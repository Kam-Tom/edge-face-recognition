import os
import cv2
import shutil
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import MTCNN

# --- FIX DLA MXNET I NUMPY 1.24+ ---
# To musi być wykonane ZANIM zaimportujemy mxnet
try:
    np.bool = np.bool_
except AttributeError:
    pass # Jeśli numpy jest stary, to zadziała samo

import mxnet as mx
from mxnet import recordio
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
RAW_REC_DIR = Path("../data/raw/casia-webface-rec")
RAW_IMAGES_DIR = Path("../data/raw/casia-webface")
PROCESSED_DIR = Path("../data/processed/casia_no_align") # Output dla Crop Only

KAGGLE_DATASET = "debarghamitraroy/casia-webface"

# Optimization Params
BATCH_SIZE = 512       # A6000 pociągnie to bez problemu
NUM_WORKERS = 8        # Szybkie ładowanie
AVAILABLE_CPU = os.cpu_count() or 4


# --- DATASET FOR BATCH PROCESSING ---
class FaceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Zwracamy pusty obrazek w razie błędu
            return np.zeros((10, 10, 3), dtype=np.uint8), str(path), False
        
        # MTCNN wymaga RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb, str(path), True

def collate_fn(batch):
    batch = [b for b in batch if b[2]]
    if not batch: return [], [], []
    images, paths, _ = zip(*batch)
    return list(images), list(paths)


# --- HELPERS (Download & Extract) ---

def download_dataset():
    if RAW_REC_DIR.exists() and any(RAW_REC_DIR.rglob("*.rec")):
        print("✅ .rec files found. Skipping download.")
        return

    print(f"⬇️ Downloading dataset {KAGGLE_DATASET}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        RAW_REC_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(KAGGLE_DATASET, path=RAW_REC_DIR, unzip=True)
        print("✅ Download completed.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)

def save_raw_image(args):
    label, img_bytes, idx, output_dir = args
    try:
        img = mx.image.imdecode(img_bytes).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        person_dir = output_dir / str(label)
        person_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(person_dir / f"{idx}.jpg"), img)
        return True
    except:
        return False

def extract_raw_images(rec_path):
    if RAW_IMAGES_DIR.exists() and any(RAW_IMAGES_DIR.iterdir()):
        print(f"✅ Directory {RAW_IMAGES_DIR} is not empty. Assuming extraction is done.")
        return

    print(f"📦 Phase 1: Extracting raw images to {RAW_IMAGES_DIR}...")
    idx_path = rec_path.with_suffix('.idx')
    
    if not idx_path.exists():
        print("❌ .idx file not found.")
        return

    record = recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), 'r')
    header, _ = recordio.unpack(record.read_idx(0))
    max_idx = int(header.label[0])

    tasks = []
    print("⏳ Reading records...")
    for i in tqdm(range(1, max_idx + 1)):
        try:
            item = record.read_idx(i)
            if item is None: continue
            header, img_bytes = recordio.unpack(item)
            label = int(header.label) if isinstance(header.label, (int, float)) else int(header.label[0])
            tasks.append((label, img_bytes, i, RAW_IMAGES_DIR))
        except:
            pass

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving {len(tasks)} images using {NUM_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(save_raw_image, tasks), total=len(tasks)))
    print("✅ Extraction completed.")

def find_rec_file(start_dir):
    rec_files = list(start_dir.rglob("*.rec"))
    for f in rec_files:
        if "train" in f.name: return f
    return rec_files[0] if rec_files else None


# --- PHASE 2: CROP ONLY (BATCHED) ---

def process_crop_only_batched():
    print(f"✂️  Phase 2: Running FAST MTCNN CROP ONLY (Batch Size: {BATCH_SIZE})...")
    
    if not RAW_IMAGES_DIR.exists():
        print("❌ Raw images directory not found. Run extraction first.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   -> Using device: {device}")
    
    # Init facenet-pytorch MTCNN
    # landmarks=False bo robimy tylko crop ramki
    detector = MTCNN(keep_all=False, select_largest=False, device=device, post_process=False)
    
    print("🔍 Scanning files...")
    image_paths = list(RAW_IMAGES_DIR.rglob("*.jpg"))
    print(f"   -> Found {len(image_paths)} images to process.")
    
    # Loader
    dataset = FaceDataset(image_paths)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn
    )

    success_count = 0
    
    for images, paths in tqdm(loader, desc="Cropping (Batched)"):
        if not images: continue
        
        try:
            # 1. Detect faces (boxes only)
            # boxes_list: [[box1], [box2], ...]
            # probs_list: [[prob1], ...]
            boxes_list, probs_list = detector.detect(images, landmarks=False)
        except Exception as e:
            print(f"⚠️ Batch error: {e}")
            continue
            
        # 2. Process results (CPU)
        for i, path_str in enumerate(paths):
            img_path = Path(path_str)
            save_dir = PROCESSED_DIR / img_path.parent.name
            save_path = save_dir / img_path.name
            
            if save_path.exists(): continue
                
            img_rgb = images[i]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            final_img = None
            
            if boxes_list[i] is not None:
                # 1. Take best face
                box = boxes_list[i][0] # Pierwszy box (najlepszy)
                
                # 2. Get coordinates
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # 3. Clamp
                h_img, w_img = img_bgr.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                
                # 4. Crop
                if x2 > x1 and y2 > y1:
                    crop_img = img_bgr[y1:y2, x1:x2]
                    final_img = cv2.resize(crop_img, (112, 112))
            
            # Fallback
            if final_img is None:
                final_img = cv2.resize(img_bgr, (112, 112))

            save_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(save_path), final_img)
            success_count += 1

    print(f"✅ Cropping completed. Processed: {success_count}/{len(image_paths)}")
    print(f"💾 Data ready at: {PROCESSED_DIR}")


def main():
    download_dataset()
    
    rec_file = find_rec_file(RAW_REC_DIR)
    if not rec_file:
        print("❌ .rec file not found.")
        return

    extract_raw_images(rec_file)
    process_crop_only_batched()

if __name__ == "__main__":
    main()