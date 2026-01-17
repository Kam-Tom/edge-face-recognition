import os
import cv2
import shutil
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset

# Numpy compatibility fix
np.bool = np.bool_ if hasattr(np, 'bool_') else bool

import mxnet as mx
from mxnet import recordio
from facenet_pytorch import MTCNN  # PyTorch-based MTCNN

# --- CONFIGURATION ---
RAW_REC_DIR = Path("../data/raw/casia-webface-rec")
RAW_IMAGES_DIR = Path("../data/raw/casia-webface")
PROCESSED_DIR = Path("../data/processed/casia")

KAGGLE_DATASET = "debarghamitraroy/casia-webface"
AVAILABLE_CPU = os.cpu_count() or 4
# Using more workers for data loading to keep GPU busy
NUM_WORKERS = min(max(AVAILABLE_CPU * 3 // 4, 4), 16)
BATCH_SIZE = 512  # Optimized for A6000 / RTX 3090

# Standard reference points for ArcFace (112x112)
REFERENCE_PTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7255, 92.3655]
], dtype=np.float32)


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
            # Return dummy if read fails (filtered out later)
            return np.zeros((10, 10, 3), dtype=np.uint8), str(path), False
        
        # MTCNN requires RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb, str(path), True

def collate_fn(batch):
    # Filter out failed reads
    batch = [b for b in batch if b[2]]
    if not batch: return [], [], []
    images, paths, _ = zip(*batch)
    return list(images), list(paths)


# --- DOWNLOAD & EXTRACT HELPERS ---

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


def find_rec_file(start_dir):
    rec_files = list(start_dir.rglob("*.rec"))
    for f in rec_files:
        if "train" in f.name: return f
    return rec_files[0] if rec_files else None


# --- PHASE 1: EXTRACTION ---

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


# --- PHASE 2: ALIGNMENT (BATCHED) ---

def process_alignment():
    print(f"🚀 Phase 2: Running FAST MTCNN Alignment (Batch Size: {BATCH_SIZE}) -> {PROCESSED_DIR}...")
    
    if not RAW_IMAGES_DIR.exists():
        print("❌ Raw images directory not found. Run extraction first.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   -> Using device: {device}")

    # Init facenet-pytorch MTCNN
    # keep_all=False -> keep only best face
    # post_process=False -> slightly faster
    detector = MTCNN(keep_all=False, select_largest=False, device=device, post_process=False)
    
    print("🔍 Scanning image files...")
    image_paths = list(RAW_IMAGES_DIR.rglob("*.jpg"))
    print(f"   -> Found {len(image_paths)} images.")
    
    # Setup Data Loader for speed
    dataset = FaceDataset(image_paths)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn
    )

    success_count = 0
    
    for images, paths in tqdm(loader, desc="Aligning (Batched)"):
        if not images: continue
        
        try:
            # 1. Detect faces in batch (GPU)
            # boxes_list: [[box1], [box2], ...] for each image in batch
            # points_list: [[landmarks1], ...] for each image in batch
            boxes_list, probs_list, points_list = detector.detect(images, landmarks=True)
        except Exception as e:
            print(f"⚠️ Batch error: {e}")
            continue
            
        # 2. Process results (CPU)
        for i, path_str in enumerate(paths):
            img_path = Path(path_str)
            save_dir = PROCESSED_DIR / img_path.parent.name
            save_path = save_dir / img_path.name
            
            if save_path.exists():
                continue
                
            # Get original image (RGB -> BGR for OpenCV save)
            img_rgb = images[i]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            final_img = None
            
            # Check if face detected
            if boxes_list[i] is not None:
                # points_list[i][0] contains landmarks for the best face
                dst_pts = points_list[i][0].astype(np.float32)
                
                tform = cv2.estimateAffinePartial2D(dst_pts, REFERENCE_PTS, method=cv2.LMEDS)[0]
                if tform is not None:
                    final_img = cv2.warpAffine(img_bgr, tform, (112, 112))
            
            # Fallback
            if final_img is None:
                final_img = cv2.resize(img_bgr, (112, 112))

            save_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(save_path), final_img)
            success_count += 1

    print(f"✅ Alignment completed. Processed: {success_count}/{len(image_paths)}")
    print(f"💾 Data ready at: {PROCESSED_DIR}")


def main():
    download_dataset()
    
    rec_file = find_rec_file(RAW_REC_DIR)
    if not rec_file:
        print("❌ .rec file not found.")
        return

    extract_raw_images(rec_file)
    process_alignment()

if __name__ == "__main__":
    main()