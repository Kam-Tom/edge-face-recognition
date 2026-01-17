import os
import cv2
import shutil
import numpy as np
import tarfile
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import MTCNN

# --- CONFIGURATION ---
KAGGLE_DATASET = "atulanandjha/lfwpeople"
DOWNLOAD_DIR = Path("../data/raw/lfw-download")
RAW_IMAGES_DIR = Path("../data/raw/lfw")
PROCESSED_DIR = Path("../data/processed/lfw")

# Optimization Params
BATCH_SIZE = 512       # Optimized for A6000
NUM_WORKERS = 8        # Fast disk loading
AVAILABLE_CPU = os.cpu_count() or 4

# ArcFace Reference Points
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
            # Return dummy if read fails
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
    if DOWNLOAD_DIR.exists() and any(DOWNLOAD_DIR.rglob("*.tgz")):
        print("✅ Dataset archive found. Skipping download.")
        return

    print(f"⬇️ Downloading {KAGGLE_DATASET}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(KAGGLE_DATASET, path=DOWNLOAD_DIR, unzip=True)
        print("✅ Download completed.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)


def extract_raw_images():
    if RAW_IMAGES_DIR.exists() and any(RAW_IMAGES_DIR.iterdir()):
        print(f"✅ Raw images found in {RAW_IMAGES_DIR}. Skipping extraction.")
        return

    print(f"📦 Extracting archives to {RAW_IMAGES_DIR}...")
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for tar_path in DOWNLOAD_DIR.glob("*.tgz"):
        print(f"   - Extracting {tar_path.name}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=DOWNLOAD_DIR)

    source_root = None
    for path in DOWNLOAD_DIR.rglob("lfw-funneled"):
        if path.is_dir():
            source_root = path
            break
    
    if not source_root:
        possible = list(DOWNLOAD_DIR.rglob("George_W_Bush"))
        if possible: source_root = possible[0].parent

    if not source_root:
        print("❌ Could not find image root directory.")
        return

    print(f"📂 Organizing raw images into {RAW_IMAGES_DIR}...")
    shutil.copytree(source_root, RAW_IMAGES_DIR, dirs_exist_ok=True)
    
    print("✅ Raw extraction completed.")


# --- PHASE 2: ALIGNMENT (BATCHED) ---

def process_alignment():
    print(f"📐 Phase 2: Running FAST Batched Alignment (LFW) -> {PROCESSED_DIR}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Copy pairs.txt
    pairs_src = list(DOWNLOAD_DIR.rglob("pairs.txt"))
    if pairs_src:
        shutil.copy(pairs_src[0], PROCESSED_DIR / "pairs.txt")
        print("📄 Copied pairs.txt successfully.")
    else:
        print("⚠️ pairs.txt not found! Validation will fail.")

    # Init MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   -> Using device: {device} | Batch Size: {BATCH_SIZE}")
    
    detector = MTCNN(keep_all=False, select_largest=False, device=device, post_process=False)
    
    image_paths = list(RAW_IMAGES_DIR.rglob("*.jpg"))
    print(f"🔍 Found {len(image_paths)} images to align.")

    # Setup DataLoader
    dataset = FaceDataset(image_paths)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn
    )

    success_count = 0
    
    for images, paths in tqdm(loader, desc="Aligning LFW"):
        if not images: continue
        
        try:
            # 1. Detect faces in batch
            boxes_list, probs_list, points_list = detector.detect(images, landmarks=True)
        except Exception as e:
            print(f"⚠️ Batch error: {e}")
            continue

        # 2. Process results
        for i, path_str in enumerate(paths):
            img_path = Path(path_str)
            save_dir = PROCESSED_DIR / img_path.parent.name
            save_path = save_dir / img_path.name
            
            if save_path.exists():
                continue
            
            # Get original image
            img_rgb = images[i]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            final_img = None

            # Check detection
            if boxes_list[i] is not None:
                # Use landmarks for alignment
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

    print(f"✅ Done. Processed: {success_count}/{len(image_paths)}")
    print(f"💾 Ready for validation in: {PROCESSED_DIR}")


def main():
    download_dataset()
    extract_raw_images()
    process_alignment()

if __name__ == "__main__":
    main()