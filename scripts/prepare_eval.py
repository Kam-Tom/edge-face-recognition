import os
import cv2
import shutil
import numpy as np
import tarfile
import torch
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN  # Using PyTorch-based MTCNN

# --- CONFIGURATION ---
KAGGLE_DATASET = "atulanandjha/lfwpeople"

# 1. Download location
DOWNLOAD_DIR = Path("../data/raw/lfw-download")

# 2. Raw images location
RAW_IMAGES_DIR = Path("../data/raw/lfw")

# 3. Final processed location
PROCESSED_DIR = Path("../data/processed/lfw")

AVAILABLE_CPU = os.cpu_count() or 4

# ArcFace Reference Points
REFERENCE_PTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7255, 92.3655]
], dtype=np.float32)


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


def process_alignment():
    print(f"📐 Running Alignment (facenet-pytorch) -> {PROCESSED_DIR}...")
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
    print(f"   -> Using device: {device}")
    
    detector = MTCNN(keep_all=False, select_largest=False, device=device)
    
    image_paths = list(RAW_IMAGES_DIR.rglob("*.jpg"))
    print(f"🔍 Found {len(image_paths)} images to align.")

    success = 0
    
    for img_path in tqdm(image_paths, desc="Aligning LFW"):
        try:
            person_name = img_path.parent.name
            save_dir = PROCESSED_DIR / person_name
            save_path = save_dir / img_path.name
            
            if save_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect
            boxes, probs, points = detector.detect(img_rgb, landmarks=True)
            
            final_img = None

            if boxes is not None:
                # points[0] are landmarks for the best face
                dst_pts = points[0].astype(np.float32)
                
                tform = cv2.estimateAffinePartial2D(dst_pts, REFERENCE_PTS, method=cv2.LMEDS)[0]
                if tform is not None:
                    final_img = cv2.warpAffine(img, tform, (112, 112))
            
            # Fallback
            if final_img is None:
                final_img = cv2.resize(img, (112, 112))

            save_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(save_path), final_img)
            success += 1

        except Exception:
            continue

    print(f"✅ Done. Processed: {success}/{len(image_paths)}")
    print(f"💾 Ready for validation in: {PROCESSED_DIR}")


def main():
    download_dataset()
    extract_raw_images()
    process_alignment()

if __name__ == "__main__":
    main()