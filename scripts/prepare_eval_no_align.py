import os
import cv2
import shutil
import numpy as np
import tarfile
from pathlib import Path
from tqdm import tqdm
from mtcnn import MTCNN

# --- CONFIGURATION ---
KAGGLE_DATASET = "atulanandjha/lfwpeople"

# 1. Download location
DOWNLOAD_DIR = Path("../data/raw/lfw-download")

# 2. Raw images location
RAW_IMAGES_DIR = Path("../data/raw/lfw")

# 3. OUTPUT LOCATION (Distinct from the aligned version)
PROCESSED_DIR = Path("../data/processed/lfw_crop_only")

AVAILABLE_CPU = os.cpu_count() or 4


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
    """Extracts images from downloaded archives to a clean RAW folder."""
    if RAW_IMAGES_DIR.exists() and any(RAW_IMAGES_DIR.iterdir()):
        print(f"✅ Raw images found in {RAW_IMAGES_DIR}. Skipping extraction.")
        return

    print(f"📦 Extracting archives to {RAW_IMAGES_DIR}...")
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract .tgz files
    for tar_path in DOWNLOAD_DIR.glob("*.tgz"):
        print(f"   - Extracting {tar_path.name}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=DOWNLOAD_DIR)

    # 2. Locate image folder
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

    # 3. Copy to RAW_IMAGES_DIR
    print(f"📂 Organizing raw images into {RAW_IMAGES_DIR}...")
    shutil.copytree(source_root, RAW_IMAGES_DIR, dirs_exist_ok=True)
    
    print("✅ Raw extraction completed.")


def process_crop_only():
    print(f"✂️  Running CROP ONLY (No Alignment) -> {PROCESSED_DIR}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Copy pairs.txt (CRITICAL)
    pairs_src = list(DOWNLOAD_DIR.rglob("pairs.txt"))
    if pairs_src:
        shutil.copy(pairs_src[0], PROCESSED_DIR / "pairs.txt")
        print("📄 Copied pairs.txt successfully.")
    else:
        print("⚠️ pairs.txt not found! Validation will fail.")

    detector = MTCNN()
    image_paths = list(RAW_IMAGES_DIR.rglob("*.jpg"))
    print(f"🔍 Found {len(image_paths)} images to process.")

    success = 0
    
    for img_path in tqdm(image_paths, desc="Cropping LFW"):
        try:
            person_name = img_path.parent.name
            save_dir = PROCESSED_DIR / person_name
            save_path = save_dir / img_path.name
            
            if save_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)
            
            final_img = None

            if results:
                # 1. Best face
                best_face = max(results, key=lambda x: x['confidence'])
                
                # 2. Bounding Box
                x, y, w, h = best_face['box']
                
                # 3. Clamp coords
                h_img, w_img = img.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                # 4. Crop & Resize
                if w > 0 and h > 0:
                    crop = img[y:y+h, x:x+w]
                    final_img = cv2.resize(crop, (112, 112))
            
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
    process_crop_only()

if __name__ == "__main__":
    main()