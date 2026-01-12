import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

KAGGLE_DATASET = "atulanandjha/lfwpeople"
RAW_DIR = Path("../data/raw_lfw_custom") 
OUTPUT_DIR = Path("../data/processed/lfw_112")
available_cpu = os.cpu_count() or 4
NUM_WORKERS = min(max(available_cpu * 3 // 4, 4), 32)


def process_image(args):
    """Just resize - LFW is already aligned!"""
    img_path, output_dir = args
    try:
        img = cv2.imread(str(img_path))
        if img is None: return False
        
        # Just resize, no face detection needed
        final_img = cv2.resize(img, (112, 112))
        
        person_name = img_path.parent.name
        save_dir = output_dir / person_name
        save_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(save_dir / img_path.name), final_img)
        return True
    except:
        return False

def main():
    print("🚀 Fast LFW preparation (no face detection - already aligned)")

    # 1. Download
    if not RAW_DIR.exists():
        print(f"⬇️  Downloading {KAGGLE_DATASET}...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)
            
            for item in RAW_DIR.glob("*.tgz"):
                import tarfile
                with tarfile.open(item) as tar:
                    tar.extractall(path=RAW_DIR)
        except Exception as e:
            print(f"❌ Error: {e}")
            return

    # 2. Find images
    img_root = None
    for path in RAW_DIR.rglob("lfw-funneled"):
        if path.is_dir():
            img_root = path
            break
    if not img_root:
        possible = list(RAW_DIR.rglob("George_W_Bush"))
        if possible: img_root = possible[0].parent
    if not img_root:
        print("❌ Images not found!")
        return

    # 3. Copy pairs.txt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_src = list(RAW_DIR.rglob("pairs.txt"))
    if pairs_src:
        shutil.copy(pairs_src[0], OUTPUT_DIR / "pairs.txt")
        print("✅ Copied pairs.txt")

    # 4. Process in parallel
    image_paths = list(img_root.rglob("*.jpg"))
    print(f"📸 Processing {len(image_paths)} images with {NUM_WORKERS} workers...")
    
    tasks = [(p, OUTPUT_DIR) for p in image_paths]
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_image, tasks), total=len(tasks)))
    
    print(f"\n✅ Done! Output: {OUTPUT_DIR}")
    print(f"   Processed: {sum(results)}/{len(results)}")

if __name__ == "__main__":
    main()