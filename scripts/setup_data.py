import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

KAGGLE_DATASET = "washingtongold/vggface2-mini"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMG_SIZE = 112
VAL_SPLIT = 0.1


def process_image(args):
    src, dst = args
    if dst.exists():
        return
    try:
        img = cv2.imread(str(src))
        if img is None:
            return

        h, w = img.shape[:2]
        size = min(h, w)
        y, x = (h - size) // 2, (w - size) // 2
        crop = img[y : y + size, x : x + size]
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), resized)
    except Exception:
        pass


def main():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install kaggle: pip install kaggle")
        return

    # Download if needed
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"Downloading {KAGGLE_DATASET}...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)

    # Build processing tasks
    all_images = list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.png"))
    tasks = []

    for src in all_images:
        identity = src.parent.name
        is_val = any(p in src.parts for p in ("val", "validation", "test"))
        split = "val" if is_val else "train"

        dst_folder = PROCESSED_DIR / split / identity
        dst_folder.mkdir(parents=True, exist_ok=True)
        tasks.append((src, dst_folder / src.name))

    print(f"Processing {len(tasks)} images...")
    with ThreadPoolExecutor() as ex:
        list(tqdm(ex.map(process_image, tasks), total=len(tasks)))

    # Create val split if missing
    val_root = PROCESSED_DIR / "val"
    train_root = PROCESSED_DIR / "train"

    if not val_root.exists() or not any(val_root.iterdir()):
        print(f"Creating validation split ({int(VAL_SPLIT * 100)}%)...")
        val_root.mkdir(parents=True, exist_ok=True)

        for identity_path in tqdm(list(train_root.iterdir())):
            if not identity_path.is_dir():
                continue
            images = list(identity_path.glob("*.jpg"))
            num_val = int(len(images) * VAL_SPLIT)
            if num_val == 0:
                continue

            dest = val_root / identity_path.name
            dest.mkdir(exist_ok=True)
            for img in random.sample(images, num_val):
                shutil.move(str(img), str(dest / img.name))

    print(f"Done. Train: {train_root}, Val: {val_root}")


if __name__ == "__main__":
    main()