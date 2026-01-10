import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

KAGGLE_DATASET = "yakhyokhuja/vggface2-112x112"
RAW_DIR = Path("data/raw/big")
PROCESSED_DIR = Path("data/processed/big")
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


def count_files(directory):
    """Pomocnicza funkcja do liczenia plików w folderze"""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*.jpg"))


def main():
    print("🚀 Rozpoczynam przygotowanie danych...")

    # 1. Sprawdzanie biblioteki Kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ Błąd: Brak biblioteki kaggle. Zainstaluj: pip install kaggle")
        return

    # 2. Pobieranie danych
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"⬇️  Pobieranie datasetu {KAGGLE_DATASET}...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)
        print("✅ Pobieranie zakończone.")
    else:
        print(f"📂 Dane surowe już istnieją w {RAW_DIR}, pomijam pobieranie.")

    # 3. Indeksowanie plików
    print("🔍 Skanowanie folderów w poszukiwaniu zdjęć (to może chwilę potrwać)...")
    all_images = list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.png"))
    
    if not all_images:
        print("❌ Nie znaleziono żadnych zdjęć w folderze raw!")
        return
        
    print(f"✅ Znaleziono łącznie {len(all_images)} obrazów.")

    # 4. Przygotowanie zadań
    tasks = []
    print("⚙️  Przygotowywanie listy zadań...")
    for src in all_images:
        identity = src.parent.name
        # Jeśli folder nazywa się 'val' lub 'test', wrzuć do walidacji, w przeciwnym razie train
        is_val = any(p in src.parts for p in ("val", "validation", "test"))
        split = "val" if is_val else "train"

        dst_folder = PROCESSED_DIR / split / identity
        dst_folder.mkdir(parents=True, exist_ok=True)
        tasks.append((src, dst_folder / src.name))

    # 5. Przetwarzanie (Resize/Crop)
    print(f"🔨 Przetwarzanie {len(tasks)} zdjęć (Crop & Resize)...")
    # Używamy ThreadPoolExecutor - uwaga, zużywa dużo CPU
    with ThreadPoolExecutor() as ex:
        list(tqdm(ex.map(process_image, tasks), total=len(tasks), unit="img"))

    # 6. Tworzenie splitu walidacyjnego (jeśli go nie ma)
    val_root = PROCESSED_DIR / "val"
    train_root = PROCESSED_DIR / "train"

    # Sprawdzamy czy w folderze val jest cokolwiek
    is_val_empty = not val_root.exists() or not any(val_root.iterdir())

    if is_val_empty:
        print(f"✂️  Tworzenie podziału walidacyjnego (automatyczne {int(VAL_SPLIT * 100)}%)...")
        val_root.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        folders = list(train_root.iterdir())
        
        for identity_path in tqdm(folders, desc="Przenoszenie plików"):
            if not identity_path.is_dir():
                continue
            images = list(identity_path.glob("*.jpg"))
            num_val = int(len(images) * VAL_SPLIT)
            
            if num_val == 0:
                continue

            dest = val_root / identity_path.name
            dest.mkdir(exist_ok=True)
            
            # Losowe przenoszenie
            for img in random.sample(images, num_val):
                shutil.move(str(img), str(dest / img.name))
                moved_count += 1
        
        print(f"✅ Przeniesiono {moved_count} zdjęć do folderu walidacyjnego.")
    else:
        print("ℹ️  Podział walidacyjny już istnieje, pomijam ten krok.")

    # 7. Podsumowanie
    print("\n" + "="*40)
    print("🏁 ZAKOŃCZONO!")
    print(f"📁 Folder wynikowy: {PROCESSED_DIR}")
    print(f"📊 Statystyki końcowe:")
    print(f"   - Train set: {count_files(train_root)} zdjęć")
    print(f"   - Val set:   {count_files(val_root)} zdjęć")
    print("="*40)


if __name__ == "__main__":
    main()