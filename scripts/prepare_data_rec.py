import os
# --- FIX DLA MXNET NA NOWYM PYTHONIE ---
import numpy as np
np.bool = bool 
# ---------------------------------------

import shutil
import mxnet as mx
from mxnet import recordio
import cv2
from pathlib import Path
from tqdm import tqdm

# --- KONFIGURACJA ---
KAGGLE_DATASET = "debarghamitraroy/casia-webface" 
RAW_DIR = Path("../data/raw")             
PROCESSED_DIR = Path("../data/processed/casia") # Jeden folder na wszystko

def download_dataset():
    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.rec")):
        print(f"📂 Pliki .rec znalezione, pomijam pobieranie.")
        return

    print(f"⬇️  Pobieranie datasetu {KAGGLE_DATASET}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)
        print("✅ Pobrano.")
    except Exception as e:
        print(f"❌ Błąd pobierania: {e}")
        exit(1)

def find_rec_file(start_dir):
    rec_files = list(start_dir.rglob("*.rec"))
    for f in rec_files:
        if "train" in f.name: return f
    return rec_files[0] if rec_files else None

def unpack_rec_file(rec_path, output_dir):
    idx_path = rec_path.with_suffix('.idx')
    if not idx_path.exists():
        print(f"❌ Brak pliku .idx")
        return

    print(f"🚀 Rozpakowywanie do jednego wora: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    record = recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), 'r')
    header, _ = recordio.unpack(record.read_idx(0))
    max_idx = int(header.label[0])

    for i in tqdm(range(1, max_idx + 1), desc="Extracting"):
        try:
            item = record.read_idx(i)
            if item is None: continue
            header, img_bytes = recordio.unpack(item)
            
            label = int(header.label) if isinstance(header.label, (int, float)) else int(header.label[0])
            
            img = mx.image.imdecode(img_bytes).asnumpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            person_dir = output_dir / str(label)
            person_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(person_dir / f"{i}.jpg"), img)
            
        except Exception:
            pass

def remove_single_image_folders(root_dir):
    """Usuwa osoby z < 2 zdjęciami, bo dataset.py tego wymaga."""
    print("🧹 Sprzątanie (usuwanie osób z tylko 1 zdjęciem)...")
    removed = 0
    folders = [f for f in root_dir.iterdir() if f.is_dir()]
    
    for folder in tqdm(folders, desc="Cleaning"):
        images = list(folder.glob("*.jpg"))
        if len(images) < 2:
            shutil.rmtree(folder) # Usuwamy folder
            removed += 1
            
    print(f"✅ Usunięto {removed} osób (zbyt mało zdjęć do podziału).")

def main():
    download_dataset()
    
    rec_file = find_rec_file(RAW_DIR)
    if not rec_file: return
    
    # Rozpakuj wszystko
    if not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.iterdir()):
        unpack_rec_file(rec_file, PROCESSED_DIR)
        # WAŻNE: Czyścimy, żeby Twój dataset.py zadziałał
        remove_single_image_folders(PROCESSED_DIR)
    else:
        print("📂 Dane już gotowe.")

    print("\n✅ GOTOWE! Ustaw w experiment.py:")
    # Uwaga: Ustawiamy to samo dla obu, bo dataset.py sam sobie podzieli
    print(f'DATA_DIR = "{PROCESSED_DIR}"')
    print(f'VAL_DIR  = "{PROCESSED_DIR}"')

if __name__ == "__main__":
    main()