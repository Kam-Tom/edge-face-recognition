import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from mtcnn import MTCNN

# --- KONFIGURACJA ---
KAGGLE_DATASET = "atulanandjha/lfwpeople"
# Uwaga: RAW_DIR to folder tymczasowy, zostanie usunięty na końcu!
RAW_DIR = Path("data/raw_lfw_custom") 
OUTPUT_DIR = Path("data/processed/lfw_112")

def main():
    print("🚀 START: Proste przygotowanie LFW (Crop & Resize)")

    # 1. POBIERANIE
    if not RAW_DIR.exists():
        print(f"⬇️  Pobieranie {KAGGLE_DATASET}...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)
            
            # Rozpakowanie tgz (często LFW jest spakowane podwójnie)
            for item in RAW_DIR.glob("*.tgz"):
                import tarfile
                with tarfile.open(item) as tar:
                    tar.extractall(path=RAW_DIR)
        except Exception as e:
            print(f"❌ Błąd pobierania: {e}")
            return

    # 2. Szukanie folderu ze zdjęciami
    img_root = None
    for path in RAW_DIR.rglob("lfw-funneled"):
        if path.is_dir():
            img_root = path
            break
            
    if not img_root:
        possible = list(RAW_DIR.rglob("George_W_Bush"))
        if possible: img_root = possible[0].parent

    if not img_root:
        print("❌ Nie znalazłem zdjęć!")
        # Jak nie ma zdjęć, to sprzątamy pusty folder i kończymy
        if RAW_DIR.exists(): shutil.rmtree(RAW_DIR)
        return

    # 3. PRZETWARZANIE
    print("🕵️  Uruchamiam wykrywacz twarzy (MTCNN)...")
    detector = MTCNN()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kopiujemy listę par zanim usuniemy raw!
    pairs_src = list(RAW_DIR.rglob("pairs.txt"))
    if pairs_src:
        shutil.copy(pairs_src[0], OUTPUT_DIR / "pairs.txt")
        print("✅ Skopiowano plik pairs.txt")
    
    image_paths = list(img_root.rglob("*.jpg"))
    print(f"📸 Przetwarzam {len(image_paths)} zdjęć...")

    for img_path in tqdm(image_paths):
        try:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            
            if faces:
                x, y, w, h = faces[0]['box']
                x, y = max(0, x), max(0, y)
                face_crop = img[y:y+h, x:x+w]
                
                if face_crop.size > 0:
                    final_img = cv2.resize(face_crop, (112, 112))
                else:
                    final_img = cv2.resize(img, (112, 112))
            else:
                final_img = cv2.resize(img, (112, 112))

            # ZAPISZ
            person_name = img_path.parent.name
            file_name = img_path.name
            
            save_dir = OUTPUT_DIR / person_name
            save_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(save_dir / file_name), final_img)
            
        except Exception:
            pass

    # 4. SPRZĄTANIE (Nowa sekcja)
    print("\n🧹 Sprzątanie plików tymczasowych (usuwanie folderu raw)...")
    if RAW_DIR.exists():
        try:
            shutil.rmtree(RAW_DIR)
            print(f"🗑️  Usunięto folder: {RAW_DIR}")
        except Exception as e:
            print(f"⚠️ Nie udało się usunąć folderu tymczasowego: {e}")

    print(f"\n✅ ZAKOŃCZONO! Gotowe dane są w: {OUTPUT_DIR}")
    print("Teraz możesz bezpiecznie odpalić evaluate_lfw_pairs.py")

if __name__ == "__main__":
    main()