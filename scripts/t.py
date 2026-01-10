import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# === KONFIGURACJA ===
SOURCE_DIR = Path("data/raw/big/vggface2_112x112") # Gdzie są foldery id_x
DEST_DIR = Path("data/processed/big")

# Ile % OSOB idzie do treningu (0.9 = 90% osób w train, 10% unikalnych osób w val)
TRAIN_RATIO = 0.9

# True = Przenieś (szybko), False = Kopiuj (bezpiecznie)
MOVE_FILES = False
# ====================

def main():
    if not SOURCE_DIR.exists():
        print(f"❌ Błąd: Nie znaleziono folderu: {SOURCE_DIR}")
        return

    # 1. Tworzymy strukturę
    train_root = DEST_DIR / "train"
    val_root = DEST_DIR / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    # 2. Pobieramy listę folderów (tożsamości)
    identity_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]
    total_ids = len(identity_folders)
    
    if total_ids == 0:
        print("❌ Nie znaleziono folderów id_x!")
        return

    # 3. Mieszamy listę OSÓB (nie zdjęć)
    print(f"🎲 Mieszanie {total_ids} tożsamości...")
    random.shuffle(identity_folders)

    # 4. Dzielimy listę osób
    split_idx = int(total_ids * TRAIN_RATIO)
    train_identities = identity_folders[:split_idx]
    val_identities = identity_folders[split_idx:]

    print(f"✅ Podział:")
    print(f"   - Train: {len(train_identities)} osób (np. {train_identities[0].name} ...)")
    print(f"   - Val:   {len(val_identities)} osób (np. {val_identities[0].name} ...)")
    print(f"   (Te zbiory są rozłączne - model nie zobaczy osób z Val podczas nauki!)")

    # 5. Funkcja wykonawcza
    action = shutil.move if MOVE_FILES else shutil.copytree

    # 6. Przenoszenie TRAIN
    print("🚀 Przenoszenie zbioru TRAIN...")
    for folder in tqdm(train_identities):
        dest = train_root / folder.name
        # copytree kopiuje cały folder z zawartością
        if not dest.exists():
            action(str(folder), str(dest))

    # 7. Przenoszenie VAL
    print("🚀 Przenoszenie zbioru VAL...")
    for folder in tqdm(val_identities):
        dest = val_root / folder.name
        if not dest.exists():
            action(str(folder), str(dest))

    print("\n🏁 Gotowe! Struktura:")
    print(f"   {train_root} -> {len(list(train_root.iterdir()))} folderów")
    print(f"   {val_root} -> {len(list(val_root.iterdir()))} folderów")

if __name__ == "__main__":
    main()