#!/bin/bash

# ID wyciagniete z Twojego linku
GDRIVE_ID="1WEyNcf0WE56KCsD-SRfSHOeNFMBxF6MB"
ARCHIVE="data_full.tar.gz"

echo "🚀 START"

# 1. Instalacja gdown (cicho)
pip install -q gdown

# 2. Pobieranie
if [ ! -f "$ARCHIVE" ]; then
    echo "⬇️ Pobieranie z GDrive..."
    gdown --id $GDRIVE_ID -O $ARCHIVE
fi

# 3. Czyszczenie starego
if [ -d "data/processed" ]; then
    rm -rf data/processed
fi

# 4. Rozpakowanie
echo "📦 Rozpakowywanie..."
if command -v pigz &> /dev/null; then
    tar -I pigz -xf $ARCHIVE
else
    tar -xf $ARCHIVE
fi

# 5. Usuwanie archiwum
echo "🗑️ Usuwanie pliku .tar.gz..."
rm $ARCHIVE

echo "✅ GOTOWE. Dane w data/processed/"