#!/usr/bin/env bash
set -e

STAGING_DIR=".kaggle_staging"

echo "==> Preparing clean staging directory..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# Copia solo i file e le cartelle necessari
cp -r config mole_classificator main.py kaggle_runner.py pyproject.toml uv.lock dataset-metadata.json kernel-metadata.json "$STAGING_DIR/"

# Usa uv run per invocare la CLI di Kaggle dall'ambiente del progetto
# Invia la cartella di staging forzando la modalità zip per includere le sottocartelle
if ! uv run kaggle datasets status filippoarcari/hult-classification-src > /dev/null 2>&1; then
    echo "==> Creating dataset on Kaggle..."
    uv run kaggle datasets create -p "$STAGING_DIR" --public --dir-mode zip
else
    echo "==> Updating dataset source code..."
    # L'ERRORE ERA QUI: Mancava --dir-mode zip nel comando version
    uv run kaggle datasets version -p "$STAGING_DIR" -m "Update code $(date +'%Y-%m-%d %H:%M:%S')" --dir-mode zip
fi

# Rimuovi la directory temporanea
rm -rf "$STAGING_DIR"

sleep 5

echo "==> Pushing and triggering job on Kaggle GPU..."
uv run kaggle kernels push -p .

echo "==> Job submitted! Checking execution status..."
uv run kaggle kernels status filippoarcari/hult-classification-runner