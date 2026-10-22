# Come applicare questa patch

> Istruzioni storiche della prima integrazione UI. Per l'avvio attuale con
> `uv run main.py step=page`, smoke test, TensorBoard e Kaggle usare [README.md](README.md).
> L'import placeholder e il preprocessing del serving sono stati sostituiti.

Struttura contenuta in questo archivio:

```
serving/main.py              -> nuovo, servizio FastAPI
serving/requirements.txt     -> nuovo, dipendenze minime di serving
frontend/components/LesionClassifier.tsx  -> nuovo componente
frontend/app/page.tsx        -> sovrascrive la home page di default
frontend/next.config.js      -> ATTENZIONE: se ne hai già uno, fai merge manuale
gitignore-additions.txt      -> righe da aggiungere al TUO .gitignore esistente
```

## 1. Sync con rsync

Dalla cartella dove hai estratto l'archivio:

```bash
rsync -av --exclude='README_PATCH.md' --exclude='gitignore-additions.txt' \
  ./ /path/al/tuo/repo/hult_classification/
```

`-a` preserva permessi/timestamp, `-v` è verbose. Non uso `--delete`, quindi
non cancella nulla di esistente nel tuo repo: solo aggiunge/sovrascrive i
file elencati sopra.

**Attenzione a `frontend/next.config.js`**: se nel tuo progetto Next.js esiste
già un `next.config.js` con altro contenuto (es. altre rewrite, plugin),
rsync lo sovrascrive senza fare merge. Controlla il diff prima di committare,
o applica manualmente solo il blocco `rewrites()`.

## 2. Cose da completare dopo il sync

Nel file `serving/main.py`, ancora segnati con `TODO`:

- Import reale della LightningModule (attualmente placeholder
  `from src.models.lightning_module import Milk10kLightningModule`)
- Verifica che `NORMALIZE_MEAN` / `NORMALIZE_STD` coincidano con la tua
  pipeline di training
- `checkpoints/best.ckpt` e `checkpoints/class_names.json` non sono inclusi
  in questa patch (vanno scaricati dall'output del training su Kaggle)

## 3. Setup e avvio locale

```bash
# backend
cd serving
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# frontend (in un altro terminale)
cd frontend
npm install
npm run dev
```

## 4. .gitignore

Aggiungi al tuo `.gitignore` esistente il contenuto di
`gitignore-additions.txt` incluso in questo archivio (non lo sovrascrive,
va unito manualmente).
