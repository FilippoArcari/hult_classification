# hult_classification

Classificatore MILK10k con ConvNeXt-Tiny e Transformer, training Lightning e UI Next.js.

## Installazione

Python >=3.13, [uv](https://docs.astral.sh/uv/) e, per la pagina, Node.js/npm.

```bash
uv sync
```

## Smoke test locale rapido

```bash
uv run main.py
# equivalente:
uv run main.py step=smoke
```

Usa la stessa architettura ConvNeXt del training, con una testa Transformer più piccola, due immagini sintetiche RGB 32×32 e un solo batch per training, validation e test. Esegue anche il backward e un aggiornamento dell'ottimizzatore. CPU, un thread, nessun worker, nessun download dei pesi e nessun accesso a MILK10k. I tempi includono gli import di PyTorch/Lightning; il primo `uv run` può inoltre installare le dipendenze.

Non salva checkpoint né sovrascrive `model.ckpt` o `class_names.json`. I log sono in `lightning_logs/smoke/`. Le metriche su dati sintetici verificano il funzionamento della pipeline, non la qualità del modello.

## Pagina locale: una sola immagine per richiesta

Per avviare la pagina è disponibile **`avvia_pagina.sh`**: scegliere **Esegui nel terminale** dal gestore dei file, oppure digitare `./avvia_pagina.sh` nel terminale aperto in questa cartella. Il browser si apre automaticamente; lasciare il terminale aperto durante l'uso e premere **Ctrl+C** per chiudere l'applicazione. Il primo avvio può richiedere più tempo per preparare le dipendenze. Sono necessari uv e Node.js/npm già installati sulla macchina.

```bash
uv run main.py step=page
```

Avvia il backend su 127.0.0.1:8000 e la UI già presente in `frontend/` su 127.0.0.1:3000; apre il browser quando entrambi rispondono. Al primo avvio esegue automaticamente `npm ci`. Ctrl+C arresta entrambi i servizi.

Nessuna scansione del dataset e nessun `Trainer.predict`. Il checkpoint viene caricato una sola volta, alla prima richiesta inviata dal pulsante **Analizza**. Ogni richiesta esegue il forward su una sola immagine; un successivo caricamento usa il modello già in memoria. Il frontend mantiene il proprio design e il flusso caricamento → Analizza.

```bash
# Utile su una macchina remota o senza browser:
uv run main.py step=page page.open_browser=false
# Percorsi/porta della UI personalizzati:
uv run main.py step=page page.checkpoint=/path/model.ckpt page.class_names=/path/class_names.json page.port=3001
```

Servono i pesi di un training già eseguito. La UI si avvia anche in assenza del checkpoint; l'API risponde 503 quando viene richiesta una predizione senza pesi disponibili. I nuovi checkpoint includono i nomi delle classi; `class_names.json` resta il fallback per quelli precedenti. Il caricamento del checkpoint non scarica i pesi ImageNet.

## Dimensioni delle immagini

Il modello interpola su una griglia 2D gli embedding posizionali alla dimensione effettiva delle feature di ConvNeXt. Non esiste più il limite fisso di 252 token: sono supportate immagini quadrate, verticali e orizzontali, anche oltre la vecchia griglia.

Training e serving condividono lo stesso preprocessing: orientamento EXIF, RGB, resize proporzionale del solo lato massimo se supera `classifier.max_image_size` (600 per default), senza ritaglio. Immagini con lati inferiori a 32 pixel ricevono padding; il modello richiede almeno 32×32. Il DataLoader applica padding ai batch contenenti forme diverse. Si tratta di supporto a dimensioni diverse, non di una garanzia che cambiare risoluzione lasci identiche le probabilità.

I nuovi training usano la normalizzazione ImageNet; i checkpoint precedenti, senza il campo `normalize`, mantengono il precedente comportamento ToTensor. I pesi posizionali conservano nome e forma per caricare checkpoint esistenti (252 posizioni = griglia 14×18). Per checkpoint storici con un'altra griglia specificare `pos_grid_size` coerente con il relativo `num_tokens`.

## TensorBoard

```bash
uv run tensorboard --logdir lightning_logs
```

Oltre a loss e accuratezza di train/validation/test, vengono registrati i learning rate dei gruppi dell'ottimizzatore, immagini in ingresso, distribuzioni/media/deviazione standard delle attivazioni dei blocchi ConvNeXt, della proiezione, del Transformer, del pooling e dei logits; inoltre norma globale dei gradienti e istogrammi di pesi/gradienti della testa e degli embedding posizionali.

Validation e test registrano anche macro-F1, F1 per classe e ROC AUC one-vs-rest (macro e per classe), calcolati sull'intera epoca con sincronizzazione tra GPU. L'AUC usa le probabilità softmax; l'F1 usa la classe con probabilità maggiore.

`classifier.diagnostics_every_n_steps=100` controlla gli istogrammi: 0 li disabilita; lo smoke test usa 1. Il numero di valori campionati per le attivazioni è limitato per contenere il costo. `logging.log_graph=true` aggiunge il grafo del modello con un esempio 64×64.

## Training locale

```bash
uv run main.py step=train
```

Il profilo trainer locale usa tutti i batch, fino a 50 epoche, e una GPU se disponibile (altrimenti CPU). Per la prova rapida senza dataset resta disponibile `step=smoke`. Richiede `MILK10k_Training_Input/` e `MILK10k_Training_GroundTruth.csv`. I parametri del backbone sono inclusi nell'ottimizzatore anche durante il congelamento, così lo sblocco dopo le prime 3 epoche abilita il fine tuning.

La CrossEntropy usa pesi class-balanced `(1-beta)/(1-beta**n)`, normalizzati a media 1 e calcolati dalle sole etichette di training senza decodificare le immagini. `dataset.class_balance_beta=0.999` controlla il bilanciamento; `0` assegna pesi uniformi. Solo il training applica flip orizzontali/verticali, rotazioni fino a 15° su un canvas espanso e lievi variazioni di luminosità/contrasto. `dataset.train_augmentation=false` le disabilita. Validation, test e serving usano il preprocessing deterministico. Gli split per lesione rimangono invariati.

Il learning rate iniziale è `3e-4` per la testa e `3e-5` per il backbone, con AdamW, weight decay `1e-4` e cosine annealing sull'orizzonte di training configurato. `dataset.seed=42` inizializza gli RNG e i worker dei DataLoader.

Checkpoint e early stopping monitorano `val_f1_macro` in modalità `max`, con pazienza 10 e miglioramento minimo `1e-4`. Il training salva `model.ckpt` con il macro-F1 migliore sulla validation, `last.ckpt`, `class_names.json` e i log TensorBoard nel `trainer.default_root_dir` (directory corrente per default). Anche il profilo Kaggle usa un massimo di 50 epoche; resta attivo il limite temporale di 8 ore.

## Training Kaggle con GPU

Configurare le proprie credenziali della [Kaggle CLI](https://github.com/Kaggle/kaggle-cli#api-credentials), abilitare l'account all'uso delle GPU e avere accesso al dataset indicato in `kernel-metadata.json` (attualmente `filippoarcari/hult-classification-data`).

```bash
# Prepara il job, senza upload o consumo GPU:
uv run main.py step=run_on_kaggle kaggle.dry_run=true

# Invia il codice attuale e avvia il training remoto:
uv run main.py step=run_on_kaggle

# Gli override del training vengono inoltrati al job:
uv run main.py step=run_on_kaggle trainer.max_epochs=50 classifier.lr=0.0002

# Lo script esistente usa lo stesso percorso:
bash run_on_kaggle.sh
```

Il job contiene solo il sorgente Python/configurazioni necessari, incorporati nello script: non pubblica un dataset del codice, non carica le immagini locali né credenziali e non dipende dalla disponibilità di una nuova versione del dataset sorgente. Il kernel resta privato. Il dataset MILK10k viene collegato separatamente tramite i metadati.

Richiede GPU T4 tramite `machine_shape=NvidiaTeslaT4`; il profilo `trainer=kaggle` usa tutte le GPU CUDA rese visibili da Kaggle, precisione mista 16 bit, gradient accumulation e clipping. Se CUDA non è disponibile il job termina con errore. Riusa PyTorch/torchvision dell'immagine Kaggle e installa soltanto le dipendenze del training, evitando una seconda installazione dello stack CUDA.

```bash
uv run kaggle kernels status filippoarcari/hult-classification-runner
uv run kaggle kernels output filippoarcari/hult-classification-runner -p kaggle_output
uv run tensorboard --logdir kaggle_output/lightning_logs
```

Il comando di invio termina dopo la sottomissione; il training continua su Kaggle. Gli output sono salvati in `/kaggle/working`: checkpoint, classi e log. Copiare/scaricare `model.ckpt` e `class_names.json` nella root locale per usare la pagina.

Riferimenti ufficiali: [push e controllo dei kernel Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [metadati GPU](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md), [Lightning TensorBoardLogger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html), [PyTorch TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard.html).

## Verifiche di regressione

```bash
uv run pytest -q
```

Coprono risoluzioni diverse e più di 252 token, gradienti degli embedding interpolati, sblocco del backbone, caricamento offline dei checkpoint, batch misti, serving con caricamento lazy e singola immagine, limiti upload e packaging Kaggle senza dati/credenziali.
