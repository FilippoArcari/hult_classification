import os
import subprocess
import sys

def find_source_dir(base_path="/kaggle/input", target_file="main.py"):
    """
    Scansiona ricorsivamente l'albero delle directory a partire da base_path.
    Ritorna il percorso assoluto della cartella che contiene target_file.
    """
    for root, dirs, files in os.walk(base_path):
        if target_file in files:
            return root
    return None

def main():
    print(f"==> Esplorazione iniziale di /kaggle/input/: {os.listdir('/kaggle/input/')}")
    
    # 1. Ricerca dinamica del codice sorgente
    src_code = find_source_dir(target_file="main.py")
    
    if not src_code:
        print("ERRORE CRITICO: Il file 'main.py' non è stato trovato in /kaggle/input/!")
        print("--- DUMP DELL'INTERO FILE SYSTEM IN /kaggle/input/ ---")
        for r, d, f in os.walk("/kaggle/input/"):
            print(f"Dir: {r} | Files: {f}")
        sys.exit(1)
        
    dst = "/kaggle/working/HULT_CLASSIFICATION"
    
    # 2. Sincronizzazione nel workspace scrivibile
    print(f"==> Trovato codice sorgente in: {src_code}")
    print(f"==> Sincronizzazione verso: {dst}...")
    os.makedirs(dst, exist_ok=True)
    os.system(f"cp -r {src_code}/* {dst}/")
    os.chdir(dst)

    # 3. Ricerca e collegamento dinamico del dataset di immagini/etichette
    src_data = find_source_dir(target_file="MILK10k_Training_GroundTruth.csv")
    
    if src_data:
        print(f"==> Trovati dati di addestramento in: {src_data}")
        print("==> Creazione symlink per evitare la copia dei file pesanti...")
        if os.path.exists(f"{src_data}/MILK10k_Training_Input"):
            os.symlink(f"{src_data}/MILK10k_Training_Input", f"{dst}/MILK10k_Training_Input")
        if os.path.exists(f"{src_data}/MILK10k_Training_GroundTruth.csv"):
            os.symlink(f"{src_data}/MILK10k_Training_GroundTruth.csv", f"{dst}/MILK10k_Training_GroundTruth.csv")
    else:
        print("ATTENZIONE: File 'MILK10k_Training_GroundTruth.csv' non trovato.")
        print("Se i dati pesanti non sono ancora stati caricati come dataset a parte, il training fallirà.")

    # 4. Inizializzazione dell'ambiente ed esecuzione
    print("==> Installazione del package manager 'uv'...")
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

    print("==> Avvio processo di addestramento (uv run --python 3.11 main.py trainer=kaggle)...")
    # Forziamo uv ad usare Python 3.11 invece dell'ultima pre-release/versione disponibile
    cmd = ["uv", "run",  "main.py", "trainer=kaggle"]
    res = subprocess.run(cmd)
    
    # Propaga l'exit code del training script al kernel Kaggle
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()