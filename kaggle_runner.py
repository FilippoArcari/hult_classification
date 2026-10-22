"""Package the current source locally; train with Kaggle's preinstalled CUDA PyTorch."""

import base64
import io
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
EMBEDDED_SOURCE = ""
TRAIN_OVERRIDES = []


def prepare_submission(output_dir, overrides=()):
    """A single self-contained script avoids source-dataset versioning races."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    paths = [ROOT / "main.py"]
    paths += sorted((ROOT / "config").rglob("*.yaml"))
    paths += sorted((ROOT / "mole_classificator").glob("*.py"))
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    source = base64.b64encode(buffer.getvalue()).decode("ascii")
    # Local-only dispatch settings must never cause recursive remote submission.
    forwarded = []
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in {"step", "step.name", "trainer"} or key.startswith(("kaggle.", "page.")):
            continue
        forwarded.append(override)
    runner = Path(__file__).read_text(encoding="utf-8")
    runner = runner.replace('EMBEDDED_SOURCE = ""', "EMBEDDED_SOURCE = " + repr(source), 1)
    runner = runner.replace("TRAIN_OVERRIDES = []", "TRAIN_OVERRIDES = " + repr(forwarded), 1)
    (output_dir / "kaggle_runner.py").write_text(runner, encoding="utf-8")
    metadata = json.loads((ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
    source_dataset = json.loads((ROOT / "dataset-metadata.json").read_text(encoding="utf-8"))["id"]
    metadata["dataset_sources"] = [item for item in metadata["dataset_sources"] if item != source_dataset]
    metadata.update(code_file="kaggle_runner.py", enable_gpu=True, enable_internet=True)
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def submit_to_kaggle(dry_run=False, output_dir=".kaggle_staging", overrides=()):
    metadata = prepare_submission(output_dir, overrides)
    log.info("Prepared current source in %s for %s", output_dir, metadata["id"])
    if dry_run:
        log.info("Dry run complete: no upload or GPU job was started.")
        return
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("Kaggle CLI missing. Run uv sync.")
    # The CLI uses the user's local Kaggle credentials; none are packaged.
    subprocess.run([kaggle, "kernels", "push", "-p", str(output_dir)], check=True)
    log.info("Submitted GPU training: https://www.kaggle.com/code/%s", metadata["id"])
    log.info("Check: uv run kaggle kernels status %s", metadata["id"])


def find_training_data(base_path="/kaggle/input"):
    base_path = Path(base_path)
    for csv in sorted(base_path.rglob("MILK10k_Training_GroundTruth.csv")):
        images = csv.parent / "MILK10k_Training_Input"
        if images.is_dir():
            return images, csv
    raise FileNotFoundError(
        "Attach the MILK10k data dataset containing MILK10k_Training_GroundTruth.csv "
        "and MILK10k_Training_Input/ to the Kaggle kernel."
    )


def main(workspace="/kaggle/working", input_dir="/kaggle/input"):
    if not EMBEDDED_SOURCE:
        raise RuntimeError("Use uv run main.py step=run_on_kaggle to submit the current source.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not provide a CUDA GPU; CPU training is disabled for this step.")
    print(f"CUDA GPUs: {torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})", flush=True)
    workspace = Path(workspace)
    source_dir = workspace / "hult_classification"
    source_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(EMBEDDED_SOURCE))) as archive:
        archive.extractall(source_dir)
    images, labels = find_training_data(input_dir)
    # Reuse the GPU image's torch/torchvision; do not install another CUDA stack.
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "lightning>=2.5,<3", "hydra-core>=1.3,<2", "tensorboard>=2.18",
        "torchmetrics>=1.9,<2",
        "pandas>=2.2", "scikit-learn>=1.5",
    ], check=True)
    command = [
        sys.executable, str(source_dir / "main.py"), "step=train", "trainer=kaggle",
        *TRAIN_OVERRIDES,
        "trainer.accelerator=gpu",
        f"dataset.path_to_folder={images}", f"dataset.path_to_csv={labels}",
        f"trainer.default_root_dir={workspace}",
    ]
    subprocess.run(command, cwd=workspace, check=True)


if __name__ == "__main__":
    main()
