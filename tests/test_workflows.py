import ast
import base64
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
import zipfile

import lightning as L
import pytest
import torch
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from PIL import Image

import kaggle_runner
from kaggle_runner import find_training_data, prepare_submission
from mole_classificator.milk10k_datamodule import Milk10kDataModule
from mole_classificator.diagnostics import InspectionTensorBoardLogger
from mole_classificator.mole_classificator import Classificator
from mole_classificator.preprocessing import ImagePreprocessor, pad_image_batch
from serving.main import create_app


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    return Classificator(lr=1e-3, weight_decay=1e-5, num_classes=2,
                         pretrained=False, d_model=32, nhead=4, num_layers=1,
                         class_names=["a", "b"])


@pytest.mark.parametrize("height,width", [(32, 32), (450, 600), (600, 450), (512, 640)])
def test_network_accepts_resolutions_beyond_old_token_limit(model, height, width):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.rand(1, 3, height, width))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_positional_interpolation_preserves_native_grid_and_gradients(model):
    torch.testing.assert_close(model.positional_embedding(14, 18), model.pos_embedding)
    model.zero_grad()
    model.positional_embedding(16, 20).square().mean().backward()
    assert model.pos_embedding.grad is not None
    assert model.pos_embedding.grad.abs().sum() > 0


def test_unfrozen_backbone_is_updated_by_existing_optimizer(model):
    model._trainer = SimpleNamespace(current_epoch=3, max_epochs=10)
    try:
        optimizer = model.configure_optimizers()["optimizer"]
        parameter = next(model.backbone.parameters())
        assert any(parameter is p for group in optimizer.param_groups for p in group["params"])
        before = parameter.detach().clone()
        model.on_train_epoch_start()
        assert parameter.requires_grad
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert not torch.equal(parameter, before)
    finally:
        model._trainer = None
        model.zero_grad()


def test_checkpoint_load_does_not_download_pretrained_weights(model, tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": L.__version__}, checkpoint)
    original = __import__("torchvision").models.convnext_tiny

    def offline_backbone(*, weights):
        assert weights is None
        return original(weights=None)

    monkeypatch.setattr("torchvision.models.convnext_tiny", offline_backbone)
    restored = Classificator.load_from_checkpoint(checkpoint, pretrained=False)
    torch.testing.assert_close(restored.pos_embedding, model.pos_embedding)
    assert restored.hparams.class_names == ["a", "b"]


def test_tensorboard_graph_exports_and_restores_attention_fastpath(model, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    previous = torch.backends.mha.get_fastpath_enabled()
    logger = InspectionTensorBoardLogger(tmp_path, log_graph=True)
    logger.log_graph(model)
    logger.experiment.flush()
    assert EventAccumulator(logger.log_dir).Reload().Graph().node
    assert torch.backends.mha.get_fastpath_enabled() == previous
    logger.finalize("success")


def test_preprocessing_keeps_aspect_ratio_and_batches_mixed_sizes():
    transform = ImagePreprocessor(max_size=600)
    landscape = transform(Image.new("RGB", (2000, 1000)))
    portrait = transform(Image.new("RGB", (80, 160)))
    tiny = transform(Image.new("L", (1, 2)))
    assert landscape.shape == (3, 300, 600)
    assert portrait.shape == (3, 160, 80)
    assert tiny.shape == (3, 32, 32)
    images, labels = pad_image_batch([(landscape, 0), (portrait, 1), (tiny, 0)])
    assert images.shape == (3, 3, 300, 600)
    assert labels.tolist() == [0, 1, 0]
    assert torch.isfinite(images).all()


def test_label_inspection_never_opens_images():
    class MetadataDataset:
        labels = __import__("numpy").array([0, 1, 1])

        def __getitem__(self, index):
            raise AssertionError("Class counting must not decode images")

    subset = SimpleNamespace(dataset=MetadataDataset(), indices=[0, 2])
    assert Milk10kDataModule("unused", "unused")._extract_labels(subset) == [0, 1]


def png_bytes(size=(91, 51)):
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    return buffer.getvalue()


def test_page_loads_once_and_only_predicts_each_uploaded_image(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    calls = {"loads": 0, "forwards": 0}

    class Predictor:
        num_classes = 2
        hparams = OmegaConf.create({"class_names": ["a", "b"], "normalize": True, "max_image_size": 600})

        def eval(self):
            return self

        def __call__(self, image):
            assert image.shape[0] == 1
            calls["forwards"] += 1
            return torch.tensor([[0.0, 1.0]])

    def load(*args, **kwargs):
        assert kwargs["pretrained"] is False
        calls["loads"] += 1
        return Predictor()

    monkeypatch.setattr(Classificator, "load_from_checkpoint", load)
    with TestClient(create_app(checkpoint)) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert calls == {"loads": 0, "forwards": 0}
        assert client.post("/predict", files={"file": ("bad", b"invalid")}).status_code == 400
        assert calls["loads"] == 0
        for size in [(91, 51), (51, 91)]:
            response = client.post("/predict", files={"file": ("test.png", png_bytes(size), "image/png")})
            assert response.status_code == 200
            predictions = response.json()["predictions"]
            assert predictions[0]["label"] == "b"
            assert sum(p["probability"] for p in predictions) == pytest.approx(1)
        assert calls == {"loads": 1, "forwards": 2}


def test_page_starts_without_checkpoint_and_reports_missing_weights(tmp_path):
    with TestClient(create_app(tmp_path / "missing.ckpt")) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", files={"file": ("test.png", png_bytes())}).status_code == 503
        assert client.post("/predict", files={"file": ("large", b"x" * (10*1024*1024+1))}).status_code == 413


def test_kaggle_bundle_uses_current_source_gpu_and_forwards_training_options(tmp_path):
    metadata = prepare_submission(tmp_path, ["step=run_on_kaggle", "classifier.lr=0.0002",
                                           "trainer.max_epochs=2", "kaggle.dry_run=true"])
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["filippoarcari/hult-classification-data"]
    tree = ast.parse((tmp_path / "kaggle_runner.py").read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value)
              for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in {"EMBEDDED_SOURCE", "TRAIN_OVERRIDES"}}
    assert values["TRAIN_OVERRIDES"] == ["classifier.lr=0.0002", "trainer.max_epochs=2"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(values["EMBEDDED_SOURCE"]))) as archive:
        assert "mole_classificator/preprocessing.py" in archive.namelist()
        assert archive.read("main.py") == Path("main.py").read_bytes()
        assert all("MILK10k" not in name and ".venv" not in name for name in archive.namelist())


def test_kaggle_finds_data_in_nested_input_mounts(tmp_path):
    dataset = tmp_path / "datasets" / "owner" / "milk10k" / "1"
    dataset.mkdir(parents=True)
    (dataset / "MILK10k_Training_GroundTruth.csv").touch()
    (dataset / "MILK10k_Training_Input").mkdir()
    images, labels = find_training_data(tmp_path)
    assert images.is_dir() and labels.is_file()


def test_remote_runner_reuses_cuda_and_starts_gpu_training(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    prepare_submission(bundle)
    parsed = ast.parse((bundle / "kaggle_runner.py").read_text())
    source = next(ast.literal_eval(node.value) for node in parsed.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == "EMBEDDED_SOURCE")
    monkeypatch.setattr(kaggle_runner, "EMBEDDED_SOURCE", source)
    monkeypatch.setattr(kaggle_runner, "TRAIN_OVERRIDES", ["trainer.max_epochs=2"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr(kaggle_runner, "find_training_data", lambda root: (tmp_path / "images", tmp_path / "labels.csv"))
    commands = []
    monkeypatch.setattr(kaggle_runner.subprocess, "run", lambda command, **kwargs: commands.append(command))
    kaggle_runner.main(workspace=tmp_path / "work")
    assert len(commands) == 2
    assert commands[0][1:4] == ["-m", "pip", "install"]
    assert not any(re.match(r"^(torch|torchvision)(?:$|[<>=!~\[])", item) for item in commands[0])
    assert "torchmetrics>=1.9,<2" in commands[0]
    assert "step=train" in commands[1] and "trainer=kaggle" in commands[1]
    assert "trainer.accelerator=gpu" in commands[1] and "trainer.max_epochs=2" in commands[1]
    assert (tmp_path / "work/hult_classification/main.py").is_file()


def test_remote_runner_refuses_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_runner, "EMBEDDED_SOURCE", "nonempty")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        kaggle_runner.main(workspace=tmp_path)
