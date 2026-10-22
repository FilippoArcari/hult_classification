import json
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import EarlyStopping
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from main import run_training
from mole_classificator.mole_classificator import Classificator


def test_real_training_exports_best_model_metrics_and_unfreezes_backbone(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    image_dir = tmp_path / "images"
    rows = []
    for index in range(40):
        lesion_id = f"lesion_{index:02}"
        folder = image_dir / lesion_id
        folder.mkdir(parents=True)
        label = index % 2
        Image.new("RGB", (32, 32), (220, 20, 40) if label == 0 else (30, 40, 220)).save(folder / "image.png")
        rows.append({"lesion_id": lesion_id, "a": int(label == 0), "b": int(label == 1)})
    ground_truth = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(ground_truth, index=False)
    root = tmp_path / "artifacts"
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "step=train", f"dataset.path_to_folder={image_dir}", f"dataset.path_to_csv={ground_truth}",
            "dataset.batch_size=16", "dataset.num_workers=0", "trainer.accelerator=cpu",
            "trainer.max_epochs=4", "trainer.log_every_n_steps=1", f"+trainer.default_root_dir={root}",
            "classifier.pretrained=false", "classifier.d_model=8", "classifier.nhead=2",
            "classifier.num_layers=1", "classifier.diagnostics_every_n_steps=0",
        ])
    monkeypatch.setattr("torchvision.models.convnext_tiny", lambda **kwargs: SimpleNamespace(
        features=torch.nn.Sequential(torch.nn.Conv2d(3, 768, kernel_size=1, stride=32)),
    ))
    backbone_training = []

    class ObserveBackbone(L.Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if batch_idx == 0:
                assert all(parameter.requires_grad for parameter in module.backbone.parameters())
                backbone_training.append((trainer.current_epoch, module.backbone.training))

    trainers = []
    real_trainer = L.Trainer

    def capture_trainer(**kwargs):
        kwargs["callbacks"].append(ObserveBackbone())
        trainer = real_trainer(**kwargs, enable_progress_bar=False, enable_model_summary=False)
        trainers.append(trainer)
        return trainer

    monkeypatch.setattr(L, "Trainer", capture_trainer)
    run_training(cfg)
    trainer, = trainers
    checkpoint = trainer.checkpoint_callback
    early_stopping = next(callback for callback in trainer.callbacks if isinstance(callback, EarlyStopping))
    assert checkpoint.monitor == early_stopping.monitor == "val_f1_macro"
    assert checkpoint.mode == early_stopping.mode == "max"
    assert backbone_training == [(0, False), (1, False), (2, False), (3, True)]
    assert trainer.lr_scheduler_configs[0].scheduler.T_max == 4
    assert len(trainer.datamodule.train_dataset) == 28
    assert len(trainer.datamodule.val_dataset) == 4
    assert len(trainer.datamodule.test_dataset) == 8
    torch.testing.assert_close(trainer.lightning_module.loss_fn.weight, trainer.datamodule.get_class_weights())
    assert (root / "last.ckpt").is_file()
    assert (root / "model.ckpt").read_bytes() == Path(checkpoint.best_model_path).read_bytes()
    assert json.loads((root / "class_names.json").read_text()) == ["a", "b"]
    restored = Classificator.load_from_checkpoint(root / "model.ckpt", pretrained=False)
    torch.testing.assert_close(restored.loss_fn.weight, trainer.datamodule.get_class_weights())
    assert "class_weights" in restored.hparams
    trainer.logger.experiment.flush()
    events = EventAccumulator(trainer.logger.log_dir).Reload()
    for stage in ("val", "test"):
        for name in ("f1_macro", "auroc_macro", "f1/a", "f1/b", "auroc/a", "auroc/b"):
            values = events.Scalars(f"{stage}_{name}")
            assert values
            assert all(0 <= event.value <= 1 for event in values)
    assert len(events.Scalars("val_f1_macro")) == 4
    assert trainer.callback_metrics["test_f1_macro"].item() == pytest.approx(events.Scalars("test_f1_macro")[-1].value)
    trainer.logger.finalize("success")
