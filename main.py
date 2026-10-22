import json
import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def run_smoke(cfg):
    """One real forward/backward/validation/test batch, without data or downloads."""
    import lightning as L
    import torch
    from lightning.pytorch.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader, TensorDataset

    from mole_classificator.mole_classificator import Classificator

    torch.set_num_threads(1)
    L.seed_everything(42, workers=True)
    loader = DataLoader(
        TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1])),
        batch_size=2, num_workers=0,
    )
    model = Classificator(
        lr=1e-3, weight_decay=1e-5, num_classes=2, pretrained=False,
        d_model=32, nhead=4, num_layers=1, freeze_backbone_epochs=0,
        diagnostics_every_n_steps=1,
    )
    logger = TensorBoardLogger(to_absolute_path("lightning_logs"), name="smoke")
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=1,
        limit_train_batches=1, limit_val_batches=1, limit_test_batches=1,
        num_sanity_val_steps=0, log_every_n_steps=1, logger=logger,
        enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(model, dataloaders=loader)
    logger.experiment.flush()
    log.info("Smoke test passed. TensorBoard: %s", logger.log_dir)


def run_training(cfg):
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from mole_classificator.diagnostics import InspectionTensorBoardLogger
    from mole_classificator.milk10k_datamodule import Milk10kDataModule
    from mole_classificator.mole_classificator import Classificator

    L.seed_everything(cfg.dataset.seed, workers=True)
    data = Milk10kDataModule(
        path_to_dir=to_absolute_path(cfg.dataset.path_to_folder),
        path_to_ground_truth=to_absolute_path(cfg.dataset.path_to_csv),
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        max_image_size=cfg.classifier.max_image_size,
        normalize=cfg.classifier.normalize,
        seed=cfg.dataset.seed,
        train_augmentation=cfg.dataset.train_augmentation,
    )
    data.setup()
    class_names = data.get_col_names()
    root = Path(to_absolute_path(cfg.trainer.get("default_root_dir", ".")))
    root.mkdir(parents=True, exist_ok=True)
    logger = InspectionTensorBoardLogger(str(root), name="lightning_logs", log_graph=cfg.logging.log_graph)
    trainer_params = OmegaConf.to_container(cfg.trainer, resolve=True)
    callbacks = [LearningRateMonitor(logging_interval="step")]

    if cfg.step.name == "train":
        early_stopping = EarlyStopping(**OmegaConf.to_container(cfg.early_stopping))
        checkpoint = ModelCheckpoint(monitor=cfg.early_stopping.monitor,
                                     mode=cfg.early_stopping.mode, save_top_k=1, save_last=True)
        callbacks.extend([early_stopping, checkpoint])
        params = OmegaConf.to_container(cfg.classifier, resolve=True)
        class_weights = data.get_class_weights(beta=cfg.dataset.class_balance_beta)
        params.update(num_classes=data.num_classes, class_names=class_names,
                      class_weights=class_weights)
        log.info("Training class weights: %s", dict(zip(class_names, class_weights.tolist())))
        model = Classificator(**params)
        trainer = L.Trainer(**trainer_params, callbacks=callbacks, logger=logger)
        log.info("Dataset: train=%d, validation=%d, test=%d; classes=%s",
                 len(data.train_dataset), len(data.val_dataset), len(data.test_dataset), class_names)
        trainer.fit(model, datamodule=data)
        trainer.save_checkpoint(str(root / "last.ckpt"))
        if trainer.is_global_zero:
            (root / "class_names.json").write_text(json.dumps(class_names), encoding="utf-8")
            # Export the best validation checkpoint, preserving its hyperparameters.
            import shutil
            if not checkpoint.best_model_path:
                raise RuntimeError("Training produced no validation checkpoint")
            shutil.copy2(checkpoint.best_model_path, root / "model.ckpt")
        trainer.test(model, datamodule=data, ckpt_path="best")
        log.info("Best checkpoint: %s", checkpoint.best_model_path)
    else:
        model = Classificator.load_from_checkpoint(to_absolute_path(cfg.page.checkpoint), pretrained=False)
        trainer = L.Trainer(**trainer_params, logger=logger)
        trainer.test(model, datamodule=data)


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info("Step: %s", cfg.step.name)
    # Dispatch before importing the data pipeline or constructing a model.
    if cfg.step.name == "page":
        from serving.launcher import run_page
        run_page(cfg.page)
    elif cfg.step.name == "run_on_kaggle":
        from hydra.core.hydra_config import HydraConfig
        from kaggle_runner import submit_to_kaggle
        submit_to_kaggle(dry_run=cfg.kaggle.dry_run, output_dir=to_absolute_path(cfg.kaggle.output_dir),
                         overrides=HydraConfig.get().overrides.task)
    elif cfg.step.name == "smoke":
        run_smoke(cfg)
    elif cfg.step.name in {"train", "test"}:
        run_training(cfg)
    else:
        raise ValueError(f"Unknown step: {cfg.step.name}")


if __name__ == "__main__":
    my_app()
