import json

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import lightning as L
from mole_classificator.milk10k_datamodule import Milk10kDataModule
from mole_classificator.mole_classificator import Classificator
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    milk10kDataModule = Milk10kDataModule(
        path_to_dir=cfg.dataset.path_to_folder,
        path_to_ground_truth=cfg.dataset.path_to_csv,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )
    milk10kDataModule.setup()
    class_names = milk10kDataModule.get_col_names()

    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    train_dataset = milk10kDataModule.train_dataset
    print(f"shape of train dataset: {train_dataset[0][0].shape}")
    print(f"Train dataset size: {len(milk10kDataModule.train_dataset)}")
    print(f"Validation dataset size: {len(milk10kDataModule.val_dataset)}")
    print(f"Test dataset size: {len(milk10kDataModule.test_dataset)}")

    # Deriva num_classes dai dati reali invece di assumerlo in config.
    num_classes = milk10kDataModule.num_classes


    if cfg.step.name == "train":
        # cfg.classifier è in modalità struct (Hydra la imposta di default),
        # quindi non puoi aggiungere chiavi nuove senza open_dict.
        with open_dict(cfg):
            if "num_classes" in cfg.classifier and cfg.classifier.num_classes != num_classes:
                print(
                    f"[WARNING] cfg.classifier.num_classes="
                    f"{cfg.classifier.num_classes} non corrisponde al valore "
                    f"derivato dai dati ({num_classes}). Uso quello derivato."
                )
            cfg.classifier.num_classes = num_classes
        early_stopping_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode=cfg.early_stopping.mode,
            verbose=True,
        )

        model = Classificator(**cfg.classifier)
        trainer = L.Trainer(**cfg.trainer, callbacks=[early_stopping_callback])
        trainer.fit(model, milk10kDataModule)
        if early_stopping_callback.stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED:    
            print(f"Early stopping triggered after {early_stopping_callback.wait_count} epochs without improvement.")
        print(f"Training completed. Best model checkpoint saved at: {trainer.checkpoint_callback.best_model_path}")
        trainer.save_checkpoint("model.ckpt")
        print("Risultati del test:")
        trainer.test(model, milk10kDataModule)
    if cfg.step.name == "test":
        model = Classificator.load_from_checkpoint("model.ckpt")
        trainer = L.Trainer(**cfg.trainer)
        trainer.test(model, milk10kDataModule)
    if cfg.step.name == "page":
        model = Classificator.load_from_checkpoint("model.ckpt")
        trainer = L.Trainer(**cfg.trainer)
        trainer.predict(model, milk10kDataModule)


if __name__ == "__main__":
    my_app()