import os
import lightning as L
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np  
from mole_classificator.preprocessing import ImagePreprocessor, pad_image_batch



class Milk10kDataset(Dataset):
    def __init__(self, image_paths, ground_truth_df, id_col="lesion_id", transform=None):
        self.image_paths = image_paths
        gt = ground_truth_df.set_index(id_col)

        self.class_names = gt.columns.tolist()
        lesion_labels = gt[self.class_names].values.argmax(axis=1)
        self.lesion_to_label = dict(zip(gt.index, lesion_labels))

        self.labels = np.array(
            [self.lesion_to_label[self._lesion_id(p)] for p in image_paths]
        )
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def _lesion_id(self, path):
        return os.path.basename(os.path.dirname(path))

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        lesion_id = self._lesion_id(path)
        with Image.open(path) as source:
            image = self.transform(source)
        label = self.lesion_to_label[lesion_id]
        return image, label

    def groups(self):
        # array parallelo a image_paths, usato per lo split group-aware
        return [self._lesion_id(p) for p in self.image_paths]


class Milk10kDataModule(L.LightningDataModule):
    def __init__(self, path_to_dir, path_to_ground_truth, batch_size=32, num_workers=4, val_frac=0.2, seed=42,
                 max_image_size=600, normalize=True, train_augmentation=True):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        assert os.path.exists(self.hparams.path_to_ground_truth)
        assert os.path.isdir(self.hparams.path_to_dir)

    def _collect_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.hparams.path_to_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)
    def setup(self, stage=None):
        ground_truth = pd.read_csv(self.hparams.path_to_ground_truth)
        image_paths = self._collect_image_paths()
        self.full_dataset = Milk10kDataset(
            image_paths, ground_truth,
            transform=ImagePreprocessor(self.hparams.max_image_size, self.hparams.normalize),
        )
        groups = self.full_dataset.groups()
        labels = self.full_dataset.labels  # array di indici di classe, uno per immagine

        # Stadio 1: 1 fold su 5 = 20% -> test
        skf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.hparams.seed)
        trainval_idx, test_idx = next(skf1.split(image_paths, labels, groups))

        # Stadio 2: 1 fold su 8 = 12.5% del restante 80% = 10% del totale -> val
        trainval_labels = labels[trainval_idx]
        trainval_groups = [groups[i] for i in trainval_idx]
        skf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=self.hparams.seed)
        train_rel_idx, val_rel_idx = next(skf2.split(trainval_idx, trainval_labels, trainval_groups))

        train_idx = trainval_idx[train_rel_idx]
        val_idx = trainval_idx[val_rel_idx]

        if stage in (None, "fit"):
            train_dataset = Milk10kDataset(
                image_paths, ground_truth,
                transform=ImagePreprocessor(self.hparams.max_image_size, self.hparams.normalize,
                                            augment=self.hparams.train_augmentation),
            )
            self.train_dataset = Subset(train_dataset, train_idx)
            self.val_dataset = Subset(self.full_dataset, val_idx)
        if stage in (None, "test"):
            self.test_dataset = Subset(self.full_dataset, test_idx)
   
        
    def get_col_names(self):
        return self.train_dataset.dataset.class_names

    def get_class_weights(self, beta=0.999):
        """Effective-number weights from training metadata, normalized to mean one."""
        if not 0 <= beta < 1:
            raise ValueError("class_balance_beta deve soddisfare 0 <= beta < 1")
        if not hasattr(self, "train_dataset"):
            raise RuntimeError("get_class_weights richiede setup('fit') prima del calcolo")
        class_names = self.train_dataset.dataset.class_names
        counts = np.bincount(self._extract_labels(self.train_dataset), minlength=len(class_names))
        missing = [name for name, count in zip(class_names, counts) if count == 0]
        if missing:
            raise ValueError(f"Classi assenti dal training: {missing}")
        weights = (1 - beta) / (1 - np.power(beta, counts))
        return torch.as_tensor(weights / weights.mean(), dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                           shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True,
                           collate_fn=pad_image_batch)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                           shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True,
                           collate_fn=pad_image_batch)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                           shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True,
                           collate_fn=pad_image_batch)
    @property
    def num_classes(self) -> int:
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            raise RuntimeError(
                "num_classes richiede che setup() sia già stato chiamato."
            )

        train_labels = set(self._extract_labels(self.train_dataset))
        val_labels = set(self._extract_labels(self.val_dataset))
        test_labels = set(self._extract_labels(self.test_dataset))

        # Verifica esplicita: se val/test contengono classi assenti dal
        # training, il modello non potrà mai predirle correttamente, e
        # se il training ne contiene di più di val/test è comunque un
        # segnale da controllare (split non stratificato?).
        if val_labels - train_labels:
            raise ValueError(
                f"Classi presenti in validation ma non in training: "
                f"{val_labels - train_labels}"
            )
        if test_labels - train_labels:
            raise ValueError(
                f"Classi presenti in test ma non in training: "
                f"{test_labels - train_labels}"
            )

        return len(self.full_dataset.class_names)

    def _extract_labels(self, dataset) -> list[int]:
        # Inspect labels without decoding every image in the dataset.
        return dataset.dataset.labels[dataset.indices].tolist()
