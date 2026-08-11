import os
import lightning as L
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np  



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
        image = self.transform(Image.open(path).convert("RGB"))
        label = self.lesion_to_label[lesion_id]
        return image, label

    def groups(self):
        # array parallelo a image_paths, usato per lo split group-aware
        return [self._lesion_id(p) for p in self.image_paths]


class Milk10kDataModule(L.LightningDataModule):
    def __init__(self, path_to_dir, path_to_ground_truth, batch_size=32, num_workers=4, val_frac=0.2, seed=42):
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
        return image_paths
    def setup(self, stage=None):
        ground_truth = pd.read_csv(self.hparams.path_to_ground_truth)
        image_paths = self._collect_image_paths()
        self.full_dataset = Milk10kDataset(image_paths, ground_truth)
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
            self.train_dataset = Subset(self.full_dataset, train_idx)
            self.val_dataset = Subset(self.full_dataset, val_idx)
        if stage in (None, "test"):
            self.test_dataset = Subset(self.full_dataset, test_idx)
   
        
    def get_col_names(self):
        return self.train_dataset.dataset.class_names

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                           shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                           shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                           shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
    @property
    def num_classes(self) -> int:
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            raise RuntimeError(
                "num_classes richiede che setup() sia già stato chiamato."
            )

        train_labels = set(self._extract_labels(self.train_dataset))
        val_labels = set(self._extract_labels(self.val_dataset))
        test_labels = set(self._extract_labels(self.test_dataset))

        all_labels = train_labels | val_labels | test_labels

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

        return len(all_labels)

    def _extract_labels(self, dataset) -> list[int]:
        # Adatta questo metodo alla struttura reale del tuo Dataset.
        # Esempio se il Dataset espone un dataframe con colonna encoded:
        # return dataset.dataframe["diagnosis_encoded"].tolist()
        return [dataset[i][1] for i in range(len(dataset))]