import csv

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import Subset

from mole_classificator.milk10k_datamodule import Milk10kDataModule
from mole_classificator.preprocessing import ImagePreprocessor


class MetadataDataset:
    class_names = ["a", "b", "c"]

    def __init__(self):
        self.labels = np.array([0, 0, 1, 2, 2, 2, 0, 1, 1, 1])

    def __getitem__(self, index):
        raise AssertionError("Class weights must not decode images")


def test_class_weights_use_only_training_metadata():
    dataset = MetadataDataset()
    data = Milk10kDataModule("unused", "unused")
    data.train_dataset = Subset(dataset, [0, 1, 2, 3, 4, 5])
    data.val_dataset = Subset(dataset, [6, 7])
    data.test_dataset = Subset(dataset, [8, 9])
    expected = torch.tensor([(1 - 0.9) / (1 - 0.9 ** count) for count in [2, 1, 3]])
    expected /= expected.mean()
    torch.testing.assert_close(data.get_class_weights(beta=0.9), expected)
    torch.testing.assert_close(data.get_class_weights(beta=0), torch.ones(3))
    dataset.labels[6:] = 2
    torch.testing.assert_close(data.get_class_weights(beta=0.9), expected)


@pytest.mark.parametrize("beta", [-0.1, 1, 1.1, float("nan")])
def test_class_weights_reject_invalid_beta(beta):
    with pytest.raises(ValueError, match="0 <= beta < 1"):
        Milk10kDataModule("unused", "unused").get_class_weights(beta)


def test_class_weights_report_missing_training_class():
    data = Milk10kDataModule("unused", "unused")
    data.train_dataset = Subset(MetadataDataset(), [0, 1, 2])
    with pytest.raises(ValueError, match="Classi assenti dal training.*c"):
        data.get_class_weights()


def test_augmentation_is_exclusive_to_training(tmp_path):
    image_folder = tmp_path / "images"
    csv_path = tmp_path / "labels.csv"
    image = Image.fromarray(np.arange(48 * 80 * 3, dtype=np.uint8).reshape(48, 80, 3))
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lesion_id", "a", "b"])
        for index in range(40):
            lesion_id = f"lesion_{index:03}"
            folder = image_folder / lesion_id
            folder.mkdir(parents=True)
            for view in range(2):
                image.save(folder / f"view_{view}.png")
            writer.writerow([lesion_id, int(index % 2 == 0), int(index % 2 == 1)])
    data = Milk10kDataModule(str(image_folder), str(csv_path), num_workers=0)
    data.setup()
    train, val, test = data.train_dataset, data.val_dataset, data.test_dataset
    assert train.dataset is not val.dataset
    assert val.dataset is test.dataset
    np.testing.assert_array_equal(train.dataset.labels, val.dataset.labels)
    assert train.dataset.transform.augmentation is not None
    assert val.dataset.transform.augmentation is None
    torch.manual_seed(7)
    first, label = train[0]
    second, next_label = train[0]
    assert label == next_label
    assert first.shape != second.shape or not torch.equal(first, second)
    for subset in (val, test):
        first, label = subset[0]
        second, next_label = subset[0]
        assert label == next_label
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, ImagePreprocessor()(image), rtol=0, atol=0)
    train_groups = {train.dataset.groups()[i] for i in train.indices}
    val_groups = {val.dataset.groups()[i] for i in val.indices}
    test_groups = {test.dataset.groups()[i] for i in test.indices}
    assert not (train_groups & val_groups or train_groups & test_groups or val_groups & test_groups)


def test_train_rotation_keeps_the_full_image_before_resize():
    torch.manual_seed(7)
    transform = ImagePreprocessor(normalize=False, augment=True)
    result = transform(Image.new("RGB", (80, 48), "white"))
    assert result.shape[1] >= 48 and result.shape[2] >= 80
    assert torch.isfinite(result).all()
