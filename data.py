from dataclasses import dataclass

import numpy as np
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HuggingFaceCIFAR10Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample["img"]
        label = sample["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


@dataclass
class ActiveLearningData:
    train_split: object
    test_split: object
    full_dataset: object
    lset: np.ndarray
    uset: np.ndarray
    val_indices: np.ndarray


def validate_split_ratios(lset_ratio: float, uset_ratio: float, val_ratio: float) -> None:
    ratio_sum = lset_ratio + uset_ratio + val_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-8):
        raise ValueError(
            f"lSet_ratio + val_ratio + uSet_ratio must be 1.0, got {ratio_sum:.6f}"
        )


def build_classification_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_transform, eval_transform


def build_feature_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_cifar10_dataset():
    train_split = load_dataset("uoft-cs/cifar10", split="train")
    test_split = load_dataset("uoft-cs/cifar10", split="test")
    full_dataset = concatenate_datasets([train_split, test_split])
    return train_split, test_split, full_dataset


def build_active_learning_data(
    lset_ratio: float,
    uset_ratio: float,
    val_ratio: float,
    seed: int,
) -> ActiveLearningData:
    validate_split_ratios(lset_ratio, uset_ratio, val_ratio)

    train_split, test_split, full_dataset = load_cifar10_dataset()
    total_size = len(full_dataset)
    indices = np.arange(total_size)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    lset_size = int(total_size * lset_ratio) + 1
    uset_size = int(total_size * uset_ratio)

    lset = indices[:lset_size]
    uset = indices[lset_size : lset_size + uset_size]
    val_indices = indices[lset_size + uset_size :]

    return ActiveLearningData(
        train_split=train_split,
        test_split=test_split,
        full_dataset=full_dataset,
        lset=lset,
        uset=uset,
        val_indices=val_indices,
    )
