import math

import torch
from torch.utils.data import Subset, random_split


def split_train_validation_dataset(dataset, validation_split_ratio: float, seed: int):
    if validation_split_ratio <= 0:
        return dataset, None
    if validation_split_ratio >= 1:
        raise ValueError("validation_split_ratio must be between 0 and 1.")

    dataset_len = len(dataset)
    validation_len = max(1, math.ceil(dataset_len * validation_split_ratio))
    train_len = dataset_len - validation_len
    if train_len <= 0:
        raise ValueError(
            "validation_split_ratio must be between 0 and 1 and leave at least one training example."
        )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_len, validation_len],
        generator=generator,
    )
    return train_dataset, validation_dataset


def raw_examples_from_dataset(dataset):
    if isinstance(dataset, Subset):
        examples = raw_examples_from_dataset(dataset.dataset)
        return [examples[idx] for idx in dataset.indices]

    if not hasattr(dataset, "data"):
        raise TypeError(
            "Validation accuracy requires a dataset with a raw `data` attribute or a Subset of one."
        )
    return dataset.data
