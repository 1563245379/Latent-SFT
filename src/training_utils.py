import math

import torch
from torch.utils.data import Subset, random_split

DEBUG_SAMPLE_LIMIT = 200
DEBUG_NUM_TRAIN_EPOCHS = 3


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value for --debug, got {value!r}.")


def parse_project_debug_flag(argv):
    sanitized_args = []
    project_debug = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg != "--debug":
            sanitized_args.append(arg)
            index += 1
            continue

        next_index = index + 1
        if next_index >= len(argv) or argv[next_index].startswith("--"):
            project_debug = True
            index += 1
            continue

        next_value = argv[next_index]
        lowered = next_value.lower()
        if lowered in {"1", "true", "yes", "y", "on", "0", "false", "no", "n", "off"}:
            project_debug = _str_to_bool(next_value)
            index += 2
            continue

        sanitized_args.extend([arg, next_value])
        index += 2

    return sanitized_args, project_debug


def apply_project_debug_flag(training_args, project_debug: bool):
    if project_debug:
        training_args.training_debug = True
    return training_args


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


def apply_debug_training_limits(dataset, training_args):
    debug = getattr(training_args, "training_debug", False)
    legacy_debug = getattr(training_args, "debug", False)
    if isinstance(legacy_debug, bool):
        debug = debug or legacy_debug

    if not debug:
        return dataset

    training_args.num_train_epochs = DEBUG_NUM_TRAIN_EPOCHS
    if hasattr(training_args, "max_steps"):
        training_args.max_steps = -1
    sample_count = min(DEBUG_SAMPLE_LIMIT, len(dataset))
    return Subset(dataset, range(sample_count))


def get_resume_from_checkpoint(training_args):
    resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    return resume_from_checkpoint or None


def should_block_existing_output_dir(training_args, output_dir_has_contents: bool) -> bool:
    return (
        output_dir_has_contents
        and getattr(training_args, "do_train", False)
        and not getattr(training_args, "overwrite_output_dir", False)
        and get_resume_from_checkpoint(training_args) is None
    )


def raw_examples_from_dataset(dataset):
    if isinstance(dataset, Subset):
        examples = raw_examples_from_dataset(dataset.dataset)
        return [examples[idx] for idx in dataset.indices]

    if not hasattr(dataset, "data"):
        raise TypeError(
            "Validation accuracy requires a dataset with a raw `data` attribute or a Subset of one."
        )
    return dataset.data
