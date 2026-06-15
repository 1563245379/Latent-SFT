import inspect

from transformers.trainer import Trainer


def normalize_trainer_init_kwargs(kwargs):
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" not in kwargs or "tokenizer" in trainer_parameters:
        return kwargs

    tokenizer = kwargs.pop("tokenizer")
    if "processing_class" in trainer_parameters and "processing_class" not in kwargs:
        kwargs["processing_class"] = tokenizer
    return kwargs
