import inspect
import logging

from transformers.trainer import Trainer

logger = logging.getLogger(__name__)

_UNSAFE_TORCH_LOAD_ERROR_MARKERS = (
    "Due to a serious vulnerability issue in `torch.load`",
    "upgrade torch to at least v2.6",
    "CVE-2025-32434",
)


def normalize_trainer_init_kwargs(kwargs):
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" not in kwargs or "tokenizer" in trainer_parameters:
        return kwargs

    tokenizer = kwargs.pop("tokenizer")
    if "processing_class" in trainer_parameters and "processing_class" not in kwargs:
        kwargs["processing_class"] = tokenizer
    return kwargs


def is_unsafe_torch_load_rng_error(error: ValueError) -> bool:
    message = str(error)
    return any(marker in message for marker in _UNSAFE_TORCH_LOAD_ERROR_MARKERS)


class SafeRngStateResumeMixin:
    def _load_rng_state(self, resume_from_checkpoint):
        try:
            return super()._load_rng_state(resume_from_checkpoint)
        except ValueError as exc:
            if not is_unsafe_torch_load_rng_error(exc):
                raise

            logger.warning(
                "Skipping RNG state restore from %s because this torch version is blocked "
                "from loading rng_state*.pth files by Transformers' CVE-2025-32434 guard. "
                "Model, optimizer, scheduler, and trainer state resume still proceed; "
                "upgrade torch to >=2.6 to restore RNG state exactly.",
                resume_from_checkpoint,
            )
            return None
