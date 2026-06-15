from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    latent_model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models for latent model"}
    )
    ce_w: float = field(
        default=1.0
    )
    kl_w: float = field(
        default=1.0
    )
    bfloat16: bool = field(
        default=True
    )
    use_flash_attention_2: bool = field(
        default=True
    )
    topk_interpolation: int = field(
        default=5, metadata={"help": "The k value for topk interpolation"}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        metadata={"help": "Path to train data"}
    )
    train_latent_soft_label_path: str = field(
        metadata={"help": "Path to train latent state chunks"}
    )
    # Gumbel noise options
    add_gumbel_noise: bool = field(
        default=False, metadata={"help": "Whether to add Gumbel noise to soft labels"}
    )
    gumbel_temperature: float = field(
        default=1.0, metadata={"help": "Temperature for Gumbel-softmax"}
    )
    noise_scale: float = field(
        default=1.0, metadata={"help": "Scale factor for Gumbel noise"}
    )
    validation_split_ratio: float = field(
        default=0.0, metadata={"help": "Ratio of training data reserved for validation accuracy during decoder training"}
    )


@dataclass
class Stage2TrainingArguments(TrainingArguments):
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the content of the output directory"}
    )
    lora_tune: bool = field(
        default=True, metadata={"help": "Whether to use lora"}
    )
    lora_path: str = field(
        default=None, metadata={"help": "Lora path"}
    )
    lora_rank: int = field(
        default=32, metadata={"help": "Lora rank, only valid when `lora_tune=True`"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Lora dropout, only valid when `lora_tune=True`"}
    )
    training: bool = field(
        default=True, metadata={"help": "Whether to training"}
    )
    save_best_total_limit: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of best validation-accuracy checkpoints to keep"}
    )
    save_recent_total_limit: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of most recent checkpoints to keep in addition to best checkpoints"}
    )
    validation_batch_size: int = field(
        default=1, metadata={"help": "Batch size used by validation accuracy generation"}
    )
    validation_max_new_tokens: int = field(
        default=256, metadata={"help": "Maximum generated tokens per validation example"}
    )
    
