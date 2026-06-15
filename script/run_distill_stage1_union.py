import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import logging
from pathlib import Path
from transformers import (
    HfArgumentParser,
    set_seed,
)

from src.modeling.modeling_stage1 import (
    LatentSFTStage1Union
)

from src.stage1.arguments import (
    ModelArguments, 
    DataArguments, 
    Stage1TrainingArguments as TrainingArguments
)

from src.stage1.data import (
    Stage1Dataset, 
    DataCollatorForDynamicPadding,
)

from src.stage1.trainer import (
    Stage1UnionTrainer
)
from src.checkpointing import prepare_best_and_recent_checkpointing
from src.training_utils import (
    apply_debug_training_limits,
    apply_project_debug_flag,
    get_resume_from_checkpoint,
    parse_project_debug_flag,
    should_block_existing_output_dir,
    split_train_validation_dataset,
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser_args, project_debug = parse_project_debug_flag(sys.argv[1:])
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=parser_args)
    apply_project_debug_flag(training_args, project_debug)
    
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    prepare_best_and_recent_checkpointing(training_args)

    if should_block_existing_output_dir(
        training_args,
        output_dir_has_contents=os.path.exists(training_args.output_dir) and bool(os.listdir(training_args.output_dir)),
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir or --resume_from_checkpoint to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    model = LatentSFTStage1Union(
        encoder_name_or_path=model_args.encoder_name_or_path,
        decoder_name_or_path=model_args.decoder_name_or_path,
        bfloat16=model_args.bfloat16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        lora_tune=training_args.lora_tune,
        lora_path=training_args.lora_path,
        lora_rank=training_args.lora_rank,
        lora_dropout=training_args.lora_dropout,
        save_path=training_args.output_dir,
        topk_interpolation=model_args.topk_interpolation,
    )

    full_train_dataset = Stage1Dataset(path=data_args.train_data_path,
        args=data_args, 
        model=model
    )
    train_dataset, validation_dataset = split_train_validation_dataset(
        full_train_dataset,
        validation_split_ratio=data_args.validation_split_ratio,
        seed=training_args.seed,
    )
    train_dataset = apply_debug_training_limits(train_dataset, training_args)

    trainer = Stage1UnionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        validation_data_args=data_args,
        data_collator=DataCollatorForDynamicPadding(model.tokenizer.pad_token_id,model.compress_token_id, model.latent_token_ids[-1]),
        tokenizer=model.tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    train_result = trainer.train(resume_from_checkpoint=get_resume_from_checkpoint(training_args))
    trainer.save_model()
    if trainer.is_world_process_zero():
        model.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
