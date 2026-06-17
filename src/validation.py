import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist

from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer
from src.checkpointing import rotate_trainer_checkpoints_for_best_and_recent
from src.modeling.modeling_stage1 import _get_decoder_embed_tokens, softmax_over_embedding_topk
from src.modeling.modeling_stage2 import LatentSFTStage2SoftEmbedding
from src.stage1.data import (
    _normalize_stage1_example,
    build_latent_token_induction_mask,
    insert_special_token_every_k,
)
from src.training_utils import raw_examples_from_dataset

logger = logging.getLogger(__name__)


def _dist_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _dist_rank_and_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rank_shard_examples(examples, rank, world_size):
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    return examples[rank::world_size]


def _dist_sum_validation_counts(correct, total, device):
    if not dist.is_available() or not dist.is_initialized():
        return correct, total

    backend = str(dist.get_backend()).lower()
    tensor_device = device if "nccl" in backend else torch.device("cpu")
    counts = torch.tensor(
        [float(correct), float(total)],
        dtype=torch.float64,
        device=tensor_device,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return int(counts[0].item()), int(counts[1].item())


def _model_device(model):
    return next(model.parameters()).device


def _model_dtype(model):
    dtype = next(model.parameters()).dtype
    return dtype if dtype.is_floating_point else torch.float32


def _left_pad_2d(sequences, fill_value, dtype=torch.long):
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    for sequence in sequences:
        pad_len = max_len - len(sequence)
        padded.append([fill_value] * pad_len + sequence)
    return torch.tensor(padded, dtype=dtype)


def _right_pad_2d(sequences, fill_value, dtype=torch.long):
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    for sequence in sequences:
        pad_len = max_len - len(sequence)
        padded.append(sequence + [fill_value] * pad_len)
    return torch.tensor(padded, dtype=dtype)


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_validation_answer(example):
    if "answer" in example and str(example["answer"]).strip():
        return str(example["answer"]).strip()
    if "cot_answer" in example and str(example["cot_answer"]).strip():
        return extract_answer(str(example["cot_answer"]))
    if "solution" in example and str(example["solution"]).strip():
        return extract_answer(str(example["solution"]))
    raise ValueError("Validation example must include `answer`, `cot_answer`, or `solution`.")


def compute_accuracy_from_predictions(predictions, examples):
    correct, total = _count_correct_predictions(predictions, examples)
    if total == 0:
        return 0.0
    return correct / total


def _count_correct_predictions(predictions, examples):
    if len(predictions) != len(examples):
        raise ValueError("Prediction count must match validation example count.")
    if not examples:
        return 0, 0

    correct = 0
    for prediction, example in zip(predictions, examples):
        pred_answer = extract_answer(prediction)
        gt_answer = extract_validation_answer(example)
        correct += int(check_is_correct(pred_answer, gt_answer))
    return correct, len(examples)


def _stage2_prompt(example, model):
    problem = example["problem"]
    latent_model_path = model.latent_model_path.lower()
    if "deepseek" in latent_model_path:
        messages = [
            {
                "role": "user",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + problem,
            },
        ]
        input_text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_prefix = input_text + "<｜Assistant｜>"
    elif "llama" in latent_model_path:
        input_text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem}"
            "<|eot_id|>"
        )
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        messages = [
            {
                "role": "user",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + problem,
            },
        ]
        input_prefix = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    input_ids = model.tokenizer(
        input_prefix,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    input_ids = input_ids + model.latent_token_ids[0]
    device = _model_device(model)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long, device=device),
    }


def _generate_stage2_predictions(model, examples, batch_size, max_new_tokens):
    predictions = []
    latent_model = model.latent_model
    latent_model.tokenizer = model.tokenizer
    latent_model.latent_token_ids = model.latent_token_ids

    for batch in _batched(examples, batch_size):
        for example in batch:
            inputs = _stage2_prompt(example, model)
            if model.lora_tune:
                decoded = model.one_example_generate_lora(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=False,
                )
            else:
                decoded = LatentSFTStage2SoftEmbedding.one_example_generate_hf(
                    latent_model,
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=False,
                    topk_interpolation=model.topk_interpolation,
                )
            predictions.append(decoded["text"])
    return predictions


def _stage1_reasoning_text(example):
    for field in ("solution", "cot", "cot_answer"):
        if field in example and str(example[field]).strip():
            return str(example[field]).strip()
    raise ValueError("Stage 1 validation example must include `solution`, `cot`, or `cot_answer`.")


def _prepare_stage1_example(example, model, compression_rate):
    example = _normalize_stage1_example(example)
    problem = example["problem"]
    encoder_name = model.encoder_name_or_path.lower()
    if "deepseek" in encoder_name:
        messages = [
            {
                "role": "user",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + problem,
            },
        ]
        input_text = cot_text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        cot_prefix = cot_text + "<｜Assistant｜>"
        input_prefix = input_text + "<｜Assistant｜>"
    elif "llama" in encoder_name:
        input_text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem}"
            "<|eot_id|>"
        )
        cot_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_prefix = cot_prefix
    elif "qwen" in encoder_name:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": problem},
        ]
        input_prefix = cot_prefix = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        raise ValueError("Unsupported model type for Stage 1 validation.")

    input_ids = model.tokenizer(
        input_prefix,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    cot_prefix_ids = model.tokenizer(
        cot_prefix,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    cot_content_ids = model.tokenizer(
        _stage1_reasoning_text(example),
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    cot_suffix_ids, count, _ = insert_special_token_every_k(
        cot_content_ids,
        model.compress_token_id,
        k=compression_rate,
    )
    return {
        "input_ids": input_ids + model.latent_token_ids[0] + [-100] * count + model.latent_token_ids[1],
        "cot_ids": cot_prefix_ids + model.latent_token_ids[0] + cot_suffix_ids + model.latent_token_ids[1],
    }


def _generate_stage1_predictions(model, examples, batch_size, max_new_tokens, compression_rate):
    device = _model_device(model)
    dtype = _model_dtype(model)
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    predictions = []
    decoder_embed_tokens = _get_decoder_embed_tokens(model.decoder)
    for batch in _batched(examples, batch_size):
        prepared = [_prepare_stage1_example(example, model, compression_rate) for example in batch]
        input_ids = _left_pad_2d(
            [item["input_ids"] for item in prepared],
            fill_value=pad_token_id,
        ).to(device)
        input_attention_mask = (input_ids != pad_token_id).long()
        input_attention_mask[input_ids == -100] = 1

        cot_ids = _right_pad_2d(
            [item["cot_ids"] for item in prepared],
            fill_value=pad_token_id,
        ).to(device)
        cot_attention_mask = build_latent_token_induction_mask(
            cot_ids,
            [model.compress_token_id],
            pad_token_id,
            dtype=dtype,
        )

        compress_embedding = model._compress(cot_ids, cot_attention_mask)
        compress_mask = cot_ids == model.compress_token_id
        decoder_mask = input_ids == -100

        input_ids_for_embed = input_ids.clone()
        input_ids_for_embed[decoder_mask] = pad_token_id
        input_embeddings = decoder_embed_tokens(input_ids_for_embed)

        for index in range(len(prepared)):
            compress_idx = compress_mask[index].nonzero(as_tuple=False).squeeze(-1)
            decoder_idx = decoder_mask[index].nonzero(as_tuple=False).squeeze(-1)
            if compress_idx.shape[0] != decoder_idx.shape[0]:
                raise ValueError(
                    f"Validation sample {index} has {compress_idx.shape[0]} compressed tokens "
                    f"but {decoder_idx.shape[0]} decoder placeholders."
                )
            new_embeddings, _, _ = softmax_over_embedding_topk(
                compress_embedding[index, compress_idx],
                decoder_embed_tokens,
                top_k=model.topk_interpolation,
                temperature=1.0,
                use_cosine=False,
            )
            input_embeddings[index, decoder_idx] = new_embeddings

        generated = model.decoder.generate(
            inputs_embeds=input_embeddings,
            attention_mask=input_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        predictions.extend(
            model.tokenizer.decode(output, skip_special_tokens=False)
            for output in generated
        )
    return predictions


def compute_validation_accuracy(
    model,
    validation_dataset,
    stage,
    data_args,
    batch_size,
    max_new_tokens,
):
    examples = raw_examples_from_dataset(validation_dataset)
    if not examples:
        return 0.0
    if stage not in {"stage2", "stage1_decoder", "stage1_union"}:
        raise ValueError(f"Unsupported validation stage: {stage}")

    rank, world_size = _dist_rank_and_world_size()
    local_examples = _rank_shard_examples(examples, rank, world_size)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if not local_examples:
                predictions = []
            elif stage == "stage2":
                predictions = _generate_stage2_predictions(
                    model,
                    local_examples,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                )
            elif stage in {"stage1_decoder", "stage1_union"}:
                predictions = _generate_stage1_predictions(
                    model,
                    local_examples,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    compression_rate=data_args.compression_rate,
                )
    finally:
        if was_training:
            model.train()

    local_correct, local_total = _count_correct_predictions(predictions, local_examples)
    correct, total = _dist_sum_validation_counts(
        local_correct,
        local_total,
        device=_model_device(model),
    )
    if total == 0:
        return 0.0
    return correct / total


def update_checkpoint_validation_accuracy(trainer, validation_dataset, validation_stage, data_args):
    if validation_dataset is None:
        return

    accuracy = compute_validation_accuracy(
        model=trainer.model,
        validation_dataset=validation_dataset,
        stage=validation_stage,
        data_args=data_args,
        batch_size=trainer.args.validation_batch_size,
        max_new_tokens=trainer.args.validation_max_new_tokens,
    )
    if not trainer.is_world_process_zero():
        _dist_barrier()
        return

    trainer.log({"validation_accuracy": accuracy})

    checkpoint_dir = Path(trainer.args.output_dir) / f"checkpoint-{trainer.state.global_step}"
    if checkpoint_dir.exists():
        trainer.state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
        summary_path = checkpoint_dir / "validation_metrics.json"
        summary_path.write_text(
            json.dumps(
                {
                    "validation_accuracy": accuracy,
                    "validation_size": len(raw_examples_from_dataset(validation_dataset)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    rotate_trainer_checkpoints_for_best_and_recent(trainer)
    _dist_barrier()
