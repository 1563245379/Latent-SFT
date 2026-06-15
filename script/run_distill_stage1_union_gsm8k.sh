#!/bin/bash
set -euo pipefail

export HF_HOME=/workspace/cache

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Runtime overrides: PYTHON=python NPROC_PER_NODE=8 MASTER_PORT=25001 bash $0
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 8)}"
MASTER_PORT="${MASTER_PORT:-25001}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Editable config
save_root="${REPO_ROOT}/output/stage1_union"
output_name=""
encoder_name_or_path="<path-to-stage1-encoder-checkpoint-hf>"
decoder_name_or_path="<path-to-stage1-decoder-checkpoint-hf>"
train_data_path="${REPO_ROOT}/<path-to-your-train-jsonl>"
compression_rate=2
topk_interpolation=10
validation_split_ratio=0.05
validation_batch_size=4
save_best_total_limit=3
save_recent_total_limit=1
debug=False
resume_from_checkpoint=""
deepspeed_config="${REPO_ROOT}/config_zero1.json"
output_dir="${save_root}/${output_name}"

# Create the run directory and archive this launcher for reproducibility.
mkdir -p "${output_dir}"
echo "Archiving launcher script to ${output_dir}/"
cp "$0" "${output_dir}/"

# ----------------------------------------------------------------------------
# 5. torchrun argument groups
# ----------------------------------------------------------------------------
distributed_args="
    --standalone \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
"

model_args="
    --encoder_name_or_path ${encoder_name_or_path} \
    --decoder_name_or_path ${decoder_name_or_path} \
    --bfloat16 True \
    --use_flash_attention_2 False \
    --topk_interpolation ${topk_interpolation} \
"

data_args="
    --train_data_path ${train_data_path} \
    --compression_rate ${compression_rate} \
    --validation_split_ratio ${validation_split_ratio} \
"

stage1_train_args="
    --lora_tune True \
    --lora_rank 64 \
    --lora_dropout 0.1 \
"

train_args="
    --debug ${debug} \
    --deepspeed ${deepspeed_config} \
    --no_remove_unused_columns \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --num_train_epochs 10 \
    --bf16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --dataloader_drop_last False \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 16 \
    --dataloader_pin_memory True \
    --logging_steps 1 \
    --save_best_total_limit ${save_best_total_limit} \
    --save_recent_total_limit ${save_recent_total_limit} \
    --validation_batch_size ${validation_batch_size} \
    --metric_for_best_model validation_accuracy \
    --greater_is_better True \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --logging_dir ${output_dir}/tensorboard \
    --run_name ${output_name} \
    --overwrite_output_dir \
    --output_dir ${output_dir}
"
if [[ -n "${resume_from_checkpoint}" ]]; then
    train_args+=" --resume_from_checkpoint ${resume_from_checkpoint}"
fi

# ----------------------------------------------------------------------------
# 6. Launch
# ----------------------------------------------------------------------------
"${PYTHON}" -m torch.distributed.run \
    ${distributed_args} \
    "${SCRIPT_DIR}/run_distill_stage1_union.py" \
    ${model_args} \
    ${data_args} \
    ${stage1_train_args} \
    ${train_args}
