# Repository Guidelines

## Project Structure & Module Organization

This repository implements the Latent-SFT two-stage training and evaluation pipeline. Core model code lives in `src/modeling/`, with Stage 1 data, argument, and trainer logic in `src/stage1/` and Stage 2 equivalents in `src/stage2/`. Launch scripts are split between Python entrypoints and task-specific shell wrappers in `script/`. Evaluation entrypoints live in `eval/`, shared grading/parsing helpers in `eval_utils/`, and figures/assets in `figs/`. Training and evaluation data should be placed under a repository-level `data/` directory, which may be absent in fresh checkouts. The customized SGLang dependency is kept separately in `sglang_latent_reasoning_pkg/`; follow its own tooling when editing that package.

## Build, Test, and Development Commands

- `conda create -n latent-sft python=3.12 -y` then `pip install -r requirements.txt`: create the reference training environment.
- `bash script/run_distill_stage1_encoder_gsm8k.sh`: run a task-specific Stage 1 encoder job; use the matching decoder and union scripts next.
- `python generate_latent_soft_label_lora_batch.py --help`: inspect latent soft-label generation options before producing `batch_*.pt` chunks.
- `python merge_lora.py --help`: inspect LoRA merge options before Stage 2.
- `bash script/run_distill_stage2_gsm8k.sh`: launch Stage 2 training after updating data/checkpoint paths.
- `python eval/eval_latent_model_hf_batch.py --help`: validate Transformer-based evaluation options. Use `eval/eval_*_sglang.py` only after installing `sglang_latent_reasoning_pkg/`.

## Coding Style & Naming Conventions

Use Python 3.12 and 4-space indentation. Follow existing `snake_case` names for functions, variables, and CLI arguments, and `PascalCase` for model/trainer classes. Keep Stage 1 and Stage 2 concerns separated; add shared evaluation logic to `eval_utils/` rather than duplicating parser or grader code. There is no root formatter configuration, so keep imports readable and avoid unrelated style-only churn.

## Testing Guidelines

There is no root pytest suite. Treat focused evaluation runs as regression checks and record the exact command, dataset, checkpoint, `compression_rate`, and `topk_interpolation`. For lightweight syntax validation after edits, run `python -m compileall src script eval eval_utils` before expensive GPU jobs.

## Commit & Pull Request Guidelines

Recent history uses short messages such as `Fix boolean CLI flags and update Latent-SFT docs`, `Update README.md`, and `Add SGLang latent reasoning package`. Prefer concise imperative commits (`Fix ...`, `Update ...`, `Add ...`) with the affected area named. Pull requests should describe the pipeline stage touched, list required data/checkpoint assumptions, include evaluation or compile commands run, and avoid committing datasets, generated checkpoints, latent-label chunks, or credentials.

## Agent-Specific Instructions

In Windows/PowerShell sessions, do not use `rg` or ripgrep. Prefer `Get-ChildItem`, `Select-String`, `Get-Content`, and `Where-Object` for repository search and file inspection.
