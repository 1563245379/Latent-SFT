import json
import logging
import math
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_TRAINER_STATE_NAME = "trainer_state.json"


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    match = _CHECKPOINT_RE.match(checkpoint_dir.name)
    if not match:
        return None
    return int(match.group(1))


def _read_checkpoint_metric(checkpoint_dir: Path, metric_name: str) -> float | None:
    state_path = checkpoint_dir / _TRAINER_STATE_NAME
    if not state_path.exists():
        return None

    try:
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read trainer state from %s: %s", state_path, exc)
        return None

    checkpoint_step = _checkpoint_step(checkpoint_dir)
    matching_logs = []

    for log_entry in trainer_state.get("log_history", []):
        if metric_name not in log_entry:
            continue
        metric_value = log_entry[metric_name]
        log_step = log_entry.get("step")
        if checkpoint_step is None or log_step is None or log_step <= checkpoint_step:
            matching_logs.append((log_step, metric_value))

    if not matching_logs:
        return None

    _, metric_value = matching_logs[-1]
    try:
        metric = float(metric_value)
    except (TypeError, ValueError):
        return None

    return metric if math.isfinite(metric) else None


def _list_checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    checkpoints = [
        path
        for path in root.glob("checkpoint-*")
        if path.is_dir() and _checkpoint_step(path) is not None
    ]
    return sorted(checkpoints, key=lambda path: _checkpoint_step(path) or -1)


def select_best_and_recent_checkpoints(
    output_dir: str | Path,
    metric_name: str,
    greater_is_better: bool,
    best_total_limit: int | None,
    recent_total_limit: int | None,
) -> list[Path]:
    checkpoints = _list_checkpoints(output_dir)
    best_limit = max(best_total_limit or 0, 0)
    recent_limit = max(recent_total_limit or 0, 0)

    keep: set[Path] = set()
    if recent_limit:
        keep.update(checkpoints[-recent_limit:])

    if best_limit:
        metric_checkpoints = []
        for checkpoint_dir in checkpoints:
            metric = _read_checkpoint_metric(checkpoint_dir, metric_name)
            if metric is None:
                continue
            step = _checkpoint_step(checkpoint_dir) or -1
            ranking_metric = metric if greater_is_better else -metric
            metric_checkpoints.append((ranking_metric, step, checkpoint_dir))

        metric_checkpoints.sort(reverse=True)
        keep.update(path for _, _, path in metric_checkpoints[:best_limit])

    return sorted(keep, key=lambda path: _checkpoint_step(path) or -1)


def rotate_best_and_recent_checkpoints(
    output_dir: str | Path,
    metric_name: str,
    greater_is_better: bool,
    best_total_limit: int | None,
    recent_total_limit: int | None,
) -> None:
    checkpoints = _list_checkpoints(output_dir)
    if not checkpoints:
        return

    keep = set(
        select_best_and_recent_checkpoints(
            output_dir=output_dir,
            metric_name=metric_name,
            greater_is_better=greater_is_better,
            best_total_limit=best_total_limit,
            recent_total_limit=recent_total_limit,
        )
    )
    for checkpoint_dir in checkpoints:
        if checkpoint_dir in keep:
            continue
        logger.info("Deleting checkpoint outside retention policy: %s", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)


def best_and_recent_checkpointing_enabled(args) -> bool:
    return (
        getattr(args, "save_best_total_limit", None) is not None
        or getattr(args, "save_recent_total_limit", None) is not None
    )


def prepare_best_and_recent_checkpointing(args) -> None:
    if not best_and_recent_checkpointing_enabled(args):
        return

    args.save_total_limit = None
    if getattr(args, "metric_for_best_model", None) is None:
        args.metric_for_best_model = "validation_accuracy"
    if getattr(args, "greater_is_better", None) is None:
        args.greater_is_better = True


def rotate_trainer_checkpoints_for_best_and_recent(trainer) -> None:
    args = trainer.args
    if not best_and_recent_checkpointing_enabled(args):
        return

    metric_name = getattr(args, "metric_for_best_model", None) or "validation_accuracy"
    greater_is_better = bool(getattr(args, "greater_is_better", True))
    rotate_best_and_recent_checkpoints(
        output_dir=args.output_dir,
        metric_name=metric_name,
        greater_is_better=greater_is_better,
        best_total_limit=getattr(args, "save_best_total_limit", None),
        recent_total_limit=getattr(args, "save_recent_total_limit", None),
    )
