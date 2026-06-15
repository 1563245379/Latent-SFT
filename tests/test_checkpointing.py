import json
import tempfile
import unittest
from pathlib import Path

from src.checkpointing import (
    rotate_best_and_recent_checkpoints,
    select_best_and_recent_checkpoints,
)


class CheckpointRetentionTest(unittest.TestCase):
    def _write_checkpoint(self, root, step, validation_accuracy):
        checkpoint_dir = Path(root) / f"checkpoint-{step}"
        checkpoint_dir.mkdir()
        trainer_state = {
            "global_step": step,
            "log_history": [{"step": step, "validation_accuracy": validation_accuracy}],
        }
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state),
            encoding="utf-8",
        )
        return checkpoint_dir

    def test_selects_three_highest_validation_accuracy_checkpoints_plus_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for step, validation_accuracy in [
                (10, 0.50),
                (20, 0.80),
                (30, 0.60),
                (40, 0.90),
                (50, 0.70),
            ]:
                self._write_checkpoint(tmpdir, step, validation_accuracy)

            keep = select_best_and_recent_checkpoints(
                tmpdir,
                metric_name="validation_accuracy",
                greater_is_better=True,
                best_total_limit=3,
                recent_total_limit=1,
            )

        self.assertEqual(
            {Path(path).name for path in keep},
            {"checkpoint-20", "checkpoint-40", "checkpoint-50"},
        )

    def test_rotation_removes_checkpoints_outside_best_and_recent_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for step, validation_accuracy in [
                (10, 0.50),
                (20, 0.80),
                (30, 0.60),
                (40, 0.90),
                (50, 0.70),
            ]:
                self._write_checkpoint(tmpdir, step, validation_accuracy)

            rotate_best_and_recent_checkpoints(
                tmpdir,
                metric_name="validation_accuracy",
                greater_is_better=True,
                best_total_limit=3,
                recent_total_limit=1,
            )

            remaining = sorted(path.name for path in Path(tmpdir).glob("checkpoint-*"))

        self.assertEqual(
            remaining,
            ["checkpoint-20", "checkpoint-40", "checkpoint-50"],
        )


if __name__ == "__main__":
    unittest.main()
