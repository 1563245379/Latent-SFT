import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers.trainer import Trainer

from src.stage1.arguments import Stage1TrainingArguments
from src.stage1.trainer import Stage1EncoderTrainer
from src.stage2.arguments import Stage2TrainingArguments
from src.stage2.trainer import Stage2Trainer


class TrainerCompatibilityTest(unittest.TestCase):
    def test_stage1_trainer_accepts_legacy_tokenizer_keyword(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Stage1EncoderTrainer(
                model=torch.nn.Linear(1, 1),
                args=Stage1TrainingArguments(output_dir=tmpdir, report_to=[]),
                tokenizer=object(),
            )

        self.assertIsNotNone(trainer)

    def test_stage2_trainer_accepts_legacy_tokenizer_keyword(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Stage2Trainer(
                model=torch.nn.Linear(1, 1),
                args=Stage2TrainingArguments(output_dir=tmpdir, report_to=[]),
                tokenizer=object(),
            )

        self.assertIsNotNone(trainer)

    def test_stage1_trainer_skips_unsafe_rng_state_resume_on_old_torch(self):
        error = ValueError(
            "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, "
            "we now require users to upgrade torch to at least v2.6 in order to use the function."
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Stage1EncoderTrainer(
                model=torch.nn.Linear(1, 1),
                args=Stage1TrainingArguments(output_dir=tmpdir, report_to=[]),
            )

            with patch.object(Trainer, "_load_rng_state", side_effect=error):
                trainer._load_rng_state(f"{tmpdir}/checkpoint-1")

    def test_stage2_trainer_reraises_unrelated_rng_state_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Stage2Trainer(
                model=torch.nn.Linear(1, 1),
                args=Stage2TrainingArguments(output_dir=tmpdir, report_to=[]),
            )

            with patch.object(Trainer, "_load_rng_state", side_effect=ValueError("different failure")):
                with self.assertRaisesRegex(ValueError, "different failure"):
                    trainer._load_rng_state(f"{tmpdir}/checkpoint-1")


if __name__ == "__main__":
    unittest.main()
