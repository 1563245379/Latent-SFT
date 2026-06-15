import tempfile
import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
