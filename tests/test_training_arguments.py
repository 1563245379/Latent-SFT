import unittest

from transformers import HfArgumentParser

from src.stage1.arguments import (
    DataArguments as Stage1DataArguments,
    ModelArguments as Stage1ModelArguments,
    Stage1TrainingArguments,
)
from src.stage2.arguments import (
    DataArguments as Stage2DataArguments,
    ModelArguments as Stage2ModelArguments,
    Stage2TrainingArguments,
)


class TrainingArgumentsCompatibilityTest(unittest.TestCase):
    def test_stage1_parser_accepts_overwrite_output_dir(self):
        parser = HfArgumentParser(
            (Stage1ModelArguments, Stage1DataArguments, Stage1TrainingArguments)
        )

        _, _, training_args = parser.parse_args_into_dataclasses(
            [
                "--encoder_name_or_path",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--decoder_name_or_path",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--train_data_path",
                "zen-E/GSM8k-Aug-NL",
                "--output_dir",
                "/tmp/latent-sft-stage1",
                "--overwrite_output_dir",
            ]
        )

        self.assertTrue(training_args.overwrite_output_dir)

    def test_stage2_parser_accepts_overwrite_output_dir(self):
        parser = HfArgumentParser(
            (Stage2ModelArguments, Stage2DataArguments, Stage2TrainingArguments)
        )

        _, _, training_args = parser.parse_args_into_dataclasses(
            [
                "--latent_model_path",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--train_data_path",
                "/tmp/train.jsonl",
                "--train_latent_soft_label_path",
                "/tmp/latent",
                "--output_dir",
                "/tmp/latent-sft-stage2",
                "--overwrite_output_dir",
            ]
        )

        self.assertTrue(training_args.overwrite_output_dir)


if __name__ == "__main__":
    unittest.main()
