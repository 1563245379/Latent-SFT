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
from src.training_utils import parse_project_debug_flag


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

    def test_stage1_parser_accepts_decoder_checkpoint_retention_arguments(self):
        parser = HfArgumentParser(
            (Stage1ModelArguments, Stage1DataArguments, Stage1TrainingArguments)
        )

        _, data_args, training_args = parser.parse_args_into_dataclasses(
            [
                "--encoder_name_or_path",
                "encoder",
                "--decoder_name_or_path",
                "decoder",
                "--train_data_path",
                "/tmp/train.jsonl",
                "--output_dir",
                "/tmp/latent-sft-stage1-decoder",
                "--validation_split_ratio",
                "0.2",
                "--save_best_total_limit",
                "3",
                "--save_recent_total_limit",
                "1",
                "--validation_batch_size",
                "4",
            ]
        )

        self.assertEqual(data_args.validation_split_ratio, 0.2)
        self.assertEqual(training_args.save_best_total_limit, 3)
        self.assertEqual(training_args.save_recent_total_limit, 1)
        self.assertEqual(training_args.validation_batch_size, 4)

    def test_stage2_parser_accepts_validation_split_and_checkpoint_retention_arguments(self):
        parser = HfArgumentParser(
            (Stage2ModelArguments, Stage2DataArguments, Stage2TrainingArguments)
        )

        _, data_args, training_args = parser.parse_args_into_dataclasses(
            [
                "--latent_model_path",
                "latent",
                "--train_data_path",
                "/tmp/train.jsonl",
                "--train_latent_soft_label_path",
                "/tmp/latent",
                "--output_dir",
                "/tmp/latent-sft-stage2",
                "--validation_split_ratio",
                "0.1",
                "--save_best_total_limit",
                "3",
                "--save_recent_total_limit",
                "1",
                "--validation_batch_size",
                "2",
            ]
        )

        self.assertEqual(data_args.validation_split_ratio, 0.1)
        self.assertEqual(training_args.save_best_total_limit, 3)
        self.assertEqual(training_args.save_recent_total_limit, 1)
        self.assertEqual(training_args.validation_batch_size, 2)

    def test_stage1_parser_accepts_debug_argument(self):
        parser = HfArgumentParser(
            (Stage1ModelArguments, Stage1DataArguments, Stage1TrainingArguments)
        )

        _, _, training_args = parser.parse_args_into_dataclasses(
            [
                "--encoder_name_or_path",
                "encoder",
                "--decoder_name_or_path",
                "decoder",
                "--train_data_path",
                "/tmp/train.jsonl",
                "--output_dir",
                "/tmp/latent-sft-stage1-debug",
                "--training_debug",
                "True",
            ]
        )

        self.assertTrue(training_args.training_debug)

    def test_stage2_parser_accepts_debug_argument(self):
        parser = HfArgumentParser(
            (Stage2ModelArguments, Stage2DataArguments, Stage2TrainingArguments)
        )

        _, _, training_args = parser.parse_args_into_dataclasses(
            [
                "--latent_model_path",
                "latent",
                "--train_data_path",
                "/tmp/train.jsonl",
                "--train_latent_soft_label_path",
                "/tmp/latent",
                "--output_dir",
                "/tmp/latent-sft-stage2-debug",
                "--training_debug",
                "True",
            ]
        )

        self.assertTrue(training_args.training_debug)

    def test_project_debug_flag_consumes_boolean_debug_alias(self):
        args, debug = parse_project_debug_flag(
            [
                "--output_dir",
                "/tmp/latent-sft-debug",
                "--debug",
                "True",
                "--save_strategy",
                "epoch",
            ]
        )

        self.assertTrue(debug)
        self.assertEqual(
            args,
            [
                "--output_dir",
                "/tmp/latent-sft-debug",
                "--save_strategy",
                "epoch",
            ],
        )


if __name__ == "__main__":
    unittest.main()
