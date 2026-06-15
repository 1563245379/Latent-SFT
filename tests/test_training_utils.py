import unittest
from types import SimpleNamespace

from src.training_utils import apply_debug_training_limits, split_train_validation_dataset
from src.training_utils import get_resume_from_checkpoint, should_block_existing_output_dir


class DatasetSplitTest(unittest.TestCase):
    def test_validation_split_ratio_carves_validation_subset_from_training_data(self):
        train_dataset, validation_dataset = split_train_validation_dataset(
            list(range(10)),
            validation_split_ratio=0.3,
            seed=13,
        )

        self.assertEqual(len(train_dataset), 7)
        self.assertEqual(len(validation_dataset), 3)

    def test_zero_validation_split_keeps_existing_train_only_behavior(self):
        dataset = list(range(10))

        train_dataset, validation_dataset = split_train_validation_dataset(
            dataset,
            validation_split_ratio=0.0,
            seed=13,
        )

        self.assertIs(train_dataset, dataset)
        self.assertIsNone(validation_dataset)

    def test_validation_split_ratio_must_leave_training_examples(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            split_train_validation_dataset(
                list(range(3)),
                validation_split_ratio=1.0,
                seed=13,
            )

    def test_debug_limits_training_dataset_to_200_examples_and_three_epochs(self):
        training_args = SimpleNamespace(debug=True, num_train_epochs=10, max_steps=500)

        debug_dataset = apply_debug_training_limits(
            list(range(500)),
            training_args=training_args,
        )

        self.assertEqual(len(debug_dataset), 200)
        self.assertEqual(training_args.num_train_epochs, 3)
        self.assertEqual(training_args.max_steps, -1)

    def test_debug_leaves_short_training_dataset_length_unchanged(self):
        training_args = SimpleNamespace(debug=True, num_train_epochs=10)

        debug_dataset = apply_debug_training_limits(
            list(range(50)),
            training_args=training_args,
        )

        self.assertEqual(len(debug_dataset), 50)
        self.assertEqual(training_args.num_train_epochs, 3)

    def test_existing_output_dir_is_allowed_when_resuming_from_checkpoint(self):
        training_args = SimpleNamespace(
            output_dir="/tmp/non-empty-run",
            do_train=True,
            overwrite_output_dir=False,
            resume_from_checkpoint="/tmp/non-empty-run/checkpoint-100",
        )

        self.assertFalse(should_block_existing_output_dir(training_args, output_dir_has_contents=True))

    def test_existing_output_dir_still_blocks_without_resume_or_overwrite(self):
        training_args = SimpleNamespace(
            output_dir="/tmp/non-empty-run",
            do_train=True,
            overwrite_output_dir=False,
            resume_from_checkpoint=None,
        )

        self.assertTrue(should_block_existing_output_dir(training_args, output_dir_has_contents=True))

    def test_resume_checkpoint_path_is_forwarded_to_trainer_train(self):
        training_args = SimpleNamespace(resume_from_checkpoint="/tmp/run/checkpoint-42")

        self.assertEqual(
            get_resume_from_checkpoint(training_args),
            "/tmp/run/checkpoint-42",
        )


if __name__ == "__main__":
    unittest.main()
