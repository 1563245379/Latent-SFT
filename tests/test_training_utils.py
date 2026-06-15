import unittest

from src.training_utils import split_train_validation_dataset


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


if __name__ == "__main__":
    unittest.main()
