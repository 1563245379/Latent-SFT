import unittest
from unittest.mock import patch

from src.stage1.data import _load_stage1_data, _normalize_stage1_example


class Stage1DataNormalizationTest(unittest.TestCase):
    def test_normalizes_huggingface_gsm8k_aug_nl_example(self):
        example = {
            "question": "How many apples are left?",
            "cot": "Alice has 5 apples and gives away 2, so she has 3 apples.",
            "answer": "3",
        }

        normalized = _normalize_stage1_example(example, idx=0)

        self.assertEqual(normalized["problem"], "How many apples are left?")
        self.assertEqual(normalized["cot"], example["cot"])
        self.assertEqual(
            normalized["cot_answer"],
            "Alice has 5 apples and gives away 2, so she has 3 apples.\n\nTherefore, the final answer is \\boxed{3}.",
        )

    def test_loads_huggingface_dataset_id_with_requested_split(self):
        with patch("src.stage1.data.load_dataset", return_value=[{"row": 1}]) as load_dataset:
            data = _load_stage1_data("zen-E/GSM8k-Aug-NL", split="train")

        self.assertEqual(data, [{"row": 1}])
        load_dataset.assert_called_once_with("zen-E/GSM8k-Aug-NL", split="train")


if __name__ == "__main__":
    unittest.main()
