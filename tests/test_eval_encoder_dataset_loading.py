import unittest
from unittest.mock import patch

from eval import eval_encoder_hf_batch as eval_encoder


class EvalEncoderDatasetLoadingTest(unittest.TestCase):
    def test_loads_openai_gsm8k_test_split_and_normalizes_fields(self):
        rows = [
            {
                "question": "Jan has 2 apples and buys 3 more. How many apples?",
                "answer": "Jan has 2 + 3 = 5 apples.\n#### 5",
            }
        ]

        with patch.object(
            eval_encoder, "load_dataset", return_value=rows, create=True
        ) as load_dataset:
            data = eval_encoder.load_eval_data("openai/gsm8k")

        self.assertEqual(
            data,
            [
                {
                    "problem": "Jan has 2 apples and buys 3 more. How many apples?",
                    "solution": "Jan has 2 + 3 = 5 apples.\n#### 5",
                    "answer": "5",
                }
            ],
        )
        load_dataset.assert_called_once_with("openai/gsm8k", "main", split="test")

    def test_loads_math500_test_split_and_normalizes_fields(self):
        rows = [
            {
                "problem": "Compute 6^2.",
                "solution": "We have 6^2 = 36.",
                "answer": "36",
                "subject": "Algebra",
            }
        ]

        with patch.object(
            eval_encoder, "load_dataset", return_value=rows, create=True
        ) as load_dataset:
            data = eval_encoder.load_eval_data("HuggingFaceH4/MATH-500")

        self.assertEqual(
            data,
            [
                {
                    "problem": "Compute 6^2.",
                    "solution": "We have 6^2 = 36.",
                    "answer": "36",
                    "subject": "Algebra",
                }
            ],
        )
        load_dataset.assert_called_once_with("HuggingFaceH4/MATH-500", split="test")


if __name__ == "__main__":
    unittest.main()
