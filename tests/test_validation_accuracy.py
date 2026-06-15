import unittest

from src.validation import compute_accuracy_from_predictions, extract_validation_answer


class ValidationAccuracyTest(unittest.TestCase):
    def test_extracts_answer_from_training_cot_answer_when_answer_field_is_absent(self):
        answer = extract_validation_answer(
            {
                "problem": "1+1?",
                "cot_answer": "We add the numbers.\n\nTherefore, the final answer is \\boxed{2}.",
            }
        )

        self.assertEqual(answer, "2")

    def test_computes_accuracy_from_generated_predictions(self):
        accuracy = compute_accuracy_from_predictions(
            predictions=[
                "Reasoning... Therefore, the final answer is \\boxed{2}.",
                "Reasoning... Therefore, the final answer is \\boxed{5}.",
            ],
            examples=[
                {"answer": "2"},
                {"answer": "4"},
            ],
        )

        self.assertEqual(accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
