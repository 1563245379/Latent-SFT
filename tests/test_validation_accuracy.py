import unittest

from src.validation import (
    _prepare_stage1_example,
    compute_accuracy_from_predictions,
    extract_validation_answer,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": [len(str(text))]}


class FakeStage1Model:
    encoder_name_or_path = "llama"
    tokenizer = FakeTokenizer()
    compress_token_id = 999
    latent_token_ids = [[101], [102]]


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

    def test_stage1_validation_accepts_huggingface_gsm8k_aug_nl_schema(self):
        prepared = _prepare_stage1_example(
            {
                "question": "How many apples are left?",
                "cot": "Alice has 5 apples and gives away 2, so she has 3 apples.",
                "answer": "3",
            },
            FakeStage1Model(),
            compression_rate=2,
        )

        self.assertEqual(prepared["input_ids"][1:], [101, -100, 102])
        self.assertEqual(prepared["cot_ids"][1], 101)
        self.assertEqual(prepared["cot_ids"][-2:], [999, 102])


if __name__ == "__main__":
    unittest.main()
