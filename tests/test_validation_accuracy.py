import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import src.validation as validation
from src.validation import (
    _prepare_stage1_example,
    compute_accuracy_from_predictions,
    extract_validation_answer,
    update_checkpoint_validation_accuracy,
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


class RawValidationDataset:
    def __init__(self, data):
        self.data = data


class FakeDistributedValidationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.seen_answers = []


def _free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_validation_worker(rank, world_size, port, result_queue):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    model = FakeDistributedValidationModel()
    dataset = RawValidationDataset(
        [
            {"answer": "0"},
            {"answer": "1"},
            {"answer": "2"},
            {"answer": "3"},
        ]
    )

    def fake_stage1_predictions(model, examples, **kwargs):
        model.seen_answers.extend(example["answer"] for example in examples)
        return [f"Therefore, the final answer is \\boxed{{{example['answer']}}}." for example in examples]

    try:
        with patch.object(
            validation,
            "_generate_stage1_predictions",
            side_effect=fake_stage1_predictions,
        ):
            accuracy = validation.compute_validation_accuracy(
                model=model,
                validation_dataset=dataset,
                stage="stage1_decoder",
                data_args=SimpleNamespace(compression_rate=2),
                batch_size=2,
                max_new_tokens=8,
            )
        result_queue.put((rank, accuracy, model.seen_answers))
    finally:
        dist.destroy_process_group()


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

    def test_validation_examples_are_sharded_across_ranks(self):
        examples = list(range(7))

        shards = [
            validation._rank_shard_examples(examples, rank=rank, world_size=3)
            for rank in range(3)
        ]

        self.assertEqual(shards, [[0, 3, 6], [1, 4], [2, 5]])
        self.assertEqual(
            sorted(example for shard in shards for example in shard),
            examples,
        )

    def test_nonzero_rank_participates_in_checkpoint_validation(self):
        calls = []
        trainer = SimpleNamespace(
            model=object(),
            args=SimpleNamespace(
                validation_batch_size=2,
                validation_max_new_tokens=8,
                output_dir="/tmp/latent-sft-validation-test",
            ),
            state=SimpleNamespace(global_step=1),
            is_world_process_zero=lambda: False,
            log=lambda metrics: self.fail(f"nonzero rank should not log metrics: {metrics}"),
        )

        with patch.object(validation, "_dist_barrier"), patch.object(
            validation,
            "compute_validation_accuracy",
            side_effect=lambda **kwargs: calls.append(kwargs) or 0.5,
        ):
            update_checkpoint_validation_accuracy(
                trainer,
                validation_dataset=[{"answer": "1"}],
                validation_stage="stage1_decoder",
                data_args=SimpleNamespace(compression_rate=2),
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["stage"], "stage1_decoder")

    def test_compute_validation_accuracy_parallelizes_across_distributed_ranks(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")

        world_size = 2
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        port = _free_tcp_port()
        processes = [
            context.Process(
                target=_distributed_validation_worker,
                args=(rank, world_size, port, result_queue),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()

        results = [result_queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)

        for process in processes:
            self.assertEqual(process.exitcode, 0)

        by_rank = {rank: (accuracy, seen_answers) for rank, accuracy, seen_answers in results}
        self.assertEqual(by_rank[0], (1.0, ["0", "2"]))
        self.assertEqual(by_rank[1], (1.0, ["1", "3"]))


if __name__ == "__main__":
    unittest.main()
