from argparse import Namespace

from fdt.data import split_sizes
from fdt.tasks import generate_example
from fdt.train_qwen import make_training_sampler, records_to_metric, run_diagnostics, select_balanced_examples


def test_qwen_diagnostics_are_grouped_from_existing_records():
    eval_results = {
        "dev": {
            "records": [
                {
                    "expected": "YES",
                    "parsed": "YES",
                    "ok": True,
                    "metadata": {"num_nodes": 6},
                },
                {
                    "expected": "YES",
                    "parsed": "NO",
                    "ok": False,
                    "metadata": {"num_nodes": 6},
                },
                {
                    "expected": "NO",
                    "parsed": "NO",
                    "ok": True,
                    "metadata": {"num_nodes": 7},
                },
            ]
        }
    }
    args = Namespace(diagnostic_metadata_keys="answer,num_nodes", case_examples=1)

    diagnostics = run_diagnostics(args, eval_results)

    assert diagnostics["dev_answer_YES"]["accuracy"] == 0.5
    assert diagnostics["dev_answer_NO"]["accuracy"] == 1.0
    assert diagnostics["dev_num_nodes_6"]["accuracy"] == 0.5
    assert diagnostics["dev_num_nodes_7"]["accuracy"] == 1.0
    assert "records" not in diagnostics["dev_answer_YES"]


def test_balanced_answer_sampler_alternates_binary_labels():
    examples = [
        generate_example("graph_reachability", seed=i, split="train", difficulty="hard_ladder") for i in range(60)
    ]
    sampler = make_training_sampler(examples, "balanced_answer", seed=0)

    sampled = [sampler(step, micro_idx).answer for step in range(1, 11) for micro_idx in range(4)]

    assert sampled.count("YES") == sampled.count("NO")


def test_large_split_sizes_are_substantially_larger_than_debug():
    debug = split_sizes("debug")
    large = split_sizes("large")

    assert large.train >= 50 * debug.train
    assert large.dev > debug.dev
    assert large.id_test > debug.id_test
    assert large.ood_test > debug.ood_test


def test_select_balanced_examples_keeps_binary_labels():
    examples = [
        generate_example("graph_reachability", seed=i, split="train", difficulty="hard_ladder") for i in range(60)
    ]

    selected = select_balanced_examples(examples, count=20, seed=0)

    assert len(selected) == 20
    assert {example.answer for example in selected} == {"YES", "NO"}


def test_records_to_metric_reports_prediction_counts_and_margin():
    metric = records_to_metric(
        [
            {"expected": "YES", "parsed": "YES", "scores": {"YES": 0.1, "NO": 0.8}, "ok": True},
            {"expected": "NO", "parsed": "YES", "scores": {"YES": 0.2, "NO": 0.7}, "ok": False},
            {"expected": "NO", "parsed": "NO", "scores": {"YES": 0.9, "NO": 0.3}, "ok": True},
        ],
        case_examples=0,
    )

    assert metric["accuracy"] == 2 / 3
    assert metric["expected_counts"] == {"NO": 2, "YES": 1}
    assert metric["prediction_counts"] == {"NO": 1, "YES": 2}
    assert metric["mean_yes_minus_no_nll"] == ((0.1 - 0.8) + (0.2 - 0.7) + (0.9 - 0.3)) / 3
