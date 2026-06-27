import json

from fdt.tasks import generate_example, list_tasks, parse_arithmetic_answer, verify_answer


def test_generators_are_deterministic():
    for task in list_tasks():
        first = generate_example(task, seed=123, split="train")
        second = generate_example(task, seed=123, split="train")
        assert first == second


def test_examples_are_json_serializable_and_self_verifying():
    for task in list_tasks():
        for split in ["train", "dev", "id_test", "ood_test"]:
            example = generate_example(task, seed=42, split=split)
            payload = json.loads(example.to_json())
            assert payload["prompt"]
            assert payload["trace"]
            assert payload["answer"]
            assert payload["metadata"]["task"] == task
            assert verify_answer(example, example.answer)


def test_ood_examples_are_larger_or_deeper():
    graph_train = generate_example("graph_reachability", seed=1, split="train")
    graph_ood = generate_example("graph_reachability", seed=1, split="ood_test")
    assert graph_ood.metadata["num_nodes"] >= graph_train.metadata["num_nodes"]

    pointer_train = generate_example("pointer_chasing", seed=1, split="train")
    pointer_ood = generate_example("pointer_chasing", seed=1, split="ood_test")
    assert pointer_ood.metadata["depth"] > pointer_train.metadata["depth"]

    expr_train = generate_example("symbolic_arithmetic", seed=1, split="train")
    expr_ood = generate_example("symbolic_arithmetic", seed=1, split="ood_test")
    assert expr_ood.metadata["max_depth"] > expr_train.metadata["max_depth"]


def test_easy_graph_reachability_is_small_and_binary():
    example = generate_example("graph_reachability", seed=7, split="train", difficulty="easy")
    assert example.metadata["difficulty"] == "easy"
    assert example.metadata["num_nodes"] == 4
    assert example.answer in {"YES", "NO"}

    ood = generate_example("graph_reachability", seed=7, split="ood_test", difficulty="easy")
    assert ood.metadata["num_nodes"] == 6
    assert ood.answer in {"YES", "NO"}


def test_hard_ladder_graph_reachability_controls_size_path_and_labels():
    train_examples = [
        generate_example("graph_reachability", seed=i, split="train", difficulty="hard_ladder") for i in range(60)
    ]
    ood_examples = [
        generate_example("graph_reachability", seed=i, split="ood_test", difficulty="hard_ladder") for i in range(100)
    ]

    assert {example.metadata["num_nodes"] for example in train_examples} == {6, 7, 8, 9, 10}
    assert {example.metadata["num_nodes"] for example in ood_examples} == {12, 13, 14, 15, 16}
    assert {example.answer for example in train_examples} == {"YES", "NO"}
    assert {example.answer for example in ood_examples} == {"YES", "NO"}
    assert all(example.metadata["difficulty"] == "hard_ladder" for example in train_examples + ood_examples)

    train_yes_paths = {
        example.metadata["path_length"] for example in train_examples if example.answer == "YES"
    }
    ood_yes_paths = {
        example.metadata["path_length"] for example in ood_examples if example.answer == "YES"
    }
    assert train_yes_paths == {1, 2, 3}
    assert ood_yes_paths == {4, 5, 6, 7, 8}


def test_pointer_chasing_depth_ladder_and_balanced_labels():
    train_examples = [generate_example("pointer_chasing", seed=i, split="train") for i in range(12)]
    ood_examples = [generate_example("pointer_chasing", seed=i, split="ood_test") for i in range(16)]

    assert {example.metadata["depth"] for example in train_examples} == {2, 3, 4}
    assert {example.metadata["depth"] for example in ood_examples} == {5, 6, 7, 8}
    assert {example.answer for example in train_examples} == {"YES", "NO"}
    assert {example.answer for example in ood_examples} == {"YES", "NO"}


def test_simple_pointer_chasing_is_shorter_and_shallower():
    train_examples = [
        generate_example("pointer_chasing", seed=i, split="train", difficulty="simple") for i in range(8)
    ]
    ood_examples = [
        generate_example("pointer_chasing", seed=i, split="ood_test", difficulty="simple") for i in range(8)
    ]

    assert {example.metadata["depth"] for example in train_examples} == {1}
    assert {example.metadata["depth"] for example in ood_examples} == {2, 3}
    assert {example.metadata["num_states"] for example in train_examples + ood_examples} == {4}
    assert {example.answer for example in train_examples} == {"YES", "NO"}
    assert all(example.metadata["difficulty"] == "simple" for example in train_examples + ood_examples)


def test_arithmetic_answer_parser_accepts_simple_expressions():
    assert parse_arithmetic_answer("Answer: (3+5)-2") == 6
    assert parse_arithmetic_answer("not arithmetic") is None
