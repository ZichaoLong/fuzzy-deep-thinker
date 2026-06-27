from fdt.data import build_split, read_jsonl
from fdt.formats import continuous_item, format_text
from fdt.tasks import generate_easy_graph_reachability_fixed_nodes, generate_example
from fdt.tokenizer import CharTokenizer


def test_build_split_writes_jsonl(tmp_path):
    path = build_split(tmp_path, "graph_reachability", "train", 3, seed_start=10)
    examples = read_jsonl(path)
    assert len(examples) == 3
    assert examples[0].metadata["seed"] == 10


def test_build_split_accepts_easy_difficulty(tmp_path):
    path = build_split(tmp_path, "graph_reachability", "train", 3, seed_start=10, difficulty="easy")
    examples = read_jsonl(path)
    assert len(examples) == 3
    assert all(example.metadata["difficulty"] == "easy" for example in examples)


def test_build_split_accepts_hard_ladder_difficulty(tmp_path):
    path = build_split(tmp_path, "graph_reachability", "train", 6, seed_start=0, difficulty="hard_ladder")
    examples = read_jsonl(path)
    assert len(examples) == 6
    assert all(example.metadata["difficulty"] == "hard_ladder" for example in examples)


def test_easy_ladder_uses_size_curriculum_and_balanced_labels(tmp_path):
    train_path = build_split(tmp_path, "graph_reachability", "train", 12, seed_start=0, difficulty="easy_ladder")
    ood_path = build_split(tmp_path, "graph_reachability", "ood_test", 8, seed_start=0, difficulty="easy_ladder")

    train_examples = read_jsonl(train_path)
    ood_examples = read_jsonl(ood_path)

    assert {example.metadata["num_nodes"] for example in train_examples} == {4, 5, 6}
    assert {example.metadata["num_nodes"] for example in ood_examples} == {7, 8}
    assert {example.answer for example in train_examples} == {"YES", "NO"}
    assert all(example.metadata["difficulty"] == "easy_ladder" for example in train_examples + ood_examples)


def test_fixed_node_easy_graph_diagnostic_examples():
    example = generate_easy_graph_reachability_fixed_nodes(seed=123, n=7)
    assert example.metadata["num_nodes"] == 7
    assert example.metadata["difficulty"] == "easy"
    assert example.answer in {"YES", "NO"}


def test_training_formats_have_expected_loss_boundaries():
    example = generate_example("graph_reachability", seed=0, split="train")

    direct = format_text(example, "direct")
    assert direct.text[direct.loss_start :].startswith(example.answer)

    cot = format_text(example, "cot")
    assert cot.text[cot.loss_start :].startswith(example.trace)
    assert f"Answer: {example.answer}" in cot.text

    masked = format_text(example, "masked_cot")
    assert masked.text[masked.loss_start :].startswith(example.answer)
    assert example.trace in masked.text[: masked.loss_start]

    latent = continuous_item(example)
    assert latent.prefix.endswith("Answer: ")
    assert latent.answer.startswith(example.answer)


def test_char_tokenizer_round_trips_text():
    tokenizer = CharTokenizer.build(["abc", "bcd"])
    text = "abcd"
    assert tokenizer.decode(tokenizer.encode(text)) == text
