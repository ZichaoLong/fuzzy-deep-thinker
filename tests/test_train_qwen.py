from argparse import Namespace

from fdt.train_qwen import run_diagnostics


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
