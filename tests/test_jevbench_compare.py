from copy import deepcopy

import pytest

from jev_inference.jevbench_compare import compare


def result(model, backend, accuracy, latency, brier=None, ece=None, changes=0.1):
    summary = {}
    for method in ("labels", "independent"):
        summary[method] = {
            "correct": int(accuracy * 100),
            "scorable": 100,
            "accuracy": accuracy,
            "invalid_outputs": 0,
            "latency_ms_median": latency,
            "latency_ms_p95": latency * 2,
            "brier_mean": brier,
            "ece": ece,
            "order_sensitivity": {"decision_change_rate": changes},
        }
    return {
        "metadata": {
            "model": model,
            "backend": backend,
            "quantization": None if backend == "transformers" else "Q5_K_M",
            "dataset_sha256": "same",
        },
        "summary": summary,
    }


def test_compare_renders_side_by_side_accuracy_and_delta():
    reference = result("Qwen/Qwen2.5-1.5B-Instruct", "transformers", 0.5, 100.0, 0.6, 0.2, 0.3)
    candidate = result(".cache/qwen.gguf", "llama_cpp", 0.55, 80.0, 0.55, 0.15, 0.2)
    text = compare(reference, candidate)
    assert "Qwen/Qwen2.5-1.5B-Instruct" in text
    assert "llama_cpp" in text
    assert "Q5_K_M" in text
    assert "`labels`" in text
    assert "+5.00 pp" in text
    assert "N/A" not in text


def test_compare_rejects_different_datasets():
    reference = result("a", "transformers", 0.5, 1.0)
    candidate = result("b", "llama_cpp", 0.5, 1.0)
    candidate["metadata"]["dataset_sha256"] = "different"
    with pytest.raises(ValueError, match="dataset"):
        compare(reference, candidate)


def test_compare_handles_missing_calibration_as_na():
    reference = result("a", "transformers", 0.5, 1.0)
    candidate = result("b", "llama_cpp", 0.5, 1.0)
    candidate["summary"] = deepcopy(candidate["summary"])
    candidate["summary"]["labels"]["brier_mean"] = None
    candidate["summary"]["labels"]["ece"] = None
    text = compare(reference, candidate)
    assert "N/A" in text


def test_compare_reads_top_label_ece_from_summary_object():
    reference = result("a", "transformers", 0.5, 1.0, 0.6, {"ece": 0.2})
    candidate = result("b", "llama_cpp", 0.5, 1.0, 0.6, {"ece": 0.15})
    text = compare(reference, candidate)
    assert "0.2000" in text
    assert "0.1500" in text
    assert "-0.05" in text or "-0.0500" in text
