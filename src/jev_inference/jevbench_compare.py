"""Render a side-by-side comparison of two JevBench result documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NA = "N/A"


def _format(value, digits: int = 4) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table_row(label: str, left, right, delta: str | None = None) -> str:
    cells = [label, _format(left), _format(right)]
    if delta is not None:
        cells.append(delta)
    return "| " + " | ".join(cells) + " |"


def _signed(value: float | None, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return NA
    return f"{value:+.{digits}f}{suffix}"


def _delta(reference, candidate, scale: float = 1.0, digits: int = 4, suffix: str = "") -> str:
    if reference is None or candidate is None:
        return NA
    return _signed((candidate - reference) * scale, digits, suffix)


def _method_summary(result: dict, method: str) -> dict:
    return result.get("summary", {}).get(method, {})


def _order_rate(summary: dict):
    order = summary.get("order_sensitivity") or {}
    return order.get("decision_change_rate")


def _metric(summary: dict, key: str):
    if key == "order_sensitivity":
        return _order_rate(summary)
    value = summary.get(key)
    if key == "ece" and isinstance(value, dict):
        return value.get("ece")
    return value


def compare(reference: dict, candidate: dict) -> str:
    ref_meta = reference.get("metadata", {})
    cand_meta = candidate.get("metadata", {})
    if ref_meta.get("dataset_sha256") != cand_meta.get("dataset_sha256"):
        raise ValueError(
            "cannot compare results from different datasets: "
            f"{ref_meta.get('dataset_sha256')!r} != {cand_meta.get('dataset_sha256')!r}"
        )

    lines = [
        "# JevBench backend comparison",
        "",
        "| Field | Reference | Candidate |",
        "| --- | --- | --- |",
        _table_row("Model", ref_meta.get("model"), cand_meta.get("model")),
        _table_row("Backend", ref_meta.get("backend", "transformers"), cand_meta.get("backend")),
        _table_row(
            "Quantization",
            ref_meta.get("quantization") or "none",
            cand_meta.get("quantization") or "none",
        ),
        _table_row(
            "Dataset SHA-256",
            ref_meta.get("dataset_sha256"),
            cand_meta.get("dataset_sha256"),
        ),
    ]
    ref_gpu = ref_meta.get("gpu")
    cand_gpu = cand_meta.get("gpu")
    if ref_gpu or cand_gpu:
        lines.append(_table_row("GPU", ref_gpu, cand_gpu))
    ref_cpu = (ref_meta.get("host") or {}).get("cpu_model")
    cand_cpu = (cand_meta.get("host") or {}).get("cpu_model")
    if ref_cpu or cand_cpu:
        lines.append(_table_row("CPU", ref_cpu, cand_cpu))
    lines.append("")
    if ref_gpu and ref_gpu == cand_gpu and ref_cpu == cand_cpu:
        ref_precision = ref_meta.get("dtype") or "unspecified precision"
        cand_precision = cand_meta.get("quantization") or cand_meta.get("dtype") or "unspecified"
        lines.extend(
            [
                f"> Both runs used the same GPU and host ({ref_gpu}, {ref_cpu}). "
                "Latency deltas therefore reflect the runtime and precision choices "
                f"(reference: {ref_precision}; candidate: {cand_precision}), "
                "not different hardware.",
                "",
            ]
        )

    methods = list(reference.get("summary", {}))
    for method in candidate.get("summary", {}):
        if method not in methods:
            methods.append(method)

    for method in methods:
        ref = _method_summary(reference, method)
        cand = _method_summary(candidate, method)
        lines.extend(
            [
                f"## `{method}`",
                "",
                "| Metric | Reference | Candidate | Delta |",
                "| --- | --- | --- | --- |",
            ]
        )
        for label, key, scale, digits, suffix in (
            ("Accuracy", "accuracy", 100.0, 2, " pp"),
            ("Brier score", "brier_mean", 1.0, 4, ""),
            ("Top-label ECE", "ece", 1.0, 4, ""),
            ("Order change rate", "order_sensitivity", 100.0, 2, " pp"),
            ("Latency median (ms)", "latency_ms_median", 1.0, 2, " ms"),
            ("Latency p95 (ms)", "latency_ms_p95", 1.0, 2, " ms"),
        ):
            ref_value = _metric(ref, key)
            cand_value = _metric(cand, key)
            lines.append(
                _table_row(
                    label,
                    _format(
                        None
                        if ref_value is None
                        else ref_value * (100.0 if suffix == " pp" else 1.0),
                        digits,
                    ),
                    _format(
                        None
                        if cand_value is None
                        else cand_value * (100.0 if suffix == " pp" else 1.0),
                        digits,
                    ),
                    _delta(ref_value, cand_value, scale, digits, suffix),
                )
            )
        extras = []
        for label, key in (
            ("Scorable", "scorable"),
            ("Correct", "correct"),
            ("Invalid outputs", "invalid_outputs"),
            ("Operational failures", "operational_failures"),
        ):
            if key in ref or key in cand:
                extras.append(_table_row(label, ref.get(key), cand.get(key)))
        lines.extend([*extras, ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two JevBench result documents")
    parser.add_argument("reference", help="Reference result JSON (for example Transformers BF16)")
    parser.add_argument("candidate", help="Candidate result JSON (for example GGUF llama.cpp)")
    parser.add_argument("--output", help="Write Markdown to this path instead of stdout")
    args = parser.parse_args()
    text = compare(
        json.loads(Path(args.reference).read_text()),
        json.loads(Path(args.candidate).read_text()),
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Saved {path}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
