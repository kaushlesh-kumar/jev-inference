"""Render measured public JevBench diagnostics; never an official leaderboard score."""

import argparse
import json
from pathlib import Path


def number(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def render(results):
    lines = [
        "## JevBench public-subset diagnostics",
        "",
        "Public tasks from [JevBench](https://github.com/fstandhartinger/jevbench), pinned to "
        "`bb05a335bc809e61b20c0f745d25499a82b326fc`. This is a public-subset experiment, "
        "not an official leaderboard result. Gold answers and provenance never enter prompts.",
        "",
        "| Model | Method | Scorer correct | Median ms | p95 ms | Peak MiB | Brier | ECE | "
        "Order changes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        model = result["metadata"]["model"]
        for method, row in result["summary"].items():
            ece = row["ece"]
            ece = ece["ece"] if isinstance(ece, dict) else ece
            lines.append(
                f"| {model} | `{method}` | {row['correct']}/{row['scorable']} | "
                f"{number(row['latency_ms_median'])} | {number(row['latency_ms_p95'])} | "
                f"{number(row['max_peak_allocated_mib'], 1)} | {number(row['brier_mean'], 3)} | "
                f"{number(ece, 3)} | "
                f"{percent(row['order_sensitivity']['decision_change_rate'])} |"
            )
    for result in results:
        summary = result["summary"]
        required = {"labels", "generate_label", "sequence_likelihood", "independent"}
        if required.issubset(summary):
            labels, generation = summary["labels"], summary["generate_label"]
            sequence, independent = summary["sequence_likelihood"], summary["independent"]
            ratio = generation["latency_ms_median"] / labels["latency_ms_median"]
            lines += [
                "",
                f"For **{result['metadata']['model']}**, restricted logits have **{ratio:.2f}x "
                f"lower median latency** than same-prompt label generation, with "
                f"**{labels['correct']}/{labels['scorable']}** versus "
                f"**{generation['correct']}/{generation['scorable']}** "
                "correct decisions under the pinned scorer. "
                "Full-option sequence likelihood scores "
                f"**{sequence['correct']}/{sequence['scorable']}** "
                f"at **{sequence['latency_ms_median']:.1f} ms**; independent scoring scores "
                f"**{independent['correct']}/{independent['scorable']}** at "
                f"**{independent['latency_ms_median']:.1f} ms**. "
                "These observed accuracy differences have no significance claim.",
            ]
    lines += [
        "",
        "Accuracy, latency and memory above use canonical presentations only. Order changes "
        "compare the canonical order with one cyclic rotation, matching canonical label IDs. "
        "Distribution accuracy uses the pinned upstream scorer, including its lexicographic "
        "canonical-label tie break; the API selects the first supplied option on a tie. "
        "Order changes compare the API choices. "
        "Invalid outputs count as incorrect; an invalid/valid change also counts as an order "
        "change. Two invalid outputs with no selected choice count as unchanged.",
        "",
        "Brier is the multiclass sum; ECE uses ten equal-width bins and top-label confidence. "
        "Both use valid score distributions only. Generation returns choices, so its calibration "
        "metrics are N/A. Softmax of restricted logits, Yes/No margins or mean sequence "
        "log-likelihoods supplies relative weights, not calibrated correctness probabilities.",
        "",
        "Label generation and restricted logits share the same prompt; JSON uses a separate "
        "canonical-label JSON prompt with full option rubrics. Independent scoring sees one "
        "option rubric at a time. Prompt differences prevent attributing every accuracy "
        "difference solely to inference.\n\n"
        "`sequence_likelihood` teacher-forces every token of each full option, divides the "
        "summed conditional log-probabilities by option token count, and excludes EOS. "
        "It uses a shared full-option prompt and one forward per option. The prompt and "
        "continuation are tokenized separately. This differs from single-token label scoring.",
        "",
        "### Recorded test rigs",
        "",
    ]
    for result in results:
        meta = result["metadata"]
        lines += [f"**{meta['model']}**", "", "```json", json.dumps(meta, indent=2), "```", ""]
    lines += [
        "Timing includes tokenization, inference and processing with backend synchronization; "
        "model loading, queueing and HTTP are excluded. Peak memory measures the Transformers "
        "PyTorch allocator when available, or N/A for llama.cpp; allocator cache and CUDA "
        "context are excluded. "
        "Host-visible resources and GPU driver/clocks are snapshots, not pinned allocations. "
        "Each GPU job runs methods serially with batch size and concurrency one. "
        "Generation uses KV cache; scoring disables it. No runtime quantization, torch.compile, "
        "TensorRT or vLLM is enabled. GGUF weights may use their storage quantization. Different "
        "model jobs may overlap on separate Modal GPU allocations.",
        "",
        "### Task breakdown",
        "",
        "| Model | Method | Tier | Correct | Invalid | Failures |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for method, summary in result["summary"].items():
            for tier, row in summary["per_tier"].items():
                lines.append(
                    f"| {result['metadata']['model']} | `{method}` | {tier} | "
                    f"{row['correct']}/{row['scorable']} | {row['invalid_outputs']} | "
                    f"{row['operational_failures']} |"
                )
    lines += [
        "",
        "### Accuracy by task type",
        "",
        "| Model | Method | Type | Correct | Invalid |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for method, summary in result["summary"].items():
            for kind, row in summary["per_type"].items():
                lines.append(
                    f"| {result['metadata']['model']} | `{method}` | {kind} | "
                    f"{row['correct']}/{row['scorable']} | {row['invalid_outputs']} |"
                )
    lines += [
        "",
        "### Additional diagnostics",
        "",
        "| Model | Method | Returned choice correct | Valid distribution n | "
        "Valid order pairs | Mean order TV | "
        "Ordinal MAE | Noul threshold accuracy | Noul abstentions | Gold probability TV |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        expected = {r["task"]["id"]: str(r["task"]["expected"]) for r in result.get("dataset", [])}
        for method, row in result["summary"].items():
            returned_correct = sum(
                r["choice"] == expected.get(r["id"])
                for r in result["results"]
                if r["method"] == method and r["variant"] == 0
            )
            order = row["order_sensitivity"]
            lines.append(
                f"| {result['metadata']['model']} | `{method}` | "
                f"{returned_correct}/{row['scorable']} | "
                f"{row['calibration_n']} | "
                f"{order['valid_pairs']} | {number(order['probability_tv_mean'], 3)} | "
                f"{number(row['ordinal_ev_mae'], 3)} | "
                f"{number(row['noul_threshold_accuracy'], 3)} | {row['noul_abstentions']} | "
                f"{number(row['gold_distribution_tv_mean'], 3)} |"
            )
    lines += [
        "",
        "Ordinal MAE uses expected value for distribution methods and the selected level "
        "for generation, on valid ordinal predictions. Noul distribution decisions use "
        "P(yes) >= 0.8 for Yes, <= 0.2 for No, otherwise abstain; abstentions count as incorrect. "
        "Generation has no confidence threshold; its missing or invalid Noul decisions "
        "are included in the abstention count. These are supplementary typed diagnostics, "
        "not the official v1.5 composite. Gold probability TV measures fidelity on tasks with "
        "an explicitly supplied reference distribution; see raw summaries for denominators.",
        "",
        "### Selected order changes",
        "",
        "| Model | Method | Task | Canonical choice | Rotated choice |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for method in result["summary"]:
            grouped = {}
            for row in result["results"]:
                if row["method"] == method:
                    grouped.setdefault(row["id"], {})[row["variant"]] = row
            for task_id, variants in grouped.items():
                if 0 in variants and 1 in variants:
                    a, b = variants[0], variants[1]
                    if a["choice"] != b["choice"]:
                        lines.append(
                            f"| {result['metadata']['model']} | `{method}` | {task_id} | "
                            f"{a['choice']} | {b['choice']} |"
                        )
                        break
    return "\n".join(lines) + "\n"


def validate_comparison(results):
    fields = (
        "dataset_sha256",
        "source",
        "source_sha256",
        "repeats",
        "permutations",
        "gpu",
        "dtype",
        "versions",
        "max_input_tokens",
        "warmup_runs_per_method",
        "batch_size",
        "concurrency",
        "generation_limits",
        "temperature",
        "gpu_total_mib",
        "gpu_compute_capability",
        "python",
        "cuda",
        "greedy",
        "scorer",
        "calibration",
        "failure_policy",
    )
    reference = results[0]["metadata"]
    for result in results[1:]:
        for field in fields:
            if result["metadata"].get(field) != reference.get(field):
                raise ValueError(f"incompatible benchmark metadata: {field}")
        if set(result["summary"]) != set(results[0]["summary"]):
            raise ValueError("incompatible benchmark methods")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="docs/jevbench-report.md")
    parser.add_argument("--readme", action="store_true")
    args = parser.parse_args()
    results = [json.loads(Path(path).read_text()) for path in args.results]
    validate_comparison(results)
    report = render(results)
    Path(args.output).write_text(report)
    if args.readme:
        path = Path("README.md")
        text = path.read_text()
        start, end = "<!-- jevbench-results:start -->", "<!-- jevbench-results:end -->"
        # Keep rig details in the full report and the concise results in README.
        report_lines = report.splitlines()
        table_start = next(
            index
            for index, line in enumerate(report_lines)
            if line.startswith("| Model | Method | Scorer correct")
        )
        table_end = table_start
        while table_end < len(report_lines) and report_lines[table_end].startswith("|"):
            table_end += 1
        snippet = "\n".join(report_lines[:table_end]) + "\n\n"
        snippet += (
            "Accuracy and timing use canonical order; order changes compare one cyclic rotation. "
            "Generation and logits share the label prompt; JSON uses a separate prompt. "
            "Brier/ECE use valid distributions; generation has no calibration score. "
            "The upstream scorer resolves ties lexicographically; "
            "the API selects the first option. "
            "See the full report for returned-choice accuracy and methodology.\n\n"
        )
        meta = results[0]["metadata"]
        snippet += (
            f"Test rig: Modal, {meta['gpu']}, {meta['dtype']}, batch/concurrency 1, "
            f"{meta['max_input_tokens']}-token context limit, "
            f"{meta['warmup_runs_per_method']} warmups per method and {meta['repeats']} "
            f"measured attempt(s) per presentation. {meta['unique_tasks']} unique tasks, "
            f"{meta['presentations']} presentations and {meta['requests']} requests per model. "
            "CUDA-synchronized timing excludes loading, HTTP and queueing. "
            + (
                "T4 has no native BF16 Tensor Core support.\n\n"
                if meta["dtype"] == "torch.bfloat16"
                else "The model dtype is explicit in the reproduction command.\n\n"
            )
        )
        snippet += f"Full rig configuration and tier breakdown: [{args.output}]({args.output}).\n"
        snippet += (
            "Raw per-task measurements: "
            + ", ".join(f"[{Path(p).name}]({p})" for p in args.results)
            + ".\n"
        )
        block = start + "\n" + snippet + end
        if start in text:
            before, remainder = text.split(start, 1)
            _, after = remainder.split(end, 1)
            text = before + block + after
        else:
            text += "\n" + block + "\n"
        path.write_text(text)


if __name__ == "__main__":
    main()
