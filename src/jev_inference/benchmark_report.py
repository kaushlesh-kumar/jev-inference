"""Render benchmark evidence without discarding invalid or incorrect outputs."""

import argparse
import json
import shlex
from pathlib import Path

from .benchmark import METHODS

START = "<!-- benchmark-results:start -->"
END = "<!-- benchmark-results:end -->"


def number(value, digits=1):
    return "N/A" if value is None else f"{value:.{digits}f}"


def summary_table(result: dict) -> str:
    lines = [
        "| Method | Correct | Invalid | Median ms | p95 ms | Peak VRAM MiB | Extra MiB |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method in METHODS:
        item = result["summary"][method]
        lines.append(
            f"| `{method}` | {item['correct']}/{item['total']} "
            f"({item['accuracy']:.1%}) | {item['invalid_outputs']} | "
            f"{item['latency_ms_median']:.1f} | {item['latency_ms_p95']:.1f} | "
            f"{number(item['max_peak_allocated_mib'])} | {number(item['max_extra_peak_mib'])} |"
        )
    return "\n".join(lines)


def grouped_results(result: dict) -> dict:
    groups = {}
    for row in result["results"]:
        groups.setdefault(row["id"], {})[row["method"]] = row
    return groups


def interesting_cases(result: dict) -> list[str]:
    groups = grouped_results(result)
    selected = []

    def pick(candidates, key):
        if remaining := [item for item in candidates if item not in selected]:
            selected.append(max(remaining, key=lambda item: key(groups[item])))

    pick(
        [
            key
            for key, group in groups.items()
            if group["generate_json"]["correct"] and group["labels"]["correct"]
        ],
        lambda group: (
            group["generate_json"]["latency_ms_median"] / group["labels"]["latency_ms_median"]
        ),
    )
    pick(
        [
            key
            for key, group in groups.items()
            if group["generate_json"]["correct"] and not group["labels"]["correct"]
        ],
        lambda group: group["generate_json"]["latency_ms_median"],
    )
    pick(
        [
            key
            for key, group in groups.items()
            if group["independent"]["correct"] and not group["labels"]["correct"]
        ],
        lambda group: group["labels"]["latency_ms_median"],
    )
    pick(list(groups), lambda group: group["labels"]["extra_peak_mib"])
    return selected


def snippet(
    result: dict,
    results_name: str = "benchmark-results.json",
    report_name: str = "benchmark-report.md",
) -> str:
    meta = result["metadata"]
    summary = result["summary"]
    json_ratio = (
        summary["generate_json"]["latency_ms_median"] / summary["labels"]["latency_ms_median"]
    )
    label_ratio = (
        summary["generate_label"]["latency_ms_median"] / summary["labels"]["latency_ms_median"]
    )
    label_memory = summary["labels"]["max_peak_allocated_mib"]
    generation_memory = summary["generate_label"]["max_peak_allocated_mib"]
    memory_direction = (
        "unknown"
        if label_memory is None or generation_memory is None
        else "more"
        if label_memory > generation_memory
        else "no more"
    )
    p95_direction = (
        "higher"
        if summary["labels"]["latency_ms_p95"] > summary["generate_label"]["latency_ms_p95"]
        else "no higher"
    )
    positions = [0, 0, 0]
    for case in result["dataset"]:
        index = case["choices"].index(case["answer"])
        if index < len(positions):
            positions[index] += 1
    lines = [
        f"## Benchmark: {meta['cases']} classification and decision tasks",
        "",
        f"Measured on **{meta['model']}**, **{meta['gpu']}**, `{meta['dtype']}`, "
        f"{meta['timestamp_utc'][:10]}. {meta['cases']} hand-labeled English cases; "
        f"{meta['repeats']} repetitions per case/method after warmup.",
        "",
        summary_table(result),
        "",
        f"On this run, `labels` was **{json_ratio:.2f}× faster than JSON generation** "
        f"and **{label_ratio:.2f}× faster than label generation** by median latency. "
        f"It used {memory_direction} maximum allocated tensor memory, and its p95 latency "
        f"was {p95_direction} than label generation. Faster inference does not establish "
        "decision quality.",
        "",
        "`generate_label` generates a label using the **same prompt** as `labels`; "
        "`generate_json` generates an object containing the choice text using a separate "
        "JSON prompt. Both baselines use greedy decoding until EOS, capped at 16/64 new "
        "tokens. `labels` reads A/B/C logits in one forward; `independent` uses three "
        "separate yes/no forwards on this three-option dataset.",
        "",
        "Selected examples (choice and median latency):",
        "",
        "| Case | Expected | Generate JSON | Generate label | Logits | Independent |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    groups = grouped_results(result)
    for case_id in interesting_cases(result):
        group = groups[case_id]
        cells = []
        for method in ("generate_json", "generate_label", "labels", "independent"):
            row = group[method]
            choice = row["choice"] if row["choice"] is not None else "INVALID"
            marker = "✓" if row["correct"] else "✗"
            cells.append(f"{choice} {marker} · {row['latency_ms_median']:.1f} ms")
        lines.append(f"| `{case_id}` | {group['labels']['expected']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Timing includes tokenization, inference, and result processing with CUDA "
            "synchronization. Network, queueing, model loading, and cold starts are excluded. "
            f"Median/p95 are computed over {meta['cases']} per-case latency medians. Peak VRAM "
            "is the maximum `max_memory_allocated` across all measurements of a method, "
            "including weights; Extra is the largest increase over pre-request allocated "
            "memory. These measure PyTorch tensors, excluding CUDA context and allocator "
            "cache, rather than total GPU usage.",
            "",
            "Accuracy includes output-format compliance: unparseable generated answers count "
            "as incorrect; invalid outputs are also reported separately. "
            "This is a small diagnostic dataset, not an external quality benchmark. "
            f"Correct positions: {positions[0]} A / {positions[1]} B / {positions[2]} C. "
            "The JSON comparison mixes inference and prompt-format effects; use "
            "`generate_label` versus `labels` for the same-prompt comparison. Avoiding "
            "generation does not guarantee lower VRAM: the current logits engine computes "
            "vocabulary logits for the entire input sequence.",
            "",
            f"[All measurements and raw outputs](docs/{results_name}) · "
            f"[Full table](docs/{report_name}) · "
            "[Dataset](benchmarks/classification30.json)",
            "",
            "Reproduce from the repository root after installing `.[modal]` "
            "and running `modal setup`:",
            "",
            "```bash",
            "python run_benchmark.py --modal "
            f"--model {shlex.quote(meta['model'])} --gpu T4 --repeats {meta['repeats']} "
            f"--output docs/{shlex.quote(results_name)}",
            f"python -m jev_inference.benchmark_report docs/{shlex.quote(results_name)} "
            "--update-readme",
            "```",
            "",
            "For a local CUDA GPU, install `.[inference]` and omit `--modal`. The benchmark "
            "runs as a separate ephemeral Modal job and does not update the deployed API. "
            f"The JSON retains all {meta['cases'] * meta['repeats'] * len(METHODS)} measurements, "
            "choices, raw outputs, token counts, dataset SHA256, model revision, and library "
            "versions. Dependency versions and model revision may change between runs; "
            "compare metadata before comparing results.",
        ]
    )
    return "\n".join(lines)


def full_report(result: dict, results_name: str = "benchmark-results.json") -> str:
    repeats = result["metadata"]["repeats"]
    lines = [
        "# Full classification benchmark",
        "",
        f"Model: **{result['metadata']['model']}**; GPU: **{result['metadata']['gpu']}**.",
        "",
        summary_table(result),
        "",
        f"Latency is the median of {repeats} repetitions; memory is the maximum. "
        "Choice and correctness use the first repetition; Stable indicates whether "
        "the choice stayed the same across repetitions. Accuracy requires a parsed allowed "
        "choice; malformed generated outputs count as incorrect. Label parsing accepts "
        "a single uppercase letter with optional trailing dot or parenthesis; JSON "
        "parsing requires an object whose choice field matches an allowed option. "
        "Cached allocator memory (peak_reserved_mib in JSON) can carry over between "
        "methods and is not used for memory comparisons. "
        f"Raw outputs and repetitions: [JSON]({results_name}).",
        "",
        "| Case | Method | Expected | Choice | Correct | ms | Peak MiB | Extra MiB | Stable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["results"]:
        lines.append(
            f"| {row['id']} | {row['method']} | {row['expected']} | "
            f"{row['choice'] or 'INVALID'} | {'✓' if row['correct'] else '✗'} | "
            f"{row['latency_ms_median']:.1f} | {number(row['peak_allocated_mib'])} | "
            f"{number(row['extra_peak_mib'])} | {row['stable_choice']} |"
        )
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(result["metadata"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def update_readme(text: str, content: str) -> str:
    block = f"{START}\n{content}\n{END}"
    if START in text and END in text:
        start = text.index(START)
        end = text.index(END, start) + len(END)
        return text[:start] + block + text[end:]
    if START in text or END in text:
        raise ValueError("README has incomplete benchmark markers")
    return text.rstrip() + "\n\n" + block + "\n"


def report_path(results_path: Path) -> Path:
    stem = results_path.stem
    name = (
        stem.replace("benchmark-results", "benchmark-report", 1)
        if stem.startswith("benchmark-results")
        else stem + "-report"
    )
    return results_path.with_name(name + ".md")


def validate_comparison(results: list[dict]):
    reference = results[0]
    models = [result["metadata"]["model"] for result in results]
    if len(set(models)) != len(models):
        raise ValueError("comparison requires different model IDs")
    keys = (
        "dataset_sha256",
        "python",
        "cases",
        "repeats",
        "gpu",
        "gpu_total_mib",
        "warmup_runs_per_method",
        "dtype",
        "versions",
        "cuda",
        "max_input_tokens",
        "generation_limits",
        "timing",
        "method_order",
        "greedy",
    )
    for result in results[1:]:
        if result["dataset"] != reference["dataset"]:
            raise ValueError("comparison requires identical datasets")
        for key in keys:
            if result["metadata"][key] != reference["metadata"][key]:
                raise ValueError(f"comparison requires matching {key}")


def test_rig(result: dict) -> list[str]:
    meta = result["metadata"]
    input_lengths = [
        run["input_tokens"]
        for row in result["results"]
        for run in row["runs"]
        if "input_tokens" in run
    ]
    versions = meta["versions"]
    rows = [
        ("Platform", "Modal, separate ephemeral GPU jobs; Debian slim Linux container"),
        ("GPU", f"1 × {meta['gpu']}; {meta['gpu_total_mib']:.1f} MiB CUDA-visible VRAM"),
        ("Precision", f"{meta['dtype']}; no quantization"),
        (
            "Software",
            f"Python {meta['python']}; PyTorch {versions['torch']}; "
            f"Transformers {versions['transformers']}; Accelerate {versions['accelerate']}",
        ),
        ("CUDA", f"{meta['cuda']} (PyTorch build version, not the NVIDIA driver version)"),
        ("Workload", "Batch size 1; one request at a time; no batching or concurrency"),
        (
            "Context",
            f"{meta['max_input_tokens']} token limit; "
            f"{min(input_lengths)}–{max(input_lengths)} tokens in generation prompts",
        ),
        (
            "Generation",
            f"Greedy, 1 beam, stop at EOS; label cap "
            f"{meta['generation_limits']['generate_label']} tokens; JSON cap "
            f"{meta['generation_limits']['generate_json']} tokens",
        ),
        ("KV cache", "Enabled for generate(); disabled for direct logits forwards"),
        (
            "Warmup / repeats",
            f"{meta['warmup_runs_per_method']} warmups per method on the first "
            f"case; {meta['repeats']} measured repeats per case/method",
        ),
        (
            "Execution",
            "Methods rotated by case/repeat; model runs sequential; "
            "no explicit torch.compile, TensorRT, or vLLM",
        ),
        (
            "Host / driver",
            "CPU model, host RAM, NVIDIA driver, GPU clocks and power limits "
            "were not captured or pinned",
        ),
        ("Dataset SHA256", f"`{meta['dataset_sha256']}`"),
    ]
    return [
        "",
        "### Test rig and protocol",
        "",
        "| Setting | Configuration |",
        "| --- | --- |",
        *[f"| {key} | {value} |" for key, value in rows],
        "",
    ]


def comparison_snippet(results: list[dict], paths: list[Path]) -> str:
    validate_comparison(results)
    meta = results[0]["metadata"]
    lines = [
        f"## Benchmark: {meta['cases']} classification tasks across model sizes",
        "",
        f"Same {meta['cases']} hand-labeled English tasks, **{meta['gpu']}**, "
        f"`{meta['dtype']}`, {meta['repeats']} repetitions per task/method, "
        "two warmup runs per method. Both model runs use the same dataset SHA256, "
        "library versions, prompt templates, and measurement protocol.",
        "",
        "| Model | Method | Correct | Invalid | Median ms | p95 ms | Peak VRAM MiB |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        model = result["metadata"]["model"].split("/")[-1]
        for method in METHODS:
            item = result["summary"][method]
            lines.append(
                f"| {model} | `{method}` | {item['correct']}/{item['total']} "
                f"({item['accuracy']:.1%}) | {item['invalid_outputs']} | "
                f"{item['latency_ms_median']:.1f} | {item['latency_ms_p95']:.1f} | "
                f"{number(item['max_peak_allocated_mib'])} |"
            )
    lines.extend(test_rig(results[0]))
    if meta["gpu"] == "Tesla T4" and meta["dtype"] == "torch.bfloat16":
        lines.extend(
            [
                "BF16 is the actual auto-selected model dtype in these runs. T4 lacks native "
                "BF16 Tensor Core support; these are not optimized FP16-throughput results. "
                "See the [NVIDIA precision support matrix]"
                "(https://docs.nvidia.com/deeplearning/tensorrt/pdf/TensorRT-Support-Matrix-Guide.pdf).",
                "",
            ]
        )
    for result in results:
        meta_model = result["metadata"]
        lines.append(
            f"- **{meta_model['model']}**: revision "
            f"`{meta_model['model_revision']}`; measured "
            f"`{meta_model['timestamp_utc']}`."
        )
    lines.extend(["", "Median latency ratios (baseline median divided by logits median):", ""])
    for result in results:
        summary = result["summary"]
        logits_ms = summary["labels"]["latency_ms_median"]
        json_ratio = summary["generate_json"]["latency_ms_median"] / logits_ms
        label_ratio = summary["generate_label"]["latency_ms_median"] / logits_ms
        lines.append(
            f"- **{result['metadata']['model'].split('/')[-1]}**: "
            f"**{json_ratio:.2f}×** versus JSON generation; "
            f"**{label_ratio:.2f}×** versus label generation."
        )
    latest = results[-1]
    latest_summary = latest["summary"]
    latest_model = latest["metadata"]["model"].split("/")[-1]
    lines.extend(
        [
            "",
            f"For **{latest_model}**, the identical-prompt comparison scored "
            f"**{latest_summary['generate_label']['correct']}/{meta['cases']}** for label "
            f"generation and **{latest_summary['labels']['correct']}/{meta['cases']}** for logits. "
            f"The separate JSON prompt scored **{latest_summary['generate_json']['correct']}"
            f"/{meta['cases']}**, and independent scoring scored "
            f"**{latest_summary['independent']['correct']}/{meta['cases']}**. "
            "A speed improvement from changing inference does not imply that a label prompt "
            "matches the quality of a JSON prompt.",
        ]
    )
    lines.extend(
        [
            "",
            "`generate_label` uses the **same prompt** as `labels`, with greedy decoding "
            "until EOS (16-token cap). `generate_json` uses a separate JSON prompt and a "
            "64-token cap. `labels` reads A/B/C logits in one forward; `independent` uses "
            "one yes/no forward per option. No training or prompt tuning between model runs.",
            "",
            "Selected examples (choice and median latency):",
            "",
            "| Case | Model | Expected | Generate JSON | Generate label | Logits | Independent |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    groups = [grouped_results(result) for result in results]
    # Include improved/degraded logits choices, then representative speed/memory cases.
    selected = []
    for target in (True, False):
        for case in results[0]["dataset"]:
            case_id = case["id"]
            if (
                groups[0][case_id]["labels"]["correct"] != target
                and groups[-1][case_id]["labels"]["correct"] == target
            ):
                selected.append(case_id)
                break
    for case_id in interesting_cases(results[-1]):
        if case_id not in selected:
            selected.append(case_id)
        if len(selected) >= 4:
            break
    for case_id in selected:
        for result, group in zip(results, groups):
            cells = []
            for method in ("generate_json", "generate_label", "labels", "independent"):
                row = group[case_id][method]
                marker = "✓" if row["correct"] else "✗"
                cells.append(
                    f"{row['choice'] or 'INVALID'} {marker} · {row['latency_ms_median']:.1f} ms"
                )
            model = result["metadata"]["model"].split("/")[-1]
            expected = group[case_id]["labels"]["expected"]
            lines.append(f"| `{case_id}` | {model} | {expected} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Timing includes tokenization, inference, and result processing with CUDA "
            "synchronization; network, queueing, model loading, and cold starts are excluded. "
            "Median/p95 are computed over per-task latency medians. Peak VRAM is the maximum "
            "allocated PyTorch tensor memory, including weights, excluding CUDA context and "
            "allocator cache. Per-request extra memory is included in the full reports.",
            "",
            "Accuracy includes output-format compliance; unparseable generated answers count "
            "as incorrect and are listed separately as Invalid. Correct answer positions are "
            "balanced (10 A / 10 B / 10 C). This small diagnostic set does not establish "
            "general decision quality. JSON comparisons mix inference and prompt-format "
            "effects; use the label baseline for identical prompts. The current logits "
            "engine computes vocabulary logits for the entire input, so it can use more "
            "VRAM and be slower on long inputs than label generation.",
            "",
            "Full evidence (all choices, raw outputs, timings, memory, and model revisions):",
            "",
        ]
    )
    for result, path in zip(results, paths):
        model = result["metadata"]["model"].split("/")[-1]
        lines.append(
            f"- **{model}**: [raw JSON]({path.as_posix()}), "
            f"[full table]({report_path(path).as_posix()})."
        )
    total = sum(len(result["results"]) * result["metadata"]["repeats"] for result in results)
    lines.extend(
        [
            "",
            f"[Dataset](benchmarks/classification30.json). {total} measured requests in total. "
            "Runs are sequential and not simultaneous. Reproduce after installing `.[modal]` "
            "and running `modal setup`:",
            "",
            "```bash",
        ]
    )
    for result, path in zip(results, paths):
        lines.append(
            "python run_benchmark.py --modal "
            f"--model {shlex.quote(result['metadata']['model'])} --gpu T4 "
            f"--repeats {meta['repeats']} --output {shlex.quote(path.as_posix())}"
        )
    other_paths = " ".join(shlex.quote(path.as_posix()) for path in paths[1:])
    lines.extend(
        [
            "python -m jev_inference.benchmark_report "
            + shlex.quote(paths[0].as_posix())
            + f" --compare {other_paths} --update-readme",
            "```",
            "",
            "For a local CUDA GPU, install `.[inference]` and omit `--modal`. These ephemeral "
            "benchmark jobs do not update the deployed API. Dependency versions and model "
            "revisions can change between runs; compare metadata before comparing results.",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Render benchmark reports and optional model comparison in README"
    )
    parser.add_argument("results")
    parser.add_argument("--compare", nargs="+", default=[])
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    paths = [Path(path) for path in [args.results, *args.compare]]
    results = [json.loads(path.read_text()) for path in paths]
    if args.compare:
        validate_comparison(results)
    for result, path in zip(results, paths):
        report = report_path(path)
        report.write_text(full_report(result, path.name))
        print(f"Saved {report}")
    if args.update_readme:
        content = (
            comparison_snippet(results, paths)
            if args.compare
            else snippet(results[0], paths[0].name, report_path(paths[0]).name)
        )
        readme = Path("README.md")
        readme.write_text(update_readme(readme.read_text(), content))


if __name__ == "__main__":
    main()
