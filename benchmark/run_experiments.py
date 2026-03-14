from __future__ import annotations

"""
Experiment Runner
=================

Runs the VCL memory system and baselines on the MemBayes benchmark.

Systems under test:
    1. VCL (full)           — three-layer architecture, all features
    2. VCL (no coreset)     — without importance-weighted coreset
    3. VCL (no decay)       — without adaptive temporal decay
    4. Naive RAG            — store everything, confidence=1.0, last-write-wins
    5. Sliding Window       — keep last N interactions in context
    6. Decay Only           — temporal decay, no Bayesian evidence tracking

Usage:
    python run_experiments.py
    python run_experiments.py --systems vcl naive
    python run_experiments.py --seed 123 --output-dir my_results
"""

import argparse
import json
import os
import time
import logging
from dataclasses import replace
from pathlib import Path

from benchmark.generator import generate_benchmark, evaluate_responses
from benchmark.baselines import NaiveRAGMemory, SlidingWindowMemory, DecayOnlyMemory
from membayes import VCLMemory, VCLConfig
from membayes.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# System factory
# =============================================================================

def create_systems(llm_client: LLMClient, systems: list[str],
                   window_size: int = 30) -> dict:
    """Create all requested memory systems.

    Returns dict of {name: system}.
    """
    config = VCLConfig()
    available = {}

    if "vcl" in systems:
        available["VCL (full)"] = VCLMemory(
            config=config, use_coreset=True, llm_client=llm_client,
        )

    if "vcl_no_coreset" in systems:
        available["VCL (no coreset)"] = VCLMemory(
            config=config, use_coreset=False, llm_client=llm_client,
        )

    if "vcl_no_decay" in systems:
        no_decay_config = VCLConfig(
            decay_rates={k: 0.0 for k in config.decay_rates},
            soft_decay_threshold=0.0,
            hard_decay_threshold=0.0,
        )
        available["VCL (no decay)"] = VCLMemory(
            config=no_decay_config, use_coreset=True, llm_client=llm_client,
        )

    if "naive" in systems:
        available["Naive RAG"] = NaiveRAGMemory(config, llm_client)

    if "sliding_window" in systems:
        available["Sliding Window"] = SlidingWindowMemory(
            config, llm_client, window_size=window_size,
        )

    if "decay_only" in systems:
        available["Decay Only"] = DecayOnlyMemory(config, llm_client)

    return available


# =============================================================================
# Benchmark runner
# =============================================================================

def run_system(system, benchmark: dict, system_name: str) -> list[dict]:
    """Run a single system on the benchmark. Returns list of test responses."""
    stream = benchmark["stream"]
    total_steps = len(stream)
    responses = []
    n_interactions = 0
    n_tests = 0
    n_forgets = 0

    for i, item in enumerate(stream):
        step = item["step"]
        itype = item["item_type"]

        if itype == "interaction":
            n_interactions += 1
            logger.info("[%s] Step %d/%d — interaction #%d: %s",
                        system_name, i + 1, total_steps, n_interactions,
                        item["content"][:80])
            system.process_interaction(item["content"], step)

        elif itype == "forget":
            n_forgets += 1
            logger.info("[%s] Step %d/%d — FORGET #%d: %s",
                        system_name, i + 1, total_steps, n_forgets,
                        item["content"][:80])
            system.forget(item["content"], step)

        elif itype == "test":
            n_tests += 1
            logger.info("[%s] Step %d/%d — test #%d: %s",
                        system_name, i + 1, total_steps, n_tests,
                        item["content"][:80])
            resp = system.answer_query(item["content"])
            responses.append(resp)
            logger.info("[%s]   → answer=%s, confidence=%.3f",
                        system_name, resp.get("answer", "")[:60],
                        resp.get("confidence", 0.0))

        # Progress log every 50 items
        if (i + 1) % 50 == 0:
            logger.info("[%s] Progress: %d/%d items (interactions=%d, tests=%d, forgets=%d)",
                        system_name, i + 1, total_steps, n_interactions, n_tests, n_forgets)

    logger.info("[%s] Completed: %d interactions, %d tests, %d forgets",
                system_name, n_interactions, n_tests, n_forgets)

    return responses


# =============================================================================
# Visualization
# =============================================================================

def plot_results(all_results: dict, output_dir: str = "results"):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    names = list(all_results.keys())
    evals = all_results
    colors = ["#1565C0", "#1E88E5", "#64B5F6",
              "#FF9800", "#FFC107", "#8BC34A"]

    # ── Plot 1: Overall Accuracy ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    accs = [evals[n]["overall_accuracy"] for n in names]
    bars = ax.bar(range(len(names)), accs, color=colors[:len(names)],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Overall Accuracy by System")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=150)
    plt.close()

    # ── Plot 2: ECE ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    eces = [evals[n]["calibration"]["ECE"] for n in names]
    bars = ax.bar(range(len(names)), eces, color=colors[:len(names)],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_title("Confidence Calibration (ECE)")
    for bar, val in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ece_comparison.png", dpi=150)
    plt.close()

    # ── Plot 3: Per-hypothesis accuracy heatmap ──────────────────────
    hypotheses = sorted(set().union(
        *(evals[n].get("by_hypothesis", {}).keys() for n in names)
    ))
    if hypotheses:
        data = []
        for n in names:
            row = [evals[n].get("by_hypothesis", {}).get(h, {}).get("accuracy", 0)
                   for h in hypotheses]
            data.append(row)
        data_arr = np.array(data)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data_arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(hypotheses)))
        ax.set_xticklabels(hypotheses, fontsize=10)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title("Accuracy by Hypothesis and System")
        for i in range(len(names)):
            for j in range(len(hypotheses)):
                ax.text(j, i, f"{data_arr[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if data_arr[i, j] > 0.5 else "white")
        fig.colorbar(im, ax=ax, label="Accuracy")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hypothesis_heatmap.png", dpi=150)
        plt.close()

    # ── Plot 4: Calibration diagram ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    for i, name in enumerate(names[:4]):
        bins = evals[name]["calibration"]["bins"]
        if bins:
            preds = [b["predicted"] for b in bins]
            actuals = [b["actual"] for b in bins]
            ax.plot(preds, actuals, "o-", color=colors[i], label=name, markersize=8)
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title("Calibration Diagram")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_diagram.png", dpi=150)
    plt.close()

    # ── Plot 5: Bayesian consistency ─────────────────────────────────
    check_keys = ["reinforcement_increases_confidence",
                  "contradiction_decreases_confidence",
                  "delay_decreases_confidence"]
    check_labels = ["Reinforcement\nIncreases Conf.",
                    "Contradiction\nDecreases Conf.",
                    "Delay\nDecreases Conf."]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(check_labels))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = []
        for ck in check_keys:
            v = evals[name]["bayesian_consistency"].get(ck)
            vals.append(1.0 if v else 0.0 if v is not None else 0.5)
        ax.bar(x + i * width, vals, width, label=name, color=colors[i],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x + width * len(names) / 2 - width / 2)
    ax.set_xticklabels(check_labels, fontsize=9)
    ax.set_ylabel("Pass (1.0) / Fail (0.0)")
    ax.set_title("Bayesian Consistency Checks")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bayesian_consistency.png", dpi=150)
    plt.close()

    # ── Plot 6: Per-test-type accuracy ───────────────────────────────
    test_types = sorted(set().union(
        *(evals[n].get("by_test_type", {}).keys() for n in names)
    ))
    if test_types:
        x = np.arange(len(test_types))
        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, name in enumerate(names):
            type_accs = [evals[name]["by_test_type"].get(tt, {}).get("accuracy", 0)
                         for tt in test_types]
            ax.bar(x + i * width, type_accs, width, label=name, color=colors[i],
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x + width * len(names) / 2 - width / 2)
        ax.set_xticklabels([tt.replace("_", "\n") for tt in test_types], fontsize=7)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Test Type")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(0, 1.15)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/per_type_accuracy.png", dpi=150)
        plt.close()

    print(f"  Plots saved to {output_dir}/")


# =============================================================================
# Summary table
# =============================================================================

def print_summary(all_results: dict):
    """Print comparison tables to stdout."""
    names = list(all_results.keys())
    evals = all_results

    print("\n" + "=" * 90)
    print("  EXPERIMENT RESULTS")
    print("=" * 90)

    # Overall
    print(f"\n{'System':<25s} {'Accuracy':>10s} {'ECE':>10s}")
    print("-" * 50)
    for name in names:
        e = evals[name]
        print(f"{name:<25s} {e['overall_accuracy']:>10.3f} "
              f"{e['calibration']['ECE']:>10.4f}")

    # Per hypothesis
    hypotheses = sorted(set().union(
        *(evals[n].get("by_hypothesis", {}).keys() for n in names)
    ))
    if hypotheses:
        print(f"\n{'Hypothesis':<15s}", end="")
        for name in names:
            print(f" {name[:14]:>14s}", end="")
        print()
        print("-" * (15 + 15 * len(names)))
        for h in hypotheses:
            print(f"{h:<15s}", end="")
            for name in names:
                acc = evals[name].get("by_hypothesis", {}).get(h, {}).get("accuracy", 0)
                print(f" {acc:>14.3f}", end="")
            print()

    # Bayesian consistency
    print(f"\n{'Bayesian Check':<40s}", end="")
    for name in names:
        print(f" {name[:14]:>14s}", end="")
    print()
    print("-" * (40 + 15 * len(names)))
    for ck in ["reinforcement_increases_confidence",
               "contradiction_decreases_confidence",
               "delay_decreases_confidence"]:
        label = ck.replace("_", " ").title()
        print(f"{label:<40s}", end="")
        for name in names:
            val = evals[name]["bayesian_consistency"].get(ck)
            s = "PASS" if val else "FAIL" if val is not None else "N/A"
            print(f" {s:>14s}", end="")
        print()


# =============================================================================
# Main
# =============================================================================

SYSTEM_NAME_MAP = {
    "vcl": "VCL (full)",
    "vcl_no_coreset": "VCL (no coreset)",
    "vcl_no_decay": "VCL (no decay)",
    "naive": "Naive RAG",
    "sliding_window": "Sliding Window",
    "decay_only": "Decay Only",
}


def merge_results(output_dir: str):
    """Merge per-system result files into all_results.json, print summary, and plot."""
    all_results = {}
    for sys_key, sys_name in SYSTEM_NAME_MAP.items():
        path = f"{output_dir}/{sys_key}_results.json"
        if os.path.exists(path):
            with open(path) as f:
                all_results[sys_name] = json.load(f)
            logger.info("Loaded %s from %s", sys_name, path)

    if not all_results:
        print("  No result files found to merge.")
        return

    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    plot_results(all_results, output_dir)
    print(f"\n  Merged {len(all_results)} systems into {output_dir}/all_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="MemBayes: Run benchmark experiments")
    parser.add_argument("--systems", nargs="+",
                        default=["vcl", "vcl_no_coreset", "vcl_no_decay",
                                 "naive", "sliding_window", "decay_only"],
                        help="Systems to run")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Sliding window size")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--benchmark-file", type=str,
                        default="results/benchmark.json",
                        help="Path to benchmark JSON (generate with: "
                             "python -m benchmark.generator)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip running systems; merge existing result files "
                             "and generate plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Merge-only mode: combine per-system results and plot
    if args.merge_only:
        print("=" * 60)
        print("  Merging results and generating plots...")
        print("=" * 60)
        merge_results(args.output_dir)
        return

    print("=" * 60)
    print("  MemBayes Benchmark Experiments")
    print("=" * 60)

    # Step 1: Load benchmark (must be pre-generated)
    print("\n[1/3] Loading benchmark...")
    if not os.path.exists(args.benchmark_file):
        print(f"  ERROR: Benchmark file not found: {args.benchmark_file}")
        print(f"  Generate it first with:")
        print(f"    python -m benchmark.generator --output {args.benchmark_file}")
        return

    with open(args.benchmark_file) as f:
        benchmark = json.load(f)
    print(f"  Loaded from {args.benchmark_file}")

    m = benchmark["metadata"]
    print(f"  {m['n_facts']} facts, {m['n_entities']} entities")
    print(f"  {m['total_interactions']} interactions, {m['total_tests']} tests, "
          f"{m['total_forgets']} forgets")

    # Step 2: Initialize LLM client
    print("\n  Initializing LLM client...")
    client = LLMClient()
    print(f"  Chat: {client.model} via DeepInfra")
    print(f"  Embeddings: {client.embedding_model} via OpenAI")

    # Step 3: Run requested systems
    print("\n[2/3] Running systems...")
    systems = create_systems(client, args.systems, args.window_size)
    all_results = {}

    # Extract test items for evaluation
    test_items = [s for s in benchmark["stream"] if s["item_type"] == "test"]

    for name, system in systems.items():
        # Find the short key for this system name
        sys_key = next((k for k, v in SYSTEM_NAME_MAP.items() if v == name), name)

        print(f"  Running: {name}...", end=" ", flush=True)
        t0 = time.time()
        responses = run_system(system, benchmark, name)
        elapsed = time.time() - t0
        evaluation = evaluate_responses(test_items, responses)
        all_results[name] = evaluation
        acc = evaluation["overall_accuracy"]
        ece = evaluation["calibration"]["ECE"]
        print(f"acc={acc:.3f}  ECE={ece:.4f}  ({elapsed:.1f}s)")

        # Save per-system result file (for distributed runs)
        per_system_path = f"{args.output_dir}/{sys_key}_results.json"
        with open(per_system_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        logger.info("Saved %s results to %s", name, per_system_path)

    # Save combined results
    with open(f"{args.output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Step 4: Print and plot
    print("\n[3/3] Generating results...")
    print_summary(all_results)
    plot_results(all_results, args.output_dir)

    # Print LLM usage
    usage = client.get_usage()
    print(f"\n  LLM usage: {usage['total_calls']} chat calls, "
          f"{usage['total_embedding_calls']} embedding calls")

    print("\n" + "=" * 60)
    print(f"  Done! Results saved to {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
