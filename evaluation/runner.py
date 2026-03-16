from __future__ import annotations

"""
Experiment Runner
=================

Runs the VCL memory system and baselines on the MemBayes benchmark.

Results are saved into timestamped directories under results/:
    results/run_2026-03-15_14-30-00/
        vcl_results.json
        naive_results.json
        ...
        all_results.json
        *.png

Systems under test:
    1. VCL (full)           — three-layer architecture, all features
    2. VCL (no coreset)     — without importance-weighted coreset
    3. VCL (no decay)       — without adaptive temporal decay
    4. Naive RAG            — store everything, confidence=1.0, last-write-wins
    5. Sliding Window       — keep last N interactions in context
    6. Decay Only           — temporal decay, no Bayesian evidence tracking

Usage:
    python -m evaluation.runner
    python -m evaluation.runner --systems vcl naive
    python -m evaluation.runner --merge-only --run-dir results/run_2026-03-15_14-30-00
"""

import argparse
import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path

from evaluation.evaluator import evaluate_responses
from evaluation.baselines import NaiveRAGMemory, SlidingWindowMemory, DecayOnlyMemory
from evaluation.plots import plot_results
from membayes import VCLMemory, VCLConfig
from membayes.llm_client import LLMClient

logger = logging.getLogger(__name__)


def setup_logging(run_dir: str | None = None):
    """Configure logging to console and optionally to a file in run_dir/logs/."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler (always)
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(console)

    # File handler (when we have a run directory)
    if run_dir:
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"runner_{stamp}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)
        logger.info("Logging to %s", log_file)


SYSTEM_NAME_MAP = {
    "vcl": "VCL (full)",
    "vcl_no_coreset": "VCL (no coreset)",
    "vcl_no_decay": "VCL (no decay)",
    "naive": "Naive RAG",
    "sliding_window": "Sliding Window",
    "decay_only": "Decay Only",
}


def make_run_dir(base: str = "results") -> str:
    """Create a timestamped run directory like results/run_2026-03-15_14-30-00."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def find_latest_run_dir(base: str = "results") -> str | None:
    """Find the most recent run_* directory under base."""
    if not os.path.isdir(base):
        return None
    dirs = sorted(
        [d for d in os.listdir(base)
         if d.startswith("run_") and os.path.isdir(os.path.join(base, d))],
        reverse=True,
    )
    return os.path.join(base, dirs[0]) if dirs else None


# =============================================================================
# System factory
# =============================================================================

def create_systems(llm_client: LLMClient, systems: list[str],
                   window_size: int = 30) -> dict:
    """Create all requested memory systems."""
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
                        system_name, (resp.get("answer") or "")[:60],
                        resp.get("confidence", 0.0))

        if (i + 1) % 50 == 0:
            logger.info("[%s] Progress: %d/%d items "
                        "(interactions=%d, tests=%d, forgets=%d)",
                        system_name, i + 1, total_steps,
                        n_interactions, n_tests, n_forgets)

    logger.info("[%s] Completed: %d interactions, %d tests, %d forgets",
                system_name, n_interactions, n_tests, n_forgets)

    return responses


# =============================================================================
# Summary table
# =============================================================================

def print_summary(all_results: dict):
    """Print comparison tables to stdout."""
    names = list(all_results.keys())

    print("\n" + "=" * 90)
    print("  EXPERIMENT RESULTS")
    print("=" * 90)

    # Overall
    print(f"\n{'System':<25s} {'Accuracy':>10s} {'ECE':>10s}")
    print("-" * 50)
    for name in names:
        e = all_results[name]
        print(f"{name:<25s} {e['overall_accuracy']:>10.3f} "
              f"{e['calibration']['ECE']:>10.4f}")

    # Per hypothesis
    hypotheses = sorted(set().union(
        *(all_results[n].get("by_hypothesis", {}).keys() for n in names)
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
                acc = (all_results[name]
                       .get("by_hypothesis", {})
                       .get(h, {})
                       .get("accuracy", 0))
                print(f" {acc:>14.3f}", end="")
            print()

    # Per category
    categories = sorted(set().union(
        *(all_results[n].get("by_category", {}).keys() for n in names)
    ))
    if categories:
        print(f"\n{'Category':<15s}", end="")
        for name in names:
            print(f" {name[:14]:>14s}", end="")
        print()
        print("-" * (15 + 15 * len(names)))
        for cat in categories:
            print(f"{cat:<15s}", end="")
            for name in names:
                acc = (all_results[name]
                       .get("by_category", {})
                       .get(cat, {})
                       .get("accuracy", 0))
                print(f" {acc:>14.3f}", end="")
            print()

    # Evidence strength
    strengths = sorted(set().union(
        *(all_results[n].get("by_strength", {}).keys() for n in names)
    ))
    if strengths:
        print(f"\n{'Strength':<15s}", end="")
        for name in names:
            print(f" {name[:14]:>14s}", end="")
        print()
        print("-" * (15 + 15 * len(names)))
        for s in strengths:
            print(f"{s:<15s}", end="")
            for name in names:
                acc = (all_results[name]
                       .get("by_strength", {})
                       .get(s, {})
                       .get("accuracy", 0))
                print(f" {acc:>14.3f}", end="")
            print()

    # Bayesian consistency
    all_checks = sorted(set().union(
        *(all_results[n].get("bayesian_consistency", {}).keys() for n in names)
    ))
    if all_checks:
        print(f"\n{'Bayesian Check':<45s}", end="")
        for name in names:
            print(f" {name[:14]:>14s}", end="")
        print()
        print("-" * (45 + 15 * len(names)))
        for ck in all_checks:
            label = ck.replace("_", " ").title()
            print(f"{label:<45s}", end="")
            for name in names:
                val = all_results[name]["bayesian_consistency"].get(ck)
                s = "PASS" if val else "FAIL" if val is not None else "N/A"
                print(f" {s:>14s}", end="")
            print()


# =============================================================================
# Merge results
# =============================================================================

def merge_results(run_dir: str):
    """Merge per-system result files in run_dir into all_results.json."""
    all_results = {}
    for sys_key, sys_name in SYSTEM_NAME_MAP.items():
        path = os.path.join(run_dir, f"{sys_key}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results[sys_name] = json.load(f)
            logger.info("Loaded %s from %s", sys_name, path)

    if not all_results:
        print(f"  No result files found in {run_dir}/")
        return

    out_path = os.path.join(run_dir, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    plot_results(all_results, run_dir)
    print(f"\n  Merged {len(all_results)} systems into {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MemBayes: Run evaluation experiments")
    parser.add_argument(
        "--systems", nargs="+",
        default=["vcl", "vcl_no_coreset", "vcl_no_decay",
                 "naive", "sliding_window", "decay_only"],
        help="Systems to run")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Sliding window size")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Base output directory")
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Specific run directory (for --merge-only or to resume into)")
    parser.add_argument(
        "--benchmark-file", type=str,
        default="results/benchmark.json",
        help="Path to benchmark JSON "
             "(generate with: python -m evaluation.generator)")
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip running systems; merge existing result files and plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging()  # console only until we have a run directory

    # ── Merge-only mode ──────────────────────────────────────────────
    if args.merge_only:
        run_dir = args.run_dir or find_latest_run_dir(args.output_dir)
        if not run_dir:
            print("  No run directories found. Nothing to merge.")
            return
        print("=" * 60)
        print(f"  Merging results from: {run_dir}")
        print("=" * 60)
        merge_results(run_dir)
        return

    # ── Normal run ───────────────────────────────────────────────────
    # Create or reuse run directory
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = make_run_dir(args.output_dir)

    setup_logging(run_dir)  # add file handler now that we have a run directory

    print("=" * 60)
    print("  MemBayes Evaluation Experiments")
    print(f"  Run directory: {run_dir}")
    print("=" * 60)

    # Load benchmark
    print("\n[1/3] Loading benchmark...")
    if not os.path.exists(args.benchmark_file):
        print(f"  ERROR: Benchmark file not found: {args.benchmark_file}")
        print(f"  Generate it first with:")
        print(f"    python -m evaluation.generator "
              f"--output {args.benchmark_file}")
        return

    with open(args.benchmark_file) as f:
        benchmark = json.load(f)
    print(f"  Loaded from {args.benchmark_file}")

    m = benchmark["metadata"]
    print(f"  {m['n_facts']} facts, {m['n_entities']} entities")
    print(f"  {m['total_interactions']} interactions, "
          f"{m['total_tests']} tests, {m['total_forgets']} forgets")

    # Initialize LLM client
    print("\n  Initializing LLM client...")
    client = LLMClient()
    print(f"  Chat: {client.model} via DeepInfra")
    print(f"  Embeddings: {client.embedding_model} via OpenAI")

    # Run systems
    print("\n[2/3] Running systems...")
    systems = create_systems(client, args.systems, args.window_size)
    all_results = {}

    test_items = [s for s in benchmark["stream"]
                  if s["item_type"] == "test"]

    for name, system in systems.items():
        sys_key = next(
            (k for k, v in SYSTEM_NAME_MAP.items() if v == name), name)

        print(f"  Running: {name}...", end=" ", flush=True)
        t0 = time.time()
        responses = run_system(system, benchmark, name)
        elapsed = time.time() - t0
        evaluation = evaluate_responses(test_items, responses)
        all_results[name] = evaluation
        acc = evaluation["overall_accuracy"]
        ece = evaluation["calibration"]["ECE"]
        print(f"acc={acc:.3f}  ECE={ece:.4f}  ({elapsed:.1f}s)")

        per_system_path = os.path.join(run_dir, f"{sys_key}_results.json")
        with open(per_system_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        logger.info("Saved %s results to %s", name, per_system_path)

    # Print summary
    print("\n[3/3] Results...")
    print_summary(all_results)

    # Save combined results and plots
    out_path = os.path.join(run_dir, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    plot_results(all_results, run_dir)

    usage = client.get_usage()
    print(f"\n  LLM usage: {usage['total_calls']} chat calls, "
          f"{usage['total_embedding_calls']} embedding calls")

    print("\n" + "=" * 60)
    print(f"  Done! Results saved to {run_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
