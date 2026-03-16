from __future__ import annotations

"""
Evaluation Plots (Expanded)
============================

Generates comparison visualizations across memory systems.
Includes per-hypothesis heatmap, per-category, evidence strength,
confidence calibration, and Bayesian consistency checks.
"""

import os


def plot_results(all_results: dict, output_dir: str = "results"):
    """Generate comparison plots for all evaluated systems."""
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

    # Color scheme: blues for VCL variants, warm for baselines
    colors = ["#1565C0", "#1E88E5", "#64B5F6",
              "#FF9800", "#FFC107", "#8BC34A"]

    # ── Plot 1: Overall Accuracy ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    accs = [all_results[n]["overall_accuracy"] for n in names]
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
    eces = [all_results[n]["calibration"]["ECE"] for n in names]
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
        *(all_results[n].get("by_hypothesis", {}).keys() for n in names)
    ))
    if hypotheses:
        data = np.zeros((len(hypotheses), len(names)))
        for j, name in enumerate(names):
            for i, h in enumerate(hypotheses):
                data[i, j] = (all_results[name]
                              .get("by_hypothesis", {})
                              .get(h, {})
                              .get("accuracy", 0))

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(len(hypotheses)))

        hyp_labels = {
            "H1": "H1: Retention", "H2": "H2: Reinforcement",
            "H3": "H3: Contradiction", "H4": "H4: Forgetting",
            "H5": "H5: Calibration", "H6": "H6: Coreset",
            "H7": "H7: Selective Forget", "H8": "H8: Retrieval",
            "H9": "H9: Distractors", "H10": "H10: Belief Revision",
            "H11": "H11: Rapid Conflict", "H12": "H12: Category Decay",
            "H13": "H13: Dependency Prop.", "H14": "H14: Evidence Strength",
        }
        labels = [hyp_labels.get(h, h) for h in hypotheses]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Accuracy by Hypothesis and System")

        for i in range(len(hypotheses)):
            for j in range(len(names)):
                ax.text(j, i, f"{data[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] < 0.5 else "black")

        fig.colorbar(im, ax=ax, label="Accuracy")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hypothesis_heatmap.png", dpi=150)
        plt.close()

    # ── Plot 4: Per test-type accuracy grouped bars ──────────────────
    test_types = sorted(set().union(
        *(all_results[n].get("by_test_type", {}).keys() for n in names)
    ))
    if test_types:
        fig, ax = plt.subplots(figsize=(16, 7))
        x = np.arange(len(test_types))
        width = 0.8 / len(names)

        for j, name in enumerate(names):
            vals = [
                all_results[name]
                .get("by_test_type", {})
                .get(tt, {})
                .get("accuracy", 0)
                for tt in test_types
            ]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width,
                   label=name, color=colors[j % len(colors)],
                   edgecolor="black", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([tt.replace("_", "\n") for tt in test_types],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Test Type")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test_type_comparison.png", dpi=150)
        plt.close()

    # ── Plot 5: Per-category accuracy ────────────────────────────────
    all_categories = sorted(set().union(
        *(all_results[n].get("by_category", {}).keys() for n in names)
    ))
    if all_categories:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_categories))
        width = 0.8 / len(names)

        for j, name in enumerate(names):
            vals = [
                all_results[name]
                .get("by_category", {})
                .get(cat, {})
                .get("accuracy", 0)
                for cat in all_categories
            ]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width,
                   label=name, color=colors[j % len(colors)],
                   edgecolor="black", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Semantic Category")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_accuracy.png", dpi=150)
        plt.close()

    # ── Plot 6: Per-category confidence ──────────────────────────────
    if all_categories:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_categories))
        width = 0.8 / len(names)

        for j, name in enumerate(names):
            vals = [
                all_results[name]
                .get("by_category", {})
                .get(cat, {})
                .get("avg_confidence", 0)
                for cat in all_categories
            ]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width,
                   label=name, color=colors[j % len(colors)],
                   edgecolor="black", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, fontsize=9)
        ax.set_ylabel("Average Confidence")
        ax.set_title("Confidence by Semantic Category")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_confidence.png", dpi=150)
        plt.close()

    # ── Plot 7: Evidence Strength Analysis ───────────────────────────
    all_strengths = sorted(set().union(
        *(all_results[n].get("by_strength", {}).keys() for n in names)
    ))
    if all_strengths:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy by strength
        ax = axes[0]
        x = np.arange(len(all_strengths))
        width = 0.8 / len(names)
        for j, name in enumerate(names):
            vals = [
                all_results[name]
                .get("by_strength", {})
                .get(s, {})
                .get("accuracy", 0)
                for s in all_strengths
            ]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width,
                   label=name, color=colors[j % len(colors)],
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(all_strengths, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Evidence Strength")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)

        # Confidence by strength
        ax = axes[1]
        for j, name in enumerate(names):
            vals = [
                all_results[name]
                .get("by_strength", {})
                .get(s, {})
                .get("avg_confidence", 0)
                for s in all_strengths
            ]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width,
                   label=name, color=colors[j % len(colors)],
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(all_strengths, fontsize=9)
        ax.set_ylabel("Average Confidence")
        ax.set_title("Confidence by Evidence Strength")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/evidence_strength.png", dpi=150)
        plt.close()

    # ── Plot 8: Bayesian Consistency Checks ──────────────────────────
    all_checks = sorted(set().union(
        *(all_results[n].get("bayesian_consistency", {}).keys() for n in names)
    ))
    if all_checks:
        fig, ax = plt.subplots(figsize=(14, 6))
        data = np.zeros((len(all_checks), len(names)))
        for j, name in enumerate(names):
            for i, ck in enumerate(all_checks):
                val = all_results[name]["bayesian_consistency"].get(ck)
                data[i, j] = 1.0 if val else 0.0 if val is not None else 0.5

        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(len(all_checks)))
        ax.set_yticklabels([ck.replace("_", " ").title() for ck in all_checks],
                           fontsize=7)
        ax.set_title("Bayesian Consistency Checks")

        for i in range(len(all_checks)):
            for j in range(len(names)):
                val = all_results[names[j]]["bayesian_consistency"].get(all_checks[i])
                label = "PASS" if val else "FAIL" if val is not None else "N/A"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        fontweight="bold",
                        color="white" if data[i, j] < 0.5 else "black")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/bayesian_consistency.png", dpi=150)
        plt.close()

    # ── Plot 9: Confidence Calibration Diagram ───────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for j, name in enumerate(names[:6]):
        ax = axes[j]
        bins = all_results[name].get("calibration", {}).get("bins", [])
        if not bins:
            ax.set_title(f"{name}\n(no data)")
            continue

        predicted = [b["predicted"] for b in bins]
        actual = [b["actual"] for b in bins]
        counts = [b["n"] for b in bins]

        ax.bar(predicted, actual, width=0.08, alpha=0.6,
               color=colors[j % len(colors)], edgecolor="black", linewidth=0.3)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Predicted Confidence", fontsize=8)
        ax.set_ylabel("Actual Accuracy", fontsize=8)
        ax.set_title(f"{name}\nECE={all_results[name]['calibration']['ECE']:.4f}",
                     fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_diagram.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {output_dir}/")
