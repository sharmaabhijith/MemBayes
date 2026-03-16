from __future__ import annotations

"""
Response Evaluator (Expanded)
==============================

Scores system responses against ground truth, computes ECE,
and checks Bayesian consistency properties across all 14 hypotheses.
"""


def is_correct(expected: str, actual: str) -> bool:
    """Check if actual answer matches expected answer."""
    if expected == "[FORGOTTEN]":
        low = actual.lower()
        return any(phrase in low for phrase in [
            "don't have", "don't know", "no information",
            "forgotten", "unknown", "not available",
        ]) or actual.strip() == ""

    exp_low = expected.strip().lower()
    act_low = actual.strip().lower()
    return exp_low in act_low or act_low in exp_low


def evaluate_responses(test_items: list[dict],
                       responses: list[dict]) -> dict:
    """Evaluate system responses against ground truth.

    Args:
        test_items: Test items from the benchmark stream (only type="test")
        responses: System responses, one per test item

    Returns:
        Evaluation dict with accuracy, ECE, per-hypothesis, consistency,
        per-category breakdown, and strength analysis
    """
    assert len(test_items) == len(responses), (
        f"Got {len(test_items)} tests but {len(responses)} responses"
    )

    all_confs: list[float] = []
    all_correct: list[int] = []
    by_type: dict[str, dict] = {}
    by_hypothesis: dict[str, dict] = {}
    by_category: dict[str, dict] = {}
    by_strength: dict[str, dict] = {}

    for test, resp in zip(test_items, responses):
        ok = is_correct(test["expected_answer"], resp.get("answer", ""))
        conf = resp.get("confidence", 0.0)

        all_confs.append(conf)
        all_correct.append(int(ok))

        # By test type
        tt = test["test_type"]
        if tt not in by_type:
            by_type[tt] = {"correct": 0, "total": 0,
                           "confs": [], "accs": []}
        by_type[tt]["correct"] += int(ok)
        by_type[tt]["total"] += 1
        by_type[tt]["confs"].append(conf)
        by_type[tt]["accs"].append(int(ok))

        # By hypothesis
        h = test["hypothesis"]
        if h not in by_hypothesis:
            by_hypothesis[h] = {"correct": 0, "total": 0,
                                "confs": [], "accs": []}
        by_hypothesis[h]["correct"] += int(ok)
        by_hypothesis[h]["total"] += 1
        by_hypothesis[h]["confs"].append(conf)
        by_hypothesis[h]["accs"].append(int(ok))

        # By category (from metadata)
        cat = test.get("metadata", {}).get("category", "")
        if cat:
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0,
                                    "confs": [], "accs": []}
            by_category[cat]["correct"] += int(ok)
            by_category[cat]["total"] += 1
            by_category[cat]["confs"].append(conf)
            by_category[cat]["accs"].append(int(ok))

        # By evidence strength (from metadata, for reinforcement tests)
        strength = test.get("metadata", {}).get("strength", "")
        if strength and strength != "none":
            if strength not in by_strength:
                by_strength[strength] = {"correct": 0, "total": 0,
                                         "confs": [], "accs": []}
            by_strength[strength]["correct"] += int(ok)
            by_strength[strength]["total"] += 1
            by_strength[strength]["confs"].append(conf)
            by_strength[strength]["accs"].append(int(ok))

    # Overall accuracy
    total_correct = sum(d["correct"] for d in by_type.values())
    total_n = sum(d["total"] for d in by_type.values())
    overall_acc = round(total_correct / total_n, 3) if total_n else 0

    def _summarize(d: dict) -> dict:
        """Build summary dict for a grouping."""
        results = {}
        for key, data in d.items():
            results[key] = {
                "accuracy": round(data["correct"] / data["total"], 3)
                            if data["total"] else 0,
                "avg_confidence": round(sum(data["confs"]) / len(data["confs"]), 3)
                                  if data["confs"] else 0,
                "n": data["total"],
            }
        return results

    type_results = _summarize(by_type)
    hyp_results = _summarize(by_hypothesis)
    cat_results = _summarize(by_category)
    strength_results = _summarize(by_strength)

    # ECE (10-bin Expected Calibration Error)
    n_bins = 10
    ece = 0.0
    cal_bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bin_items = [
            (c, a) for c, a in zip(all_confs, all_correct)
            if lo <= c < hi or (i == n_bins - 1 and c == hi)
        ]
        if bin_items:
            pred = sum(c for c, _ in bin_items) / len(bin_items)
            actual = sum(a for _, a in bin_items) / len(bin_items)
            cal_bins.append({
                "range": f"[{lo:.1f}, {hi:.1f})",
                "predicted": round(pred, 3),
                "actual": round(actual, 3),
                "n": len(bin_items),
                "error": round(abs(pred - actual), 3),
            })
            ece += (len(bin_items) / len(all_confs)) * abs(pred - actual)

    # ── Bayesian consistency checks ──────────────────────────────────
    consistency = {}

    def _avg_conf(tt_name):
        data = by_type.get(tt_name, {})
        confs = data.get("confs", [])
        return sum(confs) / len(confs) if confs else None

    # Check 1: Reinforcement increases confidence (H2)
    r_conf = _avg_conf("post_reinforcement")
    i_conf = _avg_conf("immediate_recall")
    if r_conf is not None and i_conf is not None:
        consistency["reinforcement_increases_confidence"] = r_conf > i_conf

    # Check 2: Contradiction decreases confidence (H3)
    c_conf = _avg_conf("post_contradiction")
    if c_conf is not None and r_conf is not None:
        consistency["contradiction_decreases_confidence"] = c_conf < r_conf

    # Check 3: Delay decreases confidence (H4)
    f_conf = _avg_conf("forgetting_curve")
    if f_conf is not None and i_conf is not None:
        consistency["delay_decreases_confidence"] = f_conf < i_conf

    # Check 4: Strong evidence > weak evidence confidence (H14)
    if "high" in by_strength and "low" in by_strength:
        high_conf = sum(by_strength["high"]["confs"]) / len(by_strength["high"]["confs"])
        low_conf = sum(by_strength["low"]["confs"]) / len(by_strength["low"]["confs"])
        consistency["strong_gt_weak_confidence"] = high_conf > low_conf

    # Check 5: Short delay < long delay confidence drop (H12)
    sd_conf = _avg_conf("short_delay_recall")
    fc_conf = _avg_conf("forgetting_curve")
    if sd_conf is not None and fc_conf is not None:
        consistency["longer_delay_more_decay"] = fc_conf < sd_conf

    # Check 6: Category-specific decay ordering (H12)
    # transient should decay more than identity
    if by_category:
        cat_avg_confs = {}
        for cat, data in by_category.items():
            confs = data.get("confs", [])
            if confs:
                cat_avg_confs[cat] = sum(confs) / len(confs)
        if "transient" in cat_avg_confs and "identity" in cat_avg_confs:
            consistency["transient_decays_faster_than_identity"] = (
                cat_avg_confs["transient"] < cat_avg_confs["identity"]
            )

    # Check 7: Dependency child confidence drops after parent contradiction (H13)
    dep_child_data = by_type.get("dependency_child", {})
    if dep_child_data and dep_child_data.get("confs"):
        dep_child_conf = sum(dep_child_data["confs"]) / len(dep_child_data["confs"])
        consistency["dependency_propagation_lowers_child"] = dep_child_conf < 0.7

    return {
        "overall_accuracy": overall_acc,
        "by_test_type": type_results,
        "by_hypothesis": hyp_results,
        "by_category": cat_results,
        "by_strength": strength_results,
        "calibration": {"ECE": round(ece, 4), "bins": cal_bins},
        "bayesian_consistency": consistency,
    }
