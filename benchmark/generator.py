from __future__ import annotations

"""
MemBayes Benchmark
==================

Generates a synthetic interaction stream + test probes to evaluate whether
a memory system exhibits correct Bayesian properties across all three layers.

Hypotheses tested:
    H1: Retention      — memories persist across intervening interactions
    H2: Reinforcement  — repeated evidence increases confidence
    H3: Contradiction  — conflicting evidence updates value correctly
    H4: Forgetting     — confidence decays over time without reinforcement
    H5: Calibration    — confidence scores predict accuracy (low ECE)
    H6: Coreset        — episodic replay improves long-term retention
    H7: Selective forget — targeted deletion without collateral damage
    H8: Retrieval      — system finds correct memory from paraphrased queries
    H9: Distractors    — similar but distinct facts don't interfere

All facts use synthetic vocabulary to avoid LLM pretraining contamination.
All interactions are plain natural language — no metadata shortcuts.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# =============================================================================
# Synthetic vocabulary (no overlap with real-world knowledge)
# =============================================================================

ENTITIES = [
    "Zara", "Tomas", "Priya", "Ravi", "Hana",
    "Yuki", "Emil", "Sonia", "Okon", "Lian",
]

PLACES = [
    "Karvossa", "Nelmoor", "Plistad", "Zenthari", "Drakmere",
    "Velthun", "Moraqua", "Thindar", "Orsival", "Brimholt",
]

FOODS = [
    "grillberry stew", "thornfig jam", "crescent bread",
    "velvet porridge", "moss dumplings", "sunpetal salad",
    "ashroot soup", "dewmelon tart", "silkbean curry", "frostplum pie",
]

HOBBIES = [
    "sand-singing", "fog-painting", "flame-dancing",
    "stone-whispering", "tide-reading", "moonweaving",
    "dustcarving", "cloudcharming", "rootbending", "echosculpting",
]

COLORS = [
    "cerulean", "indigo", "amber", "vermillion", "chartreuse",
    "obsidian", "coral", "teal", "saffron", "mauve",
]

OCCUPATIONS = [
    "chronosmith", "skyherder", "runekeeper", "wavesmith",
    "glowforge operator", "starcharist", "veilwright",
]

FILLERS = [
    "Can you explain what photosynthesis is?",
    "Tell me a fun fact about octopuses.",
    "What's the difference between a stack and a queue?",
    "How does gravity work on the moon?",
    "Explain the water cycle in simple terms.",
    "What are prime numbers?",
    "How do bridges stay up?",
    "Why is the sky blue?",
    "What's the tallest mountain in the solar system?",
    "How do magnets work?",
    "Tell me about the history of chess.",
    "What causes thunder?",
    "How does a refrigerator work?",
    "What's the Fibonacci sequence?",
    "Explain how rainbows form.",
    "How does a compass work?",
    "What is the speed of sound?",
    "Why do leaves change color in autumn?",
    "How do airplanes fly?",
    "What is the deepest ocean trench?",
]


# =============================================================================
# Natural language templates
# =============================================================================

PRESENT_TEMPLATES = {
    "favorite_food": [
        "{entity} told me they really love {value}.",
        "I found out that {entity}'s favorite dish is {value}.",
        "{entity} mentioned that they enjoy eating {value} more than anything.",
    ],
    "hometown": [
        "I learned that {entity} lives in {value}.",
        "{entity} told me their hometown is {value}.",
        "Apparently {entity} comes from {value}.",
    ],
    "hobby": [
        "{entity} spends their free time {value}.",
        "I heard that {entity}'s main hobby is {value}.",
        "{entity} is really into {value} these days.",
    ],
    "favorite_color": [
        "By the way, {entity} mentioned that their favorite color is {value}.",
        "{entity} said they absolutely love the color {value}.",
        "I learned that {entity}'s preferred color is {value}.",
    ],
    "occupation": [
        "{entity} works as a {value}.",
        "I found out that {entity}'s profession is {value}.",
        "Turns out {entity} is a {value} by trade.",
    ],
}

REINFORCE_TEMPLATES = {
    "favorite_food": [
        "Yeah, {entity} definitely loves {value}, they mentioned it again.",
        "Someone else confirmed that {entity}'s go-to meal is {value}.",
        "I heard from another source that {entity} really does prefer {value}.",
    ],
    "hometown": [
        "Can confirm — {entity} is indeed from {value}.",
        "Multiple people have told me {entity} lives in {value}.",
        "It's well known that {entity}'s home is {value}.",
    ],
    "hobby": [
        "{entity} was seen {value} again the other day.",
        "Others have noticed that {entity} really enjoys {value}.",
        "{entity} keeps talking about how much they love {value}.",
    ],
    "favorite_color": [
        "{entity} was wearing {value} again — clearly their favorite.",
        "Yep, {entity} confirmed once more that {value} is their color.",
        "No doubt about it, {entity}'s top color choice is {value}.",
    ],
    "occupation": [
        "Confirmed — {entity} is still working as a {value}.",
        "I saw {entity} at their {value} workshop today.",
        "Others verified that {entity}'s job is {value}.",
    ],
}

CONTRADICT_TEMPLATES = {
    "favorite_food": [
        "Actually, {entity} has changed their mind. They now prefer {new_value} over {old_value}.",
        "Correction: {entity}'s favorite food is actually {new_value}, not {old_value}.",
        "I was wrong before — {entity} told me their new favorite is {new_value}.",
    ],
    "hometown": [
        "Update: {entity} has moved to {new_value}. They no longer live in {old_value}.",
        "Actually, {entity} relocated to {new_value} recently.",
        "I need to correct myself — {entity} now resides in {new_value}, not {old_value}.",
    ],
    "hobby": [
        "{entity} has given up {old_value} and taken up {new_value} instead.",
        "Correction: {entity} no longer does {old_value}. Their hobby is now {new_value}.",
        "Actually, {entity} switched to {new_value} from {old_value}.",
    ],
    "favorite_color": [
        "{entity} told me they've switched to {new_value} — {old_value} is out.",
        "Actually, {entity}'s favorite color is now {new_value}, not {old_value}.",
        "Correction: {entity} prefers {new_value} over {old_value} now.",
    ],
    "occupation": [
        "{entity} changed careers — they're now a {new_value} instead of a {old_value}.",
        "Actually, {entity} quit being a {old_value} and became a {new_value}.",
        "Update: {entity} is now working as a {new_value}, not a {old_value}.",
    ],
}

QUERY_TEMPLATES = {
    "favorite_food": [
        "What is {entity}'s favorite food?",
        "What does {entity} like to eat the most?",
        "If {entity} could only eat one dish, what would it be?",
        "What food does {entity} prefer?",
    ],
    "hometown": [
        "Where does {entity} live?",
        "What is {entity}'s hometown?",
        "Where is {entity} from?",
        "In which place does {entity} reside?",
    ],
    "hobby": [
        "What is {entity}'s hobby?",
        "What does {entity} do in their free time?",
        "What activity does {entity} enjoy?",
        "How does {entity} spend their leisure time?",
    ],
    "favorite_color": [
        "What is {entity}'s favorite color?",
        "What color does {entity} like the most?",
        "Which color does {entity} prefer?",
        "What's {entity}'s top color choice?",
    ],
    "occupation": [
        "What does {entity} do for a living?",
        "What is {entity}'s profession?",
        "What job does {entity} have?",
        "What is {entity}'s occupation?",
    ],
}

FORGET_TEMPLATES = [
    "Please forget what I told you about {entity}'s {attribute}.",
    "Delete the information about {entity}'s {attribute}.",
    "Remove the memory about {entity}'s {attribute} — that was wrong.",
]

ATTRIBUTE_TO_CATEGORY = {
    "favorite_food": "preference",
    "hometown": "episodic",
    "hobby": "preference",
    "favorite_color": "preference",
    "occupation": "identity",
}

ATTRIBUTE_TO_POOL = {
    "favorite_food": FOODS,
    "hometown": PLACES,
    "hobby": HOBBIES,
    "favorite_color": COLORS,
    "occupation": OCCUPATIONS,
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class BenchmarkFact:
    fact_id: str
    entity: str
    attribute: str
    value: str
    category: str
    reinforcement_count: int = 0
    contradicted_by: Optional[str] = None
    forget_target: bool = False


@dataclass
class StreamItem:
    step: int
    item_type: str          # "interaction", "test", "forget"
    content: str            # text for interaction/forget, question for test
    fact_id: str = ""       # ground truth reference (for evaluation only)
    expected_answer: str = ""
    test_type: str = ""
    hypothesis: str = ""
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Fact pool generator
# =============================================================================

def generate_facts(rng: random.Random) -> list[BenchmarkFact]:
    """Generate 35 synthetic facts across 10 entities."""
    facts = []
    fid = 0

    attributes = ["favorite_food", "hometown", "hobby", "favorite_color"]

    # Give each entity 3 facts, plus 5 extras for occupation/distractors
    entity_order = list(ENTITIES)
    rng.shuffle(entity_order)

    used = set()
    for entity in entity_order:
        rng.shuffle(attributes)
        for attr in attributes[:3]:
            pool = ATTRIBUTE_TO_POOL[attr]
            value = rng.choice([v for v in pool if (entity, attr, v) not in used])
            used.add((entity, attr, value))
            facts.append(BenchmarkFact(
                fact_id=f"F{fid:03d}",
                entity=entity,
                attribute=attr,
                value=value,
                category=ATTRIBUTE_TO_CATEGORY[attr],
            ))
            fid += 1

    # Add 5 occupation facts for variety
    occ_entities = rng.sample(entity_order, 5)
    for entity in occ_entities:
        value = rng.choice(OCCUPATIONS)
        facts.append(BenchmarkFact(
            fact_id=f"F{fid:03d}",
            entity=entity,
            attribute="occupation",
            value=value,
            category="identity",
        ))
        fid += 1

    return facts


# =============================================================================
# Benchmark generator
# =============================================================================

def generate_benchmark(n_reinforce: int = 7, n_contradict: int = 7,
                       n_forget: int = 3, seed: int = 42) -> dict:
    """Generate complete benchmark with natural language stream.

    Returns dict with: metadata, facts, stream
    """
    rng = random.Random(seed)
    facts = generate_facts(rng)
    n_facts = len(facts)

    # Partition facts into roles
    indices = list(range(n_facts))
    rng.shuffle(indices)
    reinforce_set = set(indices[:n_reinforce])
    contradict_set = set(indices[n_reinforce:n_reinforce + n_contradict])
    stable_set = set(indices[n_reinforce + n_contradict:])

    # Assign contradictions
    for idx in contradict_set:
        f = facts[idx]
        pool = ATTRIBUTE_TO_POOL[f.attribute]
        new_val = rng.choice([v for v in pool if v != f.value])
        f.contradicted_by = new_val

    # Assign forget targets from stable set
    forget_indices = rng.sample(list(stable_set), min(n_forget, len(stable_set)))
    for idx in forget_indices:
        facts[idx].forget_target = True

    # Assign reinforcement counts
    for idx in reinforce_set:
        facts[idx].reinforcement_count = 3

    stream: list[StreamItem] = []
    step = 0

    def add_fillers(n_min=1, n_max=3):
        nonlocal step
        for _ in range(rng.randint(n_min, n_max)):
            stream.append(StreamItem(
                step=step, item_type="interaction",
                content=rng.choice(FILLERS),
            ))
            step += 1

    # ── Phase 1: Initial Presentation ────────────────────────────────
    order = list(range(n_facts))
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = PRESENT_TEMPLATES[f.attribute]
        text = rng.choice(templates).format(entity=f.entity, value=f.value)
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=f.fact_id,
            metadata={"phase": "initial", "entity": f.entity, "attribute": f.attribute},
        ))
        step += 1
        add_fillers(1, 2)

    # ── Phase 2: Immediate Recall (H1, H8) ──────────────────────────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        # Use direct query for H1
        question = templates[0].format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=f.value,
            test_type="immediate_recall", hypothesis="H1",
        ))
        step += 1

    # Paraphrased recall probes for H8 (sample 10 facts)
    h8_sample = rng.sample(order, min(10, len(order)))
    for idx in h8_sample:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates[1:]).format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=f.value,
            test_type="paraphrased_recall", hypothesis="H8",
        ))
        step += 1

    # ── Phase 3: Reinforcement (3 rounds) ────────────────────────────
    for rnd in range(3):
        for idx in reinforce_set:
            f = facts[idx]
            templates = REINFORCE_TEMPLATES[f.attribute]
            text = templates[rnd % len(templates)].format(entity=f.entity, value=f.value)
            stream.append(StreamItem(
                step=step, item_type="interaction", content=text,
                fact_id=f.fact_id,
                metadata={"phase": "reinforcement", "round": rnd + 1},
            ))
            step += 1
        add_fillers(2, 4)

    # ── Phase 4: Post-Reinforcement Recall (H2) ─────────────────────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates).format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=f.value,
            test_type="post_reinforcement", hypothesis="H2",
            metadata={"reinforced": idx in reinforce_set},
        ))
        step += 1

    # ── Phase 5: Contradiction ───────────────────────────────────────
    for idx in contradict_set:
        f = facts[idx]
        templates = CONTRADICT_TEMPLATES[f.attribute]
        text = rng.choice(templates).format(
            entity=f.entity, old_value=f.value, new_value=f.contradicted_by,
        )
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=f.fact_id,
            metadata={"phase": "contradiction", "old_value": f.value,
                       "new_value": f.contradicted_by},
        ))
        step += 1
    add_fillers(3, 6)

    # ── Phase 6: Post-Contradiction Recall (H3) ─────────────────────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates).format(entity=f.entity)
        if idx in contradict_set:
            expected = f.contradicted_by
        else:
            expected = f.value
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=expected,
            test_type="post_contradiction", hypothesis="H3",
            metadata={"contradicted": idx in contradict_set,
                       "reinforced": idx in reinforce_set},
        ))
        step += 1

    # ── Phase 7: Long Delay (70 fillers) ─────────────────────────────
    for _ in range(70):
        stream.append(StreamItem(
            step=step, item_type="interaction",
            content=rng.choice(FILLERS),
            metadata={"phase": "long_delay"},
        ))
        step += 1

    # ── Phase 8: Forgetting Curve Recall (H4) ────────────────────────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates).format(entity=f.entity)
        if idx in contradict_set:
            expected = f.contradicted_by
        else:
            expected = f.value
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=expected,
            test_type="forgetting_curve", hypothesis="H4",
            metadata={"reinforced": idx in reinforce_set,
                       "contradicted": idx in contradict_set},
        ))
        step += 1

    # ── Phase 9: Selective Forget (H7) ───────────────────────────────
    forget_neighbors = []
    for idx in forget_indices:
        f = facts[idx]
        text = rng.choice(FORGET_TEMPLATES).format(
            entity=f.entity, attribute=f.attribute.replace("_", " "),
        )
        stream.append(StreamItem(
            step=step, item_type="forget", content=text,
            fact_id=f.fact_id,
            metadata={"phase": "selective_forget"},
        ))
        step += 1

        # Find a neighbor fact (same entity, different attribute)
        for oidx, of in enumerate(facts):
            if oidx != idx and of.entity == f.entity and oidx not in forget_indices:
                forget_neighbors.append(oidx)
                break

    # Test forgotten targets
    for idx in forget_indices:
        f = facts[idx]
        question = QUERY_TEMPLATES[f.attribute][0].format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer="[FORGOTTEN]",
            test_type="selective_forget_target", hypothesis="H7",
        ))
        step += 1

    # Test neighbors (should be intact)
    for idx in forget_neighbors:
        f = facts[idx]
        expected = f.contradicted_by if f.contradicted_by else f.value
        question = QUERY_TEMPLATES[f.attribute][0].format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=expected,
            test_type="selective_forget_neighbor", hypothesis="H7",
        ))
        step += 1

    # ── Phase 10: Distractor Recall (H9) ─────────────────────────────
    # Find fact pairs: same attribute, different entities
    by_attr: dict[str, list[int]] = {}
    for idx, f in enumerate(facts):
        by_attr.setdefault(f.attribute, []).append(idx)

    distractor_pairs = []
    for attr, attr_indices in by_attr.items():
        if len(attr_indices) >= 2:
            pair = rng.sample(attr_indices, 2)
            distractor_pairs.append(pair)
            if len(distractor_pairs) >= 5:
                break

    for idx_a, idx_b in distractor_pairs:
        fa, fb = facts[idx_a], facts[idx_b]
        # Ask about entity A — should NOT return entity B's value
        question = QUERY_TEMPLATES[fa.attribute][0].format(entity=fa.entity)
        expected = fa.contradicted_by if fa.contradicted_by else fa.value
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=fa.fact_id, expected_answer=expected,
            test_type="distractor_recall", hypothesis="H9",
            metadata={"distractor_entity": fb.entity,
                       "distractor_value": fb.contradicted_by or fb.value},
        ))
        step += 1

    # ── Build output ─────────────────────────────────────────────────
    n_interactions = sum(1 for s in stream if s.item_type == "interaction")
    n_tests = sum(1 for s in stream if s.item_type == "test")
    n_forgets = sum(1 for s in stream if s.item_type == "forget")

    return {
        "metadata": {
            "n_entities": len(ENTITIES),
            "n_facts": n_facts,
            "n_reinforced": n_reinforce,
            "n_contradicted": n_contradict,
            "n_forget_targets": n_forget,
            "n_distractors": len(distractor_pairs),
            "total_interactions": n_interactions,
            "total_tests": n_tests,
            "total_forgets": n_forgets,
            "total_steps": step,
            "seed": seed,
            "hypotheses": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"],
        },
        "facts": [asdict(f) for f in facts],
        "stream": [asdict(s) for s in stream],
    }


# =============================================================================
# Evaluation
# =============================================================================

def is_correct(expected: str, actual: str) -> bool:
    """Check if actual answer matches expected answer."""
    if expected == "[FORGOTTEN]":
        low = actual.lower()
        return any(phrase in low for phrase in [
            "don't have", "don't know", "no information",
            "forgotten", "unknown", "not available", "",
        ]) or actual.strip() == ""

    # Exact or substring match
    exp_low = expected.strip().lower()
    act_low = actual.strip().lower()
    return exp_low in act_low or act_low in exp_low


def evaluate_responses(stream: list[dict], responses: list[dict]) -> dict:
    """Evaluate system responses against ground truth.

    Args:
        stream: Test items from the benchmark stream (only type="test" items)
        responses: System responses, one per test item

    Returns:
        Evaluation dict with accuracy, ECE, per-hypothesis, consistency checks
    """
    assert len(stream) == len(responses), (
        f"Got {len(stream)} tests but {len(responses)} responses"
    )

    all_confs, all_correct = [], []
    by_type: dict[str, dict] = {}
    by_hypothesis: dict[str, dict] = {}

    for test, resp in zip(stream, responses):
        ok = is_correct(test["expected_answer"], resp.get("answer", ""))
        conf = resp.get("confidence", 0.0)

        all_confs.append(conf)
        all_correct.append(int(ok))

        # By test type
        tt = test["test_type"]
        if tt not in by_type:
            by_type[tt] = {"correct": 0, "total": 0, "confs": [], "accs": []}
        by_type[tt]["correct"] += int(ok)
        by_type[tt]["total"] += 1
        by_type[tt]["confs"].append(conf)
        by_type[tt]["accs"].append(int(ok))

        # By hypothesis
        h = test["hypothesis"]
        if h not in by_hypothesis:
            by_hypothesis[h] = {"correct": 0, "total": 0, "confs": [], "accs": []}
        by_hypothesis[h]["correct"] += int(ok)
        by_hypothesis[h]["total"] += 1
        by_hypothesis[h]["confs"].append(conf)
        by_hypothesis[h]["accs"].append(int(ok))

    # Overall
    total_correct = sum(d["correct"] for d in by_type.values())
    total_n = sum(d["total"] for d in by_type.values())
    overall_acc = round(total_correct / total_n, 3) if total_n else 0

    # Per-type results
    type_results = {}
    for tt, d in by_type.items():
        type_results[tt] = {
            "accuracy": round(d["correct"] / d["total"], 3) if d["total"] else 0,
            "avg_confidence": round(sum(d["confs"]) / len(d["confs"]), 3) if d["confs"] else 0,
            "n": d["total"],
        }

    # Per-hypothesis results
    hyp_results = {}
    for h, d in by_hypothesis.items():
        hyp_results[h] = {
            "accuracy": round(d["correct"] / d["total"], 3) if d["total"] else 0,
            "avg_confidence": round(sum(d["confs"]) / len(d["confs"]), 3) if d["confs"] else 0,
            "n": d["total"],
        }

    # ECE
    n_bins = 10
    ece = 0.0
    cal_bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bin_items = [(c, a) for c, a in zip(all_confs, all_correct)
                     if lo <= c < hi or (i == n_bins - 1 and c == hi)]
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

    # Bayesian consistency checks
    consistency = {}

    # H2: reinforcement increases confidence
    if "post_reinforcement" in by_type and "immediate_recall" in by_type:
        r_conf = sum(by_type["post_reinforcement"]["confs"]) / len(by_type["post_reinforcement"]["confs"])
        i_conf = sum(by_type["immediate_recall"]["confs"]) / len(by_type["immediate_recall"]["confs"])
        consistency["reinforcement_increases_confidence"] = r_conf > i_conf

    # H3: contradiction decreases average confidence
    if "post_contradiction" in by_type and "post_reinforcement" in by_type:
        c_conf = sum(by_type["post_contradiction"]["confs"]) / len(by_type["post_contradiction"]["confs"])
        r_conf = sum(by_type["post_reinforcement"]["confs"]) / len(by_type["post_reinforcement"]["confs"])
        consistency["contradiction_decreases_confidence"] = c_conf < r_conf

    # H4: delay decreases confidence
    if "forgetting_curve" in by_type and "immediate_recall" in by_type:
        f_conf = sum(by_type["forgetting_curve"]["confs"]) / len(by_type["forgetting_curve"]["confs"])
        i_conf = sum(by_type["immediate_recall"]["confs"]) / len(by_type["immediate_recall"]["confs"])
        consistency["delay_decreases_confidence"] = f_conf < i_conf

    return {
        "overall_accuracy": overall_acc,
        "by_test_type": type_results,
        "by_hypothesis": hyp_results,
        "calibration": {"ECE": round(ece, 4), "bins": cal_bins},
        "bayesian_consistency": consistency,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MemBayes benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/benchmark.json",
                        help="Output file path")
    args = parser.parse_args()

    print("Generating MemBayes benchmark...")
    bm = generate_benchmark(seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(bm, f, indent=2)

    m = bm["metadata"]
    print(f"  {m['n_facts']} facts across {m['n_entities']} entities")
    print(f"  {m['total_interactions']} interactions, {m['total_tests']} tests, "
          f"{m['total_forgets']} forgets")
    print(f"  {m['total_steps']} total steps")
    print(f"  Hypotheses: {', '.join(m['hypotheses'])}")
    print(f"  Saved to {out}")
