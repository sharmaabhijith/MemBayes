from __future__ import annotations

"""
Benchmark Generator (Expanded)
==============================

Generates a synthetic interaction stream + test probes to evaluate whether
a memory system exhibits correct Bayesian properties across all three layers.

Hypotheses tested:
    H1:  Retention         — memories persist across intervening interactions
    H2:  Reinforcement     — repeated evidence increases confidence
    H3:  Contradiction     — conflicting evidence updates value correctly
    H4:  Forgetting        — confidence decays over time without reinforcement
    H5:  Calibration       — confidence scores predict accuracy (low ECE)
    H6:  Coreset           — episodic replay improves long-term retention
    H7:  Selective forget  — targeted deletion without collateral damage
    H8:  Retrieval         — correct memory found from paraphrased queries
    H9:  Distractors       — similar but distinct facts don't interfere
    H10: Belief revision   — multi-step contradiction chains (A→B→C)
    H11: Rapid conflict    — conflicting evidence within short window
    H12: Category decay    — decay rates differ by semantic category
    H13: Dependency prop.  — contradicting a parent weakens dependents
    H14: Evidence strength — high/medium/low evidence have different effects

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
    "Freya", "Kael", "Miren", "Deshi", "Noor",
    "Savik", "Belen", "Jonah", "Thera", "Caspian",
]

PLACES = [
    "Karvossa", "Nelmoor", "Plistad", "Zenthari", "Drakmere",
    "Velthun", "Moraqua", "Thindar", "Orsival", "Brimholt",
    "Quarista", "Fenhollow", "Duskara", "Thornwall", "Crestmere",
    "Rivenshade", "Galspire", "Oakholm", "Stillmount", "Ashfern",
]

FOODS = [
    "grillberry stew", "thornfig jam", "crescent bread",
    "velvet porridge", "moss dumplings", "sunpetal salad",
    "ashroot soup", "dewmelon tart", "silkbean curry", "frostplum pie",
    "pepperbloom risotto", "shellmoss broth", "glimmerbean paste",
    "crisp dawnroot", "amber nectar pudding", "stormwheat rolls",
    "moonberry compote", "glazed thornfruit", "ironleaf stir-fry",
    "cloudpuff pastry",
]

HOBBIES = [
    "sand-singing", "fog-painting", "flame-dancing",
    "stone-whispering", "tide-reading", "moonweaving",
    "dustcarving", "cloudcharming", "rootbending", "echosculpting",
    "crystaltuning", "stormwatching", "shadowbraiding", "wavesinging",
    "starcharting", "frostscribing", "windknitting", "glowstitching",
    "dewcatching", "threadspinning",
]

COLORS = [
    "cerulean", "indigo", "amber", "vermillion", "chartreuse",
    "obsidian", "coral", "teal", "saffron", "mauve",
    "heliotrope", "umber", "verdigris", "cinnabar", "aureolin",
    "smaragdine", "glaucous", "nacarat", "feldgrau", "sinopia",
]

OCCUPATIONS = [
    "chronosmith", "skyherder", "runekeeper", "wavesmith",
    "glowforge operator", "starcharist", "veilwright",
    "dreamwelder", "ashwarden", "tidecaller", "dawnkeeper",
    "rootmender", "mistwalker", "flaretender", "galewright",
    "shellcrafter", "duskscribe", "voidtapper", "embersinger",
    "crystalshaper",
]

PETS = [
    "a shimmercat", "a dusthound", "a thornwing", "a mossferret",
    "a glowfox", "a stoneowl", "an embervole", "a tideskipper",
    "a cloudmoth", "a rootsnake", "a frostbeetle", "a wavesparrow",
    "a sandlurker", "a dewspider", "a flickerfinch", "a nightcrawler",
    "an ironhare", "a shellturtle", "a blazeant", "a driftworm",
]

TRANSPORT = [
    "a skyraft", "a rootcart", "a duskcycle", "a tideboard",
    "a glowsled", "a sandbarge", "a stormskiff", "a crystalcar",
    "a windwagon", "a flamechariot", "a mossglider", "a voidhopper",
    "a wavepod", "a thornrider", "a cloudcoach", "an emberbike",
    "a shellscooter", "a driftboat", "a starship", "a dustrunner",
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
    "What are black holes made of?",
    "How do vaccines work?",
    "What is the difference between weather and climate?",
    "How do submarines navigate underwater?",
    "What causes earthquakes?",
    "How do solar panels generate electricity?",
    "What is the Doppler effect?",
    "How do birds know where to migrate?",
    "What is a supernova?",
    "How does 3D printing work?",
    "What causes the tides?",
    "How does the internet work?",
    "What is the speed of light?",
    "How do bees make honey?",
    "What is dark matter?",
    "How do electric cars work?",
    "What causes the northern lights?",
    "How do plants communicate?",
    "What is quantum entanglement?",
    "How do geysers form?",
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
    "pet": [
        "{entity} has {value} as a pet.",
        "I heard that {entity} keeps {value}.",
        "{entity} told me they own {value}.",
    ],
    "transport": [
        "{entity} gets around using {value}.",
        "I found out that {entity}'s main mode of transport is {value}.",
        "{entity} mentioned they travel by {value}.",
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
    "pet": [
        "Saw {entity} with {value} again — definitely their pet.",
        "Others confirmed that {entity} does indeed own {value}.",
        "{entity} was talking about {value} again, clearly their pet.",
    ],
    "transport": [
        "Spotted {entity} on {value} again today.",
        "Can confirm {entity} still uses {value} to get around.",
        "Others have seen {entity} traveling by {value}.",
    ],
}

# Strength-specific reinforcement templates
REINFORCE_STRONG_TEMPLATES = {
    "favorite_food": [
        "{entity} emphatically declared that {value} is their absolute favorite food.",
        "There is no doubt whatsoever — {entity}'s favorite food is definitely {value}.",
    ],
    "hometown": [
        "{entity} has lived in {value} their entire life and will never leave.",
        "Everyone knows for certain that {entity}'s home is {value}.",
    ],
    "hobby": [
        "{entity} is completely obsessed with {value} and does it every single day.",
        "There is absolutely no question — {value} is {entity}'s passion.",
    ],
    "favorite_color": [
        "{entity} is absolutely obsessed with {value} — everything they own is that color.",
        "Without any doubt, {value} is {entity}'s all-time favorite color.",
    ],
    "occupation": [
        "{entity} has been a {value} for decades and is deeply passionate about it.",
        "There is no question — {entity} is definitely a {value}, everyone knows.",
    ],
    "pet": [
        "{entity} adores {value} more than anything in the world.",
        "There's zero doubt that {entity}'s beloved pet is {value}.",
    ],
    "transport": [
        "{entity} swears by {value} and would never use anything else.",
        "Without question, {entity}'s preferred transport is {value}.",
    ],
}

REINFORCE_WEAK_TEMPLATES = {
    "favorite_food": [
        "I think {entity} might enjoy {value}, but I'm not entirely sure.",
        "Someone vaguely mentioned that {entity} perhaps likes {value}.",
    ],
    "hometown": [
        "I believe {entity} might be from {value}, though I could be wrong.",
        "Someone suggested that {entity} may live in {value}.",
    ],
    "hobby": [
        "I think {entity} might occasionally do some {value}, but it's unclear.",
        "Maybe {entity} has tried {value} once or twice.",
    ],
    "favorite_color": [
        "I think {entity} might like {value}, but they weren't very sure.",
        "Perhaps {entity} somewhat prefers {value}, though it's hard to tell.",
    ],
    "occupation": [
        "I think {entity} might work as a {value}, but I'm not certain.",
        "Someone vaguely suggested {entity} could be a {value}.",
    ],
    "pet": [
        "I think {entity} might have {value}, but I'm not entirely sure.",
        "Someone mentioned {entity} perhaps owns {value}.",
    ],
    "transport": [
        "I think {entity} might sometimes use {value}, but it's unclear.",
        "Maybe {entity} occasionally travels by {value}.",
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
    "pet": [
        "{entity} gave away their {old_value} and now has {new_value}.",
        "Actually, {entity}'s pet is {new_value} now, not {old_value}.",
        "Correction: {entity} switched from {old_value} to {new_value}.",
    ],
    "transport": [
        "{entity} ditched {old_value} and now uses {new_value}.",
        "Actually, {entity} now gets around with {new_value}, not {old_value}.",
        "Update: {entity} switched from {old_value} to {new_value}.",
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
    "pet": [
        "What kind of pet does {entity} have?",
        "What is {entity}'s pet?",
        "Does {entity} own any animals?",
        "What pet does {entity} keep?",
    ],
    "transport": [
        "How does {entity} get around?",
        "What is {entity}'s mode of transport?",
        "How does {entity} travel?",
        "What does {entity} use for transportation?",
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
    "pet": "relational",
    "transport": "transient",
}

ATTRIBUTE_TO_POOL = {
    "favorite_food": FOODS,
    "hometown": PLACES,
    "hobby": HOBBIES,
    "favorite_color": COLORS,
    "occupation": OCCUPATIONS,
    "pet": PETS,
    "transport": TRANSPORT,
}

# Dependency templates: parent attribute -> (child attribute, child template)
DEPENDENCY_DEFINITIONS = {
    "hometown": [
        ("transport", "Since {entity} lives in {parent_value}, they get around using {value}."),
        ("hobby", "Because {entity} is in {parent_value}, they spend time {value}."),
    ],
    "occupation": [
        ("transport", "As a {parent_value}, {entity} travels by {value} for work."),
    ],
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
    # Extended fields for new scenarios
    chain_values: list[str] = field(default_factory=list)  # for multi-step contradiction
    depends_on: Optional[str] = None  # parent fact_id for dependency tests
    evidence_strength: str = "medium"  # for strength tests


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
    """Generate ~80 synthetic facts across 20 entities.

    Each entity gets 4 facts from different attribute types, ensuring
    coverage across all 5 categories (identity, preference, episodic,
    relational, transient).
    """
    facts = []
    fid = 0

    # Core attributes — each entity gets these
    core_attributes = ["favorite_food", "hometown", "hobby", "favorite_color"]
    # Extra attributes — assigned to subsets
    extra_attributes = ["occupation", "pet", "transport"]

    entity_order = list(ENTITIES)
    rng.shuffle(entity_order)

    used = set()

    # Each entity gets 4 core attributes
    for entity in entity_order:
        rng.shuffle(core_attributes)
        for attr in core_attributes:
            pool = ATTRIBUTE_TO_POOL[attr]
            available = [v for v in pool if (entity, attr, v) not in used]
            value = rng.choice(available)
            used.add((entity, attr, value))
            facts.append(BenchmarkFact(
                fact_id=f"F{fid:03d}",
                entity=entity,
                attribute=attr,
                value=value,
                category=ATTRIBUTE_TO_CATEGORY[attr],
            ))
            fid += 1

    # 10 entities get occupation (identity category)
    occ_entities = rng.sample(entity_order, 10)
    for entity in occ_entities:
        pool = ATTRIBUTE_TO_POOL["occupation"]
        available = [v for v in pool if (entity, "occupation", v) not in used]
        value = rng.choice(available)
        used.add((entity, "occupation", value))
        facts.append(BenchmarkFact(
            fact_id=f"F{fid:03d}",
            entity=entity,
            attribute="occupation",
            value=value,
            category="identity",
        ))
        fid += 1

    # 10 entities get pet (relational category)
    pet_entities = rng.sample(entity_order, 10)
    for entity in pet_entities:
        pool = ATTRIBUTE_TO_POOL["pet"]
        available = [v for v in pool if (entity, "pet", v) not in used]
        value = rng.choice(available)
        used.add((entity, "pet", value))
        facts.append(BenchmarkFact(
            fact_id=f"F{fid:03d}",
            entity=entity,
            attribute="pet",
            value=value,
            category="relational",
        ))
        fid += 1

    # 10 entities get transport (transient category)
    trans_entities = rng.sample(entity_order, 10)
    for entity in trans_entities:
        pool = ATTRIBUTE_TO_POOL["transport"]
        available = [v for v in pool if (entity, "transport", v) not in used]
        value = rng.choice(available)
        used.add((entity, "transport", value))
        facts.append(BenchmarkFact(
            fact_id=f"F{fid:03d}",
            entity=entity,
            attribute="transport",
            value=value,
            category="transient",
        ))
        fid += 1

    return facts


# =============================================================================
# Benchmark generator
# =============================================================================

def generate_benchmark(
    n_reinforce: int = 15,
    n_reinforce_strong: int = 5,
    n_reinforce_weak: int = 5,
    n_contradict: int = 15,
    n_chain_contradict: int = 5,
    n_rapid_conflict: int = 4,
    n_forget: int = 6,
    n_dependency: int = 4,
    seed: int = 42,
) -> dict:
    """Generate expanded benchmark with natural language stream.

    Returns dict with: metadata, facts, stream
    """
    rng = random.Random(seed)
    facts = generate_facts(rng)
    n_facts = len(facts)

    # Build index for quick lookup
    facts_by_entity: dict[str, list[int]] = {}
    facts_by_attr: dict[str, list[int]] = {}
    facts_by_category: dict[str, list[int]] = {}
    for idx, f in enumerate(facts):
        facts_by_entity.setdefault(f.entity, []).append(idx)
        facts_by_attr.setdefault(f.attribute, []).append(idx)
        facts_by_category.setdefault(f.category, []).append(idx)

    # ── Partition facts into roles ──────────────────────────────────
    indices = list(range(n_facts))
    rng.shuffle(indices)

    pos = 0
    reinforce_set = set(indices[pos:pos + n_reinforce])
    pos += n_reinforce

    contradict_set = set(indices[pos:pos + n_contradict])
    pos += n_contradict

    # Chain contradictions: subset with A→B→C updates
    chain_set = set()
    chain_candidates = [i for i in indices[pos:] if i not in reinforce_set and i not in contradict_set]
    rng.shuffle(chain_candidates)
    chain_set = set(chain_candidates[:n_chain_contradict])
    pos_chain = n_chain_contradict

    # Rapid conflict: facts that get confirm then immediate contradict
    rapid_candidates = [i for i in indices if i not in reinforce_set
                        and i not in contradict_set and i not in chain_set]
    rng.shuffle(rapid_candidates)
    rapid_set = set(rapid_candidates[:n_rapid_conflict])

    stable_set = set(indices) - reinforce_set - contradict_set - chain_set - rapid_set

    # Split reinforcement into strong/medium/weak subsets
    reinforce_list = list(reinforce_set)
    rng.shuffle(reinforce_list)
    reinforce_strong = set(reinforce_list[:n_reinforce_strong])
    reinforce_weak = set(reinforce_list[n_reinforce_strong:n_reinforce_strong + n_reinforce_weak])
    reinforce_medium = reinforce_set - reinforce_strong - reinforce_weak

    # Mark evidence strength
    for idx in reinforce_strong:
        facts[idx].evidence_strength = "high"
    for idx in reinforce_medium:
        facts[idx].evidence_strength = "medium"
    for idx in reinforce_weak:
        facts[idx].evidence_strength = "low"

    # Assign single contradictions
    for idx in contradict_set:
        f = facts[idx]
        pool = ATTRIBUTE_TO_POOL[f.attribute]
        new_val = rng.choice([v for v in pool if v != f.value])
        f.contradicted_by = new_val

    # Assign chain contradictions (A→B→C)
    for idx in chain_set:
        f = facts[idx]
        pool = ATTRIBUTE_TO_POOL[f.attribute]
        others = [v for v in pool if v != f.value]
        rng.shuffle(others)
        val_b = others[0]
        val_c = others[1] if len(others) > 1 else others[0]
        f.chain_values = [val_b, val_c]
        f.contradicted_by = val_c  # final expected value

    # Assign rapid conflict contradictions
    for idx in rapid_set:
        f = facts[idx]
        pool = ATTRIBUTE_TO_POOL[f.attribute]
        new_val = rng.choice([v for v in pool if v != f.value])
        f.contradicted_by = new_val

    # Assign forget targets from stable set
    stable_list = list(stable_set)
    rng.shuffle(stable_list)
    forget_indices = stable_list[:min(n_forget, len(stable_list))]
    for idx in forget_indices:
        facts[idx].forget_target = True

    # Assign reinforcement counts
    for idx in reinforce_set:
        facts[idx].reinforcement_count = 3

    # ── Set up dependency pairs ─────────────────────────────────────
    # Find facts with hometown or occupation that we can build dependencies from
    # Exclude facts already assigned to other roles to ensure clean H13 testing
    dependency_pairs = []  # (parent_idx, child_idx, child_template)
    dep_parent_candidates = [
        idx for idx in range(n_facts)
        if facts[idx].attribute in DEPENDENCY_DEFINITIONS
        and idx not in forget_indices
        and idx not in contradict_set
        and idx not in chain_set
        and idx not in rapid_set
    ]
    rng.shuffle(dep_parent_candidates)

    for pidx in dep_parent_candidates[:n_dependency]:
        parent = facts[pidx]
        dep_defs = DEPENDENCY_DEFINITIONS[parent.attribute]
        child_attr, child_template = rng.choice(dep_defs)
        # Find or create a child fact for same entity
        child_idx = None
        for cidx in facts_by_entity.get(parent.entity, []):
            if facts[cidx].attribute == child_attr and cidx != pidx:
                child_idx = cidx
                break
        if child_idx is not None:
            facts[child_idx].depends_on = parent.fact_id
            dependency_pairs.append((pidx, child_idx, child_template))

    # ── Build stream ────────────────────────────────────────────────
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
            metadata={"phase": "initial", "entity": f.entity,
                       "attribute": f.attribute, "category": f.category},
        ))
        step += 1
        add_fillers(1, 2)

    # Present dependency relationships
    for pidx, cidx, child_template in dependency_pairs:
        parent = facts[pidx]
        child = facts[cidx]
        text = child_template.format(
            entity=child.entity, parent_value=parent.value, value=child.value)
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=child.fact_id,
            metadata={"phase": "dependency_setup", "parent_fact_id": parent.fact_id,
                       "entity": child.entity},
        ))
        step += 1
        add_fillers(1, 2)

    # ── Phase 2: Immediate Recall (H1, H8) ──────────────────────────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = templates[0].format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=f.value,
            test_type="immediate_recall", hypothesis="H1",
            metadata={"category": f.category},
        ))
        step += 1

    # Paraphrased recall probes for H8 (sample 20 facts)
    h8_sample = rng.sample(order, min(20, len(order)))
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

    # ── Phase 3: Reinforcement (3 rounds, with varied strength) ─────
    for rnd in range(3):
        for idx in reinforce_set:
            f = facts[idx]
            if idx in reinforce_strong:
                templates = REINFORCE_STRONG_TEMPLATES[f.attribute]
            elif idx in reinforce_weak:
                templates = REINFORCE_WEAK_TEMPLATES[f.attribute]
            else:
                templates = REINFORCE_TEMPLATES[f.attribute]
            text = templates[rnd % len(templates)].format(
                entity=f.entity, value=f.value)
            stream.append(StreamItem(
                step=step, item_type="interaction", content=text,
                fact_id=f.fact_id,
                metadata={"phase": "reinforcement", "round": rnd + 1,
                           "strength": facts[idx].evidence_strength},
            ))
            step += 1
        add_fillers(2, 4)

    # ── Phase 4: Post-Reinforcement Recall (H2 + H14) ────────────────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates).format(entity=f.entity)
        is_reinforced = idx in reinforce_set
        strength = f.evidence_strength if is_reinforced else "none"
        hyp = "H14" if is_reinforced and strength in ("high", "low") else "H2"
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=f.value,
            test_type="post_reinforcement", hypothesis=hyp,
            metadata={"reinforced": is_reinforced,
                       "strength": strength, "category": f.category},
        ))
        step += 1

    # ── Phase 5: Simple Contradiction ────────────────────────────────
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

    # ── Phase 5b: Chain Contradictions (H10: A→B→C) ──────────────────
    for idx in chain_set:
        f = facts[idx]
        # First contradiction: A→B
        val_b = f.chain_values[0]
        templates = CONTRADICT_TEMPLATES[f.attribute]
        text = rng.choice(templates).format(
            entity=f.entity, old_value=f.value, new_value=val_b)
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=f.fact_id,
            metadata={"phase": "chain_contradiction", "chain_step": 1,
                       "old_value": f.value, "new_value": val_b},
        ))
        step += 1
    add_fillers(5, 10)

    # Test intermediate value (should be B)
    for idx in chain_set:
        f = facts[idx]
        val_b = f.chain_values[0]
        templates = QUERY_TEMPLATES[f.attribute]
        question = templates[0].format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=val_b,
            test_type="chain_intermediate", hypothesis="H10",
            metadata={"chain_step": "after_first_change"},
        ))
        step += 1

    # Second contradiction: B→C
    for idx in chain_set:
        f = facts[idx]
        val_b = f.chain_values[0]
        val_c = f.chain_values[1]
        templates = CONTRADICT_TEMPLATES[f.attribute]
        text = rng.choice(templates).format(
            entity=f.entity, old_value=val_b, new_value=val_c)
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=f.fact_id,
            metadata={"phase": "chain_contradiction", "chain_step": 2,
                       "old_value": val_b, "new_value": val_c},
        ))
        step += 1
    add_fillers(3, 5)

    # ── Phase 5c: Rapid Conflict (H11) ────────────────────────────────
    # Confirm then immediately contradict within 2 steps
    for idx in rapid_set:
        f = facts[idx]
        # First: confirm the original value
        reinf_templates = REINFORCE_TEMPLATES[f.attribute]
        text = rng.choice(reinf_templates).format(entity=f.entity, value=f.value)
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=f.fact_id,
            metadata={"phase": "rapid_conflict_confirm"},
        ))
        step += 1
        # Immediately contradict (within conflict window)
        contra_templates = CONTRADICT_TEMPLATES[f.attribute]
        text = rng.choice(contra_templates).format(
            entity=f.entity, old_value=f.value, new_value=f.contradicted_by)
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=f.fact_id,
            metadata={"phase": "rapid_conflict_contradict",
                       "old_value": f.value, "new_value": f.contradicted_by},
        ))
        step += 1
    add_fillers(3, 5)

    # ── Phase 5d: Dependency Contradiction (H13) ──────────────────────
    # Contradict parent facts and later test whether children lost confidence
    dep_parent_indices_contradicted = []
    for pidx, cidx, _ in dependency_pairs:
        parent = facts[pidx]
        # Only contradict parents that aren't already contradicted
        if pidx not in contradict_set and pidx not in chain_set:
            pool = ATTRIBUTE_TO_POOL[parent.attribute]
            new_val = rng.choice([v for v in pool if v != parent.value])
            templates = CONTRADICT_TEMPLATES[parent.attribute]
            text = rng.choice(templates).format(
                entity=parent.entity, old_value=parent.value, new_value=new_val)
            stream.append(StreamItem(
                step=step, item_type="interaction", content=text,
                fact_id=parent.fact_id,
                metadata={"phase": "dependency_contradiction",
                           "old_value": parent.value, "new_value": new_val},
            ))
            step += 1
            dep_parent_indices_contradicted.append((pidx, cidx))
            parent.contradicted_by = new_val
    add_fillers(2, 4)

    # ── Phase 6: Post-Contradiction Recall (H3, H10, H11, H13) ───────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates).format(entity=f.entity)

        # Determine expected answer
        if idx in chain_set:
            expected = f.chain_values[1]  # final value C
            hyp = "H10"
        elif idx in rapid_set:
            expected = f.contradicted_by
            hyp = "H11"
        elif idx in contradict_set:
            expected = f.contradicted_by
            hyp = "H3"
        else:
            expected = f.value
            hyp = "H3"

        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=expected,
            test_type="post_contradiction", hypothesis=hyp,
            metadata={"contradicted": idx in contradict_set or idx in chain_set or idx in rapid_set,
                       "reinforced": idx in reinforce_set,
                       "chain": idx in chain_set,
                       "rapid": idx in rapid_set},
        ))
        step += 1

    # Test dependency children after parent contradiction (H13)
    for pidx, cidx in dep_parent_indices_contradicted:
        child = facts[cidx]
        templates = QUERY_TEMPLATES[child.attribute]
        question = templates[0].format(entity=child.entity)
        # Child value may still be correct but confidence should be lower
        expected = child.contradicted_by if child.contradicted_by else child.value
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=child.fact_id, expected_answer=expected,
            test_type="dependency_child", hypothesis="H13",
            metadata={"parent_fact_id": facts[pidx].fact_id,
                       "parent_contradicted": True},
        ))
        step += 1

    # ── Phase 7a: Short Delay (30 fillers) ─────────────────────────────
    for _ in range(30):
        stream.append(StreamItem(
            step=step, item_type="interaction",
            content=rng.choice(FILLERS),
            metadata={"phase": "short_delay"},
        ))
        step += 1

    # ── Phase 7b: Short Delay Recall (H12 — category decay) ──────────
    # Test each category separately to measure differential decay
    category_test_facts = {}
    for idx in order:
        f = facts[idx]
        if idx not in contradict_set and idx not in chain_set and idx not in rapid_set:
            category_test_facts.setdefault(f.category, []).append(idx)

    for cat, cat_indices in category_test_facts.items():
        sample = rng.sample(cat_indices, min(5, len(cat_indices)))
        for idx in sample:
            f = facts[idx]
            templates = QUERY_TEMPLATES[f.attribute]
            question = rng.choice(templates).format(entity=f.entity)
            stream.append(StreamItem(
                step=step, item_type="test", content=question,
                fact_id=f.fact_id, expected_answer=f.value,
                test_type="short_delay_recall", hypothesis="H12",
                metadata={"category": f.category, "delay": "short"},
            ))
            step += 1

    # ── Phase 8a: Long Delay (100 fillers) ──────────────────────────────
    for _ in range(100):
        stream.append(StreamItem(
            step=step, item_type="interaction",
            content=rng.choice(FILLERS),
            metadata={"phase": "long_delay"},
        ))
        step += 1

    # ── Phase 8b: Forgetting Curve Recall (H4 + H12) ──────────────────
    rng.shuffle(order)
    for idx in order:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates).format(entity=f.entity)
        if idx in contradict_set or idx in chain_set or idx in rapid_set:
            expected = f.contradicted_by
        else:
            expected = f.value
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=expected,
            test_type="forgetting_curve", hypothesis="H4",
            metadata={"reinforced": idx in reinforce_set,
                       "contradicted": idx in contradict_set or idx in chain_set,
                       "category": f.category,
                       "strength": f.evidence_strength if idx in reinforce_set else "none"},
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
    # Test with confusable entity-attribute pairs (same attribute, different entities)
    distractor_pairs = []
    for attr, attr_indices in facts_by_attr.items():
        if len(attr_indices) >= 2:
            avail = [i for i in attr_indices if i not in forget_indices]
            if len(avail) >= 2:
                pairs = []
                for _ in range(3):  # up to 3 pairs per attribute
                    if len(avail) < 2:
                        break
                    pair = rng.sample(avail, 2)
                    pairs.append(pair)
                    avail = [x for x in avail if x not in pair]
                distractor_pairs.extend(pairs)

    # Limit to 15 distractor tests
    rng.shuffle(distractor_pairs)
    distractor_pairs = distractor_pairs[:15]

    for idx_a, idx_b in distractor_pairs:
        fa, fb = facts[idx_a], facts[idx_b]
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

    # ── Phase 11: Coreset Stress Test (H6) ──────────────────────────
    # Add a burst of 30+ new facts for a single entity to overflow the coreset
    stress_entity = rng.choice(ENTITIES)
    stress_facts_ids = []
    stress_attrs = list(ATTRIBUTE_TO_POOL.keys())
    for i in range(25):
        attr = stress_attrs[i % len(stress_attrs)]
        pool = ATTRIBUTE_TO_POOL[attr]
        value = rng.choice(pool)
        fid_str = f"FS{i:03d}"
        stress_facts_ids.append(fid_str)
        templates = PRESENT_TEMPLATES[attr]
        text = rng.choice(templates).format(entity=stress_entity, value=value)
        stream.append(StreamItem(
            step=step, item_type="interaction", content=text,
            fact_id=fid_str,
            metadata={"phase": "coreset_stress", "entity": stress_entity,
                       "attribute": attr},
        ))
        step += 1

    add_fillers(5, 10)

    # Now test recall of facts from earlier phases (should be retained by coreset)
    coreset_test_sample = rng.sample(
        [idx for idx in order if idx not in forget_indices
         and idx not in contradict_set and idx not in chain_set
         and idx not in rapid_set],
        min(10, len(order))
    )
    for idx in coreset_test_sample:
        f = facts[idx]
        templates = QUERY_TEMPLATES[f.attribute]
        question = rng.choice(templates).format(entity=f.entity)
        stream.append(StreamItem(
            step=step, item_type="test", content=question,
            fact_id=f.fact_id, expected_answer=f.value,
            test_type="coreset_stress_recall", hypothesis="H6",
            metadata={"after_stress": True},
        ))
        step += 1

    # ── Build output ─────────────────────────────────────────────────
    n_interactions = sum(1 for s in stream if s.item_type == "interaction")
    n_tests = sum(1 for s in stream if s.item_type == "test")
    n_forgets = sum(1 for s in stream if s.item_type == "forget")

    # Count tests by type
    test_type_counts = {}
    for s in stream:
        if s.item_type == "test":
            test_type_counts[s.test_type] = test_type_counts.get(s.test_type, 0) + 1

    return {
        "metadata": {
            "n_entities": len(ENTITIES),
            "n_facts": n_facts,
            "n_reinforced": n_reinforce,
            "n_reinforced_strong": n_reinforce_strong,
            "n_reinforced_weak": n_reinforce_weak,
            "n_contradicted": n_contradict,
            "n_chain_contradicted": n_chain_contradict,
            "n_rapid_conflict": n_rapid_conflict,
            "n_forget_targets": n_forget,
            "n_dependencies": len(dependency_pairs),
            "n_distractors": len(distractor_pairs),
            "total_interactions": n_interactions,
            "total_tests": n_tests,
            "total_forgets": n_forgets,
            "total_steps": step,
            "test_type_counts": test_type_counts,
            "seed": seed,
            "hypotheses": [f"H{i}" for i in range(1, 15)],
        },
        "facts": [asdict(f) for f in facts],
        "stream": [asdict(s) for s in stream],
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

    print("Generating MemBayes benchmark (expanded)...")
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
    print(f"\n  Test type breakdown:")
    for tt, count in sorted(m["test_type_counts"].items()):
        print(f"    {tt}: {count}")
    print(f"\n  Saved to {out}")
