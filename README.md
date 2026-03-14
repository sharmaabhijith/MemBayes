# MemBayes: Scalable Symbolic VCL for LLM Memory Management

A framework that adapts **Variational Continual Learning** ([Nguyen et al., ICLR 2018](https://arxiv.org/abs/1710.10628)) from maintaining distributions over neural network weights to managing LLM agent memory. Each memory is a Bernoulli random variable whose log-odds are updated via exact Bayesian inference, decayed adaptively per category, stabilized via importance-weighted coreset replay, and propagated through a dependency graph. No gradients, no fine-tuning.

The framework features a three-layer architecture, per-category adaptive decay, dependency-aware belief propagation, bounded updates, importance-weighted coresets, memory consolidation, and conflict resolution.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with heuristic mode (no API key needed)
python run_experiments.py

# Run with LLM mode (requires DeepInfra API key)
export DEEPINFRA_API_KEY="your-key-here"
python run_experiments.py --llm
python run_experiments.py --llm --model Qwen/Qwen3-235B-A22B
```

Results are saved to `results/` (plots + JSON).

---

## Project Structure

```
MemBayes/
├── membayes/                    # Core package (three-layer architecture)
│   ├── __init__.py              # Package exports
│   ├── config.py                # VCLConfig: all hyperparameters with Bayesian interpretations
│   ├── memory_entry.py          # MemoryEntry: 13-field extended entry with entity/deps/embedding
│   ├── retrieval.py             # Retrieval Layer: EntityIndex + EmbeddingIndex + ClusterIndex
│   ├── semantic.py              # Semantic Layer: LLM fact extraction, classification, disambiguation
│   ├── bayesian.py              # Bayesian Layer: bounded updates, adaptive decay, dependency propagation
│   ├── coreset.py               # Importance-weighted coreset with diminishing replay returns
│   ├── consolidation.py         # Memory consolidation via variational compression
│   ├── dependencies.py          # DAG-based dependency graph with belief propagation
│   └── vcl_memory.py            # Main orchestrator: ties all three layers together
│
├── llm_client.py                # DeepInfra/Qwen API client (OpenAI-compatible)
├── benchmark.py                 # Synthetic benchmark generator + evaluation metrics
├── baselines.py                 # 3 baseline systems: Naive, Sliding Window, Decay Only
├── run_experiments.py           # Experiment runner (6 systems, tables, plots)
├── requirements.txt             # Python dependencies
├── methodology.tex              # LaTeX methodology paper
├── framework.md                 # Framework design document
└── results/                     # Generated outputs (plots, JSON)
```

---

## Architecture: Three-Layer Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL LAYER                              │
│                                                                      │
│  1. Entity Index:    hash map entity_id → memory set         O(1)    │
│  2. Embedding Index: approximate nearest neighbor search   O(log K)  │
│  3. Cluster Index:   GMM soft assignments per entity       O(G_i)    │
│  4. Candidate Selection: top-N by cosine similarity                  │
│                                                                      │
│  Input:  raw interaction text                                        │
│  Output: candidate set C(x), |C(x)| ≤ N (default N=10)              │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ candidate set C(x)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     SEMANTIC LAYER (LLM)                              │
│                                                                      │
│  1. Fact Extraction:     text → (entity, attribute, value)           │
│  2. Evidence Classification: confirm / contradict / unrelated        │
│     + evidence_strength: high / medium / low                         │
│     + is_correction: bool                                            │
│     + detected_dependencies: list                                    │
│  3. Conflict Disambiguation: correction / context_dependent / noise  │
│  4. Query Answering:    confidence-weighted retrieval                 │
│                                                                      │
│  Only runs against C(x), not the full memory store                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ structured classification + metadata
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     BAYESIAN LAYER (Deterministic)                    │
│                                                                      │
│  1. Source-Weighted Update: w(e) = w_base · α_strength · α_source    │
│  2. Bounded Update (KL Cap): Δℓ = clamp(w, -Δ_max, +Δ_max)         │
│  3. Dependency Propagation: BFS with depth-attenuated penalties      │
│  4. Adaptive Decay: λ_eff = λ_base(category) · access_modifier      │
│  5. Importance-Weighted Coreset Replay (diminishing returns)         │
│  6. Consolidation Trigger Check                                      │
│  7. Decision Boundary for Value Replacement                          │
└──────────────────────────────────────────────────────────────────────┘
```

**Key principle**: The Retrieval Layer reduces classification from O(K) to O(N). The Semantic Layer determines *what* information means. The Bayesian Layer determines *how much* to change confidence — all deterministic, no LLM calls.

---

## Core Concepts

### Memory Entries as Bernoulli Variables

Each memory entry `m_k` is a Bernoulli random variable: "Is this entry currently correct?"

```
p(m_k is valid) = σ(ℓ_k) = 1 / (1 + exp(-ℓ_k))
```

The **log-odds representation** `ℓ_k` is the natural parameterization of the Bernoulli exponential family. Bayesian updates become **additive**:

```
After observing evidence e:    ℓ_k ← ℓ_k + w(e)
```

This is **exact Bayesian inference** — not an approximation.

### Memory Entry (13 fields)

| Field | Type | Description |
|---|---|---|
| `entry_id` | str | Unique identifier |
| `key` | str | Structured key (entity.attribute format) |
| `value` | str | Current believed value |
| `log_odds` | float | Log-odds of validity (ℓ_k) |
| `entity_id` | str | Entity this memory belongs to |
| `attribute_type` | enum | identity / preference / episodic / relational / transient |
| `dependencies` | list | Parent memory IDs this entry depends on |
| `embedding` | list[float] | Vector representation for retrieval |
| `cluster_id` | int | GMM cluster assignment |
| `created_at` | int | Creation timestamp |
| `last_accessed` | int | Last access timestamp |
| `access_count` | int | Number of times accessed |
| `status` | enum | active / uncertain / decayed / consolidated / forgotten |

### Attribute Categories

| Category | λ_base | Half-life | Examples |
|---|---|---|---|
| identity | 0.001 | ~693 steps | Name, birthdate, nationality |
| relational | 0.005 | ~139 steps | Works with, sister is |
| preference | 0.008 | ~87 steps | Favorite color, food, hobby |
| episodic | 0.030 | ~23 steps | Met at cafe, went to store |
| transient | 0.100 | ~7 steps | Current mood, today's plan |

---

## Algorithm: Per-Interaction Update

```
VCL_MEMORY_UPDATE(interaction x_t at step t):

Step 1 — ADAPTIVE TEMPORAL DECAY
    For each active entry m_k:
        λ_eff = λ_base(category) · exp(-β_access · min(n_access, 20))
        ℓ_k = ℓ_k · exp(-λ_eff · Δt)
        if σ(ℓ_k) < 0.12: mark as "decayed"
        elif σ(ℓ_k) < 0.30: mark as "uncertain"

Step 2 — CANDIDATE RETRIEVAL (Retrieval Layer)
    C(x) = EntityIndex(entity) ∪ EmbeddingIndex.query(embed(x), top-10)
    C_filtered = {m ∈ C(x) : confidence > 0.10}

Step 3 — SEMANTIC CLASSIFICATION (on C_filtered only)
    (type, fact_id, value, strength, dependencies) = LLM_classify(x, C_filtered)

Step 4 — CONFLICT CHECK
    If CONTRADICT and recent CONFIRM within 5 steps:
        resolution = LLM_disambiguate(correction / context_dependent / noise)

Step 5 — BOUNDED BAYESIAN UPDATE
    Δℓ = clamp(w_base · α_strength, -Δ_max(ℓ), +Δ_max(ℓ))
    ℓ_k = ℓ_k + Δℓ
    If contradiction crosses decision boundary (σ(ℓ) < 0.5):
        Replace value, reset log-odds at 0.8 · ℓ_0(category)
        PROPAGATE uncertainty to dependent memories

Step 6 — IMPORTANCE-WEIGHTED CORESET UPDATE
    Add to buffer; evict lowest-importance entry if full

Step 7 — PERIODIC MAINTENANCE (every R steps)
    Coreset replay with diminishing returns
    Consolidation check per entity
    Index pruning
```

---

## Key Features

### Bounded Updates (KL Regularization Analog)

Prevents unbounded confidence accumulation:

```
Δ_max(ℓ) = Δ_base / (1 + β · |ℓ|)

Fresh memory   (ℓ ≈ 1.2): can change by up to ~1.7
Reinforced     (ℓ ≈ 4.0): can change by up to ~1.25
Entrenched     (ℓ ≈ 8.0): can change by up to ~0.9
```

Even strongly held memories remain revisable under sufficient contradicting evidence.

### Dependency-Aware Belief Propagation

When a memory's value is replaced, dependents receive depth-attenuated penalties:

```
"Zara lives in Tokyo" → "Zara commutes on Yamanote line"  (penalty = -0.5)
                       → "Zara's timezone is JST"          (penalty = -0.5)
                           → "Zara's work hours" (2nd hop)  (penalty = -0.3)
```

Propagation parameters: `w_dep = -0.5`, `γ = 0.6` (per-hop attenuation), `D_max = 3` hops.

### Importance-Weighted Coreset

Replaces FIFO eviction with composite scoring:

```
I(x) = 0.4 · confidence_impact    (σ'(ℓ) — highest near decision boundary)
     + 0.2 · diversity_score      (inverse overlap with other buffer entries)
     + 0.2 · recency_score        (exponential decay on age)
     + 0.2 · category_weight      (identity: 1.5, transient: 0.3)
```

The entry with highest confidence impact is protected from eviction.

### Memory Consolidation

When an entity accumulates > 20 entries, related high-confidence entries are merged:

```
ℓ_summary = (Σ_k c_k · ℓ_k) / (Σ_k c_k)     (confidence-weighted aggregate)
```

Source entries are marked as `consolidated` (kept for audit, excluded from queries). This bounds effective memory size to O(N_entities · (K_consol + log T)).

### Conflict Resolution

When contradicting evidence arrives within 5 steps of confirming evidence:

| Resolution | Action |
|---|---|
| **Correction** | Apply contradiction at full strength |
| **Context-dependent** | Split into two context-tagged entries |
| **Noise** | Apply contradiction at reduced strength (α=0.3) |

---

## Evidence Weights

| Evidence Type | Symbol | Base Weight | Strength-Modulated Range | Effect |
|---|---|---|---|---|
| Confirming | w+ | +0.7 | [+0.28, +0.70] | Increases confidence |
| Contradicting | w- | -1.0 | [-1.00, -0.40] | Decreases confidence |
| Coreset replay | w_c | +0.12 | Diminishing over replays | Stabilizing boost |
| Dependency hit | w_dep | -0.5 | Depth-attenuated | Weakens dependent fact |
| Unrelated | — | 0 | 0 | No change |

**Design**: |w-| > w+ because contradictions carry more information than confirmations.

Evidence strength modulation:
- **high** (α=1.0): Explicit declaration ("my name is...", "I definitely prefer...")
- **medium** (α=0.7): Conversational statement
- **low** (α=0.4): Hedged language ("I think...", "maybe...", "probably...")

---

## LLM Integration (DeepInfra / Qwen)

The semantic layer uses **Qwen models** via [DeepInfra](https://deepinfra.com)'s OpenAI-compatible API for four operations:

### 1. Fact Extraction
```
Input:  "Zara mentioned her favorite color is cerulean."
Output: {key: "zara.favorite_color", value: "cerulean",
         entity: "zara", attribute_type: "preference"}
```

### 2. Evidence Classification
```
Input:  Existing: zara.favorite_color = "cerulean"
        New: "Actually, Zara now prefers amber."
Output: {classification: "contradict", new_value: "amber",
         evidence_strength: "high", is_correction: true,
         detected_dependencies: ["zara.wardrobe_preference"]}
```

### 3. Conflict Disambiguation
```
Input:  Recent confirm + immediate contradict for same memory
Output: {resolution: "correction" | "context_dependent" | "noise"}
```

### 4. Query Answering
```
Input:  "What is Zara's favorite color?"
        Candidates: [{zara.favorite_color: "amber", confidence: 0.85}, ...]
Output: {entry_id: "F001", answer: "amber"}
```

### Configuration

```bash
export DEEPINFRA_API_KEY="your-key-here"

# Default model (Qwen3-8B — fast)
python run_experiments.py --llm

# Larger model (better accuracy)
python run_experiments.py --llm --model Qwen/Qwen3-235B-A22B
```

The system works **without an API key** using heuristic keyword matching — useful for testing the Bayesian engine independently.

---

## Benchmark Design

The benchmark generates synthetic facts using fantasy entities (Zara Melikova, Threndel, grillberry stew) to avoid overlap with LLM pretraining knowledge.

### Five Phases

| Phase | What Happens | Interactions | Hypothesis Tested |
|---|---|---|---|
| 1. Present | All 20 facts introduced with filler | ~55 | H1: Retention |
| 2. Reinforce | 5 facts repeated 3× each | ~20 | H2: Reinforcement |
| 3. Contradict | 5 facts given conflicting values | ~10 | H3: Contradiction |
| 4. Long delay | 60 filler interactions | 60 | H4: Forgetting |
| 5. Selective forget | 3 facts explicitly deleted | 3 | H7: Targeted forgetting |

### Seven Hypotheses

- **H1: Retention** — Memories persist across intervening interactions
- **H2: Reinforcement** — Repeated evidence increases confidence
- **H3: Contradiction** — Conflicting evidence decreases confidence and updates value
- **H4: Forgetting** — Confidence decays over time (exponential decay)
- **H5: Calibration** — Confidence scores predict accuracy (low ECE)
- **H6: Coreset benefit** — Episodic replay improves long-term retention
- **H7: Selective forgetting** — Targeted deletion without collateral damage

### Metrics

- **Accuracy**: fraction of test probes returning the correct value
- **ECE** (Expected Calibration Error): gap between predicted confidence and actual accuracy
- **Bayesian consistency checks**:
  - H2: avg confidence post-reinforcement > avg confidence post-initial
  - H3: avg confidence post-contradiction < avg confidence post-reinforcement
  - H4: avg confidence post-delay < avg confidence post-initial

---

## Systems Compared

| System | Bayesian Updates | Temporal Decay | Coreset | Bounded Updates | Dependencies | LLM |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **VCL (full)** | yes | adaptive per-category | importance-weighted | KL-capped | yes | optional |
| VCL (no coreset) | yes | adaptive per-category | — | KL-capped | yes | — |
| VCL (no decay) | yes | — | importance-weighted | KL-capped | yes | — |
| Naive | — | — | — | — | — | — |
| Sliding Window | — | implicit (eviction) | — | — | — | — |
| Decay Only | — | global λ | — | — | — | — |

---

## Hyperparameters

All hyperparameters have direct Bayesian interpretations. See [config.py](membayes/config.py) for the full `VCLConfig` class.

### Bayesian Core

| Parameter | Symbol | Default | Bayesian Meaning |
|---|---|---|---|
| Initial log-odds (preference) | ℓ_0 | 1.2 | Prior strength: σ(1.2) ≈ 0.77 |
| Initial log-odds (identity) | ℓ_0 | 2.0 | Strong prior: σ(2.0) ≈ 0.88 |
| Confirm weight | w+ | 0.7 | Confirming log-likelihood ratio |
| Contradict weight | w- | -1.0 | Contradicting log-likelihood ratio |
| KL cap base | Δ_base | 2.0 | Max single-step posterior shift |
| KL cap dampening | β | 0.15 | Entrenchment resistance |
| Replace threshold | τ_replace | 0.5 | Decision boundary (max entropy) |

### Temporal Decay

| Parameter | Symbol | Default | Bayesian Meaning |
|---|---|---|---|
| Decay rate (identity) | λ_ident | 0.001 | Near-permanent prior |
| Decay rate (preference) | λ_pref | 0.008 | Stable, moderate drift |
| Decay rate (transient) | λ_trans | 0.100 | Fast-expiring prior |
| Access modulation | β_access | 0.05 | Implicit relevance evidence |
| Soft decay threshold | τ_soft | 0.30 | Uncertainty flag boundary |
| Hard decay threshold | τ_hard | 0.12 | Effective forgetting boundary |

### Coreset

| Parameter | Symbol | Default | Bayesian Meaning |
|---|---|---|---|
| Buffer capacity | \|C\|_max | 20 | Episodic buffer size |
| Replay weight | w_c | 0.12 | Per-replay log-odds boost |
| Replay interval | R | 10 | Steps between replay cycles |
| Diminishing rate | δ | 0.1 | Replay returns decay speed |

### Dependencies

| Parameter | Symbol | Default | Bayesian Meaning |
|---|---|---|---|
| Dependency penalty | w_dep | -0.5 | Conditional prior invalidation |
| Depth attenuation | γ | 0.6 | Multi-hop weakening |
| Max depth | D_max | 3 | Cascade limit |

---

## Scalability Analysis

| Operation | Complexity | Details |
|---|---|---|
| Classification per step | O(N) LLM calls, N≤10 | Bounded by candidate budget |
| Retrieval per step | O(log K) via indices | Entity hash + embedding ANN |
| Decay per step | O(K_active) + amortized clusters | Per-category adaptive rates |
| Memory growth | O(N_e · (K_c + log T)) | Sublinear via consolidation |

### Projected Performance

| Metric | T=100 | T=1,000 | T=10,000 | T=100,000 |
|---|---|---|---|---|
| LLM calls/step | ~10 | ~10 | ~10 | ~10 |
| Effective memories | ~100 | ~300 | ~800 | ~2,000 |

---

## Mathematical Foundation

### VCL Mapping

| VCL (Original) | MemBayes |
|---|---|
| θ = model weights | M = memory entries |
| q_t(θ) = Gaussian posterior over weights | q_t(m_k) = Bernoulli: "is entry k valid?" |
| p(D_t\|θ) = data likelihood | w(e) · α_strength: modulated evidence weight |
| KL(q_t \|\| q_{t-1}) = bounded weight change | Δ_max(ℓ) = Δ_base / (1 + β·\|ℓ\|) |
| Coreset = stored data points | Importance-weighted episodic buffer |
| N/A | Adaptive per-category temporal decay |
| N/A | Dependency DAG with belief propagation |

### Temporal Decay (Bayesian Interpretation)

```
q_t(M) ∝ q_{t-1}(M)^α · p_0(M)^{1-α}

where α = exp(-λ_eff · Δt) is the retention rate
      p_0(M) is the uninformative prior (confidence = 0.5)
```

When α = 1 (no time passed): posterior unchanged.
When α → 0 (long delay): posterior reverts to maximum-entropy prior.

---

## Usage as a Library

```python
from membayes import VCLMemory, VCLConfig

# Initialize with custom config
config = VCLConfig(
    initial_log_odds={"identity": 2.5, "preference": 1.5, "transient": 0.3},
    decay_rates={"identity": 0.0005, "preference": 0.01, "transient": 0.15},
    coreset_size=30,
)

memory = VCLMemory(config=config, use_coreset=True)

# Process interactions
memory.process_interaction({
    "step": 0,
    "interaction_type": "inform",
    "content": "Zara's favorite color is cerulean.",
    "facts_involved": ["zara_color"],
})

# Reinforce
memory.process_interaction({
    "step": 5,
    "interaction_type": "reinforce",
    "content": "Zara confirmed she still loves cerulean.",
    "facts_involved": ["zara_color"],
})

# Query
result = memory.answer_query({
    "fact_id": "zara_color",
    "question": "What is Zara's favorite color?",
})
print(f"Answer: {result['answer']}, Confidence: {result['confidence']:.2f}")

# Inspect state
state = memory.get_state()
print(f"Active: {state['n_active']}, Avg confidence: {state['avg_confidence']:.2f}")
```

---

## References

- Nguyen, C. V., Li, Y., Bui, T. D., & Turner, R. E. (2018). *Variational Continual Learning*. ICLR 2018. [arXiv:1710.10628](https://arxiv.org/abs/1710.10628)
