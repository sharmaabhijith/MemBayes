"""
VCL Configuration
==================

All hyperparameters with direct Bayesian interpretations.
Organized by subsystem: Bayesian core, decay, coreset, dependencies, retrieval, consolidation.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class VCLConfig:
    """Complete hyperparameter configuration for MemBayes.

    Every parameter has a direct Bayesian interpretation documented inline.
    """

    # ── Bayesian Core ─────────────────────────────────────────────────────
    # Prior strength per attribute category: σ(ℓ_0) = initial confidence
    initial_log_odds: dict[str, float] = field(default_factory=lambda: {
        "identity":   2.0,   # σ(2.0) ≈ 0.88
        "preference":  1.2,   # σ(1.2) ≈ 0.77
        "episodic":    0.8,   # σ(0.8) ≈ 0.69
        "relational":  1.5,   # σ(1.5) ≈ 0.82
        "transient":   0.5,   # σ(0.5) ≈ 0.62
    })

    # Evidence weights (log-odds increments = log likelihood ratios)
    confirm_weight: float = 1.0       # w+: confirming evidence (strong enough to resist decay)
    contradict_weight: float = -1.2   # w-: contradicting evidence (|w-| > w+ by design)

    # Evidence strength multipliers: modulate w by LLM-assessed strength
    strength_multipliers: dict[str, float] = field(default_factory=lambda: {
        "high":   1.0,   # explicit declaration
        "medium": 0.7,   # conversational statement
        "low":    0.4,   # hedged language
    })

    # KL regularization analog: bounded updates
    kl_cap_base: float = 2.0     # Δ_base: max single-step log-odds change
    kl_cap_dampening: float = 0.15  # β: entrenchment resistance

    # Decision boundary for value replacement (lower = require more evidence)
    replacement_threshold: float = 0.40  # τ_replace: replace when σ(ℓ) < this

    # ── Temporal Decay ────────────────────────────────────────────────────
    # Per-category base decay rates (λ_base)
    decay_rates: dict[str, float] = field(default_factory=lambda: {
        "identity":   0.0003,  # half-life ~2310 steps (identity barely decays)
        "preference": 0.002,   # half-life ~347 steps
        "relational": 0.001,   # half-life ~693 steps
        "episodic":   0.008,   # half-life ~87 steps
        "transient":  0.030,   # half-life ~23 steps
    })

    # Access-modulated decay: λ_eff = λ_base · exp(-β_access · min(n_access, n_cap))
    access_modulation_strength: float = 0.15  # β_access (accessed memories resist decay)
    access_modulation_cap: int = 20           # n_cap

    # Decay thresholds
    soft_decay_threshold: float = 0.30   # τ_soft: flag as uncertain
    hard_decay_threshold: float = 0.12   # τ_hard: mark as decayed
    consolidation_threshold: float = 0.20  # τ_consol: eligible for merge

    # ── Coreset ───────────────────────────────────────────────────────────
    coreset_size: int = 40                # |C|_max: episodic buffer capacity
    coreset_replay_weight: float = 0.25   # w_c: base replay log-odds boost
    replay_interval: int = 10             # R: steps between replay cycles
    replay_diminishing_rate: float = 0.05 # δ: replay returns decay (slower diminishing)

    # Importance scoring weights for coreset eviction
    importance_weights: dict[str, float] = field(default_factory=lambda: {
        "confidence": 0.4,
        "diversity":  0.2,
        "recency":    0.2,
        "category":   0.2,
    })

    # Category weights for importance scoring
    category_importance: dict[str, float] = field(default_factory=lambda: {
        "identity":   1.5,
        "relational": 1.3,
        "preference": 1.0,
        "episodic":   0.7,
        "transient":  0.3,
    })

    # ── Dependencies ──────────────────────────────────────────────────────
    dependency_penalty: float = -0.5   # w_dep: base penalty on value change
    depth_attenuation: float = 0.6     # γ: per-hop weakening
    max_propagation_depth: int = 3     # D_max: cascade limit

    # ── Retrieval ─────────────────────────────────────────────────────────
    candidate_budget: int = 10         # N: max candidates for LLM classification
    retrieval_threshold: float = 0.10  # τ_retrieve: min confidence for retrieval
    embedding_similarity_threshold: float = 0.3  # min cosine sim for embedding retrieval

    # ── Consolidation ─────────────────────────────────────────────────────
    consolidation_trigger: int = 20    # K_consol: max entries per entity before merge
    amortize_threshold: int = 50       # T_amortize: steps before cluster-level decay

    # ── Conflict Resolution ───────────────────────────────────────────────
    conflict_window: int = 5           # T_conflict: steps within which rapid contradiction triggers disambiguation

    def get_initial_log_odds(self, category: str) -> float:
        """Get category-specific initial log-odds, with fallback to preference."""
        return self.initial_log_odds.get(category, self.initial_log_odds["preference"])

    def get_decay_rate(self, category: str) -> float:
        """Get category-specific base decay rate, with fallback to preference."""
        return self.decay_rates.get(category, self.decay_rates["preference"])

    def get_strength_multiplier(self, strength: str) -> float:
        """Get evidence strength multiplier, defaulting to medium."""
        return self.strength_multipliers.get(strength, 0.7)

    def get_category_importance(self, category: str) -> float:
        """Get category importance weight for coreset scoring."""
        return self.category_importance.get(category, 1.0)
