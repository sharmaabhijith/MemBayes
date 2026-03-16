"""
Bayesian Layer 
================================

Deterministic mathematical operations for confidence tracking:

    1. Source-weighted log-odds updates with strength modulation
    2. Bounded updates via KL cap (prevents unbounded accumulation)
    3. Adaptive per-category temporal decay with access modulation
    4. Decision boundary for value replacement
    5. Dependency propagation on value change

All operations are exact Bayesian inference for Bernoulli variables
in log-odds space, with the KL cap as the only approximation.
"""

from __future__ import annotations

import math
import logging

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry
from membayes.dependencies import DependencyGraph

logger = logging.getLogger(__name__)


class BayesianLayer:
    """Deterministic Bayesian update engine.

    No LLM calls — all operations are mathematical.
    """

    def __init__(self, config: VCLConfig, dependency_graph: DependencyGraph):
        self.config = config
        self.deps = dependency_graph

    # ── Evidence Updates ──────────────────────────────────────────────────

    def apply_confirmation(self, entry: MemoryEntry,
                           strength: str = "medium") -> float:
        """Apply confirming evidence: ℓ_k ← ℓ_k + clamp(w+ · α_strength)."""
        alpha = self.config.get_strength_multiplier(strength)
        raw_delta = self.config.confirm_weight * alpha
        delta = self._clamp_update(raw_delta, entry.log_odds)

        entry.log_odds += delta
        entry.evidence_for += 1
        entry.update_confidence()
        entry.last_update_type = "confirm"
        return delta

    def apply_contradiction(self, entry: MemoryEntry,
                            new_value: str | None,
                            strength: str = "medium",
                            entries: dict[str, MemoryEntry] | None = None,
                            step: int = 0) -> dict:
        """Apply contradicting evidence with decision boundary and dependency propagation.

        Returns dict with keys: delta, replaced, propagated
        """
        alpha = self.config.get_strength_multiplier(strength)
        raw_delta = self.config.contradict_weight * alpha
        delta = self._clamp_update(raw_delta, entry.log_odds)

        entry.log_odds += delta
        entry.evidence_against += 1
        entry.update_confidence()
        entry.last_update_type = "contradict"
        entry.last_update_step = step

        result = {"delta": delta, "replaced": False, "propagated": []}

        # Decision boundary: replace value when confidence < τ_replace
        if entry.confidence < self.config.replacement_threshold and new_value:
            old_value = entry.value
            entry.value = new_value
            # Reset with reduced prior
            cat_log_odds = self.config.get_initial_log_odds(entry.attribute_type)
            entry.log_odds = 0.8 * cat_log_odds * alpha
            entry.evidence_for = 1
            entry.evidence_against = 0
            entry.update_confidence()
            result["replaced"] = True

            # Dependency propagation
            if entries is not None:
                propagated = self.deps.propagate(entry.entry_id, entries)
                result["propagated"] = propagated

        return result

    def apply_forget(self, entry: MemoryEntry):
        """Explicit forgetting: set posterior to ~0."""
        entry.log_odds = -10.0
        entry.confidence = 0.0
        entry.status = "forgotten"

    # ── Temporal Decay ────────────────────────────────────────────────────

    def apply_decay(self, entry: MemoryEntry, current_step: int):
        """Apply adaptive temporal decay to a single entry.

        λ_eff = λ_base(category) · exp(-β_access · min(n_access, n_cap))
        ℓ_k(t) = ℓ_k(t_last) · exp(-λ_eff · Δt)
        """
        if not entry.is_active:
            return

        dt = current_step - entry.last_accessed
        if dt <= 0:
            return

        # Per-category base rate
        lambda_base = self.config.get_decay_rate(entry.attribute_type)

        # Access-modulated decay
        n_access = min(entry.access_count, self.config.access_modulation_cap)
        modifier = math.exp(-self.config.access_modulation_strength * n_access)
        lambda_eff = lambda_base * modifier

        # Apply exponential decay
        decay_factor = math.exp(-lambda_eff * dt)
        entry.log_odds *= decay_factor
        entry.last_accessed = current_step
        entry.update_confidence()

        # Threshold checks
        if entry.confidence < self.config.hard_decay_threshold:
            entry.status = "decayed"
        elif entry.confidence < self.config.soft_decay_threshold:
            entry.status = "uncertain"

    def apply_decay_all(self, entries: dict[str, MemoryEntry], current_step: int):
        """Apply decay to all active entries."""
        for entry in entries.values():
            if entry.is_active:
                self.apply_decay(entry, current_step)

    # ── Bounded Updates (KL Regularization Analog) ────────────────────────

    def _clamp_update(self, raw_delta: float, current_log_odds: float) -> float:
        """Clamp update magnitude to prevent unbounded accumulation.

        Δ_max(ℓ) = Δ_base / (1 + β · |ℓ|)

        This dampens updates for already-extreme beliefs:
        - Fresh memory (ℓ ≈ 1.2): Δ_max ≈ 1.7
        - Reinforced (ℓ ≈ 4.0): Δ_max ≈ 1.25
        - Entrenched (ℓ ≈ 8.0): Δ_max ≈ 0.9
        """
        delta_max = self.config.kl_cap_base / (
            1.0 + self.config.kl_cap_dampening * abs(current_log_odds)
        )
        return max(-delta_max, min(raw_delta, delta_max))

    # ── New Entry Creation ────────────────────────────────────────────────

    def initialize_entry(self, entry: MemoryEntry,
                         strength: str = "medium"):
        """Set initial log-odds based on category and evidence strength."""
        alpha = self.config.get_strength_multiplier(strength)
        cat_log_odds = self.config.get_initial_log_odds(entry.attribute_type)
        entry.log_odds = cat_log_odds * alpha
        entry.update_confidence()

    # ── Conflict Detection ────────────────────────────────────────────────

    def is_conflict(self, entry: MemoryEntry, current_step: int) -> bool:
        """Check if this contradiction is within the conflict window.

        A conflict is rapid contradicting evidence arriving shortly after
        confirming evidence for the same memory.
        """
        if entry.last_update_type != "confirm":
            return False
        time_since = current_step - entry.last_update_step
        return time_since < self.config.conflict_window
