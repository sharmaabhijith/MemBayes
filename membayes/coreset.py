"""
Importance-Weighted Coreset Management
============================================

Replaces FIFO eviction with importance-weighted selection:

    I(x) = w_confidence · confidence_impact(x)
          + w_diversity  · diversity_score(x)
          + w_recency    · recency_score(x)
          + w_category   · category_weight(x)

Replay uses diminishing returns to prevent unbounded confidence growth:
    w_replay(k, n) = w_c / (1 + δ · n_replayed_k)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class CoresetItem:
    """A single entry in the episodic buffer."""
    text: str
    step: int
    affected_entry_ids: list[str] = field(default_factory=list)


class ImportanceWeightedCoreset:
    """Episodic buffer with importance-weighted eviction and diminishing replay."""

    def __init__(self, config: VCLConfig):
        self.config = config
        self.buffer: list[CoresetItem] = []

    def add(self, text: str, step: int, affected_ids: list[str],
            entries: dict[str, MemoryEntry]):
        """Add interaction to coreset, evicting lowest-importance if full."""
        item = CoresetItem(text=text, step=step, affected_entry_ids=affected_ids)
        self.buffer.append(item)

        if len(self.buffer) > self.config.coreset_size:
            self._evict(entries)

    def replay(self, entries: dict[str, MemoryEntry]):
        """Replay coreset with diminishing returns.

        w_replay(k, n) = w_c / (1 + δ · n_replayed_k)
        """
        w_c = self.config.coreset_replay_weight
        delta = self.config.replay_diminishing_rate

        for item in self.buffer:
            for entry_id in item.affected_entry_ids:
                if entry_id in entries:
                    entry = entries[entry_id]
                    if entry.is_active:
                        # Diminishing returns
                        w_eff = w_c / (1.0 + delta * entry.n_replayed)
                        entry.log_odds += w_eff
                        entry.n_replayed += 1
                        entry.update_confidence()

    def _evict(self, entries: dict[str, MemoryEntry]):
        """Evict the lowest-importance entry, protecting the highest-impact one."""
        if len(self.buffer) <= 1:
            return

        scores = []
        for i, item in enumerate(self.buffer):
            score = self._importance_score(item, entries)
            scores.append((i, score))

        # Find entry with highest confidence impact (protected)
        best_impact_idx = max(
            range(len(self.buffer)),
            key=lambda i: self._confidence_impact(self.buffer[i], entries)
        )

        # Find lowest total importance (excluding protected)
        victim_idx = -1
        victim_score = float("inf")
        for i, score in scores:
            if i == best_impact_idx:
                continue
            if score < victim_score:
                victim_score = score
                victim_idx = i

        if victim_idx >= 0:
            self.buffer.pop(victim_idx)

    def _importance_score(self, item: CoresetItem,
                          entries: dict[str, MemoryEntry]) -> float:
        """Compute composite importance score for a coreset entry."""
        w = self.config.importance_weights

        ci = self._confidence_impact(item, entries)
        di = self._diversity_score(item)
        ri = self._recency_score(item)
        cat = self._category_weight(item, entries)

        return (
            w["confidence"] * ci
            + w["diversity"] * di
            + w["recency"] * ri
            + w["category"] * cat
        )

    def _confidence_impact(self, item: CoresetItem,
                           entries: dict[str, MemoryEntry]) -> float:
        """How much would losing replay affect referenced memories?

        Approximated as Σ σ'(ℓ_k) · w_c where σ' is the sigmoid derivative.
        Entries near the decision boundary get highest impact.
        """
        w_c = self.config.coreset_replay_weight
        total = 0.0
        for eid in item.affected_entry_ids:
            if eid in entries and entries[eid].is_active:
                c = entries[eid].confidence
                # σ'(ℓ) = σ(ℓ)(1 - σ(ℓ))
                sigmoid_deriv = c * (1.0 - c)
                total += sigmoid_deriv * w_c
        return total

    def _diversity_score(self, item: CoresetItem) -> float:
        """Does this entry reference entities not covered by other coreset entries?"""
        if not item.affected_entry_ids:
            return 0.0

        # Count how many other buffer entries reference the same entries
        overlap_count = 0
        item_ids = set(item.affected_entry_ids)
        for other in self.buffer:
            if other is item:
                continue
            if set(other.affected_entry_ids) & item_ids:
                overlap_count += 1

        return 1.0 / (1.0 + overlap_count)

    def _recency_score(self, item: CoresetItem) -> float:
        """Exponential decay on coreset entry age."""
        if not self.buffer:
            return 0.0
        max_step = max(x.step for x in self.buffer)
        age = max_step - item.step
        return math.exp(-0.05 * age)

    def _category_weight(self, item: CoresetItem,
                         entries: dict[str, MemoryEntry]) -> float:
        """Higher weight for identity/relational facts."""
        total = 0.0
        count = 0
        for eid in item.affected_entry_ids:
            if eid in entries:
                cat = entries[eid].attribute_type
                total += self.config.get_category_importance(cat)
                count += 1
        return total / max(count, 1)

    @property
    def size(self) -> int:
        return len(self.buffer)
