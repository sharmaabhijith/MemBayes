"""
Memory Consolidation via Variational Compression
==================================================

Compresses related memories within an entity cluster into summary entries,
reducing storage and improving retrieval coherence.

Trigger: entity cluster exceeds K_consol entries OR majority are uncertain.

The summary's log-odds are computed as a confidence-weighted aggregate:
    ℓ_summary = (Σ_k c_k · ℓ_k) / (Σ_k c_k)

This is the symbolic analog of variational compression in VCL.
"""

from __future__ import annotations

import logging
from typing import Optional

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Merges related memories into summaries when clusters grow too large."""

    def __init__(self, config: VCLConfig):
        self.config = config
        self._summary_counter = 0

    def should_consolidate(self, entity_entries: list[MemoryEntry]) -> bool:
        """Check if an entity's memories should be consolidated.

        Triggers when:
            - |entries| > K_consol, OR
            - majority of entries are below consolidation threshold
        """
        active = [e for e in entity_entries if e.is_active]
        if not active:
            return False

        if len(active) > self.config.consolidation_trigger:
            return True

        uncertain_count = sum(
            1 for e in active
            if e.confidence < self.config.consolidation_threshold
        )
        return uncertain_count > len(active) / 2

    def consolidate(self, entity_entries: list[MemoryEntry],
                    entity_id: str, current_step: int) -> list[MemoryEntry]:
        """Consolidate entries for an entity. Returns list of new summary entries.

        Procedure:
            1. Group by attribute_type (or cluster_id)
            2. For each group with ≥ 3 entries:
               a. Separate high-confidence and low-confidence entries
               b. Create summary from high-confidence entries
               c. Mark source entries as consolidated
               d. Push low-confidence entries toward forgetting
        """
        # Group by attribute type
        groups: dict[str, list[MemoryEntry]] = {}
        for entry in entity_entries:
            if not entry.is_active:
                continue
            key = entry.attribute_type
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)

        summaries = []
        for attr_type, group in groups.items():
            if len(group) < 3:
                continue

            # Separate by confidence
            high_conf = [
                e for e in group
                if e.confidence >= self.config.soft_decay_threshold
            ]
            low_conf = [
                e for e in group
                if e.confidence < self.config.soft_decay_threshold
            ]

            # Create summary from high-confidence entries
            if len(high_conf) >= 2:
                summary = self._create_summary(
                    high_conf, entity_id, attr_type, current_step
                )
                summaries.append(summary)

                # Mark source entries as consolidated
                for entry in high_conf:
                    entry.status = "consolidated"

            # Push low-confidence entries toward forgetting
            for entry in low_conf:
                entry.log_odds -= 0.5
                entry.update_confidence()
                if entry.confidence < self.config.hard_decay_threshold:
                    entry.status = "decayed"

        if summaries:
            logger.debug(
                "Consolidated entity %s: %d summaries created",
                entity_id, len(summaries)
            )
        return summaries

    def _create_summary(self, entries: list[MemoryEntry],
                        entity_id: str, attr_type: str,
                        current_step: int) -> MemoryEntry:
        """Create a summary memory from a group of high-confidence entries.

        ℓ_summary = (Σ_k c_k · ℓ_k) / (Σ_k c_k)
        """
        self._summary_counter += 1
        summary_id = f"S{self._summary_counter:04d}"

        # Confidence-weighted average of log-odds
        total_weight = sum(e.confidence for e in entries)
        if total_weight > 0:
            summary_log_odds = sum(
                e.confidence * e.log_odds for e in entries
            ) / total_weight
        else:
            summary_log_odds = 0.0

        # Build summary content from highest-confidence entry
        best_entry = max(entries, key=lambda e: e.confidence)

        # Combine values
        values = list(set(e.value for e in entries))
        if len(values) == 1:
            summary_value = values[0]
            summary_content = best_entry.content
        else:
            summary_value = best_entry.value
            summary_content = f"Summary of {len(entries)} {attr_type} entries for {entity_id}"

        summary = MemoryEntry(
            entry_id=summary_id,
            content=summary_content,
            key=f"{entity_id}.{attr_type}_summary",
            value=summary_value,
            log_odds=summary_log_odds,
            entity_id=entity_id,
            attribute_type=attr_type,
            created_at=current_step,
            last_accessed=current_step,
            access_count=sum(e.access_count for e in entries),
            status="active",
        )
        summary.update_confidence()
        return summary
