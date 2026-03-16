"""
Dependency-Aware Belief Propagation
====================================

Maintains a DAG of memory dependencies and propagates uncertainty
when a parent memory's value is replaced.

Example:
    "Zara lives in Tokyo" → "Zara commutes on Yamanote line"
                           → "Zara's timezone is JST"

When "Zara lives in Tokyo" is contradicted and replaced, dependent
memories receive a penalty attenuated by graph distance.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict, deque

from membayes.config import VCLConfig

logger = logging.getLogger(__name__)

# Structural dependency templates: attr_type → list of attr_types that depend on it
STRUCTURAL_DEPENDENCIES = {
    "identity":  [],
    "preference": [],
    "relational": ["episodic"],
    "episodic":   [],
    "transient":  [],
}


class DependencyGraph:
    """Directed acyclic graph of memory dependencies.

    parent → children: if parent's value changes, children become uncertain.
    """

    def __init__(self, config: VCLConfig):
        self.config = config
        # parent_id → set of child_ids
        self._children: dict[str, set[str]] = defaultdict(set)
        # child_id → set of parent_ids
        self._parents: dict[str, set[str]] = defaultdict(set)

    def add_dependency(self, parent_id: str, child_id: str):
        """Register that child depends on parent."""
        if parent_id == child_id:
            return
        self._children[parent_id].add(child_id)
        self._parents[child_id].add(parent_id)

    def add_dependencies(self, child_id: str, parent_ids: list[str]):
        """Register multiple parent dependencies for a child."""
        for pid in parent_ids:
            self.add_dependency(pid, child_id)

    def get_children(self, entry_id: str) -> set[str]:
        """Get direct dependents of an entry."""
        return self._children.get(entry_id, set())

    def get_parents(self, entry_id: str) -> set[str]:
        """Get direct dependencies of an entry."""
        return self._parents.get(entry_id, set())

    def remove_entry(self, entry_id: str):
        """Remove an entry from the dependency graph."""
        # Remove as child
        for pid in list(self._parents.get(entry_id, set())):
            self._children[pid].discard(entry_id)
        self._parents.pop(entry_id, None)
        # Remove as parent
        for cid in list(self._children.get(entry_id, set())):
            self._parents[cid].discard(entry_id)
        self._children.pop(entry_id, None)

    def propagate(self, source_id: str, entries: dict) -> list[tuple[str, float]]:
        """Propagate uncertainty from source to all dependents.

        Uses BFS with depth-attenuated penalties:
            penalty(depth) = w_dep · γ^depth

        Args:
            source_id: the entry whose value was replaced
            entries: full memory store (modified in place)

        Returns:
            List of (entry_id, penalty_applied) for logging
        """
        w_dep = self.config.dependency_penalty
        gamma = self.config.depth_attenuation
        d_max = self.config.max_propagation_depth

        visited: set[str] = {source_id}
        queue: deque[tuple[str, int]] = deque()

        # Seed with direct children
        for child_id in self._children.get(source_id, set()):
            queue.append((child_id, 0))

        updates: list[tuple[str, float]] = []

        while queue:
            entry_id, depth = queue.popleft()
            if entry_id in visited or depth >= d_max:
                continue
            visited.add(entry_id)

            if entry_id not in entries:
                continue

            entry = entries[entry_id]
            if not entry.is_active:
                continue

            # Depth-attenuated penalty
            penalty = w_dep * (gamma ** depth)
            entry.log_odds += penalty
            entry.update_confidence()
            updates.append((entry_id, penalty))

            # Always propagate to children within depth limit
            for child_id in self._children.get(entry_id, set()):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        if updates:
            logger.debug(
                "Dependency propagation from %s: %d entries affected",
                source_id, len(updates)
            )
        return updates

    def detect_structural_dependencies(
        self, new_entry_id: str, new_attr_type: str,
        entity_id: str, entries: dict
    ) -> list[str]:
        """Auto-detect structural dependencies for a new entry.

        Checks if any existing memories of the same entity have attribute
        types that structurally depend on the new entry's attribute type.

        Returns list of detected parent IDs.
        """
        parent_ids = []
        for eid, entry in entries.items():
            if eid == new_entry_id:
                continue
            if entry.entity_id != entity_id:
                continue
            if not entry.is_active:
                continue
            # Check if new_attr_type depends on entry's attr_type
            deps = STRUCTURAL_DEPENDENCIES.get(entry.attribute_type, [])
            if new_attr_type in deps:
                parent_ids.append(eid)
        return parent_ids
