"""
Retrieval Layer 
===========================

Three-index retrieval system that reduces classification from O(K) to O(log K + N):

    1. Entity Index:   hash map from entity_id → set of memory IDs (O(1) lookup)
    2. Embedding Index: approximate nearest neighbor search via sorted cosine similarity
                        (HNSW-like behavior; falls back to brute-force for small stores)
    3. Cluster Index:   GMM soft assignments for topic clustering within entities

Candidate selection merges results from all indices, applies confidence-weighted
filtering, and returns a bounded candidate set for the Semantic Layer.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from typing import Optional

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry

logger = logging.getLogger(__name__)


class EntityIndex:
    """O(1) hash map from entity_id to set of memory entry IDs."""

    def __init__(self):
        self._index: dict[str, set[str]] = defaultdict(set)

    def add(self, entity_id: str, entry_id: str):
        self._index[entity_id].add(entry_id)

    def remove(self, entity_id: str, entry_id: str):
        if entity_id in self._index:
            self._index[entity_id].discard(entry_id)
            if not self._index[entity_id]:
                del self._index[entity_id]

    def get(self, entity_id: str) -> set[str]:
        return self._index.get(entity_id, set())

    def entities(self) -> list[str]:
        return list(self._index.keys())

    def entity_size(self, entity_id: str) -> int:
        return len(self._index.get(entity_id, set()))


class EmbeddingIndex:
    """Approximate nearest-neighbor index over memory embeddings.

    Uses brute-force cosine similarity for stores < 1000 entries.
    For larger stores, a proper HNSW library (e.g., hnswlib) should be plugged in.
    """

    def __init__(self):
        self._entries: dict[str, list[float]] = {}

    def add(self, entry_id: str, embedding: list[float]):
        if embedding is not None:
            self._entries[entry_id] = embedding

    def remove(self, entry_id: str):
        self._entries.pop(entry_id, None)

    def update(self, entry_id: str, embedding: list[float]):
        if embedding is not None:
            self._entries[entry_id] = embedding

    def query(self, query_embedding: list[float], top_n: int = 10) -> list[tuple[str, float]]:
        """Return top-N entry IDs by cosine similarity to query embedding."""
        if not self._entries or query_embedding is None:
            return []

        scores = []
        for eid, emb in self._entries.items():
            sim = self._cosine_similarity(query_embedding, emb)
            scores.append((eid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def size(self) -> int:
        return len(self._entries)


class ClusterIndex:
    """GMM-inspired soft clustering within entities.

    In heuristic mode (no embeddings), clusters by attribute_type.
    With embeddings, tracks per-cluster centroid and count for incremental updates.
    """

    def __init__(self):
        # entity_id → {cluster_id → {"centroid": [...], "count": int, "entries": set}}
        self._clusters: dict[str, dict[int, dict]] = defaultdict(dict)
        self._next_cluster_id: dict[str, int] = defaultdict(int)
        # Fallback: attribute_type → cluster_id mapping per entity
        self._type_clusters: dict[str, dict[str, int]] = defaultdict(dict)

    def assign_cluster(self, entry: MemoryEntry) -> int:
        """Assign entry to a cluster. Returns cluster_id."""
        entity = entry.entity_id
        if not entity:
            return -1

        # Heuristic mode: cluster by attribute type
        if entry.embedding is None:
            return self._assign_by_type(entity, entry.attribute_type)

        # Embedding mode: find nearest cluster centroid or create new
        clusters = self._clusters[entity]
        if not clusters:
            return self._create_cluster(entity, entry.embedding, entry.entry_id)

        # Find nearest cluster
        best_id, best_sim = -1, -1.0
        for cid, info in clusters.items():
            sim = EmbeddingIndex._cosine_similarity(entry.embedding, info["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        # If close enough, join existing cluster; otherwise create new
        if best_sim > 0.7:
            self._add_to_cluster(entity, best_id, entry.embedding, entry.entry_id)
            return best_id
        else:
            return self._create_cluster(entity, entry.embedding, entry.entry_id)

    def _assign_by_type(self, entity: str, attr_type: str) -> int:
        if attr_type not in self._type_clusters[entity]:
            cid = self._next_cluster_id[entity]
            self._next_cluster_id[entity] += 1
            self._type_clusters[entity][attr_type] = cid
            self._clusters[entity][cid] = {
                "centroid": None, "count": 0, "entries": set()
            }
        return self._type_clusters[entity][attr_type]

    def _create_cluster(self, entity: str, embedding: list[float], entry_id: str) -> int:
        cid = self._next_cluster_id[entity]
        self._next_cluster_id[entity] += 1
        self._clusters[entity][cid] = {
            "centroid": list(embedding),
            "count": 1,
            "entries": {entry_id},
        }
        return cid

    def _add_to_cluster(self, entity: str, cluster_id: int,
                        embedding: list[float], entry_id: str):
        info = self._clusters[entity][cluster_id]
        info["entries"].add(entry_id)
        n = info["count"]
        info["count"] = n + 1
        # Incremental centroid update
        if info["centroid"] is not None and embedding is not None:
            eta = 1.0 / (n + 1)
            info["centroid"] = [
                c + eta * (e - c)
                for c, e in zip(info["centroid"], embedding)
            ]

    def get_cluster_entries(self, entity: str, cluster_id: int) -> set[str]:
        return self._clusters.get(entity, {}).get(cluster_id, {}).get("entries", set())

    def get_entity_clusters(self, entity: str) -> dict[int, dict]:
        return self._clusters.get(entity, {})

    def remove_entry(self, entity: str, cluster_id: int, entry_id: str):
        if entity in self._clusters and cluster_id in self._clusters[entity]:
            self._clusters[entity][cluster_id]["entries"].discard(entry_id)


class RetrievalLayer:
    """Three-index retrieval layer that produces bounded candidate sets.

    Merges results from EntityIndex, EmbeddingIndex, and ClusterIndex,
    applies confidence-weighted filtering, and returns C(x) with |C(x)| ≤ N.
    """

    def __init__(self, config: VCLConfig):
        self.config = config
        self.entity_index = EntityIndex()
        self.embedding_index = EmbeddingIndex()
        self.cluster_index = ClusterIndex()

    def add_entry(self, entry: MemoryEntry):
        """Register a new memory entry in all indices."""
        if entry.entity_id:
            self.entity_index.add(entry.entity_id, entry.entry_id)
        if entry.embedding is not None:
            self.embedding_index.add(entry.entry_id, entry.embedding)
        cluster_id = self.cluster_index.assign_cluster(entry)
        entry.cluster_id = cluster_id

    def remove_entry(self, entry: MemoryEntry):
        """Remove a memory entry from all indices."""
        if entry.entity_id:
            self.entity_index.remove(entry.entity_id, entry.entry_id)
        self.embedding_index.remove(entry.entry_id)
        if entry.entity_id and entry.cluster_id >= 0:
            self.cluster_index.remove_entry(
                entry.entity_id, entry.cluster_id, entry.entry_id)

    def update_embedding(self, entry: MemoryEntry):
        """Re-index entry after embedding change (e.g., value replacement)."""
        if entry.embedding is not None:
            self.embedding_index.update(entry.entry_id, entry.embedding)

    def get_candidates(self, query_embedding: Optional[list[float]],
                       entity_id: Optional[str],
                       entries: dict[str, MemoryEntry]) -> list[str]:
        """Retrieve bounded candidate set C(x) from all indices.

        Args:
            query_embedding: embedding of the incoming interaction (may be None)
            entity_id: extracted entity identifier (may be None)
            entries: full memory store for confidence filtering

        Returns:
            List of entry IDs, |result| ≤ config.candidate_budget
        """
        candidates: set[str] = set()
        N = self.config.candidate_budget

        # Source 1: Entity index (O(1))
        if entity_id:
            entity_entries = self.entity_index.get(entity_id)
            # Take top entries by confidence
            if entity_entries:
                scored = []
                for eid in entity_entries:
                    if eid in entries:
                        scored.append((eid, entries[eid].confidence))
                scored.sort(key=lambda x: x[1], reverse=True)
                for eid, _ in scored[:5]:
                    candidates.add(eid)

        # Source 2: Embedding index (O(log K) or O(K) brute-force)
        if query_embedding is not None:
            nn_results = self.embedding_index.query(query_embedding, top_n=N)
            for eid, sim in nn_results:
                if sim >= self.config.embedding_similarity_threshold:
                    candidates.add(eid)

        # Confidence-weighted filtering
        filtered = []
        for eid in candidates:
            if eid in entries:
                entry = entries[eid]
                if entry.is_active and entry.confidence > self.config.retrieval_threshold:
                    filtered.append(eid)

        # Budget limit
        return filtered[:N]

    def prune_inactive(self, entries: dict[str, MemoryEntry]):
        """Remove decayed/forgotten entries from indices (periodic maintenance)."""
        for eid, entry in list(entries.items()):
            if entry.status in ("decayed", "forgotten"):
                self.remove_entry(entry)
