"""
VCL Memory System — Main Orchestrator
==========================================

Three-layer architecture:
    1. Retrieval Layer  → candidate set C(x), |C(x)| ≤ N
    2. Semantic Layer   → classification + strength + facts + dependencies
    3. Bayesian Layer   → deterministic log-odds updates

Per-interaction algorithm:
    Step 1: Adaptive temporal decay
    Step 2: Compute embedding for incoming text
    Step 3: Extract entity from incoming text (Semantic Layer)
    Step 4: Retrieve candidates (Retrieval Layer)
    Step 5: Classify evidence for each candidate (Semantic Layer)
    Step 6: Bayesian update (with conflict check + dependency propagation)
    Step 7: Coreset update
    Step 8: Periodic maintenance (replay, consolidation, index pruning)
"""

from __future__ import annotations

import logging
from typing import Optional

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry
from membayes.retrieval import RetrievalLayer
from membayes.semantic import SemanticLayer
from membayes.bayesian import BayesianLayer
from membayes.coreset import ImportanceWeightedCoreset
from membayes.consolidation import ConsolidationEngine
from membayes.dependencies import DependencyGraph

logger = logging.getLogger(__name__)


class VCLMemory:
    """Scalable Symbolic VCL Memory System.

    All three layers are fully functional:
        - Retrieval: embedding-based ANN + entity hash + GMM clustering
        - Semantic: LLM-based fact extraction, classification, disambiguation
        - Bayesian: deterministic log-odds updates, decay, bounded revision
    """

    def __init__(self, config: Optional[VCLConfig] = None,
                 use_coreset: bool = True,
                 llm_client=None):
        if llm_client is None:
            raise ValueError(
                "VCLMemory requires an LLM client for the semantic layer. "
                "Pass llm_client=LLMClient() to VCLMemory()."
            )

        self.config = config or VCLConfig()
        self.entries: dict[str, MemoryEntry] = {}
        self.step = 0
        self.use_coreset = use_coreset
        self.llm_client = llm_client

        # Three layers
        self.deps = DependencyGraph(self.config)
        self.retrieval = RetrievalLayer(self.config)
        self.semantic = SemanticLayer(self.config, llm_client)
        self.bayesian = BayesianLayer(self.config, self.deps)

        # Coreset and consolidation
        self.coreset = ImportanceWeightedCoreset(self.config)
        self.consolidation = ConsolidationEngine(self.config)

        # Track last confirm text per entry (for conflict detection)
        self._last_confirm_text: dict[str, str] = {}

        # Auto-incrementing entry ID counter
        self._next_entry_id = 0

    # ── ID Generation ──────────────────────────────────────────────────

    def _new_entry_id(self) -> str:
        self._next_entry_id += 1
        return f"M{self._next_entry_id:06d}"

    # ── Embedding ──────────────────────────────────────────────────────

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding vector for text via LLM client."""
        return self.llm_client.embed_single(text)

    # ── Public API ────────────────────────────────────────────────────────

    def process_interaction(self, text: str, step: int):
        """Process one incoming text through the three-layer pipeline.

        Args:
            text: Natural language input (e.g., user message).
            step: Current interaction step number.
        """
        self.step = step
        affected_ids: list[str] = []

        # Step 1: Decay is now deferred to periodic maintenance (every
        # replay_interval steps) so that confirmations land at full
        # strength before decay erodes them.

        # Step 2: Compute embedding for incoming text
        query_embedding = self._compute_embedding(text)

        # Step 3: Extract entity from incoming text (Semantic Layer)
        entity_id = self.semantic.extract_entity(text)

        # Step 4: Retrieve candidates (Retrieval Layer)
        candidate_ids = self.retrieval.get_candidates(
            query_embedding, entity_id, self.entries
        )

        if not candidate_ids:
            # No candidates found — check if this contains a memorable fact
            # If no entity was extracted, this is likely a general question
            # (e.g., "How do magnets work?") that shouldn't be stored
            if not entity_id:
                logger.debug("No entity or candidates, skipping: %s", text[:80])
                return
            # Entity present but no match — treat as new information
            new_ids = self._handle_new_fact(text, query_embedding)
            affected_ids.extend(new_ids)
        else:
            # Step 5-6: Classify and update each candidate
            found_match = False
            for cid in candidate_ids:
                entry = self.entries[cid]
                classification = self.semantic.classify_evidence(entry, text)
                action = classification["classification"]

                if action == "confirm":
                    strength = classification.get("evidence_strength", "medium")
                    self.bayesian.apply_confirmation(entry, strength)
                    entry.last_accessed = self.step
                    entry.access_count += 1
                    self._last_confirm_text[cid] = text
                    affected_ids.append(cid)
                    found_match = True

                elif action == "contradict":
                    self._handle_contradiction(entry, text, classification)
                    affected_ids.append(cid)
                    found_match = True

                # "unrelated" → skip

            # If no candidate was confirm/contradict, this is new information
            if not found_match:
                new_ids = self._handle_new_fact(text, query_embedding)
                affected_ids.extend(new_ids)

        # Step 7: Coreset update
        if self.use_coreset and affected_ids:
            self.coreset.add(text, self.step, affected_ids, self.entries)

        # Step 8: Periodic maintenance
        if self.step > 0 and self.step % self.config.replay_interval == 0:
            self._periodic_maintenance()

    def forget(self, text: str, step: int):
        """Explicitly forget memories matching the given text.

        Uses retrieval + semantic classification to identify targets.

        Args:
            text: Description of what to forget.
            step: Current interaction step number.
        """
        self.step = step

        query_embedding = self._compute_embedding(text)
        entity_id = self.semantic.extract_entity(text)
        candidate_ids = self.retrieval.get_candidates(
            query_embedding, entity_id, self.entries
        )

        for cid in candidate_ids:
            entry = self.entries[cid]
            classification = self.semantic.classify_evidence(entry, text)
            # If the text matches (confirm or contradict), this is the target
            if classification["classification"] in ("confirm", "contradict"):
                self.bayesian.apply_forget(entry)
                # Remove from embedding index so it won't match by similarity,
                # but keep in entity index as a tombstone so entity-based
                # queries can detect the forgotten fact.
                self.retrieval.embedding_index.remove(entry.entry_id)
                logger.debug("Forgot entry %s (tombstone kept): %s",
                             cid, entry.key)

    def answer_query(self, question: str) -> dict:
        """Answer a question using retrieval + confidence-weighted ranking.

        Implements Algorithm 3 from the paper:
            1. Embed the question
            2. Retrieve candidates via Retrieval Layer
            3. Use Semantic Layer to pick best match from candidates
            4. Return answer with confidence and uncertainty flag

        Args:
            question: Natural language question.

        Returns:
            Dict with: answer, confidence, uncertainty_flag, entry_id
        """
        # Step 1: Embed the question
        query_embedding = self._compute_embedding(question)

        # Step 2: Extract entity hint
        entity_id = self.semantic.extract_entity(question)

        # Step 3: Retrieve candidates (include forgotten tombstones so we
        # can detect explicitly forgotten facts)
        candidate_ids = self.retrieval.get_candidates(
            query_embedding, entity_id, self.entries,
            include_forgotten=True,
        )

        if not candidate_ids:
            return {
                "answer": "I don't have that information",
                "confidence": 0.0,
            }

        # Step 4: Check for forgotten tombstones — if the best semantic
        # match for this entity+attribute was explicitly forgotten, return
        # "I don't have that information" instead of falling through.
        all_matched = [
            self.entries[cid] for cid in candidate_ids
            if cid in self.entries
        ]
        queryable = [e for e in all_matched if e.is_queryable]
        forgotten = [e for e in all_matched if e.status == "forgotten"]

        # If there are forgotten entries but no queryable ones for this
        # query, the user explicitly forgot this fact.
        if forgotten and not queryable:
            return {
                "answer": "I don't have that information",
                "confidence": 0.0,
            }

        if not queryable:
            return {
                "answer": "I don't have that information",
                "confidence": 0.0,
            }

        # Step 5: Use Semantic Layer to pick best match from queryable entries
        result = self.semantic.answer_query(question, queryable)

        matched_id = result.get("entry_id")
        if matched_id and matched_id in self.entries:
            entry = self.entries[matched_id]
            if entry.status == "forgotten":
                return {
                    "answer": "I don't have that information",
                    "confidence": 0.0,
                }
            entry.last_accessed = self.step
            entry.access_count += 1
            return {
                "answer": result.get("answer", entry.value),
                "confidence": round(entry.confidence, 4),
                "uncertainty_flag": entry.status == "uncertain",
                "entry_id": matched_id,
            }

        # Fallback: check if we should prefer a forgotten match
        # (e.g., the LLM didn't pick any entry but there's a tombstone)
        if forgotten:
            # Use semantic layer to check if the question matches any
            # forgotten entry — if so, this fact was explicitly forgotten
            forgotten_result = self.semantic.answer_query(question, forgotten)
            if forgotten_result.get("entry_id"):
                return {
                    "answer": "I don't have that information",
                    "confidence": 0.0,
                }

        # Fallback: highest confidence queryable candidate
        best = max(queryable, key=lambda e: e.confidence)
        best.last_accessed = self.step
        best.access_count += 1
        return {
            "answer": best.value,
            "confidence": round(best.confidence, 4),
            "entry_id": best.entry_id,
        }

    # ── Internal Handlers ─────────────────────────────────────────────────

    def _handle_new_fact(self, text: str,
                         query_embedding: list[float]) -> list[str]:
        """Create a new memory entry from incoming text.

        Returns list of created entry IDs.
        """
        extracted = self.semantic.extract_fact(text)
        entry_id = self._new_entry_id()

        entry = MemoryEntry(
            entry_id=entry_id,
            content=text,
            key=extracted.get("key", entry_id),
            value=extracted.get("value", text[:50]),
            entity_id=extracted.get("entity", ""),
            attribute_type=extracted.get("attribute_type", "preference"),
            embedding=query_embedding,
            created_at=self.step,
            last_accessed=self.step,
            last_update_type="inform",
            last_update_step=self.step,
        )
        self.bayesian.initialize_entry(entry, strength="medium")

        # Detect structural dependencies
        if entry.entity_id:
            parent_ids = self.deps.detect_structural_dependencies(
                entry_id, entry.attribute_type, entry.entity_id, self.entries
            )
            entry.dependencies = parent_ids
            self.deps.add_dependencies(entry_id, parent_ids)

        self.entries[entry_id] = entry

        # Add to retrieval indices (embedding is set, so all indices get populated)
        self.retrieval.add_entry(entry)

        return [entry_id]

    def _handle_contradiction(self, entry: MemoryEntry, text: str,
                              classification: dict):
        """Handle contradicting evidence with conflict check and dependency propagation."""
        new_value = classification.get("new_value")
        strength = classification.get("evidence_strength", "medium")

        # Conflict check: rapid contradict after recent confirm
        if self.bayesian.is_conflict(entry, self.step):
            confirm_text = self._last_confirm_text.get(entry.entry_id, "")
            resolution = self.semantic.disambiguate_conflict(
                entry, confirm_text, text)

            if resolution["resolution"] == "correction":
                strength = "high"
            elif resolution["resolution"] == "context_dependent":
                self._split_context(entry, text,
                                    resolution.get("context_1", ""),
                                    resolution.get("context_2", ""))
                return
            elif resolution["resolution"] == "noise":
                strength = "low"

        # Bounded contradicting update with dependency propagation
        result = self.bayesian.apply_contradiction(
            entry, new_value, strength, self.entries, self.step)

        # If value was replaced, re-embed and update retrieval index
        if result.get("replaced"):
            new_embedding = self._compute_embedding(entry.content)
            entry.embedding = new_embedding
            self.retrieval.update_embedding(entry)

        entry.last_accessed = self.step
        entry.access_count += 1

    def _split_context(self, entry: MemoryEntry, new_content: str,
                       context_1: str, context_2: str):
        """Split a memory into two context-tagged entries."""
        # Keep original with context tag
        entry.context = context_1 or "original"

        # Create new context-tagged entry
        new_embedding = self._compute_embedding(new_content)
        extracted = self.semantic.extract_fact(new_content)
        new_id = self._new_entry_id()

        new_entry = MemoryEntry(
            entry_id=new_id,
            content=new_content,
            key=f"{entry.key}.{context_2 or 'alternate'}",
            value=extracted.get("value", ""),
            entity_id=entry.entity_id,
            attribute_type=entry.attribute_type,
            embedding=new_embedding,
            context=context_2 or "alternate",
            created_at=self.step,
            last_accessed=self.step,
        )
        self.bayesian.initialize_entry(new_entry)
        self.entries[new_id] = new_entry
        self.retrieval.add_entry(new_entry)

    # ── Periodic Maintenance ──────────────────────────────────────────────

    def _periodic_maintenance(self):
        """Step 8: Decay, coreset replay, consolidation check, index maintenance."""
        # Adaptive temporal decay (batched, not per-step)
        self.bayesian.apply_decay_all(self.entries, self.step)

        # Coreset replay with diminishing returns (after decay, so replay
        # can counteract the decay that just happened)
        if self.use_coreset:
            self.coreset.replay(self.entries)

        # Consolidation check per entity
        entity_groups: dict[str, list[MemoryEntry]] = {}
        for entry in self.entries.values():
            if entry.entity_id and entry.is_active:
                if entry.entity_id not in entity_groups:
                    entity_groups[entry.entity_id] = []
                entity_groups[entry.entity_id].append(entry)

        for entity_id, entity_entries in entity_groups.items():
            if self.consolidation.should_consolidate(entity_entries):
                summaries = self.consolidation.consolidate(
                    entity_entries, entity_id, self.step)
                for summary in summaries:
                    self.entries[summary.entry_id] = summary
                    self.retrieval.add_entry(summary)

        # Prune inactive entries from indices
        self.retrieval.prune_inactive(self.entries)

    # ── Inspection ────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return full memory state for inspection and logging."""
        active = [e for e in self.entries.values() if e.is_active]
        return {
            "step": self.step,
            "n_entries": len(self.entries),
            "n_active": len(active),
            "n_uncertain": sum(1 for e in self.entries.values()
                               if e.status == "uncertain"),
            "n_decayed": sum(1 for e in self.entries.values()
                             if e.status == "decayed"),
            "n_consolidated": sum(1 for e in self.entries.values()
                                  if e.status == "consolidated"),
            "n_forgotten": sum(1 for e in self.entries.values()
                               if e.status == "forgotten"),
            "avg_confidence": (sum(e.confidence for e in active) /
                               max(1, len(active))),
            "coreset_size": self.coreset.size,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }
