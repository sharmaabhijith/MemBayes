from __future__ import annotations

"""
Baselines for MemBayes Benchmark
==================================

All baselines share the same LLM client for semantic understanding.
The only difference is the memory management strategy.

Systems:
    1. NaiveRAGMemory     — store everything, confidence=1.0, last-write-wins
    2. SlidingWindowMemory — keep last N interactions, LLM answers from context
    3. DecayOnlyMemory     — temporal decay but no Bayesian accumulation

VCL ablations (VCL no-coreset, VCL no-decay) are handled by the experiment
runner via VCLMemory constructor flags / config overrides.
"""

import math
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import replace
from typing import Optional

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry
from membayes.retrieval import RetrievalLayer
from membayes.semantic import SemanticLayer

logger = logging.getLogger(__name__)


# =============================================================================
# Base interface
# =============================================================================

class BaseMemory(ABC):
    @abstractmethod
    def process_interaction(self, text: str, step: int): ...

    @abstractmethod
    def answer_query(self, question: str) -> dict: ...

    @abstractmethod
    def forget(self, text: str, step: int): ...

    @abstractmethod
    def get_state(self) -> dict: ...


# =============================================================================
# 1. Naive RAG Memory
# =============================================================================

class NaiveRAGMemory(BaseMemory):
    """Store all facts, no confidence tracking, last-write-wins on contradiction.

    Represents what most RAG systems do:
        - Every stored fact has confidence = 1.0
        - Contradictions immediately overwrite the value
        - No decay, no coreset, no dependency propagation
    """

    def __init__(self, config: VCLConfig, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.semantic = SemanticLayer(config, llm_client)
        self.retrieval = RetrievalLayer(config)
        self.entries: dict[str, MemoryEntry] = {}
        self.step = 0
        self._next_id = 0

    def _new_id(self) -> str:
        self._next_id += 1
        return f"N{self._next_id:06d}"

    def process_interaction(self, text: str, step: int):
        self.step = step
        entity_id = self.semantic.extract_entity(text)
        embedding = self.llm_client.embed_single(text)

        candidate_ids = self.retrieval.get_candidates(
            embedding, entity_id, self.entries
        )

        if not candidate_ids:
            if not entity_id:
                return  # No entity and no candidates — skip filler
            self._create_entry(text, embedding)
            return

        found_match = False
        for cid in candidate_ids:
            entry = self.entries[cid]
            classification = self.semantic.classify_evidence(entry, text)
            action = classification["classification"]

            if action == "confirm":
                entry.last_accessed = step
                entry.access_count += 1
                found_match = True

            elif action == "contradict":
                # Last-write-wins: immediately replace value
                new_value = classification.get("new_value")
                if new_value:
                    entry.value = new_value
                    entry.content = text
                    new_emb = self.llm_client.embed_single(text)
                    entry.embedding = new_emb
                    self.retrieval.update_embedding(entry)
                entry.last_accessed = step
                entry.confidence = 1.0
                found_match = True

        if not found_match:
            if not entity_id:
                return
            self._create_entry(text, embedding)

    def _create_entry(self, text: str, embedding: list[float]):
        extracted = self.semantic.extract_fact(text)
        eid = self._new_id()
        entry = MemoryEntry(
            entry_id=eid,
            content=text,
            key=extracted.get("key", eid),
            value=extracted.get("value", text[:50]),
            entity_id=extracted.get("entity", ""),
            attribute_type=extracted.get("attribute_type", "preference"),
            embedding=embedding,
            log_odds=10.0,  # fixed high confidence
            confidence=1.0,
            created_at=self.step,
            last_accessed=self.step,
        )
        self.entries[eid] = entry
        self.retrieval.add_entry(entry)

    def forget(self, text: str, step: int):
        self.step = step
        embedding = self.llm_client.embed_single(text)
        entity_id = self.semantic.extract_entity(text)
        candidate_ids = self.retrieval.get_candidates(
            embedding, entity_id, self.entries
        )
        for cid in candidate_ids:
            entry = self.entries[cid]
            classification = self.semantic.classify_evidence(entry, text)
            if classification["classification"] in ("confirm", "contradict"):
                entry.status = "forgotten"
                entry.confidence = 0.0
                self.retrieval.remove_entry(entry)

    def answer_query(self, question: str) -> dict:
        embedding = self.llm_client.embed_single(question)
        entity_id = self.semantic.extract_entity(question)
        candidate_ids = self.retrieval.get_candidates(
            embedding, entity_id, self.entries
        )

        candidates = [
            self.entries[cid] for cid in candidate_ids
            if cid in self.entries and self.entries[cid].status != "forgotten"
        ]
        if not candidates:
            return {"answer": "I don't have that information", "confidence": 0.0}

        result = self.semantic.answer_query(question, candidates)
        matched_id = result.get("entry_id")
        if matched_id and matched_id in self.entries:
            entry = self.entries[matched_id]
            return {
                "answer": result.get("answer", entry.value),
                "confidence": 1.0,
                "entry_id": matched_id,
            }

        best = max(candidates, key=lambda e: e.confidence)
        return {"answer": best.value, "confidence": 1.0, "entry_id": best.entry_id}

    def get_state(self) -> dict:
        active = [e for e in self.entries.values() if e.status != "forgotten"]
        return {
            "step": self.step,
            "n_entries": len(self.entries),
            "n_active": len(active),
            "avg_confidence": 1.0 if active else 0.0,
        }


# =============================================================================
# 2. Sliding Window Memory
# =============================================================================

WINDOW_ANSWER_PROMPT = """Given the following recent interactions, answer the question.
If the information is not available in the interactions, say "I don't have that information".

Interactions:
{interactions}

Question: "{question}"

Return JSON: {{"answer": "<your answer>", "confidence": <0.0 to 1.0>}}"""


class SlidingWindowMemory(BaseMemory):
    """Keep only the last N interactions in a buffer. Answer queries from context.

    Simulates a context-window-based approach where recent messages are
    stuffed into the LLM context for answering questions.
    """

    def __init__(self, config: VCLConfig, llm_client, window_size: int = 30):
        self.config = config
        self.llm_client = llm_client
        self.window: deque[tuple[str, int]] = deque(maxlen=window_size)
        self.forgotten: set[str] = set()
        self.step = 0

    def process_interaction(self, text: str, step: int):
        self.step = step
        self.window.append((text, step))

    def forget(self, text: str, step: int):
        self.step = step
        # Remove matching interactions from window
        self.forgotten.add(text.lower().strip())
        self.window = deque(
            [(t, s) for t, s in self.window if t.lower().strip() not in self.forgotten],
            maxlen=self.window.maxlen,
        )

    def answer_query(self, question: str) -> dict:
        if not self.window:
            return {"answer": "I don't have that information", "confidence": 0.0}

        interactions_text = "\n".join(
            f"{i+1}. \"{text}\"" for i, (text, _) in enumerate(self.window)
        )
        prompt = WINDOW_ANSWER_PROMPT.format(
            interactions=interactions_text, question=question,
        )
        system_msg = (
            "You are a memory assistant. Answer questions based only on the "
            "provided interactions. Respond with valid JSON only."
        )
        result = self.llm_client.chat_json(prompt, system_message=system_msg)

        answer = result.get("answer", "I don't have that information")
        confidence = min(max(float(result.get("confidence", 0.5)), 0.0), 1.0)

        return {"answer": answer, "confidence": round(confidence, 4)}

    def get_state(self) -> dict:
        return {
            "step": self.step,
            "window_size": len(self.window),
            "max_window": self.window.maxlen,
        }


# =============================================================================
# 3. Decay Only Memory
# =============================================================================

class DecayOnlyMemory(BaseMemory):
    """Temporal decay but no Bayesian evidence accumulation.

    - New facts get category-specific initial confidence (same as VCL)
    - Confirmations do NOT increase log-odds (just refresh timestamp)
    - Contradictions immediately replace value (like Naive) and reset confidence
    - Temporal decay applied per step (same formula as VCL)
    - No coreset, no dependency propagation, no KL capping
    """

    def __init__(self, config: VCLConfig, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.semantic = SemanticLayer(config, llm_client)
        self.retrieval = RetrievalLayer(config)
        self.entries: dict[str, MemoryEntry] = {}
        self.step = 0
        self._next_id = 0

    def _new_id(self) -> str:
        self._next_id += 1
        return f"D{self._next_id:06d}"

    def _apply_decay_all(self):
        """Apply temporal decay to all active entries."""
        for entry in self.entries.values():
            if entry.status not in ("active", "uncertain"):
                continue
            dt = self.step - entry.last_accessed
            if dt <= 0:
                continue
            lambda_base = self.config.get_decay_rate(entry.attribute_type)
            n_access = min(entry.access_count, self.config.access_modulation_cap)
            modifier = math.exp(-self.config.access_modulation_strength * n_access)
            lambda_eff = lambda_base * modifier
            decay_factor = math.exp(-lambda_eff * dt)
            entry.log_odds *= decay_factor
            entry.confidence = 1.0 / (1.0 + math.exp(-entry.log_odds))
            if entry.confidence < self.config.hard_decay_threshold:
                entry.status = "decayed"
            elif entry.confidence < self.config.soft_decay_threshold:
                entry.status = "uncertain"

    def process_interaction(self, text: str, step: int):
        self.step = step
        self._apply_decay_all()

        entity_id = self.semantic.extract_entity(text)
        embedding = self.llm_client.embed_single(text)

        candidate_ids = self.retrieval.get_candidates(
            embedding, entity_id, self.entries
        )

        if not candidate_ids:
            if not entity_id:
                return  # No entity and no candidates — skip filler
            self._create_entry(text, embedding)
            return

        found_match = False
        for cid in candidate_ids:
            entry = self.entries[cid]
            classification = self.semantic.classify_evidence(entry, text)
            action = classification["classification"]

            if action == "confirm":
                # Only refresh timestamp — no log-odds increase
                entry.last_accessed = step
                entry.access_count += 1
                found_match = True

            elif action == "contradict":
                # Immediately replace value, reset confidence
                new_value = classification.get("new_value")
                if new_value:
                    entry.value = new_value
                    entry.content = text
                    new_emb = self.llm_client.embed_single(text)
                    entry.embedding = new_emb
                    self.retrieval.update_embedding(entry)
                cat_log_odds = self.config.get_initial_log_odds(entry.attribute_type)
                entry.log_odds = cat_log_odds
                entry.confidence = 1.0 / (1.0 + math.exp(-entry.log_odds))
                entry.last_accessed = step
                found_match = True

        if not found_match:
            if not entity_id:
                return
            self._create_entry(text, embedding)

    def _create_entry(self, text: str, embedding: list[float]):
        extracted = self.semantic.extract_fact(text)
        eid = self._new_id()
        attr_type = extracted.get("attribute_type", "preference")
        cat_log_odds = self.config.get_initial_log_odds(attr_type)
        entry = MemoryEntry(
            entry_id=eid,
            content=text,
            key=extracted.get("key", eid),
            value=extracted.get("value", text[:50]),
            entity_id=extracted.get("entity", ""),
            attribute_type=attr_type,
            embedding=embedding,
            log_odds=cat_log_odds,
            created_at=self.step,
            last_accessed=self.step,
        )
        entry.update_confidence()
        self.entries[eid] = entry
        self.retrieval.add_entry(entry)

    def forget(self, text: str, step: int):
        self.step = step
        embedding = self.llm_client.embed_single(text)
        entity_id = self.semantic.extract_entity(text)
        candidate_ids = self.retrieval.get_candidates(
            embedding, entity_id, self.entries
        )
        for cid in candidate_ids:
            entry = self.entries[cid]
            classification = self.semantic.classify_evidence(entry, text)
            if classification["classification"] in ("confirm", "contradict"):
                entry.log_odds = -10.0
                entry.confidence = 0.0
                entry.status = "forgotten"
                self.retrieval.remove_entry(entry)

    def answer_query(self, question: str) -> dict:
        embedding = self.llm_client.embed_single(question)
        entity_id = self.semantic.extract_entity(question)
        candidate_ids = self.retrieval.get_candidates(
            embedding, entity_id, self.entries
        )

        candidates = [
            self.entries[cid] for cid in candidate_ids
            if cid in self.entries and self.entries[cid].status not in ("decayed", "forgotten")
        ]
        if not candidates:
            return {"answer": "I don't have that information", "confidence": 0.0}

        result = self.semantic.answer_query(question, candidates)
        matched_id = result.get("entry_id")
        if matched_id and matched_id in self.entries:
            entry = self.entries[matched_id]
            return {
                "answer": result.get("answer", entry.value),
                "confidence": round(entry.confidence, 4),
                "entry_id": matched_id,
            }

        best = max(candidates, key=lambda e: e.confidence)
        return {
            "answer": best.value,
            "confidence": round(best.confidence, 4),
            "entry_id": best.entry_id,
        }

    def get_state(self) -> dict:
        active = [e for e in self.entries.values()
                  if e.status in ("active", "uncertain")]
        return {
            "step": self.step,
            "n_entries": len(self.entries),
            "n_active": len(active),
            "avg_confidence": (sum(e.confidence for e in active) /
                               max(1, len(active))),
        }
