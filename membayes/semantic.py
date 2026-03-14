"""
Semantic Layer
================================

LLM-based natural language understanding:

    1. Fact Extraction: text → structured (entity, attribute, value)
    2. Evidence Classification: confirm / contradict / unrelated
       + evidence_strength (high/medium/low)
       + is_correction (bool)
       + detected_dependencies (list)
    3. Conflict Disambiguation: correction vs context-dependent vs noise
    4. Query Answering: confidence-weighted retrieval
    5. Entity Extraction: extract primary entity from text
"""

from __future__ import annotations

import logging
from typing import Optional

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry

logger = logging.getLogger(__name__)


# ── LLM Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a memory management assistant. You analyze text to extract facts "
    "and classify how new information relates to existing memories. "
    "Always respond with valid JSON only. No explanation, no markdown."
)

EXTRACT_FACT_PROMPT = """Extract the key fact from this text as a JSON object.

Text: "{content}"

Return JSON with exactly these fields:
{{"key": "<entity.attribute format, e.g. zara.favorite_color>",
  "value": "<the fact value>",
  "entity": "<entity name, lowercase, e.g. zara>",
  "attribute_type": "<one of: identity, preference, episodic, relational, transient>"}}"""

EXTRACT_ENTITY_PROMPT = """Extract the primary entity (person, place, or thing) mentioned in this text.
If multiple entities are mentioned, return the most important one.
If no entity is mentioned, return an empty string.

Text: "{content}"

Return JSON: {{"entity": "<entity name, lowercase, or empty string>"}}"""

CLASSIFY_EVIDENCE_PROMPT = """Given an existing memory and new information, classify the relationship.

Existing memory:
  Key: {key}
  Value: {value}
  Content: "{content}"

New information: "{new_content}"

Classify and return JSON:
{{"classification": "<confirm|contradict|unrelated>",
  "new_value": "<extracted new value if contradict, else null>",
  "evidence_strength": "<high|medium|low>",
  "is_correction": <true if user explicitly correcting, else false>,
  "detected_dependencies": [<list of related memory keys that might be affected>]}}"""

DISAMBIGUATE_PROMPT = """A conflict was detected: the user recently confirmed a memory and now appears to contradict it.

Memory: {key} = {value}
Recent confirmation: "{confirm_text}"
New contradiction: "{contradict_text}"

Determine the nature of this conflict. Return JSON:
{{"resolution": "<correction|context_dependent|noise>",
  "context_1": "<context for original value, if context_dependent>",
  "context_2": "<context for new value, if context_dependent>",
  "explanation": "<brief explanation>"}}"""

RETRIEVE_PROMPT = """Given a question and a list of memory entries with confidence scores, identify the best match.

Question: "{question}"

Memory entries (sorted by relevance × confidence):
{entries_text}

Return JSON: {{"entry_id": "<id of best match, or null>", "answer": "<answer from that entry, or null>"}}"""


class SemanticLayer:
    """Handles all natural language understanding tasks via LLM."""

    def __init__(self, config: VCLConfig, llm_client=None):
        self.config = config
        self.llm = llm_client

    def _require_llm(self):
        if self.llm is None:
            raise RuntimeError(
                "SemanticLayer requires an LLM client. "
                "Pass llm_client to VCLMemory()."
            )

    # ── Fact Extraction ───────────────────────────────────────────────────

    def extract_fact(self, content: str) -> dict:
        """Extract structured fact from natural language.

        Returns dict with: key, value, entity, attribute_type
        """
        self._require_llm()
        prompt = EXTRACT_FACT_PROMPT.format(content=content)
        result = self.llm.chat_json(prompt, system_message=SYSTEM_PROMPT)
        return {
            "key": result.get("key", "unknown"),
            "value": result.get("value", content[:50]),
            "entity": result.get("entity", ""),
            "attribute_type": result.get("attribute_type", "preference"),
        }

    # ── Entity Extraction ─────────────────────────────────────────────────

    def extract_entity(self, content: str) -> str:
        """Extract primary entity name from text via LLM.

        Returns entity name (lowercase) or empty string.
        """
        self._require_llm()
        prompt = EXTRACT_ENTITY_PROMPT.format(content=content)
        result = self.llm.chat_json(prompt, system_message=SYSTEM_PROMPT)
        return result.get("entity", "").strip().lower()

    # ── Evidence Classification ───────────────────────────────────────────

    def classify_evidence(self, entry: MemoryEntry,
                          new_content: str) -> dict:
        """Classify how new content relates to existing memory.

        Returns dict with: classification, new_value, evidence_strength,
                          is_correction, detected_dependencies
        """
        self._require_llm()
        prompt = CLASSIFY_EVIDENCE_PROMPT.format(
            key=entry.key, value=entry.value,
            content=entry.content, new_content=new_content,
        )
        result = self.llm.chat_json(prompt, system_message=SYSTEM_PROMPT)
        return {
            "classification": result.get("classification", "unrelated"),
            "new_value": result.get("new_value"),
            "evidence_strength": result.get("evidence_strength", "medium"),
            "is_correction": result.get("is_correction", False),
            "detected_dependencies": result.get("detected_dependencies", []),
        }

    # ── Conflict Disambiguation ───────────────────────────────────────────

    def disambiguate_conflict(self, entry: MemoryEntry,
                              confirm_text: str,
                              contradict_text: str) -> dict:
        """Disambiguate rapid confirm→contradict conflict.

        Returns dict with: resolution (correction/context_dependent/noise),
                          context_1, context_2
        """
        self._require_llm()
        prompt = DISAMBIGUATE_PROMPT.format(
            key=entry.key, value=entry.value,
            confirm_text=confirm_text, contradict_text=contradict_text,
        )
        result = self.llm.chat_json(prompt, system_message=SYSTEM_PROMPT)
        return {
            "resolution": result.get("resolution", "correction"),
            "context_1": result.get("context_1", ""),
            "context_2": result.get("context_2", ""),
        }

    # ── Query Answering ───────────────────────────────────────────────────

    def answer_query(self, question: str,
                     candidates: list[MemoryEntry]) -> dict:
        """Find best matching memory entry for a question.

        Returns dict with: entry_id, answer
        """
        self._require_llm()
        if not candidates:
            return {"entry_id": None, "answer": None}

        entries_text = "\n".join(
            f"  [{e.entry_id}] {e.key}: {e.value} (confidence: {e.confidence:.2f})"
            for e in candidates
        )
        prompt = RETRIEVE_PROMPT.format(
            question=question, entries_text=entries_text,
        )
        result = self.llm.chat_json(prompt, system_message=SYSTEM_PROMPT)
        return {
            "entry_id": result.get("entry_id"),
            "answer": result.get("answer"),
        }
