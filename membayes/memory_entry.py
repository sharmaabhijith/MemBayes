"""
Memory Entry
============

Extended memory entry with 13 fields: entity tracking, dependency sets,
embeddings, cluster assignments, per-entry decay rates, and access counters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class AttributeCategory(str, Enum):
    """Attribute categories with semantic decay properties."""
    IDENTITY = "identity"       # name, birthdate, nationality
    PREFERENCE = "preference"   # favorite color, food, hobby
    EPISODIC = "episodic"       # time-bound events
    RELATIONAL = "relational"   # works with, sister is
    TRANSIENT = "transient"     # current mood, today's plan


class EntryStatus(str, Enum):
    """Memory entry lifecycle states."""
    ACTIVE = "active"
    UNCERTAIN = "uncertain"       # soft decay threshold crossed
    DECAYED = "decayed"           # hard decay threshold crossed
    CONSOLIDATED = "consolidated" # merged into summary (kept for audit)
    FORGOTTEN = "forgotten"       # explicitly deleted


@dataclass
class MemoryEntry:
    """Extended memory entry with Bayesian confidence tracking.

    Fields:
        - entity_id:        entity this memory belongs to
        - attribute_type:   category determining decay rate and initial prior
        - dependencies:     set of parent memory IDs this entry depends on
        - embedding:        vector representation for retrieval
        - cluster_id:       GMM cluster assignment
        - created_at:       creation timestamp
        - last_accessed:    last access timestamp
        - access_count:     number of times accessed
        - n_replayed:       number of coreset replays (for diminishing returns)
        - context:          optional context tag (for context-dependent splits)
        - last_update_type: type of last update (for conflict detection)
    """
    entry_id: str
    content: str               # natural language description
    key: str                   # structured key (entity.attribute)
    value: str                 # current believed value

    # Bayesian state
    log_odds: float = 1.2
    confidence: float = 0.77
    evidence_for: int = 1
    evidence_against: int = 0

    # Entity and category
    entity_id: str = ""
    attribute_type: str = "preference"

    # Dependencies (parent memory IDs)
    dependencies: list[str] = field(default_factory=list)

    # Embedding (None when not using embeddings)
    embedding: Optional[list[float]] = None

    # Cluster assignment
    cluster_id: int = -1

    # Temporal
    created_at: int = 0
    last_accessed: int = 0
    access_count: int = 0

    # Status
    status: str = "active"

    # Coreset tracking
    n_replayed: int = 0

    # Context tag (for context-dependent splits)
    context: str = ""

    # Conflict detection
    last_update_type: str = ""
    last_update_step: int = 0

    def update_confidence(self):
        """Convert log-odds to probability: σ(ℓ) = 1/(1+exp(-ℓ))."""
        self.confidence = 1.0 / (1.0 + math.exp(-self.log_odds))

    def to_dict(self) -> dict:
        """Serialize to dict, excluding embedding for compactness."""
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @property
    def is_active(self) -> bool:
        return self.status in ("active", "uncertain")

    @property
    def is_queryable(self) -> bool:
        return self.status not in ("decayed", "consolidated", "forgotten")
