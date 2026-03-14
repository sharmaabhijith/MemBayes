"""
MemBayes: Scalable Symbolic Variational Continual Learning for LLM Memory Management
=====================================================================================

Adapts Variational Continual Learning (Nguyen et al., ICLR 2018) from maintaining
distributions over neural network weights to maintaining distributions over memory entries.

Three-layer architecture:
    1. Retrieval Layer  — HNSW + Entity Index + GMM clustering (O(log K) candidate retrieval)
    2. Semantic Layer   — LLM-based fact extraction, classification, disambiguation
    3. Bayesian Layer   — Deterministic log-odds updates, adaptive decay, dependency propagation

Key features:
    - O(log K + N) per interaction instead of O(K)
    - Per-category adaptive decay rates
    - Dependency-aware belief propagation
    - Importance-weighted coreset management
    - Bounded updates (KL regularization analog)
    - Memory consolidation via variational compression
    - Conflict resolution protocol
"""

from membayes.config import VCLConfig
from membayes.memory_entry import MemoryEntry, AttributeCategory, EntryStatus
from membayes.retrieval import RetrievalLayer
from membayes.bayesian import BayesianLayer
from membayes.semantic import SemanticLayer
from membayes.coreset import ImportanceWeightedCoreset
from membayes.consolidation import ConsolidationEngine
from membayes.dependencies import DependencyGraph
from membayes.vcl_memory import VCLMemory
from membayes.llm_client import LLMClient

__version__ = "2.0.0"

__all__ = [
    "VCLConfig",
    "VCLMemory",
    "LLMClient",
    "MemoryEntry",
    "AttributeCategory",
    "EntryStatus",
    "RetrievalLayer",
    "BayesianLayer",
    "SemanticLayer",
    "ImportanceWeightedCoreset",
    "ConsolidationEngine",
    "DependencyGraph",
]
