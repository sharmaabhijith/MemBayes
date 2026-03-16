"""
MemBayes End-to-End Evaluation
==============================

Runs real conversations through the full VCLMemory pipeline (with LLM calls)
and compares against baseline memory systems.

Hypotheses (H1-H14):
    H1:  Retention          H8:  Retrieval (paraphrase)
    H2:  Reinforcement      H9:  Distractors
    H3:  Contradiction      H10: Belief revision (A->B->C)
    H4:  Forgetting curve   H11: Rapid conflict
    H5:  Calibration (ECE)  H12: Category-specific decay
    H6:  Coreset stress     H13: Dependency propagation
    H7:  Selective forget   H14: Evidence strength

Modules:
    generator  — synthetic benchmark generation (H1-H14)
    baselines  — NaiveRAG, SlidingWindow, DecayOnly
    evaluator  — response scoring, ECE, consistency checks
    runner     — experiment orchestrator + CLI
    plots      — comparison visualizations
"""
