"""MatAgentBench -- a verifiable benchmark and counterfactual failure-attribution
harness for LLM agents running atomistic materials workflows.

Reproducible end to end on free infrastructure: open-weight models via free API tiers,
ML interatomic potentials in place of DFT, and CC-BY / public-domain ground truth.
"""

from .schema import (
    SCHEMA_VERSION,
    FailureClass,
    InterventionKind,
    Quantity,
    Task,
    TaskFamily,
    Trajectory,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "FailureClass",
    "InterventionKind",
    "Quantity",
    "Task",
    "TaskFamily",
    "Trajectory",
    "Verdict",
    "__version__",
]
