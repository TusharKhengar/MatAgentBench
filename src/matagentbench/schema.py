"""Canonical data model for MatAgentBench.

Everything downstream is a consumer of these types: the verifier emits `Verdict`,
the runner emits `Trajectory`, the attribution harness emits `Intervention`, and the
static site is a pure renderer over `LeaderboardFile` / `TrajectoryFile`.

Locking this schema early is deliberate. The frontend is a renderer, not a
transformer -- if these shapes churn, the site gets rebuilt.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# Physical conventions
#
# These four axes are where atomistic agents silently go wrong. Every answer -- ground
# truth and agent-reported alike -- must pin all four, or a comparison is meaningless.
# --------------------------------------------------------------------------------------

Basis = Literal["per_atom", "per_formula_unit", "per_cell"]
CellChoice = Literal["primitive", "conventional", "supercell", "unspecified"]
RelaxState = Literal["relaxed", "unrelaxed", "unspecified"]


class Quantity(BaseModel):
    """A number that knows what it means."""

    value: float
    unit: str  # canonical string, e.g. "eV/atom", "GPa", "Angstrom^3"
    basis: Basis = "per_atom"
    cell: CellChoice = "primitive"
    state: RelaxState = "relaxed"

    def describe(self) -> str:
        return f"{self.value:.6g} {self.unit} [{self.basis}, {self.cell}, {self.state}]"


class Tolerance(BaseModel):
    """Pass if |reported - truth| <= abs_tol OR relative error <= rel_tol."""

    rel: float = 0.05
    abs: float = 0.0

    def accepts(self, reported: float, truth: float) -> bool:
        delta = abs(reported - truth)
        if delta <= self.abs:
            return True
        denom = abs(truth)
        if denom == 0.0:
            return delta <= self.abs
        return (delta / denom) <= self.rel


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------


class TaskFamily(str, Enum):
    """Workflow archetypes. Tier rises with the number of dependent stages."""

    FORMATION_ENERGY = "formation_energy"
    LATTICE_PARAMETER = "lattice_parameter"
    DENSITY = "density"
    RELAXATION_DELTA = "relaxation_delta"
    BULK_MODULUS = "bulk_modulus"
    EQUATION_OF_STATE = "equation_of_state"
    POLYMORPH_RANKING = "polymorph_ranking"
    VACANCY_FORMATION = "vacancy_formation"
    SURFACE_ENERGY = "surface_energy"
    ALLOY_SCREENING = "alloy_screening"


class CellContext(BaseModel):
    """Structural facts the reconciler needs to detect basis/cell confusion.

    Populated at task-generation time from the reference structure so that
    diagnosis never has to re-run a simulation.
    """

    n_atoms_primitive: int
    n_atoms_conventional: int
    n_formula_units_primitive: int
    n_formula_units_conventional: int
    volume_primitive: float | None = None  # Angstrom^3
    volume_conventional: float | None = None
    reduced_formula: str | None = None


class Provenance(BaseModel):
    source: Literal["materials_project", "jarvis_dft", "oqmd", "derived"]
    source_id: str
    retrieved_at: datetime = Field(default_factory=_utcnow)
    url: str | None = None
    license: str | None = None
    cross_checked_against: dict[str, float] | None = None


class Task(BaseModel):
    task_id: str
    tier: Literal[1, 2, 3]
    family: TaskFamily
    prompt: str
    material_ids: list[str] = Field(default_factory=list)

    answer: Quantity
    tolerance: Tolerance = Field(default_factory=Tolerance)
    cell_context: CellContext
    provenance: Provenance

    # Workflow stages a correct solution must actually execute. Used by the
    # PLAN_INVALID check and by the plan-repair intervention.
    required_stages: list[str] = Field(default_factory=list)
    max_steps: int = 20
    notes: str | None = None


class TaskSet(BaseModel):
    schema_version: str = SCHEMA_VERSION
    name: str
    generated_at: datetime = Field(default_factory=_utcnow)
    tasks: list[Task]


# --------------------------------------------------------------------------------------
# Failure taxonomy -- the core research contribution
#
# The SILENT_* classes are the point of this benchmark. They are failures that raise no
# exception, pass every syntactic check, and produce a confident, plausible number that
# is wrong by an exact, identifiable factor.
# --------------------------------------------------------------------------------------


class FailureClass(str, Enum):
    SUCCESS = "success"

    # Loud failures: something visibly broke.
    NO_ANSWER = "no_answer"
    HARD_ERROR = "hard_error"
    LOOP = "loop"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_MISUSE = "tool_misuse"
    PLAN_INVALID = "plan_invalid"

    # Silent failures: a clean run that returns the wrong number.
    SILENT_UNIT = "silent_unit"
    SILENT_BASIS = "silent_basis"
    SILENT_CELL = "silent_cell"
    SILENT_STATE = "silent_state"
    SILENT_SIGN = "silent_sign"

    WRONG_MATERIAL = "wrong_material"
    NUMERIC_WRONG = "numeric_wrong"

    # Right answer, no work done. The agent recalled a value it had memorised from
    # pretraining instead of running the workflow. Not a failure of the agent so much
    # as a contamination signal about the task -- tracked so pass rates stay honest.
    UNEARNED_PASS = "unearned_pass"

    @property
    def is_silent(self) -> bool:
        return self.value.startswith("silent_")


SILENT_CLASSES = frozenset(fc for fc in FailureClass if fc.is_silent)


class Reconciliation(BaseModel):
    """How a wrong answer can be transformed into the right one.

    If a single known convention transform reconciles the agent's number with ground
    truth, the agent did the physics correctly and the reporting incorrectly. That
    distinction is invisible to a plain numeric check.
    """

    matched: bool
    failure_class: FailureClass | None = None
    factor: float | None = None
    label: str | None = None  # e.g. "x n_atoms_primitive (4)"
    explanation: str | None = None


class Verdict(BaseModel):
    passed: bool
    failure_class: FailureClass
    reported: Quantity | None = None
    truth: Quantity
    relative_error: float | None = None
    reconciliation: Reconciliation | None = None
    stage_checks: dict[str, bool] = Field(default_factory=dict)
    detail: str | None = None


# --------------------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------------------


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    content: str
    error: str | None = None
    duration_ms: float | None = None


class Step(BaseModel):
    index: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None

    # Context accounting -- feeds the long-horizon-degradation analysis.
    context_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    raw_response: str | None = None


class ModelSpec(BaseModel):
    backend: str  # "groq" | "cerebras" | "openrouter" | "local" | "replay"
    model: str
    open_weights: bool = True
    temperature: float = 0.0
    max_context: int | None = None


class Trajectory(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    task_id: str
    model: ModelSpec
    seed: int = 0

    steps: list[Step] = Field(default_factory=list)
    final_answer: Quantity | None = None
    final_answer_raw: str | None = None
    verdict: Verdict | None = None

    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    total_tokens: int = 0
    wall_ms: float | None = None
    error: str | None = None

    # Set when this trajectory is a counterfactual replay rather than a fresh run.
    derived_from: str | None = None
    intervention: InterventionSpec | None = None


# --------------------------------------------------------------------------------------
# Counterfactual attribution
# --------------------------------------------------------------------------------------


class InterventionKind(str, Enum):
    """Each isolates one hypothesised cause by repairing it and nothing else."""

    PLAN_REPAIR = "plan_repair"  # inject the reference stage plan
    CONTEXT_RESTORE = "context_restore"  # re-inject early context that fell out
    TOOL_REPAIR = "tool_repair"  # fix one malformed call, keep everything else
    CONVENTION_CORRECT = "convention_correct"  # state the required unit/basis/cell
    NONE = "none"


class InterventionSpec(BaseModel):
    kind: InterventionKind
    step_k: int
    payload: dict[str, Any] = Field(default_factory=dict)


class InterventionOutcome(BaseModel):
    base_run_id: str
    counterfactual_run_id: str
    task_id: str
    spec: InterventionSpec
    base_failure: FailureClass
    new_failure: FailureClass
    recovered: bool
    delta_note: str | None = None


class AttributionSummary(BaseModel):
    """Aggregate answer to 'where do these agents actually break?'"""

    schema_version: str = SCHEMA_VERSION
    model: ModelSpec
    n_failures: int
    recovered_by_kind: dict[str, int] = Field(default_factory=dict)
    attempted_by_kind: dict[str, int] = Field(default_factory=dict)

    def recovery_rate(self, kind: InterventionKind) -> float:
        attempted = self.attempted_by_kind.get(kind.value, 0)
        if attempted == 0:
            return 0.0
        return self.recovered_by_kind.get(kind.value, 0) / attempted


# --------------------------------------------------------------------------------------
# Published artifacts -- exactly what the static site fetches
# --------------------------------------------------------------------------------------


class ModelResult(BaseModel):
    model: ModelSpec
    n_tasks: int
    n_passed: int
    pass_rate: float
    # Passes that were actually computed. Excludes UNEARNED_PASS, so a model that
    # recalls Materials Project values cannot inflate this number.
    earned_pass_rate: float = 0.0
    silent_failure_rate: float
    failure_counts: dict[str, int] = Field(default_factory=dict)
    pass_rate_by_tier: dict[str, float] = Field(default_factory=dict)
    mean_steps: float | None = None
    total_tokens: int = 0
    attribution: AttributionSummary | None = None


class LeaderboardFile(BaseModel):
    """results/leaderboard.json"""

    schema_version: str = SCHEMA_VERSION
    benchmark: str = "MatAgentBench"
    generated_at: datetime = Field(default_factory=_utcnow)
    taskset: str = ""
    n_tasks: int = 0
    results: list[ModelResult] = Field(default_factory=list)
    headline: dict[str, Any] = Field(default_factory=dict)


class TrajectoryIndexEntry(BaseModel):
    run_id: str
    task_id: str
    family: TaskFamily
    tier: int
    model: str
    failure_class: FailureClass
    passed: bool
    n_steps: int
    has_counterfactual: bool = False
    path: str = ""


class TrajectoryIndex(BaseModel):
    """results/trajectories/index.json -- what the viewer lists."""

    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=_utcnow)
    entries: list[TrajectoryIndexEntry] = Field(default_factory=list)


Trajectory.model_rebuild()
