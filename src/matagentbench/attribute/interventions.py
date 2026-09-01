"""Counterfactual failure attribution.

A failure taxonomy alone is descriptive: it says *what* the run looked like when it
broke. Attribution is causal: it says what would have had to be different for the run
to succeed.

The method is a replay-and-intervene design. For a failed trajectory we re-run it with
the recorded assistant messages replayed verbatim up to step k, then change exactly one
thing at step k and let the agent continue live. If success rate rises, that one thing
was load-bearing. Because every counterfactual shares a prefix with its base run,
the difference in outcome is attributable to the intervention and not to sampling.

Replaying the prefix is also what makes this affordable: only the steps after the
intervention consume API budget.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.backends import LLMBackend, ReplayBackend
from ..agent.runner import (
    observations_before,
    run_trajectory,
    save_trajectory,
    slugify,
    transcript_of,
)
from ..schema import (
    AttributionSummary,
    FailureClass,
    InterventionKind,
    InterventionOutcome,
    InterventionSpec,
    ModelSpec,
    Task,
    Trajectory,
)
from ..verify import verify

# Which repairs are worth trying for which failure. Running every intervention against
# every failure would quadruple cost for results we can predict are null.
CANDIDATES: dict[FailureClass, tuple[InterventionKind, ...]] = {
    FailureClass.SILENT_UNIT: (InterventionKind.CONVENTION_CORRECT, InterventionKind.PLAN_REPAIR),
    FailureClass.SILENT_BASIS: (InterventionKind.CONVENTION_CORRECT, InterventionKind.PLAN_REPAIR),
    FailureClass.SILENT_CELL: (InterventionKind.CONVENTION_CORRECT, InterventionKind.PLAN_REPAIR),
    FailureClass.SILENT_SIGN: (InterventionKind.CONVENTION_CORRECT,),
    FailureClass.SILENT_STATE: (InterventionKind.PLAN_REPAIR, InterventionKind.CONVENTION_CORRECT),
    FailureClass.PLAN_INVALID: (InterventionKind.PLAN_REPAIR, InterventionKind.CONTEXT_RESTORE),
    FailureClass.NUMERIC_WRONG: (InterventionKind.PLAN_REPAIR, InterventionKind.CONTEXT_RESTORE),
    FailureClass.WRONG_MATERIAL: (InterventionKind.CONTEXT_RESTORE, InterventionKind.PLAN_REPAIR),
    FailureClass.BUDGET_EXHAUSTED: (InterventionKind.PLAN_REPAIR, InterventionKind.CONTEXT_RESTORE),
    FailureClass.LOOP: (InterventionKind.PLAN_REPAIR, InterventionKind.CONTEXT_RESTORE),
    FailureClass.TOOL_MISUSE: (InterventionKind.TOOL_REPAIR, InterventionKind.PLAN_REPAIR),
    FailureClass.NO_ANSWER: (InterventionKind.PLAN_REPAIR,),
    FailureClass.HARD_ERROR: (InterventionKind.TOOL_REPAIR,),
}


@dataclass
class InterventionPlan:
    spec: InterventionSpec
    rationale: str


def _first_failed_call_index(traj: Trajectory) -> int | None:
    for step in traj.steps:
        if step.tool_call is not None and step.tool_result is not None and not step.tool_result.ok:
            return step.index
    return None


def _final_step_index(traj: Trajectory) -> int:
    return max(len(traj.steps) - 1, 0)


def plan_interventions(task: Task, traj: Trajectory) -> list[InterventionPlan]:
    """Choose where and how to intervene, given how this run actually failed.

    Step choice is the whole design. Injecting a plan at step 0 tests whether the agent
    could ever have got there; injecting a convention reminder at the last step tests
    only the reporting, holding all of the physics fixed.
    """
    if traj.verdict is None or traj.verdict.passed:
        return []

    failure = traj.verdict.failure_class
    plans: list[InterventionPlan] = []

    for kind in CANDIDATES.get(failure, ()):
        if kind is InterventionKind.PLAN_REPAIR:
            plans.append(
                InterventionPlan(
                    InterventionSpec(kind=kind, step_k=0),
                    "Inject the reference stage sequence before the agent commits to an "
                    "approach. Recovery implicates planning rather than execution.",
                )
            )

        elif kind is InterventionKind.CONVENTION_CORRECT:
            # Deliberately as late as possible: the computation is already done, so a
            # recovery isolates reporting from physics.
            step_k = _final_step_index(traj)
            plans.append(
                InterventionPlan(
                    InterventionSpec(kind=kind, step_k=step_k),
                    "Restate the required unit/basis/cell at the final step, after all "
                    "computation. Recovery proves the physics was right and only the "
                    "bookkeeping was wrong.",
                )
            )

        elif kind is InterventionKind.CONTEXT_RESTORE:
            step_k = max(len(traj.steps) // 2, 1)
            restored = observations_before(traj, step_k)
            if not restored:
                continue
            plans.append(
                InterventionPlan(
                    InterventionSpec(
                        kind=kind, step_k=step_k, payload={"restored_context": restored}
                    ),
                    "Re-inject early observations at the midpoint, where context "
                    "truncation begins to bite. Recovery implicates forgetting.",
                )
            )

        elif kind is InterventionKind.TOOL_REPAIR:
            step_k = _first_failed_call_index(traj)
            if step_k is None:
                continue
            failed_step = next((s for s in traj.steps if s.index == step_k), None)
            if failed_step is None or failed_step.tool_call is None:
                continue
            repaired = _repair_args(task, traj, failed_step.tool_call.args)
            if not repaired:
                continue
            plans.append(
                InterventionPlan(
                    InterventionSpec(kind=kind, step_k=step_k, payload={"args": repaired}),
                    "Fix the first malformed tool call and change nothing else. "
                    "Recovery implicates tool-use mechanics, not reasoning.",
                )
            )

    return plans


def _repair_args(task: Task, traj: Trajectory, args: dict[str, Any]) -> dict[str, Any]:
    """Minimal, mechanical repairs -- never a re-derivation of the agent's intent.

    Only fixes that are unambiguous from the task definition are applied, so that a
    TOOL_REPAIR recovery cannot be an artefact of us solving the task for the agent.
    """
    repaired: dict[str, Any] = {}
    if "material_id" in args and task.material_ids:
        if str(args["material_id"]).strip() not in set(task.material_ids):
            repaired["material_id"] = task.material_ids[0]
    if "handle" in args and isinstance(args["handle"], str):
        # Point a dangling handle at the most recent structure the run actually produced.
        produced = [
            s.tool_result.content
            for s in traj.steps
            if s.tool_result is not None
            and s.tool_result.ok
            and "handle" in (s.tool_result.content or "")
        ]
        if produced and args["handle"] not in "".join(produced):
            repaired["handle"] = "s0"
    return repaired


def run_counterfactual(
    task: Task,
    base_traj: Trajectory,
    plan: InterventionPlan,
    live_backend: LLMBackend,
    results_dir: Path,
    *,
    mp_source: Any | None = None,
    force: bool = False,
) -> InterventionOutcome:
    """Replay the prefix, apply one intervention, run the rest live, verify."""
    folder = (
        results_dir
        / "counterfactuals"
        / slugify(f"{base_traj.model.backend}-{base_traj.model.model}")
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{task.task_id}__{plan.spec.kind.value}__k{plan.spec.step_k}.json"

    if path.exists() and not force:
        try:
            cf = Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
            return _outcome(task, base_traj, cf, plan)
        except (OSError, ValueError):
            pass

    replay = ReplayBackend(
        transcript=transcript_of(base_traj),
        fallback=live_backend,
        diverge_at=plan.spec.step_k,
    )
    cf = run_trajectory(
        task,
        replay,
        mp_source=mp_source,
        seed=base_traj.seed,
        intervention=plan.spec,
        derived_from=base_traj.run_id,
    )
    cf.run_id = uuid.uuid4().hex[:12]
    cf.model = ModelSpec(
        backend=base_traj.model.backend,
        model=base_traj.model.model,
        open_weights=base_traj.model.open_weights,
        temperature=base_traj.model.temperature,
    )
    cf.verdict = verify(task, cf)
    save_trajectory(cf, path)
    return _outcome(task, base_traj, cf, plan)


def _outcome(
    task: Task, base: Trajectory, cf: Trajectory, plan: InterventionPlan
) -> InterventionOutcome:
    if cf.verdict is None:
        cf.verdict = verify(task, cf)
    base_failure = base.verdict.failure_class if base.verdict else FailureClass.NO_ANSWER
    return InterventionOutcome(
        base_run_id=base.run_id,
        counterfactual_run_id=cf.run_id,
        task_id=task.task_id,
        spec=plan.spec,
        base_failure=base_failure,
        new_failure=cf.verdict.failure_class,
        recovered=bool(
            cf.verdict.passed and cf.verdict.failure_class is not FailureClass.UNEARNED_PASS
        ),
        delta_note=plan.rationale,
    )


def summarise(
    model: ModelSpec, outcomes: list[InterventionOutcome], n_failures: int
) -> AttributionSummary:
    attempted: dict[str, int] = {}
    recovered: dict[str, int] = {}
    for outcome in outcomes:
        key = outcome.spec.kind.value
        attempted[key] = attempted.get(key, 0) + 1
        if outcome.recovered:
            recovered[key] = recovered.get(key, 0) + 1
    return AttributionSummary(
        model=model,
        n_failures=n_failures,
        attempted_by_kind=attempted,
        recovered_by_kind=recovered,
    )
