"""Turning a trajectory into a Verdict.

Classification order matters and is intentional:

  1. Did the agent produce an answer at all? (loud failures)
  2. Is it numerically right? -> then did it actually do the work? (contamination)
  3. Did it fetch the right material?
  4. Can a single convention transform explain the miss? (the silent classes)
  5. Did it skip a required workflow stage?
  6. Otherwise the physics is simply wrong.

Steps 3-5 are ordered so that a specific, provable diagnosis always beats a vague one.
"""

from __future__ import annotations

from collections import Counter

from ..schema import (
    FailureClass,
    Quantity,
    Task,
    Trajectory,
    Verdict,
)
from ..units import canonical_unit, reconcile, relative_error
from .parser import parse_final_answer

# Which workflow stage each tool advances. The verifier uses this to decide whether the
# agent executed the pipeline the task requires, or merely asserted an answer.
TOOL_STAGES: dict[str, str] = {
    "mp_get_structure": "fetch_structure",
    "jarvis_get_structure": "fetch_structure",
    "build_structure": "fetch_structure",
    "make_primitive": "cell_setup",
    "make_conventional": "cell_setup",
    "make_supercell": "cell_setup",
    "relax_structure": "relax",
    "compute_energy": "compute_energy",
    "compute_elastic": "compute_elastic",
    "equation_of_state": "eos",
    "make_vacancy": "defect",
    "make_slab": "surface",
    "lookup_reference_energies": "reference_energies",
    "python_eval": "analysis",
}

LOOP_THRESHOLD = 3  # identical consecutive (tool, args) calls
TOOL_FAILURE_RATIO = 0.5


def _executed_stages(traj: Trajectory) -> set[str]:
    stages: set[str] = set()
    for step in traj.steps:
        if step.tool_call is None:
            continue
        if step.tool_result is not None and not step.tool_result.ok:
            continue  # a call that errored did not advance the workflow
        stage = TOOL_STAGES.get(step.tool_call.name)
        if stage:
            stages.add(stage)
    return stages


def _detect_loop(traj: Trajectory) -> bool:
    signatures = [
        (s.tool_call.name, repr(sorted(s.tool_call.args.items())))
        for s in traj.steps
        if s.tool_call is not None
    ]
    run = 1
    for prev, cur in zip(signatures, signatures[1:], strict=False):
        run = run + 1 if cur == prev else 1
        if run >= LOOP_THRESHOLD:
            return True
    if signatures:
        most_common = Counter(signatures).most_common(1)[0][1]
        if most_common >= LOOP_THRESHOLD + 1:
            return True
    return False


def _tool_failure_ratio(traj: Trajectory) -> float:
    calls = [s for s in traj.steps if s.tool_call is not None]
    if not calls:
        return 0.0
    failed = sum(1 for s in calls if s.tool_result is not None and not s.tool_result.ok)
    return failed / len(calls)


def _fetched_material_ids(traj: Trajectory) -> set[str]:
    found: set[str] = set()
    for step in traj.steps:
        if step.tool_call is None:
            continue
        if TOOL_STAGES.get(step.tool_call.name) != "fetch_structure":
            continue
        for key in ("material_id", "mp_id", "jid", "identifier"):
            val = step.tool_call.args.get(key)
            if isinstance(val, str):
                found.add(val.strip())
    return found


def _declared_convention_matches(reported: Quantity, truth: Quantity) -> bool:
    if canonical_unit(reported.unit) and canonical_unit(reported.unit) != canonical_unit(
        truth.unit
    ):
        return False
    if reported.basis != truth.basis:
        return False
    if reported.cell not in ("unspecified", truth.cell):
        return False
    return reported.state in ("unspecified", truth.state)


def verify(task: Task, traj: Trajectory) -> Verdict:
    truth = task.answer
    executed = _executed_stages(traj)
    required = set(task.required_stages)
    missing = sorted(required - executed)
    stage_checks = {stage: (stage in executed) for stage in sorted(required)}

    last_message = traj.final_answer_raw
    if last_message is None:
        for step in reversed(traj.steps):
            if step.raw_response:
                last_message = step.raw_response
                break

    parsed = parse_final_answer(last_message)
    stage_checks["answer_parse_strict"] = parsed.method == "json"

    # --- 1. Loud failures -------------------------------------------------------------
    if parsed.quantity is None:
        if traj.error:
            failure = FailureClass.HARD_ERROR
            detail = f"Run aborted: {traj.error}"
        elif _detect_loop(traj):
            failure = FailureClass.LOOP
            detail = "Agent repeated an identical tool call without making progress."
        elif len(traj.steps) >= task.max_steps:
            failure = FailureClass.BUDGET_EXHAUSTED
            detail = f"Hit the {task.max_steps}-step budget without producing an answer."
        elif _tool_failure_ratio(traj) > TOOL_FAILURE_RATIO:
            failure = FailureClass.TOOL_MISUSE
            detail = "Majority of tool calls errored; agent never recovered."
        else:
            failure = FailureClass.NO_ANSWER
            detail = "No parseable final answer in the agent's last message."
        return Verdict(
            passed=False,
            failure_class=failure,
            reported=None,
            truth=truth,
            stage_checks=stage_checks,
            detail=detail,
        )

    reported = parsed.quantity
    rel_err = relative_error(reported.value, truth.value)
    numerically_correct = task.tolerance.accepts(reported.value, truth.value)

    # --- 2. Correct number ------------------------------------------------------------
    if numerically_correct:
        if missing:
            # The value is right but the workflow never ran. Almost always recall from
            # pretraining rather than computation. Counted separately so headline pass
            # rates cannot be inflated by memorised reference data.
            return Verdict(
                passed=True,
                failure_class=FailureClass.UNEARNED_PASS,
                reported=reported,
                truth=truth,
                relative_error=rel_err,
                stage_checks=stage_checks,
                detail=(
                    "Correct value without executing required stage(s): "
                    f"{', '.join(missing)}. Likely recalled, not computed."
                ),
            )
        detail = None
        if not _declared_convention_matches(reported, truth):
            detail = (
                "Value correct but the reported convention label disagrees with ground "
                f"truth ({reported.describe()} vs {truth.describe()})."
            )
            stage_checks["declared_convention_matches"] = False
        return Verdict(
            passed=True,
            failure_class=FailureClass.SUCCESS,
            reported=reported,
            truth=truth,
            relative_error=rel_err,
            stage_checks=stage_checks,
            detail=detail,
        )

    # --- 3. Wrong material ------------------------------------------------------------
    fetched = _fetched_material_ids(traj)
    expected = {m.strip() for m in task.material_ids}
    if expected and fetched and not (fetched & expected):
        return Verdict(
            passed=False,
            failure_class=FailureClass.WRONG_MATERIAL,
            reported=reported,
            truth=truth,
            relative_error=rel_err,
            stage_checks=stage_checks,
            detail=(
                f"Agent worked on {sorted(fetched)} but the task specifies {sorted(expected)}."
            ),
        )

    # --- 4. Silent convention errors ---------------------------------------------------
    rec = reconcile(reported.value, truth, task.cell_context, task.tolerance)
    if rec.matched and rec.failure_class is not None:
        # Relaxation was required and skipped: attribute to state, not to arithmetic.
        failure = rec.failure_class
        if "relax" in missing:
            failure = FailureClass.SILENT_STATE
        return Verdict(
            passed=False,
            failure_class=failure,
            reported=reported,
            truth=truth,
            relative_error=rel_err,
            reconciliation=rec,
            stage_checks=stage_checks,
            detail=rec.explanation,
        )

    if "relax" in missing and required:
        return Verdict(
            passed=False,
            failure_class=FailureClass.SILENT_STATE,
            reported=reported,
            truth=truth,
            relative_error=rel_err,
            stage_checks=stage_checks,
            detail="Agent reported a value without relaxing the structure.",
        )

    # --- 5. Skipped a required stage ---------------------------------------------------
    if missing:
        return Verdict(
            passed=False,
            failure_class=FailureClass.PLAN_INVALID,
            reported=reported,
            truth=truth,
            relative_error=rel_err,
            stage_checks=stage_checks,
            detail=f"Required stage(s) never executed: {', '.join(missing)}.",
        )

    # --- 6. The physics is just wrong ---------------------------------------------------
    return Verdict(
        passed=False,
        failure_class=FailureClass.NUMERIC_WRONG,
        reported=reported,
        truth=truth,
        relative_error=rel_err,
        stage_checks=stage_checks,
        detail=(
            "Full workflow executed, no convention transform reconciles the result. "
            "Genuine numerical or methodological error."
        ),
    )
