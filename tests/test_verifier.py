"""End-to-end classification tests: trajectory in, named failure class out."""

from __future__ import annotations

import pytest

from matagentbench.agent.prompts import parse_action
from matagentbench.agent.tools import safe_eval
from matagentbench.schema import (
    CellContext,
    FailureClass,
    ModelSpec,
    Provenance,
    Quantity,
    Step,
    Task,
    TaskFamily,
    Tolerance,
    ToolCall,
    ToolResult,
    Trajectory,
)
from matagentbench.verify import parse_final_answer, verify


def make_task(**overrides) -> Task:
    defaults = dict(
        task_id="formation_energy__mp-149",
        tier=2,
        family=TaskFamily.FORMATION_ENERGY,
        prompt="Compute the formation energy of Si (mp-149).",
        material_ids=["mp-149"],
        answer=Quantity(
            value=-3.5, unit="eV/atom", basis="per_atom", cell="primitive", state="relaxed"
        ),
        tolerance=Tolerance(rel=0.05),
        cell_context=CellContext(
            n_atoms_primitive=6,
            n_atoms_conventional=12,
            n_formula_units_primitive=2,
            n_formula_units_conventional=4,
            reduced_formula="TiO2",
        ),
        provenance=Provenance(source="materials_project", source_id="mp-149"),
        required_stages=["fetch_structure", "relax", "reference_energies", "analysis"],
        max_steps=20,
    )
    defaults.update(overrides)
    return Task(**defaults)


def make_traj(steps: list[Step], final: str | None = None, **overrides) -> Trajectory:
    return Trajectory(
        run_id="test0001",
        task_id="formation_energy__mp-149",
        model=ModelSpec(backend="test", model="fake"),
        steps=steps,
        final_answer_raw=final,
        **overrides,
    )


def tool_step(index: int, name: str, args: dict, ok: bool = True) -> Step:
    return Step(
        index=index,
        tool_call=ToolCall(name=name, args=args),
        tool_result=ToolResult(ok=ok, content="{}" if ok else "", error=None if ok else "boom"),
    )


def full_workflow() -> list[Step]:
    return [
        tool_step(0, "mp_get_structure", {"material_id": "mp-149"}),
        tool_step(1, "relax_structure", {"handle": "s0"}),
        tool_step(2, "lookup_reference_energies", {"elements": ["Si"]}),
        tool_step(3, "python_eval", {"expression": "1+1"}),
    ]


# --------------------------------------------------------------------------------------


def test_success():
    task = make_task()
    traj = make_traj(
        full_workflow(),
        final='FINAL_ANSWER: {"value": -3.5, "unit": "eV/atom", "basis": "per_atom", '
        '"cell": "primitive", "state": "relaxed"}',
    )
    verdict = verify(task, traj)
    assert verdict.passed
    assert verdict.failure_class is FailureClass.SUCCESS


def test_correct_value_without_doing_the_work_is_unearned():
    """The contamination guard: right answer, no tool calls."""
    task = make_task()
    traj = make_traj([], final='FINAL_ANSWER: {"value": -3.5, "unit": "eV/atom"}')
    verdict = verify(task, traj)
    assert verdict.passed
    assert verdict.failure_class is FailureClass.UNEARNED_PASS
    assert "relax" in (verdict.detail or "")


def test_silent_basis_error_is_named():
    """Full workflow, right physics, reported per cell instead of per atom."""
    task = make_task()
    traj = make_traj(
        full_workflow(),
        final='FINAL_ANSWER: {"value": -21.0, "unit": "eV", "basis": "per_cell"}',
    )
    verdict = verify(task, traj)
    assert not verdict.passed
    assert verdict.failure_class is FailureClass.SILENT_BASIS
    assert verdict.reconciliation is not None and verdict.reconciliation.matched


def test_skipping_relaxation_is_silent_state():
    task = make_task()
    steps = [
        tool_step(0, "mp_get_structure", {"material_id": "mp-149"}),
        tool_step(1, "compute_energy", {"handle": "s0"}),
        tool_step(2, "lookup_reference_energies", {"elements": ["Si"]}),
        tool_step(3, "python_eval", {"expression": "1+1"}),
    ]
    traj = make_traj(steps, final='FINAL_ANSWER: {"value": -2.9, "unit": "eV/atom"}')
    verdict = verify(task, traj)
    assert not verdict.passed
    assert verdict.failure_class is FailureClass.SILENT_STATE


def test_wrong_material():
    task = make_task()
    steps = [
        tool_step(0, "mp_get_structure", {"material_id": "mp-999"}),
        tool_step(1, "relax_structure", {"handle": "s0"}),
        tool_step(2, "lookup_reference_energies", {"elements": ["Si"]}),
        tool_step(3, "python_eval", {"expression": "1"}),
    ]
    traj = make_traj(steps, final='FINAL_ANSWER: {"value": -1.1, "unit": "eV/atom"}')
    verdict = verify(task, traj)
    assert verdict.failure_class is FailureClass.WRONG_MATERIAL


def test_no_answer():
    task = make_task()
    traj = make_traj(full_workflow(), final="I was unable to determine this.")
    verdict = verify(task, traj)
    assert not verdict.passed
    assert verdict.failure_class is FailureClass.NO_ANSWER


def test_loop_detection():
    task = make_task()
    steps = [tool_step(i, "structure_info", {"handle": "s0"}) for i in range(4)]
    traj = make_traj(steps, final="still thinking")
    verdict = verify(task, traj)
    assert verdict.failure_class is FailureClass.LOOP


def test_tool_misuse_when_most_calls_error():
    task = make_task()
    steps = [tool_step(i, "relax_structure", {"handle": f"s{i}"}, ok=False) for i in range(3)]
    traj = make_traj(steps, final="no result")
    verdict = verify(task, traj)
    assert verdict.failure_class in {FailureClass.TOOL_MISUSE, FailureClass.LOOP}


def test_genuinely_wrong_physics_is_numeric_wrong():
    task = make_task()
    traj = make_traj(
        full_workflow(),
        final='FINAL_ANSWER: {"value": -1.234, "unit": "eV/atom", "basis": "per_atom"}',
    )
    verdict = verify(task, traj)
    assert verdict.failure_class is FailureClass.NUMERIC_WRONG


# --------------------------------------------------------------------------------------


def test_parser_handles_strict_json():
    parsed = parse_final_answer('FINAL_ANSWER: {"value": -3.5, "unit": "eV/atom"}')
    assert parsed.method == "json"
    assert parsed.quantity.value == pytest.approx(-3.5)


def test_parser_handles_loose_tagged_answer():
    parsed = parse_final_answer("FINAL_ANSWER: -3.5 eV/atom")
    assert parsed.method == "tagged"
    assert parsed.quantity.unit == "eV/atom"


def test_parser_falls_back_to_trailing_number():
    parsed = parse_final_answer("After relaxing, the energy per atom is -3.5")
    assert parsed.method == "trailing_number"
    assert parsed.quantity.value == pytest.approx(-3.5)


def test_parser_returns_nothing_for_prose():
    assert parse_final_answer("I could not compute this.").quantity is None


def test_action_parsing():
    action = parse_action(
        'THOUGHT: fetch it\nACTION: mp_get_structure\nARGS: {"material_id": "mp-149"}'
    )
    assert action.name == "mp_get_structure"
    assert action.args == {"material_id": "mp-149"}
    assert not action.is_final


def test_malformed_args_are_reported_not_repaired():
    action = parse_action("THOUGHT: go\nACTION: mp_get_structure\nARGS: {material_id: mp-149}")
    assert action.name == "mp_get_structure"
    assert action.parse_error is not None


# --------------------------------------------------------------------------------------


def test_safe_eval_arithmetic():
    assert safe_eval("(-21.0 + 3) / 6") == pytest.approx(-3.0)
    assert safe_eval("sqrt(16)") == pytest.approx(4.0)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "open('x')", "structure.volume", "[].__class__"],
)
def test_safe_eval_rejects_everything_else(expression):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expression)
