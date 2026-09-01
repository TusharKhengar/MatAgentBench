"""Task generation, with oracle calibration.

The hard problem in building this benchmark is not writing questions -- it is making
sure a *correct* agent can pass them. Our ground truth comes from DFT; our tools are an
ML interatomic potential. Those disagree by a real, material-dependent margin. A task
whose tolerance is tighter than that margin is unpassable no matter how well the agent
reasons, and would silently pollute the failure taxonomy with NUMERIC_WRONG.

So every generated task is **calibrated**: we run the reference workflow ourselves with
the same tools the agent gets, and keep the task only if the oracle passes. What
survives is a set where failure is attributable to the agent, not to the potential.

That filter is the difference between a benchmark and a quiz.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..schema import (
    CellContext,
    Provenance,
    Quantity,
    Task,
    TaskFamily,
    TaskSet,
    Tolerance,
)

# Tolerances chosen to sit above the potential's intrinsic error for each observable,
# so that a correct workflow passes. Calibration then verifies that empirically.
FAMILY_TOLERANCE: dict[TaskFamily, Tolerance] = {
    TaskFamily.DENSITY: Tolerance(rel=0.02),
    TaskFamily.LATTICE_PARAMETER: Tolerance(rel=0.03),
    TaskFamily.FORMATION_ENERGY: Tolerance(rel=0.20, abs=0.10),
    TaskFamily.RELAXATION_DELTA: Tolerance(rel=0.15, abs=0.005),
    TaskFamily.BULK_MODULUS: Tolerance(rel=0.25),
    TaskFamily.POLYMORPH_RANKING: Tolerance(rel=0.30, abs=0.02),
    TaskFamily.VACANCY_FORMATION: Tolerance(rel=0.20, abs=0.10),
}

FAMILY_TIER: dict[TaskFamily, int] = {
    TaskFamily.DENSITY: 1,
    TaskFamily.LATTICE_PARAMETER: 1,
    TaskFamily.FORMATION_ENERGY: 2,
    TaskFamily.RELAXATION_DELTA: 2,
    TaskFamily.BULK_MODULUS: 2,
    TaskFamily.POLYMORPH_RANKING: 3,
    TaskFamily.VACANCY_FORMATION: 3,
}

FAMILY_STAGES: dict[TaskFamily, list[str]] = {
    TaskFamily.DENSITY: ["fetch_structure"],
    TaskFamily.LATTICE_PARAMETER: ["fetch_structure", "relax", "cell_setup"],
    TaskFamily.FORMATION_ENERGY: [
        "fetch_structure",
        "relax",
        "reference_energies",
        "analysis",
    ],
    TaskFamily.RELAXATION_DELTA: ["fetch_structure", "compute_energy", "relax"],
    TaskFamily.BULK_MODULUS: ["fetch_structure", "relax", "eos"],
    TaskFamily.POLYMORPH_RANKING: ["fetch_structure", "relax", "analysis"],
    TaskFamily.VACANCY_FORMATION: [
        "fetch_structure",
        "cell_setup",
        "defect",
        "relax",
        "analysis",
    ],
}

FAMILY_MAX_STEPS: dict[TaskFamily, int] = {
    TaskFamily.DENSITY: 8,
    TaskFamily.LATTICE_PARAMETER: 12,
    TaskFamily.FORMATION_ENERGY: 20,
    TaskFamily.RELAXATION_DELTA: 12,
    TaskFamily.BULK_MODULUS: 16,
    TaskFamily.POLYMORPH_RANKING: 26,
    TaskFamily.VACANCY_FORMATION: 30,
}


@dataclass
class OracleResult:
    value: float
    passed: bool
    detail: str = ""


# --------------------------------------------------------------------------------------
# Prompts. Phrased as a working scientist would ask, and deliberately NOT restating the
# unit convention in the body -- the required convention is delivered separately, so an
# agent that ignores it produces exactly the silent errors we are here to measure.
# --------------------------------------------------------------------------------------

PROMPTS: dict[TaskFamily, str] = {
    TaskFamily.DENSITY: (
        "What is the mass density of {formula} (Materials Project id {mp_id})? "
        "Use the structure as stored."
    ),
    TaskFamily.LATTICE_PARAMETER: (
        "Relax {formula} ({mp_id}) with the interatomic potential and report the "
        "lattice parameter a of its conventional cell after relaxation."
    ),
    TaskFamily.FORMATION_ENERGY: (
        "Compute the formation energy of {formula} ({mp_id}) relative to its elemental "
        "references. Relax the structure first, and use reference energies consistent "
        "with the potential you are using."
    ),
    TaskFamily.RELAXATION_DELTA: (
        "How much does the energy of {formula} ({mp_id}) drop when you relax it, "
        "compared with a single-point calculation on the unrelaxed structure?"
    ),
    TaskFamily.BULK_MODULUS: (
        "Estimate the bulk modulus of {formula} ({mp_id}). Relax the structure, then "
        "fit an energy-volume curve."
    ),
    TaskFamily.POLYMORPH_RANKING: (
        "{formula_a} has two reported polymorphs: {mp_id_a} and {mp_id_b}. Relax both "
        "and report how much more stable the more stable one is than the other."
    ),
    TaskFamily.VACANCY_FORMATION: (
        "Estimate the vacancy formation energy in {formula} ({mp_id}). Build a 2x2x2 "
        "supercell, remove one site, relax both the perfect and defected supercells, "
        "and account for the removed atom using its elemental reference energy."
    ),
}


def _task_id(family: TaskFamily, *parts: str) -> str:
    return f"{family.value}__{'-'.join(parts)}"


def _provenance(record: Any, derived: bool = False) -> Provenance:
    if derived:
        return Provenance(
            source="derived",
            source_id=record.material_id,
            url=f"https://next-gen.materialsproject.org/materials/{record.material_id}",
            license="CC-BY-4.0",
            cross_checked_against=None,
        )
    return record.provenance()


# --------------------------------------------------------------------------------------
# Oracles: the reference workflow, executed with the agent's own tools
# --------------------------------------------------------------------------------------


def oracle_density(record: Any) -> OracleResult:
    structure = record.structure()
    return OracleResult(value=float(structure.density), passed=True)


def oracle_lattice_parameter(record: Any) -> OracleResult:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from ..sim import relax

    relaxed = relax(record.structure()).final_structure
    conventional = SpacegroupAnalyzer(relaxed, symprec=0.1).get_conventional_standard_structure()
    return OracleResult(value=float(conventional.lattice.abc[0]), passed=True)


def oracle_relaxation_delta(record: Any) -> OracleResult:
    from ..sim import relax

    result = relax(record.structure())
    delta = result.relaxation_delta_per_atom
    if delta is None:
        return OracleResult(value=0.0, passed=False, detail="no initial energy recorded")
    # A structure already at its potential's minimum yields a delta of ~0, which makes a
    # relative-tolerance task meaningless. Drop those.
    if abs(delta) < 0.002:
        return OracleResult(value=delta, passed=False, detail="relaxation delta too small to score")
    return OracleResult(value=float(delta), passed=True)


def oracle_formation_energy(record: Any, reference_energies: dict[str, float]) -> OracleResult:
    from ..sim import formation_energy_per_atom, relax

    relaxed = relax(record.structure()).final_structure
    try:
        value = formation_energy_per_atom(relaxed, reference_energies)
    except KeyError as exc:
        return OracleResult(value=0.0, passed=False, detail=str(exc))
    return OracleResult(value=float(value), passed=True)


def oracle_bulk_modulus(record: Any) -> OracleResult:
    from ..sim import equation_of_state, relax

    relaxed = relax(record.structure()).final_structure
    try:
        eos = equation_of_state(relaxed)
    except ValueError as exc:
        return OracleResult(value=0.0, passed=False, detail=str(exc))
    if not math.isfinite(eos.bulk_modulus_gpa) or eos.bulk_modulus_gpa <= 0:
        return OracleResult(value=0.0, passed=False, detail="non-physical bulk modulus")
    return OracleResult(value=float(eos.bulk_modulus_gpa), passed=True)


def oracle_polymorph_gap(record_a: Any, record_b: Any) -> OracleResult:
    from ..sim import relax

    e_a = relax(record_a.structure()).energy_per_atom
    e_b = relax(record_b.structure()).energy_per_atom
    gap = abs(e_a - e_b)
    if gap < 0.005:
        return OracleResult(value=gap, passed=False, detail="polymorphs are degenerate")
    return OracleResult(value=float(gap), passed=True)


def oracle_vacancy_formation(record: Any, reference_energies: dict[str, float]) -> OracleResult:
    from ..sim import make_vacancy, relax

    structure = record.structure()
    if len(structure) * 8 > 200:
        return OracleResult(value=0.0, passed=False, detail="supercell too large to be quick")

    perfect = structure.copy()
    perfect.make_supercell([2, 2, 2])
    perfect_relaxed = relax(perfect)

    removed_symbol = str(perfect[0].specie)
    if removed_symbol not in reference_energies:
        return OracleResult(value=0.0, passed=False, detail=f"no reference for {removed_symbol}")

    defected = make_vacancy(structure, site_index=0, supercell=(2, 2, 2))
    defected_relaxed = relax(defected)

    value = (
        defected_relaxed.energy_total
        + reference_energies[removed_symbol]
        - perfect_relaxed.energy_total
    )
    return OracleResult(value=float(value), passed=True)


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def _build(
    family: TaskFamily,
    record: Any,
    value: float,
    unit: str,
    basis: str,
    cell: str,
    state: str,
    cell_context: CellContext,
    prompt_kwargs: dict[str, str],
    material_ids: list[str],
    *,
    notes: str | None = None,
) -> Task:
    return Task(
        task_id=_task_id(family, *material_ids),
        tier=FAMILY_TIER[family],  # type: ignore[arg-type]
        family=family,
        prompt=PROMPTS[family].format(**prompt_kwargs),
        material_ids=material_ids,
        answer=Quantity(value=value, unit=unit, basis=basis, cell=cell, state=state),  # type: ignore[arg-type]
        tolerance=FAMILY_TOLERANCE[family],
        cell_context=cell_context,
        provenance=_provenance(record, derived=family not in {TaskFamily.DENSITY}),
        required_stages=FAMILY_STAGES[family],
        max_steps=FAMILY_MAX_STEPS[family],
        notes=notes,
    )


def generate_tasks(
    records: list[Any],
    families: list[TaskFamily],
    *,
    reference_energies: dict[str, float] | None = None,
    calibrate: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> TaskSet:
    """Build and calibrate a task set from Materials Project records.

    `calibrate=False` skips the oracle run. Useful for a fast smoke test; never use it
    for a task set you intend to publish results against.
    """
    from ..data.sources import build_cell_context

    reference_energies = reference_energies or {}
    tasks: list[Task] = []
    log = on_progress or (lambda _msg: None)

    by_formula: dict[str, list[Any]] = {}
    for record in records:
        by_formula.setdefault(record.formula, []).append(record)

    for record in records:
        structure = record.structure()
        context = build_cell_context(structure)
        prompt_kwargs = {"formula": record.formula, "mp_id": record.material_id}

        for family in families:
            if family is TaskFamily.POLYMORPH_RANKING:
                continue  # handled separately: needs a pair

            try:
                task = _build_single(
                    family, record, context, prompt_kwargs, reference_energies, calibrate, log
                )
            except Exception as exc:  # a single bad material must not kill generation
                log(f"  skip {record.material_id} {family.value}: {type(exc).__name__}: {exc}")
                continue
            if task is not None:
                tasks.append(task)
                log(f"  kept {task.task_id} -> {task.answer.describe()}")

    if TaskFamily.POLYMORPH_RANKING in families:
        for formula, group in by_formula.items():
            if len(group) < 2:
                continue
            a, b = group[0], group[1]
            try:
                oracle = oracle_polymorph_gap(a, b) if calibrate else OracleResult(0.05, True)
            except Exception as exc:
                log(f"  skip polymorph {formula}: {type(exc).__name__}: {exc}")
                continue
            if not oracle.passed:
                log(f"  drop polymorph {formula}: {oracle.detail}")
                continue

            context = build_cell_context(a.structure())
            task = Task(
                task_id=_task_id(TaskFamily.POLYMORPH_RANKING, a.material_id, b.material_id),
                tier=3,
                family=TaskFamily.POLYMORPH_RANKING,
                prompt=PROMPTS[TaskFamily.POLYMORPH_RANKING].format(
                    formula_a=formula, mp_id_a=a.material_id, mp_id_b=b.material_id
                ),
                material_ids=[a.material_id, b.material_id],
                answer=Quantity(
                    value=oracle.value,
                    unit="eV/atom",
                    basis="per_atom",
                    cell="primitive",
                    state="relaxed",
                ),
                tolerance=FAMILY_TOLERANCE[TaskFamily.POLYMORPH_RANKING],
                cell_context=context,
                provenance=_provenance(a, derived=True),
                required_stages=FAMILY_STAGES[TaskFamily.POLYMORPH_RANKING],
                max_steps=FAMILY_MAX_STEPS[TaskFamily.POLYMORPH_RANKING],
                notes="Ground truth is the potential's own relaxed energy difference.",
            )
            tasks.append(task)
            log(f"  kept {task.task_id} -> {task.answer.describe()}")

    return TaskSet(name="matagentbench-v0", tasks=tasks)


def _build_single(
    family: TaskFamily,
    record: Any,
    context: CellContext,
    prompt_kwargs: dict[str, str],
    reference_energies: dict[str, float],
    calibrate: bool,
    log: Callable[[str], None],
) -> Task | None:
    ids = [record.material_id]

    if family is TaskFamily.DENSITY:
        oracle = oracle_density(record)
        return _build(
            family,
            record,
            oracle.value,
            "g/cm^3",
            "per_cell",
            "primitive",
            "unrelaxed",
            context,
            prompt_kwargs,
            ids,
        )

    if family is TaskFamily.LATTICE_PARAMETER:
        if not calibrate:
            return None
        oracle = oracle_lattice_parameter(record)
        return _build(
            family,
            record,
            oracle.value,
            "Angstrom",
            "per_cell",
            "conventional",
            "relaxed",
            context,
            prompt_kwargs,
            ids,
        )

    if family is TaskFamily.RELAXATION_DELTA:
        if not calibrate:
            return None
        oracle = oracle_relaxation_delta(record)
        if not oracle.passed:
            log(f"  drop {record.material_id} relaxation_delta: {oracle.detail}")
            return None
        return _build(
            family,
            record,
            oracle.value,
            "eV/atom",
            "per_atom",
            "primitive",
            "relaxed",
            context,
            prompt_kwargs,
            ids,
            notes="Ground truth is defined by the potential, not by DFT.",
        )

    if family is TaskFamily.FORMATION_ENERGY:
        if not calibrate:
            return None
        oracle = oracle_formation_energy(record, reference_energies)
        if not oracle.passed:
            log(f"  drop {record.material_id} formation_energy: {oracle.detail}")
            return None
        return _build(
            family,
            record,
            oracle.value,
            "eV/atom",
            "per_atom",
            "primitive",
            "relaxed",
            context,
            prompt_kwargs,
            ids,
            notes="Referenced to potential-consistent elemental energies.",
        )

    if family is TaskFamily.BULK_MODULUS:
        if not calibrate:
            return None
        oracle = oracle_bulk_modulus(record)
        if not oracle.passed:
            log(f"  drop {record.material_id} bulk_modulus: {oracle.detail}")
            return None
        return _build(
            family,
            record,
            oracle.value,
            "GPa",
            "per_cell",
            "primitive",
            "relaxed",
            context,
            prompt_kwargs,
            ids,
        )

    if family is TaskFamily.VACANCY_FORMATION:
        if not calibrate:
            return None
        oracle = oracle_vacancy_formation(record, reference_energies)
        if not oracle.passed:
            log(f"  drop {record.material_id} vacancy: {oracle.detail}")
            return None
        return _build(
            family,
            record,
            oracle.value,
            "eV",
            "per_cell",
            "supercell",
            "relaxed",
            context,
            prompt_kwargs,
            ids,
            notes="2x2x2 supercell, one site removed.",
        )

    return None
