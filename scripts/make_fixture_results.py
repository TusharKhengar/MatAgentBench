"""Generate synthetic results so the site can be developed without a full sweep.

Writes to `.demo/` (gitignored) -- never to `results/`. Publishing invented benchmark
numbers to the live site would be worse than publishing none, so the two directories
are kept strictly apart.

    python scripts/make_fixture_results.py
    python -m http.server 8000 --directory .demo
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matagentbench.report import (  # noqa: E402
    aggregate_model,
    build_leaderboard,
    build_trajectory_index,
)
from matagentbench.schema import (  # noqa: E402
    AttributionSummary,
    CellContext,
    ModelSpec,
    Provenance,
    Quantity,
    Step,
    Task,
    TaskFamily,
    TaskSet,
    Tolerance,
    ToolCall,
    ToolResult,
    Trajectory,
)
from matagentbench.verify import verify  # noqa: E402

OUT = ROOT / ".demo"
RESULTS = OUT / "results"

CTX = CellContext(
    n_atoms_primitive=6,
    n_atoms_conventional=12,
    n_formula_units_primitive=2,
    n_formula_units_conventional=4,
    volume_primitive=62.4,
    volume_conventional=124.8,
    reduced_formula="TiO2",
)


def task(
    task_id: str,
    family: TaskFamily,
    tier: int,
    value: float,
    unit: str,
    basis: str,
    stages: list[str],
) -> Task:
    return Task(
        task_id=task_id,
        tier=tier,  # type: ignore[arg-type]
        family=family,
        prompt=f"[demo] Compute the {family.value.replace('_', ' ')} of TiO2 (mp-2657).",
        material_ids=["mp-2657"],
        answer=Quantity(value=value, unit=unit, basis=basis, cell="primitive", state="relaxed"),  # type: ignore[arg-type]
        tolerance=Tolerance(rel=0.05),
        cell_context=CTX,
        provenance=Provenance(source="materials_project", source_id="mp-2657"),
        required_stages=stages,
        max_steps=20,
    )


def step(
    i: int, name: str, args: dict, content: str = "{}", ok: bool = True, thought: str = ""
) -> Step:
    return Step(
        index=i,
        thought=thought or f"Calling {name}.",
        tool_call=ToolCall(name=name, args=args),
        tool_result=ToolResult(ok=ok, content=content, error=None if ok else "handle not found"),
        context_tokens=800 + i * 450,
        prompt_tokens=700 + i * 400,
        completion_tokens=90,
        latency_ms=640.0 + i * 25,
    )


FULL = ["fetch_structure", "relax", "reference_energies", "analysis"]


def build() -> None:
    tasks = [
        task(
            "formation_energy__mp-2657",
            TaskFamily.FORMATION_ENERGY,
            2,
            -3.5,
            "eV/atom",
            "per_atom",
            FULL,
        ),
        task(
            "bulk_modulus__mp-2657",
            TaskFamily.BULK_MODULUS,
            2,
            210.0,
            "GPa",
            "per_cell",
            ["fetch_structure", "relax", "eos"],
        ),
        task(
            "density__mp-2657",
            TaskFamily.DENSITY,
            1,
            4.23,
            "g/cm^3",
            "per_cell",
            ["fetch_structure"],
        ),
        task(
            "vacancy_formation__mp-2657",
            TaskFamily.VACANCY_FORMATION,
            3,
            5.1,
            "eV",
            "per_cell",
            ["fetch_structure", "cell_setup", "defect", "relax", "analysis"],
        ),
    ]
    taskset = TaskSet(name="demo-v0", tasks=tasks)
    by_id = {t.task_id: t for t in tasks}

    models = [
        ModelSpec(backend="cerebras", model="gpt-oss-120b", open_weights=True),
        ModelSpec(backend="groq", model="qwen/qwen3-32b", open_weights=True),
    ]

    all_trajectories: list[Trajectory] = []
    model_results = []

    for m_i, model in enumerate(models):
        trajectories: list[Trajectory] = []

        # 1. A silent basis error: correct physics, reported per cell instead of per atom.
        t = by_id["formation_energy__mp-2657"]
        trajectories.append(
            Trajectory(
                run_id=f"demo{m_i}a",
                task_id=t.task_id,
                model=model,
                steps=[
                    step(
                        0,
                        "mp_get_structure",
                        {"material_id": "mp-2657"},
                        '{"handle":"s0","formula":"TiO2","n_sites":6}',
                        thought="Fetch the rutile structure from Materials Project.",
                    ),
                    step(
                        1,
                        "relax_structure",
                        {"handle": "s0"},
                        '{"handle":"s1","energy_per_atom_eV":-8.42,"n_atoms":6}',
                        thought="Relax it before computing any energy.",
                    ),
                    step(
                        2,
                        "lookup_reference_energies",
                        {"elements": ["Ti", "O"]},
                        '{"reference_energies_eV_per_atom":{"Ti":-7.89,"O":-4.95}}',
                    ),
                    step(
                        3,
                        "python_eval",
                        {"expression": "(-50.52) - (2*-7.89 + 4*-4.95)"},
                        '{"value":-21.0}',
                        thought="Subtract the elemental references from the cell energy.",
                    ),
                ],
                final_answer_raw="THOUGHT: The formation energy is the difference.\n"
                'FINAL_ANSWER: {"value": -21.0, "unit": "eV", "basis": "per_cell"}',
                total_tokens=4200,
            )
        )

        # 2. A clean pass.
        t = by_id["density__mp-2657"]
        trajectories.append(
            Trajectory(
                run_id=f"demo{m_i}b",
                task_id=t.task_id,
                model=model,
                steps=[
                    step(
                        0,
                        "mp_get_structure",
                        {"material_id": "mp-2657"},
                        '{"handle":"s0","volume_angstrom3":62.4}',
                    )
                ],
                final_answer_raw='FINAL_ANSWER: {"value": 4.23, "unit": "g/cm^3", "basis": "per_cell"}',
                total_tokens=900,
            )
        )

        # 3. Long-horizon degradation: loses the task partway and never answers.
        t = by_id["vacancy_formation__mp-2657"]
        trajectories.append(
            Trajectory(
                run_id=f"demo{m_i}c",
                task_id=t.task_id,
                model=model,
                steps=[
                    step(0, "mp_get_structure", {"material_id": "mp-2657"}, '{"handle":"s0"}'),
                    step(
                        1,
                        "make_supercell",
                        {"handle": "s0", "scaling": [2, 2, 2]},
                        '{"handle":"s1"}',
                    ),
                    step(2, "relax_structure", {"handle": "s1"}, '{"handle":"s2"}'),
                    step(
                        3,
                        "make_vacancy",
                        {"handle": "s7"},
                        "",
                        ok=False,
                        thought="Now remove a site from the supercell.",
                    ),
                    step(4, "make_vacancy", {"handle": "s7"}, "", ok=False),
                    step(5, "make_vacancy", {"handle": "s7"}, "", ok=False),
                ],
                final_answer_raw="I have lost track of which handle holds the supercell.",
                total_tokens=7800,
            )
        )

        # 4. Unearned pass on the second model only.
        if m_i == 1:
            t = by_id["bulk_modulus__mp-2657"]
            trajectories.append(
                Trajectory(
                    run_id=f"demo{m_i}d",
                    task_id=t.task_id,
                    model=model,
                    steps=[],
                    final_answer_raw='FINAL_ANSWER: {"value": 210.0, "unit": "GPa", "basis": "per_cell"}',
                    total_tokens=300,
                )
            )

        for traj in trajectories:
            traj.verdict = verify(by_id[traj.task_id], traj)

        attribution = AttributionSummary(
            model=model,
            n_failures=2,
            attempted_by_kind={"convention_correct": 1, "plan_repair": 2, "context_restore": 1},
            recovered_by_kind={"convention_correct": 1, "plan_repair": 1, "context_restore": 0},
        )
        model_results.append(aggregate_model(model, taskset, trajectories, attribution))
        all_trajectories.extend(trajectories)

    # A recovered counterfactual for the silent-basis run, so the viewer has one to show.
    slug = "cerebras-gpt-oss-120b"
    cf_dir = RESULTS / "counterfactuals" / slug
    cf_dir.mkdir(parents=True, exist_ok=True)
    base = all_trajectories[0]
    recovered = base.model_copy(deep=True)
    recovered.run_id = "demo0a-cf"
    recovered.derived_from = base.run_id
    recovered.final_answer_raw = (
        "THOUGHT: Converting to a per-atom basis over 6 atoms.\n"
        'FINAL_ANSWER: {"value": -3.5, "unit": "eV/atom", "basis": "per_atom", '
        '"cell": "primitive", "state": "relaxed"}'
    )
    recovered.verdict = verify(by_id[base.task_id], recovered)
    (cf_dir / f"{base.task_id}__convention_correct__k3.json").write_text(
        recovered.model_dump_json(indent=2), encoding="utf-8"
    )

    leaderboard = build_leaderboard(taskset, model_results)
    index = build_trajectory_index(taskset, all_trajectories, {(slug, base.task_id)})

    for traj in all_trajectories:
        from matagentbench.agent.runner import slugify

        folder = RESULTS / "trajectories" / slugify(f"{traj.model.backend}-{traj.model.model}")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{traj.task_id}__seed0.json").write_text(
            traj.model_dump_json(indent=2), encoding="utf-8"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leaderboard.json").write_text(
        leaderboard.model_dump_json(indent=2), encoding="utf-8"
    )
    (RESULTS / "trajectories" / "index.json").write_text(
        index.model_dump_json(indent=2), encoding="utf-8"
    )
    (RESULTS / "taskset.json").write_text(taskset.model_dump_json(indent=2), encoding="utf-8")

    for name in ("index.html", "style.css", "app.js"):
        shutil.copy(ROOT / "site" / name, OUT / name)

    print(f"Wrote demo site to {OUT}")
    print(f"  models: {len(model_results)}  trajectories: {len(all_trajectories)}")
    for result in leaderboard.results:
        print(
            f"  {result.model.backend}/{result.model.model}: "
            f"pass={result.pass_rate:.0%} earned={result.earned_pass_rate:.0%} "
            f"silent_share={result.silent_failure_rate:.0%}"
        )
    print(f"  headline: {leaderboard.headline}")
    print("\nServe with:  python -m http.server 8000 --directory .demo")


if __name__ == "__main__":
    build()
