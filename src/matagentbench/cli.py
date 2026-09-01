"""Command line entry point.

The pipeline is five commands, each resumable and each safe to interrupt:

    mab doctor                 # what is configured, what is missing
    mab gen-tasks --limit 60   # build + calibrate the task set
    mab run --preset mid-open  # sweep trajectories (resumable)
    mab attribute --preset mid-open
    mab report                 # emit the JSON the site renders
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .schema import (
    AttributionSummary,
    ModelSpec,
    TaskFamily,
    TaskSet,
    Trajectory,
)

app = typer.Typer(
    add_completion=False, help="MatAgentBench: agent evaluation for atomistic workflows."
)
console = Console()

DEFAULT_TASKSET = Path("tasks/matagentbench-v0.json")
DEFAULT_RESULTS = Path("results")


def _load_env() -> None:
    load_dotenv(override=False)


def _load_taskset(path: Path) -> TaskSet:
    if not path.exists():
        console.print(f"[red]No task set at {path}.[/] Run [bold]mab gen-tasks[/] first.")
        raise typer.Exit(1)
    return TaskSet.model_validate_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Report what is installed and configured, and what is missing."""
    _load_env()
    from .agent.backends import PRESETS, PROVIDERS

    table = Table(title="MatAgentBench environment", show_lines=False)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    for module, extra in [
        ("pydantic", "core"),
        ("httpx", "core"),
        ("numpy", "core"),
        ("pymatgen", "data"),
        ("mp_api", "data"),
        ("chgnet", "sim"),
        ("ase", "sim"),
    ]:
        try:
            __import__(module)
            table.add_row(module, "[green]ok[/]", extra)
        except ImportError:
            # Rich reads square brackets as markup, so the extras name needs escaping
            # or the install hint prints as `pip install -e "."`.
            hint = "core deps" if extra == "core" else rf'pip install -e ".\[{extra}]"'
            table.add_row(module, "[yellow]missing[/]", hint)

    mp_key = os.getenv("MP_API_KEY")
    table.add_row(
        "MP_API_KEY",
        "[green]set[/]" if mp_key else "[red]missing[/]",
        "https://next-gen.materialsproject.org/api (free)" if not mp_key else "ok",
    )

    for preset, (provider, model) in PRESETS.items():
        env_var = PROVIDERS[provider].api_key_env
        if provider == "local":
            # No key is needed, but that is not the same as being usable -- probe it.
            base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
            reachable = _probe(base_url)
            status = "[green]ready[/]" if reachable else "[dim]no server[/]"
            detail = f"{provider}/{model} at {base_url}"
            if not reachable:
                detail += "  (start Ollama or llama.cpp)"
        elif os.getenv(env_var):
            status, detail = "[green]ready[/]", f"{provider}/{model}"
        else:
            status, detail = "[dim]no key[/]", f"{provider}/{model}  (set {env_var})"
        table.add_row(f"preset:{preset}", status, detail)

    console.print(table)


def _probe(base_url: str, timeout: float = 1.5) -> bool:
    """Is there an OpenAI-compatible server listening?"""
    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return response.status_code < 500
    except Exception:
        return False


@app.command("gen-tasks")
def gen_tasks(
    out: Path = typer.Option(DEFAULT_TASKSET, help="Where to write the task set."),
    limit: int = typer.Option(60, help="Maximum number of source materials to consider."),
    max_sites: int = typer.Option(12, help="Skip cells larger than this (keeps runs fast)."),
    elements: str = typer.Option("", help="Comma-separated element filter, e.g. 'Si,O'."),
    families: str = typer.Option(
        "density,lattice_parameter,formation_energy,relaxation_delta,bulk_modulus",
        help="Comma-separated task families.",
    ),
    no_calibrate: bool = typer.Option(
        False, "--no-calibrate", help="Skip oracle verification. Smoke tests only."
    ),
) -> None:
    """Build a task set from Materials Project, calibrated against the oracle workflow."""
    _load_env()
    from .data.sources import MaterialsProjectSource
    from .tasks.generate import generate_tasks

    selected = [TaskFamily(f.strip()) for f in families.split(",") if f.strip()]
    element_filter = [e.strip() for e in elements.split(",") if e.strip()] or None

    source = MaterialsProjectSource()
    console.print(f"[bold]Querying Materials Project[/] (limit={limit}, max_sites={max_sites})...")
    records = source.search(elements=element_filter, max_sites=max_sites, limit=limit)
    console.print(f"  {len(records)} candidate materials")

    reference_energies: dict[str, float] = {}
    needs_references = {TaskFamily.FORMATION_ENERGY, TaskFamily.VACANCY_FORMATION} & set(selected)
    if needs_references and not no_calibrate:
        from .sim import energy_per_atom

        symbols = sorted({str(el) for r in records for el in r.structure().composition.elements})
        console.print(f"[bold]Computing elemental references[/] for {len(symbols)} elements...")
        for symbol in symbols:
            try:
                elemental = source.search(
                    elements=[symbol], num_elements=(1, 1), max_sites=8, limit=1
                )
                if elemental:
                    reference_energies[symbol] = energy_per_atom(elemental[0].structure())
            except Exception as exc:
                console.print(f"  [yellow]no reference for {symbol}: {exc}[/]")

    console.print("[bold]Generating and calibrating tasks[/]...")
    taskset = generate_tasks(
        records,
        selected,
        reference_energies=reference_energies,
        calibrate=not no_calibrate,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/]"),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(taskset.model_dump_json(indent=2), encoding="utf-8")

    by_tier: dict[int, int] = {}
    for task in taskset.tasks:
        by_tier[task.tier] = by_tier.get(task.tier, 0) + 1
    console.print(
        f"[green]Wrote {len(taskset.tasks)} calibrated tasks[/] to {out}  "
        f"(by tier: {dict(sorted(by_tier.items()))})"
    )


@app.command()
def run(
    preset: str = typer.Option("mid-open", help="Backend preset, or provider name with --model."),
    model: str = typer.Option("", help="Explicit model id when preset is a provider name."),
    taskset: Path = typer.Option(DEFAULT_TASKSET),
    results: Path = typer.Option(DEFAULT_RESULTS),
    seed: int = typer.Option(0),
    limit: int = typer.Option(0, help="Run only the first N tasks. 0 = all."),
    force: bool = typer.Option(False, "--force", help="Re-run tasks that already have results."),
    context_limit: int = typer.Option(
        24_000, help="Transcript char budget before old observations are dropped."
    ),
) -> None:
    """Sweep trajectories. Safe to interrupt -- completed tasks are skipped on resume."""
    _load_env()
    from .agent.backends import build_backend
    from .agent.runner import run_task_cached, save_trajectory, trajectory_path
    from .data.sources import MaterialsProjectSource
    from .verify import verify

    ts = _load_taskset(taskset)
    tasks = ts.tasks[:limit] if limit else ts.tasks

    backend = build_backend(preset, model or None)
    mp_source = MaterialsProjectSource()
    console.print(
        f"[bold]{backend.spec.backend}/{backend.spec.model}[/] "
        f"over {len(tasks)} tasks (seed {seed})"
    )

    n_passed = cached = 0
    for i, task in enumerate(tasks, start=1):
        traj, was_cached = run_task_cached(
            task,
            backend,
            results,
            seed=seed,
            force=force,
            mp_source=mp_source,
            context_limit_chars=context_limit,
        )
        if traj.verdict is None:
            traj.verdict = verify(task, traj)
            save_trajectory(traj, trajectory_path(results, backend.spec, task.task_id, seed))

        cached += int(was_cached)
        n_passed += int(traj.verdict.passed)
        mark = "[green]PASS[/]" if traj.verdict.passed else "[red]FAIL[/]"
        suffix = " [dim](cached)[/]" if was_cached else ""
        console.print(
            f"  [{i}/{len(tasks)}] {mark} {task.task_id} "
            f"[dim]{traj.verdict.failure_class.value}[/]{suffix}"
        )

    console.print(f"[bold]{n_passed}/{len(tasks)} passed[/]  ({cached} reused from previous runs)")


@app.command()
def attribute(
    preset: str = typer.Option("mid-open"),
    model: str = typer.Option(""),
    taskset: Path = typer.Option(DEFAULT_TASKSET),
    results: Path = typer.Option(DEFAULT_RESULTS),
    seed: int = typer.Option(0),
    limit: int = typer.Option(0, help="Attribute only the first N failures. 0 = all."),
) -> None:
    """Run counterfactual interventions over failed trajectories."""
    _load_env()
    from .agent.backends import build_backend
    from .agent.runner import load_trajectory, trajectory_path
    from .attribute import plan_interventions, run_counterfactual, summarise
    from .data.sources import MaterialsProjectSource

    ts = _load_taskset(taskset)
    backend = build_backend(preset, model or None)
    mp_source = MaterialsProjectSource()

    failures: list[tuple[object, Trajectory]] = []
    for task in ts.tasks:
        path = trajectory_path(results, backend.spec, task.task_id, seed)
        if not path.exists():
            continue
        traj = load_trajectory(path)
        if traj.verdict is not None and not traj.verdict.passed:
            failures.append((task, traj))

    if limit:
        failures = failures[:limit]
    if not failures:
        console.print("[yellow]No failed trajectories found. Run `mab run` first.[/]")
        raise typer.Exit(0)

    console.print(f"[bold]Attributing {len(failures)} failures[/]")
    outcomes = []
    for task, traj in failures:
        plans = plan_interventions(task, traj)  # type: ignore[arg-type]
        for plan in plans:
            outcome = run_counterfactual(
                task,
                traj,
                plan,
                backend,
                results,
                mp_source=mp_source,  # type: ignore[arg-type]
            )
            outcomes.append(outcome)
            mark = "[green]recovered[/]" if outcome.recovered else "[dim]no change[/]"
            console.print(
                f"  {task.task_id} :: {plan.spec.kind.value}@k={plan.spec.step_k} -> {mark} "
                f"[dim]({outcome.base_failure.value} -> {outcome.new_failure.value})[/]"
            )

    summary = summarise(backend.spec, outcomes, n_failures=len(failures))
    out = (
        results
        / "attribution"
        / f"{backend.spec.backend}-{backend.spec.model}.json".replace("/", "-")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    table = Table(title="Recovery rate by intervention")
    table.add_column("Intervention")
    table.add_column("Attempted", justify="right")
    table.add_column("Recovered", justify="right")
    table.add_column("Rate", justify="right")
    for kind, attempted in sorted(summary.attempted_by_kind.items()):
        recovered = summary.recovered_by_kind.get(kind, 0)
        table.add_row(kind, str(attempted), str(recovered), f"{recovered / attempted:.0%}")
    console.print(table)
    console.print(f"[green]Wrote[/] {out}")


@app.command()
def report(
    taskset: Path = typer.Option(DEFAULT_TASKSET),
    results: Path = typer.Option(DEFAULT_RESULTS),
) -> None:
    """Aggregate everything on disk into leaderboard.json and the trajectory index."""
    _load_env()
    from .agent.runner import load_trajectory
    from .report import (
        aggregate_model,
        build_leaderboard,
        build_trajectory_index,
        discover_counterfactuals,
    )

    ts = _load_taskset(taskset)
    root = results / "trajectories"
    if not root.exists():
        console.print("[yellow]No trajectories found. Run `mab run` first.[/]")
        raise typer.Exit(0)

    all_trajectories: list[Trajectory] = []
    model_results = []

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trajectories: list[Trajectory] = []
        for path in sorted(model_dir.glob("*.json")):
            if path.name.endswith(".partial.json"):
                continue
            try:
                trajectories.append(load_trajectory(path))
            except (OSError, ValueError) as exc:
                console.print(f"[yellow]skipping unreadable {path.name}: {exc}[/]")
        if not trajectories:
            continue

        spec: ModelSpec = trajectories[0].model
        attribution = None
        attribution_path = (
            results / "attribution" / f"{spec.backend}-{spec.model}.json".replace("/", "-")
        )
        if attribution_path.exists():
            attribution = AttributionSummary.model_validate_json(
                attribution_path.read_text(encoding="utf-8")
            )

        model_results.append(aggregate_model(spec, ts, trajectories, attribution))
        all_trajectories.extend(trajectories)

    leaderboard = build_leaderboard(ts, model_results)
    index = build_trajectory_index(ts, all_trajectories, discover_counterfactuals(results))

    (results / "leaderboard.json").write_text(
        leaderboard.model_dump_json(indent=2), encoding="utf-8"
    )
    (results / "trajectories" / "index.json").write_text(
        index.model_dump_json(indent=2), encoding="utf-8"
    )
    (results / "taskset.json").write_text(ts.model_dump_json(indent=2), encoding="utf-8")

    table = Table(title="MatAgentBench")
    table.add_column("Model")
    table.add_column("Pass", justify="right")
    table.add_column("Earned", justify="right")
    table.add_column("Silent share of failures", justify="right")
    for result in leaderboard.results:
        table.add_row(
            f"{result.model.backend}/{result.model.model}",
            f"{result.pass_rate:.0%}",
            f"{result.earned_pass_rate:.0%}",
            f"{result.silent_failure_rate:.0%}",
        )
    console.print(table)
    console.print(json.dumps(leaderboard.headline, indent=2, default=str))
    console.print(f"[green]Wrote[/] {results / 'leaderboard.json'}")


if __name__ == "__main__":
    app()
