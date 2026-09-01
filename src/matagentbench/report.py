"""Aggregation into the two files the static site reads.

The site is a pure renderer. Anything the page needs to say is computed here and
committed as JSON, which is what lets GitHub Pages host the whole thing with no backend
and no API key anywhere near the browser.

One reporting decision worth stating plainly: `pass_rate` counts unearned passes, and
`earned_pass_rate` does not. Publishing only the first would flatter every model that
has memorised Materials Project.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .agent.runner import slugify
from .schema import (
    SILENT_CLASSES,
    AttributionSummary,
    FailureClass,
    LeaderboardFile,
    ModelResult,
    ModelSpec,
    Task,
    TaskSet,
    Trajectory,
    TrajectoryIndex,
    TrajectoryIndexEntry,
)


def aggregate_model(
    model: ModelSpec,
    taskset: TaskSet,
    trajectories: Iterable[Trajectory],
    attribution: AttributionSummary | None = None,
) -> ModelResult:
    by_id = {t.task_id: t for t in taskset.tasks}
    trajectories = list(trajectories)

    counts: Counter[str] = Counter()
    tier_totals: Counter[int] = Counter()
    tier_passed: Counter[int] = Counter()
    n_passed = 0
    n_earned = 0
    total_steps = 0
    total_tokens = 0

    for traj in trajectories:
        task = by_id.get(traj.task_id)
        if task is None or traj.verdict is None:
            continue
        verdict = traj.verdict
        counts[verdict.failure_class.value] += 1
        tier_totals[task.tier] += 1
        total_steps += len(traj.steps)
        total_tokens += traj.total_tokens

        if verdict.passed:
            n_passed += 1
            tier_passed[task.tier] += 1
            if verdict.failure_class is not FailureClass.UNEARNED_PASS:
                n_earned += 1

    n_tasks = sum(tier_totals.values())
    n_failed = n_tasks - n_passed
    n_silent = sum(counts.get(fc.value, 0) for fc in SILENT_CLASSES)

    return ModelResult(
        model=model,
        n_tasks=n_tasks,
        n_passed=n_passed,
        pass_rate=(n_passed / n_tasks) if n_tasks else 0.0,
        earned_pass_rate=(n_earned / n_tasks) if n_tasks else 0.0,
        silent_failure_rate=(n_silent / n_failed) if n_failed else 0.0,
        failure_counts=dict(counts),
        pass_rate_by_tier={
            str(tier): (tier_passed[tier] / tier_totals[tier]) if tier_totals[tier] else 0.0
            for tier in sorted(tier_totals)
        },
        mean_steps=(total_steps / n_tasks) if n_tasks else None,
        total_tokens=total_tokens,
        attribution=attribution,
    )


def build_leaderboard(taskset: TaskSet, results: list[ModelResult]) -> LeaderboardFile:
    silent_shares = [r.silent_failure_rate for r in results if r.n_tasks]
    all_counts: Counter[str] = Counter()
    for result in results:
        all_counts.update(result.failure_counts)

    total_failures = sum(
        count
        for cls, count in all_counts.items()
        if cls not in (FailureClass.SUCCESS.value, FailureClass.UNEARNED_PASS.value)
    )
    silent_total = sum(all_counts.get(fc.value, 0) for fc in SILENT_CLASSES)

    dominant = None
    if total_failures:
        ranked = [
            (cls, count)
            for cls, count in all_counts.most_common()
            if cls not in (FailureClass.SUCCESS.value, FailureClass.UNEARNED_PASS.value)
        ]
        if ranked:
            dominant = {"failure_class": ranked[0][0], "share": ranked[0][1] / total_failures}

    return LeaderboardFile(
        taskset=taskset.name,
        n_tasks=len(taskset.tasks),
        results=sorted(results, key=lambda r: r.pass_rate, reverse=True),
        headline={
            "silent_failure_share_of_failures": (
                silent_total / total_failures if total_failures else 0.0
            ),
            "mean_silent_share_across_models": (
                sum(silent_shares) / len(silent_shares) if silent_shares else 0.0
            ),
            "dominant_failure": dominant,
            "unearned_passes": all_counts.get(FailureClass.UNEARNED_PASS.value, 0),
            "n_models": len(results),
        },
    )


def build_trajectory_index(
    taskset: TaskSet,
    trajectories: Iterable[Trajectory],
    counterfactual_task_ids: set[tuple[str, str]] | None = None,
) -> TrajectoryIndex:
    """`counterfactual_task_ids` holds (model_slug, task_id) pairs that have a replay."""
    by_id: dict[str, Task] = {t.task_id: t for t in taskset.tasks}
    have_cf = counterfactual_task_ids or set()
    entries: list[TrajectoryIndexEntry] = []

    for traj in trajectories:
        task = by_id.get(traj.task_id)
        if task is None or traj.verdict is None:
            continue
        model_slug = slugify(f"{traj.model.backend}-{traj.model.model}")
        entries.append(
            TrajectoryIndexEntry(
                run_id=traj.run_id,
                task_id=traj.task_id,
                family=task.family,
                tier=task.tier,
                model=f"{traj.model.backend}/{traj.model.model}",
                failure_class=traj.verdict.failure_class,
                passed=traj.verdict.passed,
                n_steps=len(traj.steps),
                has_counterfactual=(model_slug, traj.task_id) in have_cf,
                path=f"trajectories/{model_slug}/{traj.task_id}__seed{traj.seed}.json",
            )
        )

    # Failures first, and silent failures ahead of loud ones: the viewer should open on
    # the runs that are actually interesting to read.
    def sort_key(entry: TrajectoryIndexEntry) -> tuple[int, int, str]:
        if entry.passed:
            return (2, 0, entry.task_id)
        is_silent = 0 if entry.failure_class in SILENT_CLASSES else 1
        return (0, is_silent, entry.task_id)

    entries.sort(key=sort_key)
    return TrajectoryIndex(entries=entries)


def discover_counterfactuals(results_dir: Path) -> set[tuple[str, str]]:
    root = results_dir / "counterfactuals"
    if not root.exists():
        return set()
    found: set[tuple[str, str]] = set()
    for path in root.glob("*/*.json"):
        model_slug = path.parent.name
        # Filenames are `{task_id}__{kind}__k{n}.json`, and task_id itself contains a
        # `__` separator (e.g. `formation_energy__mp-2657`). Splitting from the left
        # truncates it to the family name and silently drops every match.
        stem = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
        parts = stem.rsplit("__", 2)
        if len(parts) != 3:
            continue
        found.add((model_slug, parts[0]))
    return found
