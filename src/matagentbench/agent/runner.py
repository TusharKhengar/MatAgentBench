"""The agent loop, and the persistence that makes it survivable on free tiers.

Two things here are load-bearing for the project actually finishing:

**Resumability.** Every completed trajectory is written to disk immediately and skipped
on a re-run. A sweep interrupted by a rate limit, a closed laptop or a cancelled CI job
resumes exactly where it stopped, and costs nothing for the work already done.

**Context pressure.** `context_limit_chars` truncates the oldest observations once the
transcript grows past a budget. This is not an optimisation -- it is how the benchmark
*induces* the long-horizon forgetting failure that the CONTEXT_RESTORE intervention
then repairs. Without it, tier-3 tasks would never exhibit the degradation we are
trying to measure.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schema import (
    InterventionKind,
    InterventionSpec,
    ModelSpec,
    Step,
    Task,
    ToolCall,
    ToolResult,
    Trajectory,
)
from .backends import ChatMessage, LLMBackend
from .prompts import (
    build_system_prompt,
    build_user_prompt,
    convention_reminder,
    parse_action,
    reference_plan,
)
from .tools import Session, ToolRegistry

DEFAULT_CONTEXT_LIMIT_CHARS = 24_000
TRUNCATION_NOTICE = "[... earlier observations elided to fit the context budget ...]"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def trajectory_path(results_dir: Path, model: ModelSpec, task_id: str, seed: int) -> Path:
    folder = results_dir / "trajectories" / slugify(f"{model.backend}-{model.model}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{task_id}__seed{seed}.json"


def _estimate_tokens(text: str) -> int:
    """Rough char/4 heuristic. Good enough for relative context-pressure accounting."""
    return max(1, len(text) // 4)


def _apply_context_budget(messages: list[ChatMessage], limit_chars: int) -> list[ChatMessage]:
    """Drop the oldest observations, never the system prompt or the task statement."""
    total = sum(len(m.content) for m in messages)
    if total <= limit_chars or len(messages) <= 3:
        return messages

    head, tail = messages[:2], messages[2:]
    kept: list[ChatMessage] = []
    running = sum(len(m.content) for m in head)

    for message in reversed(tail):
        if running + len(message.content) > limit_chars and len(kept) >= 2:
            break
        kept.append(message)
        running += len(message.content)

    kept.reverse()
    if len(kept) < len(tail):
        return [*head, ChatMessage("user", TRUNCATION_NOTICE), *kept]
    return [*head, *kept]


def run_trajectory(
    task: Task,
    backend: LLMBackend,
    *,
    registry: ToolRegistry | None = None,
    mp_source: Any | None = None,
    seed: int = 0,
    intervention: InterventionSpec | None = None,
    derived_from: str | None = None,
    context_limit_chars: int = DEFAULT_CONTEXT_LIMIT_CHARS,
    allow_sim: bool = True,
) -> Trajectory:
    """Run one task to completion (or budget exhaustion) and return the trajectory."""
    session = Session()
    tools = registry or ToolRegistry(session, mp_source=mp_source, allow_sim=allow_sim)

    messages: list[ChatMessage] = [
        ChatMessage("system", build_system_prompt(tools.specs())),
        ChatMessage("user", build_user_prompt(task)),
    ]

    traj = Trajectory(
        run_id=uuid.uuid4().hex[:12],
        task_id=task.task_id,
        model=backend.spec,
        seed=seed,
        derived_from=derived_from,
        intervention=intervention,
    )
    started = time.perf_counter()

    for index in range(task.max_steps):
        # --- counterfactual injection ------------------------------------------------
        if intervention is not None and intervention.step_k == index:
            injected = _intervention_message(intervention, task)
            if injected is not None:
                messages.append(ChatMessage("user", injected))

        messages = _apply_context_budget(messages, context_limit_chars)

        try:
            response = backend.complete(messages, temperature=backend.spec.temperature)
        except Exception as exc:
            traj.error = f"{type(exc).__name__}: {exc}"
            break

        step = Step(
            index=index,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            raw_response=response.text,
            context_tokens=sum(_estimate_tokens(m.content) for m in messages),
        )
        traj.total_tokens += response.total_tokens
        messages.append(ChatMessage("assistant", response.text))

        action = parse_action(response.text)
        step.thought = action.thought

        if action.is_final:
            traj.final_answer_raw = response.text
            traj.steps.append(step)
            break

        if action.parse_error or not action.name:
            step.tool_result = ToolResult(ok=False, content="", error=action.parse_error)
            traj.steps.append(step)
            messages.append(
                ChatMessage(
                    "user",
                    f"OBSERVATION: protocol error -- {action.parse_error}. "
                    "Reply with THOUGHT/ACTION/ARGS or THOUGHT/FINAL_ANSWER.",
                )
            )
            continue

        # --- tool repair intervention rewrites this one call and nothing else --------
        args = dict(action.args)
        if (
            intervention is not None
            and intervention.kind is InterventionKind.TOOL_REPAIR
            and intervention.step_k == index
        ):
            args.update(intervention.payload.get("args", {}))

        step.tool_call = ToolCall(name=action.name, args=args)
        result = tools.call(action.name, args)
        step.tool_result = result
        traj.steps.append(step)

        observation = result.content if result.ok else f"ERROR: {result.error}"
        messages.append(ChatMessage("user", f"OBSERVATION:\n{observation}"))

    traj.finished_at = datetime.now(UTC)
    traj.wall_ms = (time.perf_counter() - started) * 1000
    return traj


def _intervention_message(spec: InterventionSpec, task: Task) -> str | None:
    if spec.kind is InterventionKind.PLAN_REPAIR:
        return f"GUIDANCE: {reference_plan(task)}"
    if spec.kind is InterventionKind.CONVENTION_CORRECT:
        return f"GUIDANCE: {convention_reminder(task)}"
    if spec.kind is InterventionKind.CONTEXT_RESTORE:
        restored = spec.payload.get("restored_context")
        if restored:
            return f"REMINDER of earlier results in this task:\n{restored}"
        return f"REMINDER of the task you are solving:\n{build_user_prompt(task)}"
    return None  # TOOL_REPAIR acts on the call itself, not the transcript


def run_task_cached(
    task: Task,
    backend: LLMBackend,
    results_dir: Path,
    *,
    seed: int = 0,
    force: bool = False,
    **kwargs: Any,
) -> tuple[Trajectory, bool]:
    """Run a task unless its trajectory is already on disk. Returns (traj, was_cached)."""
    path = trajectory_path(results_dir, backend.spec, task.task_id, seed)
    if path.exists() and not force:
        try:
            return Trajectory.model_validate_json(path.read_text(encoding="utf-8")), True
        except (OSError, ValueError):
            pass  # corrupt checkpoint: fall through and re-run

    traj = run_trajectory(task, backend, seed=seed, **kwargs)
    save_trajectory(traj, path)
    return traj, False


def save_trajectory(traj: Trajectory, path: Path) -> None:
    """Write atomically -- a sweep killed mid-write must not leave a corrupt checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.json")
    temp.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    temp.replace(path)


def load_trajectory(path: Path) -> Trajectory:
    return Trajectory.model_validate_json(Path(path).read_text(encoding="utf-8"))


def transcript_of(traj: Trajectory) -> list[str]:
    """Assistant messages in order -- the replay tape for counterfactual runs."""
    return [s.raw_response or "" for s in traj.steps]


def observations_before(traj: Trajectory, step_k: int, limit: int = 4) -> str:
    """Successful tool observations from before step_k, for CONTEXT_RESTORE."""
    collected: list[str] = []
    for step in traj.steps[:step_k]:
        if step.tool_result is not None and step.tool_result.ok and step.tool_call is not None:
            collected.append(f"{step.tool_call.name} -> {step.tool_result.content}")
    if not collected:
        return ""
    return "\n".join(collected[:limit])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump_json"):
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
