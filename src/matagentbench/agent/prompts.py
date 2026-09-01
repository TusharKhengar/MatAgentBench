"""Prompt construction and the agent's action protocol.

We use a **text protocol** rather than native function calling. Free-tier open-weight
endpoints disagree wildly about tool-call support -- some omit it, some accept the
schema and ignore it, some emit malformed arguments. A plain text protocol is uniform
across every provider, which keeps the comparison across models honest: differences in
the results are differences in the models, not in their vendors' tool-calling plumbing.

It also makes malformed calls *visible* as TOOL_MISUSE instead of being silently
repaired by a provider-side parser -- and those failures are part of what we measure.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schema import Quantity, Task
from .tools import ToolSpec

SYSTEM_TEMPLATE = """You are a computational materials scientist working through a \
calculation step by step. You have tools that fetch crystal structures and run \
atomistic simulations with a machine-learned interatomic potential.

AVAILABLE TOOLS
{tools}

PROTOCOL
Reply with exactly one step at a time, in this format:

THOUGHT: <one or two sentences of reasoning>
ACTION: <tool name>
ARGS: <a single-line JSON object>

After you see the observation, produce the next step. When you have the answer:

THOUGHT: <why this is the answer>
FINAL_ANSWER: {{"value": <number>, "unit": "<unit>", "basis": \
"<per_atom|per_formula_unit|per_cell>", "cell": "<primitive|conventional|supercell>", \
"state": "<relaxed|unrelaxed>"}}

RULES THAT MATTER
- Units and basis are part of the answer, not decoration. `energy_per_atom_eV` is per \
ATOM. A value per formula unit or per cell is a different number.
- The primitive and conventional cells contain different numbers of atoms. Extensive \
quantities differ between them by exactly that ratio.
- A single-point energy is not a relaxed energy. If the task asks for a relaxed \
property, you must call relax_structure.
- Do not guess a value you remember. Compute it with the tools. Answers that match \
reference data without the corresponding tool calls are recorded as unearned.
- One ACTION per reply. Never invent tool names or arguments.
"""

USER_TEMPLATE = """TASK ({task_id}, tier {tier})

{prompt}

Report the final value in: {unit} [{basis}, {cell} cell, {state}]
Step budget: {max_steps}
"""

_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)", re.M)
_ARGS_RE = re.compile(r"^\s*ARGS\s*:\s*(?P<args>\{.*?\})\s*$", re.M | re.DOTALL)
_THOUGHT_RE = re.compile(
    r"^\s*THOUGHT\s*:\s*(?P<thought>.+?)(?=^\s*(?:ACTION|FINAL_ANSWER)\s*:|\Z)", re.M | re.DOTALL
)
_FINAL_RE = re.compile(r"FINAL_ANSWER\s*:", re.I)


def build_system_prompt(specs: list[ToolSpec]) -> str:
    return SYSTEM_TEMPLATE.format(tools="\n".join(s.render() for s in specs))


def build_user_prompt(task: Task) -> str:
    answer: Quantity = task.answer
    return USER_TEMPLATE.format(
        task_id=task.task_id,
        tier=task.tier,
        prompt=task.prompt,
        unit=answer.unit,
        basis=answer.basis,
        cell=answer.cell,
        state=answer.state,
        max_steps=task.max_steps,
    )


class ParsedAction:
    __slots__ = ("thought", "name", "args", "is_final", "parse_error")

    def __init__(
        self,
        thought: str | None = None,
        name: str | None = None,
        args: dict[str, Any] | None = None,
        is_final: bool = False,
        parse_error: str | None = None,
    ):
        self.thought = thought
        self.name = name
        self.args = args or {}
        self.is_final = is_final
        self.parse_error = parse_error


def parse_action(text: str) -> ParsedAction:
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group("thought").strip() if thought_match else None

    if _FINAL_RE.search(text):
        return ParsedAction(thought=thought, is_final=True)

    action_match = _ACTION_RE.search(text)
    if not action_match:
        return ParsedAction(
            thought=thought,
            parse_error="No ACTION: or FINAL_ANSWER: line found.",
        )

    name = action_match.group("name")
    args_match = _ARGS_RE.search(text)
    if not args_match:
        return ParsedAction(thought=thought, name=name, args={})

    raw = args_match.group("args").strip()
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("ARGS must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return ParsedAction(
            thought=thought,
            name=name,
            parse_error=f"Could not parse ARGS as JSON ({exc}). Received: {raw[:200]}",
        )
    return ParsedAction(thought=thought, name=name, args=args)


def reference_plan(task: Task) -> str:
    """The stage sequence a correct solution must execute -- injected by PLAN_REPAIR."""
    if not task.required_stages:
        return "Work through the calculation stage by stage."
    stages = "\n".join(f"  {i + 1}. {stage}" for i, stage in enumerate(task.required_stages))
    return (
        "Follow exactly this sequence of stages, in order, using the matching tools:\n"
        f"{stages}\n"
        "Do not skip a stage and do not reorder them."
    )


def convention_reminder(task: Task) -> str:
    """Injected by CONVENTION_CORRECT -- restates the required reporting convention."""
    answer = task.answer
    return (
        "Before answering, check your reporting convention. The required answer is in "
        f"{answer.unit}, on a {answer.basis} basis, for the {answer.cell} cell, in the "
        f"{answer.state} state. Convert your computed value if it is not already in "
        "those terms."
    )
