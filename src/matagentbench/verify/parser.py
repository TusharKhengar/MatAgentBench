"""Extracting the agent's final answer from its last message.

Deliberately forgiving. A strict JSON-only parser would file every formatting slip
under NO_ANSWER and inflate that class at the expense of the numeric ones -- which
would quietly corrupt the headline failure taxonomy. We parse the strict contract
first, then fall back, and record which path was taken.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schema import Basis, CellChoice, Quantity, RelaxState
from ..units import canonical_unit

ANSWER_TAG = "FINAL_ANSWER"

_JSON_BLOCK = re.compile(
    rf"{ANSWER_TAG}\s*:?\s*(?P<json>\{{.*?\}})",
    re.IGNORECASE | re.DOTALL,
)
_LOOSE_TAGGED = re.compile(
    rf"{ANSWER_TAG}\s*:?\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[^\n,;]*)",
    re.IGNORECASE,
)
_ANY_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_BASIS_HINTS: list[tuple[re.Pattern[str], Basis]] = [
    (re.compile(r"per[\s_-]*atom|/\s*atom|atom\^?-1", re.I), "per_atom"),
    (re.compile(r"per[\s_-]*(formula[\s_-]*unit|f\.?u\.?)|/\s*f\.?u\.?", re.I), "per_formula_unit"),
    (re.compile(r"per[\s_-]*cell|/\s*cell|total", re.I), "per_cell"),
]
_CELL_HINTS: list[tuple[re.Pattern[str], CellChoice]] = [
    (re.compile(r"conventional", re.I), "conventional"),
    (re.compile(r"primitive", re.I), "primitive"),
    (re.compile(r"supercell", re.I), "supercell"),
]
_STATE_HINTS: list[tuple[re.Pattern[str], RelaxState]] = [
    (re.compile(r"unrelaxed|as[\s-]*is|initial|unoptimi[sz]ed", re.I), "unrelaxed"),
    (re.compile(r"relaxed|optimi[sz]ed|converged", re.I), "relaxed"),
]


class ParsedAnswer:
    __slots__ = ("quantity", "method", "raw")

    def __init__(self, quantity: Quantity | None, method: str, raw: str | None):
        self.quantity = quantity
        self.method = method  # "json" | "tagged" | "trailing_number" | "none"
        self.raw = raw

    def __bool__(self) -> bool:
        return self.quantity is not None


def _sniff(text: str, hints: list[tuple[re.Pattern[str], Any]], default: Any) -> Any:
    for pattern, label in hints:
        if pattern.search(text):
            return label
    return default


def _quantity_from_payload(payload: dict[str, Any], context: str) -> Quantity | None:
    if "value" not in payload:
        return None
    try:
        value = float(payload["value"])
    except (TypeError, ValueError):
        return None

    unit = canonical_unit(str(payload.get("unit", "")))
    blob = f"{unit} {context}"
    return Quantity(
        value=value,
        unit=unit,
        basis=payload.get("basis") or _sniff(blob, _BASIS_HINTS, "per_atom"),
        cell=payload.get("cell") or _sniff(blob, _CELL_HINTS, "unspecified"),
        state=payload.get("state") or _sniff(blob, _STATE_HINTS, "unspecified"),
    )


def parse_final_answer(text: str | None) -> ParsedAnswer:
    if not text or not text.strip():
        return ParsedAnswer(None, "none", text)

    # 1. The declared contract: FINAL_ANSWER: {...}
    match = _JSON_BLOCK.search(text)
    if match:
        blob = match.group("json")
        try:
            payload = json.loads(blob)
            if isinstance(payload, dict):
                quantity = _quantity_from_payload(payload, text)
                if quantity is not None:
                    return ParsedAnswer(quantity, "json", blob)
        except json.JSONDecodeError:
            pass

    # 2. Tagged but unstructured: FINAL_ANSWER: -3.42 eV/atom
    match = _LOOSE_TAGGED.search(text)
    if match:
        try:
            value = float(match.group("value"))
        except ValueError:
            value = None
        if value is not None:
            unit_text = match.group("unit").strip()
            return ParsedAnswer(
                Quantity(
                    value=value,
                    unit=canonical_unit(unit_text),
                    basis=_sniff(f"{unit_text} {text}", _BASIS_HINTS, "per_atom"),
                    cell=_sniff(text, _CELL_HINTS, "unspecified"),
                    state=_sniff(text, _STATE_HINTS, "unspecified"),
                ),
                "tagged",
                match.group(0),
            )

    # 3. Last resort: the final number in the message. Recorded as a weak parse so
    #    the taxonomy can separate "no answer" from "answer we had to guess at".
    numbers = _ANY_NUMBER.findall(text)
    if numbers:
        try:
            value = float(numbers[-1])
        except ValueError:
            return ParsedAnswer(None, "none", text)
        return ParsedAnswer(
            Quantity(
                value=value,
                unit=canonical_unit(""),
                basis=_sniff(text, _BASIS_HINTS, "per_atom"),
                cell=_sniff(text, _CELL_HINTS, "unspecified"),
                state=_sniff(text, _STATE_HINTS, "unspecified"),
            ),
            "trailing_number",
            numbers[-1],
        )

    return ParsedAnswer(None, "none", text)
