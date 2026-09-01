"""The tools an agent may call, and the sandbox they run in.

Two design rules govern this file.

**Never leak the answer.** `mp_get_structure` deliberately strips
`formation_energy_per_atom`, `band_gap`, `density` and friends from what the agent sees.
If the retrieval tool handed back the DFT value, every task would collapse into a lookup
and the benchmark would measure nothing. `REDACTED_FIELDS` is that boundary, and it is
enforced in one place on purpose.

**Never silently normalise.** Tools report the unit and basis of everything they return,
and they never quietly convert between per-atom and per-cell. The agent has to get the
bookkeeping right itself -- which is precisely the behaviour under measurement.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..schema import ToolResult

# Fields an agent must never see: these are (or trivially give away) ground truth.
REDACTED_FIELDS = frozenset(
    {
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "density",
        "bulk_modulus",
        "shear_modulus",
        "uncorrected_energy_per_atom",
        "energy_per_atom",
    }
)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]
    stage: str | None = None

    def render(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({args})\n    {self.description}"


@dataclass
class Session:
    """Mutable workspace shared across one trajectory."""

    structures: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    _counter: int = 0

    def put(self, structure: Any, label: str | None = None) -> str:
        handle = label or f"s{self._counter}"
        self._counter += 1
        self.structures[handle] = structure
        return handle

    def get(self, handle: str) -> Any:
        if handle not in self.structures:
            raise KeyError(
                f"No structure named {handle!r}. Available: {sorted(self.structures) or 'none'}"
            )
        return self.structures[handle]


# --------------------------------------------------------------------------------------
# A deliberately small arithmetic evaluator
# --------------------------------------------------------------------------------------

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pi": lambda: math.pi,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic only. No names, no attributes, no imports, no calls out."""

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Only numeric literals allowed, got {node.value!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [_eval(e) for e in node.elts]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("Only abs, round, min, max, sum, sqrt, log, exp are available.")
            return _ALLOWED_FUNCS[node.func.id](*[_eval(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")

    return float(_eval(ast.parse(expression, mode="eval")))


# --------------------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------------------


class ToolRegistry:
    """Binds tool names to implementations for one trajectory."""

    def __init__(self, session: Session, mp_source: Any | None = None, *, allow_sim: bool = True):
        self.session = session
        self.mp = mp_source
        self.allow_sim = allow_sim
        self._impls: dict[str, Callable[..., str]] = {
            "mp_get_structure": self._mp_get_structure,
            "structure_info": self._structure_info,
            "make_primitive": self._make_primitive,
            "make_conventional": self._make_conventional,
            "make_supercell": self._make_supercell,
            "relax_structure": self._relax_structure,
            "compute_energy": self._compute_energy,
            "equation_of_state": self._equation_of_state,
            "make_vacancy": self._make_vacancy,
            "make_slab": self._make_slab,
            "lookup_reference_energies": self._lookup_reference_energies,
            "python_eval": self._python_eval,
        }

    # -- dispatch ----------------------------------------------------------------------

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        impl = self._impls.get(name)
        if impl is None:
            return ToolResult(
                ok=False,
                content="",
                error=f"Unknown tool {name!r}. Available: {', '.join(sorted(self._impls))}",
                duration_ms=0.0,
            )
        try:
            content = impl(**args)
            ok, error = True, None
        except TypeError as exc:
            content, ok, error = "", False, f"Bad arguments for {name}: {exc}"
        except Exception as exc:
            content, ok, error = "", False, f"{type(exc).__name__}: {exc}"
        return ToolResult(
            ok=ok,
            content=content,
            error=error,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def specs(self) -> list[ToolSpec]:
        return [s for s in TOOL_SPECS if s.name in self._impls]

    # -- retrieval ---------------------------------------------------------------------

    def _mp_get_structure(self, material_id: str) -> str:
        if self.mp is None:
            raise RuntimeError("Materials Project source unavailable (MP_API_KEY not set).")
        records = self.mp.get([material_id])
        if not records:
            raise KeyError(f"No Materials Project entry for {material_id!r}.")
        record = records[0]
        structure = record.structure()
        handle = self.session.put(structure)
        self.session.notes.setdefault("fetched", []).append(material_id)

        # Ground-truth-bearing fields are withheld by design.
        return json.dumps(
            {
                "handle": handle,
                "material_id": record.material_id,
                "formula": record.formula,
                "n_sites": record.n_sites,
                "volume_angstrom3": round(record.volume, 4),
                "spacegroup": record.spacegroup_symbol,
                "crystal_system": record.crystal_system,
                "cell_type": "as-stored (MP primitive-ish); call make_primitive or "
                "make_conventional if the task specifies a cell",
                "withheld": sorted(REDACTED_FIELDS),
            },
            indent=2,
        )

    def _structure_info(self, handle: str) -> str:
        structure = self.session.get(handle)
        composition = structure.composition
        _, n_fu = composition.get_reduced_composition_and_factor()
        return json.dumps(
            {
                "handle": handle,
                "formula": composition.formula,
                "reduced_formula": composition.reduced_formula,
                "n_atoms": len(structure),
                "n_formula_units": int(round(n_fu)),
                "volume_angstrom3": round(float(structure.volume), 4),
                "lattice_abc_angstrom": [round(x, 5) for x in structure.lattice.abc],
                "lattice_angles_deg": [round(x, 3) for x in structure.lattice.angles],
            },
            indent=2,
        )

    # -- cell manipulation --------------------------------------------------------------

    def _make_primitive(self, handle: str) -> str:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = self.session.get(handle)
        primitive = SpacegroupAnalyzer(structure, symprec=0.1).get_primitive_standard_structure()
        new_handle = self.session.put(primitive)
        return json.dumps(
            {"handle": new_handle, "n_atoms": len(primitive), "cell": "primitive"}, indent=2
        )

    def _make_conventional(self, handle: str) -> str:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = self.session.get(handle)
        conventional = SpacegroupAnalyzer(
            structure, symprec=0.1
        ).get_conventional_standard_structure()
        new_handle = self.session.put(conventional)
        return json.dumps(
            {"handle": new_handle, "n_atoms": len(conventional), "cell": "conventional"}, indent=2
        )

    def _make_supercell(self, handle: str, scaling: list[int]) -> str:
        structure = self.session.get(handle).copy()
        structure.make_supercell(list(scaling))
        new_handle = self.session.put(structure)
        return json.dumps(
            {"handle": new_handle, "n_atoms": len(structure), "cell": "supercell"}, indent=2
        )

    # -- simulation ---------------------------------------------------------------------

    def _guard_sim(self) -> None:
        if not self.allow_sim:
            raise RuntimeError("Simulation tools are disabled in this configuration.")

    def _relax_structure(self, handle: str, fmax: float = 0.05, max_steps: int = 300) -> str:
        self._guard_sim()
        from ..sim import relax

        result = relax(self.session.get(handle), fmax=fmax, max_steps=max_steps)
        new_handle = self.session.put(result.final_structure)
        return json.dumps(
            {
                "handle": new_handle,
                "state": "relaxed",
                "n_atoms": result.n_atoms,
                "energy_per_atom_eV": round(result.energy_per_atom, 6),
                "energy_total_eV": round(result.energy_total, 6),
                "n_ionic_steps": result.n_steps,
                "converged": result.converged,
                "max_force_eV_per_angstrom": (
                    round(result.max_force, 5) if result.max_force is not None else None
                ),
                "note": "energy_per_atom_eV is per ATOM, not per formula unit or per cell",
            },
            indent=2,
        )

    def _compute_energy(self, handle: str) -> str:
        self._guard_sim()
        from ..sim import energy_per_atom, total_energy

        structure = self.session.get(handle)
        return json.dumps(
            {
                "handle": handle,
                "state": "as-provided (this is a single-point energy; it does NOT relax)",
                "n_atoms": len(structure),
                "energy_per_atom_eV": round(energy_per_atom(structure), 6),
                "energy_total_eV": round(total_energy(structure), 6),
            },
            indent=2,
        )

    def _equation_of_state(self, handle: str, relax_ions: bool = False) -> str:
        self._guard_sim()
        from ..sim import equation_of_state

        result = equation_of_state(self.session.get(handle), relax_ions=relax_ions)
        return json.dumps(
            {
                "handle": handle,
                "bulk_modulus_GPa": round(result.bulk_modulus_gpa, 4),
                "v0_angstrom3": round(result.v0, 4),
                "e0_eV": round(result.e0, 6),
                "volumes_angstrom3": [round(v, 4) for v in result.volumes],
                "energies_eV": [round(e, 6) for e in result.energies],
                "note": "energies are TOTAL cell energies in eV",
            },
            indent=2,
        )

    def _make_vacancy(
        self, handle: str, site_index: int = 0, supercell: list[int] | None = None
    ) -> str:
        self._guard_sim()
        from ..sim import make_vacancy

        cell = make_vacancy(
            self.session.get(handle),
            site_index=site_index,
            supercell=tuple(supercell or (2, 2, 2)),
        )
        new_handle = self.session.put(cell)
        return json.dumps(
            {"handle": new_handle, "n_atoms": len(cell), "defect": "vacancy"}, indent=2
        )

    def _make_slab(self, handle: str, miller_index: list[int] | None = None) -> str:
        self._guard_sim()
        from ..sim import make_slab

        slab = make_slab(self.session.get(handle), miller_index=tuple(miller_index or (1, 0, 0)))
        new_handle = self.session.put(slab)
        return json.dumps(
            {
                "handle": new_handle,
                "n_atoms": len(slab),
                "surface_area_angstrom2": round(float(slab.surface_area), 4),
            },
            indent=2,
        )

    def _lookup_reference_energies(self, elements: list[str]) -> str:
        """Elemental reference energies, needed for any formation-energy workflow."""
        self._guard_sim()
        from ..sim import energy_per_atom

        if self.mp is None:
            raise RuntimeError("Materials Project source unavailable (MP_API_KEY not set).")

        out: dict[str, float] = {}
        for element in elements:
            records = self.mp.search(elements=[element], num_elements=(1, 1), max_sites=8, limit=1)
            if not records:
                raise KeyError(f"No elemental reference structure found for {element!r}.")
            out[element] = round(energy_per_atom(records[0].structure()), 6)

        return json.dumps(
            {
                "reference_energies_eV_per_atom": out,
                "note": "Computed with the same potential as compute_energy, so they are "
                "consistent with each other. Do not mix with values from another source.",
            },
            indent=2,
        )

    def _python_eval(self, expression: str) -> str:
        value = safe_eval(expression)
        return json.dumps({"expression": expression, "value": value}, indent=2)


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "mp_get_structure",
        "Fetch a crystal structure from the Materials Project by id (e.g. 'mp-149'). "
        "Returns a handle. DFT property values are withheld.",
        {"material_id": "str"},
        stage="fetch_structure",
    ),
    ToolSpec(
        "structure_info",
        "Composition, atom count, formula-unit count, volume and lattice of a handle.",
        {"handle": "str"},
    ),
    ToolSpec(
        "make_primitive",
        "Convert to the standardised primitive cell. Returns a new handle.",
        {"handle": "str"},
        stage="cell_setup",
    ),
    ToolSpec(
        "make_conventional",
        "Convert to the standardised conventional cell. Returns a new handle.",
        {"handle": "str"},
        stage="cell_setup",
    ),
    ToolSpec(
        "make_supercell",
        "Repeat the cell, e.g. scaling=[2,2,2]. Returns a new handle.",
        {"handle": "str", "scaling": "list[int]"},
        stage="cell_setup",
    ),
    ToolSpec(
        "relax_structure",
        "Geometry-optimise with the CHGNet potential (relaxes ions AND cell). "
        "Returns a new handle plus energies.",
        {"handle": "str", "fmax": "float = 0.05", "max_steps": "int = 300"},
        stage="relax",
    ),
    ToolSpec(
        "compute_energy",
        "Single-point energy of a structure exactly as given. Does NOT relax.",
        {"handle": "str"},
        stage="compute_energy",
    ),
    ToolSpec(
        "equation_of_state",
        "Energy-volume scan and quadratic fit. Returns the bulk modulus in GPa.",
        {"handle": "str", "relax_ions": "bool = false"},
        stage="eos",
    ),
    ToolSpec(
        "make_vacancy",
        "Build a supercell and remove one site. Returns a new handle.",
        {"handle": "str", "site_index": "int = 0", "supercell": "list[int] = [2,2,2]"},
        stage="defect",
    ),
    ToolSpec(
        "make_slab",
        "Cleave a surface with the given Miller index. Returns a new handle.",
        {"handle": "str", "miller_index": "list[int] = [1,0,0]"},
        stage="surface",
    ),
    ToolSpec(
        "lookup_reference_energies",
        "Elemental reference energies in eV/atom, consistent with compute_energy. "
        "Required for formation energies.",
        {"elements": "list[str]"},
        stage="reference_energies",
    ),
    ToolSpec(
        "python_eval",
        "Evaluate an arithmetic expression. Numbers and + - * / ** ( ) abs round min "
        "max sum sqrt log exp only.",
        {"expression": "str"},
        stage="analysis",
    ),
]
