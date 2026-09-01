"""Ground-truth sources.

Materials Project is primary: free API key, CC-BY-4.0, and DFT values for ~150k
materials. JARVIS-DFT (NIST, public domain) is the cross-check -- when two independent
DFT databases agree on a value, a task built from it is far less likely to be scoring
the agent against a database artefact.

Everything is cached to disk on first fetch. Task generation is then reproducible
offline, and CI never hammers the MP API.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import CellContext, Provenance

MP_SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "nsites",
    "volume",
    "density",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "symmetry",
    "structure",
]


def _cache_dir(sub: str) -> Path:
    root = Path(os.getenv("MAB_CACHE_DIR", ".mab_cache")) / sub
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _require(module: str, extra: str) -> Any:
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f'{module!r} is required here. Install it with:\n    pip install -e ".[{extra}]"'
        ) from exc


# --------------------------------------------------------------------------------------
# Cell bookkeeping -- what the reconciler needs to name a basis/cell error
# --------------------------------------------------------------------------------------


def build_cell_context(structure: Any) -> CellContext:
    """Derive primitive/conventional atom and formula-unit counts from a structure.

    Computed once at task-generation time so that diagnosing a failure never requires
    re-running symmetry analysis, let alone a simulation.
    """
    symmetry = _require("pymatgen.symmetry.analyzer", "data")
    analyzer = symmetry.SpacegroupAnalyzer(structure, symprec=0.1)

    try:
        primitive = analyzer.get_primitive_standard_structure()
    except Exception:
        primitive = structure
    try:
        conventional = analyzer.get_conventional_standard_structure()
    except Exception:
        conventional = structure

    _, n_fu_prim = primitive.composition.get_reduced_composition_and_factor()
    _, n_fu_conv = conventional.composition.get_reduced_composition_and_factor()

    return CellContext(
        n_atoms_primitive=len(primitive),
        n_atoms_conventional=len(conventional),
        n_formula_units_primitive=int(round(n_fu_prim)),
        n_formula_units_conventional=int(round(n_fu_conv)),
        volume_primitive=float(primitive.volume),
        volume_conventional=float(conventional.volume),
        reduced_formula=structure.composition.reduced_formula,
    )


# --------------------------------------------------------------------------------------
# Materials Project
# --------------------------------------------------------------------------------------


@dataclass
class MaterialRecord:
    material_id: str
    formula: str
    n_sites: int
    volume: float
    density: float
    formation_energy_per_atom: float | None
    energy_above_hull: float | None
    band_gap: float | None
    spacegroup_symbol: str | None
    crystal_system: str | None
    structure_dict: dict[str, Any]

    def structure(self) -> Any:
        core = _require("pymatgen.core", "data")
        return core.Structure.from_dict(self.structure_dict)

    def provenance(self) -> Provenance:
        return Provenance(
            source="materials_project",
            source_id=self.material_id,
            url=f"https://next-gen.materialsproject.org/materials/{self.material_id}",
            license="CC-BY-4.0",
        )


class MaterialsProjectSource:
    def __init__(self, api_key: str | None = None, use_cache: bool = True):
        self.api_key = api_key or os.getenv("MP_API_KEY", "")
        self.use_cache = use_cache
        self.cache = _cache_dir("mp")
        if not self.api_key:
            raise RuntimeError(
                "MP_API_KEY is not set.\n"
                "Get a free key (no card) at https://next-gen.materialsproject.org/api\n"
                "then put it in .env, or add it as a GitHub Actions secret."
            )

    def _cached(self, key: str) -> Any | None:
        if not self.use_cache:
            return None
        path = self.cache / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _store(self, key: str, payload: Any) -> None:
        try:
            (self.cache / f"{key}.json").write_text(
                json.dumps(payload, default=str), encoding="utf-8"
            )
        except OSError:
            pass

    def _records_from_payload(self, payload: list[dict[str, Any]]) -> list[MaterialRecord]:
        out: list[MaterialRecord] = []
        for doc in payload:
            symmetry = doc.get("symmetry") or {}
            out.append(
                MaterialRecord(
                    material_id=str(doc["material_id"]),
                    formula=doc.get("formula_pretty", ""),
                    n_sites=int(doc.get("nsites", 0)),
                    volume=float(doc.get("volume", 0.0)),
                    density=float(doc.get("density", 0.0)),
                    formation_energy_per_atom=doc.get("formation_energy_per_atom"),
                    energy_above_hull=doc.get("energy_above_hull"),
                    band_gap=doc.get("band_gap"),
                    spacegroup_symbol=symmetry.get("symbol"),
                    crystal_system=str(symmetry.get("crystal_system") or "") or None,
                    structure_dict=doc["structure"],
                )
            )
        return out

    def search(
        self,
        *,
        elements: list[str] | None = None,
        num_elements: tuple[int, int] | None = None,
        max_sites: int = 12,
        energy_above_hull_max: float = 0.0,
        limit: int = 100,
    ) -> list[MaterialRecord]:
        """Query summary docs, filtered to cells small enough for CHGNet to be quick.

        `energy_above_hull_max=0.0` restricts to thermodynamically stable phases, which
        keeps ground truth clean and keeps polymorph-ranking tasks well posed.
        """
        key = _cache_key("search", elements, num_elements, max_sites, energy_above_hull_max, limit)
        if (hit := self._cached(key)) is not None:
            return self._records_from_payload(hit)

        mp_api = _require("mp_api.client", "data")
        query: dict[str, Any] = {
            "num_sites": (1, max_sites),
            "energy_above_hull": (0.0, energy_above_hull_max),
            "fields": MP_SUMMARY_FIELDS,
        }
        if elements:
            query["elements"] = elements
        if num_elements:
            query["num_elements"] = num_elements

        with mp_api.MPRester(self.api_key) as mpr:
            docs = mpr.materials.summary.search(**query)

        payload = []
        for doc in docs[:limit]:
            raw = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
            structure = raw.get("structure")
            if structure is None:
                continue
            raw["structure"] = structure.as_dict() if hasattr(structure, "as_dict") else structure
            symmetry = raw.get("symmetry")
            if symmetry is not None and hasattr(symmetry, "model_dump"):
                raw["symmetry"] = symmetry.model_dump()
            payload.append({k: raw.get(k) for k in MP_SUMMARY_FIELDS})

        self._store(key, payload)
        return self._records_from_payload(payload)

    def get(self, material_ids: list[str]) -> list[MaterialRecord]:
        key = _cache_key("get", sorted(material_ids))
        if (hit := self._cached(key)) is not None:
            return self._records_from_payload(hit)

        mp_api = _require("mp_api.client", "data")
        with mp_api.MPRester(self.api_key) as mpr:
            docs = mpr.materials.summary.search(material_ids=material_ids, fields=MP_SUMMARY_FIELDS)

        payload = []
        for doc in docs:
            raw = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
            structure = raw.get("structure")
            if structure is None:
                continue
            raw["structure"] = structure.as_dict() if hasattr(structure, "as_dict") else structure
            symmetry = raw.get("symmetry")
            if symmetry is not None and hasattr(symmetry, "model_dump"):
                raw["symmetry"] = symmetry.model_dump()
            payload.append({k: raw.get(k) for k in MP_SUMMARY_FIELDS})

        self._store(key, payload)
        return self._records_from_payload(payload)

    def bulk_moduli(self, material_ids: list[str]) -> dict[str, float]:
        """Voigt-Reuss-Hill bulk modulus in GPa, for the elasticity task family."""
        key = _cache_key("elasticity", sorted(material_ids))
        if (hit := self._cached(key)) is not None:
            return {k: float(v) for k, v in hit.items()}

        mp_api = _require("mp_api.client", "data")
        with mp_api.MPRester(self.api_key) as mpr:
            docs = mpr.materials.elasticity.search(
                material_ids=material_ids, fields=["material_id", "bulk_modulus"]
            )

        out: dict[str, float] = {}
        for doc in docs:
            raw = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
            modulus = raw.get("bulk_modulus") or {}
            vrh = modulus.get("vrh") if isinstance(modulus, dict) else None
            if vrh is not None:
                out[str(raw["material_id"])] = float(vrh)

        self._store(key, out)
        return out


# --------------------------------------------------------------------------------------
# JARVIS-DFT cross-check (NIST, public domain, no key)
# --------------------------------------------------------------------------------------


class JarvisSource:
    """Optional second opinion on formation energies.

    The `dft_3d` dataset is a few hundred MB on first download and is cached by
    jarvis-tools thereafter. Used only during task generation, never at eval time.
    """

    def __init__(self) -> None:
        self._data: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._data is None:
            figshare = _require("jarvis.db.figshare", "data")
            self._data = figshare.data("dft_3d")
        return self._data

    def formation_energy_by_formula(self, reduced_formula: str) -> float | None:
        """eV/atom, or None if JARVIS has no entry for this composition."""
        try:
            entries = [
                d
                for d in self._load()
                if d.get("formula") == reduced_formula
                and d.get("formation_energy_peratom") is not None
            ]
        except ImportError:
            return None
        if not entries:
            return None
        return float(
            min(entries, key=lambda d: d["formation_energy_peratom"])["formation_energy_peratom"]
        )


def cross_check(mp_value: float, jarvis_value: float | None, tolerance: float = 0.1) -> bool:
    """True if the two databases agree to within `tolerance` (eV/atom).

    Disagreement is not necessarily an error -- different functionals and corrections --
    but it is a reason to exclude the material from the task set rather than score an
    agent against a contested number.
    """
    if jarvis_value is None:
        return True
    return abs(mp_value - jarvis_value) <= tolerance
