"""ML-interatomic-potential workflows -- our stand-in for DFT.

CHGNet relaxes a small unit cell in seconds on a laptop GPU, where the DFT calculation
that produced our ground truth took CPU-hours on a cluster. That gap is the entire
reason this benchmark is runnable for free.

Unit discipline is enforced here rather than assumed. CHGNet returns energy **per atom
in eV**; almost every silent failure this benchmark measures begins with someone
forgetting that. Every function below states its unit and basis in the return value.

Heavy imports are deferred so that the schema, verifier and site tooling stay importable
on a machine with no torch installed -- which is what lets the CI site job skip a 2 GB
dependency tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..units import EV_PER_ANG3_TO_GPA

DEFAULT_FMAX = 0.05  # eV/Angstrom
DEFAULT_MAX_STEPS = 300


@dataclass
class RelaxResult:
    final_structure: Any
    energy_per_atom: float  # eV/atom
    energy_total: float  # eV, whole cell
    n_atoms: int
    n_steps: int
    converged: bool
    initial_energy_per_atom: float | None = None
    max_force: float | None = None

    @property
    def relaxation_delta_per_atom(self) -> float | None:
        if self.initial_energy_per_atom is None:
            return None
        return self.energy_per_atom - self.initial_energy_per_atom


@dataclass
class EOSResult:
    volumes: list[float]  # Angstrom^3, whole cell
    energies: list[float]  # eV, whole cell
    v0: float  # Angstrom^3
    e0: float  # eV
    bulk_modulus_gpa: float


def _require(module: str, extra: str) -> Any:
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"{module!r} is required for this operation. Install it with:\n"
            f'    pip install -e ".[{extra}]"'
        ) from exc


@lru_cache(maxsize=1)
def get_model(use_device: str | None = None) -> Any:
    """Load CHGNet once per process. First call downloads ~50 MB of weights."""
    chgnet_model = _require("chgnet.model", "sim")
    return chgnet_model.CHGNet.load(use_device=use_device)


@lru_cache(maxsize=1)
def get_relaxer(use_device: str | None = None) -> Any:
    chgnet_model = _require("chgnet.model", "sim")
    return chgnet_model.StructOptimizer(use_device=use_device)


def energy_per_atom(structure: Any) -> float:
    """Single-point energy. Returns **eV/atom** -- not eV, and not per formula unit."""
    prediction = get_model().predict_structure(structure)
    return float(prediction["e"])


def total_energy(structure: Any) -> float:
    """Single-point energy of the whole cell as given. Returns **eV**."""
    return energy_per_atom(structure) * len(structure)


def relax(
    structure: Any,
    fmax: float = DEFAULT_FMAX,
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    relax_cell: bool = True,
) -> RelaxResult:
    """Geometry optimisation with CHGNet + FIRE."""
    initial = energy_per_atom(structure)
    relaxer = get_relaxer()
    result = relaxer.relax(
        structure, fmax=fmax, steps=max_steps, relax_cell=relax_cell, verbose=False
    )

    final_structure = result["final_structure"]
    trajectory = result.get("trajectory")

    n_steps = len(getattr(trajectory, "energies", []) or [])
    final_energy_total = None
    max_force = None
    if trajectory is not None:
        energies = getattr(trajectory, "energies", None)
        if energies is not None and len(energies):
            final_energy_total = float(energies[-1])
        forces = getattr(trajectory, "forces", None)
        if forces is not None and len(forces):
            max_force = float(np.abs(np.asarray(forces[-1])).max())

    n_atoms = len(final_structure)
    if final_energy_total is None:
        final_energy_total = energy_per_atom(final_structure) * n_atoms

    return RelaxResult(
        final_structure=final_structure,
        energy_per_atom=final_energy_total / n_atoms,
        energy_total=final_energy_total,
        n_atoms=n_atoms,
        n_steps=n_steps,
        converged=(max_force is not None and max_force <= fmax),
        initial_energy_per_atom=initial,
        max_force=max_force,
    )


def equation_of_state(
    structure: Any,
    strains: tuple[float, ...] = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06),
    *,
    relax_ions: bool = False,
) -> EOSResult:
    """Energy-volume curve, fitted for the bulk modulus.

    A quadratic fit around the minimum gives B0 = V0 * (d2E/dV2), which is accurate
    enough at these strains and -- unlike a Birch-Murnaghan fit -- needs only numpy.

    Returns bulk modulus in **GPa**; energies are total-cell **eV** and volumes are
    whole-cell **Angstrom^3**.
    """
    volumes: list[float] = []
    energies: list[float] = []

    for strain in strains:
        scaled = structure.copy()
        scaled.scale_lattice(structure.volume * (1.0 + strain))
        if relax_ions:
            energy = relax(scaled, relax_cell=False).energy_total
        else:
            energy = total_energy(scaled)
        volumes.append(float(scaled.volume))
        energies.append(float(energy))

    v = np.asarray(volumes)
    e = np.asarray(energies)
    coeffs = np.polyfit(v, e, 2)  # e = a v^2 + b v + c
    a, b, _c = coeffs
    if a <= 0:
        raise ValueError("E(V) curve is not convex; cannot extract a bulk modulus.")

    v0 = -b / (2 * a)
    e0 = float(np.polyval(coeffs, v0))
    d2e_dv2 = 2 * a
    bulk_modulus = float(v0 * d2e_dv2 * EV_PER_ANG3_TO_GPA)

    return EOSResult(
        volumes=volumes,
        energies=energies,
        v0=float(v0),
        e0=e0,
        bulk_modulus_gpa=bulk_modulus,
    )


def formation_energy_per_atom(
    structure: Any,
    reference_energies_per_atom: dict[str, float],
) -> float:
    """E_f = (E_cell - sum_i n_i * mu_i) / N_atoms, in **eV/atom**.

    `reference_energies_per_atom` maps element symbol -> elemental reference energy in
    eV/atom. Supplying MP's fitted elemental references rather than raw CHGNet elemental
    energies matters: mixing the two is a real and easy way to be wrong by ~0.1 eV/atom.
    """
    composition = structure.composition
    missing = [str(el) for el in composition.elements if str(el) not in reference_energies_per_atom]
    if missing:
        raise KeyError(f"No reference energy supplied for: {', '.join(sorted(missing))}")

    cell_energy = total_energy(structure)
    reference_total = sum(
        composition[el] * reference_energies_per_atom[str(el)] for el in composition.elements
    )
    return float((cell_energy - reference_total) / len(structure))


def make_vacancy(
    structure: Any, site_index: int = 0, supercell: tuple[int, int, int] = (2, 2, 2)
) -> Any:
    """Remove one site from a supercell. Returns the defected structure."""
    cell = structure.copy()
    cell.make_supercell(list(supercell))
    if not 0 <= site_index < len(cell):
        raise IndexError(f"site_index {site_index} out of range for {len(cell)} sites")
    cell.remove_sites([site_index])
    return cell


def make_slab(
    structure: Any,
    miller_index: tuple[int, int, int] = (1, 0, 0),
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 12.0,
) -> Any:
    """Cleave a surface. Returns the first (lowest-index) termination."""
    surface = _require("pymatgen.core.surface", "data")
    generator = surface.SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=True,
    )
    slabs = generator.get_slabs()
    if not slabs:
        raise ValueError(f"No slab could be generated for miller index {miller_index}")
    return slabs[0]
