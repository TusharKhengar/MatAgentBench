"""Convention reconciliation: deciding *why* a number is wrong.

A plain numeric check tells you an agent failed. It does not tell you whether the agent
botched the physics or botched the reporting -- and in atomistic workflows those are
overwhelmingly different failures with different fixes.

The reconciler searches a closed set of known convention transforms (unit, basis, cell,
sign) and asks: does exactly one of them carry the agent's number onto ground truth? If
so, the simulation was right and the bookkeeping was wrong, and we can name which
bookkeeping. This is mechanical, not a judgement call -- no LLM judge involved.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    CellContext,
    FailureClass,
    Quantity,
    Reconciliation,
    Tolerance,
)

# --------------------------------------------------------------------------------------
# Physical constants (CODATA-ish; benchmark tolerances are far looser than the last digit)
# --------------------------------------------------------------------------------------

EV_TO_KJ_PER_MOL = 96.48533212
EV_TO_KCAL_PER_MOL = 23.060548
EV_TO_HARTREE = 0.0367493224
EV_TO_RYDBERG = 0.0734986448
EV_TO_MEV = 1000.0
EV_TO_JOULE = 1.602176634e-19

EV_PER_ANG3_TO_GPA = 160.2176634
ANG_TO_BOHR = 1.8897261246
ANG3_TO_BOHR3 = ANG_TO_BOHR**3

# Unit aliases -> canonical name, so "eV/atom", "ev per atom" and "eV atom^-1" agree.
UNIT_ALIASES: dict[str, str] = {
    "ev": "eV",
    "ev/atom": "eV/atom",
    "ev per atom": "eV/atom",
    "ev/at": "eV/atom",
    "ev atom^-1": "eV/atom",
    "ev/f.u.": "eV/formula_unit",
    "ev/fu": "eV/formula_unit",
    "ev per formula unit": "eV/formula_unit",
    "mev": "meV",
    "mev/atom": "meV/atom",
    "kj/mol": "kJ/mol",
    "kjmol": "kJ/mol",
    "kcal/mol": "kcal/mol",
    "hartree": "Hartree",
    "ha": "Hartree",
    "ry": "Rydberg",
    "rydberg": "Rydberg",
    "gpa": "GPa",
    "mbar": "Mbar",
    "kbar": "kbar",
    "pa": "Pa",
    "angstrom": "Angstrom",
    "ang": "Angstrom",
    "a": "Angstrom",
    "å": "Angstrom",
    "angstrom^3": "Angstrom^3",
    "ang^3": "Angstrom^3",
    "å^3": "Angstrom^3",
    "bohr": "Bohr",
    "g/cm^3": "g/cm^3",
    "g/cc": "g/cm^3",
    "kg/m^3": "kg/m^3",
}


def canonical_unit(unit: str | None) -> str:
    if not unit:
        return ""
    key = unit.strip().lower().replace(" ", "").replace("·", "")
    if key in UNIT_ALIASES:
        return UNIT_ALIASES[key]
    return UNIT_ALIASES.get(unit.strip().lower(), unit.strip())


# A composite of two convention errors must match far more tightly than a single one.
# Two independent factor families produce a dense enough set that some product lands
# within an ordinary tolerance of almost any number by chance -- which would let the
# reconciler "explain" genuinely wrong physics and inflate the silent-failure rate.
# Single slips are also simply much more likely than simultaneous double slips.
COMPOSITE_MAX_REL = 0.01


@dataclass(frozen=True)
class Transform:
    """A candidate multiplicative correction, with the story of what it means."""

    factor: float
    label: str
    failure_class: FailureClass
    explanation: str
    components: int = 1  # 1 = a single convention slip, 2 = two compounded


def _energy_unit_transforms() -> list[Transform]:
    """Unit slips: correct magnitude in the wrong energy unit."""
    pairs = [
        (EV_TO_KJ_PER_MOL, "kJ/mol", "kJ/mol"),
        (EV_TO_KCAL_PER_MOL, "kcal/mol", "kcal/mol"),
        (EV_TO_MEV, "meV", "meV"),
        (EV_TO_RYDBERG, "Rydberg", "Ry"),
        (EV_TO_HARTREE, "Hartree", "Ha"),
    ]
    out: list[Transform] = []
    for factor, name, short in pairs:
        # Agent reported in `name`; dividing recovers eV.
        out.append(
            Transform(
                1.0 / factor,
                f"/ {factor:.6g} ({short} -> eV)",
                FailureClass.SILENT_UNIT,
                f"Agent reported the value in {name}; ground truth is in eV.",
            )
        )
        out.append(
            Transform(
                factor,
                f"x {factor:.6g} (eV -> {short})",
                FailureClass.SILENT_UNIT,
                f"Agent reported in eV; ground truth is in {name}.",
            )
        )
    return out


def _pressure_unit_transforms() -> list[Transform]:
    pairs = [
        (EV_PER_ANG3_TO_GPA, "eV/Angstrom^3", "eV/A^3"),
        (10.0, "kbar (x10 GPa)", "kbar"),
        (100.0, "Mbar", "Mbar"),
        (1e9, "Pa", "Pa"),
    ]
    out: list[Transform] = []
    for factor, name, short in pairs:
        out.append(
            Transform(
                factor,
                f"x {factor:.6g} ({short} -> GPa)",
                FailureClass.SILENT_UNIT,
                f"Agent reported in {name}; ground truth is in GPa.",
            )
        )
        out.append(
            Transform(
                1.0 / factor,
                f"/ {factor:.6g} (GPa -> {short})",
                FailureClass.SILENT_UNIT,
                f"Agent reported in GPa; ground truth is in {name}.",
            )
        )
    return out


def _length_unit_transforms() -> list[Transform]:
    return [
        Transform(
            1.0 / ANG_TO_BOHR,
            f"/ {ANG_TO_BOHR:.6g} (Bohr -> Angstrom)",
            FailureClass.SILENT_UNIT,
            "Agent reported a length in Bohr; ground truth is in Angstrom.",
        ),
        Transform(
            ANG_TO_BOHR,
            f"x {ANG_TO_BOHR:.6g} (Angstrom -> Bohr)",
            FailureClass.SILENT_UNIT,
            "Agent reported in Angstrom; ground truth is in Bohr.",
        ),
        Transform(
            1.0 / ANG3_TO_BOHR3,
            f"/ {ANG3_TO_BOHR3:.6g} (Bohr^3 -> Angstrom^3)",
            FailureClass.SILENT_UNIT,
            "Agent reported a volume in Bohr^3; ground truth is in Angstrom^3.",
        ),
    ]


def _basis_transforms(ctx: CellContext) -> list[Transform]:
    """Per-atom vs per-formula-unit vs per-cell -- the classic silent factor-of-N."""
    out: list[Transform] = []
    candidates: list[tuple[float, str]] = []

    n_at_p = ctx.n_atoms_primitive
    n_at_c = ctx.n_atoms_conventional
    n_fu_p = ctx.n_formula_units_primitive
    n_fu_c = ctx.n_formula_units_conventional

    if n_at_p > 1:
        candidates.append((float(n_at_p), f"n_atoms_primitive ({n_at_p})"))
    if n_at_c > 1 and n_at_c != n_at_p:
        candidates.append((float(n_at_c), f"n_atoms_conventional ({n_at_c})"))
    if n_fu_p > 1:
        candidates.append((float(n_fu_p), f"n_formula_units_primitive ({n_fu_p})"))
    if n_fu_c > 1 and n_fu_c != n_fu_p:
        candidates.append((float(n_fu_c), f"n_formula_units_conventional ({n_fu_c})"))
    if n_fu_p > 0 and n_at_p % max(n_fu_p, 1) == 0:
        atoms_per_fu = n_at_p // n_fu_p
        if atoms_per_fu > 1:
            candidates.append((float(atoms_per_fu), f"atoms_per_formula_unit ({atoms_per_fu})"))

    for factor, name in candidates:
        out.append(
            Transform(
                1.0 / factor,
                f"/ {name}",
                FailureClass.SILENT_BASIS,
                f"Agent reported a total (per-cell) value; ground truth is normalised by {name}.",
            )
        )
        out.append(
            Transform(
                factor,
                f"x {name}",
                FailureClass.SILENT_BASIS,
                f"Agent normalised by {name}; ground truth is a total value.",
            )
        )
    return out


def _cell_transforms(ctx: CellContext) -> list[Transform]:
    """Primitive vs conventional cell -- an extensive-quantity ratio."""
    out: list[Transform] = []
    n_p, n_c = ctx.n_atoms_primitive, ctx.n_atoms_conventional
    if n_p > 0 and n_c > 0 and n_p != n_c:
        ratio = n_c / n_p
        out.append(
            Transform(
                1.0 / ratio,
                f"/ {ratio:.6g} (conventional -> primitive cell)",
                FailureClass.SILENT_CELL,
                "Agent used the conventional cell; ground truth is defined on the primitive cell.",
            )
        )
        out.append(
            Transform(
                ratio,
                f"x {ratio:.6g} (primitive -> conventional cell)",
                FailureClass.SILENT_CELL,
                "Agent used the primitive cell; ground truth is on the conventional cell.",
            )
        )
    if ctx.volume_primitive and ctx.volume_conventional and ctx.volume_primitive > 0:
        vr = ctx.volume_conventional / ctx.volume_primitive
        if abs(vr - 1.0) > 1e-6:
            out.append(
                Transform(
                    1.0 / vr,
                    f"/ {vr:.6g} (conventional/primitive volume ratio)",
                    FailureClass.SILENT_CELL,
                    "Volume-extensive quantity reported on the wrong cell.",
                )
            )
    return out


def _sign_transforms() -> list[Transform]:
    return [
        Transform(
            -1.0,
            "x -1 (sign convention)",
            FailureClass.SILENT_SIGN,
            "Magnitude is correct but the sign convention is inverted "
            "(e.g. formation energy reported as a positive stability).",
        )
    ]


def candidate_transforms(ctx: CellContext, truth: Quantity) -> list[Transform]:
    """Build the search space, ordered so the most diagnostic hits are found first."""
    unit = canonical_unit(truth.unit)
    transforms: list[Transform] = []

    transforms.extend(_basis_transforms(ctx))
    transforms.extend(_cell_transforms(ctx))

    if unit.startswith(("eV", "meV", "kJ", "kcal", "Hartree", "Rydberg")):
        transforms.extend(_energy_unit_transforms())
    elif unit in {"GPa", "kbar", "Mbar", "Pa"}:
        transforms.extend(_pressure_unit_transforms())
    elif unit in {"Angstrom", "Angstrom^3", "Bohr"}:
        transforms.extend(_length_unit_transforms())
    else:
        transforms.extend(_energy_unit_transforms())
        transforms.extend(_pressure_unit_transforms())

    transforms.extend(_sign_transforms())

    # Composites: a basis slip and a unit slip at the same time. Bounded to keep the
    # search space honest -- an unbounded product would "explain" any number at all.
    basis_and_cell = _basis_transforms(ctx) + _cell_transforms(ctx)
    unit_only = [t for t in transforms if t.failure_class == FailureClass.SILENT_UNIT]
    for b in basis_and_cell:
        for u in unit_only:
            transforms.append(
                Transform(
                    b.factor * u.factor,
                    f"{b.label} then {u.label}",
                    b.failure_class,
                    f"Two compounded convention errors: {b.explanation} {u.explanation}",
                    components=2,
                )
            )
    return transforms


def _best_match(
    reported_value: float,
    truth: Quantity,
    transforms: list[Transform],
    tolerance: Tolerance,
    max_relative_residual: float | None = None,
) -> tuple[float, Transform] | None:
    """Tightest transform that lands within tolerance, or None."""
    best: tuple[float, Transform] | None = None
    denom = abs(truth.value) or 1.0

    for t in transforms:
        corrected = reported_value * t.factor
        if not tolerance.accepts(corrected, truth.value):
            continue
        residual = abs(corrected - truth.value) / denom
        if max_relative_residual is not None and residual > max_relative_residual:
            continue
        if best is None or residual < best[0]:
            best = (residual, t)
    return best


def reconcile(
    reported_value: float,
    truth: Quantity,
    ctx: CellContext,
    tolerance: Tolerance,
) -> Reconciliation:
    """Try to explain a wrong number as a pure convention error.

    Single-slip transforms are searched first and always win. Only if none of them fits
    do we consider two compounded slips, and those must match an order of magnitude
    more tightly (`COMPOSITE_MAX_REL`) before we will believe them.

    This ordering is what keeps the silent-failure rate honest. Without it the dense
    product set will land within an ordinary tolerance of essentially any wrong number,
    and genuinely wrong physics gets misreported as a bookkeeping slip.

    Within each pass the *tightest* match wins, so a factor of 4 is not reported when
    an exact factor of 2 also fits.
    """
    all_transforms = candidate_transforms(ctx, truth)
    singles = [t for t in all_transforms if t.components == 1]
    composites = [t for t in all_transforms if t.components > 1]

    best = _best_match(reported_value, truth, singles, tolerance)
    if best is None:
        best = _best_match(
            reported_value, truth, composites, tolerance, max_relative_residual=COMPOSITE_MAX_REL
        )

    if best is None:
        return Reconciliation(matched=False)

    _, transform = best
    return Reconciliation(
        matched=True,
        failure_class=transform.failure_class,
        factor=transform.factor,
        label=transform.label,
        explanation=transform.explanation,
    )


def relative_error(reported: float, truth: float) -> float | None:
    denom = abs(truth)
    if denom == 0.0:
        return None
    return abs(reported - truth) / denom
