"""The reconciler is the load-bearing claim of this benchmark, so it gets the most tests.

If it over-matches, every wrong answer gets excused as a unit slip and the headline
finding is fiction. If it under-matches, silent errors get filed as NUMERIC_WRONG and
the finding disappears. Both directions are tested here.
"""

from __future__ import annotations

import pytest

from matagentbench.schema import CellContext, FailureClass, Quantity, Tolerance
from matagentbench.units import canonical_unit, reconcile


@pytest.fixture
def ctx() -> CellContext:
    # Rutile TiO2: 6 atoms / 2 formula units primitive, 12 / 4 conventional.
    return CellContext(
        n_atoms_primitive=6,
        n_atoms_conventional=12,
        n_formula_units_primitive=2,
        n_formula_units_conventional=4,
        volume_primitive=62.4,
        volume_conventional=124.8,
        reduced_formula="TiO2",
    )


@pytest.fixture
def truth() -> Quantity:
    return Quantity(value=-3.5, unit="eV/atom", basis="per_atom", cell="primitive", state="relaxed")


def test_correct_value_has_no_convention_error_to_name(ctx, truth):
    """A value that already equals ground truth is not a convention error.

    There is deliberately no identity transform in the search space: `reconcile` is only
    consulted for answers that have already failed the numeric check, and its job is to
    name a *discrepancy*. Returning a match here would mean inventing one.
    """
    result = reconcile(-3.5, truth, ctx, Tolerance(rel=0.01))
    assert not result.matched
    assert result.failure_class is None


def test_per_cell_instead_of_per_atom(ctx, truth):
    """Agent reported the whole-cell energy: -3.5 * 6 atoms."""
    result = reconcile(-21.0, truth, ctx, Tolerance(rel=0.02))
    assert result.matched
    assert result.failure_class is FailureClass.SILENT_BASIS
    assert result.factor == pytest.approx(1 / 6)


def test_per_formula_unit_instead_of_per_atom(ctx, truth):
    """Agent normalised by formula units (2) instead of atoms (6)."""
    result = reconcile(-10.5, truth, ctx, Tolerance(rel=0.02))
    assert result.matched
    assert result.failure_class is FailureClass.SILENT_BASIS


def test_kj_per_mol_instead_of_ev(ctx, truth):
    result = reconcile(-3.5 * 96.48533212, truth, ctx, Tolerance(rel=0.01))
    assert result.matched
    assert result.failure_class is FailureClass.SILENT_UNIT
    assert "kJ/mol" in (result.label or "") or "kJ/mol" in (result.explanation or "")


def test_mev_instead_of_ev(ctx, truth):
    result = reconcile(-3500.0, truth, ctx, Tolerance(rel=0.01))
    assert result.matched
    assert result.failure_class is FailureClass.SILENT_UNIT


def test_sign_convention_flip(ctx, truth):
    result = reconcile(3.5, truth, ctx, Tolerance(rel=0.01))
    assert result.matched
    assert result.failure_class is FailureClass.SILENT_SIGN


def test_conventional_cell_for_extensive_quantity(ctx):
    truth = Quantity(value=-21.0, unit="eV", basis="per_cell", cell="primitive", state="relaxed")
    # Agent used the conventional cell: twice as many atoms, twice the energy.
    result = reconcile(-42.0, truth, ctx, Tolerance(rel=0.02))
    assert result.matched
    assert result.failure_class in {FailureClass.SILENT_CELL, FailureClass.SILENT_BASIS}


def test_genuinely_wrong_number_is_not_explained_away(ctx, truth):
    """The critical negative case: an arbitrary wrong value must NOT reconcile.

    -1.234 is not related to -3.5 by any atom count, formula-unit count, cell ratio or
    energy-unit conversion in the search space.
    """
    result = reconcile(-1.234, truth, ctx, Tolerance(rel=0.02))
    assert not result.matched
    assert result.failure_class is None


def test_wrong_by_a_small_random_amount_does_not_reconcile(ctx, truth):
    """Regression: composites used to explain this one away.

    -3.9 sits within 2% of ground truth once multiplied by (1/12 atoms) x (eV->Rydberg)
    = 0.88198. Two compounded slips can be found for almost any number, so composites
    now have to match ~10x more tightly than singles and only get consulted when no
    single slip fits. Without that rule the headline silent-failure rate is fiction.
    """
    result = reconcile(-3.9, truth, ctx, Tolerance(rel=0.02))
    assert not result.matched


def test_composite_still_matches_when_it_is_exact(ctx, truth):
    """The tightening must not throw out real compounded errors.

    Per-cell (x6 atoms) AND reported in meV: an exact double slip still reconciles.
    """
    result = reconcile(-3.5 * 6 * 1000, truth, ctx, Tolerance(rel=0.02))
    assert result.matched
    assert result.failure_class in {FailureClass.SILENT_BASIS, FailureClass.SILENT_UNIT}


def test_single_slip_is_preferred_over_a_composite(ctx, truth):
    """When both could fit, the simpler explanation must win."""
    result = reconcile(-21.0, truth, ctx, Tolerance(rel=0.02))
    assert result.matched
    assert result.factor == pytest.approx(1 / 6)
    assert "then" not in (result.label or "")


def test_tightest_transform_wins(ctx):
    """With both a factor-of-2 and a factor-of-4 candidate, prefer the exact one."""
    truth = Quantity(value=-10.0, unit="eV", basis="per_cell", cell="primitive", state="relaxed")
    result = reconcile(-20.0, truth, ctx, Tolerance(rel=0.10))
    assert result.matched
    assert result.factor == pytest.approx(0.5, rel=1e-6)


def test_bulk_modulus_unit_confusion(ctx):
    truth = Quantity(value=200.0, unit="GPa", basis="per_cell", cell="primitive", state="relaxed")
    # Reported in eV/Angstrom^3 instead of GPa.
    result = reconcile(200.0 / 160.2176634, truth, ctx, Tolerance(rel=0.02))
    assert result.matched
    assert result.failure_class is FailureClass.SILENT_UNIT


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eV/atom", "eV/atom"),
        ("ev per atom", "eV/atom"),
        ("EV/ATOM", "eV/atom"),
        ("kJ/mol", "kJ/mol"),
        ("gpa", "GPa"),
        ("Å", "Angstrom"),
        ("", ""),
    ],
)
def test_unit_canonicalisation(raw, expected):
    assert canonical_unit(raw) == expected
