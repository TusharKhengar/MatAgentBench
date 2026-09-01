"""Contract tests for the JSON the static site consumes.

The site is a pure renderer with no build step, so a path or field-name change here
breaks the published page silently. These tests pin the contract.
"""

from __future__ import annotations

from matagentbench.report import discover_counterfactuals


def test_discover_counterfactuals_recovers_full_task_id(tmp_path):
    """Regression: task_id contains '__', so splitting from the left truncates it.

    `formation_energy__mp-2657__convention_correct__k3.json` must map back to the task
    `formation_energy__mp-2657`, not to `formation_energy`. Getting this wrong makes
    every `has_counterfactual` flag false and hides the counterfactual panel on the
    site entirely.
    """
    folder = tmp_path / "counterfactuals" / "cerebras-gpt-oss-120b"
    folder.mkdir(parents=True)
    (folder / "formation_energy__mp-2657__convention_correct__k3.json").write_text("{}")
    (folder / "vacancy_formation__mp-149__plan_repair__k0.json").write_text("{}")

    found = discover_counterfactuals(tmp_path)

    assert ("cerebras-gpt-oss-120b", "formation_energy__mp-2657") in found
    assert ("cerebras-gpt-oss-120b", "vacancy_formation__mp-149") in found
    assert ("cerebras-gpt-oss-120b", "formation_energy") not in found


def test_discover_counterfactuals_ignores_malformed_names(tmp_path):
    folder = tmp_path / "counterfactuals" / "groq-qwen3-32b"
    folder.mkdir(parents=True)
    (folder / "not-a-counterfactual.json").write_text("{}")
    assert discover_counterfactuals(tmp_path) == set()


def test_discover_counterfactuals_on_missing_directory(tmp_path):
    assert discover_counterfactuals(tmp_path) == set()
