# SPDX-License-Identifier: GPL-3.0-only
"""Whether the file applies the fusion, G1n: one recorded decision.

``taxiout.models.FUSION_G1N_APPLIED`` records §1 and §3 of
``reports/preregistered-2026-10-06-g1n.md`` as ``reports/fusion.md`` reads its
rule, and ``scripts/make_submission.py`` reads it rather than deciding for
itself. What the written column is: the fusion's value, floored and rounded,
on the rows ``taxiout.fusion.applied_rows`` takes, and every other row as
without it, which is checked here through the script's ``with_fusion``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from taxiout import fusion  # noqa: E402
from taxiout.provenance import CHOSEN, CONSTANTS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "src" / "taxiout" / "models.py"
SCRIPT = REPO / "scripts" / "make_submission.py"
COPY = "reports/preregistered-2026-10-06-g1n.md"
REPORT = "reports/fusion.md"
DISCLOSURE = "chosen after 22d's figures"


def _script():
    spec = importlib.util.spec_from_file_location(
        "make_submission_v10_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comment_above(source: str, line: int) -> str:
    lines = source.splitlines()[: line - 1]
    block: list[str] = []
    for text in reversed(lines):
        if not text.startswith("#"):
            break
        block.append(text.lstrip("# ").rstrip())
    return " ".join(" ".join(reversed(block)).split())


def test_the_decision_is_recorded_with_its_basis_and_its_disclosure():
    source = MODELS.read_text(encoding="utf-8")
    found = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.AnnAssign)
        and getattr(n.target, "id", None) == "FUSION_G1N_APPLIED"
    ]
    assert len(found) == 1
    node = found[0]
    assert ast.unparse(node.annotation) == "bool"
    assert isinstance(node.value, ast.Constant) and node.value.value is True
    basis = _comment_above(source, node.lineno)
    assert COPY in basis and REPORT in basis and DISCLOSURE in basis
    assert "G1n: ADOPTED" in basis
    assert (REPO / COPY).exists()


def test_the_register_holds_the_decision_as_chosen_with_the_disclosure():
    (entry,) = [c for c in CONSTANTS if c.name == "FUSION_G1N_APPLIED"]
    assert entry.module == "src/taxiout/models.py"
    assert entry.cls == CHOSEN and entry.literals == (True,)
    assert DISCLOSURE in entry.basis and COPY in entry.basis and REPORT in entry.basis


def test_the_script_reads_the_decision_and_records_it():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "taxiout.models"
        for a in n.names
    }
    assert {"FUSION_G1N_APPLIED", "FUSION_BASE_BOOSTER_SHA256"} <= imported
    for field in (
        '"fusion_g1n_applied"',
        '"fusion"',
        '"gate"',
        '"model_sha256"',
        '"inputs_sha256"',
        '"rows_taken"',
    ):
        assert field in source, field


def test_with_fusion_touches_only_the_rows_the_gate_takes():
    ms = _script()
    gate = pd.Series({"EDDF": True, "EGLL": False})
    frame = pd.DataFrame(
        {
            "ADEP_mvt": ["EDDF", "EDDF", "EGLL", "EDDF", "EDDF"],
            "a_covered": [1, 0, 1, 1, 0],
            "o_covered": [0, 0, 1, 1, 1],
            "rules_row": [0, 0, 0, 1, 0],
        }
    )
    take = fusion.applied_rows(frame, gate)
    v9 = np.array([600.0, 700.0, 800.0, 52550.0, 900.0])
    prediction = np.array([611.4, 1.0, 1.0, 1.0, -3.0])
    got = ms.with_fusion(v9, prediction, take)
    # covered and kept: the fusion, rounded
    assert got[0] == 611.0
    # uncovered, a dropped airport, a rules' row: v9's values exactly
    assert got[1] == 700.0 and got[2] == 800.0 and got[3] == 52550.0
    # OPDI coverage counts, and a negative prediction is floored at zero
    assert got[4] == 0.0
    assert list(take) == [True, False, False, False, True]
