# SPDX-License-Identifier: GPL-3.0-only
"""Whether the file takes group B's new base: one recorded decision.

``taxiout.models.GROUP_B_APPLIED`` records §4 of
``reports/preregistered-2026-10-06-airports.md`` as ``reports/group-b.md``
reads it, and ``GROUP_B_BASE`` the selector's choice per airport for the
ranking rows. ``scripts/make_submission.py`` reads both rather than deciding
for itself: a row takes its airport's menu item, the booster file stays the
configuration's prediction, the rules' rows keep the rules' values, and with
the decision off nothing is fitted beside the configuration and the build is
v10's.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taxiout import group_b
from taxiout.models import GROUP_B_APPLIED, GROUP_B_BASE
from taxiout.provenance import CHOSEN, CONSTANTS

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "src" / "taxiout" / "models.py"
SCRIPT = REPO / "scripts" / "make_submission.py"
COPY = "reports/preregistered-2026-10-06-airports.md"
REPORT = "reports/group-b.md"
FUSION_REPORT = "reports/fusion-new-base.md"
DISCLOSURES = ("a selection among four weights", "inform its choice")
RECORD = REPO / "data" / "processed" / "candidates_board" / "group_b.json"


def _script():
    spec = importlib.util.spec_from_file_location(
        "make_submission_v11_under_test", SCRIPT
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


def test_the_decision_is_recorded_with_its_basis_and_its_disclosures():
    source = MODELS.read_text(encoding="utf-8")
    found = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.AnnAssign)
        and getattr(n.target, "id", None) == "GROUP_B_APPLIED"
    ]
    assert len(found) == 1
    node = found[0]
    assert ast.unparse(node.annotation) == "bool"
    assert isinstance(node.value, ast.Constant) and node.value.value is True
    basis = _comment_above(source, node.lineno)
    for words in (COPY, REPORT, FUSION_REPORT, *DISCLOSURES):
        assert words in basis, words
    assert "confirmed by the owner on 7 October 2026" in basis
    assert (REPO / COPY).exists()


def test_the_register_holds_the_decision_as_chosen_with_the_disclosures():
    (entry,) = [c for c in CONSTANTS if c.name == "GROUP_B_APPLIED"]
    assert entry.module == "src/taxiout/models.py"
    assert entry.cls == CHOSEN and entry.literals == (True,)
    for words in (COPY, REPORT, FUSION_REPORT, *DISCLOSURES):
        assert words in entry.basis, words


def test_the_choice_per_airport_is_the_selectors():
    """Lowest RMSE over all twelve held-out months: the members' blend at
    LIRF, C11's blend at the other nine."""
    assert GROUP_B_BASE["LIRF"] == "w0.7"
    assert {a for a, it in GROUP_B_BASE.items() if it == "c11"} == {
        "EDDF",
        "EDDM",
        "EGLL",
        "EHAM",
        "LEBL",
        "LEMD",
        "LFPG",
        "LSZH",
        "LTFM",
    }


def test_the_choice_per_airport_is_the_records_where_they_are_on_hand():
    """``make group-b`` writes the selector's ranking choice from the records."""
    if not RECORD.exists():
        pytest.skip("the group B records are not on hand (make group-b writes them)")
    got = json.loads(RECORD.read_text(encoding="utf-8"))
    assert got["c14_ranking"] == GROUP_B_BASE


def _rank(icao):
    return pd.DataFrame(
        {"MVT_ID_mvt": np.arange(len(icao), dtype=float), "ADEP_mvt": icao}
    )


def test_a_row_at_lirf_takes_the_members_blend_and_elsewhere_c11s():
    ms = _script()
    icao = ["LIRF", "EDDF", "LTFM", "LIRF"]
    parts = {
        "combo4_booster": np.array([500.0, 600.0, 700.0, 800.0]),
        "c3b_member": np.array([520.0, 640.0, 720.0, 780.0]),
        "c11_member": np.array([510.0, 580.0, 690.0, 830.0]),
    }
    physical = pd.Series(
        np.mean(np.vstack([parts["combo4_booster"], parts["c3b_member"]]), axis=0)
    )
    got = ms.group_b_base(_rank(icao), parts, physical)
    base = got["base"].to_numpy()
    for i in (0, 3):
        assert (
            base[i]
            == 0.7 * parts["combo4_booster"][i] + (1.0 - 0.7) * parts["c3b_member"][i]
        )
    for i in (1, 2):
        assert (
            base[i] == group_b.blend(parts["combo4_booster"], parts["c11_member"], 0.5)[i]
        )
    assert got["record"]["rows_by_item"] == {"c11": 2, "w0.7": 2}
    assert got["record"]["configuration_sha256"] == group_b.pred_sha256(physical)
    assert got["record"]["new_base_sha256"] == group_b.pred_sha256(base)
    # the booster file's prediction is the configuration's, untouched
    assert physical.to_numpy()[0] == 510.0


def test_the_rules_rows_keep_the_rules_values_whatever_the_base():
    ms = _script()
    base = pd.Series([600.0, 700.0, 800.0])
    other = pd.Series([1.0, 2.0, 3.0])
    term = pd.Series([np.nan, 52550.0, np.nan])
    shift = pd.Series([np.nan, np.nan, 41000.0])
    a = ms.submitted(base, term, shift, True)
    b = ms.submitted(other, term, shift, True)
    assert a[1] == b[1] == 52550.0 and a[2] == b[2] == 41000.0
    assert a[0] == 600.0 and b[0] == 1.0


def test_with_the_decision_off_nothing_is_fitted_beside_the_configuration(monkeypatch):
    ms = _script()

    class Member:
        def __init__(self, key):
            self.key = key

    class Combo:
        member = Member("combo4_booster")
        members = (Member("c3b_member"),)

    monkeypatch.setattr(ms, "GROUP_B_APPLIED", False)
    assert ms.group_b_members(Combo()) == ()
    monkeypatch.setattr(ms, "GROUP_B_APPLIED", True)
    assert [m.key for m in ms.group_b_members(Combo())] == ["c11_member"]


def test_the_configurations_prediction_ignores_the_member_beside_it():
    ms = _script()
    conf = ms.Configuration.__new__(ms.Configuration)

    class Member:
        def __init__(self, key):
            self.key = key

    conf.member = Member("combo4_booster")

    class Combo:
        members = (Member("c3b_member"),)

    conf.combo = Combo()
    parts = {
        "combo4_booster": np.array([1.0, 2.0]),
        "c3b_member": np.array([3.0, 6.0]),
        "c11_member": np.array([100.0, 100.0]),
    }
    conf.parts = lambda Xr: parts
    got = conf.predict(pd.DataFrame(index=[0, 1]))
    assert list(got) == [2.0, 4.0]


def test_the_script_reads_the_decision_and_records_it():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "taxiout.models"
        for a in n.names
    }
    assert {"GROUP_B_APPLIED", "GROUP_B_BASE", "FUSION_BASE_BOOSTER_SHA256"} <= imported
    for field in (
        '"group_b_applied"',
        '"group_b"',
        '"base_per_airport"',
        '"parts_prediction_sha256"',
        '"new_base_sha256"',
        '"base_2025_records"',
        '"model_sha256"',
        '"rows_taken"',
    ):
        assert field in source, field
    assert GROUP_B_APPLIED is True
