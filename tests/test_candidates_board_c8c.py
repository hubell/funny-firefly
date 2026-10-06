# SPDX-License-Identifier: GPL-3.0-only
"""C8c in ``scripts/candidates_board.py``: its arm, its combination, its numbers.

Nothing here fits a booster or reads challenge data. The amendment's numbers
are checked against its working note where it is on hand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import candidates_board as board  # noqa: E402

RULE = board.prior_rule()


def _words(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_the_amendment_of_6_october_where_it_is_on_hand():
    notes = os.environ.get("PREREGISTRATION_NOTES")
    path = Path(notes) / "preregistration_2026-10-06_c8_prior.md" if notes else None
    if path is None or not path.exists():
        pytest.skip("the amendment's working note is not on hand")
    from taxiout.provenance import digest

    assert digest(path) == RULE.prior_sha256
    text = _words(path)
    assert "multiplied, before the clip, by n0/n1" in text
    assert (
        "(p / (1 - p)) x (n0 / n1), clipped to [0.1, 10], then scaled to a mean of 1"
        in text
    )
    assert "The verdict is written by 7 October 2026, 18:00 UTC" in text
    assert RULE.deadline_utc == "2026-10-07T18:00:00+00:00"
    assert "adopted C1 and C4 in one booster" in text
    assert RULE.adopted == ("c1", "c4", "c3b")
    assert RULE.prior_sha256 != board.board_rule().amendment_sha256


def test_c8c_is_c8_with_the_factor_alone():
    c8 = board.candidates()["c8"].member
    c8c = board.c8c().member
    assert c8c.weighted and c8c.class_ratio
    assert c8.class_ratio is None
    assert (c8c.blocks, c8c.objective, c8c.backend) == (
        c8.blocks,
        c8.objective,
        c8.backend,
    )
    assert board.member_columns(c8c) == board.member_columns(c8)
    assert board.member_params(c8c, 0) == board.member_params(c8, 0)
    assert board.member_params(c8c, 1) == board.member_params(c8, 1)


def test_the_combination_is_the_configuration_with_c8cs_weights():
    combo = board.c8c_combination()
    cands = board.candidates()
    adopted = board.combination(list(RULE.adopted), cands, "combo4", "section 4's")
    assert combo.kind == "combination" and combo.layers == ()
    assert combo.member.blocks == adopted.member.blocks == ("aobt3", "flight")
    assert combo.member.weighted and combo.member.class_ratio
    assert combo.members == adopted.members
    assert [m.backend for m in combo.members] == ["catboost"]
    assert combo.member.key == "combo8c_booster"
    assert board.member_columns(combo.member) == board.member_columns(adopted.member)


def test_the_configuration_of_6_october_is_unchanged_by_the_new_field():
    adopted = board.combination(
        list(RULE.adopted), board.candidates(), "combo4", "section 4's"
    )
    assert adopted.member == board.Member("combo4_booster", blocks=("aobt3", "flight"))
    assert adopted.member.weighted is None and adopted.member.class_ratio is None


def test_what_this_amendment_writes_is_named_apart():
    w = board.WORK
    for name in (
        "december__c8c__seed0.json",
        "heldout-2025-03__combo8c_booster__seed0.parquet",
        "july__combo8c__seed0.json",
        "december__reference_refit__seed0.parquet",
        "stage_c8c_check.json",
        "clock_c8c.json",
    ):
        assert board.ours(w / name), name
    for name in (
        "december__c8__seed0.json",
        "december__reference__seed0.parquet",
        "heldout-2025-03__combo4_booster__seed0.json",
        "july__combo4__seed0.parquet",
        "heldout-2025-01__base__seed0.json",
        "stage_stage4.json",
        "clock_stage6.json",
        "c3b_path.json",
        "stands.json",
    ):
        assert not board.ours(w / name), name


def test_the_stages_of_5_october_are_unchanged():
    assert "c8c" not in board.candidates()
    assert all("c8c" not in keys for keys in board.stages().values())


def _status(dec, jul, three, octo, year=None, dec1=None, jul1=None):
    def r(x):
        return {"rows": {"contrast": x}, "short": {"contrast": x}}

    m = {
        "seed0": {
            "december:all": r(dec),
            "july:all": r(jul),
            "july:three": r(three),
            "october:all": r(octo),
        }
    }
    if year is not None:
        m["seed0"]["year:all"] = r(year)
    if dec1 is not None:
        m["seed1"] = {"december:all": r(dec1), "july:all": r(jul1)}
    return m


def test_a_combination_is_held_to_the_five_and_no_second_seed():
    assert board.combination_status({}) == "not measured"
    assert board.combination_status(_status(-0.3, -0.6, 0, 0)) == "not measured"
    assert board.combination_status(_status(-0.3, -0.6, 0, 0, 0.16)) == "holds the five"
    assert board.combination_status(_status(-0.3, -0.6, 0, 0, 0.17)) == "fails the five"
    assert board.combination_status(_status(-0.2, -0.6, 0, 0, 0.0)) == "fails the five"


def test_the_configuration_in_words():
    words = board.configuration_words
    assert (
        words({"config": ["c1", "c4", "c3b"], "combined": True})
        == "C1, C4, C3b, read as one combination"
    )
    assert words({"config": ["c1", "c4", "c3b", "c8c"], "combined": True}).startswith(
        "C1, C4, C3b, C8c"
    )
    assert words({"config": [], "combined": False}) == "none"
