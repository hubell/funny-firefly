# SPDX-License-Identifier: GPL-3.0-only
"""Group B's pieces, ``taxiout.group_b`` and ``candidates_board``'s, on synthetic frames.

Nothing here fits a booster or reads challenge data. The rule's numbers are
checked against the published copies of the last-week note and the airports
amendment; the selector, the join, the blends and the months are checked on
constructed predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import candidates as cand  # noqa: E402
import candidates_board as board  # noqa: E402
from taxiout import group_b  # noqa: E402
from taxiout.fusion import AIRPORTS  # noqa: E402

RULE = board.group_b_rule()
MONTHS = list(cand.MONTHS)


def _words(path: str) -> str:
    return " ".join((REPO / path).read_text(encoding="utf-8").split())


def test_the_numbers_are_the_copies_words():
    text = _words(board.GROUP_B_TEXT)
    assert "w in {0.3, 0.4, 0.6, 0.7}" in text
    assert RULE.grid == (0.3, 0.4, 0.6, 0.7)
    assert "`combo4_booster` refitted at `n_estimators` 9,000" in text
    assert RULE.c12_trees == 9000
    assert (
        "better on January by more than 0.47 s; not worse on February by more than"
        " 1.59 s or on December by more than 0.23 s; not worse over the twelve held-out"
        " months' short rows by more than 0.15 s" in text
    )
    assert [(k, kind, f) for k, _, kind, f, _ in RULE.w_winter] == [
        ("january", "better", 0.47),
        ("february", "not worse", 1.59),
        ("december", "not worse", 0.23),
        ("year", "not worse", 0.15),
    ]
    assert (
        "(December 0.23, July 0.54, the three July airports 1.02, October 0.21 s)" in text
    )
    assert [f for *_, f in RULE.w_legs] == [0.23, 0.54, 1.02, 0.21]
    floors = (
        "over core rows Jan 0.33, Feb 1.56, Mar 0.36, Apr 0.07, May 0.13, Jun 0.21, Jul"
        " 0.58, Aug 0.26, Sep 0.21, Oct 0.29, Nov 0.39, Dec 0.23, year 0.16 s; over short"
        " rows Jan 0.47, Feb 1.59, Dec 0.23, year 0.15 s"
    )
    assert floors in text
    assert ", ".join(f"{k} {v:.2f}" for k, v in RULE.month_floors.items()) in floors
    assert ", ".join(f"{k} {v:.2f}" for k, v in RULE.short_floors.items()) in floors
    assert "Verdicts by 9 October 2026, 18:00 UTC" in _words(board.AIRPORTS_TEXT)
    assert RULE.deadline_utc == "2026-10-09T18:00:00+00:00"


def test_the_copies_name_the_notes_digests():
    assert RULE.last_week_sha256 in _words(board.GROUP_B_TEXT)
    assert RULE.airports_sha256 in _words(board.AIRPORTS_TEXT)


def test_the_disclosures_are_the_copies_words():
    assert "The choice among four weights is disclosed as a selection." in _words(
        board.GROUP_B_TEXT
    )
    assert "this lets months after a forward leg's month inform its choice" in _words(
        board.AIRPORTS_TEXT
    )


def test_the_c13_join_covers_every_row_once_and_keeps_order():
    rng = np.random.default_rng(1)
    icao = rng.choice(list(AIRPORTS), size=5000)
    truth = rng.normal(size=5000)
    parts = {a: truth[icao == a] for a in AIRPORTS}
    got = group_b.join_by_airport(icao, dict(reversed(list(parts.items()))))
    assert np.array_equal(got, truth)


def test_the_c13_join_refuses_a_missing_or_short_airport():
    rng = np.random.default_rng(2)
    icao = rng.choice(list(AIRPORTS), size=1000)
    parts = {a: np.zeros((icao == a).sum()) for a in AIRPORTS[:-1]}
    with pytest.raises(ValueError):
        group_b.join_by_airport(icao, parts)
    parts[AIRPORTS[-1]] = np.zeros((icao == AIRPORTS[-1]).sum() - 1)
    with pytest.raises(ValueError):
        group_b.join_by_airport(icao, parts)


def _sse(errors: dict) -> dict:
    """{item: {month: {airport: (sse, n)}}} from per-item per-month constant errors."""
    return {
        it: {m: {a: (e[m] ** 2 * 10, 10) for a in AIRPORTS} for m in MONTHS}
        for it, e in errors.items()
    }


def test_the_selector_never_uses_the_held_out_month_for_its_own_choice():
    held = "2025-03"
    base = {m: 100.0 for m in MONTHS}
    x = {m: (0.0 if m == held else 200.0) for m in MONTHS}
    assert (
        group_b.choose(_sse({"base": base, "x": x}), ["base", "x"], "EDDF", held)[0]
        == "base"
    )
    y = {m: (1000.0 if m == held else 50.0) for m in MONTHS}
    assert (
        group_b.choose(_sse({"base": base, "y": y}), ["base", "y"], "EDDF", held)[0]
        == "y"
    )


def test_the_selector_keeps_the_base_unless_another_is_strictly_lower():
    same = {m: 100.0 for m in MONTHS}
    lower = {m: 90.0 for m in MONTHS}
    sse = _sse({"base": same, "a": same, "b": dict(same)})
    assert group_b.choose(sse, ["base", "a", "b"], "LIRF", "2025-07")[0] == "base"
    sse = _sse({"base": same, "a": lower, "b": dict(lower)})
    assert group_b.choose(sse, ["base", "a", "b"], "LIRF", "2025-07")[0] == "a"
    sse = _sse({"base": same, "a": lower, "b": {m: 80.0 for m in MONTHS}})
    assert group_b.choose(sse, ["base", "a", "b"], "LIRF", None)[0] == "b"


def test_the_ranking_choice_uses_all_twelve():
    base = {m: 100.0 for m in MONTHS}
    x = {m: (0.0 if m == "2025-01" else 101.0) for m in MONTHS}
    sse = _sse({"base": base, "x": x})
    assert group_b.choose(sse, ["base", "x"], "EDDM", None)[0] == "x"
    assert group_b.choose(sse, ["base", "x"], "EDDM", "2025-01")[0] == "base"


def test_apply_choice_takes_each_airport_from_its_item():
    rng = np.random.default_rng(3)
    icao = rng.choice(list(AIRPORTS), size=3000)
    base = rng.normal(600, 200, 3000)
    other = np.full(3000, -1.0)
    choice = {a: ("o" if a == "LIRF" else "base") for a in AIRPORTS}
    pred = group_b.apply_choice(icao, base, {"base": base, "o": other}, choice)
    lirf = icao == "LIRF"
    assert np.all(pred[lirf] == -1.0)
    assert np.array_equal(pred[~lirf], base[~lirf])


def test_the_blend_weights():
    rng = np.random.default_rng(4)
    b, m = rng.normal(600, 200, 100000), rng.normal(600, 200, 100000)
    assert np.array_equal(group_b.blend(b, m, 0.5), np.mean(np.vstack([b, m]), axis=0))
    for w in RULE.grid:
        assert np.array_equal(group_b.blend(b, m, w), w * b + (1.0 - w) * m)
    assert [group_b.weight_item(w) for w in RULE.grid] == ["w0.3", "w0.4", "w0.6", "w0.7"]
    assert group_b.item_blend("w0.7") == ("combo4_booster", "c3b_member", 0.7)
    assert group_b.item_blend("c11") == ("combo4_booster", "c11_member", 0.5)
    assert group_b.item_blend("base") == ("combo4_booster", "c3b_member", 0.5)
    with pytest.raises(KeyError):
        group_b.item_blend("c14")


def test_the_menu_is_the_amendments_eight_items_in_order():
    g = board.GroupB()
    assert list(g.menu()) == [
        "base",
        "w0.3",
        "w0.4",
        "w0.6",
        "w0.7",
        "c11",
        "c12",
        "c13",
    ]


def test_w_months_strictly_before_a_forward_leg():
    assert board.w_fit_months("december") == MONTHS[:11]
    assert board.w_fit_months("july") == MONTHS[:6]
    assert board.w_fit_months("october") == MONTHS[:9]
    for m in MONTHS:
        got = board.w_fit_months(f"heldout-{m}")
        assert m not in got and len(got) == 11


def test_c12_params_change_only_the_trees():
    members = board.group_b_members()
    p = board.member_params(members["c12_booster"], 0)
    q = board.member_params(members["combo4_booster"], 0)
    assert p["n_estimators"] == RULE.c12_trees
    assert {k: v for k, v in p.items() if k != "n_estimators"} == {
        k: v for k, v in q.items() if k != "n_estimators"
    }
    assert (
        board.member_params(members["c13_booster"], 1)["random_state"]
        == q["random_state"] + 1
    )


def test_c11_member_columns_are_c3b_plus_eleven():
    c11 = board.group_b_members()["c11_member"]
    c3b = board.Member("c3b_member", backend="catboost")
    assert len(board.member_columns(c11)) == len(board.member_columns(c3b)) + 11
    assert c11.backend == "catboost" and c11.blocks == ("aobt3", "flight")


def test_the_plan_is_seventeen_fits_in_order():
    p = board.group_b_plan()
    assert len(p) == 17
    assert p[:3] == [("december", 0), ("july", 0), ("october", 0)]
    assert p[-2:] == [("december", 1), ("july", 1)]


def test_the_new_base_records_are_the_configurations_months():
    from taxiout.fusion import BASE_RECORDS

    assert [n.replace("__c14__", "__combo4__") for n in group_b.NEW_BASE_RECORDS] == list(
        BASE_RECORDS
    )


def test_base_per_airport_takes_each_airports_item():
    icao = np.array(["LIRF", "EDDF", "LIRF", "EGLL"])
    parts = {
        "combo4_booster": np.array([100.0, 200.0, 300.0, 400.0]),
        "c3b_member": np.array([110.0, 210.0, 310.0, 410.0]),
        "c11_member": np.array([120.0, 220.0, 320.0, 420.0]),
    }
    choice = {"LIRF": "w0.7", "EDDF": "c11", "EGLL": "c11"}
    got = group_b.base_per_airport(icao, parts, choice)
    want = np.array(
        [
            0.7 * 100.0 + (1.0 - 0.7) * 110.0,
            0.5 * 200.0 + 0.5 * 220.0,
            0.7 * 300.0 + (1.0 - 0.7) * 310.0,
            0.5 * 400.0 + 0.5 * 420.0,
        ]
    )
    assert np.array_equal(got, want)
    with pytest.raises(ValueError):
        group_b.base_per_airport(np.array(["LTFM"]), parts, choice)
