# SPDX-License-Identifier: GPL-3.0-only
"""The reading rules of ``scripts/candidates_board.py``, on constructed figures.

Nothing here fits a booster or reads challenge data. The rule's numbers are
checked against the published copy of the board pre-registration, and against
the amendment's working note where it is on hand.
"""

from __future__ import annotations

import os
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

RULE = board.board_rule()


def _words(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_importing_leaves_the_earlier_harness_where_it_was():
    assert cand.WORK == REPO / "data" / "processed" / "candidates"
    assert REPO / "data" / "processed" / "candidates" not in cand.FORBIDDEN
    assert REPO / "data" / "processed" / "candidates" in board.FORBIDDEN


def test_the_numbers_are_section_3s_words():
    text = _words(REPO / board.BOARD_TEXT)
    assert (
        "`gamma` if λ >= 1.75, else `tweedie` with variance power λ clipped to [1.1, 1.9] and rounded to one decimal"
        in text
    )
    assert "the target is floored at 1 s for this member" in text
    assert (
        "loss RMSE, 3,000 iterations, learning rate 0.08, depth 8, L2 3, 254 borders"
        in text
    )
    assert "CatBoost on CPU at eight threads" in text
    assert "takes more than 45 minutes" in text
    assert "7 October 2026, 12:00 UTC" in text
    assert "two blend members at equal weights" in text
    assert (RULE.park_gamma_at, RULE.tweedie_low, RULE.tweedie_high) == (1.75, 1.1, 1.9)
    assert RULE.tweedie_decimals == 1 and RULE.target_floor_s == 1.0
    c = RULE.catboost
    assert (c["iterations"], c["learning_rate"], c["depth"]) == (3000, 0.08, 8)
    assert (c["l2_leaf_reg"], c["border_count"], c["thread_count"]) == (3, 254, 8)
    assert c["loss_function"] == "RMSE"
    assert RULE.catboost_gpu_after_s == 45 * 60
    assert RULE.deadline_utc == cand.the_rule().deadline_utc
    assert RULE.blend_weight == 0.5


def test_the_floors_are_section_4s_words():
    text = _words(REPO / board.BOARD_TEXT)
    for words in (
        "better by more than 0.23 s on December over all ten",
        "better by more than 0.54 s on July over all ten",
        "not worse by more than 1.02 s on the three July airports",
        "not worse by more than 0.21 s on October",
        "by more than 0.16 s",
    ):
        assert words in text
    r = cand.the_rule()
    assert (r.floor_december, r.floor_july, r.floor_july_three) == (0.23, 0.54, 1.02)
    assert (r.floor_october, r.floor_year) == (0.21, 0.16)


def test_the_amendment_where_it_is_on_hand():
    notes = os.environ.get("PREREGISTRATION_NOTES")
    path = Path(notes) / "preregistration_2026-10-04_c7_c8.md" if notes else None
    if path is None or not path.exists():
        pytest.skip("the amendment's working note is not on hand")
    from taxiout.provenance import digest

    assert digest(path) == RULE.amendment_sha256
    text = _words(path)
    assert (
        "LightGBM, objective binary, 200 trees, learning rate 0.05, 31 leaves, minimum child 500"
        in text
    )
    assert "clipped to [0.1, 10], then scaled to a mean of 1" in text
    assert "the folds being the even and the odd calendar days" in text


def test_the_park_choice():
    assert board.choose_objective(1.75) == ("gamma", None)
    assert board.choose_objective(2.4) == ("gamma", None)
    assert board.choose_objective(1.74) == ("tweedie", 1.7)
    assert board.choose_objective(1.46) == ("tweedie", 1.5)
    assert board.choose_objective(0.3) == ("tweedie", 1.1)
    assert board.choose_objective(1.749) == ("tweedie", 1.7)


def test_the_layer_months():
    legs = {s.name: s for s in cand.leg_plan()}
    assert board.months_before(legs["december"]) == [
        f"2025-{m:02d}" for m in range(1, 12)
    ]
    assert board.months_before(legs["july"]) == [f"2025-{m:02d}" for m in range(1, 7)]
    assert board.months_before(legs["october"]) == [f"2025-{m:02d}" for m in range(1, 10)]


def _kept(n=50, seed=0):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(300, 900, n)
    return pd.DataFrame(
        {
            "mvt_id": np.arange(n, dtype=float),
            "icao": rng.choice(["EDDF", "LSZH"], n),
            "runway": "25C",
            "takeoff": pd.Timestamp("2025-01-01T10:00:00Z"),
            "term": np.arange(n) == 0,
            "shift": np.arange(n) == 1,
            "y": pred * 1.1 + 20 + rng.normal(0, 5, n),
            "pred": pred,
            "with_term": np.where(np.arange(n) == 0, 5000.0, pred),
        }
    )


def test_the_recalibration_is_least_squares_per_airport():
    kept = _kept(200)
    layer = board.fit_layer("recal", kept, None)
    for a, g in kept.groupby("icao"):
        beta, alpha = np.polyfit(g["pred"], g["y"], 1)
        assert layer[a][0] == pytest.approx(alpha)
        assert layer[a][1] == pytest.approx(beta)
    got = board.apply_layer("recal", layer, kept, None)
    a = kept["icao"].to_numpy()
    want = (
        np.array([layer[x][0] for x in a])
        + np.array([layer[x][1] for x in a]) * kept["pred"]
    )
    np.testing.assert_allclose(got, want)


def test_a_replaced_prediction_keeps_the_terms_value_on_its_rows():
    kept = _kept()
    out = board.replace_pred(kept, kept["pred"] + 7.0)
    assert out.loc[0, "with_term"] == 5000.0
    np.testing.assert_allclose(out.loc[1:, "with_term"], kept.loc[1:, "pred"] + 7.0)
    np.testing.assert_allclose(out["pred"], kept["pred"] + 7.0)


def test_the_layers_rows_are_neither_rules_and_four_hours_or_less():
    kept = _kept()
    kept.loc[2, "y"] = 5 * 3600.0
    rows = board.oof_rows(kept)
    assert not rows[0] and not rows[1] and not rows[2]
    assert rows[3:].all()


def _reading(contrast):
    r = {"rows": {"contrast": contrast}, "short": {"contrast": contrast}}
    return r


def _measured(dec, jul, three, octo, year=None, dec1=None, jul1=None):
    m = {
        "seed0": {
            "december:all": _reading(dec),
            "july:all": _reading(jul),
            "july:three": _reading(three),
            "october:all": _reading(octo),
        }
    }
    if year is not None:
        m["seed0"]["year:all"] = _reading(year)
    if dec1 is not None:
        m["seed1"] = {"december:all": _reading(dec1), "july:all": _reading(jul1)}
    return m


def test_the_status_follows_the_rule():
    assert board.verdict_of({})["status"] == "legs not measured"
    assert board.verdict_of(_measured(-0.23, -0.6, 0, 0))["status"] == "fails the four"
    assert (
        board.verdict_of(_measured(-0.24, -0.6, 1.02, 0.21))["status"]
        == "months not measured"
    )
    assert (
        board.verdict_of(_measured(-0.24, -0.6, 1.03, 0.21))["status"] == "fails the four"
    )
    assert (
        board.verdict_of(_measured(-0.3, -0.6, 0, 0, 0.17))["status"] == "fails the fifth"
    )
    assert (
        board.verdict_of(_measured(-0.3, -0.6, 0, 0, 0.16))["status"]
        == "second seed not measured"
    )
    assert (
        board.verdict_of(_measured(-0.3, -0.6, 0, 0, 0.0, -0.01, 0.004))["status"]
        == "fails the sign check"
    )
    assert (
        board.verdict_of(_measured(-0.3, -0.6, 0, 0, 0.0, -0.01, -0.2))["status"]
        == "clears"
    )


def test_the_amendments_two_run_before_the_costliest():
    order = [k for n in sorted(board.stages()) for k in board.stages()[n]]
    assert order == ["c2a", "c5", "c1", "c3a", "c4", "c7", "c8", "c3b"]
    assert set(order) == set(board.candidates())


def test_member_columns_append_to_the_rung():
    from taxiout.models import LADDER_BY_KEY, SUBMISSION_KEY

    base = tuple(LADDER_BY_KEY[SUBMISSION_KEY].features)
    cands = board.candidates()
    assert board.member_columns(cands["c1"].member)[: len(base)] == base
    assert len(board.member_columns(cands["c7"].member)) == len(base) + 5
    assert board.member_columns(cands["c8"].member) == base
