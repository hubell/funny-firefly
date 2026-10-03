# SPDX-License-Identifier: GPL-3.0-only
"""The every-month report's legs and guards, on synthetic frames.

Nothing here fits a booster or reads the challenge data. What is held to a
test is the part of ``scripts/report_monthly.py`` that would be wrong without
announcing it: a held-out leg that trains on the month it scores, a forward
leg that is not the ladder's, a dry run that fits, and a report that prints an
identifier.
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

import preregistered as pr  # noqa: E402
import report_monthly as rm  # noqa: E402


# --------------------------------------------------------------------------
# the legs


def _held(legs):
    return [leg for leg in legs if leg.kind == "held-out"]


def test_there_are_two_forward_legs_and_twelve_held_out_ones_forward_first():
    legs = rm.leg_plan()
    assert [leg.kind for leg in legs] == ["forward"] * 2 + ["held-out"] * 12
    assert [leg.valid_month for leg in _held(legs)] == list(rm.MONTHS)


def test_no_held_out_leg_trains_on_the_month_it_validates():
    for leg in _held(rm.leg_plan()):
        assert leg.valid_month not in leg.train_months


def test_every_held_out_leg_trains_on_the_other_eleven_months():
    for leg in _held(rm.leg_plan()):
        assert len(leg.train_months) == 11
        assert set(leg.train_months) | {leg.valid_month} == set(rm.MONTHS)


def test_the_forward_legs_are_the_ladders():
    forward = {leg.name: leg for leg in rm.leg_plan() if leg.kind == "forward"}
    for ladder in ("december", "july"):
        leg = forward[f"forward-{ladder}"]
        assert max(leg.train_months) == pr.LAST_TRAIN_MONTH[ladder]
        # Every month up to the last training month, nothing after it.
        assert leg.train_months == tuple(
            m for m in rm.MONTHS if m <= pr.LAST_TRAIN_MONTH[ladder]
        )
        assert leg.valid_month > max(leg.train_months)


def test_the_row_masks_follow_the_months_and_never_overlap():
    month = pd.Series(["2025-01", "2025-02", "2025-07", "2025-12", "2025-12"])
    for leg in rm.build_legs(month):
        assert not bool((leg.train & leg.valid).any())
        assert int(leg.valid.sum()) == int((month == leg.valid_month).sum())
    held = _held(rm.build_legs(month))
    # Each row is validated exactly once across the twelve.
    assert sum(int(leg.valid.sum()) for leg in held) == len(month)


# --------------------------------------------------------------------------
# --dry-run fits nothing


def test_the_dry_run_fits_nothing_and_writes_nothing(monkeypatch, tmp_path, capsys):
    takeoff = pd.Series(pd.date_range("2025-01-01", "2025-12-31", freq="7D", tz="UTC"))
    monkeypatch.setattr(
        rm.rb,
        "load_departures",
        lambda _frac: pd.DataFrame({"MVT_TIME_UTC_mvt": takeoff}),
    )
    monkeypatch.setattr(rm, "WORK", tmp_path / "monthly")
    monkeypatch.setattr(rm, "FORBIDDEN", (tmp_path / "reports",))

    def refuse(*_a, **_k):
        raise AssertionError("the dry run fitted")

    monkeypatch.setattr(rm, "fit_leg", refuse)
    monkeypatch.setattr(rm.Booster, "fit", refuse)
    monkeypatch.setattr(rm, "Frame", refuse)
    assert rm.main(["--dry-run"]) == 0
    assert not (tmp_path / "monthly").exists()
    assert "nothing was touched" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the report refuses an identifier


def test_the_report_refuses_a_stand_identifier():
    with pytest.raises(SystemExit, match="stand"):
        rm.check_no_identifiers("the worst is at A12 in May", {"A12"}, set())


def test_the_report_refuses_a_movement_identifier():
    with pytest.raises(SystemExit, match="movement"):
        rm.check_no_identifiers("a row 81234567 moved", set(), {"81234567"})


def test_the_report_allows_an_airport_code_and_a_figure():
    rm.check_no_identifiers("LIRF reads 294.4 s on 12,068 rows", {"LIRF"}, {"81234567"})


# --------------------------------------------------------------------------
# the reproduction check and the cell arithmetic


def _residuals(ids, values):
    return pd.DataFrame({"mvt_id": ids, "residual": values})


def test_the_reproduction_check_passes_only_on_an_exact_match(tmp_path):
    stored = tmp_path / "s.parquet"
    _residuals([1, 2, 3], [1.0, -2.0, 3.0]).to_parquet(stored)
    assert rm.reproduces(_residuals([3, 1, 2], [3.0, 1.0, -2.0]), stored)[1]
    assert not rm.reproduces(_residuals([1, 2, 3], [1.0, -2.0, 3.0 + 1e-9]), stored)[1]
    assert not rm.reproduces(_residuals([1, 2], [1.0, -2.0]), stored)[1]
    assert not rm.reproduces(_residuals([1, 2, 4], [1.0, -2.0, 3.0]), stored)[1]
    assert not rm.reproduces(_residuals([1], [1.0]), tmp_path / "absent.parquet")[1]


def test_a_cell_drops_the_flagged_rows_only_from_the_ex_flagged_figure():
    frame = pd.DataFrame(
        {
            "residual": [3.0, -4.0, 100.0],
            "flagged": [False, False, True],
            rm.TERM_COLUMN: [3.0, -4.0, 0.0],
        }
    )
    got = rm.cell(frame)
    assert got["term_rows"] == 1
    assert got["ex_flagged"] == pytest.approx(np.sqrt((9 + 16) / 2))
    assert got["with_term"] == pytest.approx(np.sqrt((9 + 16) / 3))
    assert got["booster"] == pytest.approx(np.sqrt((9 + 16 + 10000) / 3))
    # Prediction minus target: the residual's sign reversed.
    assert got["bias"] == pytest.approx(-(3 - 4 + 100) / 3)
