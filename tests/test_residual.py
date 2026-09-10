# SPDX-License-Identifier: GPL-3.0-only
"""The residual report's arithmetic and its refusal to publish a stand.

The generator reads stored residuals and writes aggregates. Two things are
worth holding to a test rather than to a reading of the code: that the
prediction it reconstructs is the one the ladder made, and that it will not
write a literal stand identifier -- the same guard ``report_occupancy.py``
carries, asserted here because a report is published and a mistake in it is
not recoverable.
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

import report_residual as rr  # noqa: E402
from taxiout.features import TARGET  # noqa: E402


def _side() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mvt_id": [1.0, 2.0, 3.0],
            "icao": ["EDDF", "EDDF", "EGLL"],
            "stand": ["A12", "B01", "C03"],
            "month": ["2025-12"] * 3,
            TARGET: [1200.0, 600.0, 3000.0],
            "wake": ["H", "M", "J"],
            "no_record": [False, True, False],
            "lag_min": [20.0, 50.0, np.nan],
            "reoccupied": [True, False, False],
        }
    )


def test_the_prediction_is_the_target_minus_the_stored_residual(tmp_path, monkeypatch):
    """The ladder stores ``y - pred``; this report has to invert it exactly."""
    monkeypatch.setattr(rr, "RESIDUALS", tmp_path)
    pd.DataFrame(
        {
            "mvt_id": [1.0, 2.0, 3.0],
            "icao": ["EDDF", "EDDF", "EGLL"],
            "residual": [300.0, -100.0, 50.0],
            "flagged": [False, False, False],
        }
    ).to_parquet(tmp_path / f"december_{rr.RUNG}.parquet", index=False)

    got = rr.load_leg("december", None, _side())
    assert list(got["pred"]) == [900.0, 700.0, 2950.0]
    # and the squared error it sums is the residual's, not the prediction's
    assert list(got["sq"]) == [90000.0, 10000.0, 2500.0]


def test_flagged_rows_are_left_out_and_a_leg_can_be_restricted(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "RESIDUALS", tmp_path)
    pd.DataFrame(
        {
            "mvt_id": [1.0, 2.0, 3.0],
            "icao": ["EDDF", "EDDF", "EGLL"],
            "residual": [300.0, -100.0, 50.0],
            "flagged": [False, True, False],
        }
    ).to_parquet(tmp_path / f"december_{rr.RUNG}.parquet", index=False)

    assert list(rr.load_leg("december", None, _side())["mvt_id"]) == [1.0, 3.0]
    assert list(rr.load_leg("december", ("EGLL",), _side())["mvt_id"]) == [3.0]


def test_a_missing_residual_file_is_reported_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "RESIDUALS", tmp_path)
    assert rr.load_leg("december", None, _side()) is None


def test_the_lag_bands_are_contiguous_and_open_at_the_top():
    assert rr.LAG_LABELS[0] == "0-15"
    assert rr.LAG_LABELS[-1] == "over 120"
    assert len(rr.LAG_LABELS) == len(rr.LAG_EDGES_MIN) - 1


def test_the_generator_refuses_to_write_a_literal_stand_identifier():
    """The guard `report_occupancy.py` carries, on this report's own text."""
    from run_baselines import check_no_identifiers

    matrix = pd.DataFrame(
        {
            "stand_id": ["A12", "B01"],
            "designator": ["09L", "27R"],
            "stand": ["A12", "B01"],
        }
    )
    with pytest.raises(SystemExit):
        check_no_identifiers("the worst row was at stand A12", {"A12"}, matrix)
    # a report that names only airports and runway designators is fine
    check_no_identifiers("EDDF on 09L, 1,200 s", {"A12"}, matrix)


def test_the_report_reads_the_rung_the_submission_is_fitted_from():
    """Not whichever rung is convenient: the error reported is the shipped one."""
    from taxiout.models import SUBMISSION_KEY

    assert rr.RUNG == SUBMISSION_KEY


def test_the_three_legs_include_the_july_airports_alone():
    labels = [label for _s, _i, label in rr.LEGS]
    assert any("the three" in label for label in labels)
    restricted = [icaos for _s, icaos, _l in rr.LEGS if icaos is not None]
    assert restricted and all(set(i) == set(rr.JULY_AIRPORTS) for i in restricted)
