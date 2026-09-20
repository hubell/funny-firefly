# SPDX-License-Identifier: GPL-3.0-only
"""The residual report's arithmetic and its refusal to publish a stand.

The generator reads stored residuals and writes aggregates. Two things are
worth holding to a test rather than to a reading of the code: that the
prediction it reconstructs is the one the ladder made, and that it will not
write a literal stand identifier -- the same guard ``report_occupancy.py``
carries, asserted here because a report is published and a mistake in it is
not recoverable.

The two descriptive sections added later are held to the same two things, on
synthetic frames: that neither publishes a stand it was handed, and that the
day-clustered interval refuses to speak where it rests on too few days -- one
day resampled with replacement is the same day every draw, and an interval
computed from it would report a confidence it has not got.
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
            "prefix": ["DLH", "DLH", "BAW"],
        }
    )


def _stored(residual, flagged) -> pd.DataFrame:
    """A residual file in the shape ``run_baselines.write_residuals`` writes."""
    return pd.DataFrame(
        {
            "mvt_id": [1.0, 2.0, 3.0],
            "icao": ["EDDF", "EDDF", "EGLL"],
            "runway": ["07C", "07C", "27R"],
            "takeoff": pd.to_datetime(
                ["2025-12-01T06:00:00Z", "2025-12-01T16:00:00Z", "2025-12-02T16:00:00Z"]
            ),
            "residual": residual,
            "flagged": flagged,
        }
    )


def test_the_prediction_is_the_target_minus_the_stored_residual(tmp_path, monkeypatch):
    """The ladder stores ``y - pred``; this report has to invert it exactly."""
    monkeypatch.setattr(rr, "RESIDUALS", tmp_path)
    _stored([300.0, -100.0, 50.0], [False, False, False]).to_parquet(
        tmp_path / f"december_{rr.RUNG}.parquet", index=False
    )

    got = rr.load_leg("december", None, _side())
    assert list(got["pred"]) == [900.0, 700.0, 2950.0]
    # and the squared error it sums is the residual's, not the prediction's
    assert list(got["sq"]) == [90000.0, 10000.0, 2500.0]


def test_flagged_rows_are_left_out_and_a_leg_can_be_restricted(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "RESIDUALS", tmp_path)
    _stored([300.0, -100.0, 50.0], [False, True, False]).to_parquet(
        tmp_path / f"december_{rr.RUNG}.parquet", index=False
    )

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


# --------------------------------------------------------------------------
# the two descriptive sections


def _leg(n: int, *, days: int, icao: str = "EGLL", seed: int = 0) -> pd.DataFrame:
    """A synthetic leg with everything the two sections read, over ``days`` days.

    The stand column carries a literal identifier so the guard has something to
    find if a section ever prints one.
    """
    rng = np.random.default_rng(seed)
    hours = pd.to_datetime("2025-12-01T05:00:00Z") + pd.to_timedelta(
        (np.arange(n) % (days * 18)) * 3600, unit="s"
    )
    residual = rng.normal(0.0, 200.0, n)
    return pd.DataFrame(
        {
            "mvt_id": np.arange(n, dtype=float),
            "icao": icao,
            "runway": np.where(np.arange(n) % 2 == 0, "27R", "09L"),
            "takeoff": hours,
            "residual": residual,
            TARGET: 900.0 + residual,
            "stand": "A12",
            "wake": "M",
            "no_record": np.arange(n) % 7 == 0,
            "lag_min": np.where(np.arange(n) % 11 == 0, np.nan, np.arange(n) % 90),
            "reoccupied": False,
            "prefix": np.where(np.arange(n) % 3 == 0, "BAW", "DLH"),
            "pred": 900.0,
            "sq": residual**2,
        }
    )


def _guard(text: str) -> None:
    """The report's own guard, on one section's text."""
    from run_baselines import check_no_identifiers

    matrix = pd.DataFrame(
        {
            "stand_id": ["A12", "A12"],
            "designator": ["27R", "09L"],
            "stand": ["A12", "A12"],
        }
    )
    check_no_identifiers(text, {"A12"}, matrix)


def test_the_july_section_publishes_no_stand_it_was_handed():
    """It cuts by runway and by callsign prefix; neither may leak a stand."""
    text = "\n".join(rr.section_heathrow(_leg(400, days=20), _leg(400, days=20, seed=1)))
    _guard(text)
    assert "A12" not in text
    assert "27R" in text and "BAW" in text


def test_the_weather_section_publishes_no_stand_it_was_handed():
    hourly = pd.DataFrame(
        {
            "station": "EGLL",
            "hour": pd.date_range("2025-12-01T05:00:00Z", periods=400, freq="h"),
            "vis_band": 1.0,
            "precip": 0.0,
            "freezing": 0.0,
            "age_h": 0.0,
        }
    )
    text = "\n".join(rr.section_weather(_leg(400, days=20), hourly))
    _guard(text)
    assert "A12" not in text


def test_the_weather_section_says_so_without_a_cache():
    text = "\n".join(rr.section_weather(_leg(10, days=2), None))
    assert "No METAR cache" in text


def test_an_interval_needs_more_days_than_it_has_clusters():
    """One day resampled with replacement is one day, and says nothing."""
    thin = _leg(60, days=1)
    wide = _leg(600, days=30, seed=2)
    got, days = rr._difference(thin, wide)
    assert got is None and days == 1
    cell, phrase = rr._shown(thin, wide)
    assert cell == phrase == "1 day, no interval"

    got, days = rr._difference(wide, _leg(600, days=30, seed=3))
    assert got is not None and days >= rr.MIN_INTERVAL_DAYS
    assert rr._shown(wide, _leg(600, days=30, seed=3))[0].startswith("[")


def test_the_lag_cut_names_the_rows_the_lag_table_leaves_out():
    frame = pd.DataFrame({"lag_min": [5.0, np.nan, -30.0, 1000.0]})
    got = list(rr._lag_level(frame))
    assert got == ["0-15", rr.NO_ESTIMATE, rr.BEFORE_ESTIMATE, "over 120"]


def test_the_peak_hours_are_read_off_the_squared_error():
    frame = pd.DataFrame(
        {
            "takeoff": pd.to_datetime(
                ["2025-12-01T05:00:00Z", "2025-12-01T06:00:00Z", "2025-12-01T07:00:00Z"]
            ),
            "sq": [1.0, 100.0, 50.0],
        }
    )
    assert rr._peak_hours(frame, 2) == ["06", "07"]
    assert rr._hours_label(["06", "07"]) == "06:00 to 07:59 UTC"
    assert rr._hours_label(["06", "08"]) == "06:00, 08:00 UTC"
