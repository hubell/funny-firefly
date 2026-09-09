# SPDX-License-Identifier: GPL-3.0-only
"""The plan-lag block, on a hand-built quarter hour at one runway."""

from __future__ import annotations

import inspect

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS  # noqa: E402
from taxiout.features import source_phase  # noqa: E402
from taxiout.planlags import (  # noqa: E402
    DIFF_SOURCES,
    MOVEMENT_COLUMNS,
    PLAN_DIFF_FEATURES,
    PLAN_LAG_FEATURES,
    SOURCES,
    plandiff_features,
    planlag_features,
)

PLAN_COLUMNS = ("EOBT_1_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt")

T0 = pd.Timestamp("2025-03-04T10:00:00Z")


def when(minutes):
    return pd.NaT if minutes is None else T0 + pd.Timedelta(minutes=minutes)


def departure(
    mvt_id, t, runway, *, eobt, lobt=None, sched=None, icao="ZZZZ", phase="DEP"
):
    return {
        "MVT_ID_mvt": float(mvt_id),
        "PHASE_mvt": phase,
        "ADEP_mvt": icao,
        "MVT_TIME_UTC_mvt": when(t),
        "RUNWAY_mvt": runway,
        "EOBT_1_flt": when(eobt),
        "LOBT_flt": when(eobt if lobt is None else lobt),
        "SCHED_TIME_UTC_mvt": when(eobt if sched is None else sched),
    }


def _field() -> pd.DataFrame:
    """The subject is 100, off runway 27 at t=60, planned for t=40."""
    return pd.DataFrame(
        [
            departure(100, 60, "27", eobt=40),  # lag 20 min
            departure(101, 57, "27", eobt=30),  # in the window, lag 27 min
            departure(102, 58, "09", eobt=20),  # other runway
            departure(103, 40, "27", eobt=10),  # 20 min before: outside
            departure(104, 60, "27", eobt=50),  # same second, lag 10 min
            departure(105, 70, "27", eobt=None, sched=55),  # after; no EOBT; sched lag 15
            departure(106, 75, "27", eobt=60),  # exactly fifteen minutes after: inside
            departure(107, 76, "27", eobt=60),  # just outside
            departure(200, 62, "27", eobt=50, phase="ARR"),  # an arrival: never read
        ]
    )


def _subject(m):
    return m[m["MVT_ID_mvt"] == 100.0].reset_index(drop=True)


def _plans(*, eobt=40, lobt=45, sched=25) -> pd.DataFrame:
    """One departure whose three planning values differ, in minutes past ``T0``."""
    return pd.DataFrame([departure(300, 60, "27", eobt=eobt, lobt=lobt, sched=sched)])


STAMP_COLUMNS = ("MVT_TIME_UTC_mvt",) + PLAN_COLUMNS


def _microseconds(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frame with every timestamp column held at microsecond resolution.

    Both stamps of a subtraction have to move: pandas resolves a mixed pair
    at the finer of the two, so a microsecond plan against a nanosecond
    take-off would come back to nanoseconds and hide the very thing this is
    built to expose.
    """
    out = frame.copy()
    for c in STAMP_COLUMNS:
        out[c] = out[c].dt.as_unit("us")
    return out


def test_the_subjects_own_lags_are_take_off_minus_the_plan():
    f = planlag_features(_subject(_field()), _field())
    assert f["lag_eobt_s"].iloc[0] == 20 * 60
    assert f["lag_lobt_s"].iloc[0] == 20 * 60
    assert f["lag_sched_s"].iloc[0] == 20 * 60
    assert f["lag_missing"].iloc[0] == 0.0


def test_a_missing_plan_is_nan_and_flagged():
    m = _field()
    f = planlag_features(m[m["MVT_ID_mvt"] == 105.0].reset_index(drop=True), m)
    assert np.isnan(f["lag_eobt_s"].iloc[0]) and np.isnan(f["lag_lobt_s"].iloc[0])
    assert f["lag_sched_s"].iloc[0] == 15 * 60
    assert f["lag_missing"].iloc[0] == 1.0


def test_the_runway_mean_is_over_the_others_within_fifteen_minutes():
    f = planlag_features(_subject(_field()), _field())
    # on 27 within [45, 75]: 101 (27 min), 104 (10 min), 105 (no EOBT), 106 (15 min);
    # not the subject, not 103, not 107, not the other runway, not the arrival
    assert f["lag_eobt_rwy15m"].iloc[0] == pytest.approx((27 + 10 + 15) * 60 / 3)
    # the schedule lag has 105 too: 27, 10, 15, 15
    assert f["lag_sched_rwy15m"].iloc[0] == pytest.approx((27 + 10 + 15 + 15) * 60 / 4)


def test_the_subject_is_removed_by_identity_not_by_time():
    m = _field()
    # the subject's own row absent from the movements: the mean is unchanged
    f_absent = planlag_features(_subject(m), m[m["MVT_ID_mvt"] != 100.0])
    f_present = planlag_features(_subject(m), m)
    assert f_absent["lag_eobt_rwy15m"].iloc[0] == f_present["lag_eobt_rwy15m"].iloc[0]
    # a twin at the same second with the same plan is another aircraft and counts
    twin = m.copy()
    twin.loc[twin["MVT_ID_mvt"] == 104.0, "EOBT_1_flt"] = when(40)
    f_twin = planlag_features(_subject(twin), twin)
    assert f_twin["lag_eobt_rwy15m"].iloc[0] == pytest.approx((27 + 20 + 15) * 60 / 3)


def test_a_runway_with_no_other_departure_has_no_mean():
    m = _field()
    lone = m[m["MVT_ID_mvt"].isin([100.0])].reset_index(drop=True)
    f = planlag_features(lone, lone)
    assert np.isnan(f["lag_eobt_rwy15m"].iloc[0])
    assert f["lag_eobt_s"].iloc[0] == 20 * 60


def test_the_block_carries_exactly_its_columns():
    f = planlag_features(_subject(_field()), _field())
    assert list(f.columns) == list(PLAN_LAG_FEATURES)
    assert f["lag_truncated"].iloc[0] == 0.0


def test_a_window_reaching_outside_the_month_is_flagged():
    m = _field()
    m["MVT_TIME_UTC_mvt"] = pd.Timestamp("2025-03-31T23:50:00Z")
    f = planlag_features(_subject(m), m)
    assert f["lag_truncated"].iloc[0] == 1.0


def test_sources_are_the_subject_and_other_departures_and_never_a_hidden_field():
    for name in PLAN_LAG_FEATURES:
        assert name in SOURCES
        for src in SOURCES[name]:
            phase, column = source_phase(src)
            assert phase in (None, "DEP"), (name, src)
            assert column not in FORBIDDEN_FEATURE_INPUTS, (name, src)
            assert column != "AOBT_3_flt"
    assert "PHASE_mvt" in MOVEMENT_COLUMNS and "EOBT_1_flt" in MOVEMENT_COLUMNS


def test_a_movement_frame_without_the_plans_is_refused():
    m = _field().drop(columns=["EOBT_1_flt"])
    with pytest.raises(KeyError):
        planlag_features(_subject(_field()), m)


def test_the_differences_are_the_plan_against_itself():
    f = plandiff_features(_plans())
    assert f["eobt_minus_sched_s"].iloc[0] == 15 * 60
    assert f["lobt_minus_eobt_s"].iloc[0] == 5 * 60


def test_a_missing_planning_value_takes_the_differences_that_read_it():
    both = plandiff_features(_plans(eobt=None))
    assert np.isnan(both["eobt_minus_sched_s"].iloc[0])
    assert np.isnan(both["lobt_minus_eobt_s"].iloc[0])
    no_lobt = _plans()
    no_lobt["LOBT_flt"] = pd.NaT
    f = plandiff_features(no_lobt)
    assert np.isnan(f["lobt_minus_eobt_s"].iloc[0])
    assert f["eobt_minus_sched_s"].iloc[0] == 15 * 60


def test_the_differences_are_indexed_like_the_departures():
    m = _plans()
    m.index = pd.Index([41])
    f = plandiff_features(m)
    assert list(f.index) == [41]
    assert list(f.columns) == list(PLAN_DIFF_FEATURES)


def test_the_differences_read_the_subjects_own_row_and_need_no_movements():
    assert list(inspect.signature(plandiff_features).parameters) == ["departures"]
    # no take-off, no runway, no identifier: the three planning values suffice
    f = plandiff_features(_plans()[list(PLAN_COLUMNS)])
    assert f["eobt_minus_sched_s"].iloc[0] == 15 * 60
    assert f["lobt_minus_eobt_s"].iloc[0] == 5 * 60


def test_the_difference_sources_are_exposed_planning_columns_of_the_subject():
    assert list(DIFF_SOURCES) == list(PLAN_DIFF_FEATURES)
    for name, sources in DIFF_SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase is None, (name, src)
            assert column not in FORBIDDEN_FEATURE_INPUTS, (name, src)
            assert column != "AOBT_3_flt"
            assert column in PLAN_COLUMNS, (name, src)


def test_both_blocks_are_the_same_whatever_resolution_the_stamps_arrive_at():
    """The parquet files carry microseconds; every other test here builds nanoseconds.

    Both blocks reach the same subtraction, so a resolution-blind one would
    take the lags and the differences down together, by a factor of a
    thousand on the real data and with no missing value or exception to show
    for it. ``tests/test_windows.py`` pins the property for the window
    machinery, which is where this repository first paid for it; the plan
    lags run through that machinery *and* subtract on their own, so the
    guard has to be repeated on this side.
    """
    plans = _microseconds(_plans())
    assert str(plans["EOBT_1_flt"].dtype) == "datetime64[us, UTC]"  # the setup took
    d = plandiff_features(plans)
    assert d["eobt_minus_sched_s"].iloc[0] == 15 * 60
    assert d["lobt_minus_eobt_s"].iloc[0] == 5 * 60

    field = _microseconds(_field())
    f = planlag_features(_subject(field), field)
    assert f["lag_eobt_s"].iloc[0] == 20 * 60
    assert f["lag_sched_s"].iloc[0] == 20 * 60
    assert f["lag_missing"].iloc[0] == 0.0
    # the runway mean too: it is these seconds averaged, not a fresh reading
    assert f["lag_eobt_rwy15m"].iloc[0] == pytest.approx((27 + 10 + 15) * 60 / 3)


def test_the_lags_and_the_differences_are_separate_blocks():
    assert set(PLAN_LAG_FEATURES).isdisjoint(PLAN_DIFF_FEATURES)
    assert set(SOURCES).isdisjoint(DIFF_SOURCES)
    # the differences carry no flag of their own: the lags' one already says it
    assert len(PLAN_DIFF_FEATURES) == 2
    assert "lag_missing" in SOURCES
