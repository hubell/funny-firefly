# SPDX-License-Identifier: GPL-3.0-only
"""The plan-lag block, on a hand-built quarter hour at one runway."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS  # noqa: E402
from taxiout.features import source_phase  # noqa: E402
from taxiout.planlags import (  # noqa: E402
    MOVEMENT_COLUMNS,
    PLAN_LAG_FEATURES,
    SOURCES,
    planlag_features,
)

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
