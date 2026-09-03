# SPDX-License-Identifier: GPL-3.0-only
"""The saturation block, on a hand-built runway sequence."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS  # noqa: E402
from taxiout.features import source_phase  # noqa: E402
from taxiout.saturation import (  # noqa: E402
    SATURATION_FEATURES,
    SOURCES,
    saturation_features,
)

T0 = pd.Timestamp("2025-03-04T10:00:00Z")


def when(seconds_):
    return T0 + pd.Timedelta(seconds=seconds_)


def dep(mvt_id, t_s, runway="27", icao="ZZZZ", phase="DEP"):
    return {
        "MVT_ID_mvt": float(mvt_id),
        "PHASE_mvt": phase,
        "ADEP_mvt": icao,
        "MVT_TIME_UTC_mvt": when(t_s),
        "RUNWAY_mvt": runway,
    }


def _runway() -> pd.DataFrame:
    """Take-offs on 27 at 0, 1020, 1110, 1140 s; the subject 100 at 1200 s."""
    return pd.DataFrame(
        [
            dep(1, 0),  # twenty minutes before the subject
            dep(2, 1020),  # 3 min before
            dep(3, 1110),  # 90 s after 2: not under ninety
            dep(4, 1140),  # 30 s after 3
            dep(100, 1200),  # the subject
            dep(5, 1230),  # after the subject: never read
            dep(6, 1190, runway="09"),  # another runway
            dep(200, 1195, phase="ARR"),  # an arrival: never read
        ]
    )


def _one(m, mvt_id=100.0):
    return m[m["MVT_ID_mvt"] == mvt_id].reset_index(drop=True)


def test_the_share_of_short_gaps_and_the_smallest_gap_exclude_the_subject():
    m = _runway()
    f = saturation_features(_one(m), m)
    # trailing five minutes [900, 1200]: others at 1020, 1110, 1140; gaps 90, 30
    assert f["sat_gap90_5m"].iloc[0] == pytest.approx(1 / 2)
    assert f["sat_min_gap_5m"].iloc[0] == 30.0
    # fifteen minutes [300, 1200]: the same three
    assert f["sat_gap90_15m"].iloc[0] == pytest.approx(1 / 2)


def test_the_subject_in_the_middle_of_the_sequence_is_bridged():
    # a subject at 1125, between 1110 and 1140: with it out, the gap 1110->1140 is 30 s
    m = pd.concat([_runway(), pd.DataFrame([dep(101, 1125)])], ignore_index=True)
    f = saturation_features(_one(m, 101.0), m)
    # window [825, 1125]: others 1020, 1110 (1140 is after); one gap of 90 s
    assert f["sat_gap90_5m"].iloc[0] == 0.0
    assert f["sat_min_gap_5m"].iloc[0] == 90.0
    # the original subject, at 1200, now sees 1020, 1110, 1125, 1140: gaps 90, 15, 15
    g = saturation_features(_one(m), m)
    assert g["sat_gap90_5m"].iloc[0] == pytest.approx(2 / 3)
    assert g["sat_min_gap_5m"].iloc[0] == 15.0


def test_fewer_than_two_others_in_the_window_is_missing():
    m = _runway()
    f = saturation_features(
        _one(m, 2.0), m
    )  # at 1020: only 0 before, twenty minutes back
    assert np.isnan(f["sat_gap90_5m"].iloc[0]) and np.isnan(f["sat_min_gap_5m"].iloc[0])
    assert np.isnan(f["sat_gap90_15m"].iloc[0])


def test_the_span_of_the_last_three_gaps_is_the_subject_minus_the_third_before():
    m = _runway()
    f = saturation_features(_one(m), m)
    # before 1200 on 27: 0, 1020, 1110, 1140 -> the third before is 1020
    assert f["sat_span3_s"].iloc[0] == 180.0
    assert f["sat_span3_short"].iloc[0] == 0.0
    # a subject at 1110 has 0 and 1020 before it: fewer than three, so capped and flagged
    g = saturation_features(_one(m, 3.0), m)
    assert g["sat_span3_s"].iloc[0] == 1800.0 and g["sat_span3_short"].iloc[0] == 1.0
    # a subject at 1140 has 0, 1020, 1110 before it: the third before is 0
    h = saturation_features(_one(m, 4.0), m)
    assert h["sat_span3_s"].iloc[0] == 1140.0 and h["sat_span3_short"].iloc[0] == 0.0
    # and one forty minutes after a lone predecessor sits at the cap, unflagged
    far = pd.concat([m, pd.DataFrame([dep(8, 1200 + 2400)])], ignore_index=True)
    k = saturation_features(_one(far, 8.0), far)
    assert k["sat_span3_s"].iloc[0] == 1800.0 and k["sat_span3_short"].iloc[0] == 0.0


def test_a_take_off_at_the_subjects_own_second_is_one_of_the_others():
    m = pd.concat([_runway(), pd.DataFrame([dep(7, 1200)])], ignore_index=True)
    f = saturation_features(_one(m), m)
    # others in [900, 1200]: 1020, 1110, 1140, 1200 -> gaps 90, 30, 60
    assert f["sat_gap90_5m"].iloc[0] == pytest.approx(2 / 3)
    assert f["sat_min_gap_5m"].iloc[0] == 30.0
    assert f["sat_span3_s"].iloc[0] == 90.0  # third before: 1110


def test_the_subject_absent_from_the_movements_reads_the_same_sequence():
    m = _runway()
    f = saturation_features(_one(m), m[m["MVT_ID_mvt"] != 100.0])
    g = saturation_features(_one(m), m)
    for c in SATURATION_FEATURES:
        assert (f[c].iloc[0] == g[c].iloc[0]) or (
            np.isnan(f[c].iloc[0]) and np.isnan(g[c].iloc[0])
        )


def test_a_runway_never_seen_is_missing_and_short():
    m = _runway()
    s = _one(m)
    s["RUNWAY_mvt"] = "36"
    f = saturation_features(s, m)
    assert np.isnan(f["sat_gap90_5m"].iloc[0])
    assert f["sat_span3_short"].iloc[0] == 1.0 and f["sat_span3_s"].iloc[0] == 1800.0


def test_sources_are_other_departures_take_off_times_only():
    assert list(SOURCES) == list(SATURATION_FEATURES)
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (None, "DEP"), (name, src)
            assert column not in FORBIDDEN_FEATURE_INPUTS
            assert column in (
                "MVT_TIME_UTC_mvt",
                "ADEP_mvt",
                "RUNWAY_mvt",
                "MVT_ID_mvt",
            ), (name, src)
