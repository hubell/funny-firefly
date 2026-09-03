# SPDX-License-Identifier: GPL-3.0-only
"""The congestion block, on a hand-built hour at one airport.

Every count here is checkable by eye, which is the point: a congestion
feature that is off by one is not wrong-looking, it is a slightly different
model, and the only way to know is to have written the answer down first.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import (  # noqa: E402
    AVAILABLE_FOR_ARRIVALS,
    FORBIDDEN_FEATURE_INPUTS,
)
from taxiout.congestion import (  # noqa: E402
    CONGESTION_FEATURES,
    configuration_sets,
    MOVEMENT_COLUMNS,
    SOURCES,
    congestion_features,
)
from taxiout.features import source_phase  # noqa: E402

T0 = pd.Timestamp("2025-03-04T10:00:00Z")


def when(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def movement(mvt_id, phase, t, runway, *, block=None, taxi=None, sched=None, icao="ZZZZ"):
    return {
        "MVT_ID_mvt": float(mvt_id),
        "PHASE_mvt": phase,
        "ADEP_mvt": icao if phase == "DEP" else "AAAA",
        "ADES_mvt": icao if phase == "ARR" else "AAAA",
        "MVT_TIME_UTC_mvt": when(t),
        "BLOCK_TIME_UTC_mvt": when(block) if block is not None else pd.NaT,
        "TAXITIME_SEC_mvt": taxi,
        "SCHED_TIME_UTC_mvt": when(sched) if sched is not None else when(t),
        "RUNWAY_mvt": runway,
    }


def _hour() -> pd.DataFrame:
    """One airport, one hour. The subject is id 100, taking off at t=60 on 27."""
    rows = [
        # departures: take-off time, runway, (schedule)
        movement(100, "DEP", 60, "27", sched=50),  # the subject
        movement(101, "DEP", 57, "27", sched=45),
        movement(102, "DEP", 58, "09", sched=52),
        movement(103, "DEP", 40, "27", sched=30),
        movement(104, "DEP", 60, "27", sched=55),  # same second as the subject
        movement(105, "DEP", 61, "27", sched=58),  # after the subject
        movement(106, "DEP", 0, "27", sched=-20),  # exactly sixty minutes before
        movement(107, "DEP", -1, "27", sched=-30),  # just outside the hour
        # arrivals: landing time, runway, on-block, taxi-in
        movement(200, "ARR", 55, "27", block=59, taxi=240.0),
        movement(201, "ARR", 50, "09", block=58, taxi=480.0),
        movement(202, "ARR", 20, "09", block=25, taxi=300.0),
        movement(203, "ARR", 59.5, "27", block=70, taxi=630.0),  # blocks after t
        movement(204, "ARR", 30, "09", block=35, taxi=None),  # no taxi-in
    ]
    return pd.DataFrame(rows)


def _subject(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["MVT_ID_mvt"] == 100.0].reset_index(drop=True)


def test_throughput_counts_the_runway_and_the_airport_and_not_the_subject():
    m = _hour()
    f = congestion_features(_subject(m), m)
    # runway 27 in the last five minutes: 101 (57), 104 (60, same second); not
    # 100 itself, not 105 (after), not 102 (other runway)
    assert f["tko_rwy_5m"].iloc[0] == 2.0
    assert f["tko_all_5m"].iloc[0] == 3.0
    # the last hour: 101, 103, 104, 106 (exactly on the boundary) on 27
    assert f["tko_rwy_60m"].iloc[0] == 4.0
    assert f["tko_all_60m"].iloc[0] == 5.0  # plus 102; 107 is outside


def test_arrivals_are_counted_by_landing_and_by_on_block():
    m = _hour()
    f = congestion_features(_subject(m), m)
    # landings in the last five minutes: 200 (55), 203 (59.5); 201 landed at 50
    assert f["ldg_5m"].iloc[0] == 2.0
    # on-blocks in the last five minutes: 200 (59), 201 (58); 203 blocks at 70
    assert f["onb_5m"].iloc[0] == 2.0
    assert f["txi_n_5m"].iloc[0] == 2.0
    assert f["txi_mean_5m"].iloc[0] == pytest.approx((240.0 + 480.0) / 2)
    # the last hour: 200, 201, 202 (block 25) carry a taxi-in; 204 does not
    assert f["onb_60m"].iloc[0] == 4.0
    assert f["txi_n_60m"].iloc[0] == 3.0
    assert f["txi_mean_60m"].iloc[0] == pytest.approx((240.0 + 480.0 + 300.0) / 3)


def test_demand_is_centred_on_the_schedule_and_excludes_the_subject():
    m = _hour()
    f = congestion_features(_subject(m), m)
    # subject scheduled at 50: within +-15 are 45, 52, 55, 58; 30 is not
    assert f["sched_pm15m"].iloc[0] == 4.0
    # within +-30: add 30 (103) and -20? no, -20 is 70 away
    assert f["sched_pm30m"].iloc[0] == 5.0


def test_a_subject_absent_from_the_movements_is_not_subtracted():
    """Removal is by identity: nothing is subtracted for a subject not in the events."""
    m = _hour()
    subject = _subject(m)
    others = m[m["MVT_ID_mvt"] != 100.0]
    f = congestion_features(subject, others)
    assert f["tko_rwy_5m"].iloc[0] == 2.0  # the same count, not one fewer
    assert f["sched_pm15m"].iloc[0] == 4.0


def test_configuration_reads_every_movement_and_ignores_sentinels():
    m = _hour()
    m.loc[m["MVT_ID_mvt"] == 202.0, "RUNWAY_mvt"] = "NA"  # a sentinel, not a runway
    f = congestion_features(_subject(m), m)
    # in the hour before t=60 both 27 and 09 have two or more movements
    assert "cfg_runways" not in f.columns  # the set is a diagnostic, not a feature
    assert configuration_sets(_subject(m), m).iloc[0] == "09|27"
    assert f["cfg_n_runways"].iloc[0] == 2.0
    assert np.isfinite(f["cfg_since_change_s"].iloc[0])


def test_truncation_flags_come_from_the_subjects_own_months():
    m = _hour()
    early = _subject(m).copy()
    early["MVT_TIME_UTC_mvt"] = pd.Timestamp("2025-03-01T00:30:00Z")
    early["SCHED_TIME_UTC_mvt"] = pd.Timestamp("2025-03-01T00:20:00Z")
    f = congestion_features(early, m)
    assert f["win_truncated"].iloc[0] == 1.0
    assert f["sched_truncated"].iloc[0] == 1.0
    f0 = congestion_features(_subject(m), m)
    assert f0["win_truncated"].iloc[0] == 0.0


def test_the_block_carries_exactly_its_columns_and_the_subjects_index():
    m = _hour()
    subject = _subject(m)
    subject.index = [7]
    f = congestion_features(subject, m)
    assert list(f.columns) == list(CONGESTION_FEATURES)
    assert f.index.tolist() == [7]


def test_movement_columns_are_exactly_the_prefixed_sources():
    stripped = {source_phase(c)[1] for s in SOURCES.values() for c in s if "." in c}
    assert set(MOVEMENT_COLUMNS) == stripped | {"PHASE_mvt"}


# --------------------------------------------------------------------------
# the rule, per phase


def test_departure_sources_are_never_hidden_fields():
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            if phase in (None, "DEP"):
                assert column not in FORBIDDEN_FEATURE_INPUTS, (name, src)


def test_arrival_sources_are_all_exposed_for_arrivals():
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            if phase == "ARR":
                assert column in AVAILABLE_FOR_ARRIVALS, (name, src)


def test_the_excluded_field_appears_under_no_phase():
    for sources in SOURCES.values():
        for src in sources:
            assert source_phase(src)[1] != "AOBT_3_flt"


def test_poisoning_departures_hidden_fields_moves_nothing():
    """The block reads arrivals' block and taxi times, and no departure's."""
    m = _hour()
    clean = congestion_features(_subject(m), m)
    poisoned = m.copy()
    dep = poisoned["PHASE_mvt"] == "DEP"
    poisoned.loc[dep, "BLOCK_TIME_UTC_mvt"] = when(59)
    poisoned.loc[dep, "TAXITIME_SEC_mvt"] = 99999.0
    after = congestion_features(_subject(poisoned), poisoned)
    pd.testing.assert_frame_equal(clean, after)


def test_poisoning_arrivals_fields_moves_the_arrival_features():
    """And the proof it reads arrivals: change theirs and the block changes."""
    m = _hour()
    clean = congestion_features(_subject(m), m)
    poisoned = m.copy()
    arr = poisoned["PHASE_mvt"] == "ARR"
    poisoned.loc[arr, "TAXITIME_SEC_mvt"] = 1.0
    after = congestion_features(_subject(m), poisoned)
    assert after["txi_mean_5m"].iloc[0] == 1.0
    assert clean["txi_mean_5m"].iloc[0] != 1.0
    assert after["tko_rwy_5m"].iloc[0] == clean["tko_rwy_5m"].iloc[0]
