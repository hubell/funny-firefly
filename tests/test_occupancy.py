# SPDX-License-Identifier: GPL-3.0-only
"""The stand occupancy block, on a hand-built afternoon at four stands.

Every gap here is a subtraction of two stamps written down in the fixture,
which is the point: a gap that is off by one arrival is not wrong-looking,
it is a different feature, and the only way to know is to have written the
answer first.

The frames are built at microsecond resolution, as the challenge parquet
files are. ``taxiout.windows.seconds`` once divided an integer view of them
as if it were nanoseconds, and the unit tests of the day built nanosecond
stamps and never saw it (``tests/test_windows.py``).
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import (  # noqa: E402
    AVAILABLE_FOR_ARRIVALS,
    FORBIDDEN_FEATURE_INPUTS,
)
from taxiout.features import source_phase  # noqa: E402
from taxiout.occupancy import (  # noqa: E402
    MOVEMENT_COLUMNS,
    OCCUPANCY_FEATURES,
    SOURCES,
    occupancy_features,
)

# What this block reads from an arrival's row, written out so the registry is
# held to a list stated here and not only to the shared constant. Each is
# exposed in the ranking set for arrivals: the operator at 0.76% null, which
# is the share of arrivals whose Network Manager record was not found, and
# the rest at 0.00% (the null-rate table of ``reports/data_review.md``).
READ_FROM_ARRIVALS: tuple[str, ...] = (
    "ADES_mvt",
    "STAND_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_OPERATOR_flt",
)

T0 = pd.Timestamp("2025-03-04T10:00:00Z")
STAMP_COLUMNS = ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")


def when(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def at(text: str) -> pd.Timestamp:
    """An absolute stamp, for the rows that sit at the start of the covered month."""
    return pd.Timestamp(text)


def _stamp(value):
    return value if isinstance(value, pd.Timestamp) else when(value)


def movement(
    mvt_id,
    phase,
    t,
    stand,
    *,
    block=None,
    taxi=None,
    sched=None,
    kind="A320",
    operator="OP1",
    icao="ZZZZ",
):
    return {
        "MVT_ID_mvt": float(mvt_id),
        "PHASE_mvt": phase,
        "ADEP_mvt": icao if phase == "DEP" else "AAAA",
        "ADES_mvt": icao if phase == "ARR" else "AAAA",
        "MVT_TIME_UTC_mvt": _stamp(t),
        "BLOCK_TIME_UTC_mvt": _stamp(block) if block is not None else pd.NaT,
        "TAXITIME_SEC_mvt": taxi,
        "SCHED_TIME_UTC_mvt": _stamp(sched) if sched is not None else _stamp(t),
        "STAND_mvt": stand,
        "AIRCRAFT_TYPE_mvt": kind,
        "AIRCRAFT_OPERATOR_flt": operator,
    }


def _micro(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame at the resolution the challenge files carry."""
    for c in STAMP_COLUMNS:
        frame[c] = pd.to_datetime(frame[c], utc=True).dt.as_unit("us")
    return frame


def _afternoon() -> pd.DataFrame:
    """One airport, four stands. Minutes are from 2025-03-04T10:00Z."""
    rows = [
        # A12: the subject 100 takes off at 60 having off-blocked at 40, and a
        # different aircraft on-blocked at 50, between the two
        movement(100, "DEP", 60, "A12", block=40, taxi=1200.0, sched=45),
        movement(
            200,
            "ARR",
            45,
            "A12",
            block=50,
            taxi=300.0,
            sched=40,
            kind="B738",
            operator="OP2",
        ),
        movement(
            201,
            "ARR",
            15,
            "A12",
            block=20,
            taxi=280.0,
            sched=25,
            kind="B738",
            operator="OP2",
        ),
        movement(203, "ARR", 70, "A12", block=75, taxi=300.0),  # after the take-off
        movement(110, "DEP", 20, "A12", block=5, taxi=900.0),  # the departure before
        # B01: a fast turnaround, the subject's own inbound on the stand
        movement(101, "DEP", 60, "B01", block=45, taxi=900.0, sched=50),
        movement(202, "ARR", 30, "B01", block=35, taxi=300.0, sched=30),
        # C03: no arrival at all -- the one row at the stand never on-blocked
        movement(102, "DEP", 60, "C03", block=45, taxi=900.0, sched=50),
        movement(204, "ARR", 55, "C03", block=None, taxi=None),
        # E05: an arrival on-blocking in the subject's own take-off second
        movement(104, "DEP", 60, "E05", block=40, taxi=1200.0, sched=50),
        movement(205, "ARR", 50, "E05", block=60, taxi=600.0, sched=55),
        # a subject with no stand, and an arrival with none either
        movement(105, "DEP", 60, None, block=40, taxi=1200.0),
        movement(206, "ARR", 55, None, block=58, taxi=180.0),
        # D04: the first day of the covered month, six hours in
        movement(
            300,
            "ARR",
            at("2025-03-01T04:30Z"),
            "D04",
            block=at("2025-03-01T05:00Z"),
            taxi=180.0,
        ),
        movement(103, "DEP", at("2025-03-01T06:00Z"), "D04", taxi=900.0),
    ]
    return _micro(pd.DataFrame(rows))


def _subject(frame: pd.DataFrame, mvt_id: float = 100.0) -> pd.DataFrame:
    return frame[frame["MVT_ID_mvt"] == mvt_id].reset_index(drop=True)


def test_the_frames_are_built_at_the_resolution_the_parquet_files_carry():
    m = _afternoon()
    for c in STAMP_COLUMNS:
        assert str(m[c].dtype) == "datetime64[us, UTC]"
    assert occupancy_features(_subject(m), m)["occ_gap1_s"].iloc[0] == 600.0


def test_the_last_on_block_at_the_stand_bounds_the_taxi_out():
    m = _afternoon()
    f = occupancy_features(_subject(m), m)
    # on-block at 50, take-off at 60: ten minutes, and the subject's own
    # taxi-out of twenty is at least that
    assert f["occ_gap1_s"].iloc[0] == 600.0
    assert f["occ_gap1_s"].iloc[0] < _subject(m)["TAXITIME_SEC_mvt"].iloc[0]
    # scheduled off at 45, the inbound on-blocked at 50: five minutes late for it
    assert f["occ_gap1_sched_s"].iloc[0] == -300.0
    # that arrival was itself due at 40 and blocked at 50
    assert f["occ_prev_late_s"].iloc[0] == 600.0
    assert f["occ_prev_txi_s"].iloc[0] == 300.0
    assert f["occ_missing"].iloc[0] == 0.0 and f["occ_truncated"].iloc[0] == 0.0


def test_the_second_last_on_block_is_the_second_gap():
    m = _afternoon()
    f = occupancy_features(_subject(m), m)
    assert f["occ_gap2_s"].iloc[0] == 2400.0  # on-block at 20, take-off at 60
    # the stand has two arrivals before the take-off and one after it
    lone = m[m["MVT_ID_mvt"] != 201.0]
    assert np.isnan(occupancy_features(_subject(m), lone)["occ_gap2_s"].iloc[0])


def test_a_fast_turnaround_matches_on_both_type_and_operator():
    m = _afternoon()
    f = occupancy_features(_subject(m, 101.0), m)
    assert f["occ_gap1_s"].iloc[0] == 1500.0
    assert f["occ_prev_same_type"].iloc[0] == 1.0
    assert f["occ_prev_same_op"].iloc[0] == 1.0


def test_a_different_aircraft_on_the_stand_matches_neither():
    m = _afternoon()
    f = occupancy_features(_subject(m), m)
    assert f["occ_prev_same_type"].iloc[0] == 0.0
    assert f["occ_prev_same_op"].iloc[0] == 0.0


def test_a_null_field_on_either_side_leaves_the_match_unknown():
    m = _afternoon()
    m.loc[m["MVT_ID_mvt"] == 200.0, "AIRCRAFT_OPERATOR_flt"] = None
    f = occupancy_features(_subject(m), m)
    assert np.isnan(f["occ_prev_same_op"].iloc[0])
    assert f["occ_prev_same_type"].iloc[0] == 0.0  # the type is still known
    subject = _subject(m)
    subject["AIRCRAFT_TYPE_mvt"] = None
    g = occupancy_features(subject, m)
    assert np.isnan(g["occ_prev_same_type"].iloc[0])


def test_an_arrival_on_blocking_in_the_take_off_second_has_happened():
    m = _afternoon()
    f = occupancy_features(_subject(m, 104.0), m)
    assert f["occ_gap1_s"].iloc[0] == 0.0
    assert f["occ_missing"].iloc[0] == 0.0


def test_a_stand_with_no_arrival_well_inside_the_block_is_missing():
    m = _afternoon()
    f = occupancy_features(_subject(m, 102.0), m)
    # the one arrival row at C03 never on-blocked, so it is not an event: its
    # sentinel time must not become the gap
    assert np.isnan(f["occ_gap1_s"].iloc[0]) and np.isnan(f["occ_gap2_s"].iloc[0])
    assert np.isnan(f["occ_prev_late_s"].iloc[0])
    assert np.isnan(f["occ_prev_same_type"].iloc[0])
    assert f["occ_missing"].iloc[0] == 1.0 and f["occ_truncated"].iloc[0] == 0.0


def test_the_same_stand_at_the_start_of_the_block_is_truncated_and_not_missing():
    m = _afternoon()
    subject = _subject(m, 102.0)
    subject["MVT_TIME_UTC_mvt"] = pd.Series(
        [pd.Timestamp("2025-03-01T06:00:00Z")]
    ).dt.as_unit("us")
    f = occupancy_features(subject, m)
    assert f["occ_truncated"].iloc[0] == 1.0 and f["occ_missing"].iloc[0] == 0.0


def test_a_first_day_departure_whose_stand_received_an_arrival_is_flagged_neither():
    m = _afternoon()
    f = occupancy_features(_subject(m, 103.0), m)
    assert f["occ_gap1_s"].iloc[0] == 3600.0  # 05:00 to 06:00 on the first day
    assert f["occ_missing"].iloc[0] == 0.0 and f["occ_truncated"].iloc[0] == 0.0


def test_the_previous_departure_from_the_stand_excludes_the_subject():
    m = _afternoon()
    f = occupancy_features(_subject(m), m)
    assert f["occ_prev_dep_gap_s"].iloc[0] == 2400.0  # 110 took off at 20, not itself
    # and the same with the subject absent from the movement frame
    g = occupancy_features(_subject(m), m[m["MVT_ID_mvt"] != 100.0])
    assert g["occ_prev_dep_gap_s"].iloc[0] == 2400.0
    # a stand this subject is the first to leave has no previous departure
    assert np.isnan(occupancy_features(_subject(m, 102.0), m)["occ_prev_dep_gap_s"][0])


def test_a_departure_from_the_stand_in_the_same_second_is_one_of_the_others():
    """Closed at the take-off second there too, and the subject out by identity."""
    m = _afternoon()
    twin = _subject(m, 110.0)
    twin["MVT_ID_mvt"] = 111.0
    twin["MVT_TIME_UTC_mvt"] = _subject(m)["MVT_TIME_UTC_mvt"].to_numpy()
    m = _micro(pd.concat([m, twin], ignore_index=True))
    assert occupancy_features(_subject(m), m)["occ_prev_dep_gap_s"].iloc[0] == 0.0


def test_a_stand_written_differently_in_case_and_spacing_still_matches():
    m = _afternoon()
    m.loc[m["MVT_ID_mvt"] == 200.0, "STAND_mvt"] = "a 12"
    m.loc[m["MVT_ID_mvt"] == 110.0, "STAND_mvt"] = " A12 "
    f = occupancy_features(_subject(m), m)
    assert f["occ_gap1_s"].iloc[0] == 600.0
    assert f["occ_prev_dep_gap_s"].iloc[0] == 2400.0


def test_a_subject_without_a_stand_matches_nothing():
    m = _afternoon()
    f = occupancy_features(_subject(m, 105.0), m)
    assert np.isnan(f["occ_gap1_s"].iloc[0])
    assert np.isnan(f["occ_prev_dep_gap_s"].iloc[0])
    assert f["occ_missing"].iloc[0] == 1.0


def test_the_block_carries_exactly_its_columns_and_the_subjects_index():
    m = _afternoon()
    subject = _subject(m)
    subject.index = [7]
    f = occupancy_features(subject, m)
    assert list(f.columns) == list(OCCUPANCY_FEATURES)
    assert f.index.tolist() == [7]


def test_every_subject_at_once_reads_what_each_reads_alone():
    """The block is vectorised over the subjects; one row and all rows agree."""
    m = _afternoon()
    subjects = m[m["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    together = occupancy_features(subjects, m)
    for i, mvt_id in enumerate(subjects["MVT_ID_mvt"]):
        alone = occupancy_features(_subject(m, mvt_id), m)
        for c in OCCUPANCY_FEATURES:
            a, b = together[c].iloc[i], alone[c].iloc[0]
            assert (a == b) or (np.isnan(a) and np.isnan(b)), (mvt_id, c)


def test_movements_without_a_needed_column_raise():
    m = _afternoon()
    with pytest.raises(KeyError):
        occupancy_features(_subject(m), m.drop(columns=["BLOCK_TIME_UTC_mvt"]))


# --------------------------------------------------------------------------
# the rule, per phase


def test_the_registry_names_every_column_and_nothing_else():
    assert list(SOURCES) == list(OCCUPANCY_FEATURES)


def test_movement_columns_are_exactly_the_prefixed_sources():
    stripped = {source_phase(c)[1] for s in SOURCES.values() for c in s if "." in c}
    assert set(MOVEMENT_COLUMNS) == stripped | {"PHASE_mvt"}


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
                assert column in READ_FROM_ARRIVALS, (name, src)
                assert column in AVAILABLE_FOR_ARRIVALS, (name, src)


def test_the_excluded_field_appears_under_no_phase():
    for sources in SOURCES.values():
        for src in sources:
            assert source_phase(src)[1] != "AOBT_3_flt"


def test_poisoning_departures_hidden_fields_moves_nothing():
    """The block reads arrivals' block and taxi times, and no departure's."""
    m = _afternoon()
    clean = occupancy_features(_subject(m), m)
    poisoned = m.copy()
    dep = poisoned["PHASE_mvt"] == "DEP"
    poisoned.loc[dep, "BLOCK_TIME_UTC_mvt"] = when(59)
    poisoned.loc[dep, "TAXITIME_SEC_mvt"] = 99999.0
    after = occupancy_features(_subject(poisoned), poisoned)
    pd.testing.assert_frame_equal(clean, after)


def test_poisoning_the_arrivals_on_blocks_moves_the_gaps():
    """And the proof it reads arrivals: change theirs and the block changes."""
    m = _afternoon()
    poisoned = m.copy()
    arr = poisoned["PHASE_mvt"] == "ARR"
    poisoned.loc[arr, "TAXITIME_SEC_mvt"] = 1.0
    after = occupancy_features(_subject(m), poisoned)
    assert after["occ_prev_txi_s"].iloc[0] == 1.0
    assert after["occ_gap1_s"].iloc[0] == 600.0
