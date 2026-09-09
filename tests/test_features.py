# SPDX-License-Identifier: GPL-3.0-only
"""The feature frame, and the registry that says what every column reads.

The registry test is the one that matters. ``docs/modelling_notes.md`` states
the leakage rule as a paragraph; ``taxiout.challenge.FORBIDDEN_FEATURE_INPUTS``
states it as a constant; this asserts the feature layer against the constant,
so a feature that reads a blanked field fails the suite rather than scoring
well in training and badly on the leaderboard.

There are two halves to it. The registry must not *declare* a forbidden
source, and the builder must not *read* one -- a registry can be right while
the code beside it is wrong. The second half is checked by poisoning the
forbidden columns and asserting the frame does not move.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import (  # noqa: E402
    AVAILABLE_FOR_ARRIVALS,
    FORBIDDEN_FEATURE_INPUTS,
    SOURCE_PHASES,
)
from taxiout.encoding import ENCODING_FEATURES  # noqa: E402
from taxiout.features import (  # noqa: E402
    AIRCRAFT_FEATURES,
    CATEGORICAL_FEATURES,
    EXTERNAL_SOURCES,
    FEATURE_COLUMNS,
    FEATURE_SOURCES,
    FITTED_SOURCES,
    GEOMETRY_FEATURES,
    MOVEMENT_COLUMNS,
    QUALITY_FEATURES,
    REQUIRED_COLUMNS,
    TIME_FEATURES,
    airport_fallback,
    build_features,
    feature_block,
    source_phase,
    time_features,
)
from taxiout.routing import AGGREGATED_FIELDS  # noqa: E402


def _matrix(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """(icao, stand_id, designator, taxi_m) cells with plausible descriptors."""
    recs = []
    for icao, stand_id, designator, taxi_m in rows:
        rec = {
            "icao": icao,
            "stand_id": stand_id,
            "designator": designator,
            "runway_ref": f"{designator}/xx",
            "n_features": 1,
            "spread_m": 0.0,
            "snap_max_m": 0.0,
        }
        rec.update({f: 0.0 for f in AGGREGATED_FIELDS})
        rec["taxi_m"] = taxi_m
        rec["route_m"] = taxi_m
        rec["straight_m"] = 0.5 * taxi_m
        rec["turn_deg"] = 90.0
        rec["junctions"] = 4.0
        rec["gc_m"] = 0.8 * taxi_m
        rec["detour_ratio"] = 1.25
        recs.append(rec)
    return pd.DataFrame(recs)


def _movements(stands: list[str], **overrides) -> pd.DataFrame:
    n = len(stands)
    frame = pd.DataFrame(
        {
            "MVT_ID_mvt": [float(1000 + i) for i in range(n)],
            "ADEP_mvt": ["ZZZZ"] * n,
            "ADES_mvt": ["AAAA"] * n,
            "STAND_mvt": stands,
            "RUNWAY_mvt": ["27"] * n,
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-03-04T08:30:00Z"] * n, utc=True),
            "SCHED_TIME_UTC_mvt": pd.to_datetime(["2025-03-04T08:00:00Z"] * n, utc=True),
            "LOBT_flt": pd.to_datetime(["2025-03-04T08:10:00Z"] * n, utc=True),
            "IOBT_flt": pd.to_datetime(["2025-03-04T08:00:00Z"] * n, utc=True),
            "EOBT_1_flt": pd.to_datetime(["2025-03-04T08:00:00Z"] * n, utc=True),
            "WK_TBL_CAT_flt": ["M"] * n,
            "AIRCRAFT_TYPE_mvt": ["A320"] * n,
            "AIRCRAFT_OPERATOR_flt": ["OP1"] * n,
            "FLIGHT_mvt": [f"ZZZ{100 + i}" for i in range(n)],
        }
    )
    for k, v in overrides.items():
        frame[k] = v
    return frame


# --------------------------------------------------------------------------
# the registry


def test_registry_covers_the_frame_exactly():
    assert set(FEATURE_SOURCES) == set(FEATURE_COLUMNS)
    assert len(FEATURE_SOURCES) == len(FEATURE_COLUMNS)


def test_every_source_names_a_known_phase_or_none():
    for name, sources in FEATURE_SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (
                None,
                *SOURCE_PHASES,
                *EXTERNAL_SOURCES,
                *FITTED_SOURCES,
            ), (name, src)
            assert column and "." not in column, (name, src)


def test_the_fitting_sample_prefix_names_the_target_and_nothing_else():
    """The source class the registry gained for the shrunk encoding.

    ``TRAIN.`` says a feature reads the *target* of the rows the model is
    fitted on. That is a different claim from ``DEP.TAXITIME_SEC_mvt``, which
    would be false -- the ranking set blanks the taxi time of every departure
    -- and the point of giving it a prefix of its own is that it cannot be
    mistaken for one. Nothing but the target may appear under it: an ordinary
    field of another row has a phase, and a phase is held to that phase's list.
    """
    from taxiout.features import FITTED_COLUMNS, TARGET

    assert FITTED_COLUMNS == (TARGET,)
    used = {
        name: [src for src in sources if source_phase(src)[0] in FITTED_SOURCES]
        for name, sources in FEATURE_SOURCES.items()
    }
    for name, sources in used.items():
        for src in sources:
            assert source_phase(src)[1] in FITTED_COLUMNS, (name, src)
    # and only the encoding block declares one
    assert {n for n, s in used.items() if s} == set(ENCODING_FEATURES)


def test_the_encoding_block_declares_the_keys_and_the_fitted_target():
    from taxiout.features import TARGET

    for c in ENCODING_FEATURES:
        assert FEATURE_SOURCES[c] == (
            "ADEP_mvt",
            "STAND_mvt",
            "RUNWAY_mvt",
            f"TRAIN.{TARGET}",
        )


def test_every_external_source_is_a_field_of_its_archive():
    """Each block from outside the challenge data reads its own archive's fields.

    There are two, and each is held to its own list rather than to a shared
    one: a field of the METAR archive is not a field of OurAirports, and a
    single pooled list would let either block read the other's columns without
    the audit noticing.
    """
    from taxiout.destination import FIELDS as OURAIRPORTS_FIELDS
    from taxiout.metar import FIELDS as METAR_FIELDS

    allowed = {"METAR": METAR_FIELDS, "OURAIRPORTS": OURAIRPORTS_FIELDS}
    assert set(allowed) == set(EXTERNAL_SOURCES)
    seen = set()
    for name, sources in FEATURE_SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            if phase in allowed:
                assert column in allowed[phase], (name, src)
                seen.add(phase)
    # both prefixes are actually used; a prefix nobody reads is a list that
    # cannot be wrong and therefore proves nothing
    assert seen == set(allowed)


def test_the_holiday_block_is_a_dependency_and_not_a_dataset():
    """`holidays` computes calendars from rules, so it registers no external source."""
    from taxiout.calendar import HOLIDAY_FEATURES

    for c in HOLIDAY_FEATURES:
        assert FEATURE_SOURCES[c] == ("ADEP_mvt", "MVT_TIME_UTC_mvt")


def test_no_departure_source_is_a_hidden_field():
    """The subject's own row and any other departure's: nothing hidden."""
    forbidden = set(FORBIDDEN_FEATURE_INPUTS)
    offending = {}
    for name, sources in FEATURE_SOURCES.items():
        bad = [
            src
            for src in sources
            if source_phase(src)[0] in (None, "DEP") and source_phase(src)[1] in forbidden
        ]
        if bad:
            offending[name] = bad
    assert offending == {}


def test_every_arrival_source_is_exposed_for_arrivals():
    """An arrival's block time and taxi-in time are exposed, and read as such."""
    offending = {}
    for name, sources in FEATURE_SOURCES.items():
        bad = [
            src
            for src in sources
            if source_phase(src)[0] == "ARR"
            and source_phase(src)[1] not in AVAILABLE_FOR_ARRIVALS
        ]
        if bad:
            offending[name] = bad
    assert offending == {}


def test_the_excluded_field_is_read_under_no_phase():
    for name, sources in FEATURE_SOURCES.items():
        for src in sources:
            assert source_phase(src)[1] != "AOBT_3_flt", (name, src)


def test_required_columns_are_the_bare_sources_and_movement_columns_the_rest():
    bare = {
        source_phase(c)[1]
        for s_ in FEATURE_SOURCES.values()
        for c in s_
        if source_phase(c)[0] is None
    }
    prefixed = {
        source_phase(c)[1]
        for s_ in FEATURE_SOURCES.values()
        for c in s_
        if source_phase(c)[0] not in (None, *EXTERNAL_SOURCES, *FITTED_SOURCES)
    }
    assert set(REQUIRED_COLUMNS) == bare
    assert set(MOVEMENT_COLUMNS) == prefixed | {"PHASE_mvt"}
    assert not bare & set(FORBIDDEN_FEATURE_INPUTS)


def test_every_feature_declares_at_least_one_source():
    assert all(len(s_) >= 1 for s_ in FEATURE_SOURCES.values())


# --------------------------------------------------------------------------
# what the builder actually reads


def _with_arrivals(departures: pd.DataFrame) -> pd.DataFrame:
    """A movement frame: the departures plus two arrivals at the same airport.

    Both arrivals took stand ``B5``, which is where the subjects push back
    from, so the stand-occupancy block has something to read as well as the
    congestion block. The later of the two on-blocked two minutes before the
    subjects' take-off and flew the subjects' own type under a different
    operator, so the two match flags are 1.0 and 0.0 rather than both alike.
    """
    t = departures["MVT_TIME_UTC_mvt"].iloc[0]
    minutes = pd.Timedelta(minutes=1)
    arrivals = pd.DataFrame(
        {
            "MVT_ID_mvt": [5000.0, 5001.0],
            "PHASE_mvt": ["ARR", "ARR"],
            "ADEP_mvt": ["AAAA", "AAAA"],
            "ADES_mvt": ["ZZZZ", "ZZZZ"],
            "STAND_mvt": ["B5", "B5"],
            "MVT_TIME_UTC_mvt": [t - 8 * minutes, t - 12 * minutes],
            "BLOCK_TIME_UTC_mvt": [t - 2 * minutes, t - 4 * minutes],
            "TAXITIME_SEC_mvt": [360.0, 480.0],
            "SCHED_TIME_UTC_mvt": [t - 10 * minutes, t - 15 * minutes],
            "RUNWAY_mvt": ["27", "27"],
            "AIRCRAFT_TYPE_mvt": ["A320", "B738"],
            "AIRCRAFT_OPERATOR_flt": ["OP2", "OP2"],
            "FLIGHT_mvt": ["AAA900", "AAA901"],
            # an arrival carries no departure plan, as in the data
            "EOBT_1_flt": [pd.NaT, pd.NaT],
            "LOBT_flt": [pd.NaT, pd.NaT],
        }
    )
    dep = departures.assign(
        PHASE_mvt="DEP",
        BLOCK_TIME_UTC_mvt=pd.NaT,
        TAXITIME_SEC_mvt=np.nan,
    )
    return pd.concat(
        [dep[list(MOVEMENT_COLUMNS)], arrivals[list(MOVEMENT_COLUMNS)]],
        ignore_index=True,
    )


def test_builder_ignores_the_forbidden_columns():
    """Poison every hidden field of every departure; the frame must not move.

    A registry entry is a claim about the code. This checks the code -- on
    the subject's own row and on every other departure in the movement frame.
    """
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    base = _movements(["B5", "B5"])
    moves = _with_arrivals(base)
    clean = build_features(base, m, movements=moves)

    poisoned = base.copy()
    for c in FORBIDDEN_FEATURE_INPUTS:
        poisoned[c] = (
            [1.0, 99999.0]
            if c.endswith("_SEC_mvt")
            else pd.to_datetime(
                ["1999-01-01T00:00:00Z", "2031-12-31T23:59:00Z"], utc=True
            )
        )
    poisoned_moves = moves.copy()
    dep = poisoned_moves["PHASE_mvt"] == "DEP"
    poisoned_moves.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.Timestamp("2031-12-31T23:59:00Z")
    poisoned_moves.loc[dep, "TAXITIME_SEC_mvt"] = 99999.0
    after = build_features(poisoned, m, movements=poisoned_moves)
    pd.testing.assert_frame_equal(clean, after)


def test_builder_reads_arrivals_fields_and_only_arrivals():
    """Change the arrivals' taxi-in times: the arrival features move, nothing else.

    Two blocks read that field and both must move: the congestion block's
    trailing mean, and the stand-occupancy block's record of what the aircraft
    before this one on the stand had just done.
    """
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    base = _movements(["B5", "B5"])
    moves = _with_arrivals(base)
    clean = build_features(base, m, movements=moves)
    changed = moves.copy()
    arr = changed["PHASE_mvt"] == "ARR"
    changed.loc[arr, "TAXITIME_SEC_mvt"] = 1.0
    after = build_features(base, m, movements=changed)
    assert (after["txi_mean_15m"] == 1.0).all()
    assert (clean["txi_mean_15m"] == 420.0).all()
    assert (clean["occ_prev_txi_s"] == 360.0).all()
    assert (after["occ_prev_txi_s"] == 1.0).all()
    moved = [c for c in FEATURE_COLUMNS if not clean[c].equals(after[c])]
    assert set(moved) == {c for c in FEATURE_COLUMNS if c.startswith("txi_mean_")} | {
        "occ_prev_txi_s"
    }


def test_the_occupancy_block_reads_the_arrivals_on_block_times():
    """Move the arrivals' on-blocks past the take-off: the `occ_` columns go missing.

    The stand-occupancy block is the one that reads a field the ranking set
    hides for departures, from the rows that expose it. This is the proof that
    it reads *arrivals'* on-block times and not something correlated with them:
    push every arrival's on-block to after the subject's take-off and the stand
    has no previous occupant, so every column that describes one goes to NaN
    and `occ_missing` fires. What moves with it is the congestion block's own
    on-block counts, which read the same field; the landing counts, which read
    the landing time, do not.
    """
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    base = _movements(["B5", "B5"])
    moves = _with_arrivals(base)
    clean = build_features(base, m, movements=moves)
    assert clean["occ_gap1_s"].tolist() == [120.0, 120.0]
    assert clean["occ_gap2_s"].tolist() == [240.0, 240.0]
    # the schedule against that on-block: 08:00 minus 08:28, so the inbound
    # reached the stand well after the outbound was due off it
    assert clean["occ_gap1_sched_s"].tolist() == [-1680.0, -1680.0]
    assert clean["occ_prev_late_s"].tolist() == [480.0, 480.0]
    # the last arrival flew the subjects' type under a different operator
    assert clean["occ_prev_same_type"].tolist() == [1.0, 1.0]
    assert clean["occ_prev_same_op"].tolist() == [0.0, 0.0]
    assert clean["occ_missing"].tolist() == [0.0, 0.0]

    changed = moves.copy()
    arr = changed["PHASE_mvt"] == "ARR"
    changed.loc[arr, "BLOCK_TIME_UTC_mvt"] = changed.loc[
        arr, "MVT_TIME_UTC_mvt"
    ] + pd.Timedelta(minutes=20)
    after = build_features(base, m, movements=changed)
    assert after["occ_gap1_s"].isna().all()
    assert after["occ_missing"].tolist() == [1.0, 1.0]
    # the take-off is three days into the covered month, so the absence is a
    # fact about the stand and not about the edge of the data
    assert after["occ_truncated"].tolist() == [0.0, 0.0]

    moved = {c for c in FEATURE_COLUMNS if not clean[c].equals(after[c])}
    assert moved == {
        "occ_gap1_s",
        "occ_gap2_s",
        "occ_gap1_sched_s",
        "occ_prev_late_s",
        "occ_prev_txi_s",
        "occ_prev_same_type",
        "occ_prev_same_op",
        "occ_missing",
    } | {c for c in FEATURE_COLUMNS if c.startswith(("onb_", "txi_n_", "txi_mean_"))}


def test_time_features_are_anchored_on_take_off():
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    frame = _movements(["B5"])
    early = build_features(frame, m)
    later = build_features(
        frame.assign(MVT_TIME_UTC_mvt=pd.to_datetime(["2025-03-04T19:45:00Z"], utc=True)),
        m,
    )
    moved = [c for c in TIME_FEATURES if early[c].iloc[0] != later[c].iloc[0]]
    assert "tko_minute_of_day" in moved and "tko_hour_sin" in moved
    # and nothing outside the time block moved with it
    for c in GEOMETRY_FEATURES + QUALITY_FEATURES:
        assert early[c].iloc[0] == later[c].iloc[0] or (
            np.isnan(early[c].iloc[0]) and np.isnan(later[c].iloc[0])
        )


def test_time_features_are_cyclic_and_scaled():
    t = pd.Series(
        pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T12:00:00Z"], utc=True)
    )
    f = time_features(t)
    assert f["tko_minute_of_day"].tolist() == [0.0, 720.0]
    assert f["tko_hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert f["tko_hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert f["tko_hour_cos"].iloc[1] == pytest.approx(-1.0, abs=1e-6)


# --------------------------------------------------------------------------
# the airport-level fallback


def test_unmatched_stand_falls_back_to_the_airport_level():
    m = _matrix(
        [
            ("ZZZZ", "B5", "27", 1000.0),
            ("ZZZZ", "B6", "27", 3000.0),
            ("ZZZZ", "B5", "09", 5000.0),
        ]
    )
    frame = _movements(["B5", "X1"])
    f = build_features(frame, m)
    assert f["geom_imputed"].tolist() == [0.0, 1.0]
    assert f["taxi_m"].iloc[0] == pytest.approx(1000.0)
    # the median of the two cells that reach runway 27, not of all three
    assert f["taxi_m"].iloc[1] == pytest.approx(2000.0)
    assert not f[list(GEOMETRY_FEATURES)].iloc[1].isna().any()


def test_fallback_uses_the_airport_when_the_runway_names_no_runway():
    m = _matrix([("ZZZZ", "B5", "27", 1000.0), ("ZZZZ", "B5", "09", 3000.0)])
    frame = _movements(["X1"], RUNWAY_mvt=["NA"])
    f = build_features(frame, m)
    assert f["geom_imputed"].iloc[0] == 1.0
    assert f["taxi_m"].iloc[0] == pytest.approx(2000.0)


def test_airport_fallback_has_both_levels():
    m = _matrix([("ZZZZ", "B5", "27", 1000.0), ("ZZZZ", "B5", "09", 3000.0)])
    fb = airport_fallback(m)
    assert fb["designator"].isna().sum() == 1
    assert set(fb.loc[fb["designator"].notna(), "designator"]) == {"09", "27"}


# --------------------------------------------------------------------------
# quality flags and shape ratios


def test_quality_flags_record_the_join_that_happened():
    m = _matrix([("ZZZZ", "101", "27", 1000.0), ("ZZZZ", "101L", "27", 1400.0)])
    frame = _movements(["101", "101R", "X1"])
    f = build_features(frame, m)
    assert f["join_stage"].tolist() == [1.0, 3.0, 0.0]
    # 101R only matches once the trailing letter is stripped, which folds two
    # identifiers onto one key; the cell records how far they disagree
    assert f["n_ids"].iloc[1] == 2.0
    assert f["id_spread_m"].iloc[1] == pytest.approx(400.0)
    # an unmatched row has no cell, so it has no ambiguity to report either
    assert f["n_ids"].iloc[2] == 0.0
    assert f["geom_imputed"].iloc[2] == 1.0


def test_shape_ratios_are_normalised_by_route_length():
    m = _matrix([("ZZZZ", "B5", "27", 2000.0)])
    f = build_features(_movements(["B5"]), m)
    assert f["straight_frac"].iloc[0] == pytest.approx(0.5)
    assert f["turn_deg_per_km"].iloc[0] == pytest.approx(45.0)
    assert f["junctions_per_km"].iloc[0] == pytest.approx(2.0)


def test_plan_delay_is_lobt_minus_iobt():
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    f = build_features(_movements(["B5"]), m)
    assert f["lobt_iobt_s"].iloc[0] == pytest.approx(600.0)
    assert f["lobt_iobt_missing"].iloc[0] == 0.0

    frame = _movements(["B5"])
    frame["LOBT_flt"] = pd.Series([pd.NaT], dtype="datetime64[ns, UTC]")
    g = build_features(frame, m)
    assert g["lobt_iobt_missing"].iloc[0] == 1.0
    assert g["lobt_iobt_s"].iloc[0] == 0.0


# --------------------------------------------------------------------------
# shape of the result


def test_frame_shape_and_dtypes():
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    frame = _movements(["B5", "X1"])
    f = build_features(frame, m)
    assert list(f.columns) == list(FEATURE_COLUMNS)
    assert f.index.equals(frame.index)
    for c in CATEGORICAL_FEATURES:
        assert str(f[c].dtype) == "category"
    for c in set(FEATURE_COLUMNS) - set(CATEGORICAL_FEATURES):
        assert f[c].dtype == np.float32


def test_missing_input_column_is_an_error_not_a_null_column():
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    frame = _movements(["B5"]).drop(columns=["IOBT_flt"])
    with pytest.raises(KeyError):
        build_features(frame, m)


def test_feature_block_selects_in_frame_order():
    assert feature_block(("tko_dow", "icao")) == ["icao", "tko_dow"]
    with pytest.raises(KeyError):
        feature_block(("not_a_feature",))


# --------------------------------------------------------------------------
# the aircraft


def test_aircraft_columns_are_categorical_and_come_from_the_exposed_fields():
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    frame = _movements(
        ["B5", "B5"], WK_TBL_CAT_flt=["H", "M"], AIRCRAFT_TYPE_mvt=["B77W", "A320"]
    )
    f = build_features(frame, m)
    assert set(AIRCRAFT_FEATURES) <= set(CATEGORICAL_FEATURES)
    assert f["wake_class"].tolist() == ["H", "M"]
    assert f["aircraft_type"].tolist() == ["B77W", "A320"]
    assert str(f["wake_class"].dtype) == "category"


def test_a_null_class_is_a_level_not_a_hole():
    """A null wake class is the unjoined block, which is information."""
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    frame = _movements(["B5", "B5"])
    frame["WK_TBL_CAT_flt"] = pd.Categorical(["M", None])
    frame["AIRCRAFT_TYPE_mvt"] = pd.Categorical(["A320", None])
    f = build_features(frame, m)
    assert f["wake_class"].tolist() == ["M", "none"]
    assert f["aircraft_type"].tolist() == ["A320", "none"]
    assert not f[list(AIRCRAFT_FEATURES)].isna().any().any()


def test_aircraft_sources_are_the_two_exposed_columns():
    from taxiout.features import FEATURE_SOURCES

    assert FEATURE_SOURCES["wake_class"] == ("WK_TBL_CAT_flt",)
    assert FEATURE_SOURCES["aircraft_type"] == ("AIRCRAFT_TYPE_mvt",)


# --------------------------------------------------------------------------
# the ladder


def test_the_ladder_reads_the_congestion_blocks_net_of_the_plan_lags_and_keeps_them_without():
    from taxiout.models import (
        LADDER,
        LADDER_BY_KEY,
        TOP_RUNG,
        base_of,
    )
    from taxiout.planlags import PLAN_LAG_FEATURES

    keys = [r.key for r in LADDER]
    assert keys.index("plan-lags") == keys.index("identifiers-geometry-aircraft") + 1
    assert keys.index("throughput") == keys.index("plan-lags") + 1
    net = [
        k
        for k in keys
        if k
        in (
            "throughput",
            "arrivals",
            "demand",
            "configuration",
            "corridor",
            "weather",
            "saturation",
        )
    ]
    for k in net:
        assert set(PLAN_LAG_FEATURES) <= set(LADDER_BY_KEY[k].features), k
        kept = LADDER_BY_KEY[f"{k}-nolag"]
        assert not set(PLAN_LAG_FEATURES) & set(kept.features), k
        assert set(kept.features) == set(LADDER_BY_KEY[k].features) - set(
            PLAN_LAG_FEATURES
        )
    # the kept chain is read against itself, and its first rung against the aircraft
    assert (
        base_of(LADDER_BY_KEY["throughput-nolag"]).key == "identifiers-geometry-aircraft"
    )
    assert base_of(LADDER_BY_KEY["arrivals-nolag"]).key == "throughput-nolag"
    assert base_of(LADDER_BY_KEY["plan-lags"]).key == "identifiers-geometry-aircraft"
    assert base_of(LADDER_BY_KEY["throughput"]).key == "plan-lags"
    assert base_of(LADDER[0]) is None
    # the submission model carries the plan lags
    assert set(PLAN_LAG_FEATURES) <= set(TOP_RUNG.features)
    assert len(set(keys)) == len(keys)


def test_the_submission_rung_is_not_the_last_rung_of_the_ladder():
    """Rungs sit above the submission, for reasons the ladder records.

    Saturation was built, measured, and left out because it bought nothing.
    The stand-and-runway arm is another case: its score is indistinguishable
    from the shipped arm's, and it is out on a mechanism -- the designator
    carries an intersection signature the routed distance cannot compute, so
    the runway categorical is not the encoding's to replace. Any Gate 2c rung
    the chain did not reach is a third. All of them are recorded beside
    ``SUBMISSION_KEY``; this pins the shape.
    """
    from taxiout.encoding import ENCODING_FEATURES
    from taxiout.models import ABOVE_SUBMISSION, LADDER_PROPER, TOP_RUNG
    from taxiout.saturation import SATURATION_FEATURES

    assert TOP_RUNG.key == "plan-diffs"
    # The Gate 2c chain ran to the top, so the only rung of the ladder proper
    # left above the submission is the measured negative the ladder keeps.
    assert [r.key for r in ABOVE_SUBMISSION] == ["saturation"]
    # the shipped rung carries the encoding, and keeps the runway categorical
    assert set(ENCODING_FEATURES) <= set(TOP_RUNG.features)
    assert "designator" in TOP_RUNG.features
    assert "stand" not in TOP_RUNG.features
    # the rung is still on the ladder, and its block is still built
    assert LADDER_PROPER[-1].key == "saturation"
    assert set(SATURATION_FEATURES) <= set(LADDER_PROPER[-1].features)
    # but nothing the submission is fitted from reads it
    assert not set(SATURATION_FEATURES) & set(TOP_RUNG.features)


def test_the_gate_2c_rungs_chain_on_the_arm_and_take_no_no_lag_twin():
    """Four blocks, one on top of the next, the first on the arm.

    Two things this pins. Each rung is read against the one below it in the
    chain and the first against the arm the encoding section chose, so a rung's
    number is what that block is worth net of the blocks under it. And none of
    them gains a twin in the chain kept without the plan lags: that chain is the
    plan-lag ablation of the ladder *below* the arms, and there is no arm
    without the plan lags for a twin of these to be read against.
    """
    from taxiout.encoding import ENCODING_FEATURES
    from taxiout.models import (
        ENCODING_STAND_BLOCK,
        GATE_2C_RUNGS,
        LADDER_BY_KEY,
        base_of,
    )

    expected_base = ["encoding-for-stand", *GATE_2C_RUNGS[:-1]]
    for key, base in zip(GATE_2C_RUNGS, expected_base):
        rung = LADDER_BY_KEY[key]
        assert base_of(rung).key == base, key
        # each rung is the one below it plus exactly one block
        assert set(LADDER_BY_KEY[base].features) < set(rung.features), key
        # every one of them stands on the arm, so every one carries the encoding
        assert set(ENCODING_STAND_BLOCK) <= set(rung.features), key
        assert set(ENCODING_FEATURES) <= set(rung.features), key
        assert f"{key}-nolag" not in LADDER_BY_KEY, key


def test_the_callsign_prefix_is_pooled_by_the_estimator_and_not_by_the_frame():
    """The threshold belongs to the fitting sample, so it belongs to the fit.

    ``build_features`` knows nothing about splits, so it carries the raw prefix;
    the booster counts levels on the rows it is fitted on and maps the rest to
    ``other``, which is the same arrangement the geometric fit uses for a wake
    class with too few rows at an airport.
    """
    from taxiout.callsign import CALLSIGN_FEATURES, MIN_LEVEL_ROWS, OTHER_LEVEL
    from taxiout.models import POOLED_CATEGORICALS

    assert POOLED_CATEGORICALS == CALLSIGN_FEATURES
    assert set(CALLSIGN_FEATURES) <= set(CATEGORICAL_FEATURES)
    assert MIN_LEVEL_ROWS == 30
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    frame = _movements(["B5", "B5"], FLIGHT_mvt=["DLH123", "1234"])
    f = build_features(frame, m)
    # the frame carries the raw prefix, pooled by nothing
    assert f["cs_prefix"].tolist() == ["DLH", "NA"]
    assert OTHER_LEVEL not in set(f["cs_prefix"].cat.categories)
