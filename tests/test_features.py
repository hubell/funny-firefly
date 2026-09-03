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

from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS  # noqa: E402
from taxiout.features import (  # noqa: E402
    AIRCRAFT_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_SOURCES,
    GEOMETRY_FEATURES,
    QUALITY_FEATURES,
    REQUIRED_COLUMNS,
    TIME_FEATURES,
    airport_fallback,
    build_features,
    feature_block,
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
            "ADEP_mvt": ["ZZZZ"] * n,
            "STAND_mvt": stands,
            "RUNWAY_mvt": ["27"] * n,
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-03-04T08:30:00Z"] * n, utc=True),
            "LOBT_flt": pd.to_datetime(["2025-03-04T08:10:00Z"] * n, utc=True),
            "IOBT_flt": pd.to_datetime(["2025-03-04T08:00:00Z"] * n, utc=True),
            "WK_TBL_CAT_flt": ["M"] * n,
            "AIRCRAFT_TYPE_mvt": ["A320"] * n,
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


def test_no_feature_declares_a_forbidden_source():
    forbidden = set(FORBIDDEN_FEATURE_INPUTS)
    offending = {
        name: sorted(set(sources) & forbidden)
        for name, sources in FEATURE_SOURCES.items()
        if set(sources) & forbidden
    }
    assert offending == {}


def test_required_columns_are_the_union_of_the_sources():
    union = {c for sources in FEATURE_SOURCES.values() for c in sources}
    assert set(REQUIRED_COLUMNS) == union
    assert not union & set(FORBIDDEN_FEATURE_INPUTS)


def test_every_feature_declares_at_least_one_source():
    assert all(len(s) >= 1 for s in FEATURE_SOURCES.values())


# --------------------------------------------------------------------------
# what the builder actually reads


def test_builder_ignores_the_forbidden_columns():
    """Poison every forbidden field; the frame must not move.

    A registry entry is a claim about the code. This checks the code.
    """
    m = _matrix([("ZZZZ", "B5", "27", 1000.0)])
    base = _movements(["B5", "B5"])
    clean = build_features(base, m)

    poisoned = base.copy()
    for c in FORBIDDEN_FEATURE_INPUTS:
        poisoned[c] = (
            [1.0, 99999.0]
            if c.endswith("_SEC_mvt")
            else pd.to_datetime(
                ["1999-01-01T00:00:00Z", "2031-12-31T23:59:00Z"], utc=True
            )
        )
    after = build_features(poisoned, m)
    pd.testing.assert_frame_equal(clean, after)


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
