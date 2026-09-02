# SPDX-License-Identifier: GPL-3.0-only
"""The worked airport: the pipeline reproduces the hand derivation.

``taxiout.synthetic`` draws a field on which every distance, turn, junction
count and repair is known in advance. These tests run the real pipeline on
it and compare, cell by cell. A failure here is either the pipeline drifting
or the derivation being wrong, and the field is small enough that a reader
can decide which by hand -- which is the point.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from taxiout.routing import AGGREGATED_FIELDS
from taxiout.synthetic import (
    DESIGNATORS,
    DISTANCE_TOL_M,
    FAR_STAND,
    TURN_TOL_DEG,
    WORKED,
    build,
    elements,
    expected_cell_meta,
    expected_cells,
    expected_stats,
    penalty_crossovers,
    probe_merge_shift,
)

INTEGER_FIELDS = ("sharp_turns", "junctions", "runway_crossings")


@pytest.fixture(scope="module")
def built():
    return build()


@pytest.fixture(scope="module")
def cells(built):
    return {(c.stand_id, c.designator): c for c in built.cells}


def test_every_repair_fires_once_and_the_build_reports_it(built):
    stats = {**built.tg.stats, **built.route_stats, "cells": len(built.cells)}
    for key, want in expected_stats().items():
        got = stats[key]
        if isinstance(want, dict):
            assert set(got) == set(want), key
            for d in want:
                for f, v in want[d].items():
                    assert got[d][f] == pytest.approx(v, abs=1e-3), (key, d, f)
        elif isinstance(want, float):
            assert got == pytest.approx(want, abs=1e-6), key
        else:
            assert got == want, key


def test_the_far_stand_is_reported_not_attached(built):
    far = [s for s in built.tg.stands if s.stand_id == FAR_STAND[0]]
    assert len(far) == 1
    assert far[0].node is None and math.isinf(far[0].snap_m)
    assert not any(r.stand_id == FAR_STAND[0] for r in built.routes)


def test_stand_geometry_kinds_and_snaps(built):
    by_id = {}
    for s in built.tg.stands:
        by_id.setdefault(s.stand_id, []).append(s)
    assert by_id["9"][0].geometry == "point" and by_id["9"][0].snap_m == pytest.approx(
        120.0, abs=1e-3
    )
    assert by_id["10"][0].geometry == "polygon"
    assert by_id["10"][0].snap_m == pytest.approx(math.hypot(50, 50), abs=1e-3)
    assert by_id["3"][0].id_tag == "name"  # identified by name, not ref
    assert [s.lead_in_m for s in by_id["2"]] == pytest.approx(
        [70.0, 75.0, 95.0], abs=1e-6
    )
    # Drawn stop-to-entry: still attaches at the taxilane end with the full lead-in.
    assert by_id["4"][0].snap_m == pytest.approx(0.0, abs=1e-6)
    assert by_id["4"][0].lead_in_m == pytest.approx(70.0, abs=1e-6)


def test_the_matrix_has_exactly_the_expected_cells(cells):
    assert set(cells) == set(expected_cells())
    assert len(cells) == 8 * len(DESIGNATORS)


@pytest.mark.parametrize("key", sorted(expected_cells(), key=lambda k: (int(k[0]), k[1])))
def test_cell_matches_the_hand_derivation(cells, key):
    want = expected_cells()[key]
    got = cells[key].values
    for f in AGGREGATED_FIELDS:
        if f in INTEGER_FIELDS:
            assert got[f] == want[f], (key, f)
        elif f == "turn_deg":
            assert got[f] == pytest.approx(want[f], abs=TURN_TOL_DEG), (key, f)
        elif f == "detour_ratio":
            assert got[f] == pytest.approx(want[f], abs=1e-4), (key, f)
        else:
            assert got[f] == pytest.approx(want[f], abs=DISTANCE_TOL_M), (key, f)


def test_cell_bookkeeping_for_duplicates_and_snaps(cells):
    for stand_id, meta in expected_cell_meta().items():
        cell = cells[(stand_id, "09")]
        assert cell.n_features == meta["n_features"], stand_id
        assert cell.spread_m == pytest.approx(meta["spread_m"], abs=DISTANCE_TOL_M), (
            stand_id
        )
        assert cell.snap_max_m == pytest.approx(meta["snap_max_m"], abs=DISTANCE_TOL_M), (
            stand_id
        )


def test_taxi_m_is_the_sum_of_its_parts(built):
    for r in built.routes:
        assert r.taxi_m == pytest.approx(
            r.lead_in_m + r.snap_m + r.route_m + r.entry_offset_m
        )
        assert r.entry_offset_m == 0.0
        assert r.detour_ratio >= 1.0


def test_penalty_crossovers_are_where_the_arithmetic_says():
    x = penalty_crossovers()

    def backtrack(p, stand, designator):
        b = build(runway_penalty=p)
        return {(c.stand_id, c.designator): c.values["backtrack_m"] for c in b.cells}[
            (stand, designator)
        ]

    eps = 0.005
    assert backtrack(x["south 27"] - eps, "12", "27") == pytest.approx(
        900.0, abs=DISTANCE_TOL_M
    )
    assert backtrack(x["south 27"] + eps, "12", "27") == 0.0
    assert backtrack(x["south 09"] - eps, "12", "09") == pytest.approx(
        1500.0, abs=DISTANCE_TOL_M
    )
    assert backtrack(x["south 09"] + eps, "12", "09") == 0.0
    # The north stands' window between the two crossovers is 0.0018 wide.
    assert x["north 27 via A2"] < 1.0 < x["north 27"]
    assert backtrack(1.0, "1", "27") == pytest.approx(900.0, abs=DISTANCE_TOL_M)
    assert backtrack(x["north 27"] + eps, "1", "27") == 0.0
    assert backtrack(x["north 27 via A2"] - eps, "1", "27") == pytest.approx(
        1800.0, abs=DISTANCE_TOL_M
    )
    # And at the default penalty nobody runs along a runway they could taxi round.
    assert all(
        v == 0.0
        for k, v in {
            (c.stand_id, c.designator): c.values["backtrack_m"] for c in build().cells
        }.items()
        if k[1] in ("09", "27")
    )


def test_threshold_36_backtrack_is_a_constant_offset(built):
    on_36 = [r.backtrack_m for r in built.routes if r.designator == "36"]
    assert len(on_36) == 10
    assert min(on_36) == pytest.approx(900.0, abs=DISTANCE_TOL_M)
    assert max(on_36) == pytest.approx(900.0, abs=DISTANCE_TOL_M)


def test_worked_examples_name_real_cells(cells):
    for stand_id, designator, _text in WORKED:
        assert (stand_id, designator) in cells


def test_elements_carry_no_stand_identifier_that_could_collide():
    """Identifiers are small integers, so the licence gate's prose check cannot fire."""
    for el in elements():
        tags = el.get("tags") or {}
        if tags.get("aeroway") != "parking_position":
            continue
        value = tags.get("ref") or tags.get("name")
        assert value.isdigit(), value


def test_probe_measures_the_merged_node_shift():
    """The probe fires both repairs and reports the pavement the graph keeps.

    Before the fix in ``graph.py`` the merged node moves to the last way's
    position and a later split loses the 0.2 m shift; after it nothing is
    missing. Either is a valid measurement here -- the strict assertion
    belongs with the fix, in ``tests/test_graph.py`` -- but a loss beyond the
    merge tolerance would be a new defect.
    """
    p = probe_merge_shift()
    assert p["nodes_merged"] == 1 and p["edges_split"] == 1
    assert -0.01 <= p["missing_m"] <= 0.21


# --------------------------------------------------------------------------
# worked traffic: the fixture in the challenge's schema, from a known process
#
# These need pandas and pyarrow, which the geometry tests above do not, so
# they skip individually rather than taking the module with them.

HAS_FRAMES = all(
    importlib.util.find_spec(m) is not None for m in ("pandas", "numpy", "pyarrow")
)
needs_frames = pytest.mark.skipif(not HAS_FRAMES, reason="needs pandas and pyarrow")

if HAS_FRAMES:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from taxiout import synthetic as syn
    from taxiout.challenge import (
        HIDDEN_FOR_TEST_DEPARTURES,
        load_movements,
        load_ranking,
        load_submitting,
        to_pandas,
    )
    from taxiout.submission import read as read_submission
    from taxiout.submission import validate, write
    from taxiout.tails import TailModel, combine


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("challenge")
    t = syn.traffic(seed=1, per_day=12)
    syn.write_traffic(t, root)
    return root, t


@needs_frames
def test_traffic_tests_are_collected_with_frames_present():
    assert HAS_FRAMES


@needs_frames
def test_traffic_carries_the_challenge_schema(fixture_root):
    root, t = fixture_root
    names = [c for c, _ in syn.CHALLENGE_SCHEMA]
    assert list(t.training.columns) == names
    assert list(t.ranking.columns) == names
    for path in sorted((root / "movements").glob("training_*.parquet")) + [
        root / "ranking.parquet"
    ]:
        assert pq.read_schema(path).equals(syn.arrow_schema()), path.name
    tpl = pq.read_schema(root / "submitting.parquet")
    assert tpl.names == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    assert str(tpl.types[0]) == "double" and str(tpl.types[1]) == "int32"
    assert len(list((root / "movements").glob("training_*.parquet"))) == 12


@needs_frames
def test_ranking_blanks_what_the_challenge_blanks(fixture_root):
    _root, t = fixture_root
    dep = t.ranking["PHASE_mvt"] == "DEP"
    for c in HIDDEN_FOR_TEST_DEPARTURES:
        assert t.ranking.loc[dep, c].isna().all(), c
        assert t.ranking.loc[~dep, c].notna().all(), c
    assert t.ranking.loc[dep, "MVT_TIME_UTC_mvt"].notna().all()
    assert set(t.template["MVT_ID_mvt"]) == set(t.ranking.loc[dep, "MVT_ID_mvt"])
    assert list(t.truth["MVT_ID_mvt"]) == list(t.template["MVT_ID_mvt"])
    months = sorted(t.ranking["MVT_TIME_UTC_mvt"].dt.month.unique())
    assert months == [1, 7]
    assert (t.ranking["MVT_TIME_UTC_mvt"].dt.year == 2026).all()
    assert (t.training["MVT_TIME_UTC_mvt"].dt.year == 2025).all()


@needs_frames
def test_the_loaders_read_the_fixture_unchanged(fixture_root):
    root, t = fixture_root
    dep = to_pandas(load_movements(root, phase="DEP"))
    arr = to_pandas(load_movements(root, phase="ARR"))
    rk = to_pandas(load_ranking(root))
    tpl = to_pandas(load_submitting(root))
    assert len(dep) + len(arr) == len(t.training)
    assert len(rk) == len(t.ranking) and len(tpl) == len(t.template)
    assert str(dep["MVT_TIME_UTC_mvt"].dtype) == "datetime64[us, UTC]"
    # The flight side is null as a block where the join failed, as in the data.
    no_record = dep["LOBT_flt"].isna()
    assert 0 < no_record.mean() < 0.1
    for c in syn.FLT_COLUMNS:
        assert dep.loc[no_record, c].isna().all(), c
        assert dep.loc[~no_record, c].notna().all(), c


@needs_frames
def test_the_process_is_what_it_says(fixture_root):
    _root, t = fixture_root
    truth = t.truth
    ok = ~truth["defaulted"] & ~truth["late"] & (truth["stand_kind"] == "ok")
    assert (truth.loc[ok, "excess_s"] >= 0).all()
    assert (truth.loc[ok, "recorded_s"] == truth.loc[ok, "taxi_true_s"]).all()
    assert (truth.loc[ok, "taxi_true_s"] >= syn.MIN_TAXI_S).all()
    # Unimpeded time is the field's own distance over the stated speed.
    rk = t.ranking.set_index("MVT_ID_mvt")
    rows = truth[ok].head(50)
    dist = {(r.stand_id, r.designator): r.taxi_m for r in t.matrix.itertuples()}
    for r in rows.itertuples():
        m = rk.loc[r.MVT_ID_mvt]
        want = syn.PUSHBACK_S + dist[(m["STAND_mvt"], m["RUNWAY_mvt"])] / syn.SPEED_MPS
        assert r.unimpeded_s == pytest.approx(want, abs=1e-6)
    # A defaulted row records the lag, not the taxi.
    d = truth[truth["defaulted"]]
    assert len(d) > 0
    lag = (
        rk.loc[d["MVT_ID_mvt"], "MVT_TIME_UTC_mvt"]
        - rk.loc[d["MVT_ID_mvt"], "SCHED_TIME_UTC_mvt"]
    ).dt.total_seconds()
    assert (np.abs(d["recorded_s"].to_numpy() - lag.to_numpy()) <= 6).all()
    assert (d["recorded_s"].to_numpy() > 2 * 3600).all()


@needs_frames
def test_the_record_quality_term_recovers_the_planted_feed(fixture_root):
    root, t = fixture_root
    train = to_pandas(load_movements(root, phase="DEP"))
    rk = to_pandas(load_ranking(root, phase="DEP"))
    tm = TailModel().fit(train)
    covered = tm.covered
    assert len(covered) >= 1
    assert all(b.startswith("noflt:") for _, b in covered.index)
    assert all(i == syn.ICAO for i, _ in covered.index)
    pred = tm.predict(rk).to_numpy()
    truth = t.truth.set_index("MVT_ID_mvt").loc[rk["MVT_ID_mvt"]]
    y = truth["recorded_s"].to_numpy()
    flagged = ~np.isnan(pred)
    # Every planted defaulted row with a lag above four hours sits in a covered cell.
    lag = (
        (rk["MVT_TIME_UTC_mvt"] - rk["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()
    )
    planted = truth["defaulted"].to_numpy() & (lag > 4 * 3600)
    assert flagged[planted].all()
    # On the rows it covers the term beats the airport mean, by a lot.
    mean = float(train["TAXITIME_SEC_mvt"].mean())
    rmse_term = np.sqrt(np.mean((y[flagged] - pred[flagged]) ** 2))
    rmse_mean = np.sqrt(np.mean((y[flagged] - mean) ** 2))
    assert rmse_term < 0.5 * rmse_mean


@needs_frames
def test_end_to_end_submission_on_the_fixture(fixture_root, tmp_path):
    root, t = fixture_root
    train = to_pandas(load_movements(root, phase="DEP"))
    rk = to_pandas(load_ranking(root, phase="DEP"))
    tpl = to_pandas(load_submitting(root))
    # A physical model that knows the process: unimpeded from the field's
    # own matrix, plus the mean excess seen in training; airport mean where
    # the stand does not join.
    dist = {(r.stand_id, r.designator): r.taxi_m for r in t.matrix.itertuples()}
    tm = TailModel().fit(train)
    unflagged = ~tm.flagged(train)
    excess = float(
        (
            train.loc[unflagged, "TAXITIME_SEC_mvt"]
            - syn.PUSHBACK_S
            - np.array(
                [
                    dist.get((s, r), np.nan)
                    for s, r in zip(
                        train.loc[unflagged, "STAND_mvt"],
                        train.loc[unflagged, "RUNWAY_mvt"],
                    )
                ]
            )
            / syn.SPEED_MPS
        ).mean()
    )
    fallback = float(train.loc[unflagged, "TAXITIME_SEC_mvt"].mean())
    physical = pd.Series(
        [
            syn.PUSHBACK_S + dist[(s, r)] / syn.SPEED_MPS + excess
            if (s, r) in dist
            else fallback
            for s, r in zip(rk["STAND_mvt"], rk["RUNWAY_mvt"])
        ],
        index=rk.index,
    )
    pred = combine(physical, tm.predict(rk))
    frame = pd.DataFrame(
        {"MVT_ID_mvt": rk["MVT_ID_mvt"].to_numpy(), "TAXITIME_SEC_mvt": pred.to_numpy()}
    )
    path = write(frame, tpl, tmp_path / "zzzz_v1.parquet")
    back = read_submission(path)
    validate(back, tpl)
    assert list(back["MVT_ID_mvt"]) == list(tpl["MVT_ID_mvt"])
    truth = t.truth.set_index("MVT_ID_mvt").loc[back["MVT_ID_mvt"]]
    y = truth["recorded_s"].to_numpy()
    rmse = np.sqrt(np.mean((y - back["TAXITIME_SEC_mvt"].to_numpy()) ** 2))
    rmse_mean = np.sqrt(np.mean((y - fallback) ** 2))
    assert rmse < rmse_mean
    # And without the term the same model is much worse, because the
    # scoring rule counts the defaulted rows.
    plain = np.rint(physical.to_numpy())
    rmse_plain = np.sqrt(np.mean((y - plain) ** 2))
    assert rmse < 0.5 * rmse_plain
