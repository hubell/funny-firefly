# SPDX-License-Identifier: GPL-3.0-only
"""The shared-corridor block, on a field small enough to weigh by hand.

Three stands on one airport. Stands A and B share an apron spur that only
they use; stand C reaches the runway another way; all three funnel through
the same holding-point edge. The weights and overlaps below are arithmetic
on that picture, and every assertion was written before the code ran.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from taxiout.challenge import (  # noqa: E402
    AVAILABLE_FOR_ARRIVALS,
    FORBIDDEN_FEATURE_INPUTS,
)
from taxiout.corridor import (  # noqa: E402
    CORRIDOR_FEATURES,
    SOURCES,
    corridor_features,
    overlap_matrix,
)
from taxiout.features import source_phase  # noqa: E402
from taxiout.routing import AGGREGATED_FIELDS, decode_edges, encode_edges  # noqa: E402

T0 = pd.Timestamp("2025-03-04T10:00:00Z")

# Edges as (u, v) node pairs. The holding point H is edge (9, 10) and every
# route uses it; the apron spur S is (1, 2) and A and B use it; each stand
# has a private lead edge.
H, S = (9, 10), (1, 2)
ROUTES = {
    "A": (S, (2, 9), H, (0, 1)),
    "B": (S, (2, 9), H, (3, 1)),
    "C": ((5, 9), H, (4, 5)),
}


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "icao": ["ZZZZ"] * 3,
            "stand_id": list(ROUTES),
            "designator": ["27"] * 3,
            "n_edges": [len(v) for v in ROUTES.values()],
            "edges": [tuple(sorted(v)) for v in ROUTES.values()],
        }
    )


def _matrix() -> pd.DataFrame:
    recs = []
    for stand in ROUTES:
        rec = {
            "icao": "ZZZZ",
            "stand_id": stand,
            "designator": "27",
            "runway_ref": "09/27",
            "n_features": 1,
            "spread_m": 0.0,
            "snap_max_m": 0.0,
        }
        rec.update({f: 1000.0 for f in AGGREGATED_FIELDS})
        recs.append(rec)
    return pd.DataFrame(recs)


def when(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def movement(mvt_id, phase, stand, t, *, block=None):
    return {
        "MVT_ID_mvt": float(mvt_id),
        "PHASE_mvt": phase,
        "ADEP_mvt": "ZZZZ" if phase == "DEP" else "AAAA",
        "ADES_mvt": "ZZZZ" if phase == "ARR" else "AAAA",
        "STAND_mvt": stand,
        "RUNWAY_mvt": "27",
        "MVT_TIME_UTC_mvt": when(t),
        "BLOCK_TIME_UTC_mvt": when(block) if block is not None else pd.NaT,
        "TAXITIME_SEC_mvt": 300.0 if phase == "ARR" else np.nan,
        "SCHED_TIME_UTC_mvt": when(t),
    }


# --------------------------------------------------------------------------
# the codec


def test_edges_round_trip_through_the_file_form():
    edges = ((1, 2), (2, 9), (9, 10))
    assert decode_edges(encode_edges(edges)) == edges
    assert encode_edges(()) == ""
    assert (
        decode_edges("") == ()
        and decode_edges(None) == ()
        and decode_edges(float("nan")) == ()
    )


# --------------------------------------------------------------------------
# the overlap matrix


def test_edge_weights_are_inverse_use_and_overlap_is_the_subjects_share():
    ov = overlap_matrix(_edges(), ["A|27", "B|27", "C|27"])
    a, b, c = (ov.cells[k] for k in ("A|27", "B|27", "C|27"))
    # weights: H used by 3 -> 1/3; S and (2,9) used by 2 -> 1/2; leads and
    # (5,9) used by 1 -> 1
    assert ov.weight[a] == pytest.approx(1 / 3 + 1 / 2 + 1 / 2 + 1)
    assert ov.weight[c] == pytest.approx(1 / 3 + 1 + 1)
    # A and B share S, (2,9) and H: (1/2 + 1/2 + 1/3) of A's own weight
    assert ov.matrix[a, b] == pytest.approx((1 / 2 + 1 / 2 + 1 / 3) / ov.weight[a])
    # A and C share only the holding point, which weighs a third
    assert ov.matrix[a, c] == pytest.approx((1 / 3) / ov.weight[a])
    # a route overlaps itself completely
    assert ov.matrix[a, a] == pytest.approx(1.0)
    # and the matrix is not symmetric, because each row is its own share
    assert ov.matrix[c, a] == pytest.approx((1 / 3) / ov.weight[c])


def test_weights_come_from_every_routed_cell_not_only_the_used_ones():
    """The field's property: C's existence weighs H down even if C never flies."""
    ov = overlap_matrix(_edges(), ["A|27", "B|27"])
    a, b = ov.cells["A|27"], ov.cells["B|27"]
    assert ov.matrix[a, b] == pytest.approx((1 / 2 + 1 / 2 + 1 / 3) / ov.weight[a])
    assert "C|27" not in ov.cells


def test_an_unknown_cell_is_ignored_and_an_empty_airport_is_empty():
    ov = overlap_matrix(_edges(), ["A|27", "Q|27"])
    assert set(ov.cells) == {"A|27"}
    empty = overlap_matrix(_edges().iloc[:0], ["A|27"])
    assert empty.matrix.shape == (0, 0)


# --------------------------------------------------------------------------
# the features


def _scene() -> pd.DataFrame:
    rows = [
        movement(100, "DEP", "A", 60),  # the subject, on A
        movement(101, "DEP", "B", 58),  # on the shared spur, three minutes before
        movement(102, "DEP", "C", 57),  # the other side of the field
        movement(103, "DEP", "A", 40),  # same stand, twenty minutes before
        movement(104, "DEP", "B", 61),  # after the subject
        movement(200, "ARR", "B", 50, block=59),  # blocks on the spur a minute before
        movement(201, "ARR", "C", 50, block=58),
        movement(202, "ARR", "A", 30, block=50),  # ten minutes before: in 15, not 5
    ]
    return pd.DataFrame(rows)


def _subject(scene: pd.DataFrame) -> pd.DataFrame:
    return scene[scene["MVT_ID_mvt"] == 100.0].reset_index(drop=True)


def test_corridor_load_weighs_each_movement_by_the_route_it_shares():
    scene = _scene()
    f = corridor_features(_subject(scene), scene, _matrix(), _edges())
    ov = overlap_matrix(_edges(), ["A|27", "B|27", "C|27"])
    a, b, c = (ov.cells[k] for k in ("A|27", "B|27", "C|27"))
    # five minutes: 101 on B and 102 on C; not 103 (too early), 104 (after), self
    assert f["corr_dep_5m"].iloc[0] == pytest.approx(ov.matrix[a, b] + ov.matrix[a, c])
    # fifteen minutes: the same -- 103 is at twenty
    assert f["corr_dep_15m"].iloc[0] == pytest.approx(ov.matrix[a, b] + ov.matrix[a, c])
    # arrivals by on-block: 200 (59) on B and 201 (58) on C within five;
    # 202 (50) on A within fifteen, and it overlaps completely
    assert f["corr_arr_5m"].iloc[0] == pytest.approx(ov.matrix[a, b] + ov.matrix[a, c])
    assert f["corr_arr_15m"].iloc[0] == pytest.approx(
        ov.matrix[a, b] + ov.matrix[a, c] + 1.0
    )
    assert f["corr_route_weight"].iloc[0] == pytest.approx(ov.weight[a])


def test_the_subject_is_removed_by_identity_and_a_twin_on_its_stand_counts():
    scene = _scene()
    scene.loc[scene["MVT_ID_mvt"] == 103.0, "MVT_TIME_UTC_mvt"] = when(59)
    f = corridor_features(_subject(scene), scene, _matrix(), _edges())
    ov = overlap_matrix(_edges(), ["A|27", "B|27", "C|27"])
    a, b, c = (ov.cells[k] for k in ("A|27", "B|27", "C|27"))
    # 103 is another aircraft on the subject's own route: full overlap, counted
    assert f["corr_dep_5m"].iloc[0] == pytest.approx(
        ov.matrix[a, b] + ov.matrix[a, c] + 1.0
    )


def test_an_unplaced_subject_is_nan_and_an_unplaced_event_contributes_nothing():
    scene = _scene()
    scene.loc[scene["MVT_ID_mvt"] == 101.0, "STAND_mvt"] = "NOPE"
    f = corridor_features(_subject(scene), scene, _matrix(), _edges())
    ov = overlap_matrix(_edges(), ["A|27", "B|27", "C|27"])
    a, c = ov.cells["A|27"], ov.cells["C|27"]
    assert f["corr_dep_5m"].iloc[0] == pytest.approx(ov.matrix[a, c])
    lost = _subject(scene).assign(STAND_mvt="NOPE")
    g = corridor_features(lost, scene, _matrix(), _edges())
    assert g[list(CORRIDOR_FEATURES)].isna().all().all()


def test_the_block_carries_exactly_its_columns_and_the_index():
    scene = _scene()
    subject = _subject(scene)
    subject.index = [42]
    f = corridor_features(subject, scene, _matrix(), _edges())
    assert list(f.columns) == list(CORRIDOR_FEATURES)
    assert f.index.tolist() == [42]


# --------------------------------------------------------------------------
# the rule, per phase


def test_sources_respect_the_phase_rule():
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert column != "AOBT_3_flt"
            if phase in (None, "DEP"):
                assert column not in FORBIDDEN_FEATURE_INPUTS, (name, src)
            else:
                assert column in AVAILABLE_FOR_ARRIVALS, (name, src)
