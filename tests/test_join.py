# SPDX-License-Identifier: GPL-3.0-only
"""Stand-identifier join: staged normalisation, compound refs, collapsing."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from taxiout.join import (  # noqa: E402
    UNMATCHED,
    collapse,
    expand_compound,
    join_rates,
    join_stands,
    normalise,
    split_compound,
)
from taxiout.routing import AGGREGATED_FIELDS  # noqa: E402


@pytest.mark.parametrize(
    ("value", "stage", "expected"),
    [
        ("w 998", 1, "W998"),
        (" b05 ", 1, "B05"),
        ("B05", 2, "B5"),
        ("B005", 2, "B5"),
        ("100", 2, "100"),  # zeros inside a run are not leading
        ("101L", 2, "101L"),
        ("101L", 3, "101"),
        ("A1L", 3, "A1"),
        ("A1", 3, "A1"),
        ("QZ6", 3, "QZ6"),  # letter before the digit is a prefix, kept
        ("AB", 3, "AB"),
    ],
)
def test_normalise_stages(value, stage, expected):
    assert normalise(value, stage) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("9 9Z", ["9", "9Z"]),
        ("Q98;Q99", ["Q98", "Q99"]),
        ("W 998", ["W 998"]),  # a pier letter alone is not an identifier
        ("Q97 Q99", ["Q97", "Q99"]),
        ("Link W", ["Link W"]),
        ("B05", ["B05"]),
    ],
)
def test_split_compound(value, expected):
    assert split_compound(value) == expected


def _matrix(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Minimal matrix: (stand_id, designator, taxi_m) at one invented airport."""
    recs = []
    for stand_id, designator, taxi_m in rows:
        rec = {
            "icao": "ZZZZ",
            "stand_id": stand_id,
            "designator": designator,
            "runway_ref": "09/27",
            "n_features": 1,
            "spread_m": 0.0,
            "snap_max_m": 0.0,
        }
        rec.update({f: 0.0 for f in AGGREGATED_FIELDS})
        rec["taxi_m"] = taxi_m
        recs.append(rec)
    return pd.DataFrame(recs)


def test_expand_compound_duplicates_the_cell():
    m = _matrix([("9 9Z", "27", 1000.0)])
    e = expand_compound(m)
    assert sorted(e["stand_id"]) == ["9", "9Z"]
    assert (e["taxi_m"] == 1000.0).all()


def test_collapse_records_ambiguity_and_keeps_a_real_row():
    m = _matrix([("101", "27", 1000.0), ("101L", "27", 1100.0), ("101R", "27", 1300.0)])
    c = collapse(m, stage=3)
    assert len(c) == 1
    row = c.iloc[0]
    assert row["key"] == "101"
    assert row["n_ids"] == 3
    assert row["id_spread_m"] == 300.0
    assert row["spread_m"] == 300.0
    # closest to the mean (1133): the 1100 m cell, not an average of the three
    assert row["taxi_m"] == 1100.0
    assert row["stand_id"] == "101L"


def test_collapse_at_stage_one_keeps_identifiers_apart():
    m = _matrix([("101", "27", 1000.0), ("101L", "27", 1100.0)])
    c = collapse(m, stage=1)
    assert len(c) == 2
    assert (c["n_ids"] == 1).all()


def test_join_matches_at_the_first_stage_that_finds_the_stand():
    m = _matrix(
        [
            ("B5", "27", 1000.0),
            ("101", "27", 2000.0),
            ("101L", "27", 2100.0),
            ("v 9", "27", 3000.0),
        ]
    )
    movements = pd.DataFrame(
        {
            "ADEP_mvt": ["ZZZZ"] * 6,
            "STAND_mvt": ["B05", "101L", "101", "101R", "V9", "X1"],
            "RUNWAY_mvt": ["27"] * 6,
        }
    )
    j = join_stands(movements, m)
    assert list(j["join_stage"]) == [2, 1, 1, 3, 1, UNMATCHED]
    assert j.loc[0, "taxi_m"] == 1000.0
    assert j.loc[1, "taxi_m"] == 2100.0  # the lettered half, kept precise
    assert j.loc[2, "taxi_m"] == 2000.0
    # 101R falls through to stage 3, where 101 and 101L collapse: the
    # representative is the cell nearest the mean, and the ambiguity is kept
    assert j.loc[3, "taxi_m"] in (2000.0, 2100.0)
    assert j.loc[3, "n_ids"] == 2
    assert j.loc[3, "id_spread_m"] == 100.0
    assert j.loc[4, "taxi_m"] == 3000.0
    assert pd.isna(j.loc[5, "taxi_m"])


def test_join_needs_the_runway_too():
    m = _matrix([("B5", "27", 1000.0)])
    movements = pd.DataFrame(
        {
            "ADEP_mvt": ["ZZZZ", "ZZZZ"],
            "STAND_mvt": ["B5", "B5"],
            "RUNWAY_mvt": ["09", None],
        }
    )
    j = join_stands(movements, m)
    assert list(j["join_stage"]) == [UNMATCHED, UNMATCHED]


def test_join_rates_reports_shapes_not_values():
    m = _matrix([("B5", "27", 1000.0)])
    movements = pd.DataFrame(
        {
            "ADEP_mvt": ["ZZZZ"] * 4,
            "STAND_mvt": ["B5", "B05", "Q77", "Q78"],
            "RUNWAY_mvt": ["27"] * 4,
        }
    )
    r = join_rates(join_stands(movements, m))
    assert len(r) == 1
    row = r.iloc[0]
    assert row["rows"] == 4
    assert row["stage1"] == 0.25
    assert row["stage2"] == 0.25
    assert row["matched"] == 0.5
    assert "`@##` 50.0%" in row["unmatched_shapes"]
    assert "Q77" not in row["unmatched_shapes"]
