# SPDX-License-Identifier: GPL-3.0-only
"""C7's block, ``taxiout.thunder``, on constructed METAR reports."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.features import source_phase  # noqa: E402
from taxiout.metar import FIELDS as METAR_FIELDS  # noqa: E402
from taxiout.thunder import (  # noqa: E402
    SOURCES,
    THUNDER_FEATURES,
    thunder_by_hour,
    thunder_features,
)

H0 = pd.Timestamp("2025-07-10T00:00:00Z")


def _metar(reports):
    return pd.DataFrame(
        {
            "station": "LIRF",
            "valid": [H0 + pd.Timedelta(hours=h, minutes=m) for h, m, _ in reports],
            "wxcodes": [w for *_, w in reports],
        }
    )


def _row(table, hour):
    return table.loc[table["hour"] == H0 + pd.Timedelta(hours=hour)].iloc[0]


def test_the_three_flags_of_an_hour():
    table = thunder_by_hour(
        _metar(
            [
                (0, 20, "TSRA"),
                (1, 20, "VCTS"),
                (2, 20, "VCTS"),
                (2, 50, "-TSRA"),  # a storm here in the same hour outranks one nearby
                (3, 20, "+RA"),
                (4, 20, "VCSH"),
            ]
        )
    )
    assert _row(table, 0)[["wx_ts", "wx_vcts", "wx_heavy"]].tolist() == [1, 0, 0]
    assert _row(table, 1)[["wx_ts", "wx_vcts", "wx_heavy"]].tolist() == [0, 1, 0]
    assert _row(table, 2)[["wx_ts", "wx_vcts", "wx_heavy"]].tolist() == [1, 0, 0]
    assert _row(table, 3)[["wx_ts", "wx_vcts", "wx_heavy"]].tolist() == [0, 0, 1]
    assert _row(table, 4)[["wx_ts", "wx_vcts", "wx_heavy"]].tolist() == [0, 0, 0]


def test_the_hours_before():
    table = thunder_by_hour(
        _metar([(h, 20, "TS" if h in (0, 2, 3) else "NSW") for h in range(6)])
    )
    assert _row(table, 4)["wx_ts_prev3"] == 2  # hours 1, 2, 3
    assert _row(table, 5)["wx_ts_prev3"] == 2  # hours 2, 3, 4
    assert _row(table, 4)["wx_ts_prev_hour"] == 1
    assert _row(table, 5)["wx_ts_prev_hour"] == 0


def test_carried_three_hours_then_missing():
    table = thunder_by_hour(_metar([(0, 20, "TS"), (8, 20, "NSW")]))
    assert [_row(table, h)["wx_ts"] for h in (1, 2, 3)] == [1, 1, 1]
    assert np.isnan(_row(table, 4)["wx_ts"])
    assert _row(table, 4)["wx_ts_prev3"] == 3
    dep = pd.DataFrame(
        {
            "ADEP_mvt": ["LIRF", "LIRF", "EDDF"],
            "MVT_TIME_UTC_mvt": [
                H0 + pd.Timedelta(hours=h, minutes=5) for h in (2, 5, 2)
            ],
        }
    )
    got = thunder_features(dep, table)
    assert list(got.columns) == list(THUNDER_FEATURES)
    assert got.loc[0, "wx_ts"] == 1
    assert np.isnan(got.loc[1, "wx_ts"])
    assert got.loc[2].isna().all()


def test_the_sources_are_the_archive_fields_and_the_subjects_own():
    assert set(SOURCES) == set(THUNDER_FEATURES)
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (None, "METAR"), (name, src)
            if phase == "METAR":
                assert column in METAR_FIELDS, (name, src)
            else:
                assert column in ("MVT_TIME_UTC_mvt", "ADEP_mvt"), (name, src)
