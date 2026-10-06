# SPDX-License-Identifier: GPL-3.0-only
"""C2a's classes and layer, ``taxiout.winter``, on constructed METAR reports."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.features import source_phase  # noqa: E402
from taxiout.metar import FIELDS as METAR_FIELDS  # noqa: E402
from taxiout.winter import (  # noqa: E402
    OTHER,
    SOURCES,
    WinterLayer,
    classes_by_hour,
    thresholds,
    winter_class,
)

H0 = pd.Timestamp("2025-01-10T00:00:00Z")


def _f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _metar(reports):
    """(hour, wxcodes, temperature C, dew point C) per report, one station."""
    return pd.DataFrame(
        {
            "station": "EHAM",
            "valid": [H0 + pd.Timedelta(hours=h, minutes=20) for h, *_ in reports],
            "wxcodes": [w for _, w, _, _ in reports],
            "tmpf": [_f(t) for *_, t, _ in reports],
            "dwpf": [_f(d) for *_, d in reports],
        }
    )


def _class_at(table, hour):
    got = table.loc[table["hour"] == H0 + pd.Timedelta(hours=hour), "winter_class"]
    return None if got.empty else got.iloc[0]


def test_priority_order():
    table = classes_by_hour(
        _metar(
            [
                (0, "FZRA SN", -1, -2),  # freezing precipitation outranks snow
                (1, "-SN FZFG", -1, -2),  # snow outranks freezing fog
                (2, "FZFG", -1, -2),
                (3, "RA", 2.5, 1),  # rain at or below +3 C
                (4, "RA", 3.5, 1),  # warmer: other
                (5, None, 0.5, -1),  # frost risk: spread 1.5, no precipitation
                (6, None, 0.5, -2),  # spread 2.5: other
                (7, "PL", 1, 0),
            ]
        )
    )
    assert [_class_at(table, h) for h in range(8)] == [
        "fz_precip",
        "snow",
        "fz_fog",
        "cold_wet",
        OTHER,
        "frost_risk",
        OTHER,
        "fz_precip",
    ]


def test_vicinity_groups_are_dropped():
    table = classes_by_hour(_metar([(0, "VCSN", -2, -3), (1, "VCFG", 5, 4)]))
    assert _class_at(table, 0) == "frost_risk"  # VCSN is not snow here, nor precipitation
    assert _class_at(table, 1) == OTHER


def test_persistent_snow_needs_four_of_the_twelve_hours_before():
    reports = [(h, "SN", -2, -3) for h in range(5)]
    table = classes_by_hour(_metar(reports))
    assert [_class_at(table, h) for h in range(5)] == ["snow"] * 4 + ["snow_persistent"]
    assert thresholds()["persistent_hours"] == 4
    assert thresholds()["persistent_window_h"] == 12


def test_the_class_is_carried_three_hours_and_then_absent():
    table = classes_by_hour(_metar([(0, "FZFG", -1, -2), (10, None, 10, 0)]))
    assert [_class_at(table, h) for h in (1, 2, 3)] == ["fz_fog"] * 3
    assert _class_at(table, 4) is None
    dep = pd.DataFrame(
        {
            "ADEP_mvt": ["EHAM", "EHAM", "LSZH"],
            "MVT_TIME_UTC_mvt": [
                H0 + pd.Timedelta(hours=h, minutes=40) for h in (2, 5, 2)
            ],
        }
    )
    assert winter_class(dep, table).tolist() == ["fz_fog", OTHER, OTHER]


def test_the_layer_shrinks_the_cell_toward_the_class():
    k = thresholds()["prior_rows"]
    airport = ["A"] * 4 + ["B"] * 2 + ["A"]
    cls = ["snow"] * 6 + [OTHER]
    residual = [10.0, 10.0, 10.0, 10.0, 40.0, 40.0, 999.0]
    layer = WinterLayer().fit(airport, cls, residual)
    m_c = (40 + 80) / 6  # the class mean over both airports, the other row left out
    za = 4 / (4 + k)
    got = layer.adjustment(["A", "C", "A", "B"], ["snow", "snow", OTHER, "fz_fog"])
    assert got[0] == pytest.approx(za * 10.0 + (1 - za) * m_c)
    assert got[1] == pytest.approx(m_c)  # an airport the class never saw takes m(c)
    assert got[2] == 0.0  # other is unchanged
    assert got[3] == 0.0  # a class with no fitted row is unchanged
    np.testing.assert_allclose(
        layer.apply([100.0, 100.0], ["A", "A"], ["snow", OTHER]), [100.0 + got[0], 100.0]
    )


def test_the_sources_are_the_archive_fields_and_the_subjects_own():
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (None, "METAR"), (name, src)
            if phase == "METAR":
                assert column in METAR_FIELDS, (name, src)
            else:
                assert column in ("MVT_TIME_UTC_mvt", "ADEP_mvt"), (name, src)
