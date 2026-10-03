# SPDX-License-Identifier: GPL-3.0-only
"""Weather as a regime indicator: the flags, the hourly carry, the join.

The reports here are written in the archive's own format -- statute miles,
Fahrenheit, inches, ``M`` for missing -- so the unit handling is exercised
by the same rows that exercise the flags.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.features import source_phase  # noqa: E402
from taxiout.metar import (  # noqa: E402
    FIELDS,
    MAX_CARRY_H,
    SOURCES,
    WEATHER_FEATURES,
    flags,
    load_metar,
    regime_by_hour,
    weather_features,
)


def report(station, valid, *, vsby="6.21", tmpf="50.0", p01i="0.00", wxcodes="M"):
    return {
        "station": station,
        "valid": valid,
        "tmpf": tmpf,
        "dwpf": "40.0",
        "vsby": vsby,
        "p01i": p01i,
        "wxcodes": wxcodes,
        "skyc1": "FEW",
        "skyl1": "3000.00",
    }


def _frame(rows) -> pd.DataFrame:
    """As ``load_metar`` would deliver it: ``M`` already read as missing."""
    df = pd.DataFrame(
        [{k: (None if v == "M" else v) for k, v in r.items()} for r in rows]
    )
    for c in ("tmpf", "dwpf", "vsby", "p01i", "skyl1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df


# --------------------------------------------------------------------------
# the flags


def test_visibility_bands_sit_on_the_stated_edges():
    f = flags(
        _frame(
            [
                report("X", "2025-01-01 00:50", vsby="6.21"),  # 10.0 km
                report("X", "2025-01-01 01:50", vsby="4.97"),  # 8.0 km: not below
                report("X", "2025-01-01 02:50", vsby="3.00"),  # 4.8 km
                report("X", "2025-01-01 03:50", vsby="0.93"),  # 1.50 km: not below
                report("X", "2025-01-01 04:50", vsby="0.50"),  # 0.8 km
                report("X", "2025-01-01 05:50", vsby="M"),
            ]
        )
    )
    # 6.21 mi is 9,994 m; 4.97 mi is 7,998 m, just under the 8 km edge; 3.00 mi
    # is 4,828 m; 0.93 mi is 1,497 m, just under the 1.5 km edge; 0.50 mi is 805 m
    assert f["vis_band"].tolist()[:5] == [0.0, 1.0, 1.0, 2.0, 2.0]
    assert np.isnan(f["vis_band"].iloc[5])


def test_precipitation_reads_codes_or_accumulation():
    f = flags(
        _frame(
            [
                report("X", "2025-01-01 00:50", wxcodes="-RA"),
                report(
                    "X", "2025-01-01 01:50", wxcodes="BR"
                ),  # mist is not precipitation
                report("X", "2025-01-01 02:50", p01i="0.01"),
                report("X", "2025-01-01 03:50", wxcodes="TSRA"),
                report("X", "2025-01-01 04:50"),
            ]
        )
    )
    assert f["precip"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


def test_weather_in_the_vicinity_is_not_weather_here():
    """``VCSH`` is showers eight kilometres out; ``SHRA`` is rain on the field."""
    f = flags(
        _frame(
            [
                report("X", "2025-01-01 00:50", wxcodes="VCSH"),
                report("X", "2025-01-01 01:50", wxcodes="VCTS"),
                report("X", "2025-01-01 02:50", wxcodes="VCSH SHRA"),
                report("X", "2025-01-01 03:50", wxcodes="BR VCFG"),
            ]
        )
    )
    assert f["precip"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_freezing_reads_temperature_or_frozen_codes():
    f = flags(
        _frame(
            [
                report("X", "2025-01-01 00:50", tmpf="32.0"),  # exactly 0 C
                report("X", "2025-01-01 01:50", tmpf="33.8"),  # 1 C
                report("X", "2025-01-01 02:50", tmpf="40.0", wxcodes="FZFG"),
                report("X", "2025-01-01 03:50", tmpf="40.0", wxcodes="-SN"),
                report("X", "2025-01-01 04:50", tmpf="M"),
            ]
        )
    )
    assert f["freezing"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


# --------------------------------------------------------------------------
# the hourly regime


def test_the_hour_takes_its_worst_report_and_empty_hours_carry_forward():
    h = regime_by_hour(
        _frame(
            [
                report("X", "2025-01-01 00:20", vsby="6.21"),
                report("X", "2025-01-01 00:50", vsby="0.50", wxcodes="FG"),
                # 01:00 and 02:00 have no report
                report("X", "2025-01-01 03:50", vsby="6.21"),
            ]
        )
    ).set_index("hour")
    t = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    assert h.loc[t("2025-01-01 00:00"), "vis_band"] == 2.0  # the worst of the two
    assert h.loc[t("2025-01-01 00:00"), "age_h"] == 0.0
    assert h.loc[t("2025-01-01 01:00"), "vis_band"] == 2.0  # carried
    assert h.loc[t("2025-01-01 01:00"), "age_h"] == 1.0
    assert h.loc[t("2025-01-01 02:00"), "age_h"] == 2.0
    assert h.loc[t("2025-01-01 03:00"), "vis_band"] == 0.0
    assert h.loc[t("2025-01-01 03:00"), "age_h"] == 0.0


def test_a_gap_longer_than_the_carry_is_missing_not_stale():
    rows = [report("X", "2025-01-01 00:50", vsby="0.50")]
    rows.append(report("X", f"2025-01-01 {MAX_CARRY_H + 3:02d}:50", vsby="6.21"))
    h = regime_by_hour(_frame(rows))
    hours = sorted(h["hour"].dt.hour.tolist())
    # 0 and its carried hours, then the report's own hour; the ones between are absent
    assert hours == list(range(0, MAX_CARRY_H + 1)) + [MAX_CARRY_H + 3]


def test_stations_are_kept_apart():
    h = regime_by_hour(
        _frame(
            [
                report("X", "2025-01-01 00:50", vsby="0.50"),
                report("Y", "2025-01-01 00:50", vsby="6.21"),
            ]
        )
    ).set_index(["station", "hour"])
    t = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert h.loc[("X", t), "vis_band"] == 2.0
    assert h.loc[("Y", t), "vis_band"] == 0.0


# --------------------------------------------------------------------------
# the join


def _departures(*stamps, icao="X") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ADEP_mvt": [icao] * len(stamps),
            "MVT_TIME_UTC_mvt": pd.to_datetime(list(stamps), utc=True),
        }
    )


def test_a_departure_reads_the_hour_its_take_off_falls_in():
    h = regime_by_hour(
        _frame(
            [
                report("X", "2025-01-01 10:50", vsby="0.50", wxcodes="-SN", tmpf="30.0"),
                report("X", "2025-01-01 11:50", vsby="6.21"),
            ]
        )
    )
    f = weather_features(_departures("2025-01-01 10:05:00", "2025-01-01 11:59:59"), h)
    assert f["wx_vis_band"].tolist() == [2.0, 0.0]
    assert f["wx_precip"].tolist() == [1.0, 0.0]
    assert f["wx_freezing"].tolist() == [1.0, 0.0]
    assert f["wx_missing"].tolist() == [0.0, 0.0]
    assert f["wx_age_h"].tolist() == [0.0, 0.0]


def test_an_hour_with_nothing_to_read_is_missing_and_says_so():
    h = regime_by_hour(_frame([report("X", "2025-01-01 10:50")]))
    f = weather_features(_departures("2025-06-01 10:05:00", icao="X"), h)
    assert f["wx_missing"].tolist() == [1.0]
    assert np.isnan(f["wx_vis_band"].iloc[0])
    other = weather_features(_departures("2025-01-01 10:05:00", icao="Q"), h)
    assert other["wx_missing"].tolist() == [1.0]


def test_without_a_cache_the_block_is_missing_throughout():
    f = weather_features(_departures("2025-01-01 10:05:00"), None)
    assert list(f.columns) == list(WEATHER_FEATURES)
    assert f["wx_missing"].tolist() == [1.0]
    assert f.drop(columns="wx_missing").isna().all().all()


def test_the_loader_reads_the_archive_format(tmp_path):
    (tmp_path / "X_2025.csv").write_text(
        "station,valid,tmpf,dwpf,vsby,p01i,wxcodes,skyc1,skyl1\n"
        "X,2025-01-15 00:50,33.80,26.60,6.21,0.00,M,OVC,1700.00\n"
        "X,2025-01-15 01:50,M,M,M,M,M,M,M\n",
        encoding="utf-8",
    )
    m = load_metar(tmp_path)
    # The archive's header is the field list, the station first.
    assert list(m.columns) == list(FIELDS)
    assert m["valid"].dt.tz is not None
    assert np.isnan(m["vsby"].iloc[1]) and m["vsby"].iloc[0] == 6.21


# --------------------------------------------------------------------------
# the rule


def test_weather_sources_are_the_subjects_row_and_archive_fields_only():
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (None, "METAR"), (name, src)
            if phase == "METAR":
                assert column in FIELDS, (name, src)
            else:
                assert column in ("MVT_TIME_UTC_mvt", "ADEP_mvt"), (name, src)
