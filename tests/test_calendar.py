# SPDX-License-Identifier: GPL-3.0-only
"""The holiday block: the flags, the adjacency rule, and what is missing.

The dates are chosen to be unambiguous and stable -- fixed-date national
holidays, never a movable feast and never one a subdivision decides -- so
that a new release of the calendars cannot quietly change what the block is
asserted to say.

The stamps are built at microsecond resolution, as the challenge parquet
files carry them and as the ladder therefore hands them to the block. A
fixture built at nanoseconds agrees with any implementation that reads the
integer view as nanoseconds, including a wrong one: ``taxiout.windows.seconds``
was once exactly that, and the nanosecond fixtures of the day never saw it
(``tests/test_occupancy.py``).
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.calendar import (  # noqa: E402
    COUNTRY,
    HOLIDAY_FEATURES,
    SOURCES,
    holiday_features,
)
from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS  # noqa: E402
from taxiout.features import source_phase  # noqa: E402


def frame(*rows) -> pd.DataFrame:
    """``(icao, date)`` pairs as departures taking off at midday UTC.

    At the resolution of the parquet files, for the reason the module
    docstring gives.
    """
    departures = pd.DataFrame(
        {
            "ADEP_mvt": [icao for icao, _ in rows],
            "MVT_TIME_UTC_mvt": [pd.Timestamp(f"{d}T12:00:00Z") for _, d in rows],
        },
        index=[7 * i + 3 for i in range(len(rows))],
    )
    departures["MVT_TIME_UTC_mvt"] = pd.to_datetime(
        departures["MVT_TIME_UTC_mvt"], utc=True
    ).dt.as_unit("us")
    return departures


def test_the_flags_are_the_same_whatever_resolution_the_stamps_arrive_at():
    """The date is taken by a unit-aware cast, not by dividing the integer view.

    An implementation that divides by a nanosecond day agrees with this frame
    at nanoseconds and puts every microsecond stamp in January 1970, where no
    date is a holiday. Both columns then go to zero rather than to NaN, so the
    ladder's missing-block check sees a block that is present and fits on a
    column asserting that the year has no holidays in it.
    """
    d = frame(
        ("EDDF", "2025-01-01"),  # New Year's Day, with two ordinary neighbours
        ("LFPG", "2025-07-14"),  # Bastille Day, likewise
        ("EDDF", "2025-01-02"),  # the day after a holiday
        ("EGLL", "2025-06-17"),  # an ordinary Tuesday in June
    )
    assert str(d["MVT_TIME_UTC_mvt"].dtype) == "datetime64[us, UTC]"
    pytest.importorskip("holidays")
    for unit in ("s", "ms", "us", "ns"):
        at_unit = d.assign(MVT_TIME_UTC_mvt=d["MVT_TIME_UTC_mvt"].dt.as_unit(unit))
        f = holiday_features(at_unit)
        assert list(f["holiday"]) == [1.0, 1.0, 0.0, 0.0], unit
        assert list(f["holiday_adjacent"]) == [0.0, 0.0, 1.0, 0.0], unit


def test_new_year_is_a_holiday_at_every_aerodrome_in_the_map():
    pytest.importorskip("holidays")
    f = holiday_features(frame(*[(icao, "2025-01-01") for icao in COUNTRY]))
    assert list(f["holiday"]) == [1.0] * len(COUNTRY)


def test_christmas_is_read_from_the_country_and_not_from_the_calendar():
    pytest.importorskip("holidays")
    f = holiday_features(
        frame(
            ("EDDF", "2025-12-25"),
            ("EGLL", "2025-12-25"),
            ("LEMD", "2025-12-25"),
            ("LTFM", "2025-12-25"),  # not a public holiday in Turkey
        )
    )
    assert list(f["holiday"]) == [1.0, 1.0, 1.0, 0.0]


def test_a_national_holiday_is_read_at_its_own_countrys_aerodromes_only():
    pytest.importorskip("holidays")
    # German Unity Day, Bastille Day, Republic Day: each a holiday in one country
    f = holiday_features(
        frame(
            ("EDDM", "2025-10-03"),
            ("EGLL", "2025-10-03"),
            ("LFPG", "2025-07-14"),
            ("LIRF", "2025-07-14"),
            ("LTAI", "2025-10-29"),
            ("LSZH", "2025-10-29"),
        )
    )
    assert list(f["holiday"]) == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_adjacency_is_the_neighbouring_dates_and_not_the_date_itself():
    pytest.importorskip("holidays")
    f = holiday_features(
        frame(
            ("EDDF", "2025-01-02"),  # the day after New Year's Day
            ("EDDF", "2025-10-03"),  # a holiday with two ordinary neighbours
            ("EDDF", "2025-12-25"),  # a holiday beside Boxing Day
            ("LEBL", "2025-12-25"),  # Spain has no Boxing Day at country level
            ("EHAM", "2025-06-17"),  # an ordinary Tuesday in June
        )
    )
    assert list(f["holiday"]) == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert list(f["holiday_adjacent"]) == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_the_year_either_side_is_built_so_the_boundary_reads_across_it():
    pytest.importorskip("holidays")
    # the only date in the frame is in 2025; the neighbour it needs is in 2026
    f = holiday_features(frame(("LFPG", "2025-12-31")))
    assert f["holiday"].iloc[0] == 0.0 and f["holiday_adjacent"].iloc[0] == 1.0


def test_an_aerodrome_with_no_country_is_missing_rather_than_zero():
    f = holiday_features(frame(("ZZZZ", "2025-01-01"), ("EDDF", "2025-01-01")))
    assert np.isnan(f["holiday"].iloc[0]) and np.isnan(f["holiday_adjacent"].iloc[0])


def test_a_missing_take_off_time_is_missing_in_both_columns():
    d = frame(("EDDF", "2025-01-01"), ("EDDF", "2025-01-01"))
    d.loc[d.index[0], "MVT_TIME_UTC_mvt"] = pd.NaT
    f = holiday_features(d)
    assert np.isnan(f["holiday"].iloc[0]) and np.isnan(f["holiday_adjacent"].iloc[0])


def test_the_block_is_indexed_like_the_input_and_carries_exactly_the_features():
    d = frame(("EDDF", "2025-01-01"), ("LTAI", "2025-07-04"), ("ZZZZ", "2025-03-09"))
    f = holiday_features(d)
    assert list(f.columns) == list(HOLIDAY_FEATURES)
    assert f.index.equals(d.index)
    assert holiday_features(d.iloc[:0]).empty


def test_every_airport_in_scope_has_one_country():
    assert set(COUNTRY) == {a.icao for a in AIRPORTS}
    for code in COUNTRY.values():
        assert len(code) == 2 and code.isupper()


def test_sources_are_the_subjects_own_exposed_fields_under_no_prefix():
    assert list(SOURCES) == list(HOLIDAY_FEATURES)
    for name, sources in SOURCES.items():
        assert sources == ("ADEP_mvt", "MVT_TIME_UTC_mvt"), name
        for src in sources:
            phase, column = source_phase(src)
            assert phase is None, (name, src)  # no external archive, no other row
            assert column not in FORBIDDEN_FEATURE_INPUTS
