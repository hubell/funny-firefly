# SPDX-License-Identifier: GPL-3.0-only
"""The callsign prefix block, on a hand-built set of flight identifiers."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from taxiout.callsign import (  # noqa: E402
    CALLSIGN_FEATURES,
    MIN_LEVEL_ROWS,
    NA_LEVEL,
    OTHER_LEVEL,
    SOURCES,
    callsign_features,
    frequent_levels,
    pool_rare,
)
from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS  # noqa: E402
from taxiout.features import source_phase  # noqa: E402


def departures(flights, dtype=None) -> pd.DataFrame:
    """A departure frame carrying the one column the block reads.

    The index is labels rather than a range, so a block that rebuilt it
    instead of keeping it would fail rather than pass by coincidence.
    """
    frame = pd.DataFrame({"FLIGHT_mvt": pd.Series(flights, dtype=dtype)})
    frame.index = [f"m{i}" for i in range(len(flights))]
    return frame


def prefixes(flights, dtype=None) -> list:
    return list(callsign_features(departures(flights, dtype))["cs_prefix"])


def test_the_designator_is_the_leading_three_letters():
    assert prefixes(["THY1234", "AFR23AB", "DLH8W", "BAW7"]) == [
        "THY",
        "AFR",
        "DLH",
        "BAW",
    ]


def test_an_identifier_with_no_designator_falls_to_the_na_level():
    # a bare number, two letters then a digit, and a letter then digits
    assert prefixes(["1234", "OK123", "N512SP"]) == [NA_LEVEL] * 3


def test_a_null_identifier_falls_to_the_na_level_and_never_to_a_nan():
    f = callsign_features(departures(["THY1234", None]))
    assert list(f["cs_prefix"]) == ["THY", NA_LEVEL]
    assert f["cs_prefix"].notna().all()


def test_a_lower_case_identifier_still_matches():
    assert prefixes(["baw123", "Klm45x"]) == ["BAW", "KLM"]


def test_a_registration_yields_three_letters_that_are_not_an_airline():
    """The block's own limit, pinned rather than discovered later.

    A registration written without its dash matches the pattern, so the
    level is a registration prefix and not a designator. It is still a
    consistent label for a consistent kind of movement, which is why the
    pattern is not tightened against a list of real designators.
    """
    assert prefixes(["DAIBC", "GEUUU"]) == ["DAI", "GEU"]


def test_a_dictionary_encoded_column_reads_the_same_as_an_object_one():
    """The trap the block converts to strings to avoid.

    ``FLIGHT_mvt`` arrives dictionary-encoded. A categorical's null becomes
    "nan" under ``astype(str)`` and an object column's ``None`` becomes
    "None"; either upper-cases into three letters the pattern matches, so a
    null would take a level of its own and it would not be ``NA``.
    """
    flights = ["THY1234", None, "1234", "baw12"]
    encoded = departures(flights, dtype="category")
    assert isinstance(encoded["FLIGHT_mvt"].dtype, pd.CategoricalDtype)
    assert prefixes(flights, dtype="category") == prefixes(flights)
    assert prefixes(flights) == ["THY", NA_LEVEL, NA_LEVEL, "BAW"]


def test_the_block_carries_exactly_its_column_on_the_given_index():
    frame = departures(["THY1234", "AFR23"])
    f = callsign_features(frame)
    assert list(f.columns) == list(CALLSIGN_FEATURES)
    assert list(f.index) == list(frame.index)
    assert f["cs_prefix"].dtype == object


def test_a_level_at_the_threshold_is_kept_and_one_below_it_is_not():
    values = pd.Series(
        ["THY"] * MIN_LEVEL_ROWS + ["AFR"] * (MIN_LEVEL_ROWS - 1) + [NA_LEVEL]
    )
    assert frequent_levels(values) == frozenset({"THY"})
    assert frequent_levels(values, min_rows=MIN_LEVEL_ROWS - 1) == frozenset(
        {"THY", "AFR"}
    )
    assert frequent_levels(values, min_rows=1) == frozenset({"THY", "AFR", NA_LEVEL})


def test_pooling_leaves_the_listed_alone_and_maps_the_rest_to_other():
    values = pd.Series(["THY", "AFR", NA_LEVEL, "XXX"], index=list("abcd"))
    # ``NA`` is a level like any other: frequent, it survives the pooling
    pooled = pool_rare(values, frozenset({"THY", NA_LEVEL}))
    assert list(pooled) == ["THY", OTHER_LEVEL, NA_LEVEL, OTHER_LEVEL]
    assert list(pooled.index) == list(values.index)
    assert pooled.dtype == object


def test_a_level_the_fitting_sample_never_saw_is_pooled_rather_than_new():
    """What the pooling is for: the ranking set presents no unfitted level."""
    fitting = callsign_features(departures(["THY1234"] * MIN_LEVEL_ROWS))["cs_prefix"]
    levels = frequent_levels(fitting)
    ranking = callsign_features(departures(["THY9", "XYZ9", "1234"]))["cs_prefix"]
    assert list(pool_rare(ranking, levels)) == ["THY", OTHER_LEVEL, OTHER_LEVEL]


def test_the_source_is_the_one_field_the_ranking_set_exposes_for_departures():
    assert list(SOURCES) == list(CALLSIGN_FEATURES)
    for name, sources in SOURCES.items():
        assert sources == ("FLIGHT_mvt",), name
        for src in sources:
            phase, column = source_phase(src)
            assert phase is None, (name, src)
            assert column not in FORBIDDEN_FEATURE_INPUTS, (name, src)
            assert column != "AOBT_3_flt"


def test_a_frame_without_the_identifier_is_refused():
    with pytest.raises(KeyError):
        callsign_features(pd.DataFrame({"ADEP_mvt": ["ZZZZ"]}))
