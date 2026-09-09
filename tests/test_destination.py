# SPDX-License-Identifier: GPL-3.0-only
"""The destination block, on a hand-built airports table.

Nothing here reads the cache. The positions below are published aerodrome
reference points, written out so the geodesy is checked against real pairs at
the scale the feature runs at -- a few hundred to a few thousand kilometres --
rather than at the airport scale ``tests/test_geo.py`` covers.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS  # noqa: E402
from taxiout.destination import (  # noqa: E402
    DESTINATION_FEATURES,
    FIELDS,
    SOURCES,
    bearings_deg,
    destination_features,
    distances_km,
    load_airports,
)
from taxiout.features import source_phase  # noqa: E402
from taxiout.geo import bearing_deg, geodesic_m  # noqa: E402

POINTS: dict[str, tuple[float, float]] = {
    "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3105, 4.7683),
    "EDDF": (50.0333, 8.5706),
    "LEMD": (40.4719, -3.5626),
    "LTFM": (41.2753, 28.7519),
    "LTAI": (36.8987, 30.8005),
}

# Four legs, chosen to run in different directions and over different lengths:
# 371 km east-north-east, 1,423 km south-west, 2,188 km east-south-east, 517 km
# south-east. Direction matters here because the radius the scalar fits depends
# on it.
PAIRS: tuple[tuple[str, str], ...] = (
    ("EGLL", "EHAM"),
    ("EDDF", "LEMD"),
    ("EHAM", "LTFM"),
    ("LTFM", "LTAI"),
)


def _airports(idents=None) -> pd.DataFrame:
    wanted = POINTS if idents is None else {k: POINTS[k] for k in idents}
    return pd.DataFrame(
        {
            "ident": list(wanted),
            "latitude_deg": [p[0] for p in wanted.values()],
            "longitude_deg": [p[1] for p in wanted.values()],
        }
    )


def _departures(*pairs) -> pd.DataFrame:
    return pd.DataFrame(
        {"ADEP_mvt": [a for a, _ in pairs], "ADES_mvt": [b for _, b in pairs]}
    )


def _ends(pairs):
    a = [POINTS[x] for x, _ in pairs]
    b = [POINTS[y] for _, y in pairs]
    return a, b


# --------------------------------------------------------------------------
# the geodesy


def test_the_vectorised_bearing_is_the_scalar_one_to_the_last_digit():
    a, b = _ends(PAIRS)
    got = bearings_deg(
        [p[0] for p in a], [p[1] for p in a], [q[0] for q in b], [q[1] for q in b]
    )
    want = [bearing_deg(p, q) for p, q in zip(a, b)]
    # The same expression over arrays instead of floats, so anything short of
    # exact agreement is a transcription error rather than a modelling choice.
    assert list(got) == pytest.approx(want, abs=1e-9)


def test_the_vectorised_distance_matches_the_scalar_within_the_radius_spread():
    a, b = _ends(PAIRS)
    got = distances_km(
        [p[0] for p in a], [p[1] for p in a], [q[0] for q in b], [q[1] for q in b]
    )
    want = [geodesic_m(p, q) / 1000.0 for p, q in zip(a, b)]
    # The two do not agree exactly and are not meant to. ``geo.geodesic_m``
    # fits its sphere to the midpoint latitude and to the azimuth of the line,
    # which is right under fifteen kilometres and drifts on a leg of hundreds;
    # this module runs on the fixed mean radius. The gap is where the local
    # radius sits against 6,371 km -- 6,335 km meridional at the equator to
    # 6,400 km prime-vertical at the pole -- so half a percent bounds it. The
    # second assertion pins the measured worst of these four at 0.30%, so the
    # tolerance cannot loosen quietly if either formula is changed.
    assert list(got) == pytest.approx(want, rel=0.005)
    assert max(abs(g - w) / w for g, w in zip(got, want)) < 0.004


def test_bearings_from_a_point_read_north_east_and_south():
    lat, lon = 50.0, 8.0
    got = bearings_deg(
        [lat] * 3, [lon] * 3, [lat + 1.0, lat, lat - 1.0], [lon, lon + 1.0, lon]
    )
    # Due east reads a third of a degree short of 90: an initial great-circle
    # bearing along a parallel is not 90, because the meridians converge and
    # the circle sets off north of the parallel. Checked at half a degree, as
    # ``tests/test_geo.py`` checks the scalar's own cardinals.
    assert got[0] == pytest.approx(0.0, abs=1e-9)
    assert got[1] == pytest.approx(90.0, abs=0.5)
    assert got[2] == pytest.approx(180.0, abs=1e-9)


# --------------------------------------------------------------------------
# the block


def test_the_block_reads_both_ends_from_the_table():
    f = destination_features(_departures(("EGLL", "EHAM")), _airports())
    p, q = POINTS["EGLL"], POINTS["EHAM"]
    assert f["dest_bearing_deg"].iloc[0] == pytest.approx(bearing_deg(p, q), abs=1e-9)
    assert f["dest_km"].iloc[0] == pytest.approx(geodesic_m(p, q) / 1000.0, rel=0.005)
    assert f["dest_missing"].iloc[0] == 0.0


def test_a_null_destination_and_one_the_table_lacks_are_missing_not_zero():
    f = destination_features(
        _departures(("EGLL", "EHAM"), ("EGLL", None), ("EGLL", "ZZZZ")), _airports()
    )
    assert f["dest_missing"].tolist() == [0.0, 1.0, 1.0]
    assert f[["dest_bearing_deg", "dest_km"]].iloc[1:].isna().all().all()


def test_a_departure_aerodrome_the_table_lacks_is_missing_too():
    """The flag is the union: no pair of positions, no geometry, whichever end."""
    f = destination_features(
        _departures(("EGLL", "EHAM")), _airports(["EHAM", "EDDF", "LEMD"])
    )
    assert f["dest_missing"].tolist() == [1.0]
    assert f[["dest_bearing_deg", "dest_km"]].isna().all().all()


def test_without_the_cache_the_block_is_missing_throughout(tmp_path):
    d = _departures(("EGLL", "EHAM"), ("EDDF", "LEMD"))
    for airports in (None, load_airports(tmp_path / "never_fetched.csv")):
        f = destination_features(d, airports)
        assert list(f.columns) == list(DESTINATION_FEATURES)
        assert f["dest_missing"].tolist() == [1.0, 1.0]
        assert f.drop(columns="dest_missing").isna().all().all()


def test_the_result_is_indexed_like_the_input_and_carries_exactly_the_block():
    d = _departures(("EGLL", "EHAM"), ("EDDF", "LEMD"), ("LTFM", "LTAI"))
    d.index = [7, 11, 13]
    f = destination_features(d, _airports())
    assert list(f.index) == [7, 11, 13]
    assert list(f.columns) == list(DESTINATION_FEATURES)
    assert f["dest_missing"].tolist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------
# the loader


def test_the_loader_keeps_what_it_reads_and_drops_what_it_cannot_use(tmp_path):
    path = tmp_path / "airports.csv"
    path.write_text(
        "id,ident,type,name,latitude_deg,longitude_deg,elevation_ft\n"
        "1,AAAA,large_airport,First,51.4700,-0.4543,83\n"
        "2,AAAA,large_airport,First again by another name,52.0,1.0,10\n"
        "3,,heliport,No identifier,50.0,8.0,100\n"
        "4,BBBB,small_airport,No position,,,\n"
        "5,CCCC,small_airport,Position that is not a number,x,8.0,\n",
        encoding="utf-8",
    )
    a = load_airports(path)
    assert list(a.columns) == list(FIELDS)
    # The duplicate ident keeps its first row, and the three unusable ones go.
    assert a["ident"].tolist() == ["AAAA"]
    assert a["latitude_deg"].iloc[0] == pytest.approx(51.47)
    assert a["longitude_deg"].iloc[0] == pytest.approx(-0.4543)


def test_the_loader_without_the_file_is_an_empty_table_not_an_error(tmp_path):
    a = load_airports(tmp_path / "never_fetched.csv")
    assert a.empty and list(a.columns) == list(FIELDS)


# --------------------------------------------------------------------------
# the rule


def test_sources_are_the_two_aerodrome_codes_and_the_archives_own_fields():
    assert list(SOURCES) == list(DESTINATION_FEATURES)
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (None, "OURAIRPORTS"), (name, src)
            if phase == "OURAIRPORTS":
                assert column in FIELDS, (name, src)
            else:
                assert column in ("ADEP_mvt", "ADES_mvt"), (name, src)
                assert column not in FORBIDDEN_FEATURE_INPUTS, (name, src)
