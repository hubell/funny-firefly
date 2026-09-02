# SPDX-License-Identifier: GPL-3.0-only
"""The 11 airports in scope for the PRC Data Challenge 2026.

Coordinates are published aerodrome reference points. ``radius_m`` is sized to
enclose the full movement area (aprons, taxiways, all runways) with margin --
it is deliberately generous, since Overpass filtering is cheap relative to a
second round trip.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Airport:
    icao: str
    iata: str
    name: str
    lat: float
    lon: float
    radius_m: int


AIRPORTS: tuple[Airport, ...] = (
    Airport("EDDF", "FRA", "Frankfurt Main", 50.0333, 8.5706, 5500),
    Airport("EDDM", "MUC", "Munich", 48.3538, 11.7861, 5500),
    Airport("EGLL", "LHR", "London Heathrow", 51.4700, -0.4543, 4500),
    Airport("EHAM", "AMS", "Amsterdam Schiphol", 52.3105, 4.7683, 6500),
    Airport("LEBL", "BCN", "Barcelona-El Prat", 41.2974, 2.0833, 4500),
    Airport("LEMD", "MAD", "Madrid-Barajas", 40.4719, -3.5626, 6500),
    Airport("LFPG", "CDG", "Paris Charles de Gaulle", 49.0097, 2.5479, 6500),
    Airport("LIRF", "FCO", "Rome Fiumicino", 41.8003, 12.2389, 5500),
    Airport("LTAI", "AYT", "Antalya", 36.8987, 30.8005, 5000),
    Airport("LTFM", "IST", "Istanbul", 41.2753, 28.7519, 8500),
    Airport("LSZH", "ZRH", "Zurich", 47.4647, 8.5492, 4500),
)

BY_ICAO: dict[str, Airport] = {a.icao: a for a in AIRPORTS}
