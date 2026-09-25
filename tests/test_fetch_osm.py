# SPDX-License-Identifier: GPL-3.0-only
"""The OSM fetcher's pin modes and the two halves of a rebuild's proof, offline.

Nothing here reaches the network, and a test that drives ``main`` is written so
that it cannot: ``fetch_airport`` is replaced by a recorder, and the cache is
redirected under ``tmp_path``, so a mode that misbehaved could neither send a
query nor write into the real cache. Every answer below is invented; no OSM
content is committed here.

Three things are held. **The retired mode refuses, and says why**: it read each
airport's pin out of its cache header, which carries the server's database time
on the day of the fetch and never the pin, so after a rebuild it would have
pinned every airport to the rebuild day. **The mode that replaced it pins every
airport to its register entry.** And **the element comparison and the control
probe each report a difference planted where they cannot miss it**, because a
header cannot prove a pin was applied and those two are what can.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import fetch_osm  # noqa: E402
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.osm import compare_elements  # noqa: E402
from taxiout.provenance import OSM_PINS  # noqa: E402

MIRROR = "https://instance.invalid/api/interpreter"
LIRF_PIN = "2026-09-01T13:29:10Z"
CONTROL = "2020-01-01T00:00:00Z"


def _answer(elements, *, pin=None, mirror=None, base="2026-09-25T11:21:36Z"):
    """An answer in the shape the client stores, with a fetch record if asked."""
    payload = {
        "generator": "Overpass API, invented",
        "osm3s": {"timestamp_osm_base": base},
        "elements": elements,
    }
    if mirror:
        payload["taxiout:fetch"] = {
            "mirror": mirror,
            "date_pin": pin,
            "fetched_utc": "2026-09-25T11:23:03Z",
        }
    return payload


def _elements():
    """Two nodes and a way, invented."""
    return [
        {"type": "node", "id": 1, "lat": 1.5, "lon": 2.5, "tags": {"aeroway": "x"}},
        {"type": "node", "id": 2, "lat": 1.75, "lon": 2.75},
        {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"aeroway": "taxiway"}},
    ]


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """The cache under ``tmp_path``, and every fetch recorded and refused.

    Recorded rather than raised, because ``main`` catches what a fetch raises
    and prints it as that airport's failure: a raise would be swallowed, and a
    list the test asserts empty cannot be.
    """
    cache = tmp_path / "raw"
    attempts: list[str] = []

    def refuse(ap, *args, **kwargs):
        attempts.append(ap.icao)
        raise RuntimeError("this test sends nothing")

    monkeypatch.setattr(fetch_osm, "CACHE", cache)
    monkeypatch.setattr(fetch_osm, "fetch_airport", refuse)
    monkeypatch.setattr(fetch_osm.time, "sleep", lambda _s: None)
    return cache, attempts


# --------------------------------------------------------------------------
# the retired mode and the one that replaced it


def test_pin_from_cache_refuses_and_says_why(offline, capsys):
    """It fails before anything is read or sent, however it is combined.

    Kept as a flag that refuses rather than deleted, so the command recorded
    before this fails with its reason and not with "unrecognized arguments".
    """
    cache, attempts = offline
    for argv in (
        ["--pin-from-cache", "--force"],
        ["--pin-from-cache"],
        ["--pin-from-cache", "--force", "--only", "LIRF"],
    ):
        with pytest.raises(SystemExit) as exit_:
            fetch_osm.main(argv)
        assert exit_.value.code == 2
        err = capsys.readouterr().err
        assert "--pin-from-cache is retired" in err
        assert "never the pin" in err
        assert "--pin-from-register" in err
    assert attempts == []
    assert not cache.exists()


def test_pin_from_register_resolves_every_airport_to_its_own_entry(offline, monkeypatch):
    """Driven through ``main``, the path a rebuild takes, with the fetch recorded."""
    cache, _attempts = offline
    calls = []

    def record(ap, cache_dir, *, force=False, date=None, **kwargs):
        calls.append((ap.icao, date, force, cache_dir))
        return _answer(_elements(), pin=date, mirror=MIRROR), False

    monkeypatch.setattr(fetch_osm, "fetch_airport", record)
    assert fetch_osm.main(["--pin-from-register", "--force"]) == 0

    pins = {p.icao: p.timestamp_osm_base for p in OSM_PINS}
    assert [icao for icao, *_ in calls] == [a.icao for a in AIRPORTS]
    for icao, date, force, cache_dir in calls:
        assert date is not None and date == pins[icao], icao
        assert force is True
        assert cache_dir == cache
    assert [fetch_osm.register_pin(a.icao) for a in AIRPORTS] == [
        pins[a.icao] for a in AIRPORTS
    ]


def test_pin_from_register_needs_force_and_never_fetches_unpinned(
    offline, monkeypatch, capsys
):
    """Without --force the cache answers unread, and a run that pinned nothing
    would look like a pinned rebuild. An airport with no entry is a failure for
    that airport, never an unpinned fetch."""
    _cache, attempts = offline
    with pytest.raises(SystemExit) as exit_:
        fetch_osm.main(["--pin-from-register"])
    assert exit_.value.code == 2
    assert "needs --force" in capsys.readouterr().err

    monkeypatch.setattr(
        fetch_osm, "OSM_PINS", tuple(p for p in OSM_PINS if p.icao != "LIRF")
    )
    assert fetch_osm.main(["--pin-from-register", "--force", "--only", "LIRF"]) == 1
    out = capsys.readouterr().out
    assert "LIRF  FAILED  no pin" in out and "refusing to fetch unpinned" in out
    assert attempts == []


# --------------------------------------------------------------------------
# the proof: the elements, and a control the instance must answer differently


def test_the_element_comparison_reports_a_planted_difference():
    """Each kind of difference is counted, and a header difference is none."""
    reference = _answer(_elements(), base=LIRF_PIN)
    same = _answer(_elements(), pin=LIRF_PIN, mirror=MIRROR)
    c = compare_elements(same, reference)
    assert c.identical and c.same_elements
    assert c.fetched == c.reference == {"node": 2, "way": 1}

    planted = _elements()
    planted[1]["lat"] = 1.875  # a node moved
    planted[2]["tags"] = {"aeroway": "taxilane"}  # the way retagged
    planted.append({"type": "relation", "id": 99, "members": []})  # one extra
    del planted[0]  # one missing
    c = compare_elements(_answer(planted), reference)
    assert not c.identical and not c.same_elements
    assert c.fetched == {"node": 1, "way": 1, "relation": 1}
    assert c.reference == {"node": 2, "way": 1}
    assert (c.only_fetched, c.only_reference, c.in_both) == (1, 1, 2)
    assert c.differing == 2
    assert c.fields == {"lat": 1, "tags": 1}

    # The same elements in another order are not identical, and neither is a
    # repeated key, which a lookup by key would otherwise hide.
    c = compare_elements(_answer(_elements()[::-1]), reference)
    assert c.same_elements and not c.same_order and not c.identical
    c = compare_elements(_answer(_elements() + _elements()[:1]), reference)
    assert c.duplicates == 1 and not c.identical


def test_compare_prints_both_sides_and_fails_on_a_planted_difference(
    offline, tmp_path, capsys
):
    """The command a rebuild is checked with: counts on both sides, the pin the
    fetched answer was requested at, and a non-zero exit on one reordered node
    list in one way."""
    fetched, reference = tmp_path / "fetched", tmp_path / "reference"
    _write(fetched / "LIRF.json.gz", _answer(_elements(), pin=LIRF_PIN, mirror=MIRROR))
    _write(reference / "LIRF.json.gz", _answer(_elements(), base=LIRF_PIN))
    argv = ["--compare", str(fetched), str(reference), "--only", "LIRF"]

    assert fetch_osm.main(argv) == 0
    out = capsys.readouterr().out
    assert "LIRF  identical" in out
    assert "    2 nodes      1 ways      0 relations" in out
    assert "0 differing in any field" in out
    assert "was requested at it" in out
    assert MIRROR in out

    planted = _elements()
    planted[2]["nodes"] = [2, 1]
    _write(fetched / "LIRF.json.gz", _answer(planted, pin=LIRF_PIN, mirror=MIRROR))
    assert fetch_osm.main(argv) == 1
    out = capsys.readouterr().out
    assert "LIRF  NOT IDENTICAL" in out
    assert "1 differing in any field (nodes 1)" in out

    # An answer compared with itself proves nothing, and is refused.
    assert fetch_osm.main(["--compare", str(fetched), str(fetched)]) == 1
    assert "proves nothing" in capsys.readouterr().out


def test_the_control_asks_the_instance_that_answered_and_writes_nothing(
    offline, monkeypatch, capsys
):
    """A different answer at the control date proves the instance applies a pin.

    The same answer proves nothing: either the date was ignored or the airport
    has not changed since it. That is the check that cannot be made on the
    header, which carries the instance's own database state whatever it was
    asked.
    """
    cache, _attempts = offline
    cached = _write(
        cache / "LIRF.json.gz", _answer(_elements(), pin=LIRF_PIN, mirror=MIRROR)
    )
    before = cached.read_bytes()
    asked = []
    reply = {"elements": _elements()[:1]}

    def probe(ap, cache_dir, *, force=False, date=None, mirrors=(), **kwargs):
        asked.append((ap.icao, date, force, tuple(mirrors), Path(cache_dir)))
        return _answer(reply["elements"], pin=date, mirror=mirrors[0]), False

    monkeypatch.setattr(fetch_osm, "fetch_airport", probe)
    argv = ["--control-date", CONTROL, "--only", "LIRF"]

    assert fetch_osm.main(argv) == 0
    out = capsys.readouterr().out
    assert "DIFFERENT" in out and "applies [date:]" in out
    icao, date, force, mirrors, where = asked[-1]
    assert (icao, date, force, mirrors) == ("LIRF", CONTROL, True, (MIRROR,))
    assert where != cache, "the control is fetched outside the cache"

    reply["elements"] = _elements()
    assert fetch_osm.main(argv) == 1
    out = capsys.readouterr().out
    assert "THE SAME" in out and "unproven" in out

    # The airport's own pin is not a control, and one airport must be named.
    assert fetch_osm.main(["--control-date", LIRF_PIN, "--only", "LIRF"]) == 1
    assert "own pin" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        fetch_osm.main(["--control-date", CONTROL])

    assert cached.read_bytes() == before
    assert sorted(p.name for p in cache.iterdir()) == ["LIRF.json.gz"]
