# SPDX-License-Identifier: GPL-3.0-only
"""The OSM fetcher's pin modes and the two halves of a rebuild's proof, offline.

Nothing here reaches the network, and a test that drives ``main`` is written so
that it cannot: ``fetch_airport`` is replaced by a recorder, and the cache is
redirected under ``tmp_path``, so a mode that misbehaved could neither send a
query nor write into the real cache. Every answer below is invented; no OSM
content is committed here.

Four things are held. **The retired mode refuses, and says why**: it read each
airport's pin out of its cache header, which carries the server's database time
on the day of the fetch and never the pin, so after a rebuild it would have
pinned every airport to the rebuild day. **The mode that replaced it pins every
airport to its register entry**, and records each fetch in the sidecar a
rebuilt cache is read by, since its header carries the rebuild day. **The
client refuses a pinned answer from a database older than its pin**, because
an instance cannot serve a state it has not ingested: the header cannot show a
pin was applied, but it can show one could not have been. And **the element
comparison and the control probe each report a difference planted where they
cannot miss it**, because those two are what can prove a pin was applied.
"""

from __future__ import annotations

import gzip
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import fetch_osm  # noqa: E402
from taxiout import osm  # noqa: E402
from taxiout.airports import AIRPORTS, BY_ICAO  # noqa: E402
from taxiout.osm import (  # noqa: E402
    MIRRORS,
    cache_path,
    compare_elements,
    fetch_airport,
    pin_agreement,
    read_payload,
    read_sidecar,
    stamp_precedes_pin,
)
from taxiout.provenance import OSM_PINS  # noqa: E402

MIRROR = "https://instance.invalid/api/interpreter"
LIRF_PIN = "2026-09-01T13:29:10Z"
LSZH_PIN = "2026-09-01T13:32:15Z"
CONTROL = "2020-01-01T00:00:00Z"
# The two databases the fallback mirrors stood at on 26 September 2026.
MAY_6 = "2026-05-06T03:25:00Z"
MAY_31 = "2026-05-31T22:37:44Z"
REBUILD_DAY = "2026-09-26T07:36:11Z"


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
            "timestamp_osm_base": base,
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


def test_every_fetch_is_recorded_beside_the_cache_with_its_pin(offline, monkeypatch):
    """The sidecar a rebuilt cache is read by, written and read through ``main``.

    One entry per airport fetched, copied from the answer's own fetch record, so
    ``pin_agreement`` finds the pin that the header, stamped with the rebuild
    day, cannot carry. An unpinned refetch replaces its airport's entry with a
    null pin rather than leaving the pinned one to vouch for a file it no longer
    describes, and an airport served from the cache leaves its entry alone.
    """
    cache, _attempts = offline

    def fetch(ap, cache_dir, *, force=False, date=None, **kwargs):
        path = cache_path(cache_dir, ap.icao)
        if path.exists() and not force:
            return read_payload(path), True
        answer = _answer(_elements(), pin=date, mirror=MIRROR)
        _write(path, answer)
        return answer, False

    monkeypatch.setattr(fetch_osm, "fetch_airport", fetch)
    assert fetch_osm.main(["--pin-from-register", "--force"]) == 0

    pins = {p.icao: p.timestamp_osm_base for p in OSM_PINS}
    written = read_sidecar(cache)
    assert sorted(written) == sorted(pins)
    for icao, entry in written.items():
        assert entry == {
            "date_pin": pins[icao],
            "mirror": MIRROR,
            "fetched_utc": "2026-09-25T11:23:03Z",
            "timestamp_osm_base": "2026-09-25T11:21:36Z",
        }, icao
    checks = pin_agreement(cache, pins)
    assert len(checks) == len(pins)
    assert all(c.branch == "rebuilt" and c.problem is None for c in checks)

    # Served from the cache: nothing fetched, nothing recorded.
    before = (cache / "PINS.json").read_bytes()
    assert fetch_osm.main(["--only", "EDDF"]) == 0
    assert (cache / "PINS.json").read_bytes() == before

    # Refetched unpinned: the entry says so, and the check fails that airport.
    assert fetch_osm.main(["--force", "--only", "LIRF"]) == 0
    assert read_sidecar(cache)["LIRF"]["date_pin"] is None
    assert [c.icao for c in pin_agreement(cache, pins) if c.problem] == ["LIRF"]


def test_a_recorded_stamp_before_its_pin_fails_the_sidecar_check(offline, monkeypatch):
    """Block 11's LSZH, planted: requested at its pin, answered from May.

    The fetch here stands in for a client without the rule, so the stale answer
    reaches the cache and the sidecar. The sidecar records the stamp beside the
    pin requested, and the check fails that airport on the stamp alone, which
    is what it could not do when it recorded the request and nothing else.
    """
    cache, _attempts = offline
    stale = {"LSZH": MAY_6, "LIRF": MAY_31}

    def fetch(ap, cache_dir, *, force=False, date=None, **kwargs):
        answer = _answer(
            _elements(), pin=date, mirror=MIRROR, base=stale.get(ap.icao, REBUILD_DAY)
        )
        _write(cache_path(cache_dir, ap.icao), answer)
        return answer, False

    monkeypatch.setattr(fetch_osm, "fetch_airport", fetch)
    assert fetch_osm.main(["--pin-from-register", "--force"]) == 0

    pins = {p.icao: p.timestamp_osm_base for p in OSM_PINS}
    assert read_sidecar(cache)["LSZH"]["timestamp_osm_base"] == MAY_6
    bad = {c.icao: c.problem for c in pin_agreement(cache, pins) if c.problem}
    assert sorted(bad) == ["LIRF", "LSZH"]
    for icao, problem in bad.items():
        assert stale[icao] in problem and pins[icao] in problem
        assert "cannot have been served at the pin" in problem


# --------------------------------------------------------------------------
# the rule: a pinned answer from a database older than its pin is refused


@pytest.fixture
def mirrors(monkeypatch):
    """The three mirrors answered from a script, and every ask recorded.

    ``script[mirror]`` is a list of replies, one per ask, the last repeated:
    a payload is answered, an int is an HTTP error with that code.
    """
    asked: list[str] = []
    script: dict[str, list] = {}

    def post(mirror, query, timeout):
        asked.append(mirror)
        replies = script[mirror]
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, int):
            raise urllib.error.HTTPError(mirror, reply, "scripted", {}, None)
        return json.loads(json.dumps(reply))

    monkeypatch.setattr(osm, "_post", post)
    monkeypatch.setattr(osm.time, "sleep", lambda _s: None)
    return script, asked


def test_a_pinned_answer_from_a_database_older_than_its_pin_is_refused(mirrors, tmp_path):
    """The primary throttled, both fallbacks stale, then the primary answers.

    Each fallback is asked once and refused, never asked again, and nothing
    they sent is written; the primary's answer is the one cached, with its
    stamp in its fetch record. Every failure and refusal reaches the log.
    """
    script, asked = mirrors
    primary, coffee, kumi = MIRRORS
    script[primary] = [429, _answer(_elements(), base=REBUILD_DAY)]
    script[coffee] = [_answer(_elements()[:2], base=MAY_6)]
    script[kumi] = [_answer(_elements()[:1], base=MAY_31)]
    logged: list[str] = []

    payload, cached = fetch_airport(
        BY_ICAO["LSZH"], tmp_path, force=True, date=LSZH_PIN, log=logged.append
    )
    assert not cached
    assert asked == [primary, coffee, kumi, primary]
    rec = payload["taxiout:fetch"]
    assert rec["mirror"] == primary and rec["date_pin"] == LSZH_PIN
    assert rec["timestamp_osm_base"] == REBUILD_DAY
    stored = read_payload(cache_path(tmp_path, "LSZH"))
    assert stored["taxiout:fetch"] == rec
    assert compare_elements(stored, _answer(_elements())).identical
    refusals = [line for line in logged if "REFUSED" in line]
    assert len(refusals) == 2
    assert coffee in refusals[0] and MAY_6 in refusals[0]
    assert kumi in refusals[1] and MAY_31 in refusals[1]
    assert any("HTTP 429" in line and primary in line for line in logged)


def test_a_pin_no_mirror_can_serve_is_a_stop_and_writes_nothing(mirrors, tmp_path):
    """Every mirror failed or refused: the error names each, and nothing lands.

    The primary is asked as often as the whole rotation would have been, since
    it is the only mirror left once the others are set aside; an unpinned fetch
    of the same airport gets the ordinary number of rounds and takes the stale
    answer, because without a pin there is nothing to refuse.
    """
    script, asked = mirrors
    primary, coffee, kumi = MIRRORS
    script[primary] = [504]
    script[coffee] = [_answer(_elements(), base=MAY_6)]
    script[kumi] = [_answer(_elements(), base=MAY_31)]

    with pytest.raises(RuntimeError) as err:
        fetch_airport(BY_ICAO["LSZH"], tmp_path, force=True, date=LSZH_PIN)
    message = str(err.value)
    assert "cannot be served today" in message
    for mirror, stamp in ((coffee, MAY_6), (kumi, MAY_31)):
        assert f"{mirror}: refused, its database is at {stamp}" in message
    assert f"{primary}: HTTP 504" in message
    assert asked.count(coffee) == asked.count(kumi) == 1
    assert asked.count(primary) == 2 * len(MIRRORS)
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []

    asked.clear()
    payload, _ = fetch_airport(BY_ICAO["LSZH"], tmp_path, force=True)
    assert payload["taxiout:fetch"]["mirror"] == coffee
    assert payload["taxiout:fetch"]["timestamp_osm_base"] == MAY_6
    assert payload["taxiout:fetch"]["date_pin"] is None
    assert asked == [primary, coffee]


def test_the_stamp_and_the_pin_are_compared_as_moments_and_not_as_strings():
    """An offset written another way cannot reorder them, and no stamp is none.

    09:00 at minus five hours is 14:00Z, after a 13:32:15Z pin, though the
    string sorts before it.
    """
    assert stamp_precedes_pin(MAY_6, LSZH_PIN)
    assert not stamp_precedes_pin(REBUILD_DAY, LSZH_PIN)
    assert not stamp_precedes_pin(LSZH_PIN, LSZH_PIN), "the pin itself serves"
    assert "2026-09-01T09:00:00-05:00" < LSZH_PIN
    assert not stamp_precedes_pin("2026-09-01T09:00:00-05:00", LSZH_PIN)
    assert stamp_precedes_pin(None, LSZH_PIN) and stamp_precedes_pin("", LSZH_PIN)


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
