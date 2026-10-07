# SPDX-License-Identifier: GPL-3.0-only
"""Per-leg off-block events from adsb.lol's globe history, and the runner that
fetches the days and writes their tables.

The leg and its events are those of §5 of
``reports/preregistered-2026-10-04.md`` as
``reports/preregistered-2026-10-04-adsb-leg.md`` amends them, with the
operational readings fixed before any departure was matched. The logic is the
extraction the pilot ran and the scale-up's runner ran, unchanged: the same
day's release gives the same table, byte for byte (``tests/test_adsb.py`` and
the reproduction recorded in ``reports/fusion.md``).

Every position is kept. A leg starts at the latest of (i) the first position
in the circle after the previous airborne position in the circle, or the first
position of the day's file; (ii) the first position in the circle after a gap
of 20 minutes or more with no position; (iii) the last position of a run of
ground positions in the circle below 1 kt lasting 20 minutes or more. A run is
a sequence of consecutive positions each meeting its condition; its length is
the time from its first position to its last.

Readings:

- *Ground*: altitude element ``"ground"``; *airborne*: a number; a null
  altitude is neither but still counts as a position.
- *Take-off*: a ground position in the circle whose next ground-or-airborne
  position is airborne and climbing: within 300 s, before any further ground
  position, some airborne position at least 300 ft above that first one.
- The leg's ground positions are those inside the circle from the leg's start
  to the take-off position inclusive; E1, E2, E3, the count, the gaps and the
  stationary time are read from them.
- A run that ends inside the leg is read whole.
- E3 and the stationary time are missing when E2 is; a null ground speed is
  neither below 1 kt nor 5 kt or more.

**The runner.** For each day: the ``prod`` release's metadata
(``adsblol/globe_history_<YYYY>``, tag ``v<YYYY.MM.DD>-planes-readsb-prod-0``,
looked up once in the next year's repository on a 404, where 2025-12-31 was
published), every part fetched or reused from disk at its size, each part's
size and, where the metadata carries one, its sha256 checked; the parts streamed
as one tar archive through a pool of workers started by ``forkserver`` (a forked
worker would inherit the fetch thread's curl pipes and hold them open, so the
next day's fetch waited for the extraction); the table written with the ODbL
notice in its metadata; one JSON line per day to the log. It reads no challenge
data.

The tables are a Derivative Database of adsb.lol's data and carry its licence.
Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0.
"""

from __future__ import annotations

import collections
import datetime as dt
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import subprocess
import tarfile
import threading
import time
from pathlib import Path

EARTH_KM = 6371.0
KM_PER_DEG = math.pi / 180 * EARTH_KM

GAP_S = 20 * 60  # the leg amendment's §2 (ii)
LONG_STILL_S = 20 * 60  # the leg amendment's §2 (iii)
STILL_KT = 1.0  # §5: below 1 kt
MOVING_KT = 5.0  # §5: E2, 5 kt or more
E3_RUN_S = 120.0  # §5: E3, a run of at least 120 s
STATIONARY_RUN_S = 60.0  # §5: stationary runs of at least 60 s
CLIMB_WINDOW_S = 300.0  # reading: climbing confirmed within 300 s
CLIMB_FT = 300  # reading: by at least 300 ft

# §5: legs ending in a take-off within 8 km of an airport's reference point.
RADIUS_KM = 8.0

# OurAirports' reference points (``latitude_deg``, ``longitude_deg`` of
# ``airports.csv``), as the pilot tabulated and checked them; the eleventh,
# LTAI, is extracted and never matched. ``tests/test_adsb.py`` checks them
# against the cached table where it is on hand.
REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    "EDDF": (50.026706, 8.55835),
    "EDDM": (48.353802, 11.7861),
    "EGLL": (51.470748, -0.459909),
    "EHAM": (52.308601, 4.76389),
    "LEBL": (41.2971, 2.07846),
    "LEMD": (40.493407, -3.572249),
    "LFPG": (49.00896, 2.554117),
    "LIRF": (41.804532, 12.251998),
    "LSZH": (47.458056, 8.548056),
    "LTAI": (36.898701, 30.800501),
    "LTFM": (41.274874, 28.732136),
}

NOTICE = (
    "Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0"
)

# What each table's metadata says it was built under, written as the pilot
# wrote it so that a table rebuilt here is the same bytes.
DEFINITIONS = (
    "preregistration_2026-10-04_board.md section 5 (568299d1...), "
    "amended by preregistration_2026-10-04_adsb_leg.md (6cdc0f76...)"
)

API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"

REF: dict[str, tuple[float, float]] = {}
BOXES: dict[str, tuple[float, float, float, float]] = {}
_RADIUS = [RADIUS_KM]


# --------------------------------------------------------------------------
# the events


def init_worker(ref: dict[str, tuple[float, float]], radius_km: float) -> None:
    """Set the reference points and their bounding boxes in this process."""
    _RADIUS[0] = radius_km
    REF.clear()
    BOXES.clear()
    REF.update(ref)
    for a, (lat, lon) in ref.items():
        dlat = radius_km / KM_PER_DEG * 1.01
        dlon = radius_km / (KM_PER_DEG * math.cos(math.radians(lat))) * 1.01
        BOXES[a] = (lat - dlat, lat + dlat, lon - dlon, lon + dlon)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def runs(idx_ok):
    """Maximal runs of consecutive positions meeting a condition: (first, last)."""
    out, start = [], None
    for i, ok in enumerate(idx_ok):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(idx_ok) - 1))
    return out


def legs_at(trace, t, inc, airport, meta):
    """Every leg ending in a take-off inside the circle ``inc`` marks."""
    n = len(trace)
    alt = [p[3] for p in trace]
    ground = [a == "ground" for a in alt]
    air = [isinstance(a, (int, float)) and not isinstance(a, bool) for a in alt]
    gs = [p[4] for p in trace]
    still_in = [
        ground[i] and inc[i] and gs[i] is not None and gs[i] < STILL_KT for i in range(n)
    ]
    still_runs = runs(still_in)
    long_still_ends = [e for s, e in still_runs if t[e] - t[s] >= LONG_STILL_S]
    gap_idx = [i for i in range(1, n) if t[i] - t[i - 1] >= GAP_S]

    out = []
    last_air_in = -1  # last airborne position in the circle seen so far
    for k in range(n):
        if air[k] and inc[k]:
            last_air_in = k
        if not (ground[k] and inc[k]):
            continue
        j = k + 1
        while j < n and not (ground[j] or air[j]):
            j += 1
        if j >= n or not air[j]:
            continue
        climb = False
        m = j + 1
        while m < n and t[m] - t[j] <= CLIMB_WINDOW_S and not ground[m]:
            if air[m] and alt[m] >= alt[j] + CLIMB_FT:
                climb = True
                break
            m += 1
        if not climb:
            continue
        if last_air_in >= 0:
            s1 = next(i for i in range(last_air_in + 1, k + 1) if inc[i])
        else:
            s1 = 0
        s2 = -1
        gaps_before = [g for g in gap_idx if g <= k]
        if gaps_before:
            s2 = next(i for i in range(gaps_before[-1], k + 1) if inc[i])
        s3 = max((e for e in long_still_ends if e <= k), default=-1)
        s = max(s1, s2, s3)
        rule = "+".join(r for r, v in (("i", s1), ("ii", s2), ("iii", s3)) if v == s)

        g_idx = [i for i in range(s, k + 1) if ground[i] and inc[i]]
        e1 = t[g_idx[0]]
        e2_i = next((i for i in g_idx if gs[i] is not None and gs[i] >= MOVING_KT), None)
        e2 = t[e2_i] if e2_i is not None else None
        e3 = None
        stationary = None
        if e2_i is not None:
            ends = [e for a, e in still_runs if s <= e < e2_i and t[e] - t[a] >= E3_RUN_S]
            e3 = t[ends[-1]] if ends else None
            stationary = sum(
                t[e] - t[a]
                for a, e in still_runs
                if a > e2_i and e < k and t[e] - t[a] >= STATIONARY_RUN_S
            )
        gaps = [t[b] - t[a] for a, b in zip(g_idx, g_idx[1:])]
        flights = collections.Counter()
        for i in range(s, k + 1):
            det = trace[i][8]
            if det and det.get("flight"):
                f = det["flight"].replace(" ", "")
                if f:
                    flights[f] += 1
        out.append(
            {
                "airport": airport,
                "address": meta["icao"],
                "registration": meta.get("r"),
                "type": meta.get("t"),
                "takeoff": t[k],
                "first_airborne": t[j],
                "leg_first": t[s],
                "leg_last": t[k],
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "n_ground": len(g_idx),
                "max_gap_s": max(gaps) if gaps else 0.0,
                "stationary_s": stationary,
                "flight": flights.most_common(1)[0][0] if flights else None,
                "start_rule": rule,
                "n_positions": k - s + 1,
            }
        )
    return out


def process_trace(raw: bytes):
    """One gzip trace member: its legs, and whether it passed the bounding test."""
    d = json.loads(gzip.decompress(raw))
    trace = d["trace"]
    if not trace:
        return [], False
    lat = [p[1] for p in trace]
    lon = [p[2] for p in trace]
    la0, la1, lo0, lo1 = min(lat), max(lat), min(lon), max(lon)
    cands = [
        a
        for a, (b0, b1, c0, c1) in BOXES.items()
        if la1 >= b0 and la0 <= b1 and lo1 >= c0 and lo0 <= c1
    ]
    if not cands:
        return [], False
    t0 = d["timestamp"]
    t = [t0 + p[0] for p in trace]
    out = []
    for a in cands:
        rlat, rlon = REF[a]
        inc = [haversine_km(p[1], p[2], rlat, rlon) < _RADIUS[0] for p in trace]
        if any(inc):
            out.extend(legs_at(trace, t, inc, a, d))
    return out, True


def process_group(members):
    """One ``traces/<xx>/`` directory: legs, counts, this task's CPU, the worker's peak RSS."""
    cpu0 = time.process_time()
    legs, n_traces, n_box = [], 0, 0
    for name, raw in members:
        n_traces += 1
        try:
            got, boxed = process_trace(raw)
        except Exception as exc:  # a malformed member is reported, not skipped
            raise RuntimeError(f"{name}: {exc!r}") from exc
        n_box += boxed
        legs.extend(got)
    cpu = time.process_time() - cpu0
    return legs, n_traces, n_box, cpu, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


SCHEMA_FIELDS = [
    ("day", "string"),
    ("airport", "string"),
    ("address", "string"),
    ("registration", "string"),
    ("type", "string"),
    ("takeoff", "ts"),
    ("first_airborne", "ts"),
    ("leg_first", "ts"),
    ("leg_last", "ts"),
    ("e1", "ts"),
    ("e2", "ts"),
    ("e3", "ts"),
    ("n_ground", "int32"),
    ("max_gap_s", "float64"),
    ("stationary_s", "float64"),
    ("flight", "string"),
    ("start_rule", "string"),
    ("n_positions", "int32"),
]


def write_parquet(rows, path, day, release):
    """The day's legs, sorted, with the ODbL notice in the file's metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted(rows, key=lambda r: (r["airport"], r["takeoff"], r["address"]))
    ts = pa.timestamp("us", tz="UTC")
    arrays, fields = [], []
    for name, kind in SCHEMA_FIELDS:
        vals = [r[name] for r in rows]
        if kind == "ts":
            arr = pa.array(
                [None if v is None else round(v * 1e6) for v in vals], pa.int64()
            ).cast(ts)
        else:
            arr = pa.array(vals, getattr(pa, "string" if kind == "string" else kind)())
        arrays.append(arr)
        fields.append(pa.field(name, arr.type))
    meta = {
        "licence": NOTICE,
        "source": f"adsblol/globe_history_{day[:4]} release {release}",
        "definitions": DEFINITIONS,
    }
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=meta))
    pq.write_table(table, path, compression="zstd")
    return len(rows)


# --------------------------------------------------------------------------
# the runner


def tag_of(day: str) -> str:
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def curl(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["curl", "-sS", "-L", "--retry", "5", "--retry-delay", "5", *args],
        capture_output=True,
        text=True,
    )


def release_metadata(day: str, meta_dir: Path) -> dict:
    """The release's metadata, cached in ``meta_dir``; waits out GitHub's rate limit."""
    path = meta_dir / f"{day}.json"
    if path.exists():
        return json.loads(path.read_text())
    years = [int(day[:4]), int(day[:4]) + 1]
    url = API.format(year=years.pop(0), tag=tag_of(day))
    headers = ["-H", "Accept: application/vnd.github+json"]
    if os.environ.get("GITHUB_TOKEN"):
        headers += ["-H", f"Authorization: Bearer {os.environ['GITHUB_TOKEN']}"]
    code = ""
    for attempt in range(6):
        hdr = meta_dir / f"{day}.headers"
        tmp = meta_dir / f"{day}.json.part"
        r = curl([*headers, "-D", str(hdr), "-o", str(tmp), "-w", "%{http_code}", url])
        code = r.stdout.strip()[-3:]
        if code == "200":
            tmp.rename(path)
            hdr.unlink(missing_ok=True)
            return json.loads(path.read_text())
        text = hdr.read_text() if hdr.exists() else ""
        fields = {
            k.strip().lower(): v.strip()
            for k, _, v in (ln.partition(":") for ln in text.splitlines())
            if v
        }
        if code in ("403", "429") and fields.get("x-ratelimit-remaining") == "0":
            wait = max(
                5, int(fields.get("x-ratelimit-reset", "0")) - int(time.time()) + 5
            )
            print(f"{day}: GitHub API rate limit, waiting {wait} s", flush=True)
            time.sleep(wait)
            continue
        if code == "404":
            if years:
                url = API.format(year=years.pop(0), tag=tag_of(day))
                continue
            raise RuntimeError(f"{day}: no release {tag_of(day)}")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{day}: release metadata not fetched (last HTTP {code})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch_day(day: str, parts_dir: Path, meta_dir: Path) -> dict:
    """Every part of the day's release on disk, size-checked; what was fetched, reused."""
    meta = release_metadata(day, meta_dir)
    assets = sorted(meta["assets"], key=lambda a: a["name"])
    if not assets:
        raise RuntimeError(f"{day}: release has no assets")
    d = parts_dir / day
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    fetched, reused, nbytes, digests = [], [], 0, {}
    for a in assets:
        path = d / a["name"]
        if path.exists() and path.stat().st_size == a["size"]:
            reused.append(a["name"])
        else:
            for attempt in range(3):
                tmp = path.with_suffix(path.suffix + ".part")
                r = curl(["-f", "-o", str(tmp), a["browser_download_url"]])
                if r.returncode == 0 and tmp.stat().st_size == a["size"]:
                    tmp.rename(path)
                    break
                tmp.unlink(missing_ok=True)
                time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"{day}: {a['name']} not fetched at {a['size']} B")
            fetched.append(a["name"])
        size = path.stat().st_size
        if size != a["size"]:
            raise RuntimeError(
                f"{day}: {a['name']} is {size} B, metadata says {a['size']} B"
            )
        nbytes += size
        want = (a.get("digest") or "").removeprefix("sha256:")
        if want:
            got = sha256_file(path)
            if got != want:
                raise RuntimeError(
                    f"{day}: {a['name']} sha256 {got}, metadata says {want}"
                )
            digests[a["name"]] = "equal"
        else:
            digests[a["name"]] = "none in metadata"
    return {
        "release": tag_of(day),
        "parts": [a["name"] for a in assets],
        "bytes": nbytes,
        "fetched": fetched,
        "reused": reused,
        "sha256_check": digests,
        "fetch_wall_s": round(time.monotonic() - t0, 3),
    }


class ChainReader:
    """The parts read one after another, as one file object (what ``cat`` gives)."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.fh = None

    def read(self, n=-1):
        out = bytearray()
        while n < 0 or len(out) < n:
            if self.fh is None:
                if not self.paths:
                    break
                self.fh = open(self.paths.pop(0), "rb")
            chunk = self.fh.read(-1 if n < 0 else n - len(out))
            if not chunk:
                self.fh.close()
                self.fh = None
                continue
            out += chunk
        return bytes(out)

    def close(self):
        if self.fh is not None:
            self.fh.close()


def stream_groups(parts):
    """Yield the trace members of each ``traces/<xx>/`` directory, in archive order."""
    reader = ChainReader(parts)
    tf = tarfile.open(fileobj=reader, mode="r|")
    current, batch = None, []
    for m in tf:
        if not m.isfile() or "/trace_full_" not in m.name:
            continue
        group = m.name.rsplit("/", 1)[0]
        if group != current and batch:
            yield batch
            batch = []
        current = group
        batch.append((m.name, tf.extractfile(m).read()))
    if batch:
        yield batch
    tf.close()
    reader.close()


def extract_day(day, parts, out_path, release, ref, radius, workers, inflight):
    """The day's parts through ``workers`` processes; the table written; its record."""
    t0 = time.monotonic()
    main_cpu0 = time.thread_time()
    rows, stats = [], collections.Counter()
    worker_cpu, worker_rss = [0.0], [0]
    slots = threading.BoundedSemaphore(inflight)
    errors = []

    def done(res):
        legs, nt, nb, cpu, rss = res
        rows.extend(legs)
        stats["traces"] += nt
        stats["near_an_airport"] += nb
        worker_cpu[0] += cpu
        worker_rss[0] = max(worker_rss[0], rss)
        slots.release()

    def fail(exc):
        errors.append(exc)
        slots.release()

    ctx = mp.get_context("forkserver")
    with ctx.Pool(workers, initializer=init_worker, initargs=(ref, radius)) as pool:
        for members in stream_groups(parts):
            slots.acquire()
            if errors:
                break
            stats["groups"] += 1
            pool.apply_async(
                process_group, (members,), callback=done, error_callback=fail
            )
        pool.close()
        pool.join()
    if errors:
        raise RuntimeError(f"{day}: worker failed: {errors[0]!r}")
    for r in rows:
        r["day"] = day
    tmp = out_path.with_suffix(".parquet.part")
    n = write_parquet(rows, tmp, day, release)
    tmp.rename(out_path)
    main_cpu = time.thread_time() - main_cpu0
    return {
        "groups": stats["groups"],
        "traces": stats["traces"],
        "near_an_airport": stats["near_an_airport"],
        "legs": n,
        "legs_per_airport": dict(
            sorted(collections.Counter(r["airport"] for r in rows).items())
        ),
        "extract_wall_s": round(time.monotonic() - t0, 3),
        "extract_cpu_s": round(worker_cpu[0] + main_cpu, 3),
        "extract_cpu_workers_s": round(worker_cpu[0], 3),
        "extract_cpu_main_thread_s": round(main_cpu, 3),
        "peak_rss_worker_kb": worker_rss[0],
        "peak_rss_main_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "events_bytes": out_path.stat().st_size,
        "events_sha256": sha256_file(out_path),
    }


def run_days(
    days,
    out_dir: Path,
    parts_dir: Path,
    *,
    workers: int,
    downloads: int,
    inflight: int,
    keep_parts: bool,
    ref=None,
    radius: float | None = None,
) -> list[str]:
    """Fetch and extract each day, the next fetches running while one extracts.

    Writes ``<out_dir>/<day>.parquet``, caches release metadata under
    ``<out_dir>/meta/`` and appends one line per day to
    ``<out_dir>/runner_log.jsonl``. A day whose table is already there is
    still run when asked for, and a failed day is logged and the others go
    on. Returns the failed days.
    """
    from concurrent.futures import ThreadPoolExecutor

    ref = dict(REFERENCE_POINTS if ref is None else ref)
    radius = RADIUS_KM if radius is None else float(radius)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runner_log.jsonl"
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, downloads)) as pool:
        queue = collections.deque()
        todo = iter(days)

        def fill():
            while len(queue) < max(1, downloads):
                day = next(todo, None)
                if day is None:
                    return
                queue.append(
                    (day, pool.submit(fetch_day, day, parts_dir, out_dir / "meta"))
                )

        fill()
        while queue:
            day, fut = queue.popleft()
            fill()  # the next fetches run while this day is extracted
            rec = {
                "day": day,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "workers": workers,
            }
            try:
                rec.update(fut.result())
                parts = [parts_dir / day / p for p in rec["parts"]]
                out_path = out_dir / f"{day}.parquet"
                rec.update(
                    extract_day(
                        day,
                        parts,
                        out_path,
                        rec["release"],
                        ref,
                        radius,
                        workers,
                        inflight,
                    )
                )
                deleted = []
                if not keep_parts:
                    for name in rec["fetched"]:
                        (parts_dir / day / name).unlink()
                        deleted.append(name)
                    if not any((parts_dir / day).iterdir()):
                        (parts_dir / day).rmdir()
                rec["parts_deleted"] = deleted
                rec["status"] = "ok"
            except Exception as exc:  # one failed day is logged; the others go on
                rec["status"] = f"failed: {exc}"
                failed.append(day)
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(
                json.dumps(
                    {
                        k: rec.get(k)
                        for k in (
                            "day",
                            "status",
                            "bytes",
                            "fetch_wall_s",
                            "legs",
                            "extract_wall_s",
                            "extract_cpu_s",
                            "events_bytes",
                        )
                    }
                ),
                flush=True,
            )
    return failed


def has_notice(path: Path) -> bool:
    """Whether a table carries the ODbL notice in its metadata."""
    import pyarrow.parquet as pq

    md = pq.read_schema(path).metadata or {}
    return md.get(b"licence", b"").decode() == NOTICE


__all__ = [
    "DEFINITIONS",
    "NOTICE",
    "RADIUS_KM",
    "REFERENCE_POINTS",
    "extract_day",
    "fetch_day",
    "has_notice",
    "legs_at",
    "process_group",
    "process_trace",
    "run_days",
    "tag_of",
    "write_parquet",
]
