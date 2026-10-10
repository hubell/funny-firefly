# SPDX-License-Identifier: GPL-3.0-only
"""OPDI's flight events matched to the challenge's departures.

The definitions are §2 of ``reports/preregistered-2026-10-04-opdi.md``, with
``reports/preregistered-2026-10-04-opdi-tie.md``'s tie for event flights whose
id is in no flight list. OPDI v0.0.2 (EUROCONTROL's Open Performance Data
Initiative) publishes the flight events as ten-day files and the flight lists
as monthly files; no data licence is stated, and the entry uses them on the
organizers' confirmation. No OPDI file and no table derived from one is
tracked here or deposited: ``scripts/fetch_opdi.py`` fetches the published
files again and rebuilds every table from them.

Four steps, each reading the previous one's output only:

1. **The reduction** of each event file (``reduce_file``): every first
   crossing of FL50; every ``first_seen`` event with no altitude within 4 km
   of one of the ten airports' reference points (O0's candidates); and every
   ``exit-parking_position``, ``entry-taxiway``, ``exit-apron`` and
   ``entry-runway`` event whose ``osm_airport`` is one of the ten, with
   ``osm_ref``, ``opdi_time_in_use_s``, ``osn_icao24`` and the callsign. Times
   are read as UTC: a timestamp without a zone is UTC, a string is parsed as
   UTC. Ids are the same 64 bits read as int64, whether a file stores uint64
   or int64.
2. **The tie** (``tie_map``): an event flight whose id is in no fetched list
   is tied to the list flight with the same icao24 and callsign whose
   ``first_seen`` is nearest its anchor, within 3 hours; equal nearest, or two
   event flights on one list flight, ties nothing there.
3. **The month table** (``month_table``): every flight of the month's list at
   the ten, and for a departure its events at the airport before its first
   FL50 crossing: O0 (its first ``first_seen`` surface position), O1 (last
   parking exit), O2 (first taxiway entry), O3 (last apron exit), R (last
   runway entry), each with ``osm_ref``.
4. **The match** (``match_month``): a departure matches the OPDI flight at
   its airport whose callsign equals ``CALLSIGN_flt`` or ``FLIGHT_mvt`` (spaces
   removed), whose ``first_seen`` is no later than 600 s after the take-off
   and ``last_seen`` no earlier than it, the nearest ``first_seen`` if several;
   it is covered where O0, O1 or O2 lies at least 120 s before the take-off;
   its inbound is the latest OPDI flight of the same icao24 arriving at the
   airport within the 12 hours before, linked to the challenge arrival with
   the same callsign within 30 minutes of its ``last_seen``.

The pipeline is the one the matched tables of 4 October were built with,
unchanged, with one difference: ``opdi_id`` and ``inbound_opdi_id`` are kept
as 64-bit integers (nullable), where those tables stored them as float64,
which rounds a 64-bit id. Nothing downstream joins on either.
"""

from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path

import numpy as np
import pandas as pd

from .adsb import REFERENCE_POINTS

BASE_URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"

TEN: tuple[str, ...] = (
    "EDDF",
    "EDDM",
    "EGLL",
    "EHAM",
    "LEBL",
    "LEMD",
    "LFPG",
    "LIRF",
    "LSZH",
    "LTFM",
)

O0_RADIUS_M = 4000.0  # §2: O0 within 4 km of the reference point
EARTH_R = 6371008.8  # metres, the mean radius
SURFACE = ("exit-parking_position", "entry-taxiway", "exit-apron", "entry-runway")

MATCH_AFTER_S = 600  # §2: first_seen no later than 600 s after the take-off
COVER_S = 120  # §2: an event at least 120 s before the take-off
INBOUND_S = 12 * 3600  # §2: the inbound within 12 hours before
LINK_S = 30 * 60  # §2: linked within 30 minutes of its last_seen
TIE_WINDOW_S = 3 * 3600  # the tie amendment's §1: within 3 hours

# The ten-day files are a grid stepped from this day in both directions.
GRID_ANCHOR = dt.date(2025, 1, 5)
GRID_STEP_DAYS = 10

# The months whose departures are matched: the training year and the ranking
# months, which §2 of the four-month pre-registration of 8 October 2026 reads
# from the ranking file (``taxiout.challenge.ranking_months``) rather than
# naming January and July 2026 (``match_months``). The lists read are those
# and the month before each run of consecutive months (December 2024, and
# June 2026 when the ranking months are January and July 2026), so that a
# flight or an inbound arrival crossing into a run's first month is found
# (``list_months``).
TRAINING_MONTHS: tuple[str, ...] = tuple(f"2025{m:02d}" for m in range(1, 13))

EVENT_COLS = [
    "O0_time",
    "O0_latitude",
    "O0_longitude",
    "O1_time",
    "O1_osm_ref",
    "O1_time_in_use_s",
    "O2_time",
    "O2_osm_ref",
    "O3_time",
    "O3_osm_ref",
    "R_time",
    "R_osm_ref",
    "fl50",
    "fl50_missing",
]
MOVE_COLS = [
    "MVT_ID_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "STAND_mvt",
    "CALLSIGN_flt",
    "FLIGHT_mvt",
]
LIST_COLS = ["id", "icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"]


# --------------------------------------------------------------------------
# the files


def _month_bounds(ym: str) -> tuple[dt.date, dt.date]:
    y, m = int(ym[:4]), int(ym[4:])
    start = dt.date(y, m, 1)
    end = dt.date(y + (m == 12), m % 12 + 1, 1)
    return start, end


def match_months(ranking_months) -> tuple[str, ...]:
    """The training year and the ranking months, as ``YYYYMM``, in calendar order.

    ``ranking_months`` as ``taxiout.challenge.ranking_months`` reads them
    (``YYYY-MM``); ``YYYYMM`` is taken too.
    """
    rank = {str(m).replace("-", "") for m in ranking_months}
    return tuple(sorted(set(TRAINING_MONTHS) | rank))


def list_months(match) -> tuple[str, ...]:
    """The match months and the month before each run of consecutive ones, in order."""
    months = sorted(set(match))
    out = set(months)
    prev = None
    for ym in months:
        p = pd.Period(f"{ym[:4]}-{ym[4:]}", "M")
        if prev is None or p != prev + 1:
            out.add((p - 1).strftime("%Y%m"))
        prev = p
    return tuple(sorted(out))


def event_files(months) -> list[str]:
    """The ten-day event files whose span overlaps any of ``months`` (``match_months``)."""
    spans = [_month_bounds(m) for m in months]
    lo = min(s for s, _ in spans)
    hi = max(e for _, e in spans)
    k = (lo - GRID_ANCHOR).days // GRID_STEP_DAYS - 1
    out = []
    while True:
        a = GRID_ANCHOR + dt.timedelta(days=GRID_STEP_DAYS * k)
        b = a + dt.timedelta(days=GRID_STEP_DAYS)
        if a >= hi:
            break
        if any(a < e and b > s for s, e in spans):
            out.append(f"flight_events_{a:%Y%m%d}_{b:%Y%m%d}.parquet")
        k += 1
    return out


def list_files(months) -> list[str]:
    """The monthly flight lists of ``months`` (``list_months``)."""
    return [f"flight_list_{m}.parquet" for m in months]


def url_of(name: str) -> str:
    kind = "flight_events" if name.startswith("flight_events_") else "flight_list"
    return f"{BASE_URL}/{kind}/{name}"


# --------------------------------------------------------------------------
# step 1, the reduction


def _out_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("flight_id", pa.int64()),
            ("type", pa.string()),
            ("event_time", pa.timestamp("us", tz="UTC")),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
            ("altitude", pa.float64()),
            ("airport", pa.string()),  # osm_airport, or for O0 the airport within 4 km
            ("osm_ref", pa.string()),
            ("time_in_use_s", pa.float64()),
            ("osn_icao24", pa.string()),  # surface events only, from info
            ("osn_callsign", pa.string()),  # surface events only, spaces removed
        ]
    )


def utc_times(col):
    """Event or list times as UTC timestamps, whatever the file stores."""
    import pyarrow as pa
    import pyarrow.compute as pc

    t = col.type
    if pa.types.is_timestamp(t):
        if t.tz is None:
            return pc.assume_timezone(pc.cast(col, pa.timestamp("us")), "UTC")
        return pc.cast(col, pa.timestamp("us", tz="UTC"))
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        s = pd.to_datetime(pd.Series(col.to_pandas()), utc=True, format="ISO8601")
        return pa.array(s, type=pa.timestamp("us", tz="UTC"))
    raise TypeError(f"event_time of type {t}")


def id64(col):
    """The same 64 bits as int64, whether the file stores uint64 or int64."""
    import pyarrow as pa

    a = (
        col.to_numpy(zero_copy_only=False)
        if hasattr(col, "to_numpy")
        else np.asarray(col)
    )
    return pa.array(
        a.view(np.int64) if a.dtype == np.uint64 else a.astype(np.int64), pa.int64()
    )


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def info_field(info, key: str, numeric: bool):
    """One field of the events' JSON ``info`` column, by pattern."""
    import pyarrow as pa
    import pyarrow.compute as pc

    pat = f'"{key}":' + (r"(?P<v>-?[0-9.eE+]+)" if numeric else r'"(?P<v>[^"]*)"')
    s = pc.struct_field(pc.extract_regex(info, pat), [0])
    if numeric:
        return pc.cast(s, pa.float64())
    return s


def reduce_group(t):
    """One row group of an event file, reduced to what §2 reads."""
    import pyarrow as pa
    import pyarrow.compute as pc

    schema = _out_schema()
    typ = t["type"]
    parts = []

    def nulls(n, kind):
        return pa.nulls(n, kind)

    # first-xing-fl50
    x = t.filter(pc.equal(typ, "first-xing-fl50"))
    if x.num_rows:
        n = x.num_rows
        parts.append(
            pa.table(
                {
                    "flight_id": id64(x["flight_id"]),
                    "type": x["type"],
                    "event_time": utc_times(x["event_time"]),
                    "latitude": x["latitude"],
                    "longitude": x["longitude"],
                    "altitude": x["altitude"],
                    "airport": nulls(n, pa.string()),
                    "osm_ref": nulls(n, pa.string()),
                    "time_in_use_s": nulls(n, pa.float64()),
                    "osn_icao24": nulls(n, pa.string()),
                    "osn_callsign": nulls(n, pa.string()),
                },
                schema=schema,
            )
        )

    # first_seen, a surface position, within 4 km of one of the ten
    m = pc.and_(pc.equal(typ, "first_seen"), pc.is_null(t["altitude"], nan_is_null=True))
    x = t.filter(m)
    if x.num_rows:
        lat = x["latitude"].to_numpy(zero_copy_only=False)
        lon = x["longitude"].to_numpy(zero_copy_only=False)
        ap = np.full(len(lat), None, dtype=object)
        for icao in TEN:
            rl, rn = REFERENCE_POINTS[icao]
            d = haversine_m(lat, lon, rl, rn)
            # the 4 km circles do not overlap
            ap[(d <= O0_RADIUS_M) & (ap == None)] = icao  # noqa: E711
        keep = ap != None  # noqa: E711
        if keep.any():
            x = x.filter(pa.array(keep))
            n = x.num_rows
            parts.append(
                pa.table(
                    {
                        "flight_id": id64(x["flight_id"]),
                        "type": x["type"],
                        "event_time": utc_times(x["event_time"]),
                        "latitude": x["latitude"],
                        "longitude": x["longitude"],
                        "altitude": x["altitude"],
                        "airport": pa.array(ap[keep].tolist(), pa.string()),
                        "osm_ref": nulls(n, pa.string()),
                        "time_in_use_s": nulls(n, pa.float64()),
                        "osn_icao24": nulls(n, pa.string()),
                        "osn_callsign": nulls(n, pa.string()),
                    },
                    schema=schema,
                )
            )

    # surface events at the ten
    x = t.filter(pc.is_in(typ, value_set=pa.array(SURFACE)))
    if x.num_rows:
        ap = info_field(x["info"].combine_chunks(), "osm_airport", False)
        keep = pc.fill_null(pc.is_in(ap, value_set=pa.array(TEN)), False)
        x = x.filter(keep)
        ap = ap.filter(keep)
        if x.num_rows:
            info = x["info"].combine_chunks()
            parts.append(
                pa.table(
                    {
                        "flight_id": id64(x["flight_id"]),
                        "type": x["type"],
                        "event_time": utc_times(x["event_time"]),
                        "latitude": x["latitude"],
                        "longitude": x["longitude"],
                        "altitude": x["altitude"],
                        "airport": ap,
                        "osm_ref": info_field(info, "osm_ref", False),
                        "time_in_use_s": info_field(info, "opdi_time_in_use_s", True),
                        "osn_icao24": info_field(info, "osn_icao24", False),
                        "osn_callsign": pc.replace_substring(
                            info_field(info, "osn_flight_id", False), " ", ""
                        ),
                    },
                    schema=schema,
                )
            )
    return pa.concat_tables(parts) if parts else schema.empty_table()


def reduce_file(src: Path, out: Path) -> dict:
    """One event file reduced, row group by row group, written to ``out``."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    f = pq.ParquetFile(src)
    cols = [
        "flight_id",
        "type",
        "event_time",
        "latitude",
        "longitude",
        "altitude",
        "info",
    ]
    rows_in = 0
    parts = []
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=cols)
        rows_in += t.num_rows
        parts.append(reduce_group(t))
        del t
    r = pa.concat_tables(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(r, tmp, compression="zstd")
    tmp.rename(out)
    vc = {d["values"]: d["counts"] for d in pc.value_counts(r["type"]).to_pylist()}
    return {"file": src.name, "rows_in": rows_in, "rows_out": r.num_rows, "types": vc}


# --------------------------------------------------------------------------
# step 2, the tie


def list_rows(raw: Path, months):
    """Every fetched list's ids (all rows), and its rows at the ten."""
    import pyarrow.parquet as pq

    all_ids, rows = [], []
    for name in list_files(months):
        t = pq.read_table(raw / name, columns=LIST_COLS)
        ids = id64(t["id"]).to_numpy()
        all_ids.append(ids)
        d = pd.DataFrame(
            {
                "id": ids,
                "icao24": t["icao24"].to_pandas(),
                "callsign": t["flt_id"]
                .to_pandas()
                .astype("string")
                .str.replace(" ", "", regex=False),
                "adep": t["adep"].to_pandas(),
                "ades": t["ades"].to_pandas(),
                "first_seen": utc_times(t["first_seen"]).to_pandas(),
                "list_month": name[-14:-8],
            }
        )
        rows.append(d[d["adep"].isin(TEN) | d["ades"].isin(TEN)])
    return np.unique(np.concatenate(all_ids)), pd.concat(rows, ignore_index=True)


def event_flights(reduced: pd.DataFrame) -> pd.DataFrame:
    """Per event flight: its keys (icao24, callsign) and its anchor."""
    s = reduced[reduced["type"].isin(SURFACE) & reduced["osn_icao24"].notna()]
    s = s.sort_values(["flight_id", "event_time"], kind="stable")
    first_t = s.groupby("flight_id")["event_time"].min().rename("first_surface")
    pair = (
        s.groupby(["flight_id", "osn_icao24", "osn_callsign"], dropna=False)
        .agg(n=("event_time", "size"), t0=("event_time", "min"))
        .reset_index()
        .sort_values(
            ["flight_id", "n", "t0"], ascending=[True, False, True], kind="stable"
        )
        .drop_duplicates("flight_id")
        .set_index("flight_id")[["osn_icao24", "osn_callsign"]]
    )
    fl50 = (
        reduced[reduced["type"] == "first-xing-fl50"]
        .groupby("flight_id")["event_time"]
        .min()
        .rename("fl50")
    )
    e = pair.join(first_t).join(fl50, how="left")
    e["anchor"] = e["fl50"].fillna(e["first_surface"])
    return e.reset_index()


def tie(e: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """flight_id to its tied list id (or none), with the gap to the nearest."""
    j = e[["flight_id", "osn_icao24", "osn_callsign", "anchor"]].merge(
        rows[["id", "icao24", "callsign", "first_seen"]],
        left_on=["osn_icao24", "osn_callsign"],
        right_on=["icao24", "callsign"],
    )
    j["gap"] = (j["first_seen"] - j["anchor"]).abs().dt.total_seconds()
    j = j[j["gap"] <= TIE_WINDOW_S].drop_duplicates(["flight_id", "id"])
    j = j.sort_values(["flight_id", "gap"], kind="stable")
    n_near = j.groupby(["flight_id", "gap"]).size().groupby(level=0).first()
    best = j.drop_duplicates("flight_id").set_index("flight_id")[["id", "gap"]]
    best["id"] = best["id"].astype("Int64")  # exact: float64 would round 64-bit ids
    best["equal_nearest"] = n_near.reindex(best.index).fillna(1).astype(int) > 1
    best.loc[best["equal_nearest"], "id"] = pd.NA
    return best.rename(columns={"id": "tied_id"})


def tie_map(raw: Path, reduced_files: list[Path], months) -> tuple[pd.DataFrame, dict]:
    """The tie applied: event flights in no list of ``months``, to their tied list ids."""
    all_ids, rows = list_rows(raw, months)
    red = pd.concat([pd.read_parquet(f) for f in reduced_files])
    e = event_flights(red)
    e = e[~np.isin(e["flight_id"].to_numpy(), all_ids)]
    r = tie(e, rows).dropna(subset=["tied_id"])
    r["tied_id"] = r["tied_id"].astype("int64")
    if not r["tied_id"].isin(rows["id"]).all():
        raise RuntimeError("a tied id is not a list id")
    dup = r["tied_id"].duplicated(keep=False)
    m = r[~dup].reset_index()[["flight_id", "tied_id", "gap"]]
    info = {
        "event_flights_in_no_list": int(len(e)),
        "tied": int(len(r)),
        "list_flights_tied_twice": int(dup.sum() // 2),
        "kept": int(len(m)),
    }
    return m, info


# --------------------------------------------------------------------------
# step 3, the month table


def month_table(ym: str, raw: Path, reduced_files: list[Path], tie_map_frame) -> tuple:
    """The month's list flights at the ten, with each departure's events."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    fl = pq.read_table(raw / f"flight_list_{ym}.parquet", columns=LIST_COLS)
    ten = pa.array(TEN)
    m = pc.or_(
        pc.fill_null(pc.is_in(fl["adep"], value_set=ten), False),
        pc.fill_null(pc.is_in(fl["ades"], value_set=ten), False),
    )
    fl = fl.filter(m)
    f = pd.DataFrame(
        {
            "id": id64(fl["id"]).to_numpy(),
            "icao24": fl["icao24"].to_pandas(),
            "callsign": fl["flt_id"]
            .to_pandas()
            .astype("string")
            .str.replace(" ", "", regex=False),
            "adep": fl["adep"].to_pandas(),
            "ades": fl["ades"].to_pandas(),
            "first_seen": utc_times(fl["first_seen"]).to_pandas(),
            "last_seen": utc_times(fl["last_seen"]).to_pandas(),
        }
    )
    f["is_dep"] = f["adep"].isin(TEN)
    f["is_arr"] = f["ades"].isin(TEN)
    dup_ids = int(f["id"].duplicated().sum())

    dep = f.loc[f["is_dep"], ["id", "adep"]]
    tm = tie_map_frame[tie_map_frame["tied_id"].isin(dep["id"])]
    want = np.concatenate([dep["id"].to_numpy(), tm["flight_id"].to_numpy()]).astype(
        np.int64
    )
    dset = ds.dataset(sorted(reduced_files), format="parquet")
    ev = dset.to_table(
        filter=ds.field("flight_id").isin(pa.array(want, pa.int64()))
    ).to_pandas()
    remap = dict(zip(tm["flight_id"], tm["tied_id"]))
    ev["flight_id"] = ev["flight_id"].map(lambda x: remap.get(x, x)).astype(np.int64)
    ev = ev.rename(columns={"flight_id": "id"}).merge(dep, on="id", how="inner")
    f["opdi_tied"] = f["id"].isin(tm["tied_id"])

    fl50 = (
        ev[ev["type"] == "first-xing-fl50"]
        .groupby("id")["event_time"]
        .min()
        .rename("fl50")
    )
    ev = ev.merge(fl50, left_on="id", right_index=True, how="left")
    at = ev[
        (ev["type"] != "first-xing-fl50")
        & (ev["airport"] == ev["adep"])
        & (ev["fl50"].isna() | (ev["event_time"] < ev["fl50"]))
    ]
    at = at.sort_values(["id", "event_time"], kind="stable")

    out = f.copy()
    out = out.merge(fl50, left_on="id", right_index=True, how="left")
    n_first_seen = at[at["type"] == "first_seen"].groupby("id").size()

    def pick(typ, which, cols, prefix):
        g = at[at["type"] == typ].groupby("id")
        r = (g.last() if which == "last" else g.first())[["event_time"] + cols]
        r.columns = [f"{prefix}_time"] + [f"{prefix}_{c}" for c in cols]
        return r

    for r in (
        pick("first_seen", "first", ["latitude", "longitude"], "O0"),
        pick("exit-parking_position", "last", ["osm_ref", "time_in_use_s"], "O1"),
        pick("entry-taxiway", "first", ["osm_ref"], "O2"),
        pick("exit-apron", "last", ["osm_ref"], "O3"),
        pick("entry-runway", "last", ["osm_ref"], "R"),
    ):
        out = out.merge(r, left_on="id", right_index=True, how="left")
    o0_off = int((out["O0_time"].notna() & (out["O0_time"] != out["first_seen"])).sum())
    out["fl50_missing"] = out["is_dep"] & out["fl50"].isna()
    out.loc[~out["is_dep"], "fl50"] = pd.NaT
    d = out[out["is_dep"]]
    info = {
        "list_rows_ten": int(len(out)),
        "dup_ids": dup_ids,
        "dep": int(d.shape[0]),
        "arr": int(out["is_arr"].sum()),
        "fl50_missing": int(d["fl50_missing"].sum()),
        "multi_first_seen": int((n_first_seen > 1).sum()),
        "o0_differs_from_list_first_seen": o0_off,
        "tied": int(d["opdi_tied"].sum()),
        "events": {
            k: int(d[f"{k}_time"].notna().sum()) for k in ("O0", "O1", "O2", "O3", "R")
        },
    }
    return out, info


# --------------------------------------------------------------------------
# step 4, the match


@functools.lru_cache(maxsize=None)
def _ranking_yyyymm(root: Path) -> tuple[str, ...]:
    """The ranking file's months under ``root``, as ``YYYYMM``: the months with arrivals."""
    from . import challenge

    return tuple(m.replace("-", "") for m in challenge.ranking_months(root))


def months_around(ym: str) -> list[str]:
    p = pd.Period(f"{ym[:4]}-{ym[4:]}", "M")
    return [q.strftime("%Y%m") for q in (p - 1, p, p + 1)]


def movements(root: Path, ym: str, phase: str) -> pd.DataFrame:
    """The month's movements of one phase, from the training file or the ranking set."""
    from . import challenge

    if ym.startswith("2025"):
        nxt = (pd.Period(ym[:4] + "-" + ym[4:], "M") + 1).strftime("%Y-%m")
        f = root / f"movements/training_{ym[:4]}-{ym[4:]}-01_{nxt}-01.parquet"
        t = challenge.read_table(f, MOVE_COLS, phase)
    else:
        t = challenge.load_ranking(root, MOVE_COLS, phase)
    d = t.to_pandas()
    for c in ("ADEP_mvt", "ADES_mvt", "STAND_mvt", "CALLSIGN_flt", "FLIGHT_mvt"):
        d[c] = d[c].astype("string")
    d = d[d["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m") == ym]
    return d.reset_index(drop=True)


def keys(d: pd.DataFrame, airport_col: str) -> pd.DataFrame:
    """One row per (movement, callsign key), spaces removed, both fields."""
    parts = []
    for c in ("CALLSIGN_flt", "FLIGHT_mvt"):
        k = d[["MVT_ID_mvt", airport_col, "MVT_TIME_UTC_mvt"]].copy()
        k["callsign"] = d[c].str.replace(" ", "", regex=False)
        parts.append(k[k["callsign"].notna() & (k["callsign"] != "")])
    return pd.concat(parts).drop_duplicates(["MVT_ID_mvt", "callsign"])


def match_month(ym: str, root: Path, opdi: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every departure of month ``ym`` with its OPDI flight, events and inbound.

    ``opdi`` is the month tables of ``months_around(ym)`` that exist,
    concatenated; ``root`` is the challenge data's directory.
    """
    opdi = opdi.drop_duplicates("id")
    fdep = opdi[opdi["is_dep"]]
    farr = opdi[opdi["is_arr"]][["id", "icao24", "callsign", "ades", "last_seen"]]

    dep = movements(root, ym, "DEP")
    k = keys(dep, "ADEP_mvt").merge(
        fdep[["id", "adep", "callsign", "first_seen", "last_seen"]],
        left_on=["ADEP_mvt", "callsign"],
        right_on=["adep", "callsign"],
    )
    t = k["MVT_TIME_UTC_mvt"]
    k = k[
        (k["first_seen"] <= t + pd.Timedelta(seconds=MATCH_AFTER_S))
        & (k["last_seen"] >= t)
    ]
    k = k.drop_duplicates(["MVT_ID_mvt", "id"])
    k["dist"] = (k["first_seen"] - k["MVT_TIME_UTC_mvt"]).abs()
    k = k.sort_values(["MVT_ID_mvt", "dist", "id"], kind="stable")
    ncand = k.groupby("MVT_ID_mvt").size().rename("n_candidates")
    nbest = (
        k.groupby(["MVT_ID_mvt", "dist"])
        .size()
        .groupby(level=0)
        .first()
        .rename("n_nearest")
    )
    best = k.drop_duplicates("MVT_ID_mvt")[["MVT_ID_mvt", "id"]].rename(
        columns={"id": "opdi_id"}
    )
    best["opdi_id"] = best["opdi_id"].astype("Int64")

    out = dep[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]].merge(
        best, on="MVT_ID_mvt", how="left"
    )
    out = out.merge(ncand, left_on="MVT_ID_mvt", right_index=True, how="left")
    out = out.merge(nbest, left_on="MVT_ID_mvt", right_index=True, how="left")
    out["n_candidates"] = out["n_candidates"].fillna(0).astype(int)
    out["matched"] = out["opdi_id"].notna()
    fd = fdep[
        ["id", "icao24", "first_seen", "last_seen", "opdi_tied"] + EVENT_COLS
    ].rename(columns={"id": "opdi_id"})
    fd["opdi_id"] = fd["opdi_id"].astype("Int64")
    out = out.merge(fd, on="opdi_id", how="left")
    out["opdi_tied"] = out["opdi_tied"].astype("boolean").fillna(False).astype(bool)
    lead = out["MVT_TIME_UTC_mvt"] - pd.Timedelta(seconds=COVER_S)
    early = np.zeros(len(out), bool)
    for c in ("O0_time", "O1_time", "O2_time"):
        early |= (out[c].notna() & (out[c] <= lead)).to_numpy()
    out["covered"] = out["matched"].to_numpy() & early

    # the inbound: the latest OPDI arrival of the same icao24 at the airport,
    # within 12 h before first_seen
    m = out[out["matched"]][["MVT_ID_mvt", "ADEP_mvt", "icao24", "first_seen"]]
    a = m.merge(farr, left_on=["icao24", "ADEP_mvt"], right_on=["icao24", "ades"])
    a = a[
        (a["last_seen"] < a["first_seen"])
        & (a["last_seen"] >= a["first_seen"] - pd.Timedelta(seconds=INBOUND_S))
    ]
    a = a.sort_values(
        ["MVT_ID_mvt", "last_seen", "id"], ascending=[True, False, True], kind="stable"
    )
    a = a.drop_duplicates("MVT_ID_mvt")[["MVT_ID_mvt", "id", "callsign", "last_seen"]]
    a.columns = ["MVT_ID_mvt", "inbound_opdi_id", "inbound_callsign", "inbound_last_seen"]
    a["inbound_opdi_id"] = a["inbound_opdi_id"].astype("Int64")
    out = out.merge(a, on="MVT_ID_mvt", how="left")

    # linked to the challenge arrival at the airport, callsign equal, within
    # 30 min of last_seen
    arr = pd.concat(
        [
            movements(root, mm, "ARR")
            for mm in months_around(ym)[:2]
            if mm >= "202501" and (mm.startswith("2025") or mm in _ranking_yyyymm(root))
        ]
    )
    ak = keys(arr, "ADES_mvt")
    q = out[out["inbound_opdi_id"].notna()][
        ["MVT_ID_mvt", "ADEP_mvt", "inbound_callsign", "inbound_last_seen"]
    ]
    ln = q.merge(
        ak.rename(
            columns={"MVT_ID_mvt": "inbound_MVT_ID_mvt", "MVT_TIME_UTC_mvt": "arr_time"}
        ),
        left_on=["ADEP_mvt", "inbound_callsign"],
        right_on=["ADES_mvt", "callsign"],
    )
    ln["dist"] = (ln["arr_time"] - ln["inbound_last_seen"]).abs()
    ln = ln[ln["dist"] <= pd.Timedelta(seconds=LINK_S)]
    ln = ln.sort_values(["MVT_ID_mvt", "dist", "inbound_MVT_ID_mvt"], kind="stable")
    ln = ln.drop_duplicates("MVT_ID_mvt")
    ln = ln[["MVT_ID_mvt", "inbound_MVT_ID_mvt"]].merge(
        arr[
            ["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt"]
        ].rename(
            columns={
                "MVT_ID_mvt": "inbound_MVT_ID_mvt",
                "BLOCK_TIME_UTC_mvt": "inbound_block_time",
                "SCHED_TIME_UTC_mvt": "inbound_sched_time",
                "STAND_mvt": "inbound_stand",
            }
        ),
        on="inbound_MVT_ID_mvt",
        how="left",
    )
    out = out.merge(ln, on="MVT_ID_mvt", how="left")
    out["inbound_linked"] = out["inbound_MVT_ID_mvt"].notna()
    if not (out["MVT_ID_mvt"].is_unique and len(out) == len(dep)):
        raise RuntimeError(f"{ym}: a departure matched twice or lost")
    out = out.drop(columns=["ADEP_mvt", "MVT_TIME_UTC_mvt"])
    info = {
        "departures": int(len(out)),
        "matched": int(out["matched"].sum()),
        "nearest_ties": int((out["n_nearest"] > 1).sum()),
        "via_tie": int(out["opdi_tied"].sum()),
        "covered": int(out["covered"].sum()),
        "inbound_opdi": int(out["inbound_opdi_id"].notna().sum()),
        "inbound_linked": int(out["inbound_linked"].sum()),
    }
    return out, info


def write_table(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd"
    )
    tmp.rename(path)


__all__ = [
    "TEN",
    "TRAINING_MONTHS",
    "event_files",
    "list_files",
    "list_months",
    "match_month",
    "match_months",
    "month_table",
    "reduce_file",
    "tie",
    "tie_map",
    "url_of",
    "write_table",
]
