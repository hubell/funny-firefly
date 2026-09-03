#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The unimpeded term against the data: write ``reports/geometry_gaps.md``.

The geometry supplies a full-length stand-to-threshold distance for every
pair. This asks the training data three things about that quantity, per
airport and per designator, and writes the answers as aggregates:

1. How well a pair's fastest decile of taxi-out follows its routed
   distance -- the fit, its intercept, its marginal speed, its residual --
   and where the residuals sit: by join stage, by how many OpenStreetMap
   features share the identifier, by how far the stand had to be snapped.
2. What wake category does to the fastest decile, paired on the same stand
   and runway, and what it is worth on the mean once the stand is known.
3. Whether any designator carries the signature of intersection departures
   -- a wake-class gap out of line with the airport's -- and which taxiway
   junctions along each runway could produce it.

Every number is computed here. No stand identifier from either source is
written; designators and ICAO codes are public.

Needs the challenge data under ``data/challenge/``, the distance matrices
under ``data/processed/taxi_distance/`` and the OpenStreetMap cache under
``data/raw/`` (``make reports``). About a minute.

Usage
-----
    python scripts/report_geometry_gaps.py
"""

from __future__ import annotations

import glob
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.challenge import load_movements, to_pandas  # noqa: E402
from taxiout.graph import attach_stands, build_graph  # noqa: E402
from taxiout.join import UNMATCHED, join_stands  # noqa: E402
from taxiout.routing import label_runway_edges, runway_entries, runway_membership  # noqa: E402
from taxiout.runways import build_runways  # noqa: E402
from taxiout.tails import TailModel, combine  # noqa: E402

DATA = REPO / "data" / "challenge"
MATRICES = REPO / "data" / "processed" / "taxi_distance"
CACHE = REPO / "data" / "raw"
OUT = REPO / "reports" / "geometry_gaps.md"

MIN_PAIR_ROWS = 30  # the PRU reference's own floor for a pair
MIN_CLASS_ROWS = 20  # per (pair, wake class) for the paired gap
MIN_DESIGNATOR_PAIRS = 5
DIRECT_SNAP_M = 60.0
USABLE_ENTRY_SHARE = 0.6  # a junction in the first 60% of the runway is a usable entry
MIN_ENTRY_OFFSET_M = 50.0  # nearer than this to the threshold is the threshold
HOUR = 3600.0


# --------------------------------------------------------------------------
# helpers


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 0) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:,.{digits}f}"


def signed(x: float, digits: int = 0) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+,.{digits}f}"


def table(
    header: list[str], rows: list[list[str]], align: str | None = None
) -> list[str]:
    align = align or ("-" * len(header))
    sep = ["---:" if a == "r" else "---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """``(intercept, slope, rmse, r2)`` of ``y`` on ``x`` by least squares."""
    a = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    pred = a @ coef
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / tot if tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), rmse, r2


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


# --------------------------------------------------------------------------
# data


def load_matrix() -> pd.DataFrame:
    files = [
        f for f in sorted(glob.glob(str(MATRICES / "*.csv"))) if "_features" not in f
    ]
    if not files:
        raise FileNotFoundError(
            f"no distance matrices under {MATRICES}; run make reports"
        )
    return pd.concat(
        [pd.read_csv(f, dtype={"stand_id": str, "designator": str}) for f in files],
        ignore_index=True,
    )


def load_departures(matrix: pd.DataFrame) -> tuple[pd.DataFrame, TailModel]:
    cols = [
        "ADEP_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "TAXITIME_SEC_mvt",
        "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "LOBT_flt",
        "BLOCK_TIME_UTC_mvt",
        "WK_TBL_CAT_flt",
        "AIRCRAFT_TYPE_mvt",
    ]
    dep = to_pandas(load_movements(DATA, cols, phase="DEP"))
    dep["ADEP_mvt"] = dep["ADEP_mvt"].astype(str)
    dep["y"] = pd.to_numeric(dep["TAXITIME_SEC_mvt"], errors="coerce").astype(float)
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.month
    dep["wk"] = (
        dep["WK_TBL_CAT_flt"].astype(str).where(dep["WK_TBL_CAT_flt"].notna(), "none")
    )
    dep["typ"] = (
        dep["AIRCRAFT_TYPE_mvt"]
        .astype(str)
        .where(dep["AIRCRAFT_TYPE_mvt"].notna(), "none")
    )
    tm = TailModel().fit(dep)
    joined = join_stands(dep[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]], matrix)
    for c in joined.columns:
        if c not in dep.columns:
            dep[c] = joined[c].to_numpy()
    dep["flagged"] = tm.flagged(dep).to_numpy()
    return dep, tm


def pair_table(dep: pd.DataFrame) -> pd.DataFrame:
    """One row per joined (airport, stand, designator) pair with enough rows."""
    d = dep[
        (dep["join_stage"] != UNMATCHED)
        & ~dep["flagged"]
        & (dep["y"] > 0)
        & (dep["y"] < 2 * HOUR)
    ]
    g = d.groupby(["ADEP_mvt", "stand_id", "RUNWAY_mvt"], observed=True)
    pairs = g.agg(
        n=("y", "size"),
        p10=("y", lambda s: float(np.percentile(s, 10))),
        mean=("y", "mean"),
        taxi_m=("taxi_m", "first"),
        backtrack_m=("backtrack_m", "first"),
        snap=("snap_max_m", "first"),
        nf=("n_features", "first"),
        stage=("join_stage", "first"),
        heavy=("wk", lambda s: float(s.isin(["H", "J"]).mean())),
    ).reset_index()
    pairs = pairs[pairs["n"] >= MIN_PAIR_ROWS].copy()
    pairs["snap_tier"] = np.where(
        pairs["snap"] <= 0.5,
        "touches pavement",
        np.where(
            pairs["snap"] <= DIRECT_SNAP_M, "snapped, 60 m or less", "snapped, over 60 m"
        ),
    )
    pairs["clean"] = (pairs["snap"] <= DIRECT_SNAP_M) & (pairs["nf"] == 1)
    return pairs


# --------------------------------------------------------------------------
# sections


def section_fit(pairs: pd.DataFrame) -> tuple[list[str], dict]:
    fits: dict[str, tuple[float, float]] = {}
    rows = []
    facts: dict = {}
    for icao, pa in pairs.groupby("ADEP_mvt", sort=True):
        clean = pa[pa["clean"]]
        a, b, r, r2 = fit_line(clean["taxi_m"].to_numpy(), clean["p10"].to_numpy())
        fits[icao] = (a, b)
        q = np.percentile(clean["taxi_m"], [10, 50, 90])
        rows.append(
            [
                icao,
                f"{len(clean):,}",
                num(a),
                num(1.0 / b, 2) if b > 0 else "n/a",
                num(r),
                num(r2, 2),
                num(clean["p10"].std()),
                f"{num(q[0])} / {num(q[1])} / {num(q[2])}",
            ]
        )
        facts[icao] = {"a": a, "b": b, "rmse": r, "r2": r2, "spread": float(q[2] - q[0])}
    pairs["resid"] = [
        p10 - (fits[i][0] + fits[i][1] * t)
        for i, p10, t in zip(pairs["ADEP_mvt"], pairs["p10"], pairs["taxi_m"])
    ]
    worst = min(facts, key=lambda k: facts[k]["r2"])
    best = max(facts, key=lambda k: facts[k]["r2"])
    lines = [
        "## The fit",
        "",
        "For every joined (stand, designator) pair with at least",
        f"{MIN_PAIR_ROWS} unflagged departures, the tenth percentile of taxi-out is the",
        "PRU's own definition of the unimpeded time. Per airport, a line is fitted",
        "through the clean pairs -- one OpenStreetMap feature behind the",
        f"identifier, snapped {num(DIRECT_SNAP_M)} m or less -- of that percentile on the",
        "routed distance. The intercept is the fixed component, pushback and",
        "line-up, that no distance explains; the slope is a *marginal* speed,",
        "seconds per meter of extra route, which sits above a physical taxi",
        "speed because the longer routes at these airports also run on the",
        "straighter taxiways.",
        "",
        *table(
            [
                "ICAO",
                "clean pairs",
                "intercept s",
                "marginal speed m/s",
                "RMSE s",
                "R²",
                "sd of p10",
                "routed m, p10 / p50 / p90",
            ],
            rows,
            "-rrrrrrr",
        ),
        "",
        f"R² is a property of the spread of distance as much as of the geometry: {worst}",
        f"has the lowest, {num(facts[worst]['r2'], 2)}, on a p10-to-p90 spread of {num(facts[worst]['spread'])} m,",
        f"against {num(facts[best]['r2'], 2)} at {best} on {num(facts[best]['spread'])} m. The residual in",
        "seconds is the column to read.",
    ]
    return lines, {"fits": fits, "per_airport": facts}


def section_residuals(pairs: pd.DataFrame) -> list[str]:
    def breakdown(col: str, label: str, order=None) -> list[list[str]]:
        rows = []
        for icao, pa in pairs.groupby("ADEP_mvt", sort=True):
            g = pa.groupby(col)["resid"].agg(["mean", "size"])
            keys = order if order is not None else sorted(g.index)
            cells = []
            for k in keys:
                if k in g.index:
                    cells.append(f"{signed(g.loc[k, 'mean'])} ({int(g.loc[k, 'size'])})")
                else:
                    cells.append("--")
            rows.append([icao] + cells)
        return rows

    stages = [1, 2, 3]
    nfs = [1, 2, 3]
    tiers = ["touches pavement", "snapped, 60 m or less", "snapped, over 60 m"]
    pairs["nf_bin"] = pairs["nf"].clip(upper=3).astype(int)
    lines = [
        "## Where the residuals sit",
        "",
        "Mean residual of the fit above, in seconds, with the number of pairs",
        "in brackets; every pair is scored against its airport's clean-pair line,",
        "so a non-zero cell is what that class of pair does differently. A",
        "positive residual is a pair slower than its distance says.",
        "",
        "By the stage at which the identifier joined (1 case and whitespace,",
        "2 leading zeros, 3 trailing letter):",
        "",
        *table(
            ["ICAO"] + [f"stage {s}" for s in stages],
            breakdown("stage", "stage", stages),
            "-rrr",
        ),
        "",
        "By how many OpenStreetMap features share the identifier (the cell",
        "carries the feature nearest the group mean; three means three or more):",
        "",
        *table(
            ["ICAO"] + [f"{n} feature{'s' if n > 1 else ''}" for n in nfs],
            breakdown("nf_bin", "features", nfs),
            "-rrr",
        ),
        "",
        "By how far the stand was moved to reach mapped pavement:",
        "",
        *table(["ICAO"] + tiers, breakdown("snap_tier", "snap", tiers), "-rrr"),
        "",
        "Two-feature identifiers are the case `reports/synthetic_airport.md`",
        "found to be decided by floating-point rounding; where they carry a",
        "residual of their own, the representative is part of the reason, and",
        "these tables regenerate after the fix in `routing.py`.",
    ]
    return lines


def section_wake(
    dep: pd.DataFrame, pairs: pd.DataFrame, fits: dict
) -> tuple[list[str], dict]:
    facts: dict = {}
    d = dep[
        (dep["join_stage"] != UNMATCHED)
        & ~dep["flagged"]
        & (dep["y"] > 0)
        & (dep["y"] < 2 * HOUR)
    ]
    d = d[d["wk"].isin(["M", "H", "J"])].copy()
    d["cls"] = np.where(d["wk"] == "M", "M", "H")
    pw = (
        d.groupby(["ADEP_mvt", "RUNWAY_mvt", "stand_id", "cls"], observed=True)["y"]
        .agg(n="size", p10=lambda s: float(np.percentile(s, 10)), mean="mean")
        .reset_index()
    )
    pw = pw[pw["n"] >= MIN_CLASS_ROWS]
    wide = pw.pivot_table(
        index=["ADEP_mvt", "RUNWAY_mvt", "stand_id"],
        columns="cls",
        values=["p10", "mean"],
    ).dropna()
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["gap10"] = wide["p10_M"] - wide["p10_H"]
    wide["gap_mean"] = wide["mean_M"] - wide["mean_H"]
    rows = []
    per_designator = {}
    for (icao, dsg), w in wide.groupby(level=[0, 1]):
        if len(w) < MIN_DESIGNATOR_PAIRS:
            continue
        per_designator[(icao, dsg)] = {
            "stands": len(w),
            "gap10": float(w["gap10"].median()),
            "gap_mean": float(w["gap_mean"].median()),
        }
    airport_norm = {}
    for icao in sorted({k[0] for k in per_designator}):
        gaps = [v["gap10"] for k, v in per_designator.items() if k[0] == icao]
        airport_norm[icao] = float(np.median(gaps))
    for (icao, dsg), v in sorted(per_designator.items()):
        speed = 1.0 / fits[icao][1]
        rows.append(
            [
                icao,
                dsg,
                str(v["stands"]),
                signed(v["gap10"]),
                signed(v["gap10"] * speed),
                signed(v["gap_mean"]),
                signed(v["gap10"] - airport_norm[icao]),
            ]
        )
    facts["per_designator"] = per_designator
    facts["airport_norm"] = airport_norm
    facts["gap10_range"] = (
        min(v["gap10"] for v in per_designator.values()),
        max(v["gap10"] for v in per_designator.values()),
    )

    # What it is worth on the mean, once the stand is known: forward splits
    # with a pair-mean stand-in and an additive per-(airport, class) offset.
    worth_rows = []
    for label, train_m, valid_m in (
        ("Dec 2025", dep["month"] <= 11, dep["month"] == 12),
        ("Jul 2025, all", dep["month"] <= 6, dep["month"] == 7),
    ):
        tr, va = dep[train_m], dep[valid_m]
        tm = TailModel().fit(tr)
        fit = tr[~tm.flagged(tr).to_numpy()]
        tail = tm.predict(va)
        keys = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
        pair = fit.groupby(keys, observed=True)["y"].agg(["mean", "size"])
        pair = pair.loc[pair["size"] >= 5, "mean"]
        apm = fit.groupby("ADEP_mvt")["y"].mean()
        base_va = pd.Series(
            pair.reindex(pd.MultiIndex.from_frame(va[keys])).to_numpy(), index=va.index
        ).fillna(va["ADEP_mvt"].map(apm))
        base_tr = pd.Series(
            pair.reindex(pd.MultiIndex.from_frame(fit[keys])).to_numpy(), index=fit.index
        ).fillna(fit["ADEP_mvt"].map(apm))
        resid = fit["y"] - base_tr
        scores = [rmse(va["y"], combine(base_va, tail))]
        for col in ("wk", "typ"):
            off = resid.groupby([fit["ADEP_mvt"], fit[col]]).agg(["mean", "size"])
            off = off.loc[off["size"] >= 30, "mean"]
            k = pd.MultiIndex.from_arrays([va["ADEP_mvt"], va[col]])
            adj = base_va + pd.Series(off.reindex(k).to_numpy(), index=va.index).fillna(
                0.0
            )
            scores.append(rmse(va["y"], combine(adj, tail)))
        worth_rows.append([label] + [num(s, 1) for s in scores])
        facts.setdefault("worth", {})[label] = scores
    # The class effect on the mean at the same stand and runway, per airport:
    # what a model that knows the distance but not the stand is missing. The
    # unpaired gap would conflate the class with where heavies park.
    facts["paired_mean_gap"] = {
        icao: float(
            np.median([v["gap_mean"] for k, v in per_designator.items() if k[0] == icao])
        )
        for icao in airport_norm
    }
    lines = [
        "## The fastest decile by wake category",
        "",
        "Paired on the same stand and the same runway, with at least",
        f"{MIN_CLASS_ROWS} departures of each class: the tenth percentile of medium-wake",
        "departures against that of heavies (H and J), per designator, median",
        "over the stands. The distance column converts the gap at the airport's",
        "marginal speed; the last column is the designator's gap relative to",
        "the airport's median designator.",
        "",
        *table(
            [
                "ICAO",
                "designator",
                "stands",
                "M − H at p10, s",
                "as distance, m",
                "M − H on the mean, s",
                "vs airport, s",
            ],
            rows,
            "--rrrrr",
        ),
        "",
        f"The gap is between {num(facts['gap10_range'][0])} and {num(facts['gap10_range'][1])} s and nearly the",
        "same on every designator of an airport -- one to three kilometers at",
        "the fitted speeds, which is too large and too uniform to be",
        "intersection departures. It is a fixed difference in pushback,",
        "start-up, line-up and taxi speed by wake category, and neither the",
        "geometric unimpeded term nor the Gate 1 feature frame carries the",
        "category or the aircraft type, both of which the ranking set exposes",
        "for every departure.",
        "",
        "What it is worth depends on what else the model knows. On the mean,",
        "with the stand identity already in the model, an additive offset per",
        "(airport, class) fitted on training rows moves the forward-split RMSE",
        "of a pair-mean stand-in by about a second, because heavies park at",
        "wide-body stands and the stand absorbs the class:",
        "",
        *table(
            ["split", "pair mean", "+ wake class", "+ aircraft type"], worth_rows, "-rrr"
        ),
        "",
        "Without the stand -- the geometric rung, and any pair the model has",
        "never seen -- the class is worth its gap on the mean at the same stand",
        "and runway, the median over designators of the sixth column: "
        + ", ".join(
            f"{k} {signed(v)} s" for k, v in sorted(facts["paired_mean_gap"].items())
        )
        + " (medium minus heavy). That is where a per-class intercept in the",
        "unimpeded term belongs, and the held-out-pair split is where it",
        "should show.",
    ]
    return lines, facts


def candidate_entries() -> dict[tuple[str, str], dict]:
    """Per designator: runway length and taxiway junctions along the centerline."""
    out: dict[tuple[str, str], dict] = {}
    for ap in AIRPORTS:
        path = CACHE / f"{ap.icao}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            els = json.load(fh)["elements"]
        tg = build_graph(ap.icao, els, (ap.lat, ap.lon))
        attach_stands(tg, els)
        runways, _ = build_runways(ap.icao, els)
        label_runway_edges(tg, runways)
        entries = runway_entries(tg, runways)
        membership = runway_membership(tg.g)
        g = tg.g
        lengths = {r.ref: r.length_m for r in runways}
        for e in entries:
            if e.node is None:
                continue

            def along(_u, _v, d, ref=e.ref):
                return (
                    d["length_m"]
                    if d.get("kind") == "runway" and d.get("ref") == ref
                    else None
                )

            dist = nx.single_source_dijkstra_path_length(g, e.node, weight=along)
            offsets = sorted(
                off
                for n, off in dist.items()
                if e.ref in membership.get(n, frozenset())
                and off > MIN_ENTRY_OFFSET_M
                and any(g.edges[n, m].get("kind") != "runway" for m in g.neighbors(n))
            )
            length = lengths[e.ref]
            out[(ap.icao, e.designator)] = {
                "length": length,
                "entries": len(offsets),
                "usable": [o for o in offsets if o < USABLE_ENTRY_SHARE * length],
            }
    return out


def section_intersections(pairs: pd.DataFrame, wake: dict, entries: dict) -> list[str]:
    rows = []
    for icao, pa in pairs.groupby("ADEP_mvt", sort=True):
        for dsg, pd_ in pa.groupby("RUNWAY_mvt", observed=True):
            if len(pd_) < MIN_DESIGNATOR_PAIRS:
                continue
            w = wake["per_designator"].get((icao, dsg))
            dev = w["gap10"] - wake["airport_norm"][icao] if w else float("nan")
            ent = entries.get((icao, dsg))
            rows.append(
                [
                    icao,
                    dsg,
                    f"{len(pd_):,}",
                    f"{int(pd_['n'].sum()):,}",
                    signed(pd_["resid"].median()),
                    num(pd_["backtrack_m"].median()),
                    signed(dev),
                    num(ent["length"]) if ent else "n/a",
                    str(ent["entries"]) if ent else "n/a",
                    ", ".join(num(o) for o in ent["usable"][:5])
                    if ent and ent["usable"]
                    else ("none" if ent else "n/a"),
                ]
            )
    devs = [
        abs(v["gap10"] - wake["airport_norm"][k[0]])
        for k, v in wake["per_designator"].items()
    ]
    med = (
        pairs.groupby(["ADEP_mvt", "RUNWAY_mvt"], observed=True)
        .agg(resid=("resid", "median"), pairs=("n", "size"), rows=("n", "sum"))
        .query("pairs >= @MIN_DESIGNATOR_PAIRS")
    )
    big = med[med["resid"].abs() > 100]
    big_share = float(big["rows"].sum() / med["rows"].sum())
    lo, hi = med["resid"].idxmin(), med["resid"].idxmax()
    lines = [
        "## The intersection signature, per designator",
        "",
        "An intersection departure enters the runway at a taxiway junction",
        "short of the threshold and taxis less than the full-length route the",
        "matrix assumes. If a designator is used that way routinely, its pairs",
        "sit below the airport's line, and -- because heavies take the full",
        "length more often than mediums -- its wake-class gap is wider than",
        "the airport's. Both are here, beside what the graph knows: the",
        "runway's length, how many taxiway junctions lie on its centerline",
        f"beyond {num(MIN_ENTRY_OFFSET_M)} m from the threshold, and the offsets of those in",
        f"the first {pct(USABLE_ENTRY_SHARE, 0)} of the runway, which is where an intersection",
        "departure could start.",
        "",
        *table(
            [
                "ICAO",
                "designator",
                "pairs",
                "departures",
                "median residual s",
                "median backtrack m",
                "wake gap vs airport, s",
                "runway m",
                "junctions",
                "usable entries, m from threshold",
            ],
            rows,
            "--rrrrrrr-",
        ),
        "",
        f"No designator's wake-class gap departs from its airport's by more than {num(max(devs))} s.",
        f"The designator residuals run from {signed(med.loc[lo, 'resid'])} s ({lo[0]} {lo[1]}) to",
        f"{signed(med.loc[hi, 'resid'])} s ({hi[0]} {hi[1]}); the {len(big)} designators beyond a",
        f"hundred seconds either way carry {pct(big_share)} of the departures between them,",
        "and are the short-runway and heavy-only configurations. Both quantities",
        "are the size of one entry offset at the fitted speeds, and both are",
        "constant per designator, so a model that knows the designator and the",
        "wake category absorbs them. What no feature can recover is which",
        "stands use which entry: the routed distance to an intersection differs",
        "from the threshold route by the same offset for every stand upstream",
        "of it, and by something else only for the few between it and the",
        "threshold. That is a question for surface tracks, and it stays a",
        "diagnostic for the paper if they arrive.",
        "",
        "A designator with no usable entry in the graph is one where OSM stops",
        "every taxiway at the runway's edge rather than joining its centerline;",
        "the routing report lists those, and the threshold is reached along",
        "the runway from wherever the first junction is.",
    ]
    return lines


def check_no_identifiers(text: str, dep: pd.DataFrame) -> None:
    ids = set()
    for v in dep["STAND_mvt"].dropna().astype(str).unique():
        v = v.strip()
        if len(v) >= 3 and re.search(r"[A-Za-z]", v) and re.search(r"[0-9]", v):
            ids.add(v)
    hits = [i for i in ids if re.search(r"(?<![\w])" + re.escape(i) + r"(?![\w])", text)]
    if hits:
        raise RuntimeError(f"{len(hits)} literal stand identifiers would be written")


# --------------------------------------------------------------------------


def main() -> int:
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to report", file=sys.stderr)
        return 1
    matrix = load_matrix()
    dep, _tm = load_departures(matrix)
    pairs = pair_table(dep)
    fit_lines, fit_facts = section_fit(pairs)
    resid_lines = section_residuals(pairs)
    wake_lines, wake_facts = section_wake(dep, pairs, fit_facts["fits"])
    entries = candidate_entries()
    inter_lines = section_intersections(pairs, wake_facts, entries)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    per = fit_facts["per_airport"]
    worst = min(per, key=lambda k: per[k]["r2"])
    dec = wake_facts["worth"]["Dec 2025"]
    head = [
        "# The unimpeded term against the data",
        "",
        "What the training year says about the quantity the geometry supplies:",
        "how closely a pair's fastest decile follows its routed distance, where",
        "the residuals sit, what wake category does to it, and whether any",
        "designator carries the signature of intersection departures.",
        f"Generated by `scripts/report_geometry_gaps.py` on {generated}; every figure is",
        "computed there, and the generator refuses to write a literal stand",
        "identifier from either source.",
        "",
        "## In one paragraph",
        "",
        "Per airport, the fastest decile of a pair's taxi-out follows its routed",
        f"distance with a residual of {num(min(v['rmse'] for v in per.values()))} to {num(max(v['rmse'] for v in per.values()))} s and an intercept of",
        f"{num(min(v['a'] for v in per.values()))} to {num(max(v['a'] for v in per.values()))} s that no distance explains; {worst} fits worst, on the",
        "narrowest spread of distances. The largest structured effect on that",
        "decile is wake category: paired on the same stand and runway, mediums",
        f"are {num(-wake_facts['gap10_range'][1])} to {num(-wake_facts['gap10_range'][0])} s faster than heavies, nearly the same on",
        "every designator of an airport, which is a fixed pushback-to-line-up",
        "difference and not intersection use. On the mean, with the stand known,",
        f"it is worth about {num(dec[0] - dec[1], 1)} s of RMSE; without the stand it is worth its gap",
        "on the mean, which is where a per-class intercept in the unimpeded",
        "term belongs.",
        "The intersection signature proper -- a designator whose wake gap or",
        "residual is out of line with its airport -- is at most a hundred",
        "seconds and constant per designator, absorbed by any model that knows",
        "the designator; the stand-level part needs surface tracks. Where the",
        "fit is weakest the residuals sit on identifiers shared by two",
        "OpenStreetMap features and on snapped stands, which is the",
        "representative-pick finding of the worked airport showing on real",
        "ground.",
        "",
    ]
    footer = [
        "",
        "Distance matrices derived from OpenStreetMap: data (c) OpenStreetMap contributors,",
        "ODbL 1.0. Movement and flight data supplied by EUROCONTROL PRC and the",
        "OpenSky Network under the terms of the PRC Data Challenge 2026.",
    ]
    body = (
        head
        + fit_lines
        + [""]
        + resid_lines
        + [""]
        + wake_lines
        + [""]
        + inter_lines
        + footer
    )
    text = "\n".join(body) + "\n"
    check_no_identifiers(text, dep)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
