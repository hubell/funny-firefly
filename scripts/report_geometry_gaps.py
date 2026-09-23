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
4. Where memorization runs out: how much of the ranking set sits on
   (stand, runway) pairs the training year saw rarely or never, which stands
   and designators are new outright, and -- on the two forward splits --
   what a pair's own mean is worth where it is thin against the same mean
   shrunk toward the airport mean and toward the geometric prediction.

Every number is computed here. No stand identifier from either source is
written; designators and ICAO codes are public, and the identifiers of
stands the training year never saw are described by their character shape
only. The refusal covers the training set, the ranking set and the matrix.

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
from taxiout.challenge import load_movements, load_ranking, to_pandas  # noqa: E402
from taxiout.coverage import identifier_shape  # noqa: E402
from taxiout.encoding import geometric_prior  # noqa: E402
from taxiout.features import AGGREGATED_FIELDS, airport_fallback  # noqa: E402
from taxiout.graph import DIRECT_SNAP_M, attach_stands, build_graph  # noqa: E402
from taxiout.join import UNMATCHED, join_stands  # noqa: E402
from taxiout.routing import label_runway_edges, runway_entries, runway_membership  # noqa: E402
from taxiout.runways import build_runways  # noqa: E402
from taxiout.tails import TailModel, combine  # noqa: E402
from taxiout.validation import FREQUENCY_BUCKETS, frequency_bucket  # noqa: E402

DATA = REPO / "data" / "challenge"
MATRICES = REPO / "data" / "processed" / "taxi_distance"
CACHE = REPO / "data" / "raw"
OUT = REPO / "reports" / "geometry_gaps.md"

MIN_PAIR_ROWS = 30  # the PRU reference's own floor for a pair
MIN_CLASS_ROWS = 20  # per (pair, wake class) for the paired gap
MIN_DESIGNATOR_PAIRS = 5
USABLE_ENTRY_SHARE = 0.6  # a junction in the first 60% of the runway is a usable entry
MIN_ENTRY_OFFSET_M = 50.0  # nearer than this to the threshold is the threshold
HOUR = 3600.0

# The shrinkage weight's floor, as in ``taxiout.encoding``: an airport whose
# between-pair variance falls to this fraction of its uncorrected value takes
# the floor rather than a division by zero.
MIN_BETWEEN_FRACTION = 0.05
# The ranking months, in the order the test set stacks them.
RANKING_MONTHS = ((1, "January 2026"), (7, "July 2026"))
# The two forward splits, as ``taxiout.validation`` fixes them: train through
# the month before, validate the month named.
FORWARD_SPLITS = (("December 2025", 11, 12), ("July 2025", 6, 7))
THIN_BUCKETS = FREQUENCY_BUCKETS[:3]  # seen a hundred times or fewer
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


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


def raw_pair(frame: pd.DataFrame) -> pd.Series:
    """``icao|stand|designator`` from the raw triple, as the identifier model sees it.

    The raw triple rather than the joined cell: a stand the join could not
    place still forms a pair, and "seen" is a question about the identifier
    the model memorizes, not about the geometry behind it.
    """
    return (
        frame["ADEP_mvt"].astype(str)
        + "|"
        + frame["STAND_mvt"].astype(str)
        + "|"
        + frame["RUNWAY_mvt"].astype(str)
    )


def load_ranking_departures(matrix: pd.DataFrame) -> pd.DataFrame:
    """The 2026 ranking departures, keyed like the training rows and joined."""
    rk = to_pandas(
        load_ranking(
            DATA, ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"], phase="DEP"
        )
    )
    rk["ADEP_mvt"] = rk["ADEP_mvt"].astype(str)
    rk["month"] = rk["MVT_TIME_UTC_mvt"].dt.month
    rk["pair"] = raw_pair(rk)
    joined = join_stands(rk[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]], matrix)
    rk["join_stage"] = joined["join_stage"].to_numpy()
    return rk


def geometry_frame(dep: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """The columns the unimpeded fit reads, with the airport-level fallback.

    The same block ``taxiout.features.build_features`` carries, built from
    the joined departures here so the prior below is the prior the shrunk
    encoding uses -- fitted by ``taxiout.encoding.geometric_prior`` on the
    same columns -- without building the congestion and weather blocks this
    report does not read. Unmatched rows take the matrix's own (airport,
    runway) median cell, as they do in the model.
    """
    out = pd.DataFrame(index=dep.index)
    out["icao"] = dep["ADEP_mvt"].astype(str)
    out["stand"] = dep["STAND_mvt"].astype(str)
    out["designator"] = dep["RUNWAY_mvt"].astype(str)
    out["pair"] = raw_pair(dep)
    matched = (dep["join_stage"] != UNMATCHED).to_numpy()
    for c in AGGREGATED_FIELDS:
        out[c] = pd.to_numeric(dep[c], errors="coerce").to_numpy(dtype=float)
    out["geom_imputed"] = (~matched).astype(float)
    fallback = airport_fallback(matrix)
    need = ~matched
    if need.any():
        keys = out.loc[need, ["icao", "designator"]].reset_index(drop=True)
        per_runway = fallback[fallback["designator"].notna()].astype({"designator": str})
        per_airport = fallback[fallback["designator"].isna()].drop(columns=["designator"])
        by_runway = keys.merge(per_runway, on=["icao", "designator"], how="left")
        by_airport = keys[["icao"]].merge(per_airport, on="icao", how="left")
        for c in AGGREGATED_FIELDS:
            out.loc[need, c] = by_runway[c].fillna(by_airport[c]).to_numpy()

    def ratio(n: pd.Series, d: pd.Series) -> pd.Series:
        return n / d.where(d.abs() >= 1.0)

    route = out["route_m"]
    out["straight_frac"] = ratio(out["straight_m"], route)
    out["backtrack_frac"] = ratio(out["backtrack_m"], route)
    out["turn_deg_per_km"] = 1000.0 * ratio(out["turn_deg"], route)
    out["junctions_per_km"] = 1000.0 * ratio(out["junctions"], route)
    return out


def shrinkage_weights(
    fit: pd.DataFrame, prior: pd.Series, min_rows: int = MIN_PAIR_ROWS
) -> pd.DataFrame:
    """``k = sigma2_within / sigma2_between`` per airport, by the method of moments.

    ``fit`` carries ``ADEP_mvt``, ``pair`` and ``y``; ``prior`` is the value
    the pair mean is shrunk toward, aligned to it. Over the pairs with at
    least ``min_rows`` fit rows: the pooled within-pair variance of the
    residual ``y - prior``, and the variance of the pair means of that
    residual less the mean of ``sigma2_within / n`` over the same pairs --
    the sampling noise in each pair's own mean, which the observed spread
    contains and the true spread does not. The construction and its floor
    are ``taxiout.encoding``'s; it is restated here so this report's number
    is computed in this report, and the encoding's own table is the check.
    """
    r = fit["y"].to_numpy(dtype=float) - prior.to_numpy(dtype=float)
    groups = (
        pd.DataFrame(
            {"icao": fit["ADEP_mvt"].to_numpy(), "pair": fit["pair"].to_numpy(), "r": r}
        )
        .groupby(["icao", "pair"], observed=True)["r"]
        .agg(["mean", "var", "size"])
    )
    groups = groups[groups["size"] >= min_rows]
    rows = []
    for icao, g in groups.groupby(level=0):
        dof = (g["size"] - 1).to_numpy(dtype=float)
        within = float(np.sum(g["var"].to_numpy() * dof) / dof.sum())
        raw = float(np.var(g["mean"].to_numpy(), ddof=1))
        correction = float(np.mean(within / g["size"].to_numpy()))
        between = max(raw - correction, MIN_BETWEEN_FRACTION * raw)
        rows.append(
            {
                "icao": icao,
                "pairs": int(len(g)),
                "sd_within": np.sqrt(within),
                "sd_between": np.sqrt(between),
                "floored": raw - correction <= MIN_BETWEEN_FRACTION * raw,
                "k": within / between,
            }
        )
    return pd.DataFrame(rows).set_index("icao")


def thin_forward_split(
    dep: pd.DataFrame, geometry: pd.DataFrame, last_train_month: int, valid_month: int
) -> dict:
    """One forward split: the pair mean, raw and shrunk two ways, by bucket.

    Training rows are the months through ``last_train_month``, less the rows
    the record-quality term flags on them; validation is ``valid_month``,
    scored over the rows the term does not cover, as every strata table in
    ``baseline.md`` is. Five predictors, each fitted on the training rows and
    nothing else: the airport mean; the pair's own mean, falling back to the
    airport mean where the pair was never seen; the geometric prior -- the
    unimpeded prediction plus the per-airport offset, ``taxiout.encoding``'s
    ``g_hat``; and the pair mean shrunk toward each of the last two by its
    airport's ``k``. A pair never seen shrinks to its prior exactly.
    """
    train = dep["month"] <= last_train_month
    valid = dep["month"] == valid_month
    tail_columns = [
        "ADEP_mvt",
        "LOBT_flt",
        "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
    ]
    tm = TailModel().fit(dep.loc[train, tail_columns])
    flagged_train = pd.Series(False, index=dep.index)
    flagged_train[train] = tm.flagged(dep.loc[train, tail_columns]).to_numpy()
    fit_rows = train & ~flagged_train & dep["y"].notna()
    keep = ~tm.flagged(dep.loc[valid, tail_columns]).to_numpy()

    prior = geometric_prior(geometry, dep["y"], fit_rows)
    fit = dep.loc[fit_rows, ["ADEP_mvt", "pair", "y"]]
    counts = fit["pair"].value_counts()
    pair_mean = fit.groupby("pair")["y"].mean()
    airport_mean = fit.groupby("ADEP_mvt")["y"].mean()
    k_airport = shrinkage_weights(fit, fit["ADEP_mvt"].map(airport_mean))
    k_geometry = shrinkage_weights(fit, prior.values[fit_rows])

    v = dep.loc[valid, ["ADEP_mvt", "pair", "y"]].copy()
    v["n"] = v["pair"].map(counts).fillna(0.0)
    v["ybar"] = v["pair"].map(pair_mean)
    v["airport"] = v["ADEP_mvt"].map(airport_mean)
    v["geometry"] = prior.values[valid].to_numpy()
    v["raw"] = v["ybar"].fillna(v["airport"])
    for name, table in (("airport", k_airport), ("geometry", k_geometry)):
        k = v["ADEP_mvt"].map(table["k"]).fillna(float(table["k"].median()))
        v[f"shrunk_{name}"] = (v["n"] * v["ybar"].fillna(0.0) + k * v[name]) / (
            v["n"] + k
        )
    v["bucket"] = frequency_bucket(v["n"])
    v = v[keep]
    predictors = ["airport", "raw", "geometry", "shrunk_airport", "shrunk_geometry"]
    by_bucket = {}
    for b in FREQUENCY_BUCKETS:
        m = v["bucket"] == b
        by_bucket[b] = {"rows": int(m.sum())}
        for p in predictors:
            by_bucket[b][p] = (
                rmse(v.loc[m, "y"], v.loc[m, p]) if m.any() else float("nan")
            )
    return {
        "by_bucket": by_bucket,
        "k_airport": k_airport,
        "k_geometry": k_geometry,
        "offset": prior.offset,
        "speed": prior.default_speed,
        "rows": int(len(v)),
    }


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
    # Only the columns read, so the copy is a tenth of the frame: the full
    # frame copied twice here is what put this report over four gigabytes.
    d = dep.loc[
        (dep["join_stage"] != UNMATCHED)
        & ~dep["flagged"]
        & (dep["y"] > 0)
        & (dep["y"] < 2 * HOUR)
        & dep["wk"].isin(["M", "H", "J"]),
        ["ADEP_mvt", "RUNWAY_mvt", "stand_id", "wk", "y"],
    ].copy()
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
    split_columns = [
        "ADEP_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "y",
        "wk",
        "typ",
        "LOBT_flt",
        "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
    ]
    for label, train_m, valid_m in (
        ("Dec 2025", dep["month"] <= 11, dep["month"] == 12),
        ("Jul 2025, all", dep["month"] <= 6, dep["month"] == 7),
    ):
        tr, va = dep.loc[train_m, split_columns], dep.loc[valid_m, split_columns]
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


def section_thin_pairs(
    dep: pd.DataFrame, rk: pd.DataFrame, geometry: pd.DataFrame
) -> tuple[list[str], dict]:
    """Where memorization runs out: the thin pairs, the new stands, the priors."""
    facts: dict = {}
    fit = dep[~dep["flagged"] & dep["y"].notna()]
    counts = fit["pair"].value_counts()
    rk = rk.copy()
    rk["n"] = rk["pair"].map(counts).fillna(0.0)
    rk["bucket"] = frequency_bucket(rk["n"])
    total = len(rk)

    def shares(frame: pd.DataFrame) -> dict[str, float]:
        vc = frame["bucket"].value_counts()
        return {b: float(vc.get(b, 0)) / max(len(frame), 1) for b in FREQUENCY_BUCKETS}

    pooled = shares(rk)
    facts["ranking_rows"] = total
    facts["pooled"] = pooled
    facts["thin_share"] = sum(pooled[b] for b in THIN_BUCKETS)
    pooled_rows = [
        [b, f"{int((rk['bucket'] == b).sum()):,}", pct(pooled[b], 2)]
        for b in FREQUENCY_BUCKETS
    ]
    pooled_rows.append(
        [
            "a hundred or fewer",
            f"{int(rk['bucket'].isin(THIN_BUCKETS).sum()):,}",
            pct(facts["thin_share"], 2),
        ]
    )

    def share_row(label: str, frame: pd.DataFrame) -> list[str]:
        s = shares(frame)
        return (
            [label, f"{len(frame):,}"]
            + [pct(s[b], 2) for b in FREQUENCY_BUCKETS]
            + [pct(sum(s[b] for b in THIN_BUCKETS), 2)]
        )

    per_airport = {a: shares(g) for a, g in rk.groupby("ADEP_mvt", sort=True)}
    facts["per_airport"] = per_airport
    airport_rows = [share_row(a, g) for a, g in rk.groupby("ADEP_mvt", sort=True)]
    most_never = max(per_airport, key=lambda a: per_airport[a]["0"])
    facts["most_never"] = most_never

    per_month = {}
    month_rows = []
    for m, label in RANKING_MONTHS:
        g = rk[rk["month"] == m]
        per_month[label] = shares(g)
        month_rows.append(share_row(label, g))
        if g["ADEP_mvt"].nunique() < rk["ADEP_mvt"].nunique():
            for a, ga in g.groupby("ADEP_mvt", sort=True):
                month_rows.append(share_row(f"{label}, {a}", ga))
    facts["per_month"] = per_month

    # Stands and designators the training year never saw, by airport. A null
    # stand is neither new nor seen; it is unmatched by construction and is
    # counted with the unmatched rows in the coverage report.
    seen_stands = set(dep["ADEP_mvt"].astype(str) + "|" + dep["STAND_mvt"].astype(str))
    seen_runways = set(dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str))
    has_stand = rk["STAND_mvt"].notna()
    rk["stand_key"] = rk["ADEP_mvt"] + "|" + rk["STAND_mvt"].astype(str)
    rk["runway_key"] = rk["ADEP_mvt"] + "|" + rk["RUNWAY_mvt"].astype(str)
    rk["new_stand"] = has_stand & ~rk["stand_key"].isin(seen_stands)
    rk["new_runway"] = rk["RUNWAY_mvt"].notna() & ~rk["runway_key"].isin(seen_runways)
    new = rk[rk["new_stand"]]
    facts["new_stands"] = int(new["stand_key"].nunique())
    facts["new_stand_rows"] = int(len(new))
    facts["new_stand_airports"] = int(new["ADEP_mvt"].nunique())
    facts["new_runways"] = int(rk.loc[rk["new_runway"], "runway_key"].nunique())
    facts["new_runway_rows"] = int(rk["new_runway"].sum())
    new_rows = []
    facts["new_by_airport"] = {}
    for a, g in new.groupby("ADEP_mvt", sort=True):
        stands = g.drop_duplicates("stand_key")
        placed = stands["join_stage"] != UNMATCHED
        shapes = pd.Series(
            [identifier_shape(str(v)) for v in stands["STAND_mvt"]]
        ).value_counts()
        airport_rows_n = int((rk["ADEP_mvt"] == a).sum())
        facts["new_by_airport"][a] = {
            "stands": int(len(stands)),
            "rows": int(len(g)),
            "placed": int(placed.sum()),
            "placed_rows": int((g["join_stage"] != UNMATCHED).sum()),
        }
        new_rows.append(
            [
                a,
                str(len(stands)),
                f"{len(g):,}",
                pct(len(g) / airport_rows_n, 2),
                str(int(placed.sum())),
                f"{int((g['join_stage'] != UNMATCHED).sum()):,}",
                ", ".join(f"`{s}` {n}" for s, n in shapes.head(3).items()),
            ]
        )
    facts["new_placed_rows"] = int((new["join_stage"] != UNMATCHED).sum())
    never = rk[rk["bucket"] == "0"]
    facts["never_rows"] = int(len(never))
    facts["never_new_stand"] = int(never["new_stand"].sum())
    facts["never_combination"] = int((~never["new_stand"] & ~never["new_runway"]).sum())
    facts["never_placed"] = int((never["join_stage"] != UNMATCHED).sum())

    # The two forward splits.
    splits = {
        label: thin_forward_split(dep, geometry, last, valid)
        for label, last, valid in FORWARD_SPLITS
    }
    facts["splits"] = splits
    predictors = [
        ("airport", "airport mean"),
        ("raw", "pair mean, raw"),
        ("geometry", "geometric prior"),
        ("shrunk_airport", "shrunk toward the airport mean"),
        ("shrunk_geometry", "shrunk toward the geometry"),
    ]
    split_tables: list[str] = []
    for label, last, valid in FORWARD_SPLITS:
        s = splits[label]
        rows = [
            [b, f"{s['by_bucket'][b]['rows']:,}"]
            + [num(s["by_bucket"][b][p], 1) for p, _ in predictors]
            for b in FREQUENCY_BUCKETS
        ]
        split_tables += [
            f"**{label}.** Train January to {MONTHS[last - 1]}, validate "
            f"{MONTHS[valid - 1]}, all ten airports, {s['rows']:,} validation rows",
            "the record-quality term leaves to the physical model. Offset "
            f"{num(s['offset'].min())} to {num(s['offset'].max())} s per airport; "
            f"pooled speed {s['speed']:.2f} m/s.",
            "",
            *table(
                ["bucket", "rows"] + [name for _, name in predictors],
                rows,
                "-rrrrrr",
            ),
            "",
        ]
    dec = splits[FORWARD_SPLITS[0][0]]["by_bucket"]
    jul = splits[FORWARD_SPLITS[1][0]]["by_bucket"]
    facts["dec"] = dec
    facts["jul"] = jul

    kd = splits[FORWARD_SPLITS[0][0]]
    k_rows = []
    for a in sorted(set(kd["k_airport"].index) | set(kd["k_geometry"].index)):
        ka = kd["k_airport"].loc[a] if a in kd["k_airport"].index else None
        kg = kd["k_geometry"].loc[a] if a in kd["k_geometry"].index else None
        k_rows.append(
            [
                a,
                f"{int(kg['pairs']):,}" if kg is not None else "--",
                num(kg["sd_within"]) if kg is not None else "--",
                (num(ka["sd_between"]) + ("*" if ka["floored"] else ""))
                if ka is not None
                else "--",
                (num(kg["sd_between"]) + ("*" if kg["floored"] else ""))
                if kg is not None
                else "--",
                num(ka["k"], 1) if ka is not None else "--",
                num(kg["k"], 1) if kg is not None else "--",
            ]
        )
    explained = 1.0 - (kd["k_geometry"]["sd_between"] ** 2) / (
        kd["k_airport"]["sd_between"].reindex(kd["k_geometry"].index) ** 2
    )
    facts["between_explained"] = (float(explained.min()), float(explained.max()))

    thin_dec = {
        b: dec[b]["shrunk_airport"] - dec[b]["shrunk_geometry"] for b in THIN_BUCKETS
    }
    thin_jul = {
        b: jul[b]["shrunk_airport"] - jul[b]["shrunk_geometry"] for b in THIN_BUCKETS
    }
    facts["prior_gain"] = {"dec": thin_dec, "jul": thin_jul}

    lines = [
        "## Where memorization runs out",
        "",
        "The identifier model memorizes a (stand, runway) pair; the geometry",
        "computes it. This section measures where the first has nothing to",
        "memorize: how much of the 2026 ranking set sits on pairs the 2025",
        "training year saw rarely or never, which of those are stands new",
        "outright, and -- on the two forward splits -- what a pair's own mean is",
        "worth where it is thin, against that mean shrunk toward the airport",
        "mean and toward the geometric prediction. A pair is the raw",
        "(airport, stand, runway) triple, as the identifier model sees it; its",
        "count is the number of unflagged 2025 training departures on it, and the",
        "buckets are `baseline.md`'s.",
        "",
        "### How much of the ranking set is thin",
        "",
        f"Of the {total:,} ranking departures:",
        "",
        *table(["pair seen in 2025", "ranking rows", "share"], pooled_rows, "-rr"),
        "",
        "Per airport, over both ranking months. The last column is the thin",
        "share: pairs seen a hundred times or fewer.",
        "",
        *table(
            ["ICAO", "rows"] + list(FREQUENCY_BUCKETS) + ["a hundred or fewer"],
            airport_rows,
            "-rrrrrr",
        ),
        "",
        "Per ranking month, and per airport within the month that contains only",
        "three:",
        "",
        *table(
            ["month", "rows"] + list(FREQUENCY_BUCKETS) + ["a hundred or fewer"],
            month_rows,
            "-rrrrrr",
        ),
        "",
        "### Stands and designators the training year never saw",
        "",
        f"{facts['new_stands']} stands appear in the ranking set that appear in no 2025",
        f"training departure, at {facts['new_stand_airports']} airports, carrying",
        f"{facts['new_stand_rows']:,} ranking rows between them; {facts['new_placed_rows']:,} of those",
        "rows are on a stand the OpenStreetMap join can place, so the geometry",
        "has a cell for them and the identifier model has nothing. No runway",
        "designator is new"
        + (
            "."
            if facts["new_runways"] == 0
            else f" -- {facts['new_runways']} are, on {facts['new_runway_rows']:,} rows."
        ),
        "The stands are described by the shape of their identifier -- a digit is",
        "`#`, a letter `@` -- and never by the identifier itself.",
        "",
        *table(
            [
                "ICAO",
                "new stands",
                "rows",
                "share of airport's ranking rows",
                "placed by the join",
                "rows placed",
                "identifier shapes",
            ],
            new_rows,
            "-rrrrr-",
        ),
        "",
        f"The never-seen bucket is {facts['never_rows']:,} rows: {facts['never_new_stand']:,} on a new",
        f"stand and {facts['never_combination']:,} on a stand and a runway the training year saw",
        "separately and never together. The second kind is what the held-out-pair",
        f"split in `baseline.md` measures; the first is the case no split of the training data can pose. The join places {facts['never_placed']:,} of the {facts['never_rows']:,}.",
        "",
        "### The pair mean where it is thin",
        "",
        "Two forward splits, each fitted on its own training months with the",
        "record-quality rows removed and scored on the rows the term does not",
        "cover, by how often the validation row's pair appeared in those months.",
        "Five predictors, none of them a model: the airport mean; the pair's own",
        "training mean, falling back to the airport mean where the pair was never",
        "seen; the geometric prior, which is the unimpeded prediction plus the",
        "per-airport offset that carries it onto the scale of an observed",
        "taxi-out -- the same `g_hat` the shrunk encoding in `baseline.md` uses,",
        "fitted by the same function; and the pair mean shrunk toward each of the",
        "last two, `(n * ybar + k * prior) / (n + k)`, with `k` per airport from",
        "the table that follows. On a never-seen pair the raw mean *is* the",
        "airport mean and each shrunk mean *is* its prior, so the first row of",
        "each table compares the two priors bare.",
        "",
        *split_tables,
        "The shrinkage weight, December split, per airport and per prior. `sd",
        "within` is the pooled within-pair standard deviation of the residual",
        "about the prior, over pairs with at least",
        f"{MIN_PAIR_ROWS} training rows; `sd between` is the spread of pair means",
        "of that residual with the sampling noise of each mean subtracted, and",
        "`k` is the ratio of the variances. A between-spread marked `*` fell to",
        f"the floor of {pct(MIN_BETWEEN_FRACTION, 0)} of its uncorrected value. The geometry's",
        "between-spread is the part of the pair-to-pair spread it does not",
        "explain, so the two `sd between` columns read together say how much of",
        "what a pair mean carries the geometry already knows.",
        "",
        *table(
            [
                "ICAO",
                "pairs",
                "sd within",
                "sd between, airport prior",
                "sd between, geometric prior",
                "k toward the airport mean",
                "k toward the geometry",
            ],
            k_rows,
            "-rrrrrr",
        ),
        "",
        f"The geometry explains {pct(facts['between_explained'][0], 0)} to {pct(facts['between_explained'][1], 0)} of the between-pair",
        "variance the airport mean leaves, which is why `k` toward it is the",
        "larger: a pair's handful of draws is weighed against a prior that",
        "already knows most of what they would say.",
        "",
        "Read across the `1-10` row. On the December split the raw pair mean",
        f"scores {num(dec['1-10']['raw'], 1)} s where ignoring the pair scores",
        f"{num(dec['1-10']['airport'], 1)}: a mean over a handful of draws from a",
        "distribution several hundred seconds wide is noise wearing the costume",
        f"of knowledge. Shrunk toward the airport mean it scores {num(dec['1-10']['shrunk_airport'], 1)};",
        f"shrunk toward the geometry, {num(dec['1-10']['shrunk_geometry'], 1)}. The geometric prior is worth",
        f"{num(thin_dec['1-10'], 1)} s more than the group prior on that bucket in December and",
        f"{num(thin_jul['1-10'], 1)} s in July, and on the never-seen bucket, where the two priors",
        f"stand bare, {num(thin_dec['0'], 1)} and {num(thin_jul['0'], 1)} s. On pairs seen more than a",
        "hundred times the three shrunk and raw means agree to a second, because",
        "`n` swamps `k`: where memorization works, the prior does not matter, and",
        "where it runs out, which prior it runs out to is the whole question.",
    ]
    return lines, facts


def check_no_identifiers(text: str, stands: set[str], exempt: set[str]) -> None:
    """Refuse to write a report carrying a literal stand identifier.

    ``stands`` is every identifier the report could have leaked -- from the
    training departures, the ranking departures and the matrix alike, since
    a stand new in 2026 is in the ranking set and nowhere else. ``exempt`` is
    what is public anyway: designators and ICAO codes.
    """
    ids = set()
    for v in stands:
        v = str(v).strip()
        if len(v) >= 3 and re.search(r"[A-Za-z]", v) and re.search(r"[0-9]", v):
            ids.add(v)
    hits = [
        i
        for i in ids - exempt
        if re.search(r"(?<![\w])" + re.escape(i) + r"(?![\w])", text)
    ]
    if hits:
        raise RuntimeError(f"{len(hits)} literal stand identifiers would be written")


# --------------------------------------------------------------------------


def main() -> int:
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to report", file=sys.stderr)
        return 1
    matrix = load_matrix()
    dep, _tm = load_departures(matrix)
    dep["pair"] = raw_pair(dep)
    rk = load_ranking_departures(matrix)
    geometry = geometry_frame(dep, matrix)
    thin_lines, thin = section_thin_pairs(dep, rk, geometry)
    del geometry
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
    gain = thin["prior_gain"]
    head = [
        "# The unimpeded term against the data",
        "",
        "What the training year says about the quantity the geometry supplies:",
        "how closely a pair's fastest decile follows its routed distance, where",
        "the residuals sit, what wake category does to it, whether any",
        "designator carries the signature of intersection departures, and where",
        "in the ranking set the memorized pair runs out and the geometry is what",
        "remains.",
        f"Generated by `scripts/report_geometry_gaps.py` on {generated}; every figure is",
        "computed there, and the generator refuses to write a literal stand",
        "identifier from any source, the ranking set included.",
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
        f"Of the ranking set, {pct(thin['thin_share'])} sits on a pair the training year saw a",
        f"hundred times or fewer and {pct(thin['pooled']['0'], 2)} on one it never saw;",
        f"{thin['new_stands']} stands are new outright, {thin['new_by_airport'].get(thin['most_never'], {}).get('stands', 0)} of them at {thin['most_never']}, which has",
        f"the largest never-seen share at {pct(thin['per_airport'][thin['most_never']]['0'], 2)}. Where a pair is",
        "thin its own mean is worse than the airport mean, and the mean shrunk",
        f"toward the geometry beats the mean shrunk toward the airport by {num(gain['dec']['1-10'], 1)} s in",
        f"December and {num(gain['jul']['1-10'], 1)} s in July on pairs seen one to ten times.",
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
        + [""]
        + thin_lines
        + footer
    )
    text = "\n".join(body) + "\n"
    check_no_identifiers(
        text,
        set(dep["STAND_mvt"].dropna().astype(str))
        | set(rk["STAND_mvt"].dropna().astype(str))
        | set(matrix["stand_id"].dropna().astype(str)),
        set(matrix["designator"].astype(str)) | {a.icao for a in AIRPORTS},
    )
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
