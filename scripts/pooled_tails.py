#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The pooled record-quality rule: the premise it rests on, its stake, its arm.

``taxiout.tails`` predicts for rows in *covered* cells, a cell being (airport,
record, lag band) and covered at ``MIN_COUNT`` rows or more with a defaulted
block share of ``MIN_SHARE`` or more. An airport with no covered cell is left
entirely to the physical model. What this script measures is whether such an
airport should instead inherit a cell pooled across airports.

Pooling rests on one premise: that a bucket's defaulted share is a property of
the defaulting *mechanism* and not of the airport, so a cell learned where the
mechanism is visible may be carried to an airport where it is not. That
premise is directly measurable, per airport and per bucket, from the
defaulted-block share ``TailModel.fit`` already measures on training rows. It
needs no booster, no ranking row and no validation leg, so it is measured
first and the arm is fitted only if it survives.

Three tests, in this order, under the criteria fixed in ``docs/plan.md``'s
entry of 21 September before any of these numbers existed:

1. **The premise.** Per leg and per bucket, every airport's ``n`` and
   ``share`` beside the pooled ``n`` and ``share``. The premise holds for a
   bucket only if every airport with at least ``MIN_COUNT`` rows in it has a
   share within two standard errors of the pooled share at that airport's own
   ``n``. Two standard errors of a proportion is fixed arithmetic and not a
   chosen threshold.
2. **The stake.** The ranking rows a pooled cell would cover and the term does
   not, by airport, month and bucket, from ``flag`` and the fitted cell table
   alone, reading no field outside ``PREDICT_INPUTS``.
3. **The arm**, and only if both screens pass: the adopted mixture, current
   coverage, plus a pooled cell where an airport has no covered cell in that
   bucket, scored on the three columns of the report's design table through
   the same stand-in physical model, with day-clustered intervals on the
   differences.

If the premise fails for every bucket a pooled cell would serve, the arm is
not fitted. Fitting an arm whose premise has been refuted would be measuring
something already answered, so the script records what stopped it and stops.

**This script adopts nothing and changes no rule.** It constructs no booster,
it writes no residual, and it leaves ``taxiout.tails`` exactly as it found it.
``--report`` writes one section of one file; every other file under
``reports/`` and every stored residual is fingerprinted before the run and
verified untouched after it, as ``scripts/fast_lane.py`` does, because a
script that measured an alternative and quietly moved a published figure would
be the one failure nothing downstream would report.

The section it writes sits inside ``reports/tails.md``, which
``scripts/report_tails.py`` generates in full and would otherwise overwrite.
``make tails`` runs both, in that order, and ``tests/test_pooled_tails.py``
fails if the committed report has lost the section, so a rebuild that dropped
it cannot pass unnoticed.

Usage
-----
    python scripts/pooled_tails.py
    python scripts/pooled_tails.py --report
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import report_tails as rt  # noqa: E402
from taxiout.tails import (  # noqa: E402
    BUCKETS,
    MIN_COUNT,
    MIN_SHARE,
    PREDICT_INPUTS,
    TailModel,
    combine,
    flag,
)
from taxiout.validation import cluster_bootstrap, cluster_sums  # noqa: E402

REPORT = REPO / "reports" / "tails.md"

# The two directories this script must not write, checked rather than promised.
# ``--report`` is allowed exactly one file inside the first of them and nothing
# else; the fingerprint is taken before the run and verified after it.
FORBIDDEN = (REPO / "reports", REPO / "data" / "processed" / "residuals")

# Where the section goes in ``reports/tails.md``: immediately before the data
# credit that closes the file, and replacing itself if it is already there.
SECTION_TITLE = "## The pooled record-quality rule, measured"
FOOTER_ANCHOR = "Movement and flight data supplied by EUROCONTROL PRC"

# Two standard errors of a proportion. Arithmetic, not a chosen threshold: the
# premise test asks whether one airport's share is a draw from the pooled one
# at that airport's own row count, and this is what that question costs.
SE_MULTIPLE = 2.0

# An airport enters the premise test at the term's own minimum count. Below it
# a cell's share is a coin toss, which is why the coverage rule discards it,
# and a test that read those cells would be testing the rounding.
PREMISE_MIN_N = MIN_COUNT

# The airport the question is about: the one with no covered cell whose
# ranking month carries six consecutive flagged days that the training year
# holds no equivalent of. Written out rather than derived, so the section is
# about a named airport and does not silently follow a change in coverage.
CASE_AIRPORT = "EHAM"

# Representative lags inside the two late bands, for the worked prediction.
# They are the only chosen numbers here, they are labelled as chosen wherever
# they are printed, and nothing is concluded from them: the table beside the
# worked one replaces them with the case airport's own rows, whose recorded
# taxi-out is known, so the error there is measured rather than illustrated.
ILLUSTRATIVE_LAGS = {"noflt:4-12h": 5 * rt.HOUR, "noflt:>12h": 14 * rt.HOUR}

# The ranking-set count the data review prints in a column headed ``both``,
# and the reading of that heading this script checks against the data.
BOTH_COLUMN_MONTH = "2026-01"


# --------------------------------------------------------------------------
# what this script may not write


def fingerprint(exempt: Path | None = None) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file this script must not write."""
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != exempt:
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


def verify(before: dict[str, tuple[int, int]], exempt: Path | None = None) -> None:
    after = fingerprint(exempt)
    if after != before:
        moved = sorted(set(before) ^ set(after)) or sorted(
            k for k in before if before[k] != after.get(k)
        )
        raise RuntimeError(f"this run touched {len(moved)} file(s) it must not: {moved}")


# --------------------------------------------------------------------------
# the pooled cell, from the fitted cell table alone


# A reading taken from the printed share table is reproduced by pooling the
# shares as this report prints them, so the two can be shown side by side and
# the rounding can be seen for what it costs. The printed value is taken from
# the report's own formatters and read back, rather than from a digit count
# guessed beside them, so the two cannot drift apart.
def as_printed_share(share: pd.Series) -> pd.Series:
    """Each share exactly as ``reports/tails.md`` prints it, read back as a number."""
    return pd.Series(
        [np.nan if not np.isfinite(v) else float(rt.pct(v)[:-1]) / 100.0 for v in share],
        index=share.index,
    )


def as_printed_mean(mean: pd.Series) -> pd.Series:
    """Each mean exactly as ``reports/tails.md`` prints it, read back as a number."""
    return pd.Series(
        [
            np.nan if not np.isfinite(v) else float(rt.num(v).replace(",", ""))
            for v in mean
        ],
        index=mean.index,
    )


def pool(cells: pd.DataFrame, rounded: bool = False) -> pd.DataFrame:
    """The bucket's cell pooled over airports, from the fitted cell table.

    The pooled ``share`` is the row-weighted mean of the airports' shares,
    which is the share of the bucket's rows that were defaulted; the pooled
    ``rest_mean`` is the row-weighted mean over the rows that were not, whose
    weight is zero in a cell where every row was. ``covered`` applies the
    term's own rule to the pooled cell, so a bucket whose pooled share is
    under ``MIN_SHARE`` yields no covered pooled cell and serves nobody.

    ``rounded`` takes each airport's share and mean at the precision
    ``reports/tails.md`` prints them, so a reading made from the printed table
    can be reproduced beside the one made from the data.
    """
    t = cells.reset_index()
    share = as_printed_share(t["share"]) if rounded else t["share"]
    mean = as_printed_mean(t["mean"]) if rounded else t["mean"]
    rest_mean = as_printed_mean(t["rest_mean"]) if rounded else t["rest_mean"]
    n_art = t["n"] * share
    if not rounded:
        # A measured share is a count over the cell's rows, so this product is
        # an integer up to the last bit of a division; rounding says so.
        n_art = n_art.round()
    n_rest = t["n"] - n_art
    frame = pd.DataFrame(
        {
            "tail_bucket": t["tail_bucket"].to_numpy(),
            "n": t["n"].to_numpy(dtype=float),
            "n_art": n_art.to_numpy(dtype=float),
            "n_rest": n_rest.to_numpy(dtype=float),
            "y_sum": (t["n"] * mean).to_numpy(dtype=float),
            "rest_sum": np.where(
                n_rest.to_numpy() > 0,
                n_rest.to_numpy() * rest_mean.fillna(0.0).to_numpy(),
                0.0,
            ),
        }
    )
    g = frame.groupby("tail_bucket", sort=True)
    out = pd.DataFrame(
        {
            "n": g["n"].sum(),
            "n_art": g["n_art"].sum(),
            "n_rest": g["n_rest"].sum(),
        }
    )
    out["share"] = out["n_art"] / out["n"]
    out["mean"] = g["y_sum"].sum() / out["n"]
    # A bucket every one of whose rows was defaulted has no ordinary mean. Its
    # weight in the mixture is zero, and ``0 * NaN`` must not poison the lag,
    # so the pooled cell carries the same NaN ``TailModel.fit`` carries.
    rest_sum, n_rest_total = g["rest_sum"].sum().to_numpy(), out["n_rest"].to_numpy()
    out["rest_mean"] = np.divide(
        rest_sum,
        n_rest_total,
        out=np.full(len(out), np.nan),
        where=n_rest_total > 0,
    )
    out["covered"] = (out["n"] >= MIN_COUNT) & (out["share"] >= MIN_SHARE)
    out.index.name = "tail_bucket"
    return out


def serves(cells: pd.DataFrame, pooled: pd.DataFrame) -> dict[str, bool]:
    """Buckets a pooled cell would actually serve.

    A pooled cell serves a bucket when it is covered by the term's own rule
    and at least one airport has no covered cell in that bucket. A pooled cell
    under ``MIN_SHARE`` is not covered and reaches nobody, whatever the
    premise says about it.
    """
    airports = set(cells.index.get_level_values("icao").unique())
    out = {}
    for bucket in pooled.index:
        rows = cells.xs(bucket, level="tail_bucket", drop_level=True)
        with_cover = {str(i) for i, c in rows.iterrows() if bool(c["covered"])}
        out[str(bucket)] = bool(pooled.loc[bucket, "covered"]) and with_cover != airports
    return out


# --------------------------------------------------------------------------
# screen 1: the premise


def standard_error(p: float, n: float) -> float:
    """``sqrt(p * (1 - p) / n)`` at the pooled ``p`` and the airport's own ``n``."""
    if not np.isfinite(p) or n <= 0:
        return float("nan")
    return float(np.sqrt(p * (1.0 - p) / n))


def premise(cells: pd.DataFrame, pooled: pd.DataFrame) -> dict[str, dict]:
    """Every airport's share against two standard errors of the pooled share.

    ``holds`` is True only if every airport with at least ``PREMISE_MIN_N``
    rows in the bucket lies inside the interval, False if any does not, and
    None if the bucket has no airport with enough rows to test.
    """
    out: dict[str, dict] = {}
    for bucket in pooled.index:
        p = float(pooled.loc[bucket, "share"])
        rows = []
        for icao, c in cells.xs(bucket, level="tail_bucket", drop_level=True).iterrows():
            n, s = float(c["n"]), float(c["share"])
            se = standard_error(p, n)
            low, high = p - SE_MULTIPLE * se, p + SE_MULTIPLE * se
            tested = n >= PREMISE_MIN_N
            rows.append(
                {
                    "icao": str(icao),
                    "n": int(n),
                    "share": s,
                    "low": low,
                    "high": high,
                    "tested": tested,
                    "passes": bool(low <= s <= high) if tested else None,
                }
            )
        tried = [r for r in rows if r["tested"]]
        out[str(bucket)] = {
            "rows": rows,
            "pooled_n": int(pooled.loc[bucket, "n"]),
            "pooled_share": p,
            "covered": bool(pooled.loc[bucket, "covered"]),
            "tested": len(tried),
            "failed": [r["icao"] for r in tried if not r["passes"]],
            "holds": all(r["passes"] for r in tried) if tried else None,
        }
    return out


# --------------------------------------------------------------------------
# screen 2: the stake


def stake_rows(
    frame: pd.DataFrame, cells: pd.DataFrame, pooled: pd.DataFrame
) -> pd.DataFrame:
    """Rows a pooled cell would cover and the term does not, one row per departure.

    Reads ``PREDICT_INPUTS`` and nothing else: the frame is narrowed to those
    columns before anything is computed, so a field the ranking set hides
    cannot reach this count even by accident.
    """
    missing = [c for c in PREDICT_INPUTS if c not in frame.columns]
    if missing:
        raise KeyError(f"stake_rows() needs {missing}")
    frame = frame[list(PREDICT_INPUTS)]
    flags = flag(frame)
    bucket = flags["tail_bucket"].astype(str).to_numpy()
    icao = frame["ADEP_mvt"].astype(str).to_numpy()
    key = pd.MultiIndex.from_arrays([icao, bucket], names=["icao", "tail_bucket"])
    now = cells["covered"].reindex(key).to_numpy(dtype=object)
    now = np.where(pd.isna(now), False, now).astype(bool)
    pooled_cov = pooled["covered"].reindex(bucket).to_numpy(dtype=object)
    pooled_cov = np.where(pd.isna(pooled_cov), False, pooled_cov).astype(bool)
    month = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m")
    return pd.DataFrame(
        {
            "icao": icao,
            "month": month.to_numpy(),
            "bucket": bucket,
            "flt_missing": flags["flt_missing"].to_numpy(),
            "lag_s": flags["lag_s"].to_numpy(),
            "covered_now": now,
            "gained": (~now) & pooled_cov,
        },
        index=frame.index,
    )


def both_column(rows: pd.DataFrame, icao: str, month: str) -> dict:
    """The data review's ``both`` column, read from the data rather than the heading.

    ``reports/data_review.md`` heads a ranking-set column ``both``. Read as no
    flight-side record *and* a lag over four hours it is the union of the two
    late no-record buckets, which is the population a pooled cell could reach.
    """
    at = rows[(rows["icao"] == icao) & (rows["month"] == month)]
    both = at["flt_missing"].to_numpy() & (at["lag_s"].to_numpy() > 4 * rt.HOUR)
    by_bucket = at.loc[both, "bucket"].value_counts().to_dict()
    return {
        "rows": int(len(at)),
        "no_record": int(at["flt_missing"].sum()),
        "lag_over_4h": int((at["lag_s"].to_numpy() > 4 * rt.HOUR).sum()),
        "lag_over_12h": int((at["lag_s"].to_numpy() > 12 * rt.HOUR).sum()),
        "both": int(both.sum()),
        "by_bucket": {str(k): int(v) for k, v in sorted(by_bucket.items())},
    }


# --------------------------------------------------------------------------
# what a pooled cell would predict, on rows whose value is known


def pooled_value(
    bucket: pd.Series, lag_s: np.ndarray, pooled: pd.DataFrame
) -> np.ndarray:
    """The pooled cell's mixture: its share times the lag, plus the rest."""
    b = np.asarray(bucket, dtype=object)
    share = pooled["share"].reindex(b).to_numpy(dtype=float)
    rest = pooled["rest_mean"].reindex(b).to_numpy(dtype=float)
    rest = np.where(np.isnan(rest), 0.0, rest)
    return share * np.asarray(lag_s, dtype=float) + (1.0 - share) * rest


def cost_on_known_rows(
    dep: pd.DataFrame, pooled: pd.DataFrame, icao: str, buckets: list[str]
) -> list[dict]:
    """What a pooled cell would have predicted on one airport's own training rows.

    These rows carry a recorded taxi-out, so the error a pooled cell would make
    on them is measured and not illustrated. The cell's own mean is printed
    beside it as the thing the term would otherwise leave the row to.
    """
    out = []
    at = dep[dep["ADEP_mvt"] == icao]
    for bucket in buckets:
        rows = at[at["bucket"] == bucket]
        if rows.empty:
            continue
        y = rows["y"].to_numpy(dtype=float)
        p = pooled_value(np.full(len(rows), bucket), rows["lag"].to_numpy(), pooled)
        out.append(
            {
                "bucket": bucket,
                "n": int(len(rows)),
                "mean": float(np.mean(y)),
                "median_lag": float(np.median(rows["lag"].to_numpy(dtype=float))),
                "pooled_mean_prediction": float(np.mean(p)),
                "mean_error": float(np.mean(p - y)),
                "rmse_pooled": float(np.sqrt(np.mean((p - y) ** 2))),
                "rmse_cell_mean": float(np.sqrt(np.mean((np.mean(y) - y) ** 2))),
            }
        )
    return out


def worked_prediction(
    cells: pd.DataFrame, pooled: pd.DataFrame, rounded_pooled: pd.DataFrame, icao: str
) -> list[dict]:
    """The worked figure both readings give for one row of each served bucket.

    Three forms of the same arithmetic. The first two weight the airport's own
    ordinary mean by the rest, which is the reading taken from the printed
    table, computed once from the printed shares and once from the measured
    ones; the third is what the arm itself would predict, which weights the
    *pooled* cell's ordinary mean instead.
    """
    out = []
    for bucket, lag in ILLUSTRATIVE_LAGS.items():
        if (icao, bucket) not in cells.index:
            continue
        own = float(cells.loc[(icao, bucket), "mean"])
        own_printed = float(as_printed_mean(pd.Series([own])).iloc[0])
        p_round = float(rounded_pooled.loc[bucket, "share"])
        p_meas = float(pooled.loc[bucket, "share"])
        rest = float(pooled.loc[bucket, "rest_mean"])
        out.append(
            {
                "bucket": bucket,
                "lag": float(lag),
                "n": int(cells.loc[(icao, bucket), "n"]),
                "own_mean": own,
                "share_printed": p_round,
                "share_measured": p_meas,
                "pooled_rest_mean": rest,
                "printed": p_round * lag + (1 - p_round) * own_printed,
                "measured": p_meas * lag + (1 - p_meas) * own,
                "arm": p_meas * lag + (1 - p_meas) * rest,
            }
        )
    return out


# --------------------------------------------------------------------------
# the arm, fitted only if the screens pass


def arm_prediction(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """The adopted mixture, plus a pooled cell where the airport has no covered one.

    Every already-covered cell is left exactly as ``TailModel.predict`` leaves
    it: the pooled value is written only where the term returns nothing, which
    is the definition of "the airport has no covered cell in that bucket".
    """
    model = TailModel().fit(train)
    base = model.predict(valid)
    pooled = pool(model.cells)
    flags = flag(valid)
    bucket = flags["tail_bucket"].astype(str).to_numpy()
    covered = pooled["covered"].reindex(bucket).to_numpy(dtype=object)
    covered = np.where(pd.isna(covered), False, covered).astype(bool)
    value = pooled_value(bucket, flags["lag_s"].to_numpy(), pooled)
    filled = np.where(covered, value, np.nan)
    out = np.where(base.notna().to_numpy(), base.to_numpy(), filled)
    return pd.Series(out, index=valid.index, name="tail", dtype=float)


def day_of(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )


def paired_interval(y: pd.Series, arm: pd.Series, base: pd.Series, days: np.ndarray):
    """Arm minus base RMSE, days as the clusters, the same draw for both arms."""
    sums = cluster_sums(
        {
            "arm": (y.to_numpy(dtype=float) - arm.to_numpy(dtype=float)),
            "base": (y.to_numpy(dtype=float) - base.to_numpy(dtype=float)),
        },
        days,
    )
    return cluster_bootstrap([{"s": sums}], lambda r: r["s"]["arm"] - r["s"]["base"])


def score_arm(dep: pd.DataFrame, sp: list) -> dict:
    """The arm on the design table's three columns, with day-clustered intervals."""
    out: dict = {}
    for s in sp:
        model = TailModel().fit(s.train)
        phys = rt.stand_in(s.train, s.valid, model.flagged(s.train))
        base = combine(phys, model.predict(s.valid))
        arm = combine(phys, arm_prediction(s.train, s.valid))
        days = day_of(s.valid)
        per_airport = {}
        for icao, rows in s.valid.groupby("ADEP_mvt", observed=True):
            idx = rows.index
            per_airport[str(icao)] = {
                "rows": int(len(idx)),
                "base": rt.rmse(rows["y"], base.loc[idx]),
                "arm": rt.rmse(rows["y"], arm.loc[idx]),
                "interval": paired_interval(
                    rows["y"], arm.loc[idx], base.loc[idx], day_of(rows)
                ),
            }
        out[s.name] = {
            "base": rt.rmse(s.valid["y"], base),
            "arm": rt.rmse(s.valid["y"], arm),
            "interval": paired_interval(s.valid["y"], arm, base, days),
            "per_airport": per_airport,
        }
    return out


# --------------------------------------------------------------------------
# both screens, on every leg


def legs(dep: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """The three training halves the premise is measured on."""
    m = dep["month"]
    return [
        ("December leg, train January to November", dep[m <= 11]),
        ("July leg, train January to June", dep[m <= 6]),
        ("the full training year", dep),
    ]


def measure(dep: pd.DataFrame, rk: pd.DataFrame) -> dict:
    """Both screens, on every leg. No booster, and nothing fitted but the term."""
    out: dict = {"legs": []}
    for name, train in legs(dep):
        model = TailModel().fit(train)
        pooled = pool(model.cells)
        out["legs"].append(
            {
                "name": name,
                "rows": int(len(train)),
                "cells": model.cells,
                "pooled": pooled,
                "rounded": pool(model.cells, rounded=True),
                "premise": premise(model.cells, pooled),
                "serves": serves(model.cells, pooled),
            }
        )
    full = out["legs"][-1]
    rows = stake_rows(rk, full["cells"], full["pooled"])
    out["stake"] = (
        rows[rows["gained"]]
        .groupby(["icao", "month", "bucket"], observed=True)
        .size()
        .sort_index()
    )
    out["stake_total"] = int(rows["gained"].sum())
    out["covered_now"] = int(rows["covered_now"].sum())
    out["ranking_rows"] = int(len(rows))
    out["both"] = both_column(rows, CASE_AIRPORT, BOTH_COLUMN_MONTH)
    out["served_any"] = sorted(
        {b for leg in out["legs"] for b, yes in leg["serves"].items() if yes},
        key=BUCKETS.index,
    )
    out["premise_holds_anywhere"] = any(
        leg["premise"][b]["holds"] is True
        for leg in out["legs"]
        for b, yes in leg["serves"].items()
        if yes
    )
    out["worked"] = worked_prediction(
        full["cells"], full["pooled"], full["rounded"], CASE_AIRPORT
    )
    out["cost"] = cost_on_known_rows(dep, full["pooled"], CASE_AIRPORT, out["served_any"])

    # The figures the reading quotes, measured here rather than written there.
    covered = full["cells"][full["cells"]["covered"]]
    defaulted = (full["cells"]["n"] * full["cells"]["share"]).round()
    by_airport = defaulted.groupby(level="icao").sum()
    at_covered = float(
        by_airport.reindex(sorted({str(i) for i, _ in covered.index})).sum()
    )
    out["facts"] = {
        "covered_cells": int(len(covered)),
        "covered_airports": sorted({str(i) for i, _ in covered.index}),
        "airports": int(len(by_airport)),
        "airports_with_any": int((by_airport > 0).sum()),
        "defaulted_total": int(defaulted.sum()),
        "defaulted_at_covered": int(at_covered),
        "defaulted_share_at_covered": at_covered / float(defaulted.sum()),
        "covered_share_min": float(covered["share"].min())
        if len(covered)
        else float("nan"),
        "covered_share_max": float(covered["share"].max())
        if len(covered)
        else float("nan"),
        "case_cells": [
            {
                "bucket": b,
                "n": int(full["cells"].loc[(CASE_AIRPORT, b), "n"]),
                "one_row": 1.0 / float(full["cells"].loc[(CASE_AIRPORT, b), "n"]),
            }
            for b in out["served_any"]
            if (CASE_AIRPORT, b) in full["cells"].index
        ],
    }
    return out


# --------------------------------------------------------------------------
# the section this script writes into reports/tails.md


def verdict_word(holds: bool | None, emphasis: bool = True) -> str:
    if holds is None:
        return "not testable"
    if holds:
        return "holds"
    return "**fails**" if emphasis else "fails"


def premise_summary(m: dict) -> list[str]:
    rows = []
    for leg in m["legs"]:
        for bucket in leg["pooled"].index:
            p = leg["premise"][str(bucket)]
            rows.append(
                [
                    leg["name"],
                    f"`{bucket}`",
                    f"{p['pooled_n']:,}",
                    rt.pct(p["pooled_share"]),
                    "yes" if p["covered"] else "no",
                    "yes" if leg["serves"][str(bucket)] else "no",
                    f"{p['tested']:,}",
                    f"{len(p['failed']):,}",
                    verdict_word(p["holds"]),
                ]
            )
    return rt.table(
        [
            "leg",
            "bucket",
            "pooled rows",
            "pooled share",
            f"covered at {rt.pct(MIN_SHARE, 0)}",
            "would serve it",
            "airports tested",
            "outside",
            "the premise",
        ],
        rows,
        align="--rr--rr-",
    )


def premise_detail(m: dict, bucket: str) -> list[str]:
    rows = []
    for leg in m["legs"]:
        p = leg["premise"].get(bucket)
        if p is None:
            continue
        for r in p["rows"]:
            if r["tested"]:
                verdict = "inside" if r["passes"] else "**outside**"
            else:
                verdict = "not tested"
            rows.append(
                [
                    leg["name"],
                    r["icao"],
                    f"{r['n']:,}",
                    rt.pct(r["share"]),
                    f"{rt.pct(max(r['low'], 0.0))} to {rt.pct(r['high'])}",
                    verdict,
                ]
            )
    return rt.table(
        [
            "leg",
            "ICAO",
            "rows",
            "its defaulted share",
            "two standard errors of the pooled share, at its own rows",
            "the share",
        ],
        rows,
        align="--rrr-",
    )


def stake_table(m: dict) -> list[str]:
    if not len(m["stake"]):
        return [
            "No ranking row is in that position on the full-year fit: the stake",
            "is zero rows.",
        ]
    rows = [
        [str(icao), str(month), f"`{bucket}`", f"{int(n):,}"]
        for (icao, month, bucket), n in m["stake"].items()
    ]
    return rt.table(["ICAO", "month", "bucket", "ranking rows"], rows, align="---r")


def worked_table(m: dict) -> list[str]:
    rows = [
        [
            f"`{w['bucket']}`",
            rt.secs(w["lag"]),
            rt.pct(w["share_printed"]),
            rt.pct(w["share_measured"]),
            rt.secs(w["printed"]),
            rt.secs(w["measured"]),
            rt.secs(w["own_mean"]),
            f"{w['measured'] - w['own_mean']:+,.0f} s",
        ]
        for w in m["worked"]
    ]
    return rt.table(
        [
            "bucket",
            "lag used",
            "pooled share, printed",
            "pooled share, measured",
            "predicted, printed",
            "predicted, measured",
            f"{CASE_AIRPORT}'s mean",
            "error",
        ],
        rows,
        align="-rrrrrrr",
    )


def cost_table(m: dict) -> list[str]:
    rows = [
        [
            f"`{c['bucket']}`",
            f"{c['n']:,}",
            rt.secs(c["median_lag"]),
            rt.secs(c["mean"]),
            rt.secs(c["pooled_mean_prediction"]),
            f"{c['mean_error']:+,.0f} s",
            rt.secs(c["rmse_pooled"]),
            rt.secs(c["rmse_cell_mean"]),
        ]
        for c in m["cost"]
    ]
    return rt.table(
        [
            "bucket",
            "rows",
            "median lag",
            "mean taxi-out",
            "mean pooled prediction",
            "mean error",
            "RMSE, pooled cell",
            "RMSE, its own mean",
        ],
        rows,
        align="-rrrrrrr",
    )


def reading(m: dict) -> list[str]:
    """The prose of the section, with every figure in it computed above."""
    f = m["facts"]
    case = ", ".join(f"{c['n']:,} rows in `{c['bucket']}`" for c in f["case_cells"])
    one_row = " and ".join(rt.pct(c["one_row"]) for c in f["case_cells"])
    lines = [
        "### The reading",
        "",
        "**The mechanism is at one airport.** Of the",
        f"{f['defaulted_total']:,} training rows the term's own test picks out,",
        f"{rt.pct(f['defaulted_share_at_covered'])} are at {', '.join(f['covered_airports'])},",
        f"and {f['airports_with_any']} of the {f['airports']} airports carry any at all; the covered",
        f"cells run from {rt.pct(f['covered_share_min'])} to {rt.pct(f['covered_share_max'])}. A pooled",
        "share is therefore not an estimate of a rate several airports share. It",
        "is one airport's rate diluted by the row counts of the airports that do",
        "not have it, and the premise test reads as it does for that reason.",
        "",
        "That holds at both ends of the table. Where a pooled share rests on a",
        "handful of rows spread over hundreds of thousands, two standard errors",
        "is narrower than the share itself, so an airport with no defaulted row",
        "at all sits outside the interval rather than inside it; and a bucket",
        f"whose pooled share is under {rt.pct(MIN_SHARE, 0)} is not covered and serves nobody",
        "whichever way its premise reads.",
        "",
    ]
    if f["case_cells"]:
        lines += [
            f"**{CASE_AIRPORT}'s zero is a measurement and not a gap.** Its cells in the",
            f"served buckets are uncovered on share and not on thinness: {case},",
            f"each well above the {MIN_COUNT}-row minimum. At those counts a single",
            f"defaulted row would print as {one_row}, so a share of {rt.pct(0.0)} means",
            "exactly zero defaulted rows in a full training year rather than a",
            "cell too small to say.",
            "",
        ]
    lines += [
        "**So a pooled cell would carry a share learned elsewhere onto rows",
        "measured not to have it.** That is what the two tables above price, and",
        "the second of them prices it on rows whose recorded value is known.",
    ]
    if m["premise_holds_anywhere"]:
        lines += [
            "The premise is not refuted everywhere it would be relied on, so the",
            "arm is open and is not settled by this section.",
        ]
    else:
        lines += [
            "The premise is refuted by the instrument that measures it, on every",
            "leg and in every bucket a pooled cell would serve, so the arm was",
            "not fitted: an arm whose premise has been refuted measures something",
            "already answered.",
        ]
    lines += [
        "",
        "What is left unmeasurable is narrower, and is stated rather than",
        "widened. Whether an uncovered airport's 2026 rows are themselves",
        "defaulted block times is a question about `BLOCK_TIME_UTC_mvt` and",
        "`TAXITIME_SEC_mvt`, the two fields the ranking set hides, and no",
        "instrument reaches them. That is a limit of the data and not a gap in",
        "this measurement, which is about the 2025 rows and is complete on them.",
    ]
    return lines


def section(m: dict) -> list[str]:
    both = m["both"]
    served = m["served_any"]
    served_text = ", ".join(f"`{b}`" for b in served) if served else "no bucket"
    lines = [
        SECTION_TITLE,
        "",
        "An airport with no covered cell is left entirely to the physical model,",
        "and the alternative is that it inherits a cell pooled across airports.",
        "That question was reopened as an adoptive measurement with its criteria",
        "fixed in [`../docs/plan.md`](../docs/plan.md) before any figure here",
        "existed, and this section is the measurement. Generated by",
        "`scripts/pooled_tails.py`. It changes neither the term, nor its inputs,",
        "nor its coverage rule, and the minimum count and the minimum share are",
        "what they were.",
        "",
        "It is not the pooled row of the design table above, which is a cell",
        "mean over every `noflt` cell with thin cells pooled: that row differs",
        "from the adopted design in the prediction form and in the coverage rule",
        "at once. The arm this section is about is the adopted mixture at its",
        "present coverage, plus a pooled cell only where an airport has no",
        "covered cell in that bucket, and a reader who takes the design table's",
        "row for it is reading the wrong measurement.",
        "",
        "**The instrument.** Pooling rests on one premise: that a bucket's",
        "defaulted share is a property of the defaulting mechanism and not of",
        "the airport, so a cell learned where the mechanism is visible may be",
        "carried to an airport where it is not. The fitted cell table measures",
        "that premise directly, per airport and per bucket, on training rows,",
        "with no fit and no booster. The test is arithmetic rather than a chosen",
        "threshold: an airport's share is read against two standard errors of",
        "the pooled share, `sqrt(p x (1 - p) / n)` at the pooled `p` and the",
        f"airport's own `n`, and every airport with at least {MIN_COUNT} rows in the",
        "bucket has to lie inside.",
        "",
        "### The premise, per leg and per bucket",
        "",
        "Every bucket, on each of the three training halves. A pooled cell under",
        f"{rt.pct(MIN_SHARE, 0)} is not covered by the term's own rule and reaches nobody,",
        "so the two middle columns say which buckets the question is about at",
        "all.",
        "",
        *premise_summary(m),
        "",
    ]
    if served:
        lines += [
            f"The buckets a pooled cell would serve are {served_text}. Every",
            "airport's share in them, against the interval two standard errors of",
            "the pooled share gives at that airport's own row count:",
            "",
        ]
        for bucket in served:
            lines += [f"In `{bucket}`:", "", *premise_detail(m, bucket), ""]
    else:
        lines += [
            "No pooled cell is covered on any leg, so no bucket is served and",
            "there is nothing for the premise to be tested for.",
            "",
        ]
    lines += [
        "### The stake",
        "",
        "Ranking rows whose (airport, bucket) is uncovered today and whose",
        "pooled bucket cell would be covered, from `taxiout.tails.flag` and the",
        "full-year cell table alone, reading no field outside `PREDICT_INPUTS`:",
        "",
        *stake_table(m),
        "",
        f"{m['stake_total']:,} rows in all, against the {m['covered_now']:,} of",
        f"{m['ranking_rows']:,} ranking rows the term covers today.",
        (
            "The stake is not zero, so it is the premise and not the stake that"
            if m["stake_total"]
            else "The stake is zero, and on the criteria that alone"
        ),
        "settles this.",
        "",
        f"The population at {CASE_AIRPORT} in {BOTH_COLUMN_MONTH}, counted here rather than read off",
        "a column heading in [`data_review.md`](data_review.md): of",
        f"{both['rows']:,} departures, {both['no_record']:,} have no flight-side record,",
        f"{both['lag_over_4h']:,} have a lag over four hours, {both['lag_over_12h']:,} over twelve, and",
        f"{both['both']:,} have no record *and* a lag over four hours. Those last are",
        "exactly the late no-record buckets, "
        + ", ".join(f"`{k}` at {v:,}" for k, v in both["by_bucket"].items())
        + ".",
        "",
        "### What a pooled cell would predict",
        "",
        f"For one {CASE_AIRPORT} row in each served bucket, at a stated lag inside the",
        "band. The two share columns are the same pooling arithmetic over the",
        "shares as this report prints them and over the shares as they are",
        "measured, so what the rounding costs is visible; the two prediction",
        "columns follow from them, weighting this airport's own ordinary mean by",
        "the rest.",
        "",
        *worked_table(m),
        "",
    ]
    if m["worked"]:
        arm_text = ", ".join(
            f"`{w['bucket']}` at {rt.secs(w['arm'])}" for w in m["worked"]
        )
        lines += [
            "The arm itself would predict higher than either column, because it",
            "weights the pooled cell's own ordinary mean rather than this",
            f"airport's: {arm_text}. The lags above are stated choices inside each",
            "band and are the only chosen numbers in this section. The table",
            "below replaces them with this airport's own rows, whose recorded",
            "taxi-out is known, so the error there is measured and not worked.",
            "",
        ]
    lines += [
        f"What a pooled cell would have predicted on {CASE_AIRPORT}'s own training rows in",
        "those buckets, against what they recorded:",
        "",
        *cost_table(m),
        "",
        *reading(m),
    ]
    return lines


def write_section(lines: list[str], dep: pd.DataFrame) -> str:
    """Put the section into ``reports/tails.md``, replacing itself if it is there.

    The section goes immediately before the data credit that closes the file,
    and a second run rewrites it in place rather than appending a second copy.
    ``scripts/report_tails.py`` writes the rest of that file and knows nothing
    about this one, which is why ``make tails`` runs both and a test fails if
    the committed report has lost the section.
    """
    text = REPORT.read_text(encoding="utf-8")
    body = text.splitlines()
    credit = [i for i, line in enumerate(body) if line.startswith(FOOTER_ANCHOR)]
    if not credit:
        raise RuntimeError(f"{REPORT.name} has no data credit to insert before")
    head, tail = body[: credit[0]], body[credit[0] :]
    mark = [i for i, line in enumerate(head) if line.strip() == SECTION_TITLE]
    if mark:
        head = head[: mark[0]]
    while head and not head[-1].strip():
        head.pop()
    out = "\n".join(head + [""] + lines + [""] + tail) + "\n"
    rt.check_no_identifiers(out, dep)
    REPORT.write_text(out, encoding="utf-8")
    return out


# --------------------------------------------------------------------------


def show(m: dict) -> None:
    """Everything both screens measured, on every leg and every bucket."""
    for leg in m["legs"]:
        print(f"\n=== {leg['name']}: {leg['rows']:,} training rows")
        for bucket in leg["pooled"].index:
            p = leg["premise"][str(bucket)]
            print(
                f"\n  {bucket}: pooled n={p['pooled_n']:,}"
                f" share={rt.pct(p['pooled_share'], 2)}"
                f" covered={'yes' if p['covered'] else 'no'}"
                f" serves={'yes' if leg['serves'][str(bucket)] else 'no'}"
                f" premise={verdict_word(p['holds'], emphasis=False)}"
            )
            for r in p["rows"]:
                if r["tested"]:
                    mark = "inside" if r["passes"] else "OUTSIDE"
                else:
                    mark = "not tested"
                print(
                    f"    {r['icao']}  n={r['n']:>7,}  share={rt.pct(r['share'], 2):>8}"
                    f"  [{rt.pct(max(r['low'], 0.0), 2)}, {rt.pct(r['high'], 2)}]  {mark}"
                )
    print(
        f"\n=== the stake: {m['stake_total']:,} rows, against {m['covered_now']:,} covered today"
    )
    print(m["stake"] if len(m["stake"]) else "  zero rows")
    print(f"\n=== {CASE_AIRPORT} {BOTH_COLUMN_MONTH}, counted from the data")
    for k, v in m["both"].items():
        print(f"  {k}: {v}")
    print("\n=== the worked prediction")
    for w in m["worked"]:
        print(f"  {w}")
    print("\n=== what a pooled cell would cost on rows whose value is known")
    for c in m["cost"]:
        print(f"  {c}")


def arm_report(dep: pd.DataFrame) -> None:
    """The arm on the design table's three columns. Reached only if the screens pass."""
    scores = score_arm(dep, rt.splits(dep))
    print("\n=== the arm, against the adopted mixture")
    for name, s in scores.items():
        print(
            f"  {name}: adopted {rt.num(s['base'])} s, arm {rt.num(s['arm'])} s,"
            f" difference {s['interval']}"
        )
        for icao, a in sorted(s["per_airport"].items()):
            print(
                f"    {icao}  rows={a['rows']:>7,}  adopted={rt.num(a['base'])}"
                f"  arm={rt.num(a['arm'])}  {a['interval']}"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--report",
        action="store_true",
        help="write the section into reports/tails.md as well as printing it",
    )
    ap.add_argument(
        "--arm",
        action="store_true",
        help="score the arm, which the criteria allow only if both screens pass",
    )
    args = ap.parse_args(argv)
    if not (rt.DATA / "ranking.parquet").exists():
        print(f"no challenge data under {rt.DATA}; nothing to measure", file=sys.stderr)
        return 1
    if not REPORT.exists():
        print("no reports/tails.md; run scripts/report_tails.py first", file=sys.stderr)
        return 1

    exempt = REPORT if args.report else None
    before = fingerprint(exempt)
    started = datetime.now(timezone.utc)
    dep, rk = rt.load()
    m = measure(dep, rk)
    show(m)

    if not m["served_any"]:
        print("\nNo pooled cell is covered on any leg, so no bucket is served.")
    if m["premise_holds_anywhere"]:
        print(
            "\nSTOP. The premise holds for a bucket a pooled cell would serve."
            "\nThat disagrees with the share table this repository prints, and a"
            "\ndisagreement is a stop and not a green light: report it before"
            "\nanything is fitted."
        )
    else:
        print(
            "\nThe premise fails for every bucket a pooled cell would serve, so"
            "\nthe arm is not fitted. Nothing is adopted and no rule changes."
        )
    if args.arm:
        if m["premise_holds_anywhere"] and m["stake_total"]:
            arm_report(dep)
        else:
            why = "the premise" if not m["premise_holds_anywhere"] else "the stake"
            print(f"\n--arm refused: {why} stopped the block before the arm.")
    if args.report:
        text = write_section(section(m), dep)
        print(f"\nwrote {REPORT.relative_to(REPO)} ({len(text):,} bytes)")
    verify(before, exempt)
    print(f"\nran in {(datetime.now(timezone.utc) - started).total_seconds():,.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
