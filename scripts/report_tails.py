#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Profile the record-quality term and write ``reports/tails.md``.

What the tail of the target is made of, which (airport, bucket) cells carry
it, what the term predicts for them and why, and what it is worth on the
two forward validation splits fixed in ``docs/modelling_notes.md``. Every
number in the report is computed here; nothing is hand-typed.

The physical model the term sits on top of is not this script's to fit.
What is measured here is a *stand-in* -- the training mean per (airport,
stand, runway) with the airport mean as fallback, fitted with flagged rows
excluded -- so that "with and without the term" has a concrete baseline. The
baseline report, ``reports/baseline.md``, carries the same comparison for
the real model.

One section describes the class of departure the cells cannot reach: the
plan-shifted departures, whose rule ``taxiout.tails.PlanShiftModel`` implements
as ``reports/preregistered-2026-10-02.md`` fixed it, with the class's count, its
fitted share and rest means, and the ranking rows it takes.

One section answers a question rather than describing the term. A review note
of 10 September read several callsign prefixes at the covered airport as having
taxi times equal to their lag in most of their rows with a Network Manager
record. *The callsign-prefix reading* measures that claim under both readings
of "equal to the lag" -- the note's and ``taxiout.tails.defaulted_block``'s --
and prints the (airport, prefix, bucket) cells a prefix split would open under
the coverage rule as it stands. It changes neither the term nor the rule.

The report carries aggregates only. A callsign prefix is an ICAO airline
designator and an aggregate over an operator's departures; no stand identifier
from either source appears, and the script refuses to write one.

Usage
-----
    python scripts/report_tails.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.callsign import callsign_features  # noqa: E402
from taxiout.provenance import (  # noqa: E402
    FROZEN_DOCUMENTS,
    challenge_inputs,
    digest,
    report_inputs_footer,
)
from taxiout.challenge import load_movements, load_ranking, to_pandas  # noqa: E402
from taxiout.tails import (  # noqa: E402
    BUCKETS,
    FIT_ONLY_INPUTS,
    LONG_S,
    MIN_COUNT,
    MIN_SHARE,
    NEAR_S,
    PLAN_SHIFT_ADD_ONE,
    PLAN_SHIFT_PREDICT_INPUTS,
    PREDICT_INPUTS,
    UNFLAGGED_BUCKET,
    PlanShiftModel,
    TailModel,
    combine,
    defaulted_block,
    flag,
    lag_seconds,
    plan_shift_threshold_s,
    plan_shifted,
)

DATA = REPO / "data" / "challenge"
OUT = REPO / "reports" / "tails.md"

# The dated record the plan-shifted departures' rule was fixed in, repository
# relative as ``taxiout.provenance.FROZEN_DOCUMENTS`` keys it. Read for its
# digest, never written: this generator writes ``OUT`` alone.
PREREGISTRATION = "reports/preregistered-2026-10-02.md"


def _pooled_section_title() -> str:
    """The title of the section ``scripts/pooled_tails.py`` writes into this file.

    Read out of that file's own ``SECTION_TITLE`` by parsing it, not by
    importing it: ``pooled_tails`` imports this module, so an import here would
    be circular. Read rather than repeated because the header below says which
    generator wrote which part, and a header naming the section in its own words
    would drift from the section itself the first time either changed.
    """
    import ast

    tree = ast.parse((REPO / "scripts" / "pooled_tails.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(x, ast.Name) and x.id == "SECTION_TITLE" for x in node.targets
        ):
            if isinstance(node.value, ast.Constant):
                return str(node.value.value).lstrip("# ").strip()
    raise RuntimeError("scripts/pooled_tails.py carries no SECTION_TITLE")


# The section the OTHER generator writes into this file.
POOLED_SECTION = _pooled_section_title()

HOUR = 3600.0
DAY = 86400.0
JULY_AIRPORTS = ("EDDF", "EGLL", "EHAM")
STAND_IN_MIN_PAIR = 5

# The airport the callsign-prefix section is about: the one the term's covered
# cells are at, and the one the review note of 10 September was about. Written
# out rather than derived from the fitted cells so that the section is about a
# named airport and does not silently follow a change in coverage.
PREFIX_AIRPORT = "LIRF"

# How many prefixes that section prints per record side. Enough to show the
# shape of the claim; the sentence under each table is over every row.
PREFIX_ROWS = 15

# The four buckets a test departure can fall into with no flight-side record,
# in the order the tables use, and the three late-or-early ones with a record.
NOFLT = tuple(b for b in BUCKETS if b.startswith("noflt:"))
FLT_ODD = tuple(b for b in BUCKETS if b.startswith("flt:") and b != "flt:normal")


# --------------------------------------------------------------------------
# helpers


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 0) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:,.{digits}f}"


def secs(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:,.0f} s"


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


def rmse(y: pd.Series, p: pd.Series) -> float:
    e = (y.to_numpy(dtype=float) - p.to_numpy(dtype=float)) ** 2
    return float(np.sqrt(e.mean())) if len(e) else float("nan")


# --------------------------------------------------------------------------
# data


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    # The plan-shifted departures' rule reads one estimate the term does not.
    rule = [c for c in PLAN_SHIFT_PREDICT_INPUTS if c not in PREDICT_INPUTS]
    extra = ["STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt"]
    cols = list(PREDICT_INPUTS + FIT_ONLY_INPUTS) + rule + extra
    dep = to_pandas(load_movements(DATA, cols, phase="DEP"))
    dep["ADEP_mvt"] = dep["ADEP_mvt"].astype(str)
    dep["y"] = pd.to_numeric(dep["TAXITIME_SEC_mvt"], errors="coerce").astype(float)
    dep["lag"] = lag_seconds(dep)
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.month
    dep["art"] = defaulted_block(dep).fillna(False).astype(bool)
    dep["bucket"] = flag(dep)["tail_bucket"].astype(str)
    dep["prefix"] = callsign_features(dep)["cs_prefix"].to_numpy()
    # The review note's reading of "the taxi time equals the lag": the same 60 s
    # the term uses, with no condition on the lag itself. Beside it, the first
    # half of ``defaulted_block`` on its own, so the section can measure whether
    # the two tests are the same test rather than assert that they are.
    dep["near_lag"] = ((dep["y"] - dep["lag"]).abs() <= NEAR_S).fillna(False)
    dep["near_block"] = (
        (dep["BLOCK_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"])
        .dt.total_seconds()
        .abs()
        .le(NEAR_S)
        .fillna(False)
    )
    rk = to_pandas(
        load_ranking(DATA, list(PREDICT_INPUTS) + rule + ["FLIGHT_mvt"], phase="DEP")
    )
    rk["ADEP_mvt"] = rk["ADEP_mvt"].astype(str)
    rk["month"] = rk["MVT_TIME_UTC_mvt"].dt.month
    rk["bucket"] = flag(rk)["tail_bucket"].astype(str)
    rk["prefix"] = callsign_features(rk)["cs_prefix"].to_numpy()
    return dep, rk


class Split:
    def __init__(self, name: str, dep: pd.DataFrame, train: pd.Series, valid: pd.Series):
        self.name = name
        self.train = dep[train]
        self.valid = dep[valid]


def splits(dep: pd.DataFrame) -> list[Split]:
    m = dep["month"]
    july3 = dep["ADEP_mvt"].isin(JULY_AIRPORTS)
    return [
        Split("Dec 2025", dep, m <= 11, m == 12),
        Split("Jul 2025, all", dep, m <= 6, m == 7),
        Split("Jul 2025, EDDF/EGLL/EHAM", dep, m <= 6, (m == 7) & july3),
    ]


# --------------------------------------------------------------------------
# the stand-in physical model


def stand_in(train: pd.DataFrame, valid: pd.DataFrame, exclude: pd.Series) -> pd.Series:
    """Training mean per (airport, stand, runway), airport mean as fallback.

    Fitted on training rows with ``exclude`` dropped -- the flagged rows, as
    the physical model is fitted -- and a pair needs ``STAND_IN_MIN_PAIR``
    rows to have a mean of its own.
    """
    fit = train[~exclude.to_numpy()]
    keys = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
    pair = fit.groupby(keys, observed=True)["y"].agg(["mean", "size"])
    pair = pair.loc[pair["size"] >= STAND_IN_MIN_PAIR, "mean"]
    airport = fit.groupby("ADEP_mvt")["y"].mean()
    idx = pd.MultiIndex.from_frame(valid[keys])
    p = pd.Series(pair.reindex(idx).to_numpy(), index=valid.index)
    return p.fillna(valid["ADEP_mvt"].map(airport)).astype(float)


def airport_mean_sse(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    apm = train.groupby("ADEP_mvt")["y"].mean()
    return (valid["y"] - valid["ADEP_mvt"].map(apm)) ** 2


# --------------------------------------------------------------------------
# design variants, for the alternatives table


def cell_table(train: pd.DataFrame, bucket: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame(
        {
            "icao": train["ADEP_mvt"].to_numpy(),
            "bucket": bucket.to_numpy(),
            "y": train["y"].to_numpy(),
            "art": train["art"].to_numpy(),
        }
    )
    g = t.groupby(["icao", "bucket"], sort=True)
    cells = pd.DataFrame({"n": g.size(), "share": g["art"].mean(), "mean": g["y"].mean()})
    rest = t[~t["art"]].groupby(["icao", "bucket"])["y"].mean()
    cells["rest_mean"] = rest.reindex(cells.index)
    return cells


def variant_prediction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    kind: str,
    coverage: str,
    min_count: int,
    min_share: float,
    pooled_below: int | None = None,
    bucket_train: pd.Series | None = None,
    bucket_valid: pd.Series | None = None,
) -> pd.Series:
    """The term under one design, NaN where it does not apply.

    ``coverage`` is ``"share"`` (defaulted share at least ``min_share``) or
    ``"noflt"`` (every cell without a flight-side record). ``pooled_below``
    gives a ``noflt`` cell under that many rows the bucket's mean pooled over
    airports -- the plan's original fallback -- instead of no prediction.
    """
    bt = train["bucket"] if bucket_train is None else bucket_train
    bv = valid["bucket"] if bucket_valid is None else bucket_valid
    cells = cell_table(train, bt)
    if coverage == "share":
        covered = (cells["n"] >= min_count) & (cells["share"] >= min_share)
    else:
        covered = (cells["n"] >= min_count) & cells.index.get_level_values(
            1
        ).str.startswith("noflt:")
    idx = pd.MultiIndex.from_arrays([valid["ADEP_mvt"].to_numpy(), bv.to_numpy()])
    c = cells.reindex(idx)
    cov = covered.reindex(idx).to_numpy(dtype=object)
    cov = np.where(pd.isna(cov), False, cov).astype(bool)
    if kind == "mean":
        value = c["mean"].to_numpy()
    else:
        rest = c["rest_mean"].to_numpy()
        rest = np.where(np.isnan(rest), 0.0, rest)
        share = c["share"].to_numpy()
        value = share * valid["lag"].to_numpy() + (1 - share) * rest
    out = np.where(cov, value, np.nan)
    if pooled_below is not None:
        pooled = train.groupby(bt.to_numpy())["y"].mean()
        thin = bv.str.startswith("noflt:").to_numpy() & ~cov
        n = c["n"].to_numpy()
        thin &= np.isnan(n) | (n < pooled_below)
        out = np.where(thin, bv.map(pooled).to_numpy(), out)
    return pd.Series(out, index=valid.index, dtype=float)


def with_two_hour_edge(frame: pd.DataFrame) -> pd.Series:
    """Buckets with an extra edge at 2 h inside the normal band, for the test."""
    lag = frame["lag"]
    band = np.select(
        [lag > 12 * HOUR, lag > 4 * HOUR, lag > 2 * HOUR, lag < -HOUR],
        [">12h", "4-12h", "2-4h", "<-1h"],
        default="normal",
    )
    rec = np.where(frame["LOBT_flt"].isna(), "noflt:", "flt:")
    return pd.Series(rec + band, index=frame.index)


# --------------------------------------------------------------------------
# sections


def section_mechanism(dep: pd.DataFrame) -> tuple[list[str], dict]:
    facts: dict = {}
    y, lag = dep["y"], dep["lag"]
    gap = (dep["BLOCK_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs()
    long = y > 2 * HOUR
    facts["long_rows"] = int(long.sum())
    by_ap = dep[long].groupby("ADEP_mvt").size().sort_values(ascending=False)
    top = str(by_ap.index[0])
    facts["long_top"] = top
    facts["long_top_share"] = float(by_ap.iloc[0] / long.sum())
    facts["long_noflt_share"] = float(dep.loc[long, "LOBT_flt"].isna().mean())
    near_long = long & (gap <= NEAR_S)
    facts["long_near_share"] = float(near_long.sum() / long.sum())
    facts["long_top_near_share"] = float(
        near_long[dep["ADEP_mvt"] == top].sum() / long[dep["ADEP_mvt"] == top].sum()
    )
    facts["art_rows"] = int(dep["art"].sum())
    facts["art_top_share"] = float(dep.loc[dep["art"], "ADEP_mvt"].eq(top).mean())
    facts["art_noflt_share"] = float(dep.loc[dep["art"], "LOBT_flt"].isna().mean())
    d = (y - lag)[dep["art"]]
    facts["art_diff_p10"], facts["art_diff_p50"], facts["art_diff_p90"] = (
        float(v) for v in np.percentile(d, [10, 50, 90])
    )
    art_gap = gap[dep["art"]]
    facts["art_gap_le5"] = float((art_gap <= 5).mean())

    # the threshold, on the top airport's long rows without a record
    m = long & (dep["ADEP_mvt"] == top) & dep["LOBT_flt"].isna()
    thr_rows = []
    for t in (0, 5, 30, 60, 300, 1800):
        thr_rows.append([f"{t:,}", pct((gap[m] <= t).mean())])
    facts["thr_n"] = int(m.sum())

    # the day-shift population: long rows, not defaulted, within a day of a day
    day = long & ~dep["art"] & (y > 20 * HOUR) & (y < 28 * HOUR)
    facts["day_rows"] = int(day.sum())
    facts["day_top_share"] = (
        float(dep.loc[day, "ADEP_mvt"].eq(top).mean()) if day.any() else float("nan")
    )
    facts["day_minus_day_p50"] = (
        float((y[day] - DAY).median()) if day.any() else float("nan")
    )

    # per airport: what the long rows are
    rows = []
    for icao, g in dep[long].groupby("ADEP_mvt"):
        a = g["art"]
        dd = day[g.index]
        rows.append(
            [
                icao,
                f"{len(g):,}",
                pct(g["LOBT_flt"].isna().mean()),
                f"{int(a.sum()):,}",
                f"{int(dd.sum()):,}",
                f"{int((~a & ~dd).sum()):,}",
            ]
        )
    lines = [
        "## What the tail is made of",
        "",
        f"{facts['long_rows']:,} training departures record a taxi-out above two hours.",
        f"{pct(facts['long_top_share'])} of them are at {top}, and",
        f"{pct(facts['long_noflt_share'])} have no flight-side record. For",
        f"{pct(facts['long_near_share'])} of them -- {pct(facts['long_top_near_share'])} at {top}",
        "-- the recorded off-block time sits within a minute of the *scheduled*",
        "departure time while the take-off is hours later. The block time was",
        "defaulted to the schedule, so the recorded taxi-out is take-off minus",
        "schedule: the *lag*, which the ranking set exposes for every departure.",
        "",
        f"Share of {top}'s over-two-hour departures without a record whose block",
        f"time is within a given distance of the schedule ({facts['thr_n']} rows):",
        "",
        *table(["`|block - schedule|` at most (s)", "share"], thr_rows, "-r"),
        "",
        "`taxiout.tails.defaulted_block` takes the 60 s threshold (the schedule is",
        "minute-resolution) together with a lag above two hours, which is above",
        "every plausible taxi-out and below every artifact. It picks out",
        f"{facts['art_rows']:,} training rows, {pct(facts['art_top_share'])} of them at {top} and",
        f"{pct(facts['art_noflt_share'])} without a record; their taxi-out minus lag has",
        f"p10/p50/p90 of {num(facts['art_diff_p10'])}/{num(facts['art_diff_p50'])}/"
        f"{num(facts['art_diff_p90'])} s, and {pct(facts['art_gap_le5'])} of them are within 5 s.",
        "",
        "A second, smaller population is the day-shift the data review",
        f"described: {facts['day_rows']:,} over-two-hour rows that are not defaulted",
        f"sit between 20 and 28 hours, {pct(facts['day_top_share'])} of them at {top},",
        f"a median {secs(facts['day_minus_day_p50'])} above one day -- a block time from",
        "the previous day plus an ordinary taxi. The term does not model these",
        "separately: they fall into the same cells and are carried by the cell's",
        "mean over its non-defaulted rows.",
        "",
        "Over-two-hour departures per airport, by kind:",
        "",
        *table(
            [
                "ICAO",
                "rows > 2 h",
                "no flt record",
                "defaulted block",
                "day-shift",
                "other",
            ],
            rows,
            "-rrrrr",
        ),
    ]
    return lines, facts


def section_cells(dep: pd.DataFrame, model: TailModel) -> tuple[list[str], dict]:
    facts: dict = {}
    cells = model.cells
    airports = sorted(dep["ADEP_mvt"].unique())

    def cell(icao: str, b: str, col: str) -> float:
        try:
            return float(cells.loc[(icao, b), col])
        except KeyError:
            return float("nan")

    def size_mean(icao: str, b: str) -> str:
        n = cell(icao, b, "n")
        if not np.isfinite(n):
            return "--"
        return f"{int(n):,} / {num(cell(icao, b, 'mean'))}"

    header = ["ICAO"] + [f"`{b}`" for b in BUCKETS]
    rows = [[icao] + [size_mean(icao, b) for b in BUCKETS] for icao in airports]
    share_rows = [
        [icao] + [pct(cell(icao, b, "share")) for b in NOFLT + FLT_ODD]
        for icao in airports
    ]
    # The unflagged cell's defaulted share, computed rather than stated: the
    # sentence under the table below used to read "is 0.0% everywhere", which
    # was a measurement no run produced. ``nanmax`` so an airport with no rows
    # in the cell does not decide the claim.
    unflagged_values = [cell(icao, UNFLAGGED_BUCKET, "share") for icao in airports]
    finite = [v for v in unflagged_values if np.isfinite(v)]
    unflagged_share = max(finite) if finite else float("nan")
    facts["unflagged_share_max"] = unflagged_share
    cov = model.covered
    cov_rows = []
    for (icao, b), r in cov.iterrows():
        cov_rows.append(
            [
                icao,
                f"`{b}`",
                f"{int(r['n']):,}",
                pct(r["share"]),
                num(r["mean"]),
                num(r["rest_mean"]),
                f"{r['share']:.3f} x lag + {num((1 - r['share']) * (r['rest_mean'] if np.isfinite(r['rest_mean']) else 0.0))}",
            ]
        )
    facts["covered_cells"] = len(cov)
    facts["covered_airports"] = sorted(set(cov.index.get_level_values(0)))
    facts["covered_rows_train"] = int(cov["n"].sum())
    lines = [
        "## The cells",
        "",
        "A test departure exposes whether the flight-side join found a record",
        "(`LOBT_flt` null) and its lag, banded at the edges of the data review's",
        "lag table: `normal` is [-1 h, 4 h], `4-12h` is (4 h, 12 h], `>12h` and",
        "`<-1h` are open. With the airport, that is a cell. Training rows per",
        "cell and the mean taxi-out in it (rows / mean, seconds):",
        "",
        *table(header, rows, "-" + "r" * len(BUCKETS)),
        "",
        "Share of each cell's rows with a defaulted block time, for the cells",
        "that could carry one; the `"
        + UNFLAGGED_BUCKET
        + "` cell reaches "
        + pct(unflagged_share, 1)
        + " at its highest airport, which is why it is not among them):",
        "",
        *table(["ICAO"] + [f"`{b}`" for b in NOFLT + FLT_ODD], share_rows, "-" + "r" * 7),
        "",
        f"A cell is covered by the term when it has at least {model.min_count} rows and",
        f"a defaulted share of at least {pct(model.min_share, 0)}; the rule is justified",
        f"below. {facts['covered_cells']} cells qualify, all at",
        f"{', '.join(facts['covered_airports'])}, {facts['covered_rows_train']:,} training rows",
        "between them. For a row in a covered cell the term predicts",
        "`share x lag + (1 - share) x rest_mean`, where `rest_mean` is the cell's",
        "mean over the rows that were not defaulted:",
        "",
        *table(
            ["ICAO", "cell", "rows", "defaulted", "cell mean", "rest mean", "prediction"],
            cov_rows,
            "--rrrr-",
        ),
        "",
        "Every other row -- every row at the other airports, and every row with",
        "a flight-side record -- gets no prediction from the term and is left to",
        "the physical model, except the plan-shifted departures, which have a",
        "section of their own below.",
    ]
    return lines, facts


def section_design(dep: pd.DataFrame, sp: list[Split]) -> tuple[list[str], dict]:
    facts: dict = {}
    # base predictions per split: stand-in physical model with the adopted
    # flags excluded from its fit
    base: dict[str, tuple[pd.Series, pd.Series]] = {}
    for s in sp:
        tm = TailModel().fit(s.train)
        phys = stand_in(s.train, s.valid, tm.flagged(s.train))
        base[s.name] = (phys, tm.predict(s.valid))

    def score(fn) -> list[str]:
        out = []
        for s in sp:
            phys, _ = base[s.name]
            term = fn(s)
            out.append(num(rmse(s.valid["y"], combine(phys, term))))
        return out

    def variant(**kw):
        return lambda s: variant_prediction(s.train, s.valid, **kw)

    designs = [
        (
            "stand-in physical model alone",
            lambda s: pd.Series(np.nan, index=s.valid.index),
        ),
        (
            "cell mean, every `noflt` cell, cells under 20 rows pooled over airports "
            "(the plan's first proposal)",
            variant(
                kind="mean", coverage="noflt", min_count=20, min_share=0, pooled_below=20
            ),
        ),
        (
            "cell mean, every `noflt` cell with 5 rows or more, thin cells to the physical model",
            variant(kind="mean", coverage="noflt", min_count=5, min_share=0),
        ),
        (
            f"cell mean, cells with a defaulted share of {pct(MIN_SHARE, 0)} or more "
            f"and {MIN_COUNT} rows or more",
            variant(
                kind="mean", coverage="share", min_count=MIN_COUNT, min_share=MIN_SHARE
            ),
        ),
        (
            "mixture, same coverage (**adopted**)",
            variant(
                kind="mixture", coverage="share", min_count=MIN_COUNT, min_share=MIN_SHARE
            ),
        ),
        (
            "mixture, same coverage, with an extra lag edge at 2 h",
            lambda s: variant_prediction(
                s.train,
                s.valid,
                kind="mixture",
                coverage="share",
                min_count=MIN_COUNT,
                min_share=MIN_SHARE,
                bucket_train=with_two_hour_edge(s.train),
                bucket_valid=with_two_hour_edge(s.valid),
            ),
        ),
    ]
    scores = {
        name: [float(v.replace(",", "")) for v in score(fn)] for name, fn in designs
    }
    design_rows = [[name] + [num(v) for v in scores[name]] for name, _ in designs]
    facts["design_rows"] = design_rows
    names = [s.name for s in sp]
    keys = [name for name, _ in designs]
    term_keys = keys[1:]
    pooled, mean_share, mixture, edge = keys[1], keys[3], keys[4], keys[5]
    # Splits that carry artifacts: those with at least one flagged row.
    with_art = [i for i, s in enumerate(sp) if base[s.name][1].notna().any()]
    facts["pooled_worst"] = all(
        scores[pooled][i] >= max(scores[k][i] for k in term_keys) for i in range(len(sp))
    )
    facts["mixture_gain"] = [scores[mean_share][i] - scores[mixture][i] for i in with_art]
    facts["edge_delta"] = [scores[mixture][i] - scores[edge][i] for i in with_art]
    facts["with_art_names"] = [names[i] for i in with_art]

    # the adopted design must reproduce the module
    adopted = designs[4][1]
    for s in sp:
        a = adopted(s)
        b = base[s.name][1]
        if not np.allclose(a.fillna(-1), b.fillna(-1)):
            raise AssertionError(f"report variant and taxiout.tails disagree on {s.name}")

    # The rarest covered cell, and how many rows it has in each split's
    # training half: that is what a minimum count is measured against.
    full = dep
    rare_cell = TailModel().fit(full).covered["n"].idxmin()
    rare_rows = {}
    for s in sp:
        cells = TailModel().fit(s.train).cells
        rare_rows[s.name] = (
            int(cells.loc[rare_cell, "n"]) if rare_cell in cells.index else 0
        )
    facts["rare_cell"] = rare_cell
    facts["rare_rows"] = rare_rows

    count_rows = []
    for n_min in (1, 2, 3, 5, 10, 20, 50):
        m = TailModel(min_count=n_min).fit(full)
        count_rows.append(
            [str(n_min), str(len(m.covered))]
            + score(
                variant(
                    kind="mixture", coverage="share", min_count=n_min, min_share=MIN_SHARE
                )
            )
        )
    share_rows = []
    for p_min in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
        m = TailModel(min_share=p_min).fit(full)
        share_rows.append(
            [pct(p_min, 0), str(len(m.covered))]
            + score(
                variant(
                    kind="mixture", coverage="share", min_count=MIN_COUNT, min_share=p_min
                )
            )
        )
    # The gap in defaulted shares around the threshold, over cells with
    # enough rows: the flat region of the sweep is exactly this gap.
    big = TailModel().fit(full).cells.query("n >= @MIN_COUNT")["share"]
    below = float(big[big < MIN_SHARE].max()) if (big < MIN_SHARE).any() else float("nan")
    above = (
        float(big[big >= MIN_SHARE].min()) if (big >= MIN_SHARE).any() else float("nan")
    )
    facts["share_below"], facts["share_above"] = below, above
    facts["share_max"] = float(big.max())

    def signed(v: float) -> str:
        return f"{v:+,.0f} s"

    pooled_sentence = (
        "the plan's first proposal is the worst of the five forms on every split"
        if facts["pooled_worst"]
        else "the plan's first proposal is not the best form on any split"
    )
    gain = ", ".join(
        f"{signed(g)} on {n}"
        for g, n in zip(facts["mixture_gain"], facts["with_art_names"])
    )
    edge = ", ".join(
        f"{signed(d)} on {n}"
        for d, n in zip(facts["edge_delta"], facts["with_art_names"])
    )
    # The two July splits share a training half, so name the halves once.
    rare_text = (
        f"{rare_rows[sp[0].name]} in January to November and "
        f"{rare_rows[sp[1].name]} in January to June"
    )
    lines = [
        "## Why this form, and not the others",
        "",
        "Forward splits as fixed in `docs/modelling_notes.md`: train January to",
        "November 2025 and validate December at all ten airports; train January",
        "to June and validate July, for all airports and for the three July test",
        "airports alone. The physical model is the stand-in described at the top,",
        "fitted with the adopted flags excluded; each design supplies the term",
        "for the rows it covers and the physical model predicts the rest. RMSE",
        "over every validation row, in seconds:",
        "",
        *table(["design"] + names, design_rows, "-rrr"),
        "",
        "Three things the table settles. Pooling a thin cell over airports",
        "carries one airport's artifact rate to airports that have none, and",
        f"costs more than it saves: {pooled_sentence}. Knowing the mechanism is",
        "worth having: the mixture beats the cell mean with the same coverage",
        f"({gain}) because it predicts each artifact's own value rather than the",
        "cell's average of them. And an extra lag edge at two hours, which",
        "separates the defaulted rows more sharply within the `normal` band,",
        f"moves the mixture by {edge} -- the defaulted share in that band shifts",
        "between months by more than the edge gains -- so it is not adopted.",
        "",
        "The minimum count. Covered cells at the full-year fit, and RMSE, as the",
        f"count moves (defaulted share at least {pct(MIN_SHARE, 0)} throughout):",
        "",
        *table(["minimum rows", "covered cells"] + names, count_rows, "rrrrr"),
        "",
        "The cells the term exists for are rare by construction -- they are the",
        "tail -- so a minimum count high enough to look prudent discards them.",
        f"The rarest covered cell, `{rare_cell[1]}` at {rare_cell[0]}, has",
        f"{int(TailModel().fit(full).cells.loc[rare_cell, 'n'])} rows in the full year and",
        f"{rare_text}; a minimum above either count drops it from that split,",
        "which is what the July columns show. Below five, a cell's defaulted",
        f"share rests on a handful of rows. {MIN_COUNT} is taken; the table shows every",
        "value from one to five scores the same.",
        "",
        f"The minimum share, at {MIN_COUNT} rows:",
        "",
        *table(["minimum defaulted share", "covered cells"] + names, share_rows, "rrrrr"),
        "",
        "The sweep is flat because no cell with enough rows has a defaulted",
        f"share between {pct(below)} and {pct(above)}: the artifact is a property of one",
        f"airport's feed, present in its cells at {pct(above, 0)}-{pct(facts['share_max'], 0)} and absent",
        f"elsewhere. {pct(MIN_SHARE, 0)} is taken as the round number inside the gap.",
    ]
    return lines, facts


def section_worth(dep: pd.DataFrame, sp: list[Split]) -> tuple[list[str], dict]:
    facts: dict = {}
    lines = [
        "## What the term is worth",
        "",
        "The share of the airport-mean predictor's squared error that the",
        "flagged rows carry on each validation month, and what the stand-in",
        "physical model scores with and without the term. The last column is",
        "the physical model on the unflagged rows alone, which the term does not",
        "touch; it is the number the physical model is answerable for.",
        "",
    ]
    summary = []
    for s in sp:
        tm = TailModel().fit(s.train)
        flagged = tm.flagged(s.valid)
        phys = stand_in(s.train, s.valid, tm.flagged(s.train))
        both = combine(phys, tm.predict(s.valid))
        sse = airport_mean_sse(s.train, s.valid)
        y = s.valid["y"]
        facts[s.name] = {
            "rows": int(len(s.valid)),
            "flagged": int(flagged.sum()),
            "sse_share": float(sse[flagged].sum() / sse.sum()),
            "rmse_without": rmse(y, phys),
            "rmse_with": rmse(y, both),
            "rmse_unflagged": rmse(y[~flagged], phys[~flagged]),
        }
        f = facts[s.name]
        summary.append(
            [
                s.name,
                f"{f['rows']:,}",
                f"{f['flagged']:,}",
                pct(f["flagged"] / f["rows"], 3),
                pct(f["sse_share"]),
                num(f["rmse_without"]),
                num(f["rmse_with"]),
                num(f["rmse_unflagged"]),
            ]
        )
        # per airport
        rows = []
        for icao, g in s.valid.groupby("ADEP_mvt"):
            fl = flagged[g.index]
            rows.append(
                [
                    icao + (" *" if icao in JULY_AIRPORTS and "Jul" in s.name else ""),
                    f"{len(g):,}",
                    f"{int(fl.sum()):,}",
                    pct(sse[g.index][fl].sum() / sse[g.index].sum()),
                    num(rmse(g["y"], phys[g.index])),
                    num(rmse(g["y"], both[g.index])),
                    num(rmse(g["y"][~fl], phys[g.index][~fl])),
                ]
            )
        facts[s.name]["airport_rows"] = rows
    lines += table(
        [
            "split",
            "rows",
            "flagged",
            "share of rows",
            "share of airport-mean SSE",
            "RMSE without",
            "RMSE with",
            "RMSE, unflagged rows",
        ],
        summary,
        "-rrrrrrr",
    )
    for s in sp:
        if "airport_rows" not in facts[s.name] or len(facts[s.name]["airport_rows"]) < 4:
            continue  # the three-airport split repeats rows of the full July table
        lines += [
            "",
            f"Per airport, {s.name}"
            + (" (`*` marks the July test airports)" if "Jul" in s.name else "")
            + ":",
            "",
            *table(
                [
                    "ICAO",
                    "rows",
                    "flagged",
                    "SSE share",
                    "RMSE without",
                    "RMSE with",
                    "RMSE, unflagged rows",
                ],
                facts[s.name]["airport_rows"],
                "-rrrrrr",
            ),
        ]
    return lines, facts


def _names(items: list[str]) -> str:
    """``A``, ``A and B``, ``A, B and C``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _july_reach(f: dict) -> str:
    """Where the term reaches in July, read from the ranking set itself.

    Until the organizers re-issued the ranking files on 4 September 2026 this
    was a fixed clause, "whose three airports have no covered cell", true of
    the three airports the July month then held and false once it held ten.
    """
    if f["rk_flagged_jul"]:
        return f"all at {_names(f['rk_flagged_jul_airports'])}"
    if not f["rk_jul_covered"]:
        return f"whose {f['rk_jul_airports']} airports have no covered cell"
    has = "has" if len(f["rk_jul_covered"]) == 1 else "have"
    return f"although {_names(f['rk_jul_covered'])} {has} a covered cell"


def _month_split(f: dict) -> str:
    """The term's ranking rows by month, in the words the counts decide."""
    if not f["rk_flagged_jul"]:
        return "all in January"
    if not f["rk_flagged_jan"]:
        return "all in July"
    return f"{f['rk_flagged_jan']:,} in January and {f['rk_flagged_jul']:,} in July"


def _rows(n: int) -> str:
    return f"{n:,} {'row' if n == 1 else 'rows'}"


def _where(frame: pd.DataFrame, mask: pd.Series) -> list[str]:
    """``1 at LIRF in July 2026``, per (airport, month) of the masked rows."""
    at = frame.loc[mask.to_numpy(dtype=bool)]
    when = pd.to_datetime(at["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%B %Y")
    counts = at.groupby([at["ADEP_mvt"].astype(str), when.rename("when")]).size()
    return [f"{int(n):,} at {icao} in {w}" for (icao, w), n in counts.items()]


def section_ranking(
    dep: pd.DataFrame, rk: pd.DataFrame, model: TailModel, rule: PlanShiftModel
) -> tuple[list[str], dict]:
    facts: dict = {}
    flagged = model.flagged(rk)
    facts["rk_rows"] = int(len(rk))
    facts["rk_flagged"] = int(flagged.sum())
    rows = []
    cov_index = model.covered.index
    train_months = 12
    for (icao, month), g in rk.groupby(["ADEP_mvt", "month"]):
        fl = flagged[g.index]
        if not fl.any():
            continue
        cells = []
        for b in BUCKETS:
            if (icao, b) in cov_index:
                n_rk = int((g["bucket"] == b).sum())
                n_tr = float(model.cells.loc[(icao, b), "n"]) / train_months
                cells.append(f"`{b}` {n_rk:,} ({n_tr:,.0f}/month in 2025)")
        rows.append(
            [
                icao,
                f"2026-{month:02d}",
                f"{len(g):,}",
                f"{int(fl.sum()):,}",
                "; ".join(cells),
            ]
        )
    by_month = flagged.groupby(rk["month"]).sum()
    facts["rk_flagged_jan"] = int(by_month.get(1, 0))
    facts["rk_flagged_jul"] = int(by_month.get(7, 0))
    jul = rk["month"] == 7
    jul_airports = set(rk.loc[jul, "ADEP_mvt"])
    facts["rk_jul_airports"] = len(jul_airports)
    facts["rk_jul_covered"] = sorted(jul_airports & set(cov_index.get_level_values(0)))
    facts["rk_flagged_jul_airports"] = sorted(set(rk.loc[flagged & jul, "ADEP_mvt"]))
    # The plan-shifted departures' rule, on the rows the term leaves: the rows
    # ``taxiout.tails.combine`` takes it on, counted beside the term's.
    shifted = rule.predict(rk).notna() & ~flagged
    facts["rk_plan_shift"] = int(shifted.sum())
    facts["rk_plan_shift_where"] = _where(rk, shifted)
    lines = [
        "## What it touches in the ranking set",
        "",
        f"Fitted on the full training year, the term predicts for {facts['rk_flagged']:,} of the",
        f"{facts['rk_rows']:,} ranking departures: {facts['rk_flagged_jan']:,} in January 2026 and",
        f"{facts['rk_flagged_jul']:,} in July, {_july_reach(facts)}.",
        "Rows per covered cell, against the 2025 monthly rate:",
        "",
        *table(
            ["ICAO", "month", "departures", "flagged", "covered cells"], rows, "--rr-"
        ),
        "",
        "Beside the term, the rule of *The plan-shifted departures* below takes",
        f"{_rows(facts['rk_plan_shift'])} of the ranking set that the term leaves"
        + (
            f": {_names(facts['rk_plan_shift_where'])}."
            if facts["rk_plan_shift"]
            else "."
        ),
        "The two together replace the physical model's prediction on",
        f"{facts['rk_flagged'] + facts['rk_plan_shift']:,} of the {facts['rk_rows']:,} ranking departures.",
    ]
    return lines, facts


def section_plan_shift(
    dep: pd.DataFrame, rk: pd.DataFrame, model: TailModel, rule: PlanShiftModel
) -> list[str]:
    """The plan-shifted departures: the class, its fit, the ranking rows it takes.

    Aggregates only, as the rest of the report: counts, the fitted share and
    the rest means, and where the class sits in the ranking set. Neither a
    training departure's own values nor any prediction is printed.
    """
    held = {f.path: f.sha256 for f in FROZEN_DOCUMENTS}.get(PREREGISTRATION)
    sha = digest(REPO / PREREGISTRATION)
    if held is None or sha != held:
        raise RuntimeError(
            f"{PREREGISTRATION} is not at the digest"
            " taxiout.provenance.FROZEN_DOCUMENTS holds for it; the rule's source"
            " has moved"
        )
    hours = f"{plan_shift_threshold_s() / HOUR:g}"
    plus_long, plus_n = PLAN_SHIFT_ADD_ONE
    y = dep["y"]
    members = rule.in_class(dep)
    long = members & (y > plan_shift_threshold_s())
    by_airport = dep.loc[members, "ADEP_mvt"].astype(str).value_counts().sort_index()
    train_shifted = plan_shifted(dep)
    train_shifted_covered = int((train_shifted & model.flagged(dep)).sum())
    rest_pop = dep["LOBT_flt"].notna() & (y <= plan_shift_threshold_s())
    rest_rows = rest_pop.groupby(dep["ADEP_mvt"].astype(str)).sum()
    term_rows = model.flagged(dep)
    term_in_rest = int((rest_pop & term_rows).sum())
    term_with_record = int((term_rows & dep["LOBT_flt"].notna()).sum())
    class_in_rest = int((rest_pop & members).sum())
    rest = [
        [icao, f"{int(rest_rows.get(icao, 0)):,}", num(float(v), 1)]
        for icao, v in rule.rest_mean.items()
    ]
    rk_flagged = model.flagged(rk)
    rk_shifted = plan_shifted(rk)
    rk_members = rule.in_class(rk)
    when = pd.to_datetime(rk["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m")
    at = rk.loc[rk_members.to_numpy(dtype=bool)]
    counts = at.groupby([at["ADEP_mvt"].astype(str), when[at.index].rename("m")]).size()
    rk_rows = [[icao, m, f"{int(n):,}"] for (icao, m), n in counts.items()]
    reads = ", ".join(f"`{c}`" for c in PLAN_SHIFT_PREDICT_INPUTS)
    lines = [
        "## The plan-shifted departures",
        "",
        "One class of departure the cells cannot reach: a departure the term does",
        f"not cover whose take-off falls more than {hours} h after both of its flight",
        "plan's off-block estimates, `EOBT_1_flt` and `LOBT_flt`, both present. The",
        "cells key on the schedule lag and on the absence of a record, and these",
        "departures have one. `taxiout.tails.PlanShiftModel` predicts for them by the",
        "rule `reports/preregistered-2026-10-02.md` fixed before it was evaluated or",
        f"built (sha256 `{sha}`), and that document is the",
        "source of every definition in this section.",
        "",
        f"**It rests on {_rows(rule.n)} of the training year, and was added after the",
        "class was seen in the ranking set's exposed fields.** The pre-registration",
        "records both, and says of its own evaluation that it shows what the rule",
        "does to those rows and is not independent evidence that the rule is right.",
        "",
        "### The class in the training year",
        "",
        "Fitted on the full training year, against the term fitted there: the class",
        f"holds {rule.n:,} {'departure' if rule.n == 1 else 'departures'}"
        + (
            f" ({', '.join(f'{int(n):,} at {icao}' for icao, n in by_airport.items())})"
            if rule.n
            else ""
        )
        + ",",
        f"and {int(long.sum()):,} of them recorded a taxi-out over {hours} h, which the rule",
        f"calls *long*. The term covers {train_shifted_covered:,} of the"
        f" {int(train_shifted.sum()):,} training"
        f" {'departure' if int(train_shifted.sum()) == 1 else 'departures'} past {hours} h"
        " after both"
        " estimates"
        + (
            "; those are the term's and not the class's."
            if train_shifted_covered
            else "."
        ),
        "",
        "The share is the add-one estimate over the class's departures,",
        f"`(long + {plus_long}) / (n + {plus_n})`, so a class of a few rows cannot assert a",
        f"share of none or all: ({rule.long:,} + {plus_long}) / ({rule.n:,} + {plus_n}) = **{rule.share:.4f}**.",
        "The rest mean is the mean recorded taxi-out of the airport's training",
        f"departures with a flight-plan record (`LOBT_flt` present) and a taxi-out of {hours} h",
        "or less. Nothing is left out for being the term's or the class's, since",
        "the pre-registration names no such exclusion. The means hold",
        f"{term_in_rest:,} of the term's {int(term_rows.sum()):,} rows, because",
        f"{term_with_record:,} of those have a record, and {class_in_rest:,} of the class's {rule.n:,}:",
        "",
        *table(
            ["ICAO", f"departures with a record, {hours} h or less", "rest mean (s)"],
            rest,
            "-rr",
        ),
        "",
        "### The class in the ranking set",
        "",
        f"{int(rk_members.sum()):,} of the {len(rk):,} ranking departures"
        f" {'is' if int(rk_members.sum()) == 1 else 'are'} in the class. The term"
        f" covers {int((rk_shifted & rk_flagged).sum()):,} of the"
        f" {int(rk_shifted.sum()):,} ranking"
        f" {'departure' if int(rk_shifted.sum()) == 1 else 'departures'} past {hours} h"
        " after both"
        " estimates"
        + (
            "; those are the term's and not the class's."
            if int((rk_shifted & rk_flagged).sum())
            else "."
        ),
        "",
        *table(["ICAO", "month", "departures in the class"], rk_rows, "--r"),
        "",
        "### The prediction",
        "",
        "`share x lag + (1 - share) x rest_mean`, the form the term uses, with the lag",
        "taken as take-off minus `LOBT_flt`, the plan's last known off-block time,",
        "and the rest mean of the departure's airport. It is substituted",
        "(`taxiout.tails.combine`) for the physical model's prediction on the",
        "class's rows, after the term, and before the rounding and the clip at zero",
        "that `scripts/make_submission.py` applies to every prediction. It reads",
        f"{reads} and, at fit time, the recorded taxi-out; never",
        "`AOBT_3_flt`, which stays excluded by decision. The physical model, its rows",
        "and its parameters, and the term's cells and the rows it covers, are what",
        "they were without it.",
    ]
    return lines


# --------------------------------------------------------------------------
# the callsign-prefix reading of the term, at the one airport it covers


def section_prefixes(dep: pd.DataFrame, rk: pd.DataFrame, model: TailModel) -> list[str]:
    """What a callsign-prefix split of the cells would and would not add.

    A review note of 10 September, in no report, read several callsign prefixes
    at the covered airport as having taxi times equal to their lag in most of
    their rows *with* a Network Manager record -- which, if the term's cells
    were split by prefix, would be cells the term does not cover. The note's
    reading of "equal to the lag" is ``|taxi - lag| <= NEAR_S`` with no
    condition on the lag itself; the term's own reading, ``defaulted_block``,
    adds the two-hour condition. Both are printed, per prefix and per record,
    and then the cells a prefix split would actually open. Descriptive: the
    term, its inputs and its coverage rule are unchanged.
    """
    hours = f"{LONG_S / HOUR:.0f}"
    lines = [
        f"## The callsign-prefix reading, at {PREFIX_AIRPORT}",
        "",
        "A review note of 10 September, written up nowhere until now, read several",
        f"callsign prefixes at {PREFIX_AIRPORT} *with* a Network Manager record as having",
        "taxi times equal to their lag in most of their rows. If that were the",
        "defaulting the term is about, the cells would be under-drawn: the term keys on",
        "(airport, record, lag band) and would be missing an operator dimension. This",
        "section measures the claim, and then measures what acting on it would reach.",
        "It changes nothing -- not the term, not its inputs, not its coverage rule.",
        "",
        "Two readings of *the taxi time equals the lag* are printed side by side. The",
        f"note's is `|taxi - lag| <= {NEAR_S:.0f} s` and nothing else. The term's is",
        f"`taxiout.tails.defaulted_block`: the *block* time within {NEAR_S:.0f} s of the",
        f"schedule **and** a lag above {hours} hours.",
        "",
        "They are not two tests. The target is take-off minus off-block and the lag is",
        "take-off minus schedule, so `taxi - lag` is `schedule - block` exactly, and the",
        "note's test is the term's first condition written another way. The data review",
        "reports the target equal to take-off minus off-block to the second in every",
        "training departure, so the identity should hold on the data and not only in",
        f"algebra: over the {len(dep):,} training departures the two tests agree on",
        f"{pct(float((dep['near_lag'] == dep['near_block']).mean()), 1)} of rows, with",
        f"{int((dep['near_lag'] != dep['near_block']).sum()):,} disagreeing. What the",
        f"term adds is the {hours}-hour condition, and that is the whole of the",
        "difference between the two columns below.",
        "",
        f"{PREFIX_AIRPORT} is the airport the term's covered cells are at, and the one",
        "the note was about.",
        "",
    ]
    at = dep[dep["ADEP_mvt"] == PREFIX_AIRPORT]
    if not len(at):
        return lines + [f"No {PREFIX_AIRPORT} departures in the training year."]

    for norec, label in ((False, "with a record"), (True, "without a record")):
        part = at[at["LOBT_flt"].isna() == norec]
        g = part.groupby("prefix", sort=True)
        grouped = pd.DataFrame({"n": g.size(), "art": g["art"].mean()})
        grouped["near"] = g["near_lag"].mean()
        # The median lag of the rows the note's reading catches, which is the
        # quantity the two readings differ over.
        grouped["lag"] = (
            part[part["near_lag"]]
            .groupby("prefix")["lag"]
            .median()
            .reindex(grouped.index)
        )
        grouped = grouped[grouped["n"] >= MIN_COUNT].sort_values("near", ascending=False)
        caught = part[part["near_lag"]]
        rows = [
            [
                str(prefix),
                f"{int(r['n']):,}",
                pct(r["art"]),
                pct(r["near"]),
                num(r["lag"]),
            ]
            for prefix, r in grouped.head(PREFIX_ROWS).iterrows()
        ]
        lines += [
            f"### {PREFIX_AIRPORT} departures {label}",
            "",
            f"{len(part):,} rows over {int(part['prefix'].nunique()):,} prefixes, of which",
            f"{len(grouped):,} have at least {MIN_COUNT} rows. The",
            f"{min(PREFIX_ROWS, len(grouped))} with the largest share under the note's",
            "reading are printed. `median lag` is over that prefix's rows the note's",
            "reading catches, and is what separates the two readings.",
            "",
            *table(
                [
                    "prefix",
                    "rows",
                    f"defaulted, the term ({hours} h)",
                    f"taxi within {NEAR_S:.0f} s of the lag, the note",
                    "median lag (s)",
                ],
                rows,
                "-rrrr",
            ),
            "",
            f"Over all {len(part):,} rows {label}: {pct(float(part['art'].mean()))} are",
            f"defaulted under the term's reading and {pct(float(part['near_lag'].mean()))}",
            "under the note's, and the rows the note's reading catches have a median lag",
            f"of {secs(float(caught['lag'].median())) if len(caught) else 'n/a'}.",
            "",
        ]

    # What a prefix split would open: (airport, prefix, bucket) cells meeting
    # the term's own rule, against the (airport, bucket) cells it already covers.
    cells = at.groupby(["prefix", "bucket"], sort=True).agg(
        n=("y", "size"), share=("art", "mean")
    )
    meet = cells[(cells["n"] >= model.min_count) & (cells["share"] >= model.min_share)]
    covered = {b for i, b in model.covered.index if i == PREFIX_AIRPORT}
    new = meet[[b not in covered for _p, b in meet.index]]
    rk_at = rk[rk["ADEP_mvt"] == PREFIX_AIRPORT]
    nrows, touched = [], 0
    for (prefix, bucket), r in new.iterrows():
        hits = int(((rk_at["prefix"] == prefix) & (rk_at["bucket"] == bucket)).sum())
        touched += hits
        nrows.append(
            [str(prefix), f"`{bucket}`", f"{int(r['n']):,}", pct(r["share"]), f"{hits:,}"]
        )
    lines += [
        "### The cells a prefix split would open",
        "",
        f"Every (airport, prefix, bucket) cell at {PREFIX_AIRPORT} meeting the term's own",
        f"coverage rule -- at least {model.min_count} rows and a defaulted share of at",
        f"least {pct(model.min_share, 0)} -- that the term's present cells do not already",
        f"cover. {len(meet):,} cells meet the rule. {len(meet) - len(new):,} of them lie",
        f"inside the {len(covered)} (airport, bucket) cells the term already covers at",
        f"{PREFIX_AIRPORT}, so every row in them is predicted for already and splitting",
        f"by prefix would move nothing there. {len(new):,}",
        f"{'lies' if len(new) == 1 else 'lie'} outside.",
        "",
    ]
    if nrows:
        lines += [
            *table(
                [
                    "prefix",
                    "cell",
                    "training rows",
                    "defaulted",
                    "ranking rows it would touch",
                ],
                nrows,
                "--rrr",
            ),
            "",
            f"{len(new):,} {'cell' if len(new) == 1 else 'cells'} over",
            f"{int(new['n'].sum()):,} training rows, touching {touched:,} of the",
            f"{len(rk_at):,} {PREFIX_AIRPORT} ranking departures.",
            "",
        ]
    else:
        lines += [
            "None: every cell that meets the rule is inside a cell the term already",
            "covers, so a prefix split would open nothing and touch no ranking row.",
            "",
        ]
    with_record = at[~at["LOBT_flt"].isna()]
    without = at[at["LOBT_flt"].isna()]
    lines += [
        "**What the two readings say.** The note's arithmetic holds and its reading of",
        "it does not. The taxi time does equal the lag in most of the rows of several",
        "prefixes with a record, and what that says is that those operators pushed back",
        f"within {NEAR_S:.0f} s of their scheduled off-block time -- which is the same",
        "statement, because the one quantity is the other. On a row where the block time",
        "is the scheduled time, the lag simply *is* the taxi, so the column reads as high",
        "as it does at a punctual operator and says nothing about the record. The median",
        "lag of the rows the note catches is what shows it: with a record it is",
        f"{secs(float(with_record.loc[with_record['near_lag'], 'lag'].median()))} against",
        f"a median recorded taxi-out of {secs(float(with_record['y'].median()))} over all",
        "of this airport's departures with a record; without a record it is",
        f"{secs(float(without.loc[without['near_lag'], 'lag'].median()))}, which is where",
        f"the {hours}-hour condition begins to bite, and",
        f"{pct(float(without['art'].mean()))} of those rows are defaulted under the",
        "term's reading against",
        f"{pct(float(with_record['art'].mean()), 3)} of the rows with one. The",
        f"{hours}-hour condition is not a refinement of the test; it is the test.",
        "",
        "**What the cells say.** The table above is the whole exposure of a prefix",
        "split under the coverage rule as it stands, measured on the training year and",
        "counted against the ranking set. It is printed so the size of the question is",
        "on record. This report does not decide it: the term's key, its coverage rule",
        "and the rows it predicts for are exactly what they were.",
    ]
    return lines


def section_statement(f_mech: dict, f_worth: dict, f_rank: dict) -> list[str]:
    dec = f_worth["Dec 2025"]
    jul = f_worth["Jul 2025, all"]
    return [
        "## What the term is, stated plainly",
        "",
        "The record-quality term is the RMSE-optimal prediction under the stated",
        "scoring rule for rows whose recorded value no aircraft produced. The",
        'ranking page states the rule in these words: teams "will be ranked based',
        "on the best Root Mean Square Error (RMSE) achieved across their",
        'submissions", and "the ranking script will issue an error if ... any rows',
        'are missing", so every row of the template is scored and none is',
        "excluded. For a row whose block time was defaulted to its schedule, the",
        "value the scorer holds is the lag, and the prediction that minimizes",
        "expected squared error is the mixture above -- the lag weighted by how",
        "often the cell defaults, and the cell's ordinary mean weighted by the",
        "rest. That is a conditional mean, not a taxi time; it is presented as",
        "such, kept in its own module, and the physical model is reported",
        "without it wherever it is reported with it.",
        "",
        "Two limits, stated so they need not be discovered. First, the term is",
        "fitted per (airport, cell) and covers only cells that showed the",
        f"mechanism in 2025 -- {', '.join(f_rank['covered_airports'])} alone. An airport whose",
        "feed starts defaulting block times in 2026 is not covered, and no rule",
        "fitted on 2025 could cover it; the exposure is one day-scale row per",
        "occurrence. Pooling across airports was the alternative, and the",
        "design table shows what it costs on the splits that were run.",
        f"Second, it is small: {dec['flagged']:,} rows of {dec['rows']:,} in the December",
        f"validation month, carrying {pct(dec['sse_share'])} of the airport-mean predictor's",
        f"squared error; {jul['flagged']:,} of {jul['rows']:,} in July, carrying {pct(jul['sse_share'])};",
        f"and {f_rank['rk_flagged']:,} of {f_rank['rk_rows']:,} ranking rows, {_month_split(f_rank)}. Its",
        *(
            [
                "whole effect on the leaderboard runs through those rows, and the July",
                "half of the test set does not see it at all.",
            ]
            if not f_rank["rk_flagged_jul"]
            else ["whole effect on the leaderboard runs through those rows."]
        ),
        "",
        "Beside those rows sits the rule of the plan-shifted departures: the term's",
        "form, with an add-one share, for departures whose take-off left their",
        "flight plan hours behind. Its share is estimated from",
        f"{_rows(f_rank['ps_n'])} of the training year, {f_rank['ps_long']:,} of them long, and",
        "its rest mean from the airport's ordinary departures; it is a conditional",
        "mean and not a taxi time. It was added after the class was seen in the ranking",
        "set's exposed fields, was fixed in advance of any evaluation, and takes",
        f"{_rows(f_rank['rk_plan_shift'])} of the ranking set"
        + (
            f", {_names(f_rank['rk_plan_shift_where'])}."
            if f_rank["rk_plan_shift"]
            else "."
        ),
        f"Its whole effect on the leaderboard runs through {'that row' if f_rank['rk_plan_shift'] == 1 else 'those rows'}.",
    ]


# --------------------------------------------------------------------------
# identifier guard


def check_no_identifiers(text: str, dep: pd.DataFrame) -> None:
    """Refuse to write a literal stand identifier from the challenge data."""
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
    dep, rk = load()
    model = TailModel().fit(dep)
    rule = PlanShiftModel(model).fit(dep)
    sp = splits(dep)

    mech_lines, f_mech = section_mechanism(dep)
    cell_lines, f_cells = section_cells(dep, model)
    design_lines, _ = section_design(dep, sp)
    worth_lines, f_worth = section_worth(dep, sp)
    rank_lines, f_rank = section_ranking(dep, rk, model, rule)
    f_rank["covered_airports"] = f_cells["covered_airports"]
    f_rank["ps_n"], f_rank["ps_long"] = rule.n, rule.long
    plan_lines = section_plan_shift(dep, rk, model, rule)
    prefix_lines = section_prefixes(dep, rk, model)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dec = f_worth["Dec 2025"]
    jul = f_worth["Jul 2025, all"]
    head = [
        "# The record-quality term",
        "",
        "Gate 1 of [`../docs/plan.md`](../docs/plan.md), the third term of the",
        "decomposition in [`../docs/modelling_notes.md`](../docs/modelling_notes.md):",
        "what the tail of the target is made of, which cells carry it, what the",
        "term predicts for them, and what it is worth.",
        "",
        "**This report has two generators, and each section says which wrote it.**",
        f"Everything above *{POOLED_SECTION}* is written by",
        f"`scripts/report_tails.py` from the data under `data/challenge/` on {generated},",
        "and every figure in it is computed there. That section is written into this",
        "file afterwards by `scripts/pooled_tails.py --report`, carries its own date,",
        "and its figures are computed there; `make tails` runs the two in that order.",
        'Before this header said so, the file claimed one generator and "every figure',
        'below is computed there", which stopped being true the day it gained the',
        "second. Aggregates only in both: no identifier from the challenge data",
        "appears, and both generators refuse to write one.",
        "",
        'The physical model the term sits on is not fitted here. The "with and',
        'without" figures use a stand-in -- the training mean per (airport,',
        "stand, runway), airport mean as fallback, fitted with the flagged rows",
        "excluded -- so the comparison has a concrete baseline;",
        "[`baseline.md`](baseline.md) carries it for the real model.",
        "",
        "## In one paragraph",
        "",
        f"The tail is one airport's data feed. {pct(f_mech['long_top_share'])} of the",
        f"training departures recorded above two hours are at {f_mech['long_top']}, and for",
        f"{pct(f_mech['long_top_near_share'])} of those the block time sits within a minute of",
        "the schedule while the take-off is hours later: the block time was",
        "defaulted, and the recorded taxi-out is the schedule lag, which every",
        "ranking departure exposes. The term covers the",
        f"{f_cells['covered_cells']} (airport, record, lag-band) cells where that happens at a",
        f"material rate -- all at {', '.join(f_cells['covered_airports'])} -- and predicts, for",
        "a row in one of them, the lag weighted by the cell's defaulting rate plus",
        "the cell's ordinary mean weighted by the rest. On the December 2025",
        f"validation month those are {dec['flagged']:,} rows of {dec['rows']:,} carrying",
        f"{pct(dec['sse_share'])} of the airport-mean squared error, and the stand-in physical",
        f"model scores {secs(dec['rmse_without'])} without the term and {secs(dec['rmse_with'])} with it;",
        f"on July, {jul['flagged']:,} rows of {jul['rows']:,} carrying {pct(jul['sse_share'])}, and",
        f"{secs(jul['rmse_without'])} against {secs(jul['rmse_with'])}."
        + (
            f" The {f_rank['rk_jul_airports']} July test airports"
            if not f_rank["rk_jul_covered"]
            else ""
        ),
        *(
            ["have no covered cell, so the term is inert on that half of the test set."]
            if not f_rank["rk_jul_covered"]
            else []
        ),
        f"In the ranking set it predicts for {f_rank['rk_flagged']:,} rows, {_month_split(f_rank)}.",
        "It is the RMSE-optimal prediction under the stated scoring rule for",
        "rows whose recorded value no aircraft produced, and nothing else.",
        "",
    ]
    footer = [
        "",
        *report_inputs_footer(REPO, challenge_inputs(DATA) + [REPO / PREREGISTRATION]),
        "",
        "Movement and flight data supplied by EUROCONTROL PRC and the OpenSky",
        "Network under the terms of the PRC Data Challenge 2026. This report uses",
        "no OpenStreetMap data; the repository's distance matrices elsewhere are",
        "derived from it (data (c) OpenStreetMap contributors, ODbL 1.0).",
    ]
    body = (
        head
        + mech_lines
        + [""]
        + cell_lines
        + [""]
        + design_lines
        + [""]
        + worth_lines
        + [""]
        + rank_lines
        + [""]
        + plan_lines
        + [""]
        + prefix_lines
        + [""]
        + section_statement(f_mech, f_worth, f_rank)
        + footer
    )
    text = "\n".join(body) + "\n"
    check_no_identifiers(text, dep)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
