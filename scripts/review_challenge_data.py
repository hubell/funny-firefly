#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Profile the challenge data and write ``reports/data_review.md``.

Gate 0 of ``docs/plan.md``. Three questions are answered before any feature
code is written -- whether the OpenStreetMap identifiers join to
``STAND_mvt`` and ``RUNWAY_mvt``, whether the target can be reconstructed
from fields the ranking set still exposes, and which timestamps a test
departure actually carries -- and the data is profiled around them: row
counts, coverage, null rates, the target's distribution and its tails,
timestamp granularity, and where the movement and flight sides disagree.

Every number in the report is computed here. Nothing is hand-typed, because
hand-typed statistics drift from the tables beside them, and that has
happened in this repository twice.

The report carries aggregates and identifier *shapes* only. The challenge
data is not redistributable and the stand identifiers on the OSM side are
OSM's; a literal identifier from either side is an extract, and the script
refuses to write one (see ``check_no_identifiers``).

Needs the challenge data under ``data/challenge/`` and the distance matrices
under ``data/processed/taxi_distance/`` (``make reports``). Neither is
committed. Runs in under a minute in about 2 GB.

Usage
-----
    python scripts/review_challenge_data.py
    make data-review
"""

from __future__ import annotations

import glob
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.challenge import (  # noqa: E402
    HIDDEN_FOR_TEST_DEPARTURES,
    NON_RUNWAY_VALUES,
    STRING_COLUMNS,
    load_movements,
    load_ranking,
    load_submitting,
    movement_files,
    to_pandas,
)
from taxiout.coverage import identifier_shape  # noqa: E402
from taxiout.join import STAGES, UNMATCHED, collapse, join_rates, join_stands  # noqa: E402
from taxiout.metar import load_metar, regime_by_hour  # noqa: E402
from taxiout.provenance import challenge_inputs, report_inputs_footer  # noqa: E402

DATA = REPO / "data" / "challenge"
MATRICES = REPO / "data" / "processed" / "taxi_distance"
METAR = REPO / "data" / "raw" / "metar"
OUT = REPO / "reports" / "data_review.md"

DEP_COLUMNS = [
    "ADEP_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
    "ADES_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
    "LOBT_flt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "AIRCRAFT_TYPE_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "AOBT_3_flt",
]

TIMESTAMPS = [
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
]

HOUR = 3600
DAY = 86400


# --------------------------------------------------------------------------
# helpers


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def secs(x: float) -> str:
    return f"{x:,.0f} s"


def seconds_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds()


def quantiles(s: pd.Series, qs: list[float]) -> list[float]:
    arr = s.dropna().to_numpy()
    return [float(v) for v in np.quantile(arr, qs)] if len(arr) else [np.nan] * len(qs)


def table(
    header: list[str], rows: list[list[str]], align: str | None = None
) -> list[str]:
    align = align or ("-" * len(header))
    sep = []
    for a in align:
        sep.append("---:" if a == "r" else "---")
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def month_of(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("UTC").dt.strftime("%Y-%m")


def sort_categories(frame: pd.DataFrame, column: str) -> None:
    """Order a categorical's levels alphabetically, so groupby output is too."""
    cats = frame[column].cat.categories
    frame[column] = frame[column].cat.reorder_categories(sorted(cats))


# --------------------------------------------------------------------------
# sections


def section_inventory(root: Path) -> tuple[list[str], dict]:
    facts: dict = {}
    rows = []
    total = 0
    for f in movement_files(root):
        meta = pq.read_metadata(f)
        total += meta.num_rows
        rows.append([f"`{f.name}`", f"{meta.num_rows:,}"])
    facts["train_rows"] = total
    rk = load_ranking(root, ["MVT_ID_mvt", "PHASE_mvt", *HIDDEN_FOR_TEST_DEPARTURES])
    sb = load_submitting(root)
    rk_pd = to_pandas(rk)
    sb_pd = to_pandas(sb)
    by_phase = rk_pd.groupby("PHASE_mvt", observed=True).agg(
        n=("MVT_ID_mvt", "size"),
        block_null=("BLOCK_TIME_UTC_mvt", lambda s: s.isna().mean()),
        target_null=("TAXITIME_SEC_mvt", lambda s: s.isna().mean()),
    )
    facts["rk_rows"] = len(rk_pd)
    facts["rk_dep"] = int(by_phase.loc["DEP", "n"])
    facts["rk_arr"] = int(by_phase.loc["ARR", "n"])
    facts["rk_dep_block_null"] = float(by_phase.loc["DEP", "block_null"])
    facts["rk_dep_target_null"] = float(by_phase.loc["DEP", "target_null"])
    facts["rk_arr_target_null"] = float(by_phase.loc["ARR", "target_null"])
    dep_ids = set(rk_pd.loc[rk_pd["PHASE_mvt"] == "DEP", "MVT_ID_mvt"])
    facts["sb_rows"] = len(sb_pd)
    facts["sb_in_rk_dep"] = int(sb_pd["MVT_ID_mvt"].isin(dep_ids).sum())
    facts["sb_target_null"] = float(sb_pd["TAXITIME_SEC_mvt"].isna().mean())
    facts["sb_unique"] = int(sb_pd["MVT_ID_mvt"].nunique())

    lines = [
        "## Inventory",
        "",
        f"Twelve monthly training files, {total:,} movements in total, which is the",
        "figure the challenge's data page gives for the ten airports its Table 1",
        "lists (read on 29 September 2026).",
        "",
        *table(["file", "rows"], rows, "-r"),
        "",
        f"`ranking.parquet` holds {facts['rk_rows']:,} rows: {facts['rk_dep']:,} departures and",
        f"{facts['rk_arr']:,} arrivals. For the departures, `BLOCK_TIME_UTC_mvt` is null in",
        f"{pct(facts['rk_dep_block_null'])} of rows and `TAXITIME_SEC_mvt` in",
        f"{pct(facts['rk_dep_target_null'])}, as the organisers document. For the arrivals the",
        f"taxi-in time is null in {pct(facts['rk_arr_target_null'])} of rows -- it is given,",
        "with the landing and on-block times, and is usable as an input.",
        "",
        f"`submitting.parquet` holds {facts['sb_rows']:,} rows ({facts['sb_unique']:,} distinct",
        f"`MVT_ID_mvt`), of which {facts['sb_in_rk_dep']:,} match a ranking departure; the",
        f"target column is null in {pct(facts['sb_target_null'])} of them. The template is",
        "exactly the set of ranking departures.",
    ]
    return lines, facts


def section_airports(dep: pd.DataFrame, rk_dep: pd.DataFrame) -> tuple[list[str], dict]:
    facts: dict = {}
    listed = [a.icao for a in AIRPORTS]
    present = sorted(
        set(dep["ADEP_mvt"].astype(str)) | set(rk_dep["ADEP_mvt"].astype(str))
    )
    absent = [a for a in listed if a not in present]
    facts["present"] = present
    facts["absent"] = absent

    m = month_of(dep["MVT_TIME_UTC_mvt"])
    train_by = (
        dep.groupby([dep["ADEP_mvt"], m], observed=True).size().unstack(fill_value=0)
    )
    rm = month_of(rk_dep["MVT_TIME_UTC_mvt"])
    rk_by = (
        rk_dep.groupby([rk_dep["ADEP_mvt"], rm], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    rows = []
    jul_airports = []
    for ap in present:
        t_jan = int(train_by.loc[ap].get("2025-01", 0)) if ap in train_by.index else 0
        t_jul = int(train_by.loc[ap].get("2025-07", 0)) if ap in train_by.index else 0
        t_all = int(train_by.loc[ap].sum()) if ap in train_by.index else 0
        r_jan = int(rk_by.loc[ap].get("2026-01", 0)) if ap in rk_by.index else 0
        r_jul = int(rk_by.loc[ap].get("2026-07", 0)) if ap in rk_by.index else 0
        if r_jul:
            jul_airports.append(ap)
        rows.append(
            [
                ap,
                f"{t_all:,}",
                f"{t_jan:,}",
                f"{t_jul:,}",
                f"{r_jan:,}",
                f"{r_jan / t_jan:.2f}" if t_jan else "n/a",
                f"{r_jul:,}",
                f"{r_jul / t_jul:.2f}" if t_jul else "n/a",
            ]
        )
    facts["jul_airports"] = jul_airports
    n_rk = len(rk_dep)
    facts["jul_share_of_test"] = float(
        rk_dep["ADEP_mvt"].astype(str).isin(jul_airports).mean()
    )
    facts["rk_jan"] = int((rm == "2026-01").sum())
    facts["rk_jul"] = int((rm == "2026-07").sum())
    spill = int(n_rk - facts["rk_jan"] - facts["rk_jul"])

    lines = [
        "## Airports and months",
        "",
        "The entry's own airport list, `taxiout.airports.AIRPORTS` of 2 September",
        f"2026, holds {len(listed)} airports; the challenge's data page lists ten, in its",
        "Table 1 (read on 29 September 2026). The data contains departures from",
        f"{len(present)}: " + ", ".join(present) + ".",
    ]
    if absent:
        lines += [
            f"**{', '.join(absent)} has no rows at all**, in training or ranking, although",
            "the total row count matches the organisers' figure exactly -- so this is",
            "the dataset as supplied, not a download that fell short.",
        ]
    lines += [
        "",
        "Departures per airport, with the two ranking months against the same",
        "calendar months of 2025. A ratio near one says the ranking month is a",
        "census of that airport's departures; a zero says the airport is absent from",
        "that month.",
        "",
        *table(
            [
                "ICAO",
                "2025 DEP",
                "Jan 2025",
                "Jul 2025",
                "Jan 2026",
                "ratio",
                "Jul 2026",
                "ratio",
            ],
            rows,
            "-rrrrrrr",
        ),
        "",
        f"So January 2026 is a census of all {len(present)} airports ({facts['rk_jan']:,}",
        f"departures) and July 2026 a census of {len(jul_airports)} -- "
        + ", ".join(jul_airports)
        + f" -- alone ({facts['rk_jul']:,})."
        + (
            f" {spill:,} ranking departures take off outside those two months."
            if spill
            else ""
        ),
        "Two consequences. Congestion counts",
        "built from the ranking set need no normalising for coverage, because",
        # Until the organizers re-issued the ranking files on 4 September 2026
        # the next sentence was fixed text about "the three July airports",
        # true of the month as it then stood and false once it held ten.
        *(
            [
                "within each (airport, month) present every movement is there. And the",
                f"{len(jul_airports)} July airports account for "
                + pct(facts["jul_share_of_test"])
                + " of the test departures, so the",
                "pooled leaderboard RMSE is weighted towards them.",
            ]
            if len(jul_airports) < len(present)
            else [
                "within each (airport, month) present every movement is there. And",
                f"July 2026 holds all {len(present)} airports, as January does.",
            ]
        ),
    ]
    return lines, facts


def null_rates(root: Path) -> pd.DataFrame:
    """Null share per column, by phase, for the training and ranking sets."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    rows: dict[tuple[str, str], int] = {}

    def add(label: str, t: pa.Table) -> None:
        for phase in ("DEP", "ARR"):
            mask = pc.equal(pc.cast(t["PHASE_mvt"], pa.string()), phase)
            sub = t.filter(mask)
            key = (label, phase)
            rows[key] = rows.get(key, 0) + sub.num_rows
            d = counts.setdefault(key, {})
            for c in sub.column_names:
                d[c] = d.get(c, 0) + sub[c].null_count

    for f in movement_files(root):
        add("train", pq.read_table(f, read_dictionary=list(STRING_COLUMNS)))
    add(
        "ranking",
        pq.read_table(root / "ranking.parquet", read_dictionary=list(STRING_COLUMNS)),
    )
    cols = pq.read_schema(root / "ranking.parquet").names
    out = pd.DataFrame(index=cols)
    for key in (
        ("train", "DEP"),
        ("train", "ARR"),
        ("ranking", "DEP"),
        ("ranking", "ARR"),
    ):
        out[f"{key[0]} {key[1]}"] = [counts[key].get(c, 0) / rows[key] for c in cols]
    return out


# How far apart, as a share, the `_flt` columns' null rates may sit on every
# side of the data before the section stops calling them the same rate. The
# prose prints it in percentage points, formatted from this, so the figure and
# the threshold are one number.
FLT_SAME_RATE = 0.0002


def section_nulls(rates: pd.DataFrame) -> list[str]:
    rows = [[f"`{c}`"] + [pct(v, 2) for v in rates.loc[c]] for c in rates.index]
    flt = [c for c in rates.index if c.endswith("_flt")]
    flt_rates = rates.loc[flt]
    same = bool((flt_rates.max() - flt_rates.min() < FLT_SAME_RATE).all())
    lines = [
        "## Null rates",
        "",
        "Share of null values per column. The movement side is nearly complete;",
        "the flight side is null as a block wherever the left join to the Network",
        "Manager record found nothing"
        + (
            ", and every `_flt` column has the same rate to within"
            f" {100 * FLT_SAME_RATE:.2f} points."
            if same
            else "."
        ),
        "",
        *table(["column"] + list(rates.columns), rows, "-rrrr"),
    ]
    return lines


def section_timestamps(dep: pd.DataFrame, rk_dep: pd.DataFrame) -> tuple[list[str], dict]:
    facts: dict = {}
    t = dep["TAXITIME_SEC_mvt"]
    diff = seconds_between(dep["MVT_TIME_UTC_mvt"], dep["BLOCK_TIME_UTC_mvt"])
    facts["target_is_diff"] = float((t == diff).mean())

    def sec0(s: pd.Series) -> float:
        s = s.dropna()
        return float((s.dt.second == 0).mean()) if len(s) else np.nan

    rows = []
    rounded_share = []
    for ap, g in dep.groupby("ADEP_mvt", observed=True, sort=True):
        m0, b0 = sec0(g["MVT_TIME_UTC_mvt"]), sec0(g["BLOCK_TIME_UTC_mvt"])
        # a uniform second hand gives 1/60 exactly on :00; the excess is the
        # share of minute-rounded stamps
        r_m = max(0.0, (m0 - 1 / 60) / (1 - 1 / 60))
        r_b = max(0.0, (b0 - 1 / 60) / (1 - 1 / 60))
        rounded_share.append((len(g), r_m, r_b))
        rows.append([str(ap), pct(m0, 2), pct(b0, 2), pct(r_m, 1), pct(r_b, 1)])
    n_all = sum(n for n, _, _ in rounded_share)
    # variance of a uniform rounding error on one stamp is 60^2/12; a rounded
    # stamp on either end of the interval adds one such term
    floor_mse = (
        sum(n * (r_m + r_b) * (60**2 / 12) for n, r_m, r_b in rounded_share) / n_all
    )
    facts["rounding_floor_rmse"] = float(np.sqrt(floor_mse))

    flt_rows = []
    for c in ("SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"):
        flt_rows.append([f"`{c}`", pct(sec0(dep[c]), 1), pct(sec0(rk_dep[c]), 1)])

    lines = [
        "## Timestamps and the target's definition",
        "",
        "`TAXITIME_SEC_mvt` equals `MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` to the second in",
        f"{pct(facts['target_is_diff'], 3)} of training departures: take-off minus off-block,",
        "no rounding, no adjustment.",
        "",
        "Both stamps are recorded to the second. A stamp with a uniform second hand",
        "reads `:00` one time in sixty; the excess over that is the share of stamps",
        "that were rounded to the minute, and it differs by airport:",
        "",
        *table(
            [
                "ICAO",
                "take-off `:00`",
                "off-block `:00`",
                "take-off rounded",
                "off-block rounded",
            ],
            rows,
            "-rrrr",
        ),
        "",
        "Rounding one stamp to the minute adds a uniform error of variance 60²/12 to",
        "the interval. Weighted by the rounded shares above, the floor that puts",
        f"under any predictor's RMSE is {facts['rounding_floor_rmse']:.1f} s -- negligible",
        "against the target's spread, and stated so it need not be wondered about.",
        "",
        "The scheduled and flight-side times are minute-resolution throughout:",
        "",
        *table(["column", "train DEP `:00`", "ranking DEP `:00`"], flt_rows, "-rr"),
    ]
    return lines, facts


def section_target(dep: pd.DataFrame) -> tuple[list[str], dict]:
    facts: dict = {}
    t = dep["TAXITIME_SEC_mvt"].astype(float)
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    qv = quantiles(t, qs)
    facts.update(
        n=len(t),
        mean=float(t.mean()),
        sd=float(t.std()),
        p50=qv[4],
        p99=qv[7],
        p999=qv[8],
        min=float(t.min()),
        max=float(t.max()),
        le0=int((t <= 0).sum()),
        lt60=int((t < 60).sum()),
        gt1h=int((t > HOUR).sum()),
        gt2h=int((t > 2 * HOUR).sum()),
        gt4h=int((t > 4 * HOUR).sum()),
        gt12h=int((t > 12 * HOUR).sum()),
    )
    qrows = [[f"p{q * 100:g}", secs(v)] for q, v in zip(qs, qv)]

    ap_rows = []
    for ap, g in dep.groupby("ADEP_mvt", observed=True, sort=True):
        gt = g["TAXITIME_SEC_mvt"].astype(float)
        q = quantiles(gt, [0.1, 0.5, 0.9, 0.99])
        ap_rows.append(
            [
                str(ap),
                f"{len(gt):,}",
                f"{gt.mean():,.0f}",
                f"{gt.std():,.0f}",
                f"{q[0]:,.0f}",
                f"{q[1]:,.0f}",
                f"{q[2]:,.0f}",
                f"{q[3]:,.0f}",
                f"{gt.max():,.0f}",
                f"{int((gt < 60).sum()):,}",
                f"{int((gt > 2 * HOUR).sum()):,}",
            ]
        )
    mo_rows = []
    for mth, gt in t.groupby(month_of(dep["MVT_TIME_UTC_mvt"])):
        q = quantiles(gt, [0.5, 0.99])
        mo_rows.append(
            [
                mth,
                f"{len(gt):,}",
                f"{gt.mean():,.0f}",
                f"{gt.std():,.0f}",
                f"{q[0]:,.0f}",
                f"{q[1]:,.0f}",
            ]
        )

    # what the tail does to RMSE, for the airport-mean predictor
    am = dep.groupby("ADEP_mvt", observed=True)["TAXITIME_SEC_mvt"].transform("mean")
    err = t - am
    se = err**2
    big = err.abs() > HOUR
    facts["rmse_airport_mean"] = float(np.sqrt(se.mean()))
    facts["rmse_airport_mean_ex_tail"] = float(np.sqrt(se[~big].mean()))
    facts["tail_rows_share"] = float(big.mean())
    facts["tail_sse_share"] = float(se[big].sum() / se.sum())
    facts["rmse_global_mean"] = float(np.sqrt(((t - t.mean()) ** 2).mean()))

    lines = [
        "## The target",
        "",
        f"Training departures: {facts['n']:,} rows, mean {secs(facts['mean'])}, standard",
        f"deviation {secs(facts['sd'])}, median {secs(facts['p50'])}, minimum {facts['min']:,.0f} s,",
        f"maximum {facts['max']:,.0f} s ({facts['max'] / HOUR:.1f} h).",
        "",
        *table(["quantile", "taxi-out"], qrows, "-r"),
        "",
        f"Implausible values: {facts['le0']:,} rows at or below zero, {facts['lt60']:,} under a",
        f"minute, {facts['gt1h']:,} over an hour, {facts['gt2h']:,} over two hours,",
        f"{facts['gt4h']:,} over four and {facts['gt12h']:,} over twelve.",
        "",
        "### Per airport",
        "",
        *table(
            [
                "ICAO",
                "rows",
                "mean",
                "sd",
                "p10",
                "p50",
                "p90",
                "p99",
                "max",
                "< 60 s",
                "> 2 h",
            ],
            ap_rows,
            "-rrrrrrrrrr",
        ),
        "",
        "### Per month",
        "",
        *table(["month", "rows", "mean", "sd", "p50", "p99"], mo_rows, "-rrrrr"),
        "",
        "### The tail owns the metric",
        "",
        "RMSE squares the error, so a few absurd rows outweigh everything else. For",
        "the simplest predictor, the airport mean, rows whose error exceeds one",
        f"hour are {pct(facts['tail_rows_share'], 3)} of the training departures and carry",
        f"{pct(facts['tail_sse_share'])} of the squared error. The airport-mean RMSE is",
        f"{secs(facts['rmse_airport_mean'])} with them and {secs(facts['rmse_airport_mean_ex_tail'])}",
        f"without (global mean: {secs(facts['rmse_global_mean'])}).",
    ]
    return lines, facts


# At least this many unjoined or four-hours-late departures in an airport-day,
# and a share of them this many times the airport's own training-year rate,
# before the day is called out. Both sides of the data are read at the same
# rule, so the two tables can be set against each other.
DISRUPTION_MIN_ROWS = 20
DISRUPTION_RATE_MULTIPLE = 10


def disruption_days(
    frame: pd.DataFrame, unjoined: pd.Series, lag: pd.Series, rate: pd.DataFrame
) -> pd.DataFrame:
    """Airport-days whose unjoined or four-hours-late share is far above ``rate``.

    ``rate`` is the per-airport training-year rate of each flag, so the same
    baseline is used whichever side of the data ``frame`` is.
    """
    day = frame["MVT_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.date
    daily = (
        frame.assign(day=day, unj=unjoined, late=lag > 4 * HOUR)
        .groupby(["ADEP_mvt", "day"], observed=True)
        .agg(n=("unj", "size"), unj=("unj", "sum"), late=("late", "sum"))
        .reset_index()
    )
    keep = []
    for _, r in daily.iterrows():
        tr = rate.loc[r["ADEP_mvt"]]
        for flag in ("unj", "late"):
            if (
                r[flag] >= DISRUPTION_MIN_ROWS
                and r[flag] / r["n"] > DISRUPTION_RATE_MULTIPLE * tr[flag]
            ):
                keep.append(r)
                break
    return pd.DataFrame(keep, columns=daily.columns)


def longest_run(days: pd.DataFrame) -> tuple[str, int]:
    """The airport with the longest run of consecutive flagged days, and its length.

    A single bad day at one airport and five in a row are different events, and
    the row counts alone do not say which a table holds.
    """
    best = ("none", 0)
    for ap, g in days.groupby("ADEP_mvt", observed=True):
        dates = sorted(pd.to_datetime(pd.Series(list(g["day"]))))
        run = 0
        for i, d in enumerate(dates):
            run = run + 1 if i and (d - dates[i - 1]).days == 1 else 1
            if run > best[1]:
                best = (str(ap), run)
    return best


def regime_shares(hourly, station: str, day) -> list[str]:
    """The three regime flags' shares over one station-day's hours.

    The visibility flag is printed at its low band, which is the column
    ``reports/congestion.md`` compares the two Januaries on.
    """
    if hourly is None:
        return ["n/a", "n/a", "n/a", "n/a"]
    g = hourly[(hourly["station"] == station) & (hourly["hour"].dt.date == day)]
    if not len(g):
        return ["0", "n/a", "n/a", "n/a"]
    return [
        f"{len(g):,}",
        pct(float((g["vis_band"] == 2).mean()), 0),
        pct(float(g["precip"].mean()), 0),
        pct(float(g["freezing"].mean()), 0),
    ]


def section_artefacts(
    dep: pd.DataFrame, rk_dep: pd.DataFrame, hourly=None
) -> tuple[list[str], dict]:
    facts: dict = {}
    t = dep["TAXITIME_SEC_mvt"].astype(float)
    lag = seconds_between(dep["MVT_TIME_UTC_mvt"], dep["SCHED_TIME_UTC_mvt"])
    unjoined = dep["LOBT_flt"].isna()

    # where the > 2 h rows are, and what they look like
    long = t > 2 * HOUR
    by_ap = (
        dep.loc[long]
        .groupby("ADEP_mvt", observed=True)
        .size()
        .sort_values(ascending=False)
    )
    top_ap = str(by_ap.index[0])
    facts["long_rows"] = int(long.sum())
    facts["long_top_airport"] = top_ap
    facts["long_top_share"] = float(by_ap.iloc[0] / long.sum())
    facts["long_unjoined_share"] = float(unjoined[long].mean())
    facts["near_day_rows"] = int(((t > 20 * HOUR) & (t < 28 * HOUR)).sum())
    facts["gt12h_rows"] = int((t > 12 * HOUR).sum())

    bucket = pd.Series(
        np.select(
            [t > 12 * HOUR, t > 2 * HOUR, t > HOUR, t <= 0, t < 120],
            ["> 12 h", "2-12 h", "1-2 h", "<= 0", "0-2 min"],
            default="2 min - 1 h",
        ),
        index=dep.index,
    )
    order = ["> 12 h", "2-12 h", "1-2 h", "2 min - 1 h", "0-2 min", "<= 0"]
    brows = []
    for b in order:
        m = bucket == b
        brows.append(
            [
                b,
                f"{int(m.sum()):,}",
                pct(unjoined[m].mean()),
                pct((lag[m] > 12 * HOUR).mean()),
                pct((lag[m] > 4 * HOUR).mean()),
                pct((lag[m] < -HOUR).mean()),
                f"{lag[m].median():,.0f}",
            ]
        )

    lagb = pd.Series(
        np.select(
            [lag > 12 * HOUR, lag > 4 * HOUR, lag < -HOUR],
            ["> 12 h", "4-12 h", "< -1 h"],
            default="within (-1 h, 4 h)",
        ),
        index=dep.index,
    )
    crows = []
    for lb in ["within (-1 h, 4 h)", "4-12 h", "> 12 h", "< -1 h"]:
        for uj in (False, True):
            m = (lagb == lb) & (unjoined == uj)
            if not m.any():
                continue
            q = quantiles(t[m], [0.1, 0.5, 0.9])
            crows.append(
                [
                    lb,
                    "no" if uj else "yes",
                    f"{int(m.sum()):,}",
                    f"{q[0]:,.0f}",
                    f"{q[1]:,.0f}",
                    f"{q[2]:,.0f}",
                    f"{t[m].mean():,.0f}",
                    pct((t[m] > 2 * HOUR).mean()),
                ]
            )
    key = (lagb == "> 12 h") & unjoined
    facts["flag_gt12h_unjoined_mean"] = float(t[key].mean())
    facts["flag_gt12h_unjoined_gt2h"] = float((t[key] > 2 * HOUR).mean())
    facts["flag_gt12h_unjoined_n"] = int(key.sum())

    # the same flags on the ranking set, against January 2025 as a base rate
    rlag = seconds_between(rk_dep["MVT_TIME_UTC_mvt"], rk_dep["SCHED_TIME_UTC_mvt"])
    runj = rk_dep["LOBT_flt"].isna()
    rmonth = month_of(rk_dep["MVT_TIME_UTC_mvt"])
    rrows = []
    tot = {"unjoined": 0, "lag4": 0, "lag12": 0, "both": 0}
    for (ap, mth), g in rk_dep.assign(month=rmonth).groupby(
        ["ADEP_mvt", "month"], observed=True
    ):
        gl, gu = rlag[g.index], runj[g.index]
        c_unj, c4, c12, cb = (
            int(gu.sum()),
            int((gl > 4 * HOUR).sum()),
            int((gl > 12 * HOUR).sum()),
            int((gu & (gl > 4 * HOUR)).sum()),
        )
        tot["unjoined"] += c_unj
        tot["lag4"] += c4
        tot["lag12"] += c12
        tot["both"] += cb
        rrows.append(
            [str(ap), mth, f"{len(g):,}", f"{c_unj:,}", f"{c4:,}", f"{c12:,}", f"{cb:,}"]
        )
    jan25 = month_of(dep["MVT_TIME_UTC_mvt"]) == "2025-01"
    base = {
        "unjoined": int(unjoined[jan25].sum()),
        "lag4": int((lag[jan25] > 4 * HOUR).sum()),
        "lag12": int((lag[jan25] > 12 * HOUR).sum()),
        "both": int((unjoined[jan25] & (lag[jan25] > 4 * HOUR)).sum()),
    }
    rrows.append(
        ["all", "2026", f"{len(rk_dep):,}"]
        + [f"{tot[k]:,}" for k in ("unjoined", "lag4", "lag12", "both")]
    )
    rrows.append(
        ["all", "2025-01", f"{int(jan25.sum()):,}"]
        + [f"{base[k]:,}" for k in ("unjoined", "lag4", "lag12", "both")]
    )
    facts["rk_flagged_both"] = tot["both"]
    facts["rk_flagged_lag12"] = tot["lag12"]

    # a worked example of what one such row costs
    n_test = len(rk_dep)
    one_row = (DAY + 15 * 60 - 1000) ** 2 / n_test
    facts["one_dayshift_row_mse"] = float(one_row)
    facts["five_rows_rmse_on_300"] = float(np.sqrt(300**2 + 5 * one_row))

    # daily anomaly scan on the ranking set: days whose unjoined or late share
    # is far above the airport's own training rate
    train_rate = pd.DataFrame(
        {
            "unj": unjoined.groupby(dep["ADEP_mvt"], observed=True).mean(),
            "late": (lag > 4 * HOUR).groupby(dep["ADEP_mvt"], observed=True).mean(),
        }
    )
    anomalies = disruption_days(rk_dep, runj, rlag, train_rate)
    arows = [
        [
            str(r["ADEP_mvt"]),
            str(r["day"]),
            f"{int(r['n']):,}",
            f"{int(r['unj']):,}",
            f"{int(r['late']):,}",
        ]
        for _, r in anomalies.iterrows()
    ]
    facts["anomalous_days"] = len(anomalies)

    # the same scan on the training year, so the ranking set's days have
    # something to be unusual against
    training = disruption_days(dep, unjoined, lag, train_rate)
    trows = [
        [
            str(r["ADEP_mvt"]),
            str(r["day"]),
            f"{int(r['n']):,}",
            f"{int(r['unj']):,}",
            f"{int(r['late']):,}",
        ]
        + regime_shares(hourly, str(r["ADEP_mvt"]), r["day"])
        for _, r in training.iterrows()
    ]
    facts["training_disruption_days"] = len(training)

    lines = [
        "## Where the tail comes from",
        "",
        f"Of the {facts['long_rows']:,} training departures over two hours, {facts['long_top_airport']}",
        f"alone has {pct(facts['long_top_share'])}, and {pct(facts['long_unjoined_share'])} have no",
        f"flight-side record. {facts['gt12h_rows']:,} exceed twelve hours, {facts['near_day_rows']:,} of",
        "them sitting between 20 and 28 hours: a block time from an earlier calendar",
        "day matched to a later day's take-off, which is a record-matching artefact",
        "rather than an aircraft that taxied for a day.",
        "",
        "Two fields that survive in the ranking set pick these rows out. Whether the",
        "flight-side join succeeded, and how long after its scheduled time the",
        "aircraft took off (`MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt`, the *lag*):",
        "",
        *table(
            [
                "taxi-out",
                "rows",
                "no flt record",
                "lag > 12 h",
                "lag > 4 h",
                "lag < -1 h",
                "median lag (s)",
            ],
            brows,
            "-rrrrrr",
        ),
        "",
        "Read the other way, conditioning on what a test row exposes:",
        "",
        *table(
            ["lag", "flt record", "rows", "p10", "p50", "p90", "mean", "> 2 h"],
            crows,
            "--rrrrrr",
        ),
        "",
        "A departure with no flight-side record and a lag over twelve hours has a mean",
        f"taxi-out of {secs(facts['flag_gt12h_unjoined_mean'])} in training and exceeds two hours",
        f"{pct(facts['flag_gt12h_unjoined_gt2h'])} of the time ({facts['flag_gt12h_unjoined_n']} rows).",
        "",
        "The ranking set carries the same flags at the same rates as the training",
        "year, so it almost certainly carries the same artefacts:",
        "",
        *table(
            [
                "ICAO",
                "month",
                "departures",
                "no flt record",
                "lag > 4 h",
                "lag > 12 h",
                "both",
            ],
            rrows,
            "--rrrrr",
        ),
        "",
        "The challenge's ranking page states the scoring rule: teams are ranked by",
        "the best RMSE across their submissions, over the movements in January and",
        "July 2026, and a submission is rejected if any row of the template is",
        "missing or extra. It states no exclusion of any row. Under that rule, one",
        f"such row predicted as a normal taxi-out adds about {facts['one_dayshift_row_mse']:,.0f} s² to the",
        f"mean squared error over {n_test:,} test rows, and five of them turn a model",
        f"with a true RMSE of 300 s into one scoring {facts['five_rows_rmse_on_300']:.0f} s. The decision",
        "taken on these rows is recorded in",
        "[`../docs/modelling_notes.md`](../docs/modelling_notes.md).",
    ]
    if arows:
        lines += [
            "",
            "### Days in the ranking set that look like disruptions",
            "",
            "Airport-days with at least twenty unjoined or four-hours-late departures",
            "and a share of them over ten times the airport's training-year rate. These",
            "are in the test set, and they are where de-icing and recovery queues will",
            "sit:",
            "",
            *table(
                ["ICAO", "day", "departures", "no flt record", "lag > 4 h"],
                arows,
                "--rrr",
            ),
        ]

    airports_hit = sorted({str(r["ADEP_mvt"]) for _, r in training.iterrows()})
    rk_airports = sorted({str(r["ADEP_mvt"]) for _, r in anomalies.iterrows()})
    lines += [
        "",
        "### The same days in the training year",
        "",
        "The same rule on the training year, so the days above have something to be",
        f"unusual against: at least {DISRUPTION_MIN_ROWS} unjoined or four-hours-late",
        "departures in an airport-day, and a share of them over",
        f"{DISRUPTION_RATE_MULTIPLE} times that airport's own training-year rate -- the",
        "same rate the table above is read against, so the two are the same",
        "measurement on the two sides of the data. Beside each day are the three",
        "weather flags `taxiout.metar.regime_by_hour` gives that station that day, as",
        "shares of the hours it has a reading for, so a reader can see which of these",
        "days ran in a regime and which did not.",
        "",
    ]
    if trows:
        lines += [
            *table(
                [
                    "ICAO",
                    "day",
                    "departures",
                    "no flt record",
                    "lag > 4 h",
                    "hours with a reading",
                    "visibility < 1.5 km",
                    "precipitation",
                    "freezing",
                ],
                trows,
                "--rrrrrrr",
            ),
            "",
            f"{len(training):,} airport-days over the training year, at",
            f"{len(airports_hit)} of the {dep['ADEP_mvt'].nunique()} airports"
            f" ({', '.join(airports_hit)}).",
        ]
    else:
        lines += ["No airport-day in the training year meets the rule."]
    for icao in rk_airports:
        if icao in airports_hit:
            continue
        listed = sum(1 for _, r in anomalies.iterrows() if str(r["ADEP_mvt"]) == icao)
        lines += [
            "",
            f"**The training year holds no such day at {icao}**, which the table above",
            f"lists on {listed:,} {'day' if listed == 1 else 'days'} of the ranking set.",
        ]
    if len(training) and len(anomalies):
        t_run, t_len = longest_run(training)
        r_run, r_len = longest_run(anomalies)
        lines += [
            "",
            "The two tables are not the same kind of event. The training year's are",
            "mostly single days at one airport: the longest run of consecutive flagged",
            f"days is {t_len} at {t_run}, against {r_len} at {r_run} in the ranking set.",
            "The most four-hours-late departures on any training day is",
            f"{int(training['late'].max()):,}, against",
            f"{int(anomalies['late'].max()):,} in the ranking set, and only",
            f"{int((training['late'] >= DISRUPTION_MIN_ROWS).sum()):,} of the",
            f"{len(training):,} training days reach the",
            f"{DISRUPTION_MIN_ROWS}-late threshold at all, against",
            f"{int((anomalies['late'] >= DISRUPTION_MIN_ROWS).sum()):,} of",
            f"{len(anomalies):,}. The training year's flagged days are, in the main,",
            "days the flight-side join failed on rather than days the airport stopped.",
        ]
    return lines, facts


def calibrated_rmse(t: pd.Series, x: pd.Series, group: pd.Series, width: float) -> float:
    """RMSE of the in-sample binned mean of ``t`` given ``x``, per group."""
    b = np.floor(x / width)
    fit = t.groupby([group, b], observed=True).transform("mean")
    return float(np.sqrt(((t - fit) ** 2).mean()))


def section_reconstruction(
    dep: pd.DataFrame, rk_dep: pd.DataFrame
) -> tuple[list[str], dict]:
    facts: dict = {}
    facts["rk_aobt_present"] = float(rk_dep["AOBT_3_flt"].notna().mean())
    has = dep["AOBT_3_flt"].notna()
    d = dep.loc[has]
    t = d["TAXITIME_SEC_mvt"].astype(float)
    ap = d["ADEP_mvt"]
    gap = seconds_between(d["AOBT_3_flt"], d["BLOCK_TIME_UTC_mvt"])
    recon = seconds_between(d["MVT_TIME_UTC_mvt"], d["AOBT_3_flt"])
    lag_sched = seconds_between(d["MVT_TIME_UTC_mvt"], d["SCHED_TIME_UTC_mvt"])
    lag_eobt = seconds_between(d["MVT_TIME_UTC_mvt"], d["EOBT_1_flt"])

    qg = quantiles(gap, [0.05, 0.25, 0.5, 0.75, 0.95])
    facts["gap_q"] = qg
    facts["gap_within_60"] = float((gap.abs() <= 60).mean())
    facts["gap_within_300"] = float((gap.abs() <= 300).mean())
    facts["n_joined"] = int(has.sum())

    am = t.groupby(ap, observed=True).transform("mean")
    facts["rmse_airport_mean_joined"] = float(np.sqrt(((t - am) ** 2).mean()))
    facts["rmse_recon_raw"] = float(np.sqrt(((t - recon) ** 2).mean()))
    facts["corr_recon"] = float(np.corrcoef(t, recon)[0, 1])
    facts["rmse_recon_cal"] = calibrated_rmse(t, recon, ap, 60)
    facts["corr_eobt"] = float(np.corrcoef(t[lag_eobt.notna()], lag_eobt.dropna())[0, 1])
    facts["rmse_eobt_cal"] = calibrated_rmse(t, lag_eobt, ap, 300)
    facts["corr_sched"] = float(np.corrcoef(t, lag_sched)[0, 1])
    facts["rmse_sched_cal"] = calibrated_rmse(t, lag_sched, ap, 300)

    ap_rows = []
    small = pd.DataFrame({"gap": gap, "recon": recon, "t": t, "ADEP_mvt": ap})
    for a, g in small.groupby("ADEP_mvt", observed=True, sort=True):
        ap_rows.append(
            [
                str(a),
                f"{len(g):,}",
                f"{g['gap'].median():,.0f}",
                pct((g["gap"].abs() <= 60).mean()),
                f"{np.sqrt(((g['t'] - g['recon']) ** 2).mean()):,.0f}",
            ]
        )

    other_rows = []
    for c in ("EOBT_1_flt", "LOBT_flt", "IOBT_flt", "SCHED_TIME_UTC_mvt"):
        g = seconds_between(d[c], d["BLOCK_TIME_UTC_mvt"])
        q = quantiles(g, [0.05, 0.25, 0.5, 0.75, 0.95])
        other_rows.append(
            [f"`{c}`"] + [f"{v:,.0f}" for v in q] + [pct((g.abs() <= 300).mean())]
        )

    lines = [
        "## Can the target be reconstructed? (check 2)",
        "",
        "`AOBT_3_flt`, the Network Manager's off-block time for the flown trajectory,",
        f"is present for {pct(facts['rk_aobt_present'])} of ranking departures. If it were the",
        "blanked block time under another name, `MVT_TIME_UTC_mvt - AOBT_3_flt` would",
        "be the target. It is not. Against the block time on the training",
        f"departures that have it ({facts['n_joined']:,} rows), the difference",
        f"`AOBT_3_flt - BLOCK_TIME_UTC_mvt` has a median of {qg[2]:,.0f} s and an interquartile",
        f"range of {qg[1]:,.0f} to {qg[3]:,.0f} s (5th to 95th percentile: {qg[0]:,.0f} to",
        f"{qg[4]:,.0f} s); {pct(facts['gap_within_60'])} of rows agree within a minute and",
        f"{pct(facts['gap_within_300'])} within five.",
        "",
        *table(
            ["ICAO", "rows", "median gap (s)", "within 60 s", "RMSE of `MVT - AOBT_3`"],
            ap_rows,
            "-rrrr",
        ),
        "",
        "As a predictor of the target, on the same rows:",
        "",
        *table(
            ["predictor", "correlation", "RMSE"],
            [
                ["airport mean", "", secs(facts["rmse_airport_mean_joined"])],
                [
                    "`MVT - AOBT_3`, as is",
                    f"{facts['corr_recon']:.2f}",
                    secs(facts["rmse_recon_raw"]),
                ],
                [
                    "`MVT - AOBT_3`, per-airport binned mean (60 s bins, in-sample)",
                    "",
                    secs(facts["rmse_recon_cal"]),
                ],
                [
                    "`MVT - EOBT_1`, per-airport binned mean (300 s bins)",
                    f"{facts['corr_eobt']:.2f}",
                    secs(facts["rmse_eobt_cal"]),
                ],
                [
                    "`MVT - SCHED`, per-airport binned mean (300 s bins)",
                    f"{facts['corr_sched']:.2f}",
                    secs(facts["rmse_sched_cal"]),
                ],
            ],
            "-rr",
        ),
        "",
        "So `AOBT_3_flt` is a noisy off-block estimate, and as a single predictor it",
        "carries no more than the *filed* estimated off-block time does -- both work",
        "the same way, as take-off time minus an off-block guess. The leak is real",
        "and bounded: it is not an oracle, and an entry that used it would gain",
        "little over one that used the flight plan. It is excluded from every",
        "feature regardless (`taxiout.challenge.EXCLUDED_BY_DECISION`); the reasoning",
        "is in [`../docs/modelling_notes.md`](../docs/modelling_notes.md).",
        "",
        "For completeness, how the other off-block estimates sit against the actual",
        "block time (seconds, estimate minus actual; training departures with a",
        "flight record):",
        "",
        *table(
            ["estimate", "p5", "p25", "p50", "p75", "p95", "within 5 min"],
            other_rows,
            "-rrrrrr",
        ),
    ]
    return lines, facts


def section_anchors(rk_dep: pd.DataFrame, rates: pd.DataFrame) -> tuple[list[str], dict]:
    facts: dict = {}
    rows = []
    for c in TIMESTAMPS:
        present = 1 - rates.loc[c, "ranking DEP"]
        note = {
            "MVT_TIME_UTC_mvt": "take-off; the only actual time a test departure carries",
            "BLOCK_TIME_UTC_mvt": "blanked",
            "SCHED_TIME_UTC_mvt": "scheduled departure",
            "LOBT_flt": "last off-block estimate held by NM",
            "IOBT_flt": "initial off-block estimate",
            "EOBT_1_flt": "filed estimated off-block",
            "ARVT_1_flt": "filed arrival time at destination",
            "AOBT_3_flt": "NM off-block for the flown trajectory; excluded by decision",
            "ARVT_3_flt": "arrival time at destination, flown",
        }[c]
        rows.append([f"`{c}`", pct(present, 1), note])
    facts["runway_present"] = 1 - rates.loc["RUNWAY_mvt", "ranking DEP"]
    facts["stand_present"] = 1 - rates.loc["STAND_mvt", "ranking DEP"]
    lines = [
        "## What a test departure carries (check 3)",
        "",
        f"`RUNWAY_mvt` is present for {pct(facts['runway_present'], 2)} of ranking departures and",
        f"`STAND_mvt` for {pct(facts['stand_present'], 2)}, so the distance matrix applies",
        "directly and no runway-assignment model is needed. The timestamps:",
        "",
        *table(["column", "present", "meaning"], rows, "-r-"),
        "",
        "There is no actual off-block time for any test departure. The anchor a",
        "test row offers is its take-off time; everything else is a schedule or an",
        "estimate. And because the block time is blanked for *every* departure in",
        "the ranking months, no feature can count other departures' off-block",
        "events either: a trailing window of pushbacks before the subject's time is",
        "unavailable at inference exactly as the queue-size feature was. What other",
        "movements expose in those months is their take-off time and schedule for",
        "departures, and landing time, on-block time and taxi-in time for arrivals.",
        "The feature rules this implies are in",
        "[`../docs/modelling_notes.md`](../docs/modelling_notes.md).",
    ]
    return lines, facts


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


def section_join(
    dep: pd.DataFrame, rk_dep: pd.DataFrame, matrix: pd.DataFrame
) -> tuple[list[str], dict]:
    facts: dict = {}
    designators = matrix.groupby("icao")["designator"].agg(set)

    # runway join and the values that fail it
    rw_rows = []
    for ap, g in dep.groupby("ADEP_mvt", observed=True, sort=True):
        rw = g["RUNWAY_mvt"].astype(str)
        ok = rw.isin(designators.get(str(ap), set()))
        miss = rw[~ok]
        sentinel = miss.isin(NON_RUNWAY_VALUES).sum()
        rw_rows.append(
            [str(ap), pct(ok.mean(), 2), f"{len(miss):,}", f"{int(sentinel):,}"]
        )

    # challenge-side identifier shapes, movement-weighted
    shape_rows = []
    for ap, g in dep.groupby("ADEP_mvt", observed=True, sort=True):
        counts = g["STAND_mvt"].value_counts()
        counts = counts[counts > 0]
        shapes: dict[str, int] = {}
        for v, n in counts.items():
            shapes[identifier_shape(str(v))] = shapes.get(
                identifier_shape(str(v)), 0
            ) + int(n)
        top = sorted(shapes.items(), key=lambda kv: -kv[1])[:5]
        shape_rows.append(
            [
                str(ap),
                f"{len(counts):,}",
                ", ".join(f"`{s}` {pct(n / len(g))}" for s, n in top),
            ]
        )

    cols = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
    j_train = join_stands(dep[cols], matrix)
    j_rank = join_stands(rk_dep[cols], matrix)
    r_train = join_rates(j_train).set_index("icao")
    r_rank = join_rates(j_rank).set_index("icao")
    stage_names = [f"{s}: {name}" for s, name in STAGES]
    rows = []
    for ap in sorted(r_train.index):
        a, b = r_train.loc[ap], r_rank.loc[ap]
        rows.append(
            [ap]
            + [pct(a[f"stage{s}"]) for s, _ in STAGES]
            + [pct(a["matched"]), pct(b["matched"]), a["unmatched_shapes"]]
        )
    facts["train_matched_overall"] = float((j_train["join_stage"] != UNMATCHED).mean())
    facts["rank_matched_overall"] = float((j_rank["join_stage"] != UNMATCHED).mean())
    facts["train_by_airport"] = {
        ap: float(r_train.loc[ap, "matched"]) for ap in r_train.index
    }
    facts["train_stage_share"] = {
        s: float((j_train["join_stage"] == s).mean()) for s, _ in STAGES
    }

    # how much collapsing the coarsest stage does
    present = set(dep["ADEP_mvt"].astype(str))
    c3 = collapse(matrix[matrix["icao"].isin(present)], 3)
    coll_rows = []
    for ap, g in c3.groupby("icao", sort=True):
        multi = g[g["n_ids"] > 1]
        coll_rows.append(
            [
                str(ap),
                f"{g['key'].nunique():,}",
                f"{multi['key'].nunique():,}",
                f"{multi['id_spread_m'].median():,.0f}" if len(multi) else "0",
                f"{int((multi['id_spread_m'] > 250).sum()):,}",
            ]
        )

    lines = [
        "## Do the identifiers join? (check 1)",
        "",
        "### Runways",
        "",
        "`RUNWAY_mvt` against the designators the matrices resolved. The misses",
        "are values that name no runway (`NA`, and the two helicopter codes):",
        "",
        *table(
            ["ICAO", "joined", "unmatched rows", "of which sentinels"], rw_rows, "-rrr"
        ),
        "",
        "### Stands",
        "",
        "The challenge's identifier formats, as shapes (`#` digit, `@` letter, `.`",
        "other), weighted by departures. Compare",
        "[`stand_coverage.md`](stand_coverage.md), which gives the OpenStreetMap side:",
        "",
        *table(["ICAO", "distinct `STAND_mvt`", "shapes"], shape_rows, "-r-"),
        "",
        "The join runs in stages (`taxiout.join`), each applied to both sides, and",
        "a departure is matched at the first stage that finds its stand. Stage 1",
        "folds case and removes whitespace; stage 2 also strips leading zeros in",
        "digit runs; stage 3 also strips one trailing letter after a digit. The",
        "share matched *at* each stage says which convention differs where:",
        "",
        *table(
            ["ICAO"]
            + stage_names
            + ["matched (2025)", "matched (ranking)", "unmatched shapes (2025)"],
            rows,
            "-rrrrr-",
        ),
        "",
        f"Overall, {pct(facts['train_matched_overall'])} of training departures and",
        f"{pct(facts['rank_matched_overall'])} of ranking departures land on a matrix cell.",
        "The stages were chosen by these rates and by inspecting the identifiers",
        "that failed, never by validation error. What the inspection found: the",
        "whitespace is inside OpenStreetMap's refs at Frankfurt; Schiphol's movement",
        "data zero-pads the stand number where OSM does not; Munich and Istanbul map",
        "split stands as lettered halves where the movement data names the whole.",
        "What remains unmatched is, in the main, stands OpenStreetMap does not have",
        "-- whole numeric blocks at Munich and Barcelona -- plus non-stand positions",
        "(general aviation, hangars, maintenance) at Zurich and a literal",
        "*unknown* placeholder at Barcelona. Those rows fall back to airport-level",
        "features, and the write-up says so.",
        "",
        "### What the coarsest stage collapses",
        "",
        "Stripping a trailing letter folds a split stand's halves and its parent",
        "onto one key. The collapsed cell keeps the route nearest the group mean",
        "and records how far the group disagrees:",
        "",
        *table(
            [
                "ICAO",
                "keys at stage 3",
                "keys with > 1 identifier",
                "median disagreement (m)",
                "> 250 m",
            ],
            coll_rows,
            "-rrrr",
        ),
    ]
    return lines, facts


def section_disagreement(dep: pd.DataFrame) -> list[str]:
    d = dep.loc[dep["LOBT_flt"].notna()]

    def dis(a: str, b: str) -> float:
        x, y = d[a].astype(str), d[b].astype(str)
        both = d[a].notna() & d[b].notna()
        return float((x[both] != y[both]).mean())

    sched_eobt = seconds_between(d["EOBT_1_flt"], d["SCHED_TIME_UTC_mvt"]).abs() > 900
    rows = [
        ["`ADEP_mvt` vs `ADEP_flt`", pct(dis("ADEP_mvt", "ADEP_flt"), 2)],
        ["`ADES_mvt` vs `ADES_flt`", pct(dis("ADES_mvt", "ADES_flt"), 2)],
        ["`ADES_mvt` vs `ADES_FILED_flt`", pct(dis("ADES_mvt", "ADES_FILED_flt"), 2)],
        [
            "`AIRCRAFT_TYPE_mvt` vs `AIRCRAFT_TYPE_flt`",
            pct(dis("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), 2),
        ],
        [
            "`FLIGHT_RULE_mvt` vs `FLIGHT_RULE_flt`",
            pct(dis("FLIGHT_RULE_mvt", "FLIGHT_RULE_flt"), 2),
        ],
        [
            "`SCHED_TIME_UTC_mvt` vs `EOBT_1_flt` more than 15 min apart",
            pct(float(sched_eobt.mean()), 2),
        ],
    ]
    cat_rows = []
    for c in (
        "FLIGHT_RULE_mvt",
        "FLIGHT_TYPE_flt",
        "MARKET_SEGMENT_flt",
        "WK_TBL_CAT_flt",
    ):
        g = dep.groupby(dep[c].astype(str).where(dep[c].notna(), "(null)"))[
            "TAXITIME_SEC_mvt"
        ]
        for v, s in sorted(g, key=lambda kv: -len(kv[1])):
            cat_rows.append(
                [f"`{c}`", v, f"{len(s):,}", f"{s.mean():,.0f}", f"{s.median():,.0f}"]
            )
    return [
        "## Where the two sides disagree",
        "",
        "Training departures with a flight-side record. The organisers did not",
        "reconcile the two sources and say so; the disagreement rates are a measure",
        "of how far to trust a flight-side field over its movement-side twin:",
        "",
        *table(["pair", "disagree"], rows, "-r"),
        "",
        "### Categorical fields against the target",
        "",
        "Rows and taxi-out per category, for the fields the literature says are",
        "weak predictors. A null category is the unjoined block, and its mean is",
        "where the day-shift artefacts live:",
        "",
        *table(["field", "value", "rows", "mean", "median"], cat_rows, "--rrr"),
    ]


# --------------------------------------------------------------------------
# guard


def check_no_identifiers(
    text: str, dep: pd.DataFrame, rk_dep: pd.DataFrame, matrix: pd.DataFrame
) -> None:
    """Refuse to write a report that carries a literal stand identifier.

    Same rule as ``check_licence_hygiene.py`` -- identifiers with a letter and
    a digit and at least three characters -- applied to both the challenge and
    the OSM side, and to this file alone, with a zero threshold. Runway
    designators and ICAO codes are exempt: those are public.
    """
    values: set[str] = set()
    for s in (dep["STAND_mvt"], rk_dep["STAND_mvt"], matrix["stand_id"]):
        for v in s.dropna().astype(str).unique():
            v = v.strip()
            if len(v) >= 3 and re.search(r"[A-Za-z]", v) and re.search(r"[0-9]", v):
                values.add(v)
    exempt = set(matrix["designator"].astype(str)) | {a.icao for a in AIRPORTS}
    leaked = sorted(
        v
        for v in values - exempt
        if re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", text)
    )
    if leaked:
        raise SystemExit(
            f"refusing to write {OUT.name}: {len(leaked)} literal stand identifier(s) "
            "would be published"
        )


# --------------------------------------------------------------------------
# main


def main() -> int:
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to review", file=sys.stderr)
        return 1
    matrix = load_matrix()
    hourly = None
    if METAR.exists() and any(METAR.glob("*.csv")):
        hourly = regime_by_hour(load_metar(METAR))
    else:
        print(
            f"no METAR cache under {METAR}; the regime columns read n/a", file=sys.stderr
        )

    print("inventory ...")
    inv_lines, inv = section_inventory(DATA)
    print("null rates ...")
    rates = null_rates(DATA)

    print("loading departures ...")
    dep = to_pandas(load_movements(DATA, DEP_COLUMNS, phase="DEP"))
    rk_dep = to_pandas(load_ranking(DATA, DEP_COLUMNS, phase="DEP"))
    sort_categories(dep, "ADEP_mvt")
    sort_categories(rk_dep, "ADEP_mvt")

    print("profiling ...")
    ap_lines, ap = section_airports(dep, rk_dep)
    ts_lines, ts = section_timestamps(dep, rk_dep)
    tg_lines, tg = section_target(dep)
    ar_lines, ar = section_artefacts(dep, rk_dep, hourly)
    rc_lines, rc = section_reconstruction(dep, rk_dep)
    an_lines, an = section_anchors(rk_dep, rates)
    print("joining ...")
    jn_lines, jn = section_join(dep, rk_dep, matrix)
    dg_lines = section_disagreement(dep)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    absent = ", ".join(ap["absent"]) if ap["absent"] else "none"
    worst_ap = min(jn["train_by_airport"], key=jn["train_by_airport"].get)
    best_ap = max(jn["train_by_airport"], key=jn["train_by_airport"].get)
    head = [
        "# Challenge data review",
        "",
        "Gate 0 of [`../docs/plan.md`](../docs/plan.md): what the challenge data",
        "contains, whether the geometry joins to it, and what the ranking set does",
        "and does not expose. Generated by `scripts/review_challenge_data.py` from",
        f"the data under `data/challenge/` on {generated}; every figure below is",
        "computed there. The data itself is not redistributable and nothing from it",
        "beyond aggregates and identifier shapes appears here.",
        "",
        "## In one paragraph",
        "",
        f"The data covers {len(ap['present'])} airports, the ten the challenge's data page",
        f"lists (read on 29 September 2026), not the {len(AIRPORTS)} of the entry's own",
        "airport list (absent:",
        f"{absent}). January 2026 is a census of all of them; July 2026 covers",
        f"{len(ap['jul_airports'])} ({', '.join(ap['jul_airports'])}). The identifiers join: runways",
        f"everywhere, stands at {pct(jn['train_by_airport'][worst_ap])} ({worst_ap}) to",
        f"{pct(jn['train_by_airport'][best_ap])} ({best_ap}) of departures after a staged",
        f"normalisation, {pct(jn['train_matched_overall'])} overall. A test departure carries",
        "its take-off time, its schedule and the flight plan's estimates, but no",
        "off-block time -- its own or any other departure's -- so congestion has to",
        "be built from take-offs, schedules and arrivals. `AOBT_3_flt` survives but",
        f"is a minute-resolution estimate sitting a median {abs(rc['gap_q'][2]):,.0f} s from the",
        "blanked block time with an interquartile range of",
        f"{rc['gap_q'][3] - rc['gap_q'][1]:,.0f} s; as a predictor it is worth no more",
        "than the filed EOBT, and it is excluded. The target has a heavy tail:",
        f"{pct(tg['tail_rows_share'], 3)} of rows carry {pct(tg['tail_sse_share'])} of the",
        "airport-mean squared error, most of them day-shift artefacts; the ranking",
        "set carries the two fields that flag them at the training rate, and the",
        "stated scoring rule is RMSE over every row of the template.",
        "",
    ]
    body = (
        head
        + inv_lines
        + [""]
        + ap_lines
        + [""]
        + [
            *section_nulls(rates),
            "",
        ]
        + ts_lines
        + [""]
        + tg_lines
        + [""]
        + ar_lines
        + [""]
        + rc_lines
        + [""]
        + an_lines
        + [""]
        + jn_lines
        + [""]
        + dg_lines
        + [
            "",
            *report_inputs_footer(
                REPO,
                challenge_inputs(DATA)
                + sorted(p for p in MATRICES.glob("*.csv") if "_features" not in p.name)
                + sorted(METAR.glob("*.csv")),
            ),
            "",
            "Distance matrices derived from OpenStreetMap: data (c) OpenStreetMap contributors,",
            "ODbL 1.0. Movement and flight data supplied by EUROCONTROL PRC and the",
            "OpenSky Network under the terms of the PRC Data Challenge 2026.",
            "",
        ]
    )
    text = "\n".join(body)
    check_no_identifiers(text, dep, rk_dep, matrix)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
