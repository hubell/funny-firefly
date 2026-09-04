#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""How the excess term's inputs were built, and write ``reports/congestion.md``.

Gate 2 of ``docs/plan.md``. ``reports/baseline.md`` says what each
congestion block is worth on the ladder; this report says how the blocks
were constructed and whether the construction holds on both sides of the
data. It fits no model. What it measures:

1. **The leakage audit.** Every feature column against the phase-qualified
   raw columns it reads, grouped by block, and the two lists each phase is
   held to -- the table the test suite asserts, printed so a reader can
   check it without running the tests.
2. **The windows on both sides.** The same feature, the same airport, the
   training year against the ranking months: medians of the counts, and how
   many rows had a window truncated by the covered block. A feature that is
   constructed identically on both sides has the same shape on both sides,
   census for census; one that is not shows up here before it shows up on a
   leaderboard.
3. **The shared corridor.** Whether the weighted corridor load is a new
   column or the take-off count renamed: its correlation with throughput per
   airport, the share of subjects whose load is zero while the count is not,
   and how much of each airport's traffic the geometry could place at all.
4. **The runway configuration.** How many distinct configurations each
   airport runs, how long they hold, and how often the set is empty.

Every number is computed here; nothing is hand-typed. The report carries
aggregates and runway designators only, and the generator refuses to write a
literal stand identifier from either source.

Needs the challenge data under ``data/challenge/``, the distance matrices and
edge sets under ``data/processed/taxi_distance/`` (``make reports``). A minute
or two.

Usage
-----
    python scripts/report_congestion.py
    make congestion
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
sys.path.insert(0, str(REPO / "scripts"))

from run_baselines import load_edge_sets, load_matrix, num, pct, table  # noqa: E402
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.challenge import (  # noqa: E402
    AVAILABLE_FOR_ARRIVALS,
    FORBIDDEN_FEATURE_INPUTS,
    load_movements,
    load_ranking,
    to_pandas,
)
from taxiout.congestion import (  # noqa: E402
    configuration_sets,
    ARRIVAL_FEATURES,
    CONFIGURATION_FEATURES,
    COVERAGE_FEATURES,
    DEMAND_FEATURES,
    THROUGHPUT_FEATURES,
    WINDOWS_MIN,
)
from taxiout.corridor import CORRIDOR_FEATURES, degeneracy  # noqa: E402
from taxiout.ceiling import (  # noqa: E402
    GAP_BUCKETS_MIN,
    HALF_WINDOW_S,
    bucket_label,
    lag1_autocorrelation,
    oracle,
)
from taxiout.models import (  # noqa: E402
    ABOVE_SUBMISSION,
    LADDER_BY_KEY,
    TOP_RUNG,
)
from taxiout.planlags import PLAN_LAG_FEATURES  # noqa: E402
from taxiout.saturation import SATURATION_FEATURES  # noqa: E402
from taxiout.metar import (  # noqa: E402
    MAX_CARRY_H,
    VIS_HIGH_M,
    VIS_LOW_M,
    WEATHER_FEATURES,
    load_metar,
    regime_by_hour,
)
from taxiout.features import (  # noqa: E402
    AIRCRAFT_FEATURES,
    FEATURE_SOURCES,
    GEOMETRY_FEATURES,
    IDENTIFIER_FEATURES,
    MOVEMENT_COLUMNS,
    PLAN_FEATURES,
    QUALITY_FEATURES,
    REQUIRED_COLUMNS,
    TIME_FEATURES,
    airport_fallback,
    build_features,
    source_phase,
)
from taxiout.exposure import (  # noqa: E402
    ARRIVAL_COLUMNS,
    ARRIVAL_LATE_S,
    LAG_COLUMNS,
    LAG_HI_S,
    LAG_LO_S,
    MONTH_PAIRS,
    TAXI_IN_HI_S,
    TAXI_IN_LO_S,
    arrival_rows,
    departure_rows,
    lag_slopes,
    pooled_cost,
    joint_window_rows,
    shift_table,
    window_losses,
    worst_case,
)
from taxiout.validation import month_of  # noqa: E402

DATA = REPO / "data" / "challenge"
METAR = REPO / "data" / "raw" / "metar"
RESIDUALS = REPO / "data" / "processed" / "residuals"
OUT = REPO / "reports" / "congestion.md"

# The forward splits whose validation residuals the ladder keeps.
FORWARD_SPLITS = ("december", "july")
# Which forward split answers for which ranking month. December is the
# winter proxy for January -- the same choice the ladder's splits make -- and
# July answers for July.
SPLIT_FOR_MONTH = {"2026-01": "december", "2026-07": "july"}
TARGET = "TAXITIME_SEC_mvt"
# The rungs the ceiling is read on: the submission rung and the saturation
# rung above it, so the cheap queue proxy is measured against the ceiling of
# the rung it was added to; then the same two of the chain kept without the
# plan lags.
CEILING_RUNGS = tuple(
    dict.fromkeys(
        k
        for k in (
            TOP_RUNG.key,
            *(r.key for r in ABOVE_SUBMISSION),
            f"{TOP_RUNG.key}-nolag",
            *(f"{r.key}-nolag" for r in ABOVE_SUBMISSION),
        )
        if k in LADDER_BY_KEY
    )
)

BLOCKS = (
    ("identifiers", IDENTIFIER_FEATURES),
    ("time", TIME_FEATURES),
    ("geometry", GEOMETRY_FEATURES),
    ("join quality", QUALITY_FEATURES),
    ("flight plan", PLAN_FEATURES),
    ("aircraft", AIRCRAFT_FEATURES),
    ("plan lags", PLAN_LAG_FEATURES),
    ("throughput", THROUGHPUT_FEATURES),
    ("arrivals", ARRIVAL_FEATURES),
    ("demand", DEMAND_FEATURES),
    ("configuration", CONFIGURATION_FEATURES),
    ("coverage", COVERAGE_FEATURES),
    ("corridor", CORRIDOR_FEATURES),
    ("weather", WEATHER_FEATURES),
    ("saturation", SATURATION_FEATURES),
)

# The columns whose two-sided shape is printed. One per block, the widest
# window, because the widest window is the one most exposed to a boundary.
COMPARED = (
    "lag_eobt_rwy15m",
    "tko_all_60m",
    "tko_rwy_60m",
    "ldg_60m",
    "onb_60m",
    "txi_mean_60m",
    "sched_pm60m",
    "cfg_n_runways",
    "corr_dep_15m",
    "corr_arr_15m",
    "sat_gap90_15m",
    "sat_span3_s",
)


def load_side(side: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Departures and the whole movement frame of one side."""
    columns = sorted(set(REQUIRED_COLUMNS) | set(MOVEMENT_COLUMNS))
    if side == "training":
        dep = to_pandas(load_movements(DATA, columns, phase="DEP"))
        moves = to_pandas(load_movements(DATA, list(MOVEMENT_COLUMNS)))
    else:
        dep = to_pandas(load_ranking(DATA, columns, phase="DEP"))
        moves = to_pandas(load_ranking(DATA, list(MOVEMENT_COLUMNS)))
    return dep.reset_index(drop=True), moves.reset_index(drop=True)


# --------------------------------------------------------------------------
# sections


def section_audit() -> list[str]:
    rows = []
    for block, columns in BLOCKS:
        buckets: dict[str, set[str]] = {
            "bare": set(),
            "DEP": set(),
            "ARR": set(),
            "ext": set(),
        }
        for c in columns:
            for src in FEATURE_SOURCES[c]:
                phase, column = source_phase(src)
                key = (
                    "bare"
                    if phase is None
                    else phase
                    if phase in ("DEP", "ARR")
                    else "ext"
                )
                buckets[key].add(f"{phase}.{column}" if key == "ext" else column)
        rows.append(
            [
                block,
                str(len(columns)),
                ", ".join(f"`{c}`" for c in sorted(buckets["bare"])) or "--",
                ", ".join(f"`{c}`" for c in sorted(buckets["DEP"])) or "--",
                ", ".join(f"`{c}`" for c in sorted(buckets["ARR"])) or "--",
                ", ".join(f"`{c}`" for c in sorted(buckets["ext"])) or "--",
            ]
        )
    return [
        "## The leakage audit",
        "",
        "Every feature column is registered against the raw columns it reads,",
        "and the registry names the phase each is read from: a bare column is",
        "the subject's own row, `DEP.` another departure's, `ARR.` an arrival's.",
        "The rule is phase-specific -- a feature may read what the ranking set",
        "exposes *for the rows it reads from* -- so the test suite holds each",
        "phase to its own list, and this table is that assertion printed.",
        "",
        "Hidden for every departure, the subject's or a neighbour's, and never",
        "read under a bare or `DEP.` source: "
        + ", ".join(f"`{c}`" for c in FORBIDDEN_FEATURE_INPUTS)
        + ".",
        "Exposed for every arrival in the ranking set, with no nulls by the data",
        "review's count, and the only",
        "columns an `ARR.` source may name: "
        + ", ".join(f"`{c}`" for c in AVAILABLE_FOR_ARRIVALS)
        + ". `AOBT_3_flt` is read under no phase. The only source outside the",
        "challenge data is the Mesonet's METAR archive, under its own prefix,",
        "held to that archive's field list.",
        "",
        *table(
            [
                "block",
                "columns",
                "subject's own row",
                "other departures",
                "arrivals",
                "outside the challenge data",
            ],
            rows,
            "-r----",
        ),
        "",
        "The arrivals block is the one that reads a departure-hidden column --",
        "`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, from arrivals. The poison",
        "test in `tests/test_features.py` sets every departure's copy of those",
        "columns to nonsense on both the subject and the movement frame and",
        "asserts the whole frame is unchanged; then sets every arrival's",
        "taxi-in time to one second and asserts that the arrival features, and",
        "only they, move.",
    ]


def section_windows(train: pd.DataFrame, rank: pd.DataFrame) -> list[str]:
    """The same counts on both sides, by airport and month-type."""
    t_month = month_of(train["_takeoff"])
    r_month = month_of(rank["_takeoff"])
    # January against January, July against July, per airport
    rows = []
    for col in COMPARED:
        for months, label in (
            (("2025-01", "2026-01"), "January"),
            (("2025-07", "2026-07"), "July"),
        ):
            a = train[t_month == months[0]]
            b = rank[r_month == months[1]]
            for ap in sorted(set(b["icao"].astype(str))):
                x = a.loc[a["icao"].astype(str) == ap, col]
                y = b.loc[b["icao"].astype(str) == ap, col]
                if not len(y):
                    continue
                rows.append(
                    [
                        f"`{col}`",
                        label,
                        ap,
                        num(float(x.median()), 1),
                        num(float(y.median()), 1),
                        num(float(x.quantile(0.9)), 1),
                        num(float(y.quantile(0.9)), 1),
                    ]
                )
    trunc = []
    for name, f in (("training year", train), ("ranking months", rank)):
        n = len(f)
        trunc.append(
            [
                name,
                f"{n:,}",
                f"{int(f['win_truncated'].sum()):,} ({pct(float(f['win_truncated'].mean()), 3)})",
                f"{int(f['sched_truncated'].sum()):,} ({pct(float(f['sched_truncated'].mean()), 3)})",
            ]
        )
    return [
        "## The windows, on both sides",
        "",
        "Every congestion feature is a count or a mean over movements in a window",
        "anchored on the subject's take-off (its schedule, for demand), built by",
        "one function from each side's own movement frame. A feature built",
        "identically on both sides has the same shape on both sides -- and it",
        "should, because each ranking month is a census of every airport it",
        "contains (`reports/data_review.md`), so the counts need no coverage",
        "normalisation: a movement in a window means the same thing in January",
        "2026 as in January 2025 within any (airport, month) that is present.",
        "",
        "Median and 90th percentile, the same calendar month a year apart:",
        "",
        *table(
            [
                "feature",
                "month",
                "ICAO",
                "median 2025",
                "median 2026",
                "p90 2025",
                "p90 2026",
            ],
            rows,
            "---rrrr",
        ),
        "",
        "A window that reaches before the first instant of its covered block",
        "-- the first hour of January 2025 on the training side, of January and",
        "July 2026 on the ranking side -- is flagged rather than left short:",
        "",
        *table(
            [
                "side",
                "departures",
                "take-off window truncated",
                "schedule window truncated",
            ],
            trunc,
            "-rrr",
        ),
    ]


def fit_tail(train: pd.DataFrame, note: str):
    """The record-quality term fitted on the training side, if it is in the tree.

    Stream B owns ``taxiout.tails``; this reads the interface its docstring
    fixes and reimplements none of it. Without it the exposure section keeps
    every row and the report says so, rather than printing a mean that looks
    like it excluded the tail and did not.
    """
    try:
        from taxiout.tails import TailModel
    except ImportError:
        return (
            None,
            note + " (the record-quality term is not in the tree, so no row is flagged)",
        )
    return TailModel().fit(train), "unflagged by the record-quality term and " + note


def signed(x: float, digits: int = 0) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+,.{digits}f}"


def residual_rmse_by_airport(split: str, rung: str) -> pd.DataFrame:
    """Per-airport ex-flagged RMSE of one rung on one forward split."""
    r = load_residuals(split, rung)
    if r is None or not len(r):
        return pd.DataFrame(columns=["icao", "n", "rmse_s"])
    g = r.groupby(r["icao"].astype(str), sort=True)["residual"]
    return pd.DataFrame(
        {
            "n": g.size().astype(float),
            "rmse_s": g.apply(lambda v: float(np.sqrt(np.mean(v**2)))),
        }
    ).reset_index(names="icao")


def _window_sentence(losses: dict, plans: pd.DataFrame, joint_plans: pd.DataFrame) -> str:
    """What each window kept and what the joint one would have reported instead.

    Every number here is computed over exactly the cells the tables below
    compare: the smallest keep rate of any of the three lags' own windows,
    the smallest a joint window would leave, and the cell whose reported
    shift the joint window would move furthest.
    """
    narrowest, eobt, joint = [], [], []
    for _, row in plans.iterrows():
        for side, month in (
            ("training", row["training_month"]),
            ("ranking", row["ranking_month"]),
        ):
            cell = losses[side].loc[(row["icao"], month)]
            n = float(cell["in_record"])
            if n:
                narrowest.append(float(cell["in_window_min"]) / n)
                eobt.append(float(cell["in_window"]) / n)
                joint.append(float(cell["in_joint"]) / n)
    if not narrowest:
        return ""
    head = (
        f"Over the cells compared below, the narrowest of the three lags' own windows "
        f"keeps {pct(min(narrowest))} of the record set and the take-off-minus-EOBT "
        f"window -- the one the headline shift is read from -- keeps "
        f"{pct(min(eobt))}, where a joint window over the three would leave that "
        f"same mean only {pct(min(joint))} of its rows."
    )
    merged = plans.merge(
        joint_plans[["icao", "ranking_month", "lag_eobt_s_shift"]],
        on=["icao", "ranking_month"],
        suffixes=("", "_joint"),
        how="inner",
    ).dropna(subset=["lag_eobt_s_shift", "lag_eobt_s_shift_joint"])
    if merged.empty:
        return head
    merged["moved"] = (
        merged["lag_eobt_s_shift"] - merged["lag_eobt_s_shift_joint"]
    ).abs()
    w = merged.loc[merged["moved"].idxmax()]
    label = pd.Period(w["ranking_month"]).strftime("%B %Y")
    return (
        head + f" The difference is not cosmetic: a joint window reports {w['icao']}'s "
        f"{label} shift as {signed(float(w['lag_eobt_s_shift_joint']))} s where its own window "
        f"reports {signed(float(w['lag_eobt_s_shift']))} s."
    )


def section_exposure(
    dep_rows: dict, arr_rows: dict, train_target, tail_note: str
) -> list[str]:
    """How each ranking month's regime differs from its training-year twin."""
    train_dep, rank_dep = dep_rows["training"], dep_rows["ranking"]
    plans = shift_table(train_dep, rank_dep, LAG_COLUMNS)
    arrivals = shift_table(arr_rows["training"], arr_rows["ranking"], ARRIVAL_COLUMNS)
    slopes = lag_slopes(train_dep, train_target).set_index("icao")
    losses = {k: window_losses(v) for k, v in dep_rows.items()}
    # what a single window across all three lags would have reported instead,
    # so the argument for windowing each lag on its own carries its own number
    joint_plans = shift_table(
        joint_window_rows(train_dep), joint_window_rows(rank_dep), LAG_COLUMNS
    )

    lines = [
        "## Regime exposure: the ranking months against their twins",
        "",
        "The plan-lag block reads take-off minus a planning value fixed before",
        "departure, which is the taxi plus the pushback delay against that plan.",
        "The ranking months' departures take off later against their plans than the",
        "same calendar month of 2025 did, and the exposed fields cannot say which of",
        "the two moved: a ranking departure's block time is hidden, so the split is",
        "unobservable. They can bound it, and that is what this section does. It",
        "fits no model, and nothing here is a feature.",
        "",
        f"The rows: departures {tail_note}, with each lag then windowed to between "
        f"{LAG_LO_S / 60:.0f} and {LAG_HI_S / 60:.0f} minutes **on its own**. A departure taking off half an hour",
        "before its own off-block estimate is a record artefact rather than a fast",
        "taxi; one taking off two hours after it is the record-quality tail, which",
        "has its own term. The window is per lag and not across the three together,",
        "because a row whose schedule lag runs long is usually a row that took off",
        "late: windowing the three together drops it from the take-off-minus-EOBT",
        "mean as well, and drops more such rows on the side that ran later, which is",
        "the side whose shift is being measured. "
        + _window_sentence(losses, plans, joint_plans),
        "",
        "The control reads arrivals at the same airport-month -- no departure plan",
        "and nothing hidden, exposed on both sides -- with a taxi-in between "
        f"{TAXI_IN_LO_S / 60:.0f} and {TAXI_IN_HI_S / 60:.0f} minutes and a lateness within {ARRIVAL_LATE_S / 3600:.0f} hours either way,",
        "symmetric because an arrival landing early is ordinary where a departure",
        "taking off before its off-block estimate is not.",
    ]

    worst_by_month = {}
    for rank_month, train_month in MONTH_PAIRS:
        p_m = plans[plans["ranking_month"] == rank_month].set_index("icao")
        a_m = arrivals[arrivals["ranking_month"] == rank_month].set_index("icao")
        if p_m.empty:
            continue
        label = pd.Period(rank_month).strftime("%B %Y")
        twin = pd.Period(train_month).strftime("%B %Y")

        rows = []
        for icao in p_m.index:
            rows.append(
                [
                    str(icao),
                    f"{int(p_m.loc[icao, 'lag_eobt_s_n_train']):,}",
                    f"{int(p_m.loc[icao, 'lag_eobt_s_n_rank']):,}",
                    signed(float(p_m.loc[icao, "lag_eobt_s_shift"])),
                    signed(float(p_m.loc[icao, "lag_lobt_s_shift"])),
                    signed(float(p_m.loc[icao, "lag_sched_s_shift"])),
                    signed(float(a_m.loc[icao, "arr_taxi_in_s_shift"]))
                    if icao in a_m.index
                    else "n/a",
                    signed(float(a_m.loc[icao, "arr_late_s_shift"]))
                    if icao in a_m.index
                    else "n/a",
                ]
            )
        lines += [
            "",
            f"### {label} against {twin}",
            "",
            "The shift in each mean, in seconds. The first three are the departure's",
            "own plan; the last two are the control, and read no plan of the",
            "departure's at all:",
            "",
            *table(
                [
                    "ICAO",
                    f"rows, {twin}",
                    f"rows, {label}",
                    "take-off - EOBT",
                    "- LOBT",
                    "- schedule",
                    "arrivals' taxi-in",
                    "arrivals' lateness",
                ],
                rows,
                "-rrrrrrr",
            ),
        ]

        split = SPLIT_FOR_MONTH.get(rank_month)
        rmse = residual_rmse_by_airport(split, TOP_RUNG.key) if split else pd.DataFrame()
        if rmse.empty:
            lines += [
                "",
                "No residuals for this split under `data/processed/residuals/`, so the",
                "worst case is not computed; `python scripts/run_baselines.py` writes them.",
            ]
            continue
        rmse = rmse.set_index("icao")
        worst, rows = [], []
        for icao in p_m.index:
            shift = float(p_m.loc[icao, "lag_eobt_s_shift"])
            slope = float(slopes.loc[icao, "slope"]) if icao in slopes.index else np.nan
            base = float(rmse.loc[icao, "rmse_s"]) if icao in rmse.index else np.nan
            n = float(rmse.loc[icao, "n"]) if icao in rmse.index else np.nan
            w = worst_case(shift, slope, base)
            worst.append(
                {
                    "icao": icao,
                    "n": n,
                    "rmse_s": base,
                    "bias_s": w["bias_s"],
                    "cost_s": w["cost_s"],
                }
            )
            rows.append(
                [
                    str(icao),
                    num(slope, 2),
                    signed(shift),
                    signed(w["bias_s"]),
                    num(base, 1),
                    num(w["rmse_with_bias_s"], 1),
                    signed(w["cost_s"], 1),
                ]
            )
        worst = pd.DataFrame(worst)
        pooled = pooled_cost(worst)
        worst_by_month[rank_month] = {
            "per_airport": worst,
            "pooled": pooled,
            "split": split,
        }
        rows.append(
            [
                "**pooled**",
                "--",
                "--",
                "--",
                num(pooled["rmse_s"], 1),
                num(pooled["rmse_with_bias_s"], 1),
                signed(pooled["cost_s"], 1),
            ]
        )
        lines += [
            "",
            "The worst case: what the model would mispredict if the entire shift were",
            "pushback delay and none of it taxi. The slope is the airport's",
            "training-year ordinary least squares of taxi-out on take-off minus the",
            "first off-block estimate, over the same rows and the same window; the",
            "bias is that slope times the shift; and the cost is that bias added in",
            f"quadrature to `{TOP_RUNG.label}`'s ex-flagged RMSE on the {split} split at",
            "that airport. A per-airport constant bias enters that airport's squared",
            "error, so biases of opposite sign do not cancel in the pooled row:",
            "",
            *table(
                [
                    "ICAO",
                    "slope, s per s",
                    "shift, s",
                    "worst-case bias, s",
                    "RMSE (s)",
                    "with the bias (s)",
                    "cost (s)",
                ],
                rows,
                "-rrrrrr",
            ),
        ]

    lines += ["", *_exposure_reading(plans, arrivals, slopes, worst_by_month)]
    return lines


def _exposure_reading(plans, arrivals, slopes, worst_by_month) -> list[str]:
    """What the three quantities say, in the report's own numbers."""
    jan = plans[plans["ranking_month"] == "2026-01"].set_index("icao")
    jan_a = arrivals[arrivals["ranking_month"] == "2026-01"].set_index("icao")
    if jan.empty:
        return []
    moved = jan["lag_eobt_s_shift"].sort_values(ascending=False)
    top = [i for i in moved.index[:3] if i in jan_a.index]
    lo, hi = float(np.nanmin(slopes["slope"])), float(np.nanmax(slopes["slope"]))
    lines = [
        "### What it bounds",
        "",
        "Where the plan lag moves most, the surface does not. At the three airports",
        "whose January lag moved furthest -- "
        + ", ".join(f"{i} at {signed(float(moved[i]))} s" for i in top)
        + " --",
        "arrivals' taxi-in moved by "
        + ", ".join(signed(float(jan_a.loc[i, "arr_taxi_in_s_shift"])) for i in top)
        + " s, while arrivals' own lateness moved by "
        + ", ".join(signed(float(jan_a.loc[i, "arr_late_s_shift"])) for i in top)
        + " s.",
        "Taxi-in is the surface running slower or not; arrivals' lateness is the day",
        "running late or not. The second moves with the departure plan lag and the",
        "first barely moves, so the shift is mostly the day running late against its",
        "plans rather than the surface running slower.",
        "",
        "That is a reading of the control, not a measurement of the split, and what",
        "stands is the bound. Taking the whole shift as pushback delay -- the worst",
        "case, which the control says it is not -- the slopes run "
        f"{num(lo, 2)} to {num(hi, 2)} seconds of",
        "taxi-out per second of lag.",
    ]
    if not worst_by_month:
        return lines
    lines[-1] = lines[-1].rstrip(".") + ", and the cost in RMSE is:"
    for month, r in worst_by_month.items():
        label = pd.Period(month).strftime("%B %Y")
        per = (
            r["per_airport"]
            .dropna(subset=["cost_s"])
            .sort_values("cost_s", ascending=False)
        )
        if per.empty:
            continue
        worst_two = " and ".join(
            f"{signed(row.cost_s, 1)} s at {row.icao}" for row in per.head(2).itertuples()
        )
        lines.append(
            f"in {label}, {worst_two} at its two worst airports, "
            f"and {signed(r['pooled']['cost_s'], 1)} s pooled;"
        )
    if lines[-1].endswith(";"):
        lines[-1] = lines[-1][:-1] + "."
    lines += [
        "",
        "It is a bound on the exposure and not an estimate of it. The true shift is",
        "a mixture, no exposed field separates the two, and nothing in the model is",
        "corrected for it: a correction would be a guess at the mixture, fitted on",
        "nothing.",
    ]
    return lines


def section_corridor(train: pd.DataFrame) -> list[str]:
    d = degeneracy(train)
    rows = [
        [
            r["icao"],
            pct(r["placed"]),
            f"{r['corr_5m']:.2f}",
            f"{r['corr_15m']:.2f}",
            pct(r["zero_load_share_5m"]),
            pct(r["zero_load_share_15m"]),
            f"{r['median_load_5m']:.2f}",
            f"{r['median_load_15m']:.2f}",
        ]
        for _, r in d.iterrows()
    ]
    overall = d[d["icao"] == "all"].iloc[0]
    return [
        "## The shared corridor",
        "",
        "The one feature here no other entry has. Each other movement in the",
        "window is weighted by how much of the subject's own taxiway route its",
        "route shares, with every edge weighted by the inverse of the number of",
        "(stand, runway) routes that use it -- a property of the field, so a",
        "holding point every route funnels through weighs almost nothing and an",
        "apron spur three stands share weighs a third. Departures are anchored",
        "on their take-off, arrivals on their on-block, windows of 5 and 15",
        "minutes only.",
        "",
        "The degeneracy test that matters is not whether routes overlap -- at a",
        "hub they all do somewhere -- but whether the weighted load is a new",
        "column or the take-off count renamed. Per airport, the correlation of",
        "the departure-side load with the plain take-off count in the same",
        "window; the share of subjects with traffic in the window but none of it",
        "on their route; the median load in route-equivalents; and how much of",
        "the traffic the geometry could place on a route at all:",
        "",
        *table(
            [
                "ICAO",
                "placed",
                "corr with count, 5 min",
                "15 min",
                "zero load, traffic present, 5 min",
                "15 min",
                "median load, 5 min",
                "15 min",
            ],
            rows,
            "-rrrrrrr",
        ),
        "",
        f"Pooled, the load correlates with the count at {overall['corr_15m']:.2f} over",
        "fifteen minutes: the same traffic, a different question about it. Whether",
        "the question is worth asking is the corridor rung of the ladder in",
        "[`baseline.md`](baseline.md), which sits after throughput so that its",
        "step is exactly what the routing adds to the count.",
    ]


def section_configuration(train: pd.DataFrame) -> list[str]:
    rows = []
    icao = train["icao"].astype(str)
    for ap, g in train.groupby(icao, observed=True):
        cfg = g["cfg_runways"].astype(str)
        since = g["cfg_since_change_s"]
        rows.append(
            [
                str(ap),
                f"{cfg.nunique():,}",
                f"{cfg[cfg != ''].value_counts(normalize=True).head(3).sum():.0%}",
                pct(float((cfg == "").mean()), 2),
                num(float(g["cfg_n_runways"].median()), 0),
                num(float(since.median() / 60.0), 0),
                pct(float((since < 1800).mean())),
            ]
        )
    return [
        "## The runway configuration",
        "",
        "The set of designators with at least two movements in the trailing hour,",
        "from every movement's designator, and the seconds since that set last",
        "changed, capped at six hours. The floor of two suppresses the flicker of",
        "a stray movement on an idle runway; a runway the window has forgotten is",
        "a change like any other. The set itself is a diagnostic here and not a",
        "feature: as a categorical the ladder found it taking a large share of the",
        "model's splits and buying nothing on any split, so the model reads instead",
        "the count of runways in use and",
        "the seconds since the set last changed. Per airport: how many distinct sets the year",
        "runs, how much of the year the three commonest cover, how often the set",
        "is empty, the median number of runways in use, the median time since the",
        "last change, and the share of departures within half an hour of one:",
        "",
        *table(
            [
                "ICAO",
                "distinct sets",
                "top 3 cover",
                "empty",
                "runways in use",
                "median since change, min",
                "within 30 min of a change",
            ],
            rows,
            "-rrrrrr",
        ),
    ]


def section_weather(hourly: pd.DataFrame | None) -> list[str]:
    if hourly is None:
        return [
            "## The weather regime",
            "",
            "No METAR cache under `data/raw/metar/`; `python scripts/fetch_metar.py`",
            "fetches it, and the weather rung of the ladder is skipped until it does.",
        ]
    h = hourly.copy()
    h["month"] = h["hour"].dt.strftime("%Y-%m")
    rows = []
    for st, g in h.groupby("station", sort=True):
        rows.append(
            [
                str(st),
                f"{len(g):,}",
                pct(float((g["age_h"] == 0).mean())),
                pct(float((g["vis_band"] == 2).mean()), 2),
                pct(float((g["vis_band"] == 1).mean())),
                pct(float(g["precip"].mean())),
                pct(float(g["freezing"].mean())),
            ]
        )
    jan = []
    for st, g in h[h["month"].isin(["2025-01", "2026-01"])].groupby("station", sort=True):
        a = g[g["month"] == "2025-01"]
        b = g[g["month"] == "2026-01"]
        if not len(a) or not len(b):
            continue
        jan.append(
            [
                str(st),
                pct(float((a["vis_band"] == 2).mean()), 1),
                pct(float((b["vis_band"] == 2).mean()), 1),
                pct(float(a["precip"].mean())),
                pct(float(b["precip"].mean())),
                pct(float(a["freezing"].mean())),
                pct(float(b["freezing"].mean())),
            ]
        )
    return [
        "## The weather regime",
        "",
        "Gate 2b. Three flags per airport-hour from the Mesonet's METAR archive,",
        "and nothing continuous: a visibility band (2 below "
        f"{VIS_LOW_M / 1000:.1f} km, where low-visibility procedures are in force; 1 below "
        f"{VIS_HIGH_M / 1000:.0f} km; 0 above), precipitation at the aerodrome (present-weather",
        "groups in the vicinity are dropped), and freezing conditions (air at or",
        "below 0 C, or a freezing or frozen group). Each hour takes the worst of",
        f"its reports and an empty hour inherits the last for up to {MAX_CARRY_H} hours;",
        "a subject reads the hour its take-off falls in. The fetch and the terms",
        "are recorded in the cache's manifest and in the README.",
        "",
        "Coverage and the flag rates over the fetched period, per station:",
        "",
        *table(
            [
                "ICAO",
                "hours",
                "own report",
                "visibility below 1.5 km",
                "1.5 to 8 km",
                "precipitation",
                "freezing",
            ],
            rows,
            "-rrrrrr",
        ),
        "",
        "January 2025 against January 2026, the training year's winter month",
        "against the ranking set's. Where the second column is the larger, the",
        "ranking month ran in a regime the training year saw less of:",
        "",
        *table(
            [
                "ICAO",
                "< 1.5 km, 2025",
                "2026",
                "precipitation, 2025",
                "2026",
                "freezing, 2025",
                "2026",
            ],
            jan,
            "-rrrrrr",
        ),
    ]


def load_residuals(split: str, rung: str) -> pd.DataFrame | None:
    f = RESIDUALS / f"{split}_{rung}.parquet"
    if not f.exists():
        return None
    r = pd.read_parquet(f)
    return r[~r["flagged"].astype(bool)].reset_index(drop=True)


def section_ceiling() -> list[str]:
    lines = [
        "## The ceiling: what any queue model could still add",
        "",
        "An oracle diagnostic, never a feature. On each forward split's validation",
        "rows, over the rows the record-quality term leaves to the physical model,",
        "each rung's residuals are read against the residuals of the *other*",
        "departures on the same runway -- which are their taxi times minus what",
        "the model already knew, and which no admissible feature may read. Two",
        "questions. First, the lag-1 autocorrelation: the correlation between a",
        "departure's residual and the residual of the departure before it on the",
        "same runway, by the take-off gap between them. A queue is a run of",
        "departures late together; if the residuals of consecutive take-offs at",
        "ninety-second spacing are uncorrelated, there is no queue signal left to",
        "find. Second, the RMSE after regressing the residual on the mean residual",
        "of the other same-runway departures within "
        f"{HALF_WINDOW_S // 60} minutes either side, the",
        "subject excluded -- one least-squares fit, pooled, in sample. That is",
        "what a model could reach if it knew its neighbours' outcomes exactly, and",
        "it bounds what any queue model built from the runway sequence could add.",
        "Read on the rung the submission is fitted from and on the saturation",
        "rung above it -- so the cheap queue proxy is measured against the",
        "ceiling of the rung it was added to -- and on the same two of the chain",
        "kept without the plan lags.",
    ]
    any_found = False
    for split in FORWARD_SPLITS:
        loaded = {k: load_residuals(split, k) for k in CEILING_RUNGS}
        loaded = {k: v for k, v in loaded.items() if v is not None and len(v)}
        if not loaded:
            continue
        any_found = True
        lines += ["", f"### {split.title()} split", ""]
        # -- autocorrelation by gap, both rungs
        header = ["rung", "rows"]
        for lo, hi in GAP_BUCKETS_MIN:
            header += [f"{bucket_label(lo, hi)}, pairs", "corr"]
        rows = []
        for k, r in loaded.items():
            ac = lag1_autocorrelation(r).set_index("bucket")
            cells = [LADDER_BY_KEY[k].label, f"{len(r):,}"]
            for lo, hi in GAP_BUCKETS_MIN:
                b = ac.loc[bucket_label(lo, hi)]
                cells += [f"{int(b['n']):,}", num(float(b["corr"]), 2)]
            rows.append(cells)
        lines += [
            "Lag-1 autocorrelation of consecutive same-runway residuals, by the",
            "gap between the two take-offs:",
            "",
            *table(header, rows, "-r" + "rr" * len(GAP_BUCKETS_MIN)),
            "",
        ]
        # -- the oracle, both rungs
        rows = []
        oracles = {}
        for k, r in loaded.items():
            o = oracle(r)
            oracles[k] = o
            rows.append(
                [
                    LADDER_BY_KEY[k].label,
                    pct(o["with_neighbour"]),
                    num(o["slope"], 2),
                    num(o["rmse_before"], 1),
                    num(o["rmse_after"], 1),
                    num(o["rmse_before"] - o["rmse_after"], 1),
                ]
            )
        lines += [
            "The oracle: RMSE over the same rows before and after regressing the",
            "residual on its neighbours' mean residual. The last column is the",
            "ceiling -- the most any queue model reading the sequence could add:",
            "",
            *table(
                [
                    "rung",
                    "rows with a neighbour",
                    "slope",
                    "RMSE (s)",
                    "after the oracle (s)",
                    "ceiling (s)",
                ],
                rows,
                "-rrrrr",
            ),
            "",
        ]
        # -- per airport, on the rung the submission is fitted from
        top = TOP_RUNG.key
        if top in loaded:
            r = loaded[top]
            ac = lag1_autocorrelation(r, by_airport=True)
            first = bucket_label(*GAP_BUCKETS_MIN[0])
            ac0 = ac[ac["bucket"] == first].set_index("icao")
            per = oracles[top]["per_airport"].set_index("icao")
            rows = []
            for icao in per.index:
                rows.append(
                    [
                        str(icao),
                        f"{int(per.loc[icao, 'n']):,}",
                        num(float(ac0.loc[icao, "corr"]), 2)
                        if icao in ac0.index
                        else "--",
                        num(float(per.loc[icao, "rmse_before"]), 1),
                        num(float(per.loc[icao, "rmse_after"]), 1),
                        num(
                            float(
                                per.loc[icao, "rmse_before"] - per.loc[icao, "rmse_after"]
                            ),
                            1,
                        ),
                    ]
                )
            lines += [
                f"Per airport, {LADDER_BY_KEY[top].label}: the {first} autocorrelation",
                "and the oracle from the pooled fit:",
                "",
                *table(
                    [
                        "ICAO",
                        "rows",
                        f"corr, {first}",
                        "RMSE (s)",
                        "after (s)",
                        "ceiling (s)",
                    ],
                    rows,
                    "-rrrrr",
                ),
            ]
    if not any_found:
        lines += [
            "",
            "No residuals under `data/processed/residuals/`; `python scripts/run_baselines.py`",
            "writes them, and this section reads them on the next run.",
        ]
    return lines


def check_no_identifiers(text: str, stands: set[str], matrix: pd.DataFrame) -> None:
    values = {
        v.strip()
        for v in stands | set(matrix["stand_id"].dropna().astype(str))
        if len(v.strip()) >= 3 and re.search(r"[A-Za-z]", v) and re.search(r"[0-9]", v)
    }
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


def main() -> int:
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to report", file=sys.stderr)
        return 1
    matrix = load_matrix()
    edges = load_edge_sets()
    fallback = airport_fallback(matrix)
    hourly = None
    if METAR.exists() and any(METAR.glob("*.csv")):
        hourly = regime_by_hour(load_metar(METAR))
    sides = {}
    stands: set[str] = set()
    dep_rows: dict[str, pd.DataFrame] = {}
    arr_rows: dict[str, pd.DataFrame] = {}
    train_target = None
    tail_model = None
    tail_note = "carrying a flight-plan record"
    for side in ("training", "ranking"):
        print(f"{side} ...")
        dep, moves = load_side(side)
        stands |= set(dep["STAND_mvt"].dropna().astype(str))
        f = build_features(
            dep, matrix, fallback=fallback, movements=moves, edges=edges, weather=hourly
        )
        f["_takeoff"] = dep["MVT_TIME_UTC_mvt"].to_numpy()
        # the configuration set, for its own table below: a diagnostic, not a feature
        f["cfg_runways"] = configuration_sets(dep, moves).to_numpy()
        sides[side] = f

        # The regime-exposure section, from the raw rows rather than the frame:
        # it compares the ranking months against their training-year twins and
        # is not a feature of anything. The record-quality term is fitted on
        # the training side and predicted on both, as everywhere else.
        if side == "training":
            tail_model, tail_note = fit_tail(dep, tail_note)
            train_target = pd.to_numeric(dep[TARGET], errors="coerce")
        flagged = tail_model.flagged(dep) if tail_model is not None else None
        dep_rows[side] = departure_rows(dep, flagged)
        arr_rows[side] = arrival_rows(moves)
        del moves
    train, rank = sides["training"], sides["ranking"]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    head = [
        "# The excess term's inputs",
        "",
        "Gate 2 of [`../docs/plan.md`](../docs/plan.md): how the congestion",
        "features were built, and whether the construction holds on both sides",
        "of the data. What each is *worth* is the ladder in",
        "[`baseline.md`](baseline.md); this report fits no model. Generated by",
        "`scripts/report_congestion.py` from the data under `data/challenge/` on",
        f"{generated}; every figure below is computed there, and the generator",
        "refuses to write a literal stand identifier from either source.",
        "",
        "## In one paragraph",
        "",
        f"{len(train):,} training departures and {len(rank):,} ranking departures",
        "get the same "
        + str(sum(len(c) for _, c in BLOCKS[6:]))
        + " congestion columns, from windows of "
        + ", ".join(str(w) for w in WINDOWS_MIN)
        + " minutes anchored on the subject's take-off, built by one function",
        "from each side's own movement frame with the subject removed from its",
        "own count by identity. The same calendar month a year apart has the",
        "same shape on both sides. "
        f"{int(train['win_truncated'].sum()):,} training rows and "
        f"{int(rank['win_truncated'].sum()):,} ranking rows have a window",
        "truncated by the start of their covered block, and carry a flag for it.",
        "The shared-corridor load is a distinct column from the take-off count",
        "at every airport, and the ladder says what it is worth.",
        "",
    ]
    body = (
        head
        + section_audit()
        + [""]
        + section_windows(train, rank)
        + [""]
        + section_exposure(dep_rows, arr_rows, train_target, tail_note)
        + [""]
        + section_corridor(train)
        + [""]
        + section_configuration(train)
        + [""]
        + section_weather(hourly)
        + [""]
        + section_ceiling()
        + [
            "",
            "Distance matrices and edge sets derived from OpenStreetMap: data (c)",
            "OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied",
            "by EUROCONTROL PRC and the OpenSky Network under the terms of the PRC",
            "Data Challenge 2026. METARs from the Iowa Environmental Mesonet, Iowa",
            "State University.",
            "",
        ]
    )
    text = "\n".join(body)
    check_no_identifiers(text, stands, matrix)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
