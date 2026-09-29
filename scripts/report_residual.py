#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""What the model's remaining error is, and write ``reports/residual.md``.

``reports/baseline.md`` says what each block is worth; this says what is left
after all of them. It fits no model: every figure comes from the stored
validation residuals of the submission rung, joined back to the validation
month on the movement identifier.

The claim it exists to support is that the residual is a **thin tail of
individual holds** rather than a queue everybody stands in, and that the
classes it is enriched in are the ones where an individual hold lives. Four
measurements say so:

1. **The shape of the tail.** How few rows carry how much of the squared
   error, and how one-sided they are. A conditional mean sitting between two
   modes under-predicts the held rows, so the tail should be mostly
   under-predictions, and it is.
2. **What it is enriched in.** Heavies, departures with no Network Manager
   record, and departures whose stand was taken by a different aircraft
   twenty to forty-five minutes before take-off -- the rows the
   stand-occupancy bound speaks about. Each as a share of rows, of the worst
   one percent of residuals, and of the squared error.
3. **The error by plan lag.** It grows with the lag, as a conditional mean
   between a gate delay and a hold must.
4. **The largest rows.** The handful that carry a tenth of a split's squared
   error each, as airport, recorded taxi and prediction -- never a stand or a
   movement identifier.

The per-airport table at the foot is what the paper's results section reads
for its per-airport prose.

Two further sections describe rather than decide. **Heathrow's July** cuts the
submitted rung's July residuals at one airport five ways -- by take-off hour,
runway, plan-lag band, callsign prefix, and whether the row has a Network
Manager record -- with December at the same airport beside each cut, because
that airport's July figure is the one that stands out. **December by weather
flag** cuts the December residuals by the three regime flags
``taxiout.metar.regime_by_hour`` produces. Neither fits anything and neither
changes a feature; each difference their prose states carries the day-clustered
bootstrap interval ``taxiout.validation.cluster_bootstrap`` gives, which is what
the standing rule asks of a question put inside one split.

Every number is computed here; nothing is hand-typed. The report carries
aggregates and runway designators only, and the generator refuses to write a
literal stand identifier.

Needs the challenge data under ``data/challenge/``, the distance matrices from
``make reports``, and the residuals ``scripts/run_baselines.py`` writes under
``data/processed/residuals/``. A few minutes.

Usage
-----
    python scripts/report_residual.py
    make residual
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_baselines import (  # noqa: E402
    check_no_identifiers,
    load_matrix,
    num,
    pct,
    table,
)
from taxiout.provenance import challenge_inputs, report_inputs_footer  # noqa: E402
from taxiout.callsign import callsign_features  # noqa: E402
from taxiout.challenge import load_movements, load_ranking, to_pandas  # noqa: E402
from taxiout.features import TARGET  # noqa: E402
from taxiout.metar import load_metar, regime_by_hour  # noqa: E402
from taxiout.models import SUBMISSION_KEY  # noqa: E402
from taxiout.occupancy import MOVEMENT_COLUMNS, occupancy_features  # noqa: E402
from taxiout.validation import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    JULY_AIRPORTS,
    cluster_bootstrap,
    cluster_sums,
    month_of,
)

DATA = REPO / "data" / "challenge"
RESIDUALS = REPO / "data" / "processed" / "residuals"


def _inputs() -> list:
    """Every path this run read, for the provenance footer.

    Named individually rather than as a directory: a directory has no
    digest, and "the data under `data/challenge/`" is exactly the
    declaration `scripts/check_provenance.py` check 8 exists to replace.
    """
    return challenge_inputs(DATA) + [
        RESIDUALS / f"{split}_{RUNG}.parquet"
        for split, _i, _l in LEGS
        if (RESIDUALS / f"{split}_{RUNG}.parquet").exists()
    ]


METAR = REPO / "data" / "raw" / "metar"
OUT = REPO / "reports" / "residual.md"

# The rung the submission is fitted from: this report is about the error that
# model actually leaves, not the error of a rung nothing ships.
RUNG = SUBMISSION_KEY

# The legs, and the reason there are three. The July split is over all ten
# airports; the third leg reads the three airports the ranking file of
# 2 September held in July 2026 (``JULY_AIRPORTS``), which was the whole of
# the July test month until the organizers re-issued the file on 4 September
# with all ten. The leg is kept as it was fixed. On this rung one Fiumicino row
# carries a third of the pooled July figure, which the *largest rows* section
# shows, and since 28 September that section says whether the scored month
# holds rows of that airport rather than asserting that it does not.
LEGS: tuple[tuple[str, tuple[str, ...] | None, str], ...] = (
    ("december", None, "December, all ten airports"),
    ("july", None, "July, all ten airports"),
    ("july", JULY_AIRPORTS, "July, the three"),
)

# The tail is cut at these, in seconds of absolute residual.
TAIL_CUTS_S: tuple[float, ...] = (600.0, 1200.0)

# The band where the stand-occupancy bound is the mechanism, in minutes, and
# the flag that says the stand was taken by a *different* aircraft.
REOCCUPIED_BAND_MIN: tuple[float, float] = (20.0, 45.0)

# Plan-lag bands, in minutes: take-off minus the flight plan's first off-block
# estimate. The last is open and nulls are their own row.
LAG_EDGES_MIN: tuple[float, ...] = (0, 15, 30, 45, 60, 120, np.inf)

# How many of the largest rows to print per leg.
LARGEST_ROWS = 5

# The share of |residual| that counts as "the worst one percent".
WORST_QUANTILE = 0.99

# The two row sets outside the lag bands. *The error by plan lag* prints
# ``NO_ESTIMATE`` as a row of its own and stops there; the cut below adds
# ``BEFORE_ESTIMATE``, which that table does not print, so that its shares add
# up over the whole leg.
NO_ESTIMATE = "no estimate"
BEFORE_ESTIMATE = "before the estimate"

# The airport the July section is about. Its July figure is the one that stands
# out on the ten-airport leg once the one Fiumicino day-shift is set aside, and
# it is in the July test month under both issues of the ranking file, so what
# July does to it is a question about the test set, not only the training year.
HEATHROW = "EGLL"

# How many take-off hours that section names as the ones the error sits in, and
# how many callsign prefixes it names before pooling the rest. Both are read off
# the leg's own squared error rather than fixed in advance.
PEAK_HOURS = 6
NAMED_PREFIXES = 10
POOLED_PREFIX = "the rest"

# What the weather section says of a flag that never fires on a leg, rather
# than leaving a gap in a sentence that expects a figure.
NOT_FIRED = "not present on this leg"

# Below this many distinct take-off days on the thinner side, a day-clustered
# resample has too few clusters to say anything: with one day it draws that day
# every time and reports no variation at all, which prints as a narrow interval
# and means the opposite. Those cells print their day count instead.
MIN_INTERVAL_DAYS = 5


def _band_label(lo: float, hi: float) -> str:
    return f"over {lo:.0f}" if not np.isfinite(hi) else f"{lo:.0f}-{hi:.0f}"


LAG_LABELS = [
    _band_label(LAG_EDGES_MIN[i], LAG_EDGES_MIN[i + 1])
    for i in range(len(LAG_EDGES_MIN) - 1)
]


def load_side() -> pd.DataFrame:
    """The training year's departures with what this report reads of them.

    The occupancy block is built from the whole movement frame, as the model
    builds it, because the stand's previous arrival is looked for over the
    side's own covered block.
    """
    columns = sorted(
        set(MOVEMENT_COLUMNS)
        | {
            TARGET,
            "MVT_ID_mvt",
            "WK_TBL_CAT_flt",
            "EOBT_1_flt",
            "RUNWAY_mvt",
            "FLIGHT_mvt",
        }
    )
    frame = to_pandas(load_movements(DATA, columns)).reset_index(drop=True)
    departures = frame[frame["PHASE_mvt"].astype(str) == "DEP"]
    occ = occupancy_features(departures, frame)
    out = pd.DataFrame(index=departures.index)
    out["mvt_id"] = departures["MVT_ID_mvt"].to_numpy()
    out["icao"] = departures["ADEP_mvt"].astype(str).to_numpy()
    # Carried only so the identifier guard has something to look for; nothing
    # in the report reads it.
    out["stand"] = departures["STAND_mvt"].astype(str).to_numpy()
    out["month"] = month_of(departures["MVT_TIME_UTC_mvt"]).to_numpy()
    out[TARGET] = pd.to_numeric(departures[TARGET], errors="coerce").to_numpy()
    out["wake"] = departures["WK_TBL_CAT_flt"].astype(object).to_numpy()
    # The ICAO airline designator, as taxiout.callsign reads it for the rung of
    # that name. An aggregate over an operator's departures, not an identifier.
    out["prefix"] = callsign_features(departures)["cs_prefix"].to_numpy()
    out["no_record"] = departures["AIRCRAFT_OPERATOR_flt"].isna().to_numpy()
    lag = pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True) - pd.to_datetime(
        departures["EOBT_1_flt"], utc=True
    )
    out["lag_min"] = (lag.dt.total_seconds() / 60.0).to_numpy()
    gap_min = pd.to_numeric(occ["occ_gap1_s"], errors="coerce").to_numpy() / 60.0
    lo, hi = REOCCUPIED_BAND_MIN
    out["reoccupied"] = (
        (gap_min >= lo) & (gap_min < hi) & (occ["occ_prev_same_type"].to_numpy() == 0.0)
    )
    return out.reset_index(drop=True)


def load_leg(split: str, icaos, side: pd.DataFrame) -> pd.DataFrame | None:
    """One leg's ex-flagged rows, with the residual and everything read of it."""
    path = RESIDUALS / f"{split}_{RUNG}.parquet"
    if not path.exists():
        return None
    got = pd.read_parquet(path)
    if "mvt_id" not in got.columns:
        return None
    got = got[~got["flagged"].astype(bool)]
    # The runway and the take-off come from the residual file rather than from
    # the movement side: they are what the ladder scored the row under.
    frame = got[["mvt_id", "icao", "runway", "takeoff", "residual"]].merge(
        side.drop(columns=["icao"]), on="mvt_id", how="inner"
    )
    if icaos is not None:
        frame = frame[frame["icao"].isin(icaos)]
    # residual is target minus prediction, so a positive one is the model
    # under-predicting and the prediction is recoverable exactly.
    frame["pred"] = frame[TARGET] - frame["residual"]
    frame["sq"] = frame["residual"] ** 2
    return frame.reset_index(drop=True)


def _rmse(r: pd.Series) -> float:
    return float(np.sqrt(float((r**2).mean()))) if len(r) else float("nan")


def section_shape(legs: dict) -> list[str]:
    """How few rows carry how much of the squared error, and how one-sided."""
    rows = []
    for _split, _icaos, label in LEGS:
        frame = legs.get(label)
        if frame is None or not len(frame):
            continue
        sse = float(frame["sq"].sum())
        for cut in TAIL_CUTS_S:
            tail = frame[frame["residual"].abs() > cut]
            if not len(tail):
                continue
            under = tail[tail["residual"] > 0]
            rows.append(
                [
                    label,
                    f"over {cut:,.0f}",
                    f"{len(tail):,}",
                    pct(len(tail) / len(frame), 2),
                    pct(float(tail["sq"].sum()) / sse, 1),
                    pct(len(under) / len(tail), 0),
                    num(float(under[TARGET].median()) if len(under) else np.nan, 0),
                    num(float(under["pred"].median()) if len(under) else np.nan, 0),
                    num(float(under[TARGET].quantile(0.9)) if len(under) else np.nan, 0),
                ]
            )
    return [
        "## The shape of the tail",
        "",
        "Rows whose residual is large, as a share of the rows and of the squared",
        "error. `under` is the share of those rows the model **under**-predicted:",
        "a conditional mean sitting between a gate delay and a hold under-predicts",
        "the held rows, so a one-sided tail is what the decomposition predicts and",
        "a symmetric one would be evidence against it. The last three columns are",
        "the under-predicted rows alone.",
        "",
        *table(
            [
                "leg",
                "|residual| (s)",
                "rows",
                "of rows",
                "of SSE",
                "under",
                "median actual (s)",
                "median predicted",
                "p90 actual",
            ],
            rows,
            "--rrrrrrr",
        ),
    ]


def section_enrichment(legs: dict) -> list[str]:
    """The three classes an individual hold lives in."""
    rows = []
    for _split, _icaos, label in LEGS:
        frame = legs.get(label)
        if frame is None or not len(frame):
            continue
        sse = float(frame["sq"].sum())
        worst = frame["residual"].abs() >= frame["residual"].abs().quantile(
            WORST_QUANTILE
        )
        n_worst = int(worst.sum())
        for name, mask in (
            ("heavies", frame["wake"].isin(["H", "J"]).to_numpy()),
            ("no Network Manager record", frame["no_record"].to_numpy()),
            ("stand re-occupied, other type", frame["reoccupied"].to_numpy()),
        ):
            part = frame[mask]
            if not len(part):
                continue
            rows.append(
                [
                    label,
                    name,
                    f"{len(part):,}",
                    pct(len(part) / len(frame), 2),
                    pct(int((worst & mask).sum()) / n_worst, 1) if n_worst else "n/a",
                    pct(float(part["sq"].sum()) / sse, 1),
                    num(_rmse(part["residual"]), 0),
                    f"{float(part['residual'].mean()):+,.0f}",
                ]
            )
    return [
        "## What the tail is enriched in",
        "",
        "Three classes, each one where an individual hold rather than a shared",
        "queue would put the error. `of the worst 1%` is the class's share of the",
        f"rows above the {WORST_QUANTILE:.0%} quantile of |residual| on that leg; a",
        "class carrying more of the worst one percent than of the rows is enriched.",
        "`bias` is the mean residual, so positive is the model under-predicting.",
        "",
        "The re-occupied class is the rows the stand-occupancy bound speaks about --",
        "the stand taken by an aircraft of a different type",
        f"{REOCCUPIED_BAND_MIN[0]:.0f} to {REOCCUPIED_BAND_MIN[1]:.0f} minutes before",
        "take-off -- measured here **after** the block that reads them, so what is",
        "left is what the block did not recover.",
        "",
        *table(
            [
                "leg",
                "class",
                "rows",
                "of rows",
                "of the worst 1%",
                "of SSE",
                "RMSE (s)",
                "bias",
            ],
            rows,
            "--rrrrrr",
        ),
    ]


def section_lag(legs: dict) -> list[str]:
    """The error against the plan lag, which it grows with."""
    rows = []
    for _split, _icaos, label in LEGS:
        frame = legs.get(label)
        if frame is None or not len(frame):
            continue
        sse = float(frame["sq"].sum())
        band = pd.cut(
            frame["lag_min"],
            bins=list(LAG_EDGES_MIN),
            labels=LAG_LABELS,
            right=False,
            ordered=True,
        )
        for name in LAG_LABELS + [NO_ESTIMATE]:
            take = (
                frame["lag_min"].isna().to_numpy()
                if name == NO_ESTIMATE
                else (band == name).to_numpy()
            )
            if not take.any():
                continue
            part = frame[take]
            rows.append(
                [
                    label,
                    name,
                    f"{len(part):,}",
                    pct(len(part) / len(frame), 1),
                    num(_rmse(part["residual"]), 0),
                    f"{float(part['residual'].mean()):+,.0f}",
                    pct(float(part["sq"].sum()) / sse, 1),
                ]
            )
    return [
        "## The error by plan lag",
        "",
        "Take-off minus the flight plan's first off-block estimate (`EOBT_1_flt`),",
        "which is the taxi plus whatever the aircraft waited at the stand against",
        "its own plan. The error grows with it, as a conditional mean between two",
        "modes must: a long lag is sometimes a gate delay the aircraft absorbs",
        "before pushing and sometimes a hold it absorbs after, and no field the",
        "ranking set exposes separates them.",
        "",
        *table(
            ["leg", "lag (min)", "rows", "of rows", "RMSE (s)", "bias", "of SSE"],
            rows,
            "--rrrrr",
        ),
    ]


def ranking_july_departures(icao: str) -> int:
    """How many July 2026 departures the ranking set holds at ``icao``.

    Read from the ranking file, which this report declares among its inputs.
    Until 28 September 2026 the sentence this feeds was fixed text, "July 2026
    contains no LIRF rows": true of the ranking file of 2 September, which
    held July at three airports, and false of the file the organizers
    re-issued on 4 September, which holds all ten. The count is computed so
    the sentence follows the file.
    """
    frame = to_pandas(load_ranking(DATA, ["ADEP_mvt", "MVT_TIME_UTC_mvt"], phase="DEP"))
    at = frame[frame["ADEP_mvt"].astype(str) == icao]
    return int((month_of(at["MVT_TIME_UTC_mvt"]) == "2026-07").sum())


def _july_note(first, sse: float) -> list[str]:
    """The paragraph under the largest-rows table, on the one July row.

    What it says about the scored month is computed from the ranking file:
    whether July 2026 holds departures of the row's airport decides which
    July figure the paper quotes, and the sentence says so either way.
    """
    icao = str(first["icao"])
    held = ranking_july_departures(icao)
    if held:
        scored = (
            f" **July 2026 holds {held:,} {icao} departures** in the ranking set the"
            " organizers re-issued on 4 September 2026, so a row of this kind can"
            " occur in the scored month and would be scored at its recorded value"
            " exactly as this one is. That is why the paper quotes the ten-airport"
            " July figure for the submitted model and keeps the three-airport one,"
            " read on the airports the file of 2 September held in July, beside it."
        )
    else:
        scored = (
            f" **July 2026 contains no {icao} rows.** That is why the paper quotes"
            " the three-airport July figure for the submitted model and why the"
            " pooled one stays in [`baseline.md`](baseline.md) with this reason"
            " beside it."
        )
    return [
        "",
        f"**One row is {pct(float(first['sq']) / sse, 0)} of the July split's"
        f" squared error.** A departure at {icao} recording a taxi-out"
        f" of {float(first[TARGET]):,.0f} seconds -- a day-shift in a cell the"
        " record-quality term does not cover, so no term touches it. The"
        " four-hour exclusion does not reach it either: that rule is"
        " **fit-side only**, keeping day-shifts out of the fitting sample and"
        " never out of the scoring set, and this is a validation row. So it is"
        " scored at its recorded value." + scored + " The rule that admitted the"
        " Gate 2c blocks was read on the pooled figure as it was fixed, and is"
        " not re-read here.",
    ]


def section_largest(legs: dict) -> list[str]:
    """The handful of rows that carry a tenth of a split each."""
    rows = []
    note = []
    for _split, _icaos, label in LEGS:
        frame = legs.get(label)
        if frame is None or not len(frame):
            continue
        sse = float(frame["sq"].sum())
        top = frame.nlargest(LARGEST_ROWS, "sq")
        for _, r in top.iterrows():
            rows.append(
                [
                    label,
                    str(r["icao"]),
                    f"{float(r[TARGET]):,.0f}",
                    f"{float(r['pred']):,.0f}",
                    f"{float(r['residual']):+,.0f}",
                    pct(float(r["sq"]) / sse, 1),
                ]
            )
        first = top.iloc[0] if len(top) else None
        if first is not None and label == "July, all ten airports":
            note = _july_note(first, sse)
    return [
        "## The largest rows",
        "",
        f"The {LARGEST_ROWS} rows of each leg with the largest squared error, as the",
        "airport, the recorded taxi-out and what the model predicted. No stand and no",
        "movement identifier is printed, here or anywhere in this report.",
        "",
        *table(
            ["leg", "ICAO", "recorded (s)", "predicted", "residual", "of SSE"],
            rows,
            "--rrrr",
        ),
        *note,
    ]


def section_per_airport(legs: dict) -> list[str]:
    """The same tail shares per airport, so the paper's prose has a source."""
    rows = []
    for _split, icaos, label in LEGS:
        if icaos is not None:
            continue
        frame = legs.get(label)
        if frame is None or not len(frame):
            continue
        sse = float(frame["sq"].sum())
        for icao in sorted(frame["icao"].unique()):
            part = frame[frame["icao"] == icao]
            tail = part[part["residual"].abs() > TAIL_CUTS_S[0]]
            rows.append(
                [
                    label.split(",")[0],
                    icao,
                    f"{len(part):,}",
                    num(_rmse(part["residual"]), 1),
                    f"{float(part['residual'].mean()):+,.0f}",
                    pct(float(part["sq"].sum()) / sse, 1),
                    pct(len(tail) / len(part), 2),
                    pct(float(tail["sq"].sum()) / float(part["sq"].sum()), 0)
                    if float(part["sq"].sum())
                    else "n/a",
                ]
            )
    return [
        "## Per airport",
        "",
        "The same quantities per airport, over the ten-airport legs. `of the split's",
        "SSE` says which airports the pooled figure is actually about; `tail rows`",
        f"and `tail SSE` are the rows over {TAIL_CUTS_S[0]:,.0f} s and their share of",
        "that airport's own squared error.",
        "",
        *table(
            [
                "split",
                "ICAO",
                "rows",
                "RMSE (s)",
                "bias",
                "of the split's SSE",
                "tail rows",
                "tail SSE",
            ],
            rows,
            "--rrrrrr",
        ),
    ]


# --------------------------------------------------------------------------
# Heathrow's July, and December by weather flag
#
# Both sections describe. Neither changes a feature, a rung or a rule, and
# neither fits anything: they cut the same stored residuals the sections above
# read. Where the prose states a difference it carries the day-clustered
# bootstrap interval, which is what the standing rule asks of a question put
# inside one split.


def _day(frame: pd.DataFrame) -> np.ndarray:
    """The take-off's UTC calendar day: the cluster a same-split interval uses."""
    return pd.to_datetime(frame["takeoff"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()


def _difference(a: pd.DataFrame, b: pd.DataFrame):
    """RMSE over ``a`` minus RMSE over ``b``, day-clustered; and the day count.

    The day is the cluster because congestion is autocorrelated over hours, so
    two departures an hour apart share queue state and are not independent
    draws, while two a week apart very nearly are -- the reasoning
    ``taxiout.validation`` used to rule out a random K-fold. The two row sets
    are resampled separately, as the held-out-pair gap resamples its two sides.
    Where a day contributes rows to both sides that is conservative: the
    day-level variation the two share is counted twice rather than netted out.
    No model is refitted, which is what makes a thousand draws free.

    Returns ``(result, days)``, ``result`` being ``None`` where the thinner
    side rests on fewer than ``MIN_INTERVAL_DAYS`` days.
    """
    if not len(a) or not len(b):
        return None, 0
    days = min(len(set(_day(a))), len(set(_day(b))))
    if days < MIN_INTERVAL_DAYS:
        return None, days
    fold = {
        "a": cluster_sums({"r": a["residual"].to_numpy(dtype=float)}, _day(a)),
        "b": cluster_sums({"r": b["residual"].to_numpy(dtype=float)}, _day(b)),
    }
    got = cluster_bootstrap(
        [fold], lambda r: r["a"]["r"] - r["b"]["r"], draws=BOOTSTRAP_DRAWS
    )
    return got, days


def _shown(a: pd.DataFrame, b: pd.DataFrame) -> tuple[str, str]:
    """That difference as (the table's cell, the phrase the prose uses)."""
    got, days = _difference(a, b)
    if got is None:
        note = f"{days} day{'' if days == 1 else 's'}, no interval" if days else "n/a"
        return note, note
    return (
        f"[{got.low:+,.1f}, {got.high:+,.1f}]",
        f"{got.point:+,.1f} s [{got.low:+,.1f}, {got.high:+,.1f}]",
    )


def _quintuple(part: pd.DataFrame, rows: int, sse: float) -> list[str]:
    """One cell of a cut: rows, row share, RMSE, bias and share of the leg's SSE."""
    if not rows or not sse:
        return [f"{len(part):,}", "n/a", "n/a", "n/a", "n/a"]
    if not len(part):
        return ["0", pct(0.0, 1), "n/a", "n/a", pct(0.0, 1)]
    return [
        f"{len(part):,}",
        pct(len(part) / rows, 1),
        num(_rmse(part["residual"]), 0),
        f"{float(part['residual'].mean()):+,.0f}",
        pct(float(part["sq"].sum()) / sse, 1),
    ]


def _cut(july, december, label: str, level_of, levels) -> list[str]:
    """One cut of the two legs, December's five columns beside July's."""
    jn, dn = len(july), len(december)
    js, ds = float(july["sq"].sum()), float(december["sq"].sum())
    jl, dl = np.asarray(level_of(july)), np.asarray(level_of(december))
    rows = [
        [name]
        + _quintuple(july[jl == name], jn, js)
        + _quintuple(december[dl == name], dn, ds)
        for name in levels
    ]
    tail = ["of rows", "RMSE (s)", "bias", "of SSE"]
    header = [label, "July rows"] + tail + ["Dec rows"] + tail
    return table(header, rows, "-" + "r" * 10)


def _hour_level(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(frame["takeoff"], utc=True).dt.strftime("%H").to_numpy()


def _lag_level(frame: pd.DataFrame) -> np.ndarray:
    """The bands the report already uses, with the two rows they leave out named.

    *The error by plan lag* prints the bands and the missing estimates and lets
    a take-off earlier than its own estimate fall out of the table. Here the
    shares have to add up, so that row set gets a name of its own.
    """
    lag = frame["lag_min"].to_numpy(dtype=float)
    band = (
        pd.cut(
            frame["lag_min"],
            bins=list(LAG_EDGES_MIN),
            labels=LAG_LABELS,
            right=False,
            ordered=True,
        )
        .astype(object)
        .to_numpy()
    )
    return np.where(
        np.isnan(lag), NO_ESTIMATE, np.where(pd.isna(band), BEFORE_ESTIMATE, band)
    )


def _record_level(frame: pd.DataFrame) -> np.ndarray:
    return np.where(frame["no_record"].to_numpy(), "no record", "a record")


def _named_prefixes(frame: pd.DataFrame, k: int) -> list[str]:
    """The ``k`` callsign prefixes carrying the largest share of a leg's SSE."""
    by = (
        frame.groupby(frame["prefix"].astype(str))["sq"]
        .sum()
        .sort_values(ascending=False)
    )
    return [str(p) for p in by.index[:k]]


def _peak_hours(frame: pd.DataFrame, k: int) -> list[str]:
    """The ``k`` take-off hours carrying the largest share of a leg's SSE."""
    by = (
        pd.Series(frame["sq"].to_numpy(), index=_hour_level(frame))
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )
    return sorted(str(h) for h in by.index[:k])


def _hours_label(hours: list[str]) -> str:
    """``14:00 to 19:59 UTC`` where the hours run together, else the list."""
    values = sorted(int(h) for h in hours)
    if values and values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]:02d}:00 to {values[-1]:02d}:59 UTC"
    return ", ".join(f"{v:02d}:00" for v in values) + " UTC"


def section_heathrow(july, december) -> list[str]:
    """What July changes at Heathrow: five cuts of the two legs, side by side."""
    head = [
        f"## {HEATHROW}'s July",
        "",
        f"[`baseline.md`](baseline.md) reads {HEATHROW} far above its December figure on",
        "the July split, and the per-airport table above says a thin share of its July",
        "rows carries most of its squared error. This section asks where in the day, on",
        "which runway, at which plan lag, under which callsign prefix and with or",
        "without a Network Manager record those rows sit, with December at the same",
        "airport beside each cut, so a reader sees what July changes rather than what",
        "this airport always is. It fits nothing and it decides nothing. Every",
        "difference the prose states carries the day-clustered bootstrap interval",
        "`taxiout.validation.cluster_bootstrap` gives -- the day as the cluster, which",
        "is what the standing rule asks of a question put inside one split.",
        "",
    ]
    if july is None or december is None or not len(july) or not len(december):
        return head + [
            f"No stored residuals for {HEATHROW} on one of the two legs, so the cuts are",
            "not computed. `make baseline` writes them.",
        ]

    jsse = float(july["sq"].sum())
    tail = july[july["residual"].abs() > TAIL_CUTS_S[0]]
    peak = _peak_hours(july, PEAK_HOURS)
    in_peak_j = np.isin(_hour_level(july), peak)
    in_peak_d = np.isin(_hour_level(december), peak)
    no_record_j = july["no_record"].to_numpy()
    runway_j = july["runway"].astype(str).to_numpy()
    worst_runway = str(july.groupby(july["runway"].astype(str))["sq"].sum().idxmax())
    on_worst = runway_j == worst_runway

    # The differences this section states, and no others. Fixed in one list so
    # that it can be read as a list rather than picked out of five tables.
    contrasts = [
        (f"July at {HEATHROW}", july, "December at the same airport", december),
        (
            f"July, take-off hours {_hours_label(peak)}",
            july[in_peak_j],
            "July, its other hours",
            july[~in_peak_j],
        ),
        (
            "December, the same hours",
            december[in_peak_d],
            "December, its other hours",
            december[~in_peak_d],
        ),
        (
            "July, no Network Manager record",
            july[no_record_j],
            "July, a record",
            july[~no_record_j],
        ),
        (
            "December, no Network Manager record",
            december[december["no_record"].to_numpy()],
            "December, a record",
            december[~december["no_record"].to_numpy()],
        ),
        (
            f"July, runway {worst_runway}",
            july[on_worst],
            "July, its other runways",
            july[~on_worst],
        ),
    ]
    crows, spoken = [], []
    for name, a, against, b in contrasts:
        cell, phrase = _shown(a, b)
        spoken.append(phrase)
        crows.append(
            [
                name,
                f"{len(a):,}",
                num(_rmse(a["residual"]), 1),
                against,
                f"{len(b):,}",
                num(_rmse(b["residual"]), 1),
                f"{_rmse(a['residual']) - _rmse(b['residual']):+,.1f}",
                cell,
            ]
        )

    prefixes = _named_prefixes(july, NAMED_PREFIXES)

    def prefix_level(frame: pd.DataFrame) -> np.ndarray:
        p = frame["prefix"].astype(str).to_numpy()
        return np.where(np.isin(p, prefixes), p, POOLED_PREFIX)

    hours = sorted(set(_hour_level(july)) | set(_hour_level(december)))
    runways = sorted(
        set(july["runway"].astype(str)) | set(december["runway"].astype(str))
    )
    return head + [
        f"**The two legs.** {len(july):,} ex-flagged July rows at {HEATHROW} against",
        f"{len(december):,} in December, {num(_rmse(july['residual']), 1)} s RMSE against",
        f"{num(_rmse(december['residual']), 1)} s; {pct(len(tail) / len(july), 2)} of the",
        f"July rows are over {TAIL_CUTS_S[0]:,.0f} s of absolute residual and carry",
        f"{pct(float(tail['sq'].sum()) / jsse, 0)} of the airport's July squared error.",
        "Every share below is of its own leg's rows or of its own leg's squared error,",
        "so the two halves of each table are read down and not across.",
        "",
        "### By take-off hour (UTC)",
        "",
        *_cut(july, december, "hour", _hour_level, hours),
        "",
        "### By runway",
        "",
        *_cut(july, december, "runway", lambda f: f["runway"].astype(str), runways),
        "",
        "### By plan-lag band",
        "",
        "The bands *The error by plan lag* uses. One row set that table does not",
        "print is named here so the shares add up: a take-off earlier than its own",
        f"estimate (`{BEFORE_ESTIMATE}`). A take-off with no estimate is already a",
        f"row of that table (`{NO_ESTIMATE}`), and it is a row here too.",
        "",
        *_cut(
            july,
            december,
            "lag (min)",
            _lag_level,
            list(LAG_LABELS) + [NO_ESTIMATE, BEFORE_ESTIMATE],
        ),
        "",
        "### By callsign prefix",
        "",
        f"The {NAMED_PREFIXES} prefixes with the largest share of July's squared error,",
        f"the rest pooled as `{POOLED_PREFIX}`. The prefix is the ICAO airline designator",
        "read off `FLIGHT_mvt`, as `taxiout.callsign` reads it: an aggregate over an",
        "operator's departures and not an identifier of any of them. The same prefixes",
        "are named on the December side, so the two columns are comparable.",
        "",
        *_cut(july, december, "prefix", prefix_level, prefixes + [POOLED_PREFIX]),
        "",
        "### By whether the row has a Network Manager record",
        "",
        *_cut(july, december, "record", _record_level, ["a record", "no record"]),
        "",
        "### The differences, with their intervals",
        "",
        "Each row is one set of that leg's rows against another, as RMSE, with the",
        "day-clustered percentile interval on the difference. Nothing here is a",
        "decision: the intervals are printed because the prose states the differences,",
        "and a stated difference carries one. A contrast whose thinner side rests on",
        f"fewer than {MIN_INTERVAL_DAYS} distinct take-off days prints that count and no",
        "interval, because resampling one day draws the same day every time and would",
        "report a confidence it has not got.",
        "",
        *table(
            [
                "this",
                "rows",
                "RMSE (s)",
                "against",
                "rows",
                "RMSE (s)",
                "difference (s)",
                "95% interval",
            ],
            crows,
            "-rr-rrrr",
        ),
        "",
        f"**What July changes.** {HEATHROW}'s July error is {spoken[0]} against December",
        "at the same airport. It is not spread over the day: the",
        f"{PEAK_HOURS} hours carrying the most squared error are {_hours_label(peak)}, and",
        f"they hold {pct(float(july.loc[in_peak_j, 'sq'].sum()) / jsse, 0)} of July's",
        f"squared error on {pct(float(in_peak_j.sum()) / len(july), 0)} of its rows,",
        f"running {spoken[1]} above July's other hours. The same hours in December run",
        f"{spoken[2]} against December's other hours, so the afternoon is a July effect",
        "at this airport and not a property of the airport. The rows with no Network",
        f"Manager record are {int(no_record_j.sum()):,} of July's {len(july):,} and carry",
        f"{pct(float(july.loc[no_record_j, 'sq'].sum()) / jsse, 1)} of its squared error,",
        f"running {spoken[3]} above the rows that have one; in December the same class",
        f"runs {spoken[4]} above. Runway {worst_runway} carries",
        f"{pct(float(july.loc[on_worst, 'sq'].sum()) / jsse, 1)} of July's squared error",
        f"on {pct(float(on_worst.sum()) / len(july), 1)} of its rows and runs {spoken[5]}",
        "above the other runways.",
        "",
        "None of this is a feature. The hour of take-off, the runway and the plan lag",
        "are already in the frame the submitted rung reads, and the callsign prefix is",
        "the `+ callsign prefix` rung's own level; what the cuts say is where the error",
        "the fitted model still makes sits, not that anything should be added. The",
        "oracle diagnostic in [`congestion.md`](congestion.md) bounds what knowing the",
        "same-runway sequence exactly could still buy at single figures of seconds, and",
        "that is the scale any queue reading of the afternoon would have to live inside.",
    ]


# --------------------------------------------------------------------------
# December by weather flag


def section_weather(december, hourly) -> list[str]:
    """The December leg cut by the three regime flags, per airport and pooled."""
    head = [
        "## December by weather flag",
        "",
        "[`congestion.md`](congestion.md) reports January 2026's freezing-hour shares",
        "above 2025's at Frankfurt, Zurich and Istanbul, which makes the winter regime",
        "something the ranking month has more of than the training year has. This",
        "section asks what the submitted rung's error already does under each regime,",
        "on the December validation month. The flags are the three",
        "`taxiout.metar.regime_by_hour` produces -- a visibility band, precipitation at",
        "the aerodrome, and freezing conditions -- joined to a residual on the station",
        "and the hour its take-off falls in. It fits nothing and changes nothing: the",
        "weather block is already in the frame the rung reads, so what is below is what",
        "is left after it.",
        "",
    ]
    if december is None or not len(december):
        return head + ["No stored December residuals; `make baseline` writes them."]
    if hourly is None or not len(hourly):
        return head + [
            "No METAR cache under `data/raw/metar/`; `python scripts/fetch_metar.py`",
            "fetches it, and this section is empty until it does.",
        ]

    frame = december.copy()
    frame["hour"] = pd.to_datetime(frame["takeoff"], utc=True).dt.floor("h")
    got = frame.merge(
        hourly.rename(columns={"station": "icao"})[
            ["icao", "hour", "vis_band", "precip", "freezing"]
        ],
        on=["icao", "hour"],
        how="left",
    )
    n, sse = len(got), float(got["sq"].sum())
    read = got["vis_band"].notna().to_numpy()
    levels = [
        ("visibility below 1.5 km", (got["vis_band"] == 2).to_numpy()),
        ("visibility 1.5 to 8 km", (got["vis_band"] == 1).to_numpy()),
        ("visibility 8 km or more", (got["vis_band"] == 0).to_numpy()),
        ("precipitation", (got["precip"] == 1).to_numpy()),
        ("no precipitation", (got["precip"] == 0).to_numpy()),
        ("freezing", (got["freezing"] == 1).to_numpy()),
        ("not freezing", (got["freezing"] == 0).to_numpy()),
        ("no reading for the hour", ~read),
    ]
    lrows = [[name] + _quintuple(got[mask], n, sse) for name, mask in levels]

    # The binary form of each flag, which is what "flagged against unflagged"
    # means here: the visibility flag is the hour that is not in the clear band.
    flags = (
        ("visibility below 8 km", lambda g: (g["vis_band"] >= 1).to_numpy()),
        ("precipitation", lambda g: (g["precip"] == 1).to_numpy()),
        ("freezing", lambda g: (g["freezing"] == 1).to_numpy()),
    )
    prows: list[list[str]] = []
    pooled: dict[str, str] = {}
    better: list[str] = []
    absent: list[str] = []
    groups = [("pooled", got)] + [
        (str(k), g) for k, g in got.groupby(got["icao"].astype(str), sort=True)
    ]
    for label, part in groups:
        has = part["vis_band"].notna().to_numpy()
        base = part[has]
        psse = float(base["sq"].sum())
        for name, of in flags:
            on = of(part) & has
            a, b = part[on], part[has & ~on]
            if not len(a) or not len(b):
                continue
            step = _rmse(a["residual"]) - _rmse(b["residual"])
            cell, phrase = _shown(a, b)
            if label == "pooled":
                pooled[name] = (
                    f"{len(a):,} rows at {num(_rmse(a['residual']), 0)} s against"
                    f" {num(_rmse(b['residual']), 0)} s, {phrase}"
                )
            elif name == "precipitation" and step < 0:
                better.append(label)
            prows.append(
                [
                    label,
                    name,
                    f"{len(a):,}",
                    f"{len(set(_day(a))):,}",
                    pct(len(a) / len(base), 1) if len(base) else "n/a",
                    num(_rmse(a["residual"]), 0),
                    f"{float(a['residual'].mean()):+,.0f}",
                    pct(float(a["sq"].sum()) / psse, 1) if psse else "n/a",
                    num(_rmse(b["residual"]), 0),
                    f"{float(b['residual'].mean()):+,.0f}",
                    f"{step:+,.0f}",
                    cell,
                ]
            )
        for name, of in flags:
            if label != "pooled" and not (of(part) & has).any():
                absent.append(f"{label} {name}")
    freezing_rows = [r for r in prows if r[0] != "pooled" and r[1] == "freezing"]
    worst = max(
        freezing_rows,
        key=lambda r: float(r[7].rstrip("%")) if r[7] != "n/a" else -1.0,
        default=None,
    )
    # The widest and the narrowest freezing feed, so the sentence about the
    # spread across airports quotes the spread rather than describing it.
    by_share = sorted(freezing_rows, key=lambda r: float(r[4].rstrip("%")))
    spread = (
        [
            "The same holds of how often a flag fires at all: freezing covers",
            f"{by_share[-1][4]} of {by_share[-1][0]}'s December rows and",
            f"{by_share[0][4]} of {by_share[0][0]}'s.",
        ]
        if by_share
        else ["No airport fires the freezing flag on this leg."]
    )
    return head + [
        "Every share is of the December leg's own ex-flagged rows and of its own",
        "squared error. An hour more than three hours from the last report inherits",
        "nothing and carries no flags at all, so those rows are their own line rather",
        "than being counted as clear.",
        "",
        "### Pooled, by regime",
        "",
        *table(
            ["regime", "rows", "of rows", "RMSE (s)", "bias", "of SSE"],
            lrows,
            "-rrrrr",
        ),
        "",
        "### Flagged hours against unflagged, per airport and pooled",
        "",
        "Each flag in its binary form -- the visibility flag is any hour below 8 km, the",
        "two bands together -- over the rows whose hour has a reading. The interval is",
        "the day-clustered one, the same rule the section above uses, and `days` is how",
        "many distinct take-off days the flagged rows fall on: a flag that fires on two",
        "days of the month is two draws and not two hundred, so those cells print the",
        "count and no interval.",
        "",
        *table(
            [
                "ICAO",
                "flag",
                "flagged rows",
                "days",
                "of rows",
                "RMSE (s)",
                "bias",
                "of SSE",
                "unflagged RMSE (s)",
                "bias",
                "difference (s)",
                "95% interval",
            ],
            prows,
            "--rrrrrrrrrr",
        ),
        "",
        (
            "Flags that never fire on this leg have no row: " + "; ".join(absent) + "."
            if absent
            else "Every airport fires every flag at least once on this leg."
        ),
        "",
        "**What the flags say.** Pooled over the ten airports, freezing hours are",
        f"{pooled.get('freezing', NOT_FIRED)}; hours below 8 km of visibility are",
        f"{pooled.get('visibility below 8 km', NOT_FIRED)}; hours with precipitation are",
        f"{pooled.get('precipitation', NOT_FIRED)}. The three are not one thing.",
        "Freezing separates the",
        "widest, and it is the flag the ranking month has more of than the training",
        "year. Precipitation separates the least: at",
        f"{len(better)} of the {len(groups) - 1} airports the rows in precipitation score",
        f"*better* than the rows without it ({', '.join(better) if better else 'none'}),",
        "so the flag does not separate the same way at every feed and the pooled figure",
        "is an average over feeds that do different things.",
        *spread,
        "",
        *(
            [
                f"The largest single cell is {worst[0]}'s freezing hours: {worst[2]} rows",
                f"on {worst[3]} days, carrying {worst[7]} of that airport's December",
                f"squared error, at {worst[5]} s against {worst[8]} s on its unflagged",
                f"hours, a difference of {worst[10]} s {worst[11]}.",
            ]
            if worst is not None
            else ["No airport carries both a freezing and an unflagged cell here."]
        ),
        "",
        "This describes the fitted model's residual; it proposes nothing. The weather",
        "block went into the frame at Gate 2b and the ladder measured what it was",
        "worth. What is left under a flag is what the block did not recover, and",
        "nothing here re-reads that decision.",
    ]


# --------------------------------------------------------------------------


def main() -> int:
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to run", file=sys.stderr)
        return 1

    print("loading the training side ...")
    side = load_side()
    print(f"  {len(side):,} departures")

    legs = {}
    for split, icaos, label in LEGS:
        frame = load_leg(split, icaos, side)
        if frame is None:
            print(
                f"WARNING: no residuals for {split} at rung {RUNG}; "
                "run scripts/run_baselines.py",
                file=sys.stderr,
            )
            continue
        legs[label] = frame
        print(f"  {label}: {len(frame):,} ex-flagged rows")
    if not legs:
        print("no residuals to read", file=sys.stderr)
        return 1

    # The two legs the Heathrow section reads. They are loaded here rather than
    # added to LEGS because LEGS is what the five tables above are over, and
    # this section is beside them and not inside them.
    heathrow = {
        split: load_leg(split, (HEATHROW,), side) for split in ("july", "december")
    }
    for split, frame in heathrow.items():
        if frame is not None:
            print(f"  {HEATHROW} {split}: {len(frame):,} ex-flagged rows")

    hourly = None
    if METAR.exists() and any(METAR.glob("*.csv")):
        hourly = regime_by_hour(load_metar(METAR))
        print(f"  weather: {len(hourly):,} station-hours")
    else:
        print(f"WARNING: no METAR cache under {METAR}", file=sys.stderr)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = (
        [
            "# What the remaining error is",
            "",
            "[`baseline.md`](baseline.md) says what each block of the model is worth.",
            "This says what is left after all of them: the error the submitted rung",
            f"(`{RUNG}`) still makes, over the rows the record-quality term leaves to",
            "the physical model. It fits no model -- every figure is computed from",
            "that rung's stored validation residuals, joined back to the validation",
            "month on the movement identifier. Generated by",
            "`scripts/report_residual.py` from the data under `data/challenge/` on",
            f"{generated}; every figure below is computed there, and the generator",
            "refuses to write a literal stand identifier.",
            "",
            "**The claim.** What is left is a thin tail of *individual* holds, not a",
            "queue everyone stands in. The oracle diagnostic in",
            "[`congestion.md`](congestion.md) already bounds what knowing every",
            "same-runway neighbour's outcome exactly would buy, and it is single",
            "figures of seconds; this report says what the rows that remain look",
            "like, and that they are concentrated, one-sided and enriched exactly",
            "where an individual hold would put them.",
            "",
        ]
        + section_shape(legs)
        + [""]
        + section_enrichment(legs)
        + [""]
        + section_lag(legs)
        + [""]
        + section_largest(legs)
        + [""]
        + section_per_airport(legs)
        + [""]
        + section_heathrow(heathrow.get("july"), heathrow.get("december"))
        + [""]
        + section_weather(legs.get("December, all ten airports"), hourly)
        + [
            "",
            "## What would move it",
            "",
            "Nothing the challenge exposes. The unused exposed fields were measured",
            "before they were built (`docs/review_headroom.md` §2) and the four that",
            "carried anything are in the model. What would separate a gate delay from",
            "a hold is information about the hold itself -- a regulation, a de-icing",
            "queue, a runway inspection -- which the challenge data does not carry,",
            "and the open dataset that might, EUROCONTROL's own daily ATFM delay by",
            "airport and cause, is excluded on its licence.",
            "",
            *report_inputs_footer(REPO, _inputs()),
            "",
            "Distance matrices derived from OpenStreetMap: data (c) "
            "OpenStreetMap contributors, ODbL 1.0. Movement and flight data",
            "supplied by EUROCONTROL PRC and the OpenSky Network under the terms",
            "of the PRC Data Challenge 2026.",
            "",
        ]
    )
    text = "\n".join(body)
    check_no_identifiers(text, set(side["stand"].dropna()), load_matrix())
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
