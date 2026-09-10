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
from taxiout.challenge import load_movements, to_pandas  # noqa: E402
from taxiout.features import TARGET  # noqa: E402
from taxiout.models import SUBMISSION_KEY  # noqa: E402
from taxiout.occupancy import MOVEMENT_COLUMNS, occupancy_features  # noqa: E402
from taxiout.validation import JULY_AIRPORTS, month_of  # noqa: E402

DATA = REPO / "data" / "challenge"
RESIDUALS = REPO / "data" / "processed" / "residuals"
OUT = REPO / "reports" / "residual.md"

# The rung the submission is fitted from: this report is about the error that
# model actually leaves, not the error of a rung nothing ships.
RUNG = SUBMISSION_KEY

# The legs, and the reason there are three. The July split is over all ten
# airports and July 2026 contains three, so the pooled July figure answers a
# question the ranking month does not ask -- and on this rung one Fiumicino row
# carries a third of it, which the *largest rows* section shows.
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
        | {TARGET, "MVT_ID_mvt", "WK_TBL_CAT_flt", "EOBT_1_flt", "RUNWAY_mvt"}
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
    frame = got[["mvt_id", "icao", "residual"]].merge(
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
        for name in LAG_LABELS + ["no estimate"]:
            take = (
                frame["lag_min"].isna().to_numpy()
                if name == "no estimate"
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
            note = [
                "",
                f"**One row is {pct(float(first['sq']) / sse, 0)} of the July split's"
                f" squared error.** A departure at {first['icao']} recording a taxi-out"
                f" of {float(first[TARGET]):,.0f} seconds -- a day-shift in a cell the"
                " record-quality term does not cover, so no term touches it. The"
                " four-hour exclusion does not reach it either: that rule is"
                " **fit-side only**, keeping day-shifts out of the fitting sample and"
                " never out of the scoring set, and this is a validation row. So it is"
                " scored at its recorded value. **July 2026 contains no"
                f" {first['icao']} rows.** That is why the paper quotes the"
                " three-airport July figure for the submitted model and why the pooled"
                " one stays in [`baseline.md`](baseline.md) with this reason beside"
                " it. The rule that admitted the Gate 2c blocks was read on the pooled"
                " figure as it was fixed, and is not re-read here.",
            ]
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
