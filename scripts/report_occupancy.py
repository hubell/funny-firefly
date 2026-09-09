#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The stand as a shared resource, and write ``reports/occupancy.md``.

Gate 2c of ``docs/plan.md``. ``reports/baseline.md`` says what the
stand-occupancy block is worth on the ladder; this report says what the block
is, whether the quantity it reads means the same thing on both sides of the
data, and where the physical bound behind it actually holds. It fits no model.

The mechanism in one sentence: a stand holds one aircraft, so an arrival that
on-blocks at the subject's stand before the subject takes off means the subject
had already left the stand, and the gap between that on-block and the take-off
is a lower bound on the subject's taxi-out. The ranking set gives every
arrival's stand and on-block time in full, so the bound is built from exposed
rows -- which is the whole point, and also the thing to be careful about, since
it bounds a field the ranking set hides.

What is measured:

1. **The gap on both sides.** The same feature, the same airport, the same
   calendar month a year apart: the median and the 90th percentile of the gap
   to the stand's last arrival, the share of subjects under twenty minutes, and
   the share whose stand received no arrival at all. A feature built identically
   on both sides has the same shape on both sides; where it does not, the
   difference is a regime the model carries into the ranking month, and the
   paper has to say so rather than discover it on a leaderboard.
2. **Where the bound holds.** On the validation rows of the forward splits,
   over the rows the record-quality term leaves to the physical model: by
   five-minute band of the gap, how often the actual taxi-out really is at
   least the gap, how often the model already predicts at least the gap, and
   what the model's bias is there. Split by whether the previous arrival flew
   the subject's own aircraft type, because that is what separates the
   subject's own inbound after a fast turnaround from a different aircraft that
   took the stand once the subject had pushed.
3. **The hard floor, and what it costs.** The bound as a rule rather than as a
   feature: clip the prediction up to the gap and see what happens to the RMSE.
   It is a negative result and it is the reason the block enters as a feature.

The model is read from the stored validation residuals of the rung the
occupancy block was **added to**, not from whichever rung the submission has
since moved to. Read against the rung that already carries the block, table 2
would be asking how well a model respects a bound it has been given, which is a
different and much less interesting question.

Every number is computed here; nothing is hand-typed. The report carries
aggregates and runway designators only, and the generator refuses to write a
literal stand identifier from either source.

Needs the challenge data under ``data/challenge/``, the distance matrices from
``make reports``, and the residuals ``scripts/run_baselines.py`` writes under
``data/processed/residuals/``. A few minutes.

Usage
-----
    python scripts/report_occupancy.py
    make occupancy
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
from taxiout.challenge import load_movements, load_ranking, to_pandas  # noqa: E402
from taxiout.features import TARGET  # noqa: E402
from taxiout.occupancy import (  # noqa: E402
    MOVEMENT_COLUMNS,
    occupancy_features,
)
from taxiout.validation import month_of, rmse  # noqa: E402

DATA = REPO / "data" / "challenge"
RESIDUALS = REPO / "data" / "processed" / "residuals"
OUT = REPO / "reports" / "occupancy.md"

# The rung whose stored residuals stand in for "the model" in the bound table.
# Deliberately the rung the block was ADDED TO rather than
# ``taxiout.models.TOP_RUNG``: a model that already reads the gap will respect
# the gap, and a table showing that would be measuring its own feature.
BOUND_RUNG = "encoding-for-stand"

# Which forward split's residuals answer for which month, and which month of the
# ranking set each stands in for. December is the winter proxy for January --
# the choice the ladder's splits already make -- and July answers for July.
SPLITS: tuple[tuple[str, str, str], ...] = (
    ("december", "2025-12", "2026-01"),
    ("july", "2025-07", "2026-07"),
)

# The months compared across the year boundary, training side against ranking.
MONTH_PAIRS: tuple[tuple[str, str], ...] = (
    ("2025-01", "2026-01"),
    ("2025-07", "2026-07"),
)

# Five-minute bands of the gap, in minutes, with the two wide ones the review
# used at the top: past an hour the bound is not a bound and the bands only
# need to show that it is not.
BAND_EDGES_MIN: tuple[float, ...] = (
    0,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    120,
    np.inf,
)

# Below twenty minutes the bound holds essentially always; the share under it is
# the one-number summary of how much stand pressure an airport is under.
PRESSURE_MIN = 20

# The caps the hard floor is tried at, in minutes: apply the floor only where
# the gap is under the cap, on the reasoning that the bound is sound at short
# gaps and collapses at long ones. ``inf`` is the floor applied wherever there
# is a gap at all.
FLOOR_CAPS_MIN: tuple[float, ...] = (20, 30, 45, 60, np.inf)


def band_label(edge_lo: float, edge_hi: float) -> str:
    if not np.isfinite(edge_hi):
        return f"over {edge_lo:.0f}"
    return f"{edge_lo:.0f}-{edge_hi:.0f}"


def bands(gap_s: pd.Series) -> pd.Series:
    """Each gap's five-minute band label; NaN where there is no gap."""
    minutes = pd.to_numeric(gap_s, errors="coerce") / 60.0
    labels = [
        band_label(BAND_EDGES_MIN[i], BAND_EDGES_MIN[i + 1])
        for i in range(len(BAND_EDGES_MIN) - 1)
    ]
    return pd.cut(
        minutes, bins=list(BAND_EDGES_MIN), labels=labels, right=False, ordered=True
    )


# --------------------------------------------------------------------------
# data


def side(frame: pd.DataFrame) -> pd.DataFrame:
    """The occupancy block for one side's departures, from that side's movements.

    The movement frame is the whole side, exactly as the model builds it: the
    stand's previous arrival is looked for over the side's own covered block, so
    a subset would move the coverage flags and the gaps near a month boundary.
    """
    departures = frame[frame["PHASE_mvt"].astype(str) == "DEP"]
    occ = occupancy_features(departures, frame)
    occ["icao"] = departures["ADEP_mvt"].astype(str).to_numpy()
    # Carried only so the guard below can be given the identifiers it is meant
    # to look for. Nothing in the report reads it, and a stand identifier is
    # exactly what this generator refuses to write.
    occ["stand"] = departures["STAND_mvt"].astype(str).to_numpy()
    occ["month"] = month_of(departures["MVT_TIME_UTC_mvt"]).to_numpy()
    occ["mvt_id"] = departures["MVT_ID_mvt"].to_numpy()
    occ[TARGET] = pd.to_numeric(
        departures.get(TARGET, pd.Series(np.nan, index=departures.index)),
        errors="coerce",
    ).to_numpy()
    return occ.reset_index(drop=True)


def load_side(ranking: bool) -> pd.DataFrame:
    """One side's occupancy block, with the target where the side carries it."""
    columns = sorted(set(MOVEMENT_COLUMNS) | {TARGET})
    frame = to_pandas(
        load_ranking(DATA, columns) if ranking else load_movements(DATA, columns)
    ).reset_index(drop=True)
    return side(frame)


def load_residuals(split: str) -> pd.DataFrame | None:
    """One forward split's stored validation residuals for ``BOUND_RUNG``."""
    path = RESIDUALS / f"{split}_{BOUND_RUNG}.parquet"
    if not path.exists():
        return None
    got = pd.read_parquet(path)
    if "mvt_id" not in got.columns:
        return None
    return got


# --------------------------------------------------------------------------
# sections


def section_both_sides(train: pd.DataFrame, rank: pd.DataFrame) -> list[str]:
    """The gap to the stand's last arrival, the same month a year apart."""
    rows = []
    shifts = []
    for before, after in MONTH_PAIRS:
        a = train[train["month"] == before]
        b = rank[rank["month"] == after]
        for icao in sorted(set(a["icao"]) & set(b["icao"])):
            left = a[a["icao"] == icao]
            right = b[b["icao"] == icao]
            cells = [icao, before.split("-")[1] + " / " + after.split("-")[1]]
            for got in (left, right):
                gap = got["occ_gap1_s"].dropna()
                cells += [
                    num(gap.median() / 60.0, 1) if len(gap) else "n/a",
                    num(gap.quantile(0.9) / 60.0, 1) if len(gap) else "n/a",
                    pct(float((gap < 60 * PRESSURE_MIN).mean()), 1)
                    if len(gap)
                    else "n/a",
                    pct(float(got["occ_missing"].mean()), 2),
                ]
            rows.append(cells)
            if len(left) and len(right):
                shifts.append(
                    (
                        icao,
                        before,
                        float((left["occ_gap1_s"].dropna() < 60 * PRESSURE_MIN).mean()),
                        float((right["occ_gap1_s"].dropna() < 60 * PRESSURE_MIN).mean()),
                    )
                )
    worse = [s for s in shifts if s[3] - s[2] > 0.02]
    return [
        "## The gap, on both sides",
        "",
        "The minutes from the last arrival's on-block at the subject's stand to the",
        "subject's take-off, per airport, the same calendar month a year apart. The",
        "training side is 2025 and the ranking side 2026; July 2026 contains three",
        "airports, so only those three have a July row.",
        "",
        f"`under {PRESSURE_MIN} min` is the share of subjects whose stand was taken by an",
        "arrival within that time -- the share where the bound is sound and therefore",
        "the share where it says anything -- and `no arrival` is the share whose stand",
        "received none at all in the covered block, which is the block abstaining.",
        "Both are over the rows that have a gap; the medians and percentiles are too.",
        "",
        *table(
            [
                "ICAO",
                "month",
                "median 2025 (min)",
                "p90 2025",
                f"under {PRESSURE_MIN} min 2025",
                "no arrival 2025",
                "median 2026 (min)",
                "p90 2026",
                f"under {PRESSURE_MIN} min 2026",
                "no arrival 2026",
            ],
            rows,
            "--rrrrrrrr",
        ),
        "",
        (
            "**Stand pressure is higher in the ranking month at "
            + ", ".join(
                f"{icao} ({100 * (b - a):+.1f} points in {m.split('-')[0]})"
                for icao, m, a, b in worse
            )
            + ".** That is a regime the model carries into the ranking months rather"
            " than a defect of the feature: the gap means the same thing on both"
            " sides, and there is more of it where the stands are turned faster."
            " It is the same class of exposure as the plan-lag shift in"
            " [`congestion.md`](congestion.md), and it is bounded the same way --"
            " by what the block is worth at all, which"
            " [`baseline.md`](baseline.md) measures."
            if worse
            else "**No airport's stand pressure moves by more than two points between"
            " the two years**, so the gap is the same quantity on both sides and"
            " the model reads in 2026 what it was fitted on in 2025. That is what"
            " a feature built identically on both sides should look like, and it"
            " is checked rather than assumed."
        ),
    ]


def section_bound(occ: pd.DataFrame, residuals: dict) -> list[str]:
    """Where the bound holds, by gap band, split by whether the type matches."""
    lines = [
        "## Where the bound holds",
        "",
        "The bound says the taxi-out is at least the gap. That is sound reasoning",
        "about a stand and unsound reasoning about a stand *identifier*: an",
        "identifier can denote a position that holds two aircraft, and aircraft are",
        "towed off stands as well as taxied off them. So the question is empirical,",
        "and this is the measurement.",
        "",
        f"Validation rows of each forward split, scored by `{BOUND_RUNG}` -- the rung",
        "the block was added to, and therefore a model that has never read the gap.",
        "Over the rows the record-quality term leaves to the physical model. `actual",
        "at least gap` is how often the bound is true; `model below gap` is how often",
        "the model violates it; `bias` is the mean of prediction minus actual, so a",
        "positive figure is the model under-predicting.",
        "",
        "The last four columns split the same rows by whether the arrival that took",
        "the stand flew the subject's own aircraft type. A different type means a",
        "different aircraft took the stand after the subject pushed back; the same",
        "type is consistent with the subject's own inbound and a fast turnaround.",
        "That distinction is what the block's match flags carry, and it is where the",
        "signal is.",
    ]
    for split, train_month, rank_month in SPLITS:
        got = residuals.get(split)
        if got is None:
            continue
        joined = got.merge(
            occ[["mvt_id", "occ_gap1_s", "occ_prev_same_type", TARGET]],
            on="mvt_id",
            how="inner",
        )
        joined = joined[~joined["flagged"].astype(bool)]
        joined["pred"] = joined[TARGET] - joined["residual"]
        joined["band"] = bands(joined["occ_gap1_s"])
        rows = []
        for label in joined["band"].cat.categories:
            band = joined[joined["band"] == label]
            if not len(band):
                continue
            cells = [
                label,
                f"{len(band):,}",
                pct(float((band[TARGET] >= band["occ_gap1_s"]).mean()), 0),
                pct(float((band["pred"] < band["occ_gap1_s"]).mean()), 0),
                f"{float((band[TARGET] - band['pred']).mean()):+,.0f}",
            ]
            for same in (0.0, 1.0):
                part = band[band["occ_prev_same_type"] == same]
                cells += [
                    f"{len(part):,}",
                    (
                        f"{float((part[TARGET] - part['pred']).mean()):+,.0f}"
                        if len(part)
                        else "--"
                    ),
                ]
            rows.append(cells)
        lines += [
            "",
            f"### {split.title()} split ({train_month}, standing in for {rank_month})",
            "",
            *table(
                [
                    "gap (min)",
                    "rows",
                    "actual at least gap",
                    "model below gap",
                    "bias (s)",
                    "rows, type differs",
                    "bias",
                    "rows, type matches",
                    "bias",
                ],
                rows,
                "-rrrrrrrr",
            ),
        ]
    return lines


def section_floor(occ: pd.DataFrame, residuals: dict) -> list[str]:
    """The bound as a rule: clip the prediction up to the gap, and count the cost."""
    rows: list[list[str]] = []
    changes: list[float] = []
    for split, _, _ in SPLITS:
        got = residuals.get(split)
        if got is None:
            continue
        joined = got.merge(
            occ[["mvt_id", "occ_gap1_s", TARGET]], on="mvt_id", how="inner"
        )
        joined = joined[~joined["flagged"].astype(bool)]
        pred = joined[TARGET] - joined["residual"]
        gap = joined["occ_gap1_s"]
        base = rmse(joined[TARGET], pred)
        for cap in FLOOR_CAPS_MIN:
            applies = gap.notna() & (gap < 60 * cap) & (pred < gap)
            floored = rmse(joined[TARGET], pred.where(~applies, gap))
            changes.append(floored - base)
            rows.append(
                [
                    split.title(),
                    "any gap" if not np.isfinite(cap) else f"under {cap:.0f} min",
                    f"{int(applies.sum()):,}",
                    num(base, 1),
                    num(floored, 1),
                    f"{floored - base:+,.1f}",
                ]
            )
    best = min(changes, default=float("nan"))
    return [
        "## The hard floor, and what it costs",
        "",
        "The same bound applied as a rule rather than carried as a feature: where",
        "the stand's last arrival on-blocked at least the prediction ago, raise the",
        "prediction to the gap. It is the obvious thing to do with a lower bound and",
        "it is worth doing once, to find out that it is wrong.",
        "",
        "Each row applies the floor only where the gap is under a cap, on the",
        "reasoning that the bound is sound at short gaps and collapses at long ones.",
        "The rows it touches are those where the model predicted below the gap, which",
        "is the only place a floor can act. Ex-flagged RMSE, validation rows.",
        "",
        *table(
            [
                "split",
                "floor applied where",
                "rows raised",
                "before (s)",
                "after (s)",
                "change",
            ],
            rows,
            "--rrrr",
        ),
        "",
        (
            f"**Every arrangement of the floor loses**, the cheapest by {best:,.1f} s."
            if np.isfinite(best) and best > 0
            else "**One arrangement of the floor gains**, which is not what the bands"
            " above predict and is worth checking before it is believed."
        )
        + " The reason is in the band table: the bound is true where the model already"
        " respects it, and false where the model does not. A floor can therefore only"
        " act where the bound is wrong. Carried as a feature instead, the same"
        " quantity lets the booster find the band it holds in, and"
        " [`baseline.md`](baseline.md) says what that is worth.",
    ]


# --------------------------------------------------------------------------
# main


def main() -> int:
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to run", file=sys.stderr)
        return 1

    print("loading the training side ...")
    train = load_side(ranking=False)
    print(f"  {len(train):,} departures")
    print("loading the ranking side ...")
    rank = load_side(ranking=True)
    print(f"  {len(rank):,} departures")

    residuals = {split: load_residuals(split) for split, _, _ in SPLITS}
    absent = [s for s, got in residuals.items() if got is None]
    if absent:
        print(
            "WARNING: no residuals with a movement identifier for "
            + ", ".join(absent)
            + f" at rung {BOUND_RUNG}; run scripts/run_baselines.py",
            file=sys.stderr,
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = (
        [
            "# The stand as a shared resource",
            "",
            "Gate 2c of [`../docs/plan.md`](../docs/plan.md). A stand holds one",
            "aircraft. If an arrival on-blocks at the subject's stand *before the",
            "subject takes off*, the subject had already left the stand, so its",
            "taxi-out is at least the gap between that on-block and its take-off.",
            "The ranking set exposes every arrival's stand and on-block time in full,",
            "so that bound is built entirely from rows the data gives -- and it bounds",
            "a field the same data hides for departures, which is stated here rather",
            "than left to be noticed. Generated by `scripts/report_occupancy.py` from",
            f"the data under `data/challenge/` on {generated}; every figure below is",
            "computed there, and the generator refuses to write a literal stand",
            "identifier.",
            "",
            "What the block is worth on the ladder is in",
            "[`baseline.md`](baseline.md). This report is the construction and the",
            "two things a reader should want to know about it: whether the quantity",
            "means the same thing on both sides of the data, and where the bound",
            "behind it is actually true.",
            "",
        ]
        + section_both_sides(train, rank)
        + [""]
        + section_bound(train, residuals)
        + [""]
        + section_floor(train, residuals)
        + [
            "",
            "## What this block does not settle",
            "",
            "The gap is read from an arrival that on-blocked *after* the subject's own",
            "off-block. Under the challenge's post-hoc framing that is admissible --",
            "the subject's take-off time is given, and every take-off-anchored window",
            "the model already carries has the same standing -- and under an",
            "operational framing it is not. The distinction is recorded in",
            "*Leakage watch* in [`../docs/modelling_notes.md`](../docs/modelling_notes.md)",
            "and it is a reading a reader should be given rather than one this report",
            "should make quietly.",
            "",
            "Nothing here says which of the two aircraft the previous arrival was. The",
            "match flags are a proxy -- an aircraft of the subject's own type and",
            "operator on the subject's stand is consistent with the subject's own",
            "inbound, and consistent is not the same as identified -- and no exposed",
            "field links an inbound leg to its outbound. Rotation linkage by flight",
            "number was considered and left out for that reason: the same-type,",
            "same-operator match on the stand already carries most of it.",
            "",
            "Distance matrices derived from OpenStreetMap: data (c) "
            "OpenStreetMap contributors, ODbL 1.0. Movement and flight data",
            "supplied by EUROCONTROL PRC and the OpenSky Network under the terms",
            "of the PRC Data Challenge 2026.",
            "",
        ]
    )
    text = "\n".join(body)
    check_no_identifiers(
        text,
        set(train["stand"].dropna()) | set(rank["stand"].dropna()),
        load_matrix(),
    )
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
