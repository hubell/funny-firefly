#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Recompute Gate 2c's residual-fit headroom, and write ``reports/headroom.md``.

The script ``docs/review_headroom.md`` line 8 promised. That line says: "If any
of it is adopted, the script becomes ``scripts/report_headroom.py`` and the
numbers move to ``reports/``." §7 of the same file records that all four blocks
were built and all four earned their place, so the condition was met on
8 September 2026. Until this file existed, ``report_headroom.py`` appeared
exactly once in the repository -- in the sentence promising it -- and the
consequence was a gate whose expected side nobody could regenerate from the
published repository: ``scripts/run_baselines.py``'s ``EXPECTED_HEADROOM_S`` is
quoted from a review note whose own figures came from a diagnostic held
outside the tree.

What this is NOT. It does not move ``EXPECTED_HEADROOM_S``, it does not touch
``HEADROOM_TRIGGER_S``, and it changes no verdict. The leakage audit in
``reports/baseline.md`` goes on reading the quoted constant, and the figures
here are printed *beside* it as a second measurement with its own date and its
own method. A change that both adds an instrument and makes a firing gate stop
firing is indistinguishable from tuning the gate, whatever its commit message
says, so the two jobs are kept apart and this file is only the first of them.

The method, which is ``docs/review_headroom.md`` line 11 and nothing more:
take the stored validation residuals of the rung the review fitted against
(``data/processed/residuals/<split>_encoding-for-stand.parquet``), join them
back to the raw month, build a candidate block from the feature registry's own
columns for that block, and fit a small booster *on the residual* with
five-fold cross-validation grouped by take-off day. The reduction in RMSE
between the residual itself and the out-of-fold prediction of it is the
headroom that block could add to the rung, read the same way as the queue
ceiling in ``reports/congestion.md`` -- before against after, one fit, no
model refitted.

The capacity is a named constant here, and that is the substantive addition.
Line 11 says "a small booster" and leaves it there. The estimate depends on
the extractor's capacity -- heavily, and the report measures how much -- and an
unnamed capacity is how this quantity drifted out of comparability in the
first place. ``HEADROOM_BOOSTER_PARAMS`` below writes every parameter out and
says where each value comes from.

Needs the challenge data under ``data/challenge/``, the distance matrices from
``make reports`` (for the identifier guard alone), and the residuals
``scripts/run_baselines.py`` writes under ``data/processed/residuals/``. Fits
forty small boosters and nothing of the ladder: a few minutes.

Usage
-----
    python scripts/report_headroom.py
    make headroom
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_baselines import (  # noqa: E402
    EXPECTED_HEADROOM_S,
    HEADROOM_TRIGGER_S,
    check_no_identifiers,
    load_airport_table,
    load_matrix,
    num,
    table,
)
from sweep_params import FIRST_SWEEP  # noqa: E402
from taxiout.calendar import HOLIDAY_FEATURES, holiday_features  # noqa: E402
from taxiout.callsign import CALLSIGN_FEATURES, callsign_features  # noqa: E402
from taxiout.challenge import load_movements, to_pandas  # noqa: E402
from taxiout.destination import DESTINATION_FEATURES, destination_features  # noqa: E402
from taxiout.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS  # noqa: E402
from taxiout.models import BOOSTER_PARAMS, Booster, booster_backend  # noqa: E402
from taxiout.occupancy import OCCUPANCY_FEATURES, occupancy_features  # noqa: E402
from taxiout.occupancy import MOVEMENT_COLUMNS as OCCUPANCY_MOVEMENT_COLUMNS  # noqa: E402
from taxiout.planlags import PLAN_DIFF_FEATURES, plandiff_features  # noqa: E402
from taxiout.validation import JULY_AIRPORTS, rmse  # noqa: E402

DATA = REPO / "data" / "challenge"
RESIDUALS = REPO / "data" / "processed" / "residuals"
OUT = REPO / "reports" / "headroom.md"

# The rung whose stored residuals the estimate is made on. Named as a literal
# and NOT as ``taxiout.models.SUBMISSION_KEY``, which is now ``plan-diffs``:
# ``docs/review_headroom.md`` line 12 names this file, the four blocks of §2
# did not exist when the review was written, and all four of its estimates are
# made against this one frozen rung. Following ``SUBMISSION_KEY`` instead would
# fit each block on a residual that already carries the blocks beside it, which
# is a different quantity wearing the same name.
RESIDUAL_RUNG = "encoding-for-stand"

# The capacity of the residual fit, and the point of this file.
#
# ``docs/review_headroom.md`` line 11 says "a small booster" and says nothing
# else, so every figure in §2 is a figure about a capacity nobody wrote down.
# It is NOT ``taxiout.models.BOOSTER_PARAMS``: the ladder's cell is a model
# fitted on 1.9 million training rows to predict a taxi-out, and a cell that
# size turned loose on 166 thousand rows of a frozen model's residual overfits
# inside its own folds and reads a SMALLER headroom than a smaller cell does.
# *The Capacity* below measures that, and it is why this constant exists rather
# than an import.
#
# The reconstruction rule, stated so it can be argued with: the smallest value
# on each of the four axes of the grid that was live when §2 was written --
# ``scripts/sweep_params.py``'s ``FIRST_SWEEP``, the sweep of 3 September 2026
# -- read out of that constant rather than retyped. Trees, leaves and learning
# rate are then also that grid's smallest capacity. ``min_child_samples`` is
# the one axis where the smallest value is the LEAST regularized of the three,
# and 50 is taken for a second reason: it is the value the ladder itself
# carried on 8 September, so it is what a diagnostic written that week would
# have inherited rather than chosen.
#
# THE FIGURE IS A PROPERTY OF THIS CAPACITY AND NOT OF THE DATA ALONE. A
# headroom printed here is the answer to "what would a correction of this size,
# bolted on top of that frozen rung, recover"; change the size and the answer
# changes. That is the whole reason the constant is named, and the reason the
# report prints the same block at three capacities.
HEADROOM_BOOSTER_PARAMS: dict[str, object] = {
    "objective": "regression",
    # The four axes of ``FIRST_SWEEP``, at that grid's smallest value each.
    "n_estimators": min(FIRST_SWEEP["trees"]),
    "learning_rate": min(FIRST_SWEEP["learning_rate"]),
    "num_leaves": min(FIRST_SWEEP["num_leaves"]),
    "min_child_samples": min(FIRST_SWEEP["min_child_samples"]),
    # No column or row subsampling. The ladder carries 0.9 and 0.8; both are
    # deliberate additions to a fit, and "a small booster" is the reading that
    # did not add them. Written out rather than left to the library so that a
    # default changing under a release cannot move a published figure.
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "subsample_freq": 0,
    # LightGBM's own values for the two categorical controls, not the ladder's
    # 64 and 20.0, by the same reading: the ladder's pair was chosen for the
    # ladder's cell. One block here carries one categorical.
    "max_cat_threshold": 32,
    "cat_smooth": 10.0,
    # The repository's seed, as everywhere else.
    "random_state": 20260902,
    # These three are this file's own and are NOT part of what "a small
    # booster" could have meant: ``deterministic`` and ``force_col_wise`` were
    # adopted on 12 September, four days after §2 was written, and they are
    # here because a figure this repository publishes has to come back the same
    # on a second run. See the note beside ``BOOSTER_PARAMS`` for what each
    # removes. The thread count is pinned for the same reason and is part of
    # the result, not only of the speed.
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 8,
    "verbose": -1,
}

# Folds, and the grouping. Line 11 fixes both: "five-fold cross-validation
# grouped by day". Grouped by take-off day because the residual of a frozen
# model is correlated within a day -- a queue, a weather regime, a runway
# configuration all last hours -- and folds that split a day would let a fit
# read its own afternoon off its own morning and report the leak as headroom.
HEADROOM_FOLDS = 5

# The fold assignment's seed. Days are dealt round robin and the deal is
# permuted, exactly as ``taxiout.validation.assign_pair_folds`` deals pairs, so
# the assignment is a function of this constant and of the calendar and of
# nothing else.
HEADROOM_FOLD_SEED = 20260902

# The four blocks ``EXPECTED_HEADROOM_S`` carries, each as the feature
# registry's own columns for that block. The columns are imported and never
# listed here, so this estimator cannot read a column the block itself may not:
# ``block_columns`` refuses anything outside the entry, and the rung
# definitions in ``taxiout.models`` build the same four sets from the same
# constants.
HEADROOM_BLOCKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("occupancy", "stand occupancy", OCCUPANCY_FEATURES),
    ("callsign", "callsign prefix", CALLSIGN_FEATURES),
    (
        "destination",
        "destination and holidays",
        DESTINATION_FEATURES + HOLIDAY_FEATURES,
    ),
    ("plan-diffs", "plan differences", PLAN_DIFF_FEATURES),
)

# The columns the table is read on. The first two are what
# ``EXPECTED_HEADROOM_S`` is keyed on -- December over all ten airports, July
# over the three July 2026 contains -- and the third is §2's own middle column,
# printed so that every cell §2 filled in has a recomputed counterpart. The
# index is the position in ``EXPECTED_HEADROOM_S``'s pair, or ``None`` where
# the constant carries no figure for the column.
#
# One fit per (split, block), read on the airports each column names. That is
# ``reports/baseline.md``'s own convention for a three-airport column -- "each
# rung's ex-flagged RMSE over Frankfurt, Heathrow and Schiphol alone, pooled
# exactly from their own per-airport figures" -- and the measured side of the
# audit is read that way, so the estimate has to be read that way too or the
# two would differ by the reading rather than by the instrument.
LEGS: tuple[tuple[str, tuple[str, ...] | None, str, int | None], ...] = (
    ("december", None, "December, all ten", 0),
    ("december", JULY_AIRPORTS, "December, the three", None),
    ("july", JULY_AIRPORTS, "July, the three", 1),
)

# §2's table, quoted and not computed, for the one column
# ``EXPECTED_HEADROOM_S`` does not carry. Same arrangement as that constant and
# as ``HEADROOM_READINGS``: a number from a named document, quoted so that two
# counts of one quantity cannot disagree in the third digit. §2 left the cell
# blank for three of the four blocks, and a blank stays blank here.
SECTION_2_DECEMBER_THREE: dict[str, float] = {"occupancy": 5.0}

# ``docs/review_headroom.md`` §7's two chain-level readings, quoted: the refit
# against the estimate, July the three and December. The ratios are computed
# from them.
SECTION_7_READINGS: tuple[tuple[str, float, float], ...] = (
    ("the chain, July the three", 18.3, 10.8),
    ("the chain, December", 13.8, 6.1),
)

# What ``reports/baseline.md`` prints for the step this report's first ratio is
# about, quoted from that file. It is also computed here from the same stored
# residuals the ladder wrote, and the two are required to agree at the
# precision the report prints: a quoted figure that no longer matches the files
# it was read off is the defect this whole block is about.
LADDER_STEP_QUOTED_S = 17.60

# Capacities the occupancy block is re-read at, so that the sentence "the
# figure is a property of this capacity" is a measurement in the report and
# not an assertion in a comment. The ladder's own cell is imported rather than
# written down, so moving the parameters moves this with them.
CAPACITY_CHECK: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "the ladder's own cell (`taxiout.models.BOOSTER_PARAMS`)",
        {k: BOOSTER_PARAMS[k] for k in ("n_estimators", "learning_rate", "num_leaves")}
        | {"min_child_samples": BOOSTER_PARAMS["min_child_samples"]},
    ),
    (
        "LightGBM's own defaults on the same four axes",
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 20,
        },
    ),
)

CAPACITY_AXES = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")


# --------------------------------------------------------------------------
# the method


def block_columns(key: str) -> tuple[str, ...]:
    """The registry's own columns for one block, refusing anything else."""
    entries = {k: columns for k, _label, columns in HEADROOM_BLOCKS}
    if key not in entries:
        raise KeyError(f"not a headroom block: {key!r}")
    columns = entries[key]
    outside = [c for c in columns if c not in FEATURE_COLUMNS]
    if outside:
        raise KeyError(f"{key}: not feature columns: {sorted(outside)}")
    return tuple(columns)


def check_block_columns(key: str, columns: tuple[str, ...]) -> tuple[str, ...]:
    """``columns``, or ``KeyError`` naming any that the block may not read.

    The guard the review's method needs and does not state. A headroom that
    read one column from outside the block would not be an estimate of that
    block at all, and the failure would be invisible in the number.
    """
    allowed = set(block_columns(key))
    outside = sorted(set(columns) - allowed)
    if outside:
        raise KeyError(f"{key} may not read {outside}; its block is {sorted(allowed)}")
    return tuple(columns)


def day_folds(
    takeoff: pd.Series, *, folds: int = HEADROOM_FOLDS, seed: int = HEADROOM_FOLD_SEED
) -> pd.Series:
    """Fold index per row, dealing whole take-off days round robin.

    Whole days, so no day appears in two folds. Dealt round robin and then
    permuted rather than drawn independently, which keeps the folds within one
    day of each other in size on a month of any length.
    """
    day = pd.to_datetime(takeoff, utc=True).dt.strftime("%Y-%m-%d")
    days = np.array(sorted(day.dropna().unique()))
    rng = np.random.default_rng(seed)
    deal = rng.permutation(np.arange(len(days)) % folds)
    lookup = pd.Series(deal, index=days)
    return pd.Series(lookup.reindex(day).to_numpy(), index=takeoff.index, dtype=int)


def design(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """The block's columns, typed as ``taxiout.features.build_features`` types them."""
    return pd.DataFrame(
        {
            c: (
                frame[c].astype(str).astype("category")
                if c in CATEGORICAL_FEATURES
                else pd.to_numeric(frame[c], errors="coerce").astype("float32")
            )
            for c in columns
        },
        index=frame.index,
    )


def out_of_fold(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """Out-of-fold prediction of ``frame['residual']`` from ``columns``.

    One small booster per fold, fitted on the other four and predicting the
    held one, so every row's prediction is made by a model that never saw its
    day.
    """
    X = design(frame, columns)
    y = frame["residual"].to_numpy(dtype=float)
    fold = frame["fold"].to_numpy()
    oof = np.full(len(frame), np.nan)
    for k in range(HEADROOM_FOLDS):
        held = fold == k
        if not held.any() or held.all():
            raise ValueError(f"fold {k} holds {int(held.sum())} of {len(frame)} rows")
        fitted = Booster(
            columns, dict(HEADROOM_BOOSTER_PARAMS if params is None else params)
        ).fit(X[~held], y[~held])
        oof[held] = fitted.predict(X[held]).to_numpy()
    if not np.isfinite(oof).all():
        raise ValueError("a row was never held out")
    return oof


def headroom_on(
    frame: pd.DataFrame, oof: np.ndarray, icaos: tuple[str, ...] | None
) -> dict:
    """The RMSE before and after, and their difference, over one column's rows."""
    keep = (
        np.ones(len(frame), dtype=bool)
        if icaos is None
        else frame["icao"].isin(icaos).to_numpy()
    )
    y = frame["residual"].to_numpy(dtype=float)[keep]
    before = rmse(y, np.zeros(len(y)))
    after = rmse(y, oof[keep])
    return {
        "n": int(keep.sum()),
        "before": before,
        "after": after,
        "gain": before - after,
    }


# --------------------------------------------------------------------------
# data


def load_side() -> pd.DataFrame:
    """The training year's movements, both phases, with what the four blocks read."""
    columns = sorted(
        set(OCCUPANCY_MOVEMENT_COLUMNS)
        | {"FLIGHT_mvt", "ADES_mvt", "EOBT_1_flt", "LOBT_flt", "RUNWAY_mvt"}
    )
    return to_pandas(load_movements(DATA, columns)).reset_index(drop=True)


def build_blocks(frame: pd.DataFrame, airports: pd.DataFrame | None) -> pd.DataFrame:
    """Every block's columns for the side's departures, keyed on the movement id.

    Each block is built by its own module from that module's own inputs, and
    the occupancy block is given the whole side as its event source, exactly as
    the model builds it: a subset would move the coverage flags and the gaps
    near a month boundary.
    """
    departures = frame[frame["PHASE_mvt"].astype(str) == "DEP"]
    blocks = [
        occupancy_features(departures, frame),
        callsign_features(departures),
        destination_features(departures, airports),
        holiday_features(departures),
        plandiff_features(departures),
    ]
    out = pd.concat(blocks, axis=1)
    out["mvt_id"] = departures["MVT_ID_mvt"].to_numpy()
    # Carried only so the identifier guard can be given the identifiers it is
    # meant to look for. Nothing in the report reads it.
    out["stand"] = departures["STAND_mvt"].astype(str).to_numpy()
    return out.reset_index(drop=True)


def load_residuals(split: str) -> pd.DataFrame | None:
    """One forward split's stored ex-flagged validation residuals at the arm."""
    path = RESIDUALS / f"{split}_{RESIDUAL_RUNG}.parquet"
    if not path.exists():
        return None
    got = pd.read_parquet(path)
    if "mvt_id" not in got.columns:
        return None
    # Ex-flagged, because §2's table is ex-flagged and because the ladder fits
    # nothing on a flagged row. ``reports/baseline.md`` quotes December's file
    # at 165,677 rows, which is the file including its flagged rows.
    return got[~got["flagged"].astype(bool)].reset_index(drop=True)


def residual_files() -> list[list[str]]:
    """The two input files, with their sizes and modification times."""
    rows = []
    for split in dict.fromkeys(s for s, _i, _l, _x in LEGS):
        path = RESIDUALS / f"{split}_{RESIDUAL_RUNG}.parquet"
        if not path.exists():
            continue
        written = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        rows.append(
            [
                f"`{path.relative_to(REPO).as_posix()}`",
                written.strftime("%Y-%m-%d %H:%M:%S UTC"),
                num(path.stat().st_size / 1e6, 1),
            ]
        )
    return rows


def ladder_step(split: str, key: str, icaos: tuple[str, ...] | None) -> float | None:
    """The ladder's own measured step for one block, from its stored residuals.

    ``reports/baseline.md`` prints these; this reads them back off the files
    the ladder wrote, so the quoted figure can be checked rather than trusted.
    """
    order = [k for k, _l, _c in HEADROOM_BLOCKS]
    below = RESIDUAL_RUNG if key == order[0] else order[order.index(key) - 1]
    rmses = []
    for rung in (below, key):
        path = RESIDUALS / f"{split}_{rung}.parquet"
        if not path.exists():
            return None
        got = pd.read_parquet(path)
        got = got[~got["flagged"].astype(bool)]
        if icaos is not None:
            got = got[got["icao"].isin(icaos)]
        rmses.append(rmse(got["residual"].to_numpy(dtype=float), np.zeros(len(got))))
    return rmses[0] - rmses[1]


# --------------------------------------------------------------------------
# sections


def section_method(joined: dict) -> list[str]:
    """What was fitted, on what, and the fold assignment's own check."""
    rows = []
    for split in dict.fromkeys(s for s, _i, _l, _x in LEGS):
        if split not in joined:
            continue
        got = joined[split]
        sizes = got["fold"].value_counts().sort_index()
        days = got.groupby("fold")["day"].nunique()
        rows.append(
            [
                f"`{split}`",
                num(len(got)),
                num(int(got["day"].nunique())),
                num(int(sizes.min())) + " to " + num(int(sizes.max())),
                num(int(days.min())) + " to " + num(int(days.max())),
            ]
        )
    return [
        "## The method, and what it was run on",
        "",
        "`docs/review_headroom.md` line 11, which is the whole specification:",
        "take the stored validation residuals of the rung the review fitted against,",
        "join them back to the raw month, build a candidate block, and fit a small",
        "booster *on the residual* with five-fold cross-validation grouped by day. The",
        "RMSE reduction is the headroom. Read the same way as the queue ceiling in",
        "[`congestion.md`](congestion.md): the residual against an out-of-fold",
        "prediction of it, before against after, with no model refitted.",
        "",
        f"The rung is `{RESIDUAL_RUNG}` and not `taxiout.models.SUBMISSION_KEY`, which",
        "is now `plan-diffs`. That is the file line 12 names, and it is the right one:",
        "none of the four blocks existed when the review was written, so all four of",
        "its estimates are made against this one frozen rung. Fitting each block on a",
        "residual that already carried the blocks beside it would be a different",
        "quantity wearing the same name.",
        "",
        "Rows are the split's own validation rows, ex-flagged, since §2's table is",
        "ex-flagged and the ladder fits nothing on a flagged row.",
        "",
        *table(
            ["split", "rows", "days", "fold sizes", "days per fold"],
            rows,
            "-rrrr",
        ),
        "",
        f"The folds are {HEADROOM_FOLDS}, whole take-off days dealt round robin from",
        f"`HEADROOM_FOLD_SEED = {HEADROOM_FOLD_SEED}` and permuted. **No day appears in",
        "two folds**, which is checked in the generator and in",
        "`tests/test_headroom.py` rather than asserted here: the residual of a frozen",
        "model is correlated within a day -- a queue, a weather regime and a runway",
        "configuration all last hours -- and a fold that split a day would let a fit",
        "read its own afternoon off its own morning and report the leak as headroom.",
        "",
        "Every column of the table below is one fit per (split, block), read on the",
        "airports the column names. That is [`baseline.md`](baseline.md)'s own",
        "convention for a three-airport column, and the measured side of the leakage",
        "audit is read that way, so the estimate is read that way too: otherwise the",
        "two sides would differ by the reading rule and not by the instrument.",
        "",
        "The blocks are the four `scripts/run_baselines.py`'s `EXPECTED_HEADROOM_S`",
        "carries, each built from the feature registry's own columns for that block:",
        "",
        *table(
            ["block", "columns"],
            [
                [label, ", ".join(f"`{c}`" for c in block_columns(key))]
                for key, label, _columns in HEADROOM_BLOCKS
            ],
            "--",
        ),
        "",
        "The estimator refuses a column outside the block's registry entry. A headroom",
        "that read one column from outside the block would not be an estimate of that",
        "block at all, and the failure would be invisible in the number.",
    ]


def section_capacity(capacity: list[list[str]]) -> list[str]:
    """The capacity constant, written out, and what it is worth."""
    written = [
        [f"`{k}`", str(HEADROOM_BOOSTER_PARAMS[k])]
        for k in (
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "colsample_bytree",
            "subsample",
            "subsample_freq",
            "max_cat_threshold",
            "cat_smooth",
            "random_state",
            "deterministic",
            "force_col_wise",
            "n_jobs",
        )
    ]
    return [
        "## The capacity, named",
        "",
        '`docs/review_headroom.md` line 11 says "a small booster" and says nothing',
        "else. Every figure in §2 is therefore a figure about a capacity nobody wrote",
        "down, and **this report's figures are not a recovery of the original ones: the",
        "capacity below is a reconstruction.** The original script is not in the",
        "repository, its parameters are recorded nowhere in it, and no reading of line",
        "11 fixes them. What can be fixed is that they are never unwritten again.",
        "",
        "`HEADROOM_BOOSTER_PARAMS`, in full:",
        "",
        *table(["parameter", "value"], written, "--"),
        "",
        "**The rule that chose the first four**, stated so it can be argued with: the",
        "smallest value on each of the four axes of the grid that was live when §2 was",
        "written, which is `scripts/sweep_params.py`'s `FIRST_SWEEP`, the sweep of",
        "3 September 2026. Those values are read out of that constant rather than",
        "retyped here. On trees, leaves and learning rate the smallest value is also",
        "that grid's smallest capacity. `min_child_samples` is the one axis where the",
        "grid's smallest value is the *least* regularized of the three, and 50 is taken",
        "for a second reason as well: it is the value the ladder itself carried on",
        "8 September, so it is what a diagnostic written that week would have inherited",
        "rather than chosen.",
        "",
        "**What the rule is not.** It is not pre-registered. The rule was fixed after",
        "four candidate cells had been read on the stand-occupancy block in a scratch",
        "directory outside this repository, so it was chosen by someone who had already",
        "seen what capacity does to this number. It is mechanical, it is stated, and it",
        "is checkable; it is not blind, and the difference is worth the sentence.",
        "",
        "The last three parameters are this file's own and are **not** part of what \"a",
        'small booster" could have meant. `deterministic` and `force_col_wise` were',
        "adopted on 12 September, four days after §2 was written, and they are here",
        "because a figure this repository publishes has to come back the same on a",
        "second run; the note beside `taxiout.models.BOOSTER_PARAMS` says what each of",
        "them removes. `n_jobs` is pinned for the same reason and is part of the result",
        "and not only of the speed.",
        "",
        "### What the capacity is worth",
        "",
        "The stand-occupancy block, the same rows and the same folds, at three",
        "capacities. This is the measurement behind the sentence in the constant's own",
        'comment, so that "the figure is a property of this capacity" is a number here',
        "and not an assertion in a comment.",
        "",
        *table(
            [
                "capacity",
                "trees",
                "rate",
                "leaves",
                "min child",
                "December, all ten (s)",
                "July, the three (s)",
            ],
            capacity,
            "-rrrrrr",
        ),
        "",
        "**The ladder's own cell reads the smallest headroom of the three.** That is",
        "the direction to expect and the reason this constant is not an import: a cell",
        "fitted on 1.9 million training rows to predict a taxi-out, turned loose on a",
        "few hundred thousand rows of a frozen model's residual, has capacity to spend",
        "inside its own folds and spends it. A residual fit at that size would not",
        "reproduce §2's numbers even in principle, and it would not be the instrument",
        "§2 used.",
    ]


def section_table(results: dict) -> list[str]:
    """The recomputed table beside §2's quoted figures, cell by cell."""
    rows = []
    keyed: list[tuple[str, str, float, float]] = []
    for key, label, _columns in HEADROOM_BLOCKS:
        for _split, _icaos, leg, idx in LEGS:
            got = results.get((key, leg))
            if got is None:
                continue
            quoted = (
                EXPECTED_HEADROOM_S[key][idx]
                if idx is not None
                else SECTION_2_DECEMBER_THREE.get(key)
            )
            if idx is not None:
                keyed.append((label, leg, got["gain"], quoted))
            rows.append(
                [
                    label,
                    leg,
                    num(got["n"]),
                    num(got["before"], 2),
                    num(got["after"], 2),
                    f"{got['gain']:+.2f}",
                    "n/a" if quoted is None else f"{quoted:+.1f}",
                    "n/a" if quoted is None else f"{got['gain'] - quoted:+.2f}",
                ]
            )
    gaps = [abs(new - quoted) for _label, _leg, new, quoted in keyed]
    closest = min(keyed, key=lambda r: abs(r[2] - r[3]))
    widest = max(keyed, key=lambda r: abs(r[2] - r[3]))
    under = sum(1 for _label, _leg, new, quoted in keyed if new < quoted)
    # Whether the instrument still ranks the four blocks the way §2's figures
    # do, which §7 says is the one thing a residual fit is good for. Computed
    # per column rather than claimed: the ordering is the finding either way.
    orders = {}
    for leg in dict.fromkeys(leg for _label, leg, _n, _q in keyed):
        cells = [(label, new, quoted) for label, got, new, quoted in keyed if got == leg]
        orders[leg] = (
            [label for label, _n, _q in sorted(cells, key=lambda c: -c[1])],
            [label for label, _n, _q in sorted(cells, key=lambda c: -c[2])],
        )
    agreeing = [leg for leg, (a, b) in orders.items() if a == b]
    tops = [leg for leg, (a, b) in orders.items() if a[0] == b[0]]
    # Where the ordering disagrees, how far apart §2 itself put the blocks that
    # swapped. A swap between two figures a tenth of a second apart is a
    # different finding from a swap between two a second apart.
    swaps = []
    for leg, (mine, quoted) in orders.items():
        if mine == quoted:
            continue
        moved = [q for q, m in zip(quoted, mine) if q != m]
        cells = {label: q for label, got, _n, q in keyed if got == leg}
        if len(moved) >= 2:
            span = max(cells[m] for m in moved) - min(cells[m] for m in moved)
            swaps.append((leg, moved, span))
    return [
        "## The recomputed headroom, beside the quoted figures",
        "",
        "`before` is the RMSE of the arm's residual itself over the column's rows;",
        "`after` is the RMSE of the same residual against the out-of-fold prediction of",
        "it. `recomputed` is the difference, and it is the headroom. `quoted` is",
        "`scripts/run_baselines.py`'s `EXPECTED_HEADROOM_S` for the two columns that",
        "constant is keyed on and §2's own table for `December, the three`, which §2",
        "filled in for one block only; a cell §2 left blank is left blank here.",
        "",
        *table(
            [
                "block",
                "column",
                "rows",
                "before (s)",
                "after (s)",
                "recomputed (s)",
                "quoted (s)",
                "difference (s)",
            ],
            rows,
            "--rrrrrr",
        ),
        "",
        "**The two columns are not two measurements of one thing, and nothing here",
        "should be read as a reproduction.** The quoted figures were made on",
        "8 September 2026 by a diagnostic script held outside this repository, at a",
        "capacity recorded nowhere, on residuals that the ladder has since rewritten:",
        "the two files this report reads were written on 21 September 2026 by the",
        "ladder that followed the one permitted wider sweep, which moved the booster to",
        "3,000 trees and 255 leaves. A frozen model's residual is the output of that",
        "model, so a different arm gives a different residual and the same method gives",
        "a different number. The `difference` column is therefore the sum of three",
        "things -- the arm moved, the capacity is a reconstruction, and the fold",
        "assignment is this file's -- and it is not an error bar.",
        "",
        "**The plan-difference row is not a signed comparison of like with like.**",
        "`EXPECTED_HEADROOM_S`'s own comment says why: the review gave the two",
        'difference columns as "expected +/- 1 s" rather than as a signed estimate, so',
        "that row's quoted figure is a magnitude and not a direction, and its",
        "`difference` cell inherits that.",
        "",
        "**How far the reconstruction lands.** Over the eight cells",
        "`EXPECTED_HEADROOM_S` is keyed on, the absolute difference runs from",
        f"{min(gaps):.2f} s to {max(gaps):.2f} s and averages",
        f"{sum(gaps) / len(gaps):.2f} s, and {under} of the eight come in **under** the",
        "quoted figure. The closest is",
        f"{closest[0]} on {closest[1]}, at {abs(closest[2] - closest[3]):.2f} s, which is",
        "the cell the leakage audit reads; the widest is",
        f"{widest[0]} on {widest[1]}, at {abs(widest[2] - widest[3]):.2f} s.",
        "",
        "**The ranking is where it should be read"
        + (
            ", and it holds.**"
            if len(agreeing) == len(orders)
            else ", and it holds in part.**"
            if tops
            else ", and it does not hold.**"
        )
        + " §7 says a",
        'headroom made this way "is still the right instrument for deciding what to',
        "*build* -- it is cheap, and it ranks candidates correctly, which is all §2 was",
        f'used for". Against that claim, on the {len(orders)} keyed columns:',
        "",
        *table(
            ["column", "recomputed, largest first", "quoted, largest first", "same"],
            [
                [leg, " > ".join(a), " > ".join(b), "yes" if a == b else "no"]
                for leg, (a, b) in orders.items()
            ],
            "----",
        ),
        "",
        f"The ordering agrees on {len(agreeing)} of the {len(orders)} columns and the"
        f" largest block is the same on {len(tops)} of {len(orders)}."
        + (
            " Where it disagrees, what swaps is "
            + " and ".join(
                f"{' against '.join(moved)} on {leg}, which §2 put {span:.1f} s apart"
                for leg, moved, span in swaps
            )
            + ". So what the reconstruction fails to reproduce is an ordering §2"
            " itself read off figures that close together, and not the ranking that"
            " decided anything: §2's recommended set took the largest block and the"
            " two that swapped both entered the chain."
            if swaps
            else ""
        ),
        "",
        "So: a reconstruction that agreed on every cell would be evidence the capacity",
        "had been recovered rather than reconstructed, and this is not that. It sits",
        f"below the quoted figures in {under} of the eight keyed cells, it picks the",
        f"same block out at the top of {len(tops)} of {len(orders)} columns by a wide",
        "margin, and it does not order the two middle blocks the way §2 did.",
        "",
        "What the two columns do share is the method of line 11 and the rung of line",
        "12, and what this report buys is that the left column can now be regenerated",
        "by anyone with the challenge data and one `make` target.",
    ]


def section_audit(results: dict, step: dict) -> list[str]:
    """The ratios the analysis note needs, and the audit's verdict, unchanged."""
    key = "occupancy"
    leg = "July, the three"
    new = results[(key, leg)]["gain"]
    quoted_estimate = EXPECTED_HEADROOM_S[key][1]
    measured = step[(key, leg)]
    lines = [
        "## What this says about the leakage audit, and what it does not",
        "",
        "**The audit is unchanged by this report.** `scripts/run_baselines.py`'s",
        "leakage audit reads the quoted `EXPECTED_HEADROOM_S` and the",
        f"`HEADROOM_TRIGGER_S = {HEADROOM_TRIGGER_S:.1f}` s allowance, both untouched",
        "here, and [`baseline.md`](baseline.md) is not regenerated by this target. So",
        "that a reader of this file alone is not misled into thinking the gate passes:",
        "on `+ stand occupancy` and July, the three it **fires**, at an overshoot of",
        f"{measured - quoted_estimate:.3f} s against the {HEADROOM_TRIGGER_S:.3f} s the",
        "trigger allows, and it has fired since the ladder of 21 September.",
        "",
        "### The ratios",
        "",
        "The ladder's measured step for this block on this column, read back off the",
        "residual files the ladder wrote rather than quoted from the report that prints",
        f"it: **{measured:+.4f} s**, against the {LADDER_STEP_QUOTED_S:+.2f} s",
        "[`baseline.md`](baseline.md) prints. The two agree, which is worth checking",
        "and not assuming, because a quoted figure that no longer matches the files it",
        "was read off is the whole defect this report exists to close.",
        "",
        *table(
            ["the estimate it is read against", "estimate (s)", "measured (s)", "ratio"],
            [
                [
                    "recomputed here, at the named capacity, on the 21 September"
                    " residuals",
                    f"{new:+.2f}",
                    f"{measured:+.2f}",
                    f"{measured / new:.2f}",
                ],
                [
                    "quoted `EXPECTED_HEADROOM_S`, from 8 September",
                    f"{quoted_estimate:+.1f}",
                    f"{measured:+.2f}",
                    f"{measured / quoted_estimate:.2f}",
                ],
            ],
            "-rrr",
        ),
        "",
        f"The overshoot against the recomputed figure is {measured - new:+.2f} s; the",
        "overshoot the audit reads, against the quoted one, is",
        f"{measured - quoted_estimate:+.3f} s.",
        "",
        "### §7's finding, which is the reason to expect a ratio near two",
        "",
        "`docs/review_headroom.md` §7, written 8 September after the four blocks were",
        "built and the ladder refitted, quoted:",
        "",
        "> **The estimates above are a floor and not an estimate, and that is the",
        "> methodological finding.** [...] Fitting a small booster *on the stored",
        "> residuals* of a frozen model measures what a correction bolted on top of",
        "> that model would recover. A refit does something strictly larger: it",
        "> restructures every split in the model around the new column and can use it",
        "> to condition the blocks already there. So a residual fit bounds a refit from",
        "> below, and the gap between them is the value of the interaction rather than",
        "> evidence of anything wrong.",
        "",
        "Its own two chain-level readings, quoted from the same section, with the",
        "ratios computed:",
        "",
        *table(
            ["reading", "measured (s)", "estimate (s)", "ratio"],
            [
                [label, f"{m:+.1f}", f"{e:+.1f}", f"{m / e:.2f}"]
                for label, m, e in SECTION_7_READINGS
            ],
            "-rrr",
        ),
        "",
        "Those two are chain-level and the block figures above are one block's, which",
        "is a difference worth stating; the mechanism §7 gives applies to both.",
        "",
        "**This report draws no conclusion about the trigger.** The decision rule that",
        "reads these ratios was fixed in an analysis note outside the repository before",
        "the recomputed figure existed, and acting on it -- refreshing the constant,",
        "restating the audit as a ratio test, or facing an overshoot that interaction",
        "does not explain -- is a second job and a separate change. Doing both in one",
        "motion would add an instrument and move a firing gate at the same time, which",
        "is indistinguishable from tuning the gate whatever the commit message says.",
        "This report supplies the recomputed headroom and stops.",
    ]
    return lines


def section_inputs() -> list[str]:
    """The residual files read, with their modification times."""
    return [
        "## The inputs, and why their dates are printed",
        "",
        "This quantity is a function of which residuals it was measured on, so the",
        "files are named with the times they were written. A headroom recomputed on",
        "other residuals is another number, and the defect this report closes was",
        "exactly a figure whose inputs nobody could identify.",
        "",
        *table(
            ["file", "written (UTC)", "size (MB)"],
            residual_files(),
            "--r",
        ),
        "",
        "Both were written by the ladder of 21 September 2026, the first run after",
        "the one permitted wider sweep. The figures they are compared against were made on",
        "8 September, on residuals of an arm fitted at a smaller capacity, by a script",
        "that is not in this repository.",
    ]


# --------------------------------------------------------------------------
# main


def main() -> int:
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to run", file=sys.stderr)
        return 1
    if booster_backend() == "none":
        print('no booster installed; pip install -e ".[modelling]"', file=sys.stderr)
        return 1
    absent = [
        f"{split}_{RESIDUAL_RUNG}.parquet"
        for split in dict.fromkeys(s for s, _i, _l, _x in LEGS)
        if not (RESIDUALS / f"{split}_{RESIDUAL_RUNG}.parquet").exists()
    ]
    if absent:
        print(
            "missing residuals: " + ", ".join(absent) + "; run scripts/run_baselines.py",
            file=sys.stderr,
        )
        return 1

    t0 = time.time()
    print("loading the training year ...")
    frame = load_side()
    print(f"  {len(frame):,} movements in {time.time() - t0:.0f} s")
    airports = load_airport_table()
    if airports is None:
        print(
            "no OurAirports cache under data/raw; the destination block would be a "
            "column of nothing (python scripts/fetch_ourairports.py)",
            file=sys.stderr,
        )
        return 1

    t0 = time.time()
    print("building the four blocks ...")
    blocks = build_blocks(frame, airports)
    del frame
    print(f"  {len(blocks):,} departures in {time.time() - t0:.0f} s")

    joined: dict[str, pd.DataFrame] = {}
    for split in dict.fromkeys(s for s, _i, _l, _x in LEGS):
        residuals = load_residuals(split)
        if residuals is None:
            print(f"no usable residuals for {split}", file=sys.stderr)
            return 1
        got = residuals.merge(blocks, on="mvt_id", how="inner")
        if len(got) != len(residuals):
            print(
                f"{split}: {len(residuals):,} residual rows joined to {len(got):,} "
                "block rows; the join is not one to one",
                file=sys.stderr,
            )
            return 1
        got["day"] = pd.to_datetime(got["takeoff"], utc=True).dt.strftime("%Y-%m-%d")
        got["fold"] = day_folds(got["takeoff"])
        crossing = got.groupby("day")["fold"].nunique()
        if int(crossing.max()) != 1:
            print("a day was dealt to two folds", file=sys.stderr)
            return 1
        joined[split] = got

    results: dict[tuple[str, str], dict] = {}
    for key, label, _columns in HEADROOM_BLOCKS:
        checked = check_block_columns(key, block_columns(key))
        for split in joined:
            t0 = time.time()
            oof = out_of_fold(joined[split], checked)
            for leg_split, icaos, leg, _idx in LEGS:
                if leg_split != split:
                    continue
                results[(key, leg)] = headroom_on(joined[split], oof, icaos)
            print(
                f"  {label}, {split}: "
                + ", ".join(
                    f"{leg} {results[(key, leg)]['gain']:+.2f} s"
                    for s, _i, leg, _x in LEGS
                    if s == split
                )
                + f"   {time.time() - t0:.0f} s"
            )

    print("the capacity check ...")
    capacity = [
        [
            "**this report** (`HEADROOM_BOOSTER_PARAMS`)",
            *(str(HEADROOM_BOOSTER_PARAMS[a]) for a in CAPACITY_AXES),
            f"{results[('occupancy', 'December, all ten')]['gain']:+.2f}",
            f"{results[('occupancy', 'July, the three')]['gain']:+.2f}",
        ]
    ]
    for label, cell in CAPACITY_CHECK:
        params = dict(HEADROOM_BOOSTER_PARAMS) | cell
        cells = []
        for leg_split, icaos, leg, _idx in LEGS:
            if leg not in ("December, all ten", "July, the three"):
                continue
            t0 = time.time()
            oof = out_of_fold(
                joined[leg_split], block_columns("occupancy"), params=params
            )
            got = headroom_on(joined[leg_split], oof, icaos)
            cells.append(f"{got['gain']:+.2f}")
            print(f"  {label}, {leg}: {got['gain']:+.2f} s   {time.time() - t0:.0f} s")
        capacity.append([label, *(str(cell[a]) for a in CAPACITY_AXES), *cells])

    step: dict[tuple[str, str], float] = {}
    for key, _label, _columns in HEADROOM_BLOCKS:
        for split, icaos, leg, _idx in LEGS:
            got = ladder_step(split, key, icaos)
            if got is not None:
                step[(key, leg)] = got
    quoted = LADDER_STEP_QUOTED_S
    measured = step.get(("occupancy", "July, the three"))
    if measured is None or abs(measured - quoted) >= 0.005:
        print(
            f"the ladder's step on July, the three reads {measured} against the "
            f"{quoted} baseline.md prints; the quoted figure and the files disagree",
            file=sys.stderr,
        )
        return 1

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = (
        [
            "# The residual-fit headroom, recomputed",
            "",
            "The script [`../docs/review_headroom.md`](../docs/review_headroom.md) line",
            "8 promised. That line says that if any of the review's four candidate",
            'blocks were adopted, "the script becomes `scripts/report_headroom.py` and',
            'the numbers move to `reports/`". §7 of the same file records that all four',
            "were built and all four earned their place, so the condition was met on",
            "8 September 2026 and this file is the number moving. Generated by",
            f"`scripts/report_headroom.py` on {generated}; every figure below is",
            "computed there, and the generator refuses to write a literal stand",
            "identifier.",
            "",
            "**Why it matters that this exists.**",
            "`scripts/run_baselines.py`'s `EXPECTED_HEADROOM_S` gates the",
            "stand-occupancy leakage audit and is quoted, not computed, from a review",
            "note whose own figures came from a script held outside the repository.",
            "Until this file, nobody could regenerate the expected side of a firing gate",
            "from the published repository. That was true whether or not it fired, and",
            "it is the thing this report fixes.",
            "",
            "**What it does not do.** It moves no threshold, no constant and no verdict.",
            "`EXPECTED_HEADROOM_S` and `HEADROOM_TRIGGER_S` are untouched, the audit in",
            "[`baseline.md`](baseline.md) still reads the quoted constant, and the",
            "figures here are printed beside it as a second measurement with its own",
            "date and method rather than in place of it.",
            "",
        ]
        + section_method(joined)
        + [""]
        + section_capacity(capacity)
        + [""]
        + section_table(results)
        + [""]
        + section_audit(results, step)
        + [""]
        + section_inputs()
        + [
            "",
            "Distance matrices derived from OpenStreetMap: data (c) "
            "OpenStreetMap contributors, ODbL 1.0. Movement and flight data",
            "supplied by EUROCONTROL PRC and the OpenSky Network under the terms",
            "of the PRC Data Challenge 2026.",
            "",
        ]
    )
    text = "\n".join(body)
    check_no_identifiers(text, set(blocks["stand"].dropna()), load_matrix())
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
