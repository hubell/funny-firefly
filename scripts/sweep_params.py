#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The parameter sweeps: fit January to September 2025, score October.

Permitted twice, on October 2025 alone, and never reported: the ladder's
splits -- December, July, the held-out pairs -- are the sets the write-up
reads, so nothing may be chosen on them, and the leaderboard is not a
validation set. October is a month the ladder never scores, so a number read
from it chooses parameters without touching what is reported.

Two sweeps have run on this leg. This file runs the second and keeps the
first one's terms, because the first one is what chose the parameters
``taxiout.models`` carries.

**The first sweep** (its numbers are dated 3 September 2026 on disk, under
``data/processed/``) ran on the saturation rung, the top of the ladder at the
time, over learning rate (0.03, 0.06, 0.1), leaves (31, 63, 127), minimum
child samples (50, 200, 500) and trees (300, 600, 1000), fitting on
``train & ~flagged`` with no four-hour exclusion and no shrunk encoding,
because that is how the ladder fitted then. Its terms are kept below as
``FIRST_SWEEP`` so that what produced the numbers in ``taxiout.models`` stays
readable here; the numbers themselves are in the comment beside
``BOOSTER_PARAMS``.

**The second sweep**, pre-registered on 20 September 2026 and permitted once,
runs on the submission rung under the fit rules that stand today:
``taxiout.models.fit_rows`` (train, unflagged, recorded taxi-out at most four
hours), the shrunk encoding fitted on those rows and overlaid, the
record-quality model fitted on the leg's training months, the submission
rung's features including the season columns, and no recency schedule. Those
rules are ``scripts/preregistered.py``'s ``Frame.prepare`` and ``fit_once``,
which follow ``scripts/run_baselines.run``. This file imports them rather
than restating them, so that a change to the fit rules cannot pass it by.

**The two sweeps' October figures are not comparable, and the difference
between them is not something a sweep bought.** The first scored the
saturation rung with no shrunk encoding and no four-hour exclusion; the same
parameter cell under today's rules scores 217.28 s
(``reports/preregistered-2026-09-20.md``, Experiment 1). Between 235.0 s and
217.3 s lie the feature block and the fit rules, not the parameters.

The number of trees is read from one fit per cell of the other three: the
first ``k`` trees of a boosted model are the same model whatever the total,
so a 3,000-tree fit scores 300, 600, 1,000, 1,500, 2,000 and 3,000 at once.
That claim is checked rather than assumed here: the current cell's 3,000-tree
fit read at 1,000 trees must equal the 1,000-tree fit of the same cell at the
same seed, to the bit. It is a check and not a criterion; it changes no
reading rule and no grid value, and it protects every figure the grid makes.

The chosen parameters are written into ``BOOSTER_PARAMS`` in
``taxiout.models`` by hand, with the October number that chose them, and the
ladder is run once more. The grid's own numbers stay under
``data/processed/``, which is gitignored: they are not a result, they are how
the parameters were chosen, and the README says so.

Cell labels are written out in full -- ``learning_rate=0.015,
num_leaves=255, min_child_samples=20, n_estimators=3000`` -- and never
abbreviated. A short label of letters and digits can equal a stand
identifier at one of the ten airports, and the report guards refuse a whole
report over one such token, through a code span as readily as outside one.
Written out in full, a label carries no token of that shape at all.

Usage
-----
    python scripts/sweep_params.py --largest   # the largest cell alone, timed
    python scripts/sweep_params.py --gate --grid   # the gate, then the grid
    python scripts/sweep_params.py --read      # the reading rule, no fit
    python scripts/sweep_params.py             # --gate --grid --read
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import preregistered as pr  # noqa: E402
from taxiout.encoding import ENCODING_FEATURES  # noqa: E402
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER_BY_KEY,
    SUBMISSION_KEY,
    Booster,
    pool_rare,
)
from taxiout.validation import evaluate  # noqa: E402

OUT = REPO / "data" / "processed" / "sweep_october.csv"
WORK = REPO / "data" / "processed" / "sweep"

# The leg. ``preregistered.build_legs`` constructs it through
# ``taxiout.validation.forward_split`` with ``name="october"``: train
# 2025-01 to 2025-09, validate 2025-10, all ten airports.
LEG = "october"

# The rung, read from ``SUBMISSION_KEY`` and no longer pinned by name. The
# first sweep pinned ``LADDER_BY_KEY["saturation"]`` so that following
# ``TOP_RUNG`` could not silently re-run it on a different feature block and
# stop reproducing the October numbers recorded in ``taxiout.models``. That
# reason held while this file ran the first sweep and only that one. The
# second sweep is pre-registered on the submission rung, the numbers it
# reads are its own, and the first sweep's account is kept beside
# ``BOOSTER_PARAMS`` and under ``FIRST_SWEEP`` below rather than in a live
# constant that would have to be wrong for one of the two sweeps.
SWEEP_RUNG = LADDER_BY_KEY[SUBMISSION_KEY]

# Fixed in the pre-registration of 20 September 2026 and not reopened here.
# The chosen cell (0.03, 127, 50, 1,000) sat on the edge of the first grid on
# all four axes; here it is interior on three and the trees axis extends past
# it. Each new value is one step past the old edge at the old grid's own
# spacing, and the far values of the old grid (0.1, 31, 500) are dropped.
GRID = {
    "learning_rate": (0.015, 0.03, 0.06),
    "num_leaves": (63, 127, 255),
    "min_child_samples": (20, 50, 200),
}
TREES = (300, 600, 1000, 1500, 2000, 3000)

# The first sweep's terms, kept so that the account beside ``BOOSTER_PARAMS``
# can be read against what made it. Nothing below is run by this file any
# more; it is the record of a run that happened.
FIRST_SWEEP = {
    "rung": "saturation",
    "learning_rate": (0.03, 0.06, 0.1),
    "num_leaves": (31, 63, 127),
    "min_child_samples": (50, 200, 500),
    "trees": (300, 600, 1000),
    "fit_rows": "train & ~flagged, no four-hour exclusion",
    "encoding": "none",
    "chosen_cell_october_s": 235.0,
    "grid_october_range_s": (235.0, 244.8),
    "comparable_with_this_sweep": False,
}

# The cost drivers, measured before the grid is launched. Learning rate does
# not change how much work a tree is, so the largest cell is the widest tree
# at the smallest leaf floor, read to the end of the trees axis. Its learning
# rate is the current cell's, which makes it a member of the grid rather than
# a fit made only to be timed.
LARGEST_CELL = {"learning_rate": 0.03, "num_leaves": 255, "min_child_samples": 20}

# The current cell, whose figure the reading rule compares the best cell
# against. Read out of ``BOOSTER_PARAMS`` rather than written down, so that
# moving the parameters moves this with them.
CURRENT_CELL = {k: BOOSTER_PARAMS[k] for k in GRID}
CURRENT_TREES = int(BOOSTER_PARAMS["n_estimators"])

# The five single-seed fits the floor is measured from, and the fits the gate
# reads against the run that made them. The records are on disk; the printed
# figures below are the ones ``reports/preregistered-2026-09-20.md``'s
# Experiment 1 table carries, and the gate checks the stored doubles against
# these and against each other.
SEEDS = 5
BLOCK_07_RECORDS = REPO / "data" / "processed" / "preregistered"
BLOCK_07_OCTOBER_PRINTED = (217.28, 217.65, 217.43, 217.37, 217.43)
BLOCK_07_OCTOBER_FLOOR = 0.19
BLOCK_07_OCTOBER_ROWS = 185674
BLOCK_07_OCTOBER_FLAGGED = 115
BLOCK_07_OCTOBER_FIT_ROWS = 1570127


# --------------------------------------------------------------------------
# labels, records and the things this script may not write


def cell_label(values: dict, trees: int | None = None) -> str:
    """A cell written out in full, which carries no letter-and-digit token.

    ``learning_rate=0.015, num_leaves=255, min_child_samples=20`` and, where
    a reading is named, ``, n_estimators=3000``. Nothing here is three
    characters of letters and digits run together, so nothing here can equal
    a stand identifier, and the report guards have nothing to refuse. A short
    label would have to be checked against the stand table on every run; this
    one cannot collide in the first place.
    """
    parts = [f"{k}={values[k]}" for k in GRID]
    if trees is not None:
        parts.append(f"n_estimators={int(trees)}")
    return ", ".join(parts)


def cell_index(values: dict) -> int:
    return CELLS.index(tuple(values[k] for k in GRID))


CELLS = [tuple(v) for v in itertools.product(*GRID.values())]


def cell_values(cell: tuple) -> dict:
    return dict(zip(GRID.keys(), cell))


def record_path(stem: str) -> Path:
    return WORK / f"{stem}.json"


def digest(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def mask_digest(mask) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def booster_backend_is_not_lightgbm() -> bool:
    from taxiout.models import booster_backend

    return booster_backend() != "lightgbm"


# --------------------------------------------------------------------------
# reading a prefix of one fit


def predict_at(model: Booster, X: pd.DataFrame, trees: int) -> pd.Series:
    """``Booster.predict`` reading the first ``trees`` trees and no more.

    The coding is the wrapper's own, step for step: the rare callsign
    prefixes pooled to the levels the fit kept, then every categorical cast
    to the categories the fit saw, so that reading a prefix of the model
    reads the design matrix the wrapper would have built. ``num_iteration``
    is the only difference between this and ``Booster.predict``, and the
    tree-reading check is what proves that reading 1,000 trees of a
    3,000-tree fit gives the 1,000-tree fit of the same cell and seed.

    The first sweep's own reader omitted the pooling step. The saturation
    rung it ran on carried no pooled categorical, so nothing came of it; the
    submission rung carries ``cs_prefix``, so it would have here.
    """
    design = X[list(model.features)].copy()
    for c in model.pooled:
        design[c] = pool_rare(design[c].astype(str), model._kept.get(c, frozenset()))
    for c in model.categorical:
        levels = model._levels.get(c)
        if levels is not None:
            design[c] = pd.Categorical(design[c].astype(str), categories=levels)
        else:
            design[c] = design[c].astype("category")
    return pd.Series(
        model.model.predict(design, num_iteration=int(trees)),
        index=X.index,
        dtype=float,
    )


def pooled_ex_flagged(frame: pr.Frame, prep: dict, mask, pred) -> float:
    """The RMSE over the rows the record-quality term leaves to the model.

    ``taxiout.validation.evaluate`` scored exactly this way, which is how the
    ladder and the first sweep both read October.
    """
    ev = evaluate(
        frame.y[mask],
        pred,
        icao=frame.features.loc[mask, "icao"],
        flagged=prep["flagged"][mask],
    )
    return pr._pooled(ev)


# --------------------------------------------------------------------------
# the leg, prepared once and pinned so that two runs cannot differ


def leg_fingerprint(frame: pr.Frame, prep: dict) -> dict:
    """What makes this leg this leg, so a second process can be held to it.

    The three masks and the two encoded columns. Every figure this script
    reads is a function of these plus the booster's parameters, so two runs
    that agree here and disagree on a cell would have to disagree inside
    LightGBM, which is what ``deterministic`` and ``force_col_wise`` are set
    for.
    """
    mask = frame.sets(LEG)["valid"]
    encoded = prep["frame"].loc[mask, list(ENCODING_FEATURES)]
    return {
        "train_rows": int(prep["leg"].train.sum()),
        "valid_rows": int(mask.sum()),
        "fit_rows": int(prep["rows"].sum()),
        "flagged_valid_rows": int(prep["flagged"][mask].sum()),
        "fit_mask_sha256": mask_digest(prep["rows"]),
        "valid_mask_sha256": mask_digest(mask),
        "flagged_mask_sha256": mask_digest(prep["flagged"]),
        "encoded_columns_sha256": digest(encoded.to_numpy(dtype=float)),
        "columns": len(SWEEP_RUNG.features),
        "rung": SWEEP_RUNG.key,
    }


def pin_leg(frame: pr.Frame, prep: dict) -> dict:
    """Write the leg's fingerprint, or check it against the one on disk."""
    got = leg_fingerprint(frame, prep)
    path = record_path("leg")
    if path.exists():
        was = json.loads(path.read_text(encoding="utf-8"))
        if was != got:
            for key in sorted(set(was) | set(got)):
                if was.get(key) != got.get(key):
                    print(f"  {key}: {was.get(key)} -> {got.get(key)}", file=sys.stderr)
            raise SystemExit(
                "the leg is not the leg the earlier stage prepared; stop and report"
            )
        print("  leg: identical to the one the earlier stage pinned")
    else:
        write_json(path, got)
        print(
            f"  leg: pinned, {got['fit_rows']:,} fit rows, {got['valid_rows']:,} scored"
        )
    return got


def keys_frame(frame: pr.Frame, prep: dict) -> pd.DataFrame:
    """What the day-clustered interval needs, once, for every cell.

    The movement identifier is deliberately not here. The interval clusters
    on the take-off day and reads the target, the flag and the prediction,
    and nothing this script writes has any use for an identifier it must
    never print.
    """
    mask = frame.sets(LEG)["valid"]
    return pd.DataFrame(
        {
            "icao": frame.features.loc[mask, "icao"].astype(str).to_numpy(),
            "takeoff": pd.to_datetime(
                frame.dep.loc[mask, "MVT_TIME_UTC_mvt"], utc=True
            ).to_numpy(),
            "flagged": np.asarray(prep["flagged"][mask], dtype=bool),
            "y": np.asarray(frame.y[mask], dtype=float),
        }
    )


# --------------------------------------------------------------------------
# one cell


def fit_cell(frame: pr.Frame, values: dict, *, seed: int = 0) -> tuple[dict, dict]:
    """One 3,000-tree fit of the submission rung, read at every tree count.

    Everything but the three swept parameters and ``n_estimators`` is
    ``preregistered.fit_once``'s, line for line: the same rung, the same
    prepared frame, the same fitting rows, ``sample_weight`` left at ``None``
    so that no weight reaches the backend, and the same scoring.
    """
    prep = frame.prepare(LEG)
    params = dict(BOOSTER_PARAMS)
    params.update({k: values[k] for k in GRID})
    params["n_estimators"] = max(TREES)
    params["random_state"] = int(BOOSTER_PARAMS["random_state"]) + seed
    t0 = time.time()
    model = Booster(tuple(SWEEP_RUNG.features), params).fit(
        prep["frame"].loc[prep["rows"]], frame.y[prep["rows"]], sample_weight=None
    )
    fit_wall = time.time() - t0
    mask = frame.sets(LEG)["valid"]
    design = prep["frame"].loc[mask]
    readings: dict[str, float] = {}
    digests: dict[str, str] = {}
    predictions: dict[str, np.ndarray] = {}
    t1 = time.time()
    for trees in TREES:
        pred = predict_at(model, design, trees)
        readings[str(trees)] = pooled_ex_flagged(frame, prep, mask, pred)
        digests[str(trees)] = digest(pred)
        predictions[str(trees)] = np.asarray(pred, dtype=float)
    record = {
        "label": cell_label(values),
        "params": {k: values[k] for k in GRID},
        "seed": seed,
        "random_state": int(params["random_state"]),
        "n_estimators_fitted": int(params["n_estimators"]),
        "fit_rows": int(prep["rows"].sum()),
        "scored_rows": int(mask.sum()),
        "flagged_scored_rows": int(prep["flagged"][mask].sum()),
        "fit_wall_s": fit_wall,
        "score_wall_s": time.time() - t1,
        "peak_rss_kib": peak_rss_kib(),
        "rmse_oct_s": readings,
        "sha256": digests,
    }
    return record, predictions


# --------------------------------------------------------------------------
# the gate, before any cell is read


def block_07_october() -> list[dict]:
    """The five October single-seed records the earlier run left on disk."""
    got = []
    for seed in range(SEEDS):
        path = BLOCK_07_RECORDS / f"{LEG}_seed{seed}.json"
        if not path.exists():
            raise SystemExit(f"the gate needs {path}, which is not there")
        got.append(json.loads(path.read_text(encoding="utf-8")))
    return got


def gate(frame: pr.Frame) -> dict:
    """The five singles, the floor, and the tree-reading check.

    A gate and not an observation. The parameters and the fit rules are
    unchanged since the run that made the records on disk, so the five fits
    here must reproduce that run's, and they are compared against the stored
    doubles rather than against the two decimals a report prints. The
    prediction digests are compared as well, because this harness scores the
    identical row set in the identical order: the same leg construction, the
    same mask, the same frame.
    """
    stored = block_07_october()
    rows = []
    kept: dict[int, np.ndarray] = {}
    for seed in range(SEEDS):
        record, frames = pr.fit_once(
            frame, LEG, seed=seed, schedule="none", features="submission"
        )
        here = record["sets"]["valid"]
        there = stored[seed]["sets"]["valid"]
        same_shape = (
            here["rows"] == there["rows"]
            and here["flagged"] == there["flagged"]
            and record["fit_rows"] == stored[seed]["fit_rows"]
        )
        rows.append(
            {
                "seed": seed,
                "expected_s": there["pooled_ex_flagged"],
                "found_s": here["pooled_ex_flagged"],
                "expected_printed_s": BLOCK_07_OCTOBER_PRINTED[seed],
                "expected_sha256": there["sha256"],
                "found_sha256": here["sha256"],
                "rows": here["rows"],
                "flagged": here["flagged"],
                "fit_rows": record["fit_rows"],
                "same_row_set": bool(same_shape),
                "equal_double": here["pooled_ex_flagged"] == there["pooled_ex_flagged"],
                "equal_digest": here["sha256"] == there["sha256"],
                "wall_s": record["wall_s"],
            }
        )
        kept[seed] = np.asarray(frames["valid"]["pred"], dtype=float)
        print(
            f"  seed +{seed}: expected {there['pooled_ex_flagged']:.6f} s, "
            f"found {here['pooled_ex_flagged']:.6f} s, "
            f"digest {'equal' if rows[-1]['equal_digest'] else 'DIFFERENT'} "
            f"({record['wall_s']:.0f} s)"
        )
    singles = [r["found_s"] for r in rows]
    floor = pr.floor_of(singles)
    print(f"  floor: {floor:.6f} s, expected {BLOCK_07_OCTOBER_FLOOR} s printed")
    return {
        "singles": rows,
        "floor_s": floor,
        "floor_printed_s": round(floor, 2),
        "floor_expected_printed_s": BLOCK_07_OCTOBER_FLOOR,
        "predictions": kept,
    }


def tree_reading_check(gate_result: dict, cell: dict) -> dict:
    """The current cell's 3,000-tree fit, read at 1,000, against seed + 0.

    A check and not a criterion: it changes no reading rule and no grid
    value. What it protects is every figure the grid produces, because the
    grid reads six tree counts out of one fit on the claim that the first
    ``k`` trees of a boosted model are the same model whatever the total. If
    ``n_estimators`` changed the fit, the claim would be false and the grid
    would be reading models nobody fitted.
    """
    seed0 = next(r for r in gate_result["singles"] if r["seed"] == 0)
    got = {
        "expected_s": seed0["found_s"],
        "found_s": cell["rmse_oct_s"][str(CURRENT_TREES)],
        "expected_sha256": seed0["found_sha256"],
        "found_sha256": cell["sha256"][str(CURRENT_TREES)],
    }
    got["equal_double"] = got["expected_s"] == got["found_s"]
    got["equal_digest"] = got["expected_sha256"] == got["found_sha256"]
    print(
        f"  tree reading: {CURRENT_TREES} trees of a {max(TREES)}-tree fit gives "
        f"{got['found_s']:.6f} s against the {CURRENT_TREES}-tree fit's "
        f"{got['expected_s']:.6f} s, digest "
        f"{'equal' if got['equal_digest'] else 'DIFFERENT'}"
    )
    return got


def gate_verdict(gate_result: dict, tree_check: dict | None) -> list[str]:
    """Everything that disagreed with its expectation, named."""
    bad = []
    for row in gate_result["singles"]:
        if not row["same_row_set"]:
            bad.append(f"seed +{row['seed']}: the row set is not the stored run's")
        if not row["equal_double"]:
            bad.append(
                f"seed +{row['seed']}: {row['found_s']!r} against the stored "
                f"{row['expected_s']!r}"
            )
        if row["same_row_set"] and not row["equal_digest"]:
            bad.append(f"seed +{row['seed']}: the prediction digest differs")
    if round(gate_result["floor_s"], 2) != gate_result["floor_expected_printed_s"]:
        bad.append(
            f"the floor is {gate_result['floor_s']:.4f} s against the expected "
            f"{gate_result['floor_expected_printed_s']} s"
        )
    if tree_check is not None and not tree_check["equal_digest"]:
        bad.append(
            "the tree-reading check: reading "
            f"{CURRENT_TREES} trees of a {max(TREES)}-tree fit does not give the "
            f"{CURRENT_TREES}-tree fit"
        )
    return bad


# --------------------------------------------------------------------------
# the reading rule, fixed in the pre-registration and not reopened


EDGES = {
    "learning_rate": (min(GRID["learning_rate"]), max(GRID["learning_rate"])),
    "num_leaves": (min(GRID["num_leaves"]), max(GRID["num_leaves"])),
    "min_child_samples": (min(GRID["min_child_samples"]), max(GRID["min_child_samples"])),
    "n_estimators": (min(TREES), max(TREES)),
}


def as_cell(row: dict) -> dict:
    """One row of the grid with its axes back in the types the grid holds.

    A row taken out of a frame with ``iloc`` is a Series of one dtype, so the
    leaf count and the minimum child samples come back as floats. They go into
    ``BOOSTER_PARAMS`` and into a printed label, and 255.0 leaves is not a
    value this grid holds.
    """
    got = dict(row)
    for axis, values in GRID.items():
        got[axis] = type(values[0])(got[axis])
    got["n_estimators"] = int(got["n_estimators"])
    got["rmse_oct_s"] = float(got["rmse_oct_s"])
    return got


def edges_of(row: dict) -> list[str]:
    """The axes a cell sits on an edge of, named, or an empty list."""
    return [axis for axis, ends in EDGES.items() if row[axis] in ends]


def read_grid() -> pd.DataFrame:
    """The 162 rows, one per cell and tree count, from the records on disk."""
    rows = []
    for index in range(len(CELLS)):
        path = record_path(f"cell_{index:02d}")
        if not path.exists():
            raise SystemExit(f"{path} is not there; the grid is not complete")
        record = json.loads(path.read_text(encoding="utf-8"))
        for trees in TREES:
            rows.append(
                {
                    **record["params"],
                    "n_estimators": trees,
                    "rmse_oct_s": record["rmse_oct_s"][str(trees)],
                }
            )
    return pd.DataFrame(rows)


def apply_reading_rule(grid: pd.DataFrame, floor: float) -> dict:
    """The pre-registered rule, and nothing else.

    "The best cell replaces ``BOOSTER_PARAMS`` only if its October figure is
    below the current cell's figure from the same run by more than the floor;
    a difference inside the floor keeps the current cell. If the best cell
    lies on an edge of this grid, that is recorded and the grid is not
    extended again. Nothing is chosen on December, July or the held-out
    pairs."

    The current cell's figure is this run's, read from the grid, and not the
    figure any earlier run printed. Equal to the floor is not inside it, the
    same reading ``clears_floor`` uses.
    """
    want = grid
    for axis, value in CURRENT_CELL.items():
        want = want[want[axis] == value]
    want = want[want["n_estimators"] == CURRENT_TREES]
    if len(want) != 1:
        raise SystemExit("the current cell is not exactly one row of the grid")
    current = float(want["rmse_oct_s"].iloc[0])
    ordered = grid.sort_values("rmse_oct_s", kind="mergesort").reset_index(drop=True)
    best = as_cell(ordered.iloc[0].to_dict())
    moves = pr.clears_floor(float(best["rmse_oct_s"]), current, floor)
    chosen = (
        best
        if moves
        else {**CURRENT_CELL, "n_estimators": CURRENT_TREES, "rmse_oct_s": current}
    )
    return {
        "current": {**CURRENT_CELL, "n_estimators": CURRENT_TREES, "rmse_oct_s": current},
        "current_label": cell_label(CURRENT_CELL, CURRENT_TREES),
        "best": best,
        "best_label": cell_label(best, int(best["n_estimators"])),
        "difference_s": float(best["rmse_oct_s"]) - current,
        "floor_s": floor,
        "clears_floor": bool(moves),
        "booster_params_moves": bool(moves),
        "chosen": chosen,
        "chosen_label": cell_label(chosen, int(chosen["n_estimators"])),
        "best_on_edges": edges_of(best),
        "chosen_on_edges": edges_of(chosen),
        "top_eight": [as_cell(r) for r in ordered.head(8).to_dict("records")],
        "rows": int(len(grid)),
        "grid_range_s": (
            float(ordered["rmse_oct_s"].iloc[0]),
            float(ordered["rmse_oct_s"].iloc[-1]),
        ),
    }


def chosen_interval(verdict: dict) -> dict | None:
    """The day-clustered interval on the chosen cell against the current one.

    Printed, and it decides nothing: the reading rule is floor-only and stays
    floor-only. It is here because a cell can be chosen whose interval spans
    zero, and the paper should not meet that for the first time in the
    documentation pass. If the chosen cell is the current cell there is no
    contrast and nothing is printed.
    """
    if not verdict["booster_params_moves"]:
        return None
    keys = pd.read_parquet(WORK / "keys.parquet")
    arm = keys.copy()
    base = keys.copy()
    chosen = verdict["chosen"]
    arm["pred"] = _prediction(chosen, int(chosen["n_estimators"]))
    base["pred"] = _prediction(CURRENT_CELL, CURRENT_TREES)
    got = pr.paired_interval(arm, base)
    if got is None:
        return None
    return {"point": got.point, "low": got.low, "high": got.high, "draws": got.draws}


def _prediction(values: dict, trees: int) -> np.ndarray:
    index = cell_index({k: values[k] for k in GRID})
    frame = pd.read_parquet(WORK / f"cell_{index:02d}__pred.parquet")
    return frame[str(int(trees))].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# the stages


def build() -> tuple[pr.Frame, dict]:
    frame = pr.Frame()
    prep = frame.prepare(LEG)
    pin_leg(frame, prep)
    path = WORK / "keys.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        keys_frame(frame, prep).to_parquet(path, index=False)
    return frame, prep


def store_cell(
    index: int, record: dict, predictions: dict, stem: str | None = None
) -> None:
    stem = stem or f"cell_{index:02d}"
    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_parquet(WORK / f"{stem}__pred.parquet", index=False)
    write_json(record_path(stem), record)


def run_largest(frame: pr.Frame) -> dict:
    """The largest cell, alone, so the grid's cost is measured and not guessed.

    Learning rate does not drive the cost; the leaf count, the minimum child
    samples and the number of trees do. Twenty-seven times this cell's wall
    is an upper bound on the grid, because every other cell is cheaper on at
    least one of those three.
    """
    print(f"the largest cell: {cell_label(LARGEST_CELL, max(TREES))}")
    record, predictions = fit_cell(frame, LARGEST_CELL, seed=0)
    store_cell(cell_index(LARGEST_CELL), record, predictions, stem="largest_timed")
    total = record["fit_wall_s"] + record["score_wall_s"]
    print(
        f"  fitted in {record['fit_wall_s']:.0f} s, scored at six tree counts in "
        f"{record['score_wall_s']:.0f} s, peak {record['peak_rss_kib'] / 1048576:.1f} GiB"
    )
    print(
        f"  upper bound on {len(CELLS)} cells: {len(CELLS) * total:.0f} s "
        f"({len(CELLS) * total / 3600:.1f} h), plus one frame build"
    )
    for trees in TREES:
        print(f"    {trees:5d} trees: {record['rmse_oct_s'][str(trees)]:.4f} s")
    return record


def run_gate(frame: pr.Frame) -> dict:
    print("the gate: five single-seed fits of the current cell")
    got = gate(frame)
    return got


def run_grid(frame: pr.Frame, floor: float) -> list[dict]:
    """The twenty-seven cells, in grid order, each written as it finishes."""
    made = []
    for index, cell in enumerate(CELLS):
        values = cell_values(cell)
        path = record_path(f"cell_{index:02d}")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            print(f"  [{index + 1:2d}/{len(CELLS)}] {record['label']}: already on disk")
            made.append(record)
            continue
        record, predictions = fit_cell(frame, values, seed=0)
        store_cell(index, record, predictions)
        made.append(record)
        print(
            f"  [{index + 1:2d}/{len(CELLS)}] {record['label']} -> "
            + ", ".join(f"{t}: {record['rmse_oct_s'][str(t)]:.2f}" for t in TREES)
            + f"  ({record['fit_wall_s']:.0f} s fit, "
            f"{record['score_wall_s']:.0f} s score)"
        )
    return made


def cross_process_check() -> dict | None:
    """The timed fit of the largest cell against the grid's own fit of it.

    The timing fit is made in its own process, so that ``/usr/bin/time -v``
    measures that cell and nothing else. The grid then fits it again with the
    other twenty-six, and the two must agree to the bit. It is a free reading
    of the same determinism the booster's own parameters are set for, across
    two processes rather than two fits.
    """
    timed = record_path("largest_timed")
    if not timed.exists():
        return None
    was = json.loads(timed.read_text(encoding="utf-8"))
    now = json.loads(record_path(f"cell_{cell_index(LARGEST_CELL):02d}").read_text())
    return {
        "label": was["label"],
        "equal_readings": was["rmse_oct_s"] == now["rmse_oct_s"],
        "equal_digests": was["sha256"] == now["sha256"],
        "timed_wall_s": was["fit_wall_s"] + was["score_wall_s"],
        "grid_wall_s": now["fit_wall_s"] + now["score_wall_s"],
    }


def run_read(floor: float | None = None) -> dict:
    """The CSV, the reading rule and the interval, with no fit."""
    grid = read_grid()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.sort_values("rmse_oct_s", kind="mergesort").to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}: {len(grid)} rows")
    if floor is None:
        floor = json.loads(record_path("gate").read_text(encoding="utf-8"))["floor_s"]
    verdict = apply_reading_rule(grid, floor)
    print()
    print(grid.sort_values("rmse_oct_s", kind="mergesort").head(8).to_string(index=False))
    print()
    print(
        f"current: {verdict['current_label']} -> {verdict['current']['rmse_oct_s']:.4f} s"
    )
    print(f"best:    {verdict['best_label']} -> {verdict['best']['rmse_oct_s']:.4f} s")
    print(
        f"difference {verdict['difference_s']:+.4f} s against a floor of {floor:.4f} s: "
        + (
            "BOOSTER_PARAMS moves"
            if verdict["booster_params_moves"]
            else "the current cell stands"
        )
    )
    edges = verdict["best_on_edges"]
    print(
        "the best cell is on the edge of: " + (", ".join(edges) if edges else "no axis")
    )
    interval = chosen_interval(verdict)
    verdict["interval"] = interval
    if interval is not None:
        print(
            f"the day-clustered interval, chosen against current: "
            f"{interval['point']:+.2f} s [{interval['low']:+.2f}, {interval['high']:+.2f}]. "
            "It decides nothing."
        )
    else:
        print("no interval: the chosen cell is the current cell.")
    verdict["cross_process"] = cross_process_check()
    write_json(
        record_path("verdict"),
        {k: v for k, v in verdict.items() if k != "top_eight"}
        | {"top_eight": verdict["top_eight"]},
    )
    return verdict


# --------------------------------------------------------------------------
# entry point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--largest",
        action="store_true",
        help="fit the largest cell alone and print the bound it gives on the grid",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="the five single-seed fits, the floor and the tree-reading check",
    )
    parser.add_argument("--grid", action="store_true", help="the twenty-seven cells")
    parser.add_argument(
        "--read", action="store_true", help="the CSV and the reading rule, with no fit"
    )
    args = parser.parse_args(argv)
    stages = (args.largest, args.gate, args.grid, args.read)
    if not any(stages):
        args.gate = args.grid = args.read = True

    if (args.largest or args.gate or args.grid) and booster_backend_is_not_lightgbm():
        print("the sweep reads trees from one fit and needs lightgbm", file=sys.stderr)
        return 1

    # What this script may not write, checked rather than promised, as
    # scripts/fast_lane.py and scripts/preregistered.py check it. The sweep
    # writes under data/processed/ and nowhere else; a cell that quietly
    # overwrote a stored residual or a report would put an unreported fit
    # where a reported one belongs.
    before = pr.fingerprint()

    floor = None
    gate_result = None
    if args.largest or args.gate or args.grid:
        frame, _ = build()
        if args.largest:
            run_largest(frame)
        if args.gate:
            gate_result = run_gate(frame)
            index = cell_index(CURRENT_CELL)
            path = record_path(f"cell_{index:02d}")
            if path.exists():
                cell = json.loads(path.read_text(encoding="utf-8"))
            else:
                cell, predictions = fit_cell(frame, CURRENT_CELL, seed=0)
                store_cell(index, cell, predictions)
            check = tree_reading_check(gate_result, cell)
            bad = gate_verdict(gate_result, check)
            write_json(
                record_path("gate"),
                {
                    "singles": gate_result["singles"],
                    "floor_s": gate_result["floor_s"],
                    "floor_printed_s": gate_result["floor_printed_s"],
                    "floor_expected_printed_s": gate_result["floor_expected_printed_s"],
                    "tree_reading": check,
                    "disagreed": bad,
                },
            )
            floor = gate_result["floor_s"]
            if bad:
                for line in bad:
                    print(f"  DISAGREED: {line}", file=sys.stderr)
                print(
                    "the gate disagreed with its expectation; no cell is read",
                    file=sys.stderr,
                )
                return 1
            print("  the gate holds; the grid may be read")
        if args.grid:
            if not record_path("gate").exists():
                print("the grid does not run before the gate", file=sys.stderr)
                return 1
            run_grid(frame, floor or 0.0)

    touched = pr.touched(before, pr.fingerprint())
    if touched:
        for name in touched:
            print(f"  TOUCHED: {name}", file=sys.stderr)
        print("this run wrote where it may not write", file=sys.stderr)
        return 1

    if args.read:
        run_read(floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
