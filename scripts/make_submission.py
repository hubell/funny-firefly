#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fit the model on the training year and write a submission file.

One model, the submission rung of the ladder in ``taxiout.models``: every block
of the feature frame except the runway-saturation one -- the identifiers, the
take-off clock, the plan's own delay signal and its lags, the distance-matrix
block with its join-quality flags, the aircraft, the congestion blocks and the
shared corridor, and the weather regimes. Saturation was built and measured and
bought nothing, and ``taxiout.models`` says so where it names the rung. Fitted
on the full 2025 movement files with the rows ``taxiout.tails`` covers excluded,
then combined with that term on the rows it covers. Nothing here is chosen by a
leaderboard score; the rung and the hyperparameters are the ones
``reports/baseline.md`` was generated with.

The file goes to ``data/submissions/<team>_v<n>.parquet``, and the script
checks with git that the path is ignored before writing a byte. Predictions
built from the challenge data are not redistributable and neither is the
frame that produced them; ``.gitignore`` already excludes every parquet file
anywhere in the tree, and this verifies rather than assumes it.

Validation, the template's schema and the read-back check are all
``taxiout.submission``'s -- a submission is rejected outright if any row of
the template is missing or extra, so none of that is reimplemented here.

Needs the challenge data under ``data/challenge/``, the distance matrices and
edge sets under ``data/processed/taxi_distance/`` (``make reports``), the METAR
cache under ``data/raw/metar/`` (``make metar``), which the weather block reads
and without which this refuses to write, and the modelling extras. Takes a few
minutes.

Usage
-----
    python scripts/make_submission.py --team myteam
    python scripts/make_submission.py --team myteam --version 3
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.corridor import load_edges  # noqa: E402
from taxiout.metar import load_metar, regime_by_hour  # noqa: E402
from taxiout.challenge import (  # noqa: E402
    load_movements,
    load_ranking,
    load_submitting,
    to_pandas,
)
from taxiout.features import (  # noqa: E402
    MOVEMENT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET,
    airport_fallback,
    build_features,
)
from taxiout.models import TOP_RUNG, Booster, booster_backend  # noqa: E402

DATA = REPO / "data" / "challenge"
MATRICES = REPO / "data" / "processed" / "taxi_distance"
METAR = REPO / "data" / "raw" / "metar"
SUBMISSIONS = REPO / "data" / "submissions"

ID = "MVT_ID_mvt"

# The rung the submission uses. Not the last rung of the ladder: the
# saturation rung sits above it as a measured negative and is deliberately
# out of the model. ``taxiout.models`` names which rung this is and why.
RUNG = TOP_RUNG

# Read by the record-quality term and never entered as a feature. The block
# time is one of the fields the ranking set hides and is loaded for the
# training side only, which is what fitting on training rows means.
TAILS_TRAIN_COLUMNS = ("SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "ADEP_mvt")
TAILS_PREDICT_COLUMNS = ("SCHED_TIME_UTC_mvt", "ADEP_mvt", "LOBT_flt", "MVT_TIME_UTC_mvt")

# What the two Stream B modules must export for this script to run.
NEEDED = {
    "taxiout.submission": ("write", "check_filename"),
    "taxiout.tails": ("TailModel", "combine"),
}


def stream_b():
    """``(submission, tails)`` modules, or ``(None, reason)`` if not there yet."""
    found = {}
    for name, exports in NEEDED.items():
        try:
            module = __import__(name, fromlist=["_"])
        except ImportError:
            return None, f"`{name}` is not in the tree yet (Stream B)"
        missing = [e for e in exports if not hasattr(module, e)]
        if missing:
            return None, (
                f"`{name}` exists but exports none of "
                + ", ".join(f"`{e}`" for e in missing)
            )
        found[name] = module
    return (found["taxiout.submission"], found["taxiout.tails"]), ""


def load_matrix() -> pd.DataFrame:
    files = [
        f
        for f in sorted(glob.glob(str(MATRICES / "*.csv")))
        if "_features" not in f and "_edges" not in f
    ]
    if not files:
        raise FileNotFoundError(f"no distance matrices under {MATRICES}; make reports")
    return pd.concat(
        [pd.read_csv(f, dtype={"stand_id": str, "designator": str}) for f in files],
        ignore_index=True,
    )


def load_edge_sets() -> pd.DataFrame:
    """The per-cell taxiway edge sets written beside the matrices."""
    files = sorted(glob.glob(str(MATRICES / "*_edges.csv")))
    if not files:
        raise FileNotFoundError(f"no edge sets under {MATRICES}; make reports")
    return load_edges(files)


def is_ignored(path: Path) -> bool:
    """Whether git would ignore ``path``. Asked, not assumed."""
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=REPO, capture_output=True
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0


def next_version(team: str) -> int:
    """One past the highest version already written for ``team``."""
    used = []
    for p in SUBMISSIONS.glob(f"{team}_v*.parquet"):
        tail = p.stem.rsplit("_v", 1)
        if len(tail) == 2 and tail[0] == team and tail[1].isdigit():
            used.append(int(tail[1]))
    return max(used, default=0) + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", required=True, help="team name, as the organisers hold it")
    ap.add_argument(
        "--version",
        type=int,
        default=None,
        help="submission version; defaults to one past the highest written",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing file of that version",
    )
    args = ap.parse_args()

    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to submit", file=sys.stderr)
        return 1
    if booster_backend() == "none":
        print('no booster installed; pip install -e ".[modelling]"', file=sys.stderr)
        return 1
    modules, reason = stream_b()
    if modules is None:
        print(f"WARNING: {reason}; cannot write a submission yet", file=sys.stderr)
        return 1
    submission, tails = modules

    version = args.version or next_version(args.team)
    path = SUBMISSIONS / f"{args.team}_v{version}.parquet"
    submission.check_filename(path)
    if not is_ignored(path):
        print(
            f"refusing to write {path.relative_to(REPO)}: git does not ignore it, and "
            "a submission is built from data that may not be redistributed",
            file=sys.stderr,
        )
        return 1

    if not METAR.exists() or not any(METAR.glob("*.csv")):
        print(
            "no METAR cache under data/raw/metar; the submission rung reads it. "
            "Run: python scripts/fetch_metar.py",
            file=sys.stderr,
        )
        return 1

    print("loading ...")
    matrix = load_matrix()
    fallback = airport_fallback(matrix)
    weather = regime_by_hour(load_metar(METAR))
    train_columns = sorted(set(REQUIRED_COLUMNS) | set(TAILS_TRAIN_COLUMNS) | {TARGET})
    rank_columns = sorted(set(REQUIRED_COLUMNS) | set(TAILS_PREDICT_COLUMNS) | {ID})
    train = to_pandas(load_movements(DATA, train_columns, phase="DEP")).reset_index(
        drop=True
    )
    rank = to_pandas(load_ranking(DATA, rank_columns, phase="DEP")).reset_index(drop=True)
    template = to_pandas(load_submitting(DATA))
    # The congestion block counts the movements around each subject, so each
    # side gets its own movement frame: the training year's for the training
    # departures, the ranking months' for the ranking departures. Nothing
    # from one side is ever counted on the other.
    train_moves = to_pandas(load_movements(DATA, list(MOVEMENT_COLUMNS))).reset_index(
        drop=True
    )
    rank_moves = to_pandas(load_ranking(DATA, list(MOVEMENT_COLUMNS))).reset_index(
        drop=True
    )
    print(f"  {len(train):,} training departures, {len(rank):,} to predict")

    print("fitting the record-quality term ...")
    tail_model = tails.TailModel().fit(train)
    covered = tail_model.flagged(train)
    print(
        f"  {int(tail_model.cells['covered'].sum()):,} covered cells, "
        f"{int(covered.sum()):,} training rows excluded from the physical fit"
    )

    print(f"fitting {RUNG.label} ...")
    t0 = time.time()
    edges = load_edge_sets()
    X = build_features(
        train,
        matrix,
        fallback=fallback,
        movements=train_moves,
        edges=edges,
        weather=weather,
    )
    del train_moves
    y = pd.to_numeric(train[TARGET], errors="coerce")
    fit_rows = (~covered) & y.notna()
    model = Booster(RUNG.features).fit(X.loc[fit_rows], y[fit_rows])
    print(f"  fitted on {int(fit_rows.sum()):,} rows in {time.time() - t0:.0f} s")

    print("predicting ...")
    Xr = build_features(
        rank,
        matrix,
        fallback=fallback,
        movements=rank_moves,
        edges=edges,
        weather=weather,
    )
    physical = model.predict(Xr)
    tail = tail_model.predict(rank)
    pred = tails.combine(physical, tail)
    # The submission's target is a non-negative int32; a boosted model can
    # extrapolate below zero on a row unlike anything it was fitted on, and a
    # negative taxi-out is not a prediction anyone should submit.
    negative = int((pred < 0).sum())
    pred = pred.clip(lower=0)
    print(
        f"  {int(tail.notna().sum()):,} rows take the record-quality term; "
        f"{negative:,} negative predictions clipped to zero"
    )
    print(
        f"  mean {pred.mean():,.0f} s, median {pred.median():,.0f} s, "
        f"max {pred.max():,.0f} s"
    )

    out = pd.DataFrame({ID: rank[ID].to_numpy(), TARGET: np.rint(pred.to_numpy())})
    written = submission.write(out, template, path, overwrite=args.overwrite)
    print(f"wrote {written.relative_to(REPO)} ({written.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
