#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fit the model on the training year and write a submission file.

One model, the submission rung of the ladder in ``taxiout.models``: every block
of the feature frame except the runway-saturation one -- the airport and runway
identifiers, the take-off clock, the plan's own delay signal, its lags and their
differences, the distance-matrix block with its join-quality flags, the
aircraft, the congestion blocks and the shared corridor, the weather regimes,
the shrunk identifier encoding in place of the stand categorical, and the four
Gate 2c blocks: stand occupancy, the callsign prefix, the destination geometry
and the holiday calendar. Saturation was built and measured and bought nothing,
and ``taxiout.models`` says so where it names the rung. Fitted on the full 2025
movement files with the rows ``taxiout.tails`` covers excluded, then combined
with that term on the rows it covers. Nothing here is chosen by a leaderboard
score; the rung and the hyperparameters are the ones ``reports/baseline.md`` was
generated with.

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
cache under ``data/raw/metar/`` (``make metar``) and the OurAirports table under
``data/raw/`` (``make ourairports``), which the weather and destination blocks
read and without either of which this refuses to write, and the modelling
extras. Takes a few minutes.

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
from taxiout.destination import load_airports  # noqa: E402
from taxiout.metar import load_metar, regime_by_hour  # noqa: E402
from taxiout.challenge import (  # noqa: E402
    load_movements,
    load_ranking,
    load_submitting,
    to_pandas,
)
from taxiout.encoding import (  # noqa: E402
    ENCODING_FEATURES,
    encode_split,
    fit_encoding,
    group_key,
    with_encoding,
)
from taxiout.features import (  # noqa: E402
    MOVEMENT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET,
    airport_fallback,
    build_features,
)
from taxiout.models import (  # noqa: E402
    BAGGING_SEEDS,
    TOP_RUNG,
    Booster,
    SEED_AVERAGE_ADOPTED,
    SeedAverage,
    booster_backend,
    fit_rows as fitting_rows,
)

DATA = REPO / "data" / "challenge"
MATRICES = REPO / "data" / "processed" / "taxi_distance"
METAR = REPO / "data" / "raw" / "metar"
OURAIRPORTS = REPO / "data" / "raw" / "ourairports_airports.csv"
SUBMISSIONS = REPO / "data" / "submissions"

ID = "MVT_ID_mvt"

# The rung the submission uses. Not the last rung of the ladder: the
# saturation rung sits above it as a measured negative and is deliberately
# out of the model. ``taxiout.models`` names which rung this is and why.
#
# If that rung reads the shrunk encoding, this script computes it -- once over
# the training year, out of fold for the rows it fits on and by lookup for the
# rows it predicts. The refusal below still stands for any block this script
# cannot build; the encoding is simply no longer one of them.
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
    if not OURAIRPORTS.exists():
        print(
            "no OurAirports cache under data/raw; the submission rung reads the "
            "destination geometry. Run: python scripts/fetch_ourairports.py",
            file=sys.stderr,
        )
        return 1

    print("loading ...")
    matrix = load_matrix()
    fallback = airport_fallback(matrix)
    weather = regime_by_hour(load_metar(METAR))
    airports = load_airports(OURAIRPORTS)
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
        airports=airports,
    )
    del train_moves
    y = pd.to_numeric(train[TARGET], errors="coerce")
    # Guard 3, at submission. The record-quality rows are out of everything the
    # encoding reads, for the same reason they are out of the physical fit: a
    # defaulted block time is not a taxi time, and one of them inside a thin
    # pair would otherwise *be* that pair's encoding in the shipped model.
    # ...and the recorded-taxi tail, by the same rule and the same constant the
    # ladder fits under (``taxiout.models.fit_rows``). If these two came apart,
    # the model that ships would not be the model reports/baseline.md measures.
    fit_rows = fitting_rows(y.notna(), covered, y)

    # The encoding, if the rung reads it. Computed once over the whole training
    # year and in two forms, because the two sides of this script are asking
    # different questions of it.
    #
    #   the training rows are encoded OUT OF FOLD, by ``encode_split``. Their
    #   targets are in the fitting sample, so a row's own target must not reach
    #   its own feature; fitting on in-sample encodings would teach the booster
    #   to trust a column that is sharper at fit than at use.
    #
    #   the ranking rows are encoded by LOOKUP, from the whole sample. Their
    #   targets are not in the sample at all, so there is no fold to be out of
    #   -- which is what ``taxiout.encoding`` means by the ranking set carrying
    #   no fold -- and a pair the training year never saw arrives with n = 0
    #   and encodes to the bare geometric prior.
    #
    # Both share one fitted prior, passed explicitly, so the two sides cannot
    # drift apart through two separate fits of the same quantity.
    encoder = None
    encoded_train = None
    if set(ENCODING_FEATURES) & set(RUNG.features):
        t_enc = time.time()
        encoder = fit_encoding(X, y, fit_rows)
        encoded_train = encode_split(X, y, fit_rows, prior=encoder.prior)
        X = with_encoding(X, encoded_train)
        print(
            f"  encoding fitted on {encoder.fit_rows:,} unflagged rows "
            f"({time.time() - t_enc:.0f} s)"
        )

    # A rung whose block this script cannot build must not be fitted on a
    # column of nothing. ``build_features`` leaves a block missing rather than
    # zero when its input was not supplied, and a booster given an all-null
    # column will fit happily and predict from the rest, producing a submission
    # that silently is not the rung it claims to be.
    empty = [c for c in RUNG.features if X[c].isna().all()]
    if empty:
        print(
            f"refusing to fit {RUNG.label}: this script cannot build "
            + ", ".join(f"`{c}`" for c in empty)
            + "; the column arrives missing and the rung would be fitted without it",
            file=sys.stderr,
        )
        return 1
    # Seed averaging, if the rule in ``reports/baseline.md``'s *Seed averaging*
    # adopted it. It is an estimator choice and not a rung, so the ladder stays
    # single-seed and this is the one place it would apply -- and whether it
    # applies is ``taxiout.models.SEED_AVERAGE_ADOPTED`` rather than a decision
    # taken here, so that this script cannot ship a model the report says was
    # measured and not adopted. It is currently False: once the recorded-taxi
    # tail came out of every fit, averaging five seeds gained 0.86 s and 0.54 s
    # on the two forward splits, with the booster made repeatable, against a
    # one-second threshold fixed before any of it ran.
    if SEED_AVERAGE_ADOPTED:
        model = SeedAverage(RUNG.features).fit(X.loc[fit_rows], y[fit_rows])
        how = f"under {BAGGING_SEEDS} seeds"
    else:
        model = Booster(RUNG.features).fit(X.loc[fit_rows], y[fit_rows])
        how = "under a single seed (seed averaging measured, not adopted)"
    print(f"  fitted on {int(fit_rows.sum()):,} rows {how} in {time.time() - t0:.0f} s")

    print("predicting ...")
    Xr = build_features(
        rank,
        matrix,
        fallback=fallback,
        movements=rank_moves,
        edges=edges,
        weather=weather,
        airports=airports,
    )
    if encoder is not None:
        columns = encoder.transform(Xr)
        Xr = with_encoding(Xr, columns)
        seen = pd.Series(
            group_key(Xr, "pair").map(encoder.moments["pair"]["n"]).to_numpy(),
            index=Xr.index,
        ).fillna(0.0)
        print(
            f"  {int((seen <= 0).sum()):,} of {len(Xr):,} ranking rows sit on a pair "
            "the training year never saw and encode to the geometric prior"
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
