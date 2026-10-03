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
and the holiday calendar. Saturation was built and measured against the ceiling
in the week of 3 September, when it bought nothing. Under the parameters that
ship, ``reports/baseline.md`` reads it better than the rung it was added to by
more than the seed floor on both forward splits, and it has never been measured
on top of the rung this script fits, so what it would add here is not known and
it is not in the model. Fitted on the full 2025
movement files with the rows ``taxiout.tails`` covers excluded, then combined
with that term on the rows it covers, and with the rule of the plan-shifted
departures (``taxiout.tails.PlanShiftModel``, fixed by
``reports/preregistered-2026-10-02.md``) on the rows of that class the term
leaves. Nothing here is chosen by a leaderboard score; the rung and the
hyperparameters are the ones ``reports/baseline.md`` was generated with.

The file goes to ``data/submissions/<team>_v<n>.parquet``, with a manifest
beside it at ``<team>_v<n>.manifest.json``, and the script checks with git
that both paths are ignored before writing a byte. Predictions built from the
challenge data are not redistributable and neither is the frame that produced
them; ``.gitignore`` excludes every parquet file anywhere in the tree and the
whole of ``data/submissions/``, and this verifies rather than assumes it.

The manifest says what the file was built from and by: the git revision and
whether the tree was clean, the booster, its version and the parameters it was
fitted with, how many rows it was fitted on, how many ranking rows took the
record-quality term, how many took the plan-shifted departures' rule and what
that rule was fitted to, and how many sit on a pair the training year never saw,
the sha256 of the template read and of the file written, and the time. Two
fields are left empty for the only values the scorer is ever read for,
``usedPairs`` and the score. Two builds of one version should differ in the
time alone.

Only LightGBM is fitted. ``taxiout.models.Booster`` falls back to CatBoost
where lightgbm is not importable, but that fallback reads three of
``BOOSTER_PARAMS`` -- the tree count, the learning rate and the seed -- and none
of the settings that make a fit repeat, and no report was fitted with it. A
file fitted by it would not be the model the reports measure, so this refuses.

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
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.corridor import load_edges  # noqa: E402
from taxiout.destination import load_airports  # noqa: E402
from taxiout.metar import load_metar, regime_by_hour  # noqa: E402
from taxiout.challenge import (  # noqa: E402
    SUBMITTING,
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
# saturation rung sits above it, has never been fitted on top of it, and is
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
    "taxiout.tails": ("TailModel", "PlanShiftModel", "combine"),
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


def _git(*args: str) -> subprocess.CompletedProcess | None:
    """A read-only git command in the repository, or ``None`` if there is no git.

    ``GIT_OPTIONAL_LOCKS=0`` so that asking about the tree never refreshes the
    index: the tree is read here, not written.
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError:
        return None


def is_ignored(path: Path) -> bool:
    """Whether git would ignore ``path``. Asked, not assumed."""
    proc = _git("check-ignore", "-q", str(path))
    return proc is not None and proc.returncode == 0


def revision() -> tuple[str | None, bool | None]:
    """The commit checked out, and whether the tree matches it, as git says."""
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if head is None or head.returncode != 0:
        return None, None
    if status is None or status.returncode != 0:
        return head.stdout.strip(), None
    return head.stdout.strip(), not status.stdout.strip()


def _length_dependence(encoding) -> dict:
    """Per level, the measured ratio and whether ``k`` moves with the route."""
    out = {}
    for level, shrink in encoding.shrinkage.items():
        ratio = float(shrink.within.ratio)
        out[level] = {
            "ratio": round(ratio, 6) if np.isfinite(ratio) else None,
            "material": bool(shrink.within.material),
        }
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    backend = booster_backend()
    if backend == "none":
        print('no booster installed; pip install -e ".[modelling]"', file=sys.stderr)
        return 1
    if backend != "lightgbm":
        print(
            f"refusing to fit with {backend}: lightgbm is not importable, and the "
            "fallback carries none of BOOSTER_PARAMS' determinism settings and was "
            'never measured by any report; pip install -e ".[modelling]"',
            file=sys.stderr,
        )
        return 1
    import lightgbm

    modules, reason = stream_b()
    if modules is None:
        print(f"WARNING: {reason}; cannot write a submission yet", file=sys.stderr)
        return 1
    submission, tails = modules

    version = args.version or next_version(args.team)
    path = SUBMISSIONS / f"{args.team}_v{version}.parquet"
    manifest = SUBMISSIONS / f"{args.team}_v{version}.manifest.json"
    submission.check_filename(path)
    for p in (path, manifest):
        if not is_ignored(p):
            print(
                f"refusing to write {p.relative_to(REPO)}: git does not ignore it, and "
                "a submission is built from data that may not be redistributed",
                file=sys.stderr,
            )
            return 1
        # Asked before the fit rather than after it, so that a version already
        # written costs a message and not a fit.
        if p.exists() and not args.overwrite:
            print(
                f"{p.relative_to(REPO)} exists; bump the version or pass --overwrite",
                file=sys.stderr,
            )
            return 1
    git_revision, git_clean = revision()

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
    # The plan-shifted departures' rule reads the take-off, the two off-block
    # estimates and the airport, all four in REQUIRED_COLUMNS already, and the
    # target at fit time; named here so that its inputs are loaded whatever the
    # feature set becomes.
    plan_shift_columns = set(tails.PlanShiftModel.PREDICT_INPUTS)
    train_columns = sorted(
        set(REQUIRED_COLUMNS) | set(TAILS_TRAIN_COLUMNS) | plan_shift_columns | {TARGET}
    )
    rank_columns = sorted(
        set(REQUIRED_COLUMNS) | set(TAILS_PREDICT_COLUMNS) | plan_shift_columns | {ID}
    )
    train = to_pandas(load_movements(DATA, train_columns, phase="DEP")).reset_index(
        drop=True
    )
    rank = to_pandas(load_ranking(DATA, rank_columns, phase="DEP")).reset_index(drop=True)
    template_sha256 = sha256(DATA / SUBMITTING)
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
    # The plan-shifted departures: fitted on the same rows as the term and
    # against it, and read nowhere in the physical fit, whose rows stay
    # ``fitting_rows``'s.
    plan_shift_model = tails.PlanShiftModel(tail_model).fit(train)
    print(
        f"  {plan_shift_model.n:,} plan-shifted training departures, "
        f"{plan_shift_model.long:,} of them long; add-one share "
        f"{plan_shift_model.share:.4f}"
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
    # measured and not adopted. It is True: the rule of 19 September, read in
    # ``reports/baseline.md`` under the parameters that ship with both further
    # measurements printed and the guard holding, adopts it, and the constant's
    # own comment carries the readings.
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
    bare_prior = None
    if encoder is not None:
        columns = encoder.transform(Xr)
        Xr = with_encoding(Xr, columns)
        pair = group_key(Xr, "pair")
        seen = pd.Series(
            pair.map(encoder.moments["pair"]["n"]).to_numpy(),
            index=Xr.index,
        ).fillna(0.0)
        unseen = seen <= 0
        bare_prior = {"pairs": int(pair[unseen].nunique()), "rows": int(unseen.sum())}
        print(
            f"  {bare_prior['rows']:,} of {len(Xr):,} ranking rows, on "
            f"{bare_prior['pairs']:,} pairs, sit on a pair the training year never saw "
            "and encode to the geometric prior"
        )
    physical = model.predict(Xr)
    tail = tail_model.predict(rank)
    plan_shift = plan_shift_model.predict(rank)
    # The term first, then the plan-shifted rule on the rows the term leaves.
    shifted = plan_shift.notna() & tail.isna()
    pred = tails.combine(physical, tail, plan_shift)
    # The submission's target is a non-negative int32; a boosted model can
    # extrapolate below zero on a row unlike anything it was fitted on, and a
    # negative taxi-out is not a prediction anyone should submit.
    negative = int((pred < 0).sum())
    pred = pred.clip(lower=0)
    print(
        f"  {int(tail.notna().sum()):,} rows take the record-quality term, "
        f"{int(shifted.sum()):,} the plan-shifted departures' rule; "
        f"{negative:,} negative predictions clipped to zero"
    )
    print(
        f"  mean {pred.mean():,.0f} s, median {pred.median():,.0f} s, "
        f"max {pred.max():,.0f} s"
    )

    out = pd.DataFrame({ID: rank[ID].to_numpy(), TARGET: np.rint(pred.to_numpy())})
    written = submission.write(out, template, path, overwrite=args.overwrite)
    print(f"wrote {written.relative_to(REPO)} ({written.stat().st_size:,} bytes)")

    # Written only once the file has read back and validated against the
    # template, so a manifest never describes a file that was refused.
    month = pd.to_datetime(rank["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m")
    # Whether each encoding's within-group variance was found materially
    # length-dependent, which ``taxiout.encoding`` decides from the fit rows
    # and which decides whether ``k`` moves with the route: the training rows'
    # out-of-fold encoding and the ranking rows' lookup each carry their own.
    sides = {"training": encoded_train, "ranking": encoder} if encoder is not None else {}
    record = {
        "file": written.name,
        "git_revision": git_revision,
        "git_tree_clean": git_clean,
        "booster_backend": backend,
        "booster_version": lightgbm.__version__,
        "booster_params": dict(model.params),
        "seeds": BAGGING_SEEDS if SEED_AVERAGE_ADOPTED else 1,
        "fit_rows": int(fit_rows.sum()),
        "record_quality_covered_cells": int(tail_model.cells["covered"].sum()),
        "record_quality_rows": int(tail.notna().sum()),
        "record_quality_rows_by_month": {
            m: int(n) for m, n in tail.notna().groupby(month.to_numpy()).sum().items()
        },
        "plan_shift_rows": {
            m: int(n) for m, n in shifted.groupby(month.to_numpy()).sum().items()
        },
        "plan_shift_n": plan_shift_model.n,
        "plan_shift_long": plan_shift_model.long,
        "plan_shift_share": plan_shift_model.share,
        "plan_shift_rest_mean": {
            icao: float(plan_shift_model.rest_mean[icao])
            for icao in sorted(set(rank.loc[shifted, "ADEP_mvt"].astype(str)))
        },
        "ranking_bare_prior": bare_prior,
        "encoding_length_dependence": {
            side: _length_dependence(enc) for side, enc in sides.items()
        },
        "template_sha256": template_sha256,
        "file_sha256": sha256(written),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer_used_pairs": None,
        "scorer_score": None,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
