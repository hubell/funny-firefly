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
movement files with the rows ``taxiout.tails`` covers excluded. The
record-quality term and the rule of the plan-shifted departures
(``taxiout.tails.PlanShiftModel``, fixed by
``reports/preregistered-2026-10-02.md``) are fitted and counted beside it, and
``taxiout.models.TAIL_RULES_SUBSTITUTED`` decides whether their predictions
replace the model's on the rows they cover: the term on its rows, then the rule
on the rows of its class the term leaves. It was False for v8, under
``reports/preregistered-2026-10-04.md``, and is True again from v9, under
``reports/preregistered-2026-10-04-tails.md``. Where
``taxiout.models.RECORD_QUALITY_FROM_C6B`` is True the term's rows take C6b's
prediction (``taxiout.tails.C6Model``, ``reports/c6.md``) instead of the
term's, on exactly the term's rows. Nothing here is chosen by a leaderboard
score; the rung and the hyperparameters are the ones ``reports/baseline.md``
was generated with.

**The configuration.** ``taxiout.models.SUBMISSION_CONFIGURATION`` names the
candidates the board's rule adopted (``reports/candidates-2026-10-05.md``), or
is ``None`` for the rung alone. When it names some, the model is formed as
``scripts/candidates_board.py`` forms that combination, through the harness's
own ``combination``, ``member_columns``, ``member_params`` and
``pinned_catboost``: C1's block (``taxiout.aobt3``) and C4's
(``taxiout.flight_history``) added to the rung in one LightGBM booster, and
C3b's CatBoost member on the rung's columns, each at ``BAGGING_SEEDS`` seeds
from the ladder's, and the two averaged at equal weights. C1's queue counts
are over the side's own departures: the training year for the training rows,
and each ranking month for its rows. C4's values for the ranking rows are
looked up over the whole training year (``fit_flight_history``).
``--december-check`` fits the same path at the ladder's seed on the December
leg (January to November 2025 against December) and compares its predictions
with the harness's stored ones, to the bit, writing nothing.

Beside the file it writes ``<team>_v<n>.booster.parquet``: the model's
prediction for every ranking row before any substitution, in the template's
order, as written (``TAXITIME_SEC_mvt``) and unrounded (``prediction``).

The file goes to ``data/submissions/<team>_v<n>.parquet``, with a manifest
beside it at ``<team>_v<n>.manifest.json``, and the script checks with git
that both paths are ignored before writing a byte. Predictions built from the
challenge data are not redistributable and neither is the frame that produced
them; ``.gitignore`` excludes every parquet file anywhere in the tree and the
whole of ``data/submissions/``, and this verifies rather than assumes it.

The manifest says what the file was built from and by: the git revision and
whether the tree was clean, the booster, its version and the parameters it was
fitted with, how many rows it was fitted on, whether the two rules'
predictions were substituted, how many ranking rows the record-quality term
covers and how many the plan-shifted departures' rule covers, which are the
rows each would have taken when they are not, what that rule was fitted to, and how many sit on a pair the training year never saw,
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
extras. Takes about half an hour on eight cores: v7, the five-seed file, took
32 min 26 s on 2 October 2026 by ``/usr/bin/time``, and its second build that
day, byte for byte the same file, 31 min 2 s.

Usage
-----
    python scripts/make_submission.py --team myteam
    python scripts/make_submission.py --team myteam --version 3
    python scripts/make_submission.py --december-check
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
# The harness that forms the adopted configuration, ``candidates_board``.
sys.path.insert(0, str(REPO / "scripts"))

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
    RECORD_QUALITY_FROM_C6B,
    SUBMISSION_CONFIGURATION,
    TOP_RUNG,
    Booster,
    SEED_AVERAGE_ADOPTED,
    TAIL_RULES_SUBSTITUTED,
    SeedAverage,
    booster_backend,
    fit_rows as fitting_rows,
)
from taxiout.aobt3 import REQUIRED_COLUMNS as AOBT3_COLUMNS  # noqa: E402
from taxiout.aobt3 import aobt3_features  # noqa: E402
from taxiout.flight_history import (  # noqa: E402
    FLIGHT,
    fit_flight_history,
    flight_history,
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
    "taxiout.tails": ("TailModel", "PlanShiftModel", "C6Model", "combine"),
}

# The harness's name for the configuration, and the December record it is
# checked against; the stored per-row predictions live under its directory.
CONFIGURATION_KEY = "combo4"
CHECK_SPLIT = "december"


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


def submitted(physical, tail: pd.Series, plan_shift: pd.Series, substitute: bool):
    """What the file predicts, before rounding and clipping.

    With ``substitute`` the term on the rows it covers, then the plan-shifted
    rule on the rows of its class the term leaves, then the model, which is
    ``taxiout.tails.combine``. Without it the model on every row, passed
    through ``combine`` with nothing to substitute so that the alignment checks
    are the same either way. ``substitute`` is
    ``taxiout.models.TAIL_RULES_SUBSTITUTED`` and is decided there, not here.
    """
    from taxiout.tails import combine

    if substitute:
        return combine(physical, tail, plan_shift)
    combine(physical, tail, plan_shift)
    return combine(physical, pd.Series(np.nan, index=tail.index, dtype=float))


def configuration():
    """The adopted configuration as the harness forms it, or ``None`` for the rung.

    ``taxiout.models.SUBMISSION_CONFIGURATION`` names the candidates; the
    candidate objects and their combination are ``candidates_board``'s own, so
    the members, their columns and their parameters are the ones the board's
    records were fitted with.
    """
    if SUBMISSION_CONFIGURATION is None:
        return None
    import candidates_board as board

    combo = board.combination(
        list(SUBMISSION_CONFIGURATION),
        board.candidates(),
        CONFIGURATION_KEY,
        "the adopted configuration",
    )
    if combo.layers or (combo.member is not None and combo.member.weighted):
        raise SystemExit(
            "the configuration carries a layer or weights, which this script does"
            " not form; refusing rather than fitting something else"
        )
    return combo


class Configuration:
    """The adopted combination, fitted as ``candidates_board`` fits its members.

    The booster is the rung with the configuration's blocks added, under
    ``member_params`` at ``seeds`` consecutive seeds from the ladder's, its
    predictions averaged as ``SeedAverage`` averages them; each blend member is
    fitted the same way through ``Booster`` (a CatBoost member with the
    harness's ``pinned_catboost``, its trained parameters checked as the
    harness checks them), its seeds averaged the same way; and the booster and
    the members are averaged at equal weights, as ``form_combination`` averages
    them. At one seed every part is the harness's fit at the ladder's seed.
    """

    def __init__(self, combo, seeds: int) -> None:
        import candidates_board as board

        self.board = board
        self.combo = combo
        self.seeds = int(seeds)
        self.member = combo.member
        self.columns = board.member_columns(combo.member)
        self.params = board.member_params(combo.member, 0)
        self.booster = None
        self.blend: list[list[Booster]] = []
        self.catboost_params: dict | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, rows: pd.Series) -> "Configuration":
        board = self.board
        self.booster = SeedAverage(self.columns, self.params, seeds=self.seeds).fit(
            X.loc[rows], y[rows]
        )
        self.blend = []
        for member in self.combo.members:
            fits = []
            for s in range(self.seeds):
                model = Booster(
                    board.member_columns(member), board.member_params(member, s)
                )
                if member.backend == "catboost":
                    model.backend = "catboost"
                    with board.pinned_catboost(member.gpu):
                        model.fit(X.loc[rows], y[rows])
                    want = board.board_rule().catboost
                    trained = model.model.get_all_params()
                    got = {k: trained.get(k) for k in want}
                    got["thread_count"] = model.model.get_params().get("thread_count")
                    if any(
                        not board.same_parameter(got.get(k), v) for k, v in want.items()
                    ):
                        raise SystemExit(f"the CatBoost fit ran with {got}, not {want}")
                    self.catboost_params = got
                else:
                    model.fit(X.loc[rows], y[rows])
                fits.append(model)
            self.blend.append(fits)
        return self

    def parts(self, Xr: pd.DataFrame) -> dict[str, np.ndarray]:
        """The booster's and each member's seed-averaged predictions."""
        out = {self.member.key: np.asarray(self.booster.predict(Xr), dtype=float)}
        for member, fits in zip(self.combo.members, self.blend):
            total = None
            for model in fits:
                p = model.predict(Xr)
                total = p if total is None else total + p
            out[member.key] = np.asarray(total / float(len(fits)), dtype=float)
        return out

    def predict(self, Xr: pd.DataFrame) -> pd.Series:
        parts = self.parts(Xr)
        pred = np.mean(np.vstack(list(parts.values())), axis=0)
        return pd.Series(pred, index=Xr.index, dtype=float)


def with_blocks(
    combo, X, y, fit_rows, Xr, blocks_train, blocks_rank, flight_train, flight_rank
):
    """The configuration's blocks on both sides, as the harness adds them.

    C1's columns are computed per side by the caller (``aobt3_features`` counts
    the departures of the frame it is given); C4's are the split's out-of-fold
    history for the fitting side and the whole training year's lookup for the
    other.
    """
    X = X.copy(deep=False)
    Xr = Xr.copy(deep=False)
    for block in combo.member.blocks if combo.member is not None else ():
        if block == "aobt3":
            for c in blocks_train.columns:
                X[c] = blocks_train[c].to_numpy()
                Xr[c] = blocks_rank[c].to_numpy()
        elif block == "flight":
            got = flight_history(X, flight_train, y, fit_rows)
            for c in got.columns:
                X[c] = got[c]
            looked = fit_flight_history(X, flight_train, y, fit_rows).transform(
                Xr, flight_rank
            )
            for c in looked.columns:
                Xr[c] = looked[c]
        else:
            raise SystemExit(f"this script does not build the block {block!r}")
    return X, Xr


def aobt3_side(frame: pd.DataFrame, extra: pd.DataFrame, by_month: bool) -> pd.DataFrame:
    """C1's columns for one side, aligned with ``frame``'s rows.

    ``extra`` carries ``AOBT_3_flt`` and the fields C1 differences it against,
    read for C1's module alone, and must be ``frame``'s rows in order. The
    training year counts the training year, as the harness does; a ranking side
    counts each month on its own (``by_month``).
    """
    same = len(extra) == len(frame) and np.array_equal(
        extra[ID].to_numpy(dtype=float), frame[ID].to_numpy(dtype=float)
    )
    if not same:
        raise SystemExit("C1's columns are not of the departures' rows in order")
    if not by_month:
        got = aobt3_features(extra.reset_index(drop=True))
    else:
        month = pd.to_datetime(extra["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m")
        parts = []
        for m in sorted(month.unique()):
            rows = np.flatnonzero((month == m).to_numpy())
            part = aobt3_features(extra.iloc[rows].reset_index(drop=True))
            part.index = rows
            parts.append(part)
        got = pd.concat(parts).sort_index()
    got.index = frame.index
    return got


def fit_and_predict(
    train, X, rank, Xr, tails, *, seeds, blocks_train=None, blocks_rank=None
) -> dict:
    """Fit on ``train`` and predict ``rank``: the rules, the encoding and the model.

    The ranking build calls it with the training year and the ranking set; the
    December check with January to November and December of the same year.
    """
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
    # C6b, on the term's rows, fitted on every fitting row of the term's cells.
    c6_model = None
    if RECORD_QUALITY_FROM_C6B:
        c6_model = tails.C6Model(tail_model).fit(train)
        print(f"  C6b fitted on {c6_model.fitting_rows:,} rows of the term's cells")

    print(f"fitting {RUNG.label} ...")
    t0 = time.time()
    y = pd.to_numeric(train[TARGET], errors="coerce")
    # Guard 3, at submission. The record-quality rows are out of everything the
    # encoding reads, for the same reason they are out of the physical fit: a
    # defaulted block time is not a taxi time, and one of them inside a thin
    # pair would otherwise *be* that pair's encoding in the shipped model.
    # ...and the recorded-taxi tail, by the same rule and the same constant the
    # ladder fits under (``taxiout.models.fit_rows``). If these two came apart,
    # the model that ships would not be the model reports/baseline.md measures.
    fit_rows = fitting_rows(y.notna(), covered, y)

    # The encoding, if the rung reads it. Computed once over the training
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

    combo = configuration()
    if combo is not None:
        X, Xr = with_blocks(
            combo,
            X,
            y,
            fit_rows,
            Xr,
            blocks_train,
            blocks_rank,
            train[FLIGHT],
            rank[FLIGHT],
        )

    # A rung whose block this script cannot build must not be fitted on a
    # column of nothing. ``build_features`` leaves a block missing rather than
    # zero when its input was not supplied, and a booster given an all-null
    # column will fit happily and predict from the rest, producing a submission
    # that silently is not the rung it claims to be.
    wanted = RUNG.features if combo is None else Configuration(combo, 1).columns
    empty = [c for c in wanted if X[c].isna().all()]
    if empty:
        raise SystemExit(
            f"refusing to fit {RUNG.label}: this script cannot build "
            + ", ".join(f"`{c}`" for c in empty)
            + "; the column arrives missing and the rung would be fitted without it"
        )
    if combo is not None:
        # The adopted configuration, at the seeds the rule below gives the rung.
        model = Configuration(combo, seeds).fit(X, y, fit_rows)
        how = f"as the adopted configuration under {seeds} seed(s)"
    # Seed averaging, if the rule in ``reports/baseline.md``'s *Seed averaging*
    # adopted it. It is an estimator choice and not a rung, so the ladder stays
    # single-seed and this is the one place it would apply -- and whether it
    # applies is ``taxiout.models.SEED_AVERAGE_ADOPTED`` rather than a decision
    # taken here, so that this script cannot ship a model the report says was
    # measured and not adopted. It is True: the rule of 19 September, read in
    # ``reports/baseline.md`` under the parameters that ship with both further
    # measurements printed and the guard holding, adopts it, and the constant's
    # own comment carries the readings.
    elif SEED_AVERAGE_ADOPTED:
        model = SeedAverage(RUNG.features).fit(X.loc[fit_rows], y[fit_rows])
        how = f"under {BAGGING_SEEDS} seeds"
    else:
        model = Booster(RUNG.features).fit(X.loc[fit_rows], y[fit_rows])
        how = "under a single seed (seed averaging measured, not adopted)"
    print(f"  fitted on {int(fit_rows.sum()):,} rows {how} in {time.time() - t0:.0f} s")

    print("predicting ...")
    if bare_prior is not None:
        print(
            f"  {bare_prior['rows']:,} of {len(Xr):,} rows to predict, on "
            f"{bare_prior['pairs']:,} pairs, sit on a pair the training year never saw "
            "and encode to the geometric prior"
        )
    parts = model.parts(Xr) if combo is not None else None
    physical = model.predict(Xr)
    return {
        "tail_model": tail_model,
        "plan_shift_model": plan_shift_model,
        "c6_model": c6_model,
        "fit_rows": fit_rows,
        "encoder": encoder,
        "encoded_train": encoded_train,
        "bare_prior": bare_prior,
        "combo": combo,
        "model": model,
        "parts": parts,
        "physical": physical,
        "tail": tail_model.predict(rank),
        "plan_shift": plan_shift_model.predict(rank),
        "c6": None if c6_model is None else c6_model.predict(rank),
    }


def load_inputs() -> dict | None:
    """The tables every fit reads besides the departures, or ``None`` with a reason printed."""
    if not METAR.exists() or not any(METAR.glob("*.csv")):
        print(
            "no METAR cache under data/raw/metar; the submission rung reads it. "
            "Run: python scripts/fetch_metar.py",
            file=sys.stderr,
        )
        return None
    if not OURAIRPORTS.exists():
        print(
            "no OurAirports cache under data/raw; the submission rung reads the "
            "destination geometry. Run: python scripts/fetch_ourairports.py",
            file=sys.stderr,
        )
        return None
    matrix = load_matrix()
    return {
        "matrix": matrix,
        "fallback": airport_fallback(matrix),
        "weather": regime_by_hour(load_metar(METAR)),
        "airports": load_airports(OURAIRPORTS),
        "edges": load_edge_sets(),
    }


def features_of(frame, moves, inputs) -> pd.DataFrame:
    return build_features(
        frame,
        inputs["matrix"],
        fallback=inputs["fallback"],
        movements=moves,
        edges=inputs["edges"],
        weather=inputs["weather"],
        airports=inputs["airports"],
    )


def columns_needed(tails) -> tuple[list[str], list[str]]:
    """The training and the ranking columns: the features', the rules' and C6's."""
    # The plan-shifted departures' rule reads the take-off, the two off-block
    # estimates and the airport, all four in REQUIRED_COLUMNS already, and the
    # target at fit time; named here so that its inputs are loaded whatever the
    # feature set becomes. C6 reads the callsign, the stand and the type too.
    plan_shift_columns = set(tails.PlanShiftModel.PREDICT_INPUTS)
    c6_columns = set(tails.C6Model.PREDICT_INPUTS) | {FLIGHT}
    train_columns = sorted(
        set(REQUIRED_COLUMNS)
        | set(TAILS_TRAIN_COLUMNS)
        | plan_shift_columns
        | c6_columns
        | set(tails.C6Model.FIT_ONLY_INPUTS)
        | {TARGET, ID}
    )
    rank_columns = sorted(
        set(REQUIRED_COLUMNS)
        | set(TAILS_PREDICT_COLUMNS)
        | plan_shift_columns
        | c6_columns
        | {ID}
    )
    return train_columns, rank_columns


def aobt3_columns() -> list[str]:
    return sorted(set(AOBT3_COLUMNS) | {ID})


def december_check() -> int:
    """The submission path at the ladder's seed on the December leg, against the harness.

    Fits January to November 2025 and predicts December exactly as the ranking
    build fits the year and predicts the ranking set, at one seed, and compares
    the predictions, the booster's and each member's with the harness's stored
    records of the configuration, to the bit. Writes nothing.
    """
    import candidates as cb
    import candidates_board as board

    modules, reason = stream_b()
    if modules is None:
        print(f"WARNING: {reason}", file=sys.stderr)
        return 1
    _, tails = modules
    combo = configuration()
    if combo is None:
        print("no configuration is adopted; nothing to check", file=sys.stderr)
        return 1
    inputs = load_inputs()
    if inputs is None:
        return 1
    train_columns, _ = columns_needed(tails)
    print("loading ...")
    year = to_pandas(load_movements(DATA, train_columns, phase="DEP")).reset_index(
        drop=True
    )
    moves = to_pandas(load_movements(DATA, list(MOVEMENT_COLUMNS))).reset_index(drop=True)
    X_year = features_of(year, moves, inputs)
    del moves
    blocks = None
    if "aobt3" in combo.member.blocks:
        extra = to_pandas(load_movements(DATA, aobt3_columns(), phase="DEP")).reset_index(
            drop=True
        )
        blocks = aobt3_side(year, extra, by_month=False)
    split = next(s for s in cb.leg_plan() if s.name == CHECK_SPLIT)
    month = cb.month_of(year["MVT_TIME_UTC_mvt"])
    fit_side = month.isin(split.train_months).to_numpy()
    check_side = (month == split.valid_month).to_numpy()
    got = fit_and_predict(
        year[fit_side],
        X_year[fit_side],
        year[check_side],
        X_year[check_side],
        tails,
        seeds=1,
        blocks_train=None if blocks is None else blocks[fit_side],
        blocks_rank=None if blocks is None else blocks[check_side],
    )
    work = board.WORK
    ids = year.loc[check_side, ID].to_numpy(dtype=float)
    wanted = {
        cb.record_name(split.name, combo.key, 0): got["physical"].to_numpy(dtype=float),
        **{
            cb.record_name(split.name, key, 0): pred for key, pred in got["parts"].items()
        },
    }
    ok = True
    for name, pred in wanted.items():
        stored = pd.read_parquet(work / f"{name}.parquet")
        record = json.loads((work / f"{name}.json").read_text(encoding="utf-8"))
        same_ids = np.array_equal(stored["mvt_id"].to_numpy(dtype=float), ids)
        want = stored["pred"].to_numpy(dtype=float)
        same = same_ids and np.array_equal(pred, want)
        digest = hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()
        diff = float(np.max(np.abs(pred - want))) if same_ids else float("nan")
        print(
            f"{name}: {len(pred):,} rows, identifiers in order {same_ids}, "
            f"equal to the bit {same}, largest absolute difference {diff!r} s; "
            f"sha256 {digest}, the record's {record['pred_sha256']}"
        )
        ok = ok and same and digest == record["pred_sha256"]
    print("DECEMBER CHECK: " + ("equal to the bit" if ok else "DIFFERENT"))
    return 0 if ok else 1


def write_booster_file(path: Path, rank, physical, template, submission) -> Path:
    """The model's prediction before any substitution, in the template's order.

    ``TAXITIME_SEC_mvt`` as the file writes a prediction (the floor at zero and
    the rounding, int32) and ``prediction`` unrounded, both by identifier.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = pd.Series(physical.to_numpy(dtype=float), index=rank[ID].to_numpy(dtype=float))
    written = pd.DataFrame(
        {ID: rank[ID].to_numpy(), TARGET: np.rint(physical.clip(lower=0).to_numpy())}
    )
    out = submission.prepare(written, template)
    table = pa.table(
        {
            ID: pa.array(out[ID].to_numpy(), type=pa.float64()),
            TARGET: pa.array(out[TARGET].to_numpy(), type=pa.int32()),
            "prediction": pa.array(
                raw.reindex(out[ID].to_numpy()).to_numpy(dtype=float), type=pa.float64()
            ),
        }
    )
    pq.write_table(table, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", help="team name, as the organisers hold it")
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
    ap.add_argument(
        "--december-check",
        action="store_true",
        help="fit the configuration on the December leg at the ladder's seed and"
        " compare with the harness's stored predictions; writes nothing",
    )
    args = ap.parse_args()
    if not args.december_check and not args.team:
        ap.error("--team is required unless --december-check")

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

    if args.december_check:
        return december_check()

    modules, reason = stream_b()
    if modules is None:
        print(f"WARNING: {reason}; cannot write a submission yet", file=sys.stderr)
        return 1
    submission, tails = modules

    version = args.version or next_version(args.team)
    path = SUBMISSIONS / f"{args.team}_v{version}.parquet"
    manifest = SUBMISSIONS / f"{args.team}_v{version}.manifest.json"
    booster_path = SUBMISSIONS / f"{args.team}_v{version}.booster.parquet"
    submission.check_filename(path)
    for p in (path, manifest, booster_path):
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

    inputs = load_inputs()
    if inputs is None:
        return 1

    print("loading ...")
    train_columns, rank_columns = columns_needed(tails)
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

    X = features_of(train, train_moves, inputs)
    del train_moves
    Xr = features_of(rank, rank_moves, inputs)
    del rank_moves
    combo = configuration()
    blocks_train = blocks_rank = None
    if combo is not None and "aobt3" in combo.member.blocks:
        blocks_train = aobt3_side(
            train,
            to_pandas(load_movements(DATA, aobt3_columns(), phase="DEP")).reset_index(
                drop=True
            ),
            by_month=False,
        )
        blocks_rank = aobt3_side(
            rank,
            to_pandas(load_ranking(DATA, aobt3_columns(), phase="DEP")).reset_index(
                drop=True
            ),
            by_month=True,
        )
    seeds = BAGGING_SEEDS if SEED_AVERAGE_ADOPTED else 1
    got = fit_and_predict(
        train,
        X,
        rank,
        Xr,
        tails,
        seeds=seeds,
        blocks_train=blocks_train,
        blocks_rank=blocks_rank,
    )
    model, encoder, encoded_train = got["model"], got["encoder"], got["encoded_train"]
    tail_model, plan_shift_model = got["tail_model"], got["plan_shift_model"]
    fit_rows, bare_prior, physical = got["fit_rows"], got["bare_prior"], got["physical"]
    tail, plan_shift = got["tail"], got["plan_shift"]
    # What the term's rows take: the term itself, or C6b on exactly its rows.
    record_quality = got["c6"] if RECORD_QUALITY_FROM_C6B else tail
    if not record_quality.notna().equals(tail.notna()):
        print("C6b does not cover exactly the term's rows; refusing", file=sys.stderr)
        return 1
    # The term first, then the plan-shifted rule on the rows the term leaves:
    # the rows each rule takes when substituted, and would take when not.
    shifted = plan_shift.notna() & tail.isna()
    pred = submitted(physical, record_quality, plan_shift, TAIL_RULES_SUBSTITUTED)
    # The submission's target is a non-negative int32; a boosted model can
    # extrapolate below zero on a row unlike anything it was fitted on, and a
    # negative taxi-out is not a prediction anyone should submit.
    negative = int((pred < 0).sum())
    pred = pred.clip(lower=0)
    print(
        f"  {int(tail.notna().sum()):,} rows "
        + ("take" if TAIL_RULES_SUBSTITUTED else "would take")
        + " the record-quality term"
        + (" (C6b)" if RECORD_QUALITY_FROM_C6B else "")
        + f", {int(shifted.sum()):,} the plan-shifted "
        "departures' rule"
        + ("" if TAIL_RULES_SUBSTITUTED else ", neither substituted")
        + f"; {negative:,} negative predictions clipped to zero"
    )
    print(
        f"  mean {pred.mean():,.0f} s, median {pred.median():,.0f} s, "
        f"max {pred.max():,.0f} s"
    )

    out = pd.DataFrame({ID: rank[ID].to_numpy(), TARGET: np.rint(pred.to_numpy())})
    written = submission.write(out, template, path, overwrite=args.overwrite)
    print(f"wrote {written.relative_to(REPO)} ({written.stat().st_size:,} bytes)")
    booster_file = write_booster_file(booster_path, rank, physical, template, submission)
    print(
        f"wrote {booster_file.relative_to(REPO)} ({booster_file.stat().st_size:,} bytes)"
    )

    # Written only once the file has read back and validated against the
    # template, so a manifest never describes a file that was refused.
    month = pd.to_datetime(rank["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m")
    # Whether each encoding's within-group variance was found materially
    # length-dependent, which ``taxiout.encoding`` decides from the fit rows
    # and which decides whether ``k`` moves with the route: the training rows'
    # out-of-fold encoding and the ranking rows' lookup each carry their own.
    sides = {"training": encoded_train, "ranking": encoder} if encoder is not None else {}
    c6_model = got["c6_model"]
    record = {
        "file": written.name,
        "git_revision": git_revision,
        "git_tree_clean": git_clean,
        "booster_backend": backend,
        "booster_version": lightgbm.__version__,
        "booster_params": dict(model.params),
        "seeds": BAGGING_SEEDS if SEED_AVERAGE_ADOPTED else 1,
        "fit_rows": int(fit_rows.sum()),
        "submission_configuration": None
        if combo is None
        else {
            "candidates": list(SUBMISSION_CONFIGURATION),
            "booster": {
                "key": combo.member.key,
                "blocks": list(combo.member.blocks),
                "columns": len(model.columns),
            },
            "members": [
                {"key": m.key, "backend": m.backend, "columns": len(RUNG.features)}
                for m in combo.members
            ],
            "blend": "equal weights",
            "catboost_version": __import__("catboost").__version__,
            "catboost_params": model.catboost_params,
        },
        "tail_rules_substituted": TAIL_RULES_SUBSTITUTED,
        "record_quality_from_c6b": RECORD_QUALITY_FROM_C6B,
        "record_quality_covered_cells": int(tail_model.cells["covered"].sum()),
        "record_quality_rows": int(tail.notna().sum()),
        "record_quality_rows_by_month": {
            m: int(n) for m, n in tail.notna().groupby(month.to_numpy()).sum().items()
        },
        "c6b_fitting_rows": None if c6_model is None else c6_model.fitting_rows,
        "plan_shift_rows": {
            m: int(n) for m, n in shifted.groupby(month.to_numpy()).sum().items()
        },
        "rows_each_rule_would_take": {
            "record_quality": int(tail.notna().sum()),
            "plan_shift": int(shifted.sum()),
        },
        "rows_each_rule_took": {
            "record_quality": int(record_quality.notna().sum())
            if TAIL_RULES_SUBSTITUTED
            else 0,
            "plan_shift": int(shifted.sum()) if TAIL_RULES_SUBSTITUTED else 0,
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
        "booster_file": booster_file.name,
        "booster_file_sha256": sha256(booster_file),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer_used_pairs": None,
        "scorer_score": None,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
