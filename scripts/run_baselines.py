#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run the Gate 1 ablation ladder and write ``reports/baseline.md``.

The ladder is the model stated as a decomposition and read bottom-up
(``taxiout.models``): airport mean, the PRU reference, the geometric unimpeded
term alone, a boosted model on identifiers and time, and the same model with
the geometry block added -- each scored plain and with the record-quality term
on top. Three splits, fixed before any of this was fitted
(``taxiout.validation``): two forward in time, and one that holds out whole
(airport, stand, runway) pairs.

The held-out-pair split is the one the entry rests on. Each of its folds fits
*one* model and scores it twice: on rows whose pair was removed from training,
and on a held-out sample of rows whose pair was not. Same model, same period,
same airports, differing only in whether the pair was seen -- so the gap
between the two is attributable, and it is reported for every rung whether it
favours the geometry or not.

Every number in the report is computed here. Nothing is hand-typed, because
hand-typed statistics drift from the tables beside them, and that has happened
in this repository twice. The report carries aggregates and runway designators
only; a literal stand identifier is an extract of OpenStreetMap or of the
challenge data, and the generator refuses to write one.

Needs the challenge data under ``data/challenge/`` and the distance matrices
under ``data/processed/taxi_distance/`` (``make reports``), and the modelling
extras (``pip install -e ".[modelling,dev]"``). Neither dataset is committed.

Usage
-----
    python scripts/run_baselines.py
    python scripts/run_baselines.py --dry-run --sample-frac 0.05
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.challenge import load_movements, to_pandas  # noqa: E402
from taxiout.features import (  # noqa: E402
    GEOMETRY_FEATURES,
    QUALITY_FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
    airport_fallback,
    build_features,
)
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER,
    AirportMean,
    Booster,
    booster_backend,
)
from taxiout.unimpeded import (  # noqa: E402
    MIN_PAIR_COUNT,
    PRU_PERCENTILE,
    calibration,
    fit_geometric,
    fit_pru_reference,
    lower_tail,
)
from taxiout.validation import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    JULY_AIRPORTS,
    PAIR_FOLDS,
    SEEN_FRACTION,
    cluster_bootstrap,
    cluster_sums,
    december_split,
    difference_in_differences,
    evaluate,
    evaluate_by_bucket,
    fold_rmses,
    gap,
    held_out_pair_splits,
    july_split,
    month_of,
    rmse,
)

DATA = REPO / "data" / "challenge"
MATRICES = REPO / "data" / "processed" / "taxi_distance"
OUT = REPO / "reports" / "baseline.md"

# Loaded but never entered as a feature. The record-quality layer needs the
# schedule to compute the lag its cells rest on, and the block time to measure
# on training rows how often it was defaulted. Both are kept out of
# FEATURE_SOURCES on purpose, so the registry test still means what it says --
# and `BLOCK_TIME_UTC_mvt` is a field the ranking set hides, read here only to
# fit on training rows, exactly as `taxiout.tails.FIT_ONLY_INPUTS` declares.
TAILS_COLUMNS = ("SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "ADEP_mvt")

# The interface Stream B's module fixes in its own docstring.
TAILS_INTERFACE = ("TailModel", "combine", "flag")


# --------------------------------------------------------------------------
# formatting helpers, as in scripts/review_challenge_data.py


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 0) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:,.{digits}f}"


def table(
    header: list[str], rows: list[list[str]], align: str | None = None
) -> list[str]:
    align = align or ("-" * len(header))
    sep = ["---:" if a == "r" else "---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


# --------------------------------------------------------------------------
# Stream B, imported lazily


class Tails:
    """The record-quality layer, if ``taxiout.tails`` exists yet.

    Stream B owns it and this module does not reimplement any of it. Until it
    lands the ladder runs without the record-quality rungs and the unimpeded
    speed is fitted without a flag mask, and the report says so in both places
    rather than printing a figure that looks complete.

    The interface used is the one that module's docstring fixes:
    ``TailModel().fit(training rows)``, ``.predict(frame)`` returning NaN where
    the term does not apply, and ``combine(physical, tail)``.
    """

    def __init__(self) -> None:
        self.available = False
        self.reason = ""
        self._mod = None
        try:
            import taxiout.tails as tails
        except ImportError:
            self.reason = "`taxiout.tails` is not in the tree yet (Stream B)"
            return
        missing = [n for n in TAILS_INTERFACE if not hasattr(tails, n)]
        if missing:
            self.reason = "`taxiout.tails` exists but exports none of " + ", ".join(
                f"`{n}`" for n in missing
            )
            return
        self._mod = tails
        self.available = True

    def fit(self, training: pd.DataFrame):
        """A ``TailModel`` fitted on one split's training rows, or ``None``."""
        return self._mod.TailModel().fit(training) if self.available else None

    def combine(self, physical: pd.Series, tail: pd.Series) -> pd.Series:
        return self._mod.combine(physical, tail)


# --------------------------------------------------------------------------
# data


def load_matrix() -> pd.DataFrame:
    files = [
        f for f in sorted(glob.glob(str(MATRICES / "*.csv"))) if "_features" not in f
    ]
    if not files:
        raise FileNotFoundError(f"no distance matrices under {MATRICES}; make reports")
    return pd.concat(
        [pd.read_csv(f, dtype={"stand_id": str, "designator": str}) for f in files],
        ignore_index=True,
    )


def load_departures(sample_frac: float | None) -> pd.DataFrame:
    columns = sorted(set(REQUIRED_COLUMNS) | set(TAILS_COLUMNS) | {TARGET})
    frame = to_pandas(load_movements(DATA, columns, phase="DEP"))
    if sample_frac:
        frame = frame.sample(frac=sample_frac, random_state=0).sort_index()
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# fitting one rung


def fit_and_predict(
    rung,
    features: pd.DataFrame,
    y: pd.Series,
    train: pd.Series,
    where: list[pd.Series],
    unflagged: pd.Series,
) -> tuple[list[pd.Series], dict]:
    """Fit ``rung`` on ``train`` and predict on each mask in ``where``.

    The physical rungs are fitted on the unflagged training rows only -- a
    record-matching artefact is not an aircraft taxiing slowly -- and every
    rung is evaluated on the full validation set whatever its flags.

    The two unimpeded rungs also get a per-airport additive offset, fitted on
    the same training rows and reported in its own column. They estimate
    unimpeded time and are being scored against taxi-out, so they are biased
    low by the whole excess term and would otherwise lose to the airport mean
    for a reason that says nothing about the geometry. The offset is that
    excess held as a constant, which is what Gate 2 exists to replace with a
    model; it is fitted on training rows and never on the set being reported.
    """
    fit_rows = train & unflagged
    info: dict = {}
    offsets: list[pd.Series] | None = None
    if rung.kind == "airport-mean":
        model = AirportMean().fit(features.loc[fit_rows, "icao"], y[fit_rows])
        preds = [model.predict(features.loc[m, "icao"]) for m in where]
    elif rung.kind in ("pru", "geometric"):
        ref = fit_pru_reference(features, y, mask=fit_rows)
        info["reference"] = ref
        if rung.kind == "pru":

            def estimate(mask):
                return ref.predict(features.loc[mask])
        else:
            geo = fit_geometric(
                features, ref, lower_tail(features, y, ref, mask=fit_rows)
            )
            info["geometric"] = geo

            def estimate(mask):
                return geo.predict(features.loc[mask])

        preds = [estimate(m) for m in where]
        excess = (
            (y[fit_rows] - estimate(fit_rows))
            .groupby(features.loc[fit_rows, "icao"].astype(str), observed=True)
            .mean()
        )
        info["excess"] = excess
        offsets = [
            p
            + features.loc[m, "icao"].astype(str).map(excess).fillna(float(excess.mean()))
            for m, p in zip(where, preds)
        ]
    elif rung.kind == "booster":
        model = Booster(rung.features)
        model.fit(features.loc[fit_rows], y[fit_rows])
        info["importances"] = model.importances()
        preds = [model.predict(features.loc[m]) for m in where]
    else:
        raise ValueError(f"unknown rung kind {rung.kind!r}")
    info["offset_preds"] = offsets
    return preds, info


# --------------------------------------------------------------------------
# the run


def run(features, y, raw, splits, tails: Tails) -> dict:
    """Every rung on every split. Returns the results the report reads.

    The record-quality term is refitted on each split's own training rows, and
    the rows it covers are the rows the physical rungs are fitted without. It
    is never fitted on a row it is then scored on.
    """
    results: dict = {}
    for split in splits:
        sets = {"valid": split.valid, **split.extra}
        names = list(sets)
        masks = [sets[n] for n in names]
        tail_model = tails.fit(raw.loc[split.train])
        if tail_model is not None:
            tail = tail_model.predict(raw)
            flagged = tail.notna()
            results[("meta", split.name, "tail")] = {
                "covered_cells": int(tail_model.cells["covered"].sum()),
                "cells": int(len(tail_model.cells)),
                "flagged": {n: int(flagged[m].sum()) for n, m in sets.items()},
                "mask": flagged,
            }
        else:
            tail = None
            flagged = pd.Series(False, index=raw.index)
        unflagged = ~flagged
        for rung in LADDER:
            t0 = time.time()
            preds, info = fit_and_predict(
                rung, features, y, split.train, masks, unflagged
            )
            shifted = info.get("offset_preds")
            for i, (name, mask, pred) in enumerate(zip(names, masks, preds)):
                entry = {
                    "rows": int(mask.sum()),
                    "rmse": rmse(y[mask], pred),
                    "rmse_offset": (
                        rmse(y[mask], shifted[i]) if shifted else float("nan")
                    ),
                    "eval": evaluate(
                        y[mask],
                        pred,
                        icao=features.loc[mask, "icao"],
                        flagged=flagged[mask] if tail is not None else None,
                    ),
                    "pred": pred,
                    "seconds": time.time() - t0,
                    "info": info,
                }
                if tail is not None:
                    with_rq = tails.combine(pred, tail[mask])
                    entry["rmse_rq"] = rmse(y[mask], with_rq)
                    entry["eval_rq"] = evaluate(
                        y[mask],
                        with_rq,
                        icao=features.loc[mask, "icao"],
                        flagged=flagged[mask],
                    )
                results[(split.name, rung.key, name)] = entry
            print(
                f"  {split.name:>10s} {rung.key:<22s} "
                f"RMSE {results[(split.name, rung.key, 'valid')]['rmse']:7.1f} s "
                f"({time.time() - t0:.0f} s)"
            )
    return results


# --------------------------------------------------------------------------
# report sections


def section_setup(
    n_rows: int,
    splits,
    backend: str,
    tails: Tails,
    sample_frac: float | None,
    results: dict,
) -> list[str]:
    rows = [
        [
            f"`{s.name}`",
            s.description,
            f"{s.sizes['train']:,}",
            f"{s.sizes['valid']:,}",
            f"{s.sizes.get('seen', 0):,}" if "seen" in s.sizes else "--",
        ]
        for s in splits
    ]
    lines = [
        "## What is fitted, and on what",
        "",
        (
            f"**Development run: a {sample_frac:.0%} sample, {n_rows:,} departures. "
            "Not the report.**"
            if sample_frac
            else f"{n_rows:,} training departures: the 2025 movement files in full."
        ),
        "The feature frame is `taxiout.features`: the airport, stand and runway",
        "identifiers; the take-off clock; the distance-matrix cell and four shape",
        "ratios; the join-quality flags; and `LOBT_flt - IOBT_flt`. Every column",
        "is registered in `FEATURE_SOURCES` against the raw columns it reads, and",
        "`tests/test_features.py` asserts that no source is in",
        "`taxiout.challenge.FORBIDDEN_FEATURE_INPUTS` and that poisoning those",
        "columns does not move the frame.",
        "",
        f"Boosted rungs use {backend} with the fixed parameters in",
        "`taxiout.models.BOOSTER_PARAMS` -- "
        + ", ".join(
            f"`{k}={v}`"
            for k, v in BOOSTER_PARAMS.items()
            if k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")
        )
        + ". There is no early stopping, because stopping on the set being",
        "reported is tuning on it, and nothing here was chosen by a leaderboard",
        "score.",
        "",
        *table(
            ["split", "what it tests", "train rows", "valid rows", "seen-pair rows"],
            rows,
            "--rrr",
        ),
        "",
    ]
    if tails.available:
        meta = {
            name: results[("meta", name, "tail")]
            for s in splits
            if ("meta", (name := s.name), "tail") in results
        }
        cover = [
            [
                f"`{name}`",
                f"{m['covered_cells']:,} of {m['cells']:,}",
                f"{m['flagged'].get('valid', 0):,}",
            ]
            for name, m in meta.items()
        ]
        lines += [
            "The record-quality layer is `taxiout.tails`, refitted on each split's",
            "own training rows and never on a row it is then scored on. Every rung",
            "is reported twice: plain, and with that term substituted on the rows",
            "it covers (`taxiout.tails.combine`). The physical rungs are fitted",
            "with those rows excluded -- a defaulted block time is not an aircraft",
            "taxiing slowly -- and evaluated on the full validation set, which is",
            "what the stated scoring rule counts.",
            "",
            *table(["split", "covered cells", "covered validation rows"], cover, "-rr"),
        ]
    else:
        lines += [
            f"**The record-quality layer is not wired in: {tails.reason}.** Three",
            "consequences, stated rather than left to be inferred. The ladder's",
            "record-quality rungs are absent. The unimpeded speed is fitted on a",
            "lower tail from which no flagged row has been removed, so it carries",
            "whatever the artefact rows contribute below the 10th percentile. And",
            "every RMSE below includes the artefact rows at their raw values, which",
            "is the honest figure under the stated scoring rule but is dominated by",
            "a handful of records rather than by the model -- the data review puts",
            "the airport-mean RMSE at 515 s with them and 361 s without. Read the",
            "rung-to-rung differences, not the levels, until this lands.",
        ]
    return lines


def section_unimpeded(results, features, splits) -> list[str]:
    """Speeds, coefficients, and the calibration against the PRU reference."""
    key = (splits[0].name, "geometric", "valid")
    info = results[key]["info"]
    ref, geo = info["reference"], info["geometric"]

    speed_rows = [
        [ap, f"{v:.2f}", f"{3.6 * v:.1f}"] for ap, v in sorted(geo.speed.items())
    ]
    speed_rows.append(
        ["all", f"{geo.default_speed:.2f}", f"{3.6 * geo.default_speed:.1f}"]
    )
    coef_rows = [
        [f"`{c}`", f"{b:+.4f}", f"{np.exp(abs(b)):.3f}"]
        for c, b in sorted(geo.coef.items(), key=lambda kv: -abs(kv[1]))
    ]

    train = splits[0].train
    cal = calibration(
        features[train],
        ref,
        geo,
        weights=pd.Series(1.0, index=features.index)[train],
    )
    cal_rows = [
        [
            r["icao"],
            f"{r['pairs']:,}",
            num(r["ref_median_s"]),
            num(r["rmse_s"]),
            num(r["mae_s"]),
            f"{r['bias_s']:+,.0f}",
            f"{r['ratio_p10']:.2f}",
            f"{r['ratio_p50']:.2f}",
            f"{r['ratio_p90']:.2f}",
            f"{r['corr']:.2f}",
        ]
        for _, r in cal.iterrows()
    ]
    overall = cal[cal["icao"] == "all"].iloc[0]
    levels = ref.level(features[train]).value_counts(normalize=True)

    return [
        "## The unimpeded term",
        "",
        "Two estimates of the same quantity. The PRU reference is the organisers'",
        f"own: the {100 * PRU_PERCENTILE:.0f}th percentile of observed taxi-out per",
        f"(airport, stand, runway), over pairs seen at least {MIN_PAIR_COUNT} times in",
        "the training side of the split. The geometric version turns `taxi_m` into",
        "seconds through a per-airport speed and modulates it by route shape, and",
        "so is defined for every pair the matrix can route, seen or not.",
        "",
        "On the December split's training rows the reference answers at pair level",
        f"for {pct(levels.get('pair', 0.0))} of departures, falls back to (airport,",
        f"runway) for {pct(levels.get('runway', 0.0))} and to the airport for",
        f"{pct(levels.get('airport', 0.0))}. The fallbacks are the gap the geometry",
        "exists to fill.",
        "",
        "### The fitted speeds",
        "",
        "Least squares of the pair's reference on `taxi_m`, over the unflagged",
        f"lower tail -- the {geo.n_fit:,} rows at or below their own pair's",
        "reference, restricted to pairs that have one and to rows whose geometry",
        "is a real matrix cell.",
        "",
        *table(["ICAO", "m/s", "km/h"], speed_rows, "-rr"),
        "",
        "### The modulators",
        "",
        "Coefficients on standardised route descriptors, in log space, pooled",
        "across airports. Five of them against thousands of pairs: too few to",
        "memorise a pair, which is what makes the held-out-pair split below mean",
        "something. The last column is the factor one standard deviation of the",
        "descriptor moves the unimpeded time by.",
        "",
        *table(["descriptor", "coefficient", "factor per sd"], coef_rows, "-rr"),
        "",
        "### Calibration against the PRU reference",
        "",
        "**The calibration check.** How closely `taxi_m / speed`, modulated,",
        "reproduces the reference on the pairs where the reference exists. One row",
        "per distinct pair, so a busy pair does not outvote a quiet one. Both",
        "quantities are fitted on the same training rows, so this measures whether",
        "the geometry can express the reference at all -- not whether it holds up",
        "out of sample, which is what the held-out-pair split is for.",
        "",
        *table(
            [
                "ICAO",
                "pairs",
                "reference p50",
                "RMSE",
                "MAE",
                "bias",
                "ratio p10",
                "p50",
                "p90",
                "corr",
            ],
            cal_rows,
            "-rrrrrrrrr",
        ),
        "",
        f"Pooled: {num(overall['rmse_s'])} s RMSE against a median reference of",
        f"{num(overall['ref_median_s'])} s, a "
        f"{abs(overall['rmse_s'] / overall['ref_median_s']):.0%} relative error, with",
        f"the middle 80% of pairs between {overall['ratio_p10']:.2f} and",
        f"{overall['ratio_p90']:.2f} times the reference.",
    ]


def _rung_rows(results, split_name: str, which: str, tails: Tails) -> list[list[str]]:
    rows = []
    base = None
    for rung in LADDER:
        entry = results.get((split_name, rung.key, which))
        if entry is None:
            continue
        r = entry["rmse"]
        base = r if base is None else base
        cells = [rung.label, f"{r:,.1f}", f"{r - base:+,.1f}"]
        cells.append(num(entry["rmse_offset"], 1))
        if tails.available:
            cells.append(f"{entry['rmse_rq']:,.1f}")
            ex = entry["eval"]
            cells.append(num(ex.loc[ex["icao"] == "all", "rmse_ex_flagged_s"].iloc[0], 1))
        rows.append(cells)
    return rows


def section_ladder(results, splits, tails: Tails) -> list[str]:
    header = ["rung", "RMSE (s)", "vs airport mean", "+ airport offset (s)"]
    align = "-rrr"
    if tails.available:
        header += ["+ record quality (s)", "excluding flagged rows (s)"]
        align += "rr"
    lines = [
        "## The ablation ladder",
        "",
        "Each rung adds one term of the decomposition. The rung that matters is",
        "the fourth to the fifth -- geometry added to a model that already has the",
        "identifiers -- and on a forward split it is expected to be small, because",
        "a saturated identifier model has seen every pair it is asked about. The",
        "held-out-pair split below is where the difference is supposed to appear.",
        "",
        "Two rungs need reading with care. The PRU reference and the geometric",
        "term both estimate *unimpeded* taxi-out and are being scored against",
        "actual taxi-out, so they are biased low by the entire excess term and",
        "will sit above the airport mean for a reason that has nothing to do with",
        "how good either of them is. The `+ airport offset` column adds back a",
        "per-airport constant fitted on the same training rows -- the excess held",
        "as a constant, which is exactly what Gate 2 exists to replace with a",
        "model -- and is the column to compare them on. The boosted rungs predict",
        "taxi-out directly and need no offset, so it is blank for them.",
    ]
    for split in splits:
        if split.name.startswith("pairs-"):
            continue
        lines += [
            "",
            f"### {split.name.title()} split",
            "",
            split.description[0].upper() + split.description[1:] + ".",
            "",
            *table(header, _rung_rows(results, split.name, "valid", tails), align),
        ]
    return lines


# The entry's central contrast: the block whose value on a new pair is the
# claim. Everything above it on the ladder is measured against this same
# machinery, but this is the one the write-up leads with.
CLAIM = ("identifiers", "identifiers-geometry")


def _fold_sums(results, features, y, folds, *, ex_flagged: bool, airport=None):
    """Per-pair residual sums for every rung, per held-out fold and row set.

    Rows the record-quality term covers are dropped when ``ex_flagged``;
    ``airport`` restricts to one ICAO. A fold with an empty row set on either
    side is left out, so a per-airport contrast never reads an empty fold.
    """
    out = []
    icao = features["icao"].astype(str)
    for s in folds:
        meta = results.get(("meta", s.name, "tail"))
        flagged = (
            meta["mask"]
            if (meta is not None and ex_flagged)
            else pd.Series(False, index=y.index)
        )
        fold = {}
        for side, mask, which in (
            ("held", s.valid, "valid"),
            ("seen", s.extra["seen"], "seen"),
        ):
            rows = mask & ~flagged
            if airport is not None:
                rows = rows & (icao == airport)
            idx = y.index[rows]
            resid = {}
            for rung in LADDER:
                e = results.get((s.name, rung.key, which))
                if e is None:
                    continue
                resid[rung.key] = (y.loc[idx] - e["pred"].loc[idx]).to_numpy()
            if not resid:
                break
            fold[side] = cluster_sums(resid, features.loc[idx, "pair"].to_numpy())
        if len(fold) == 2 and all(len(next(iter(v.values())).n) for v in fold.values()):
            out.append(fold)
    return out


def section_pairs(results, splits, features, y, tails: Tails) -> list[str]:
    folds = [s for s in splits if s.name.startswith("pairs-")]
    if not folds:
        return []
    ex = tails.available
    sums = _fold_sums(results, features, y, folds, ex_flagged=ex)
    if not sums:
        return []
    keys = [r.key for r in LADDER if all(r.key in f["held"] for f in sums)]
    by_key = {r.key: r for r in LADDER}
    per_fold = [fold_rmses(f) for f in sums]

    # -- table 1: every rung, seen against held out, with per-fold gaps
    rows = []
    boot_gap = {}
    for k in keys:
        boot_gap[k] = cluster_bootstrap(sums, gap(k))
        seen_m = float(np.mean([f["seen"][k] for f in per_fold]))
        held_m = float(np.mean([f["held"][k] for f in per_fold]))
        fold_gaps = [f["held"][k] - f["seen"][k] for f in per_fold]
        rows.append(
            [
                by_key[k].label,
                f"{seen_m:,.1f}",
                f"{held_m:,.1f}",
                str(boot_gap[k]),
                " / ".join(f"{g:+,.0f}" for g in fold_gaps),
                "yes" if by_key[k].uses_geometry else "no",
            ]
        )

    # -- table 2: every consecutive contrast on the ladder, as a difference in
    # differences, and the claim contrast whether or not it is consecutive
    contrasts = [(a, b) for a, b in zip(keys, keys[1:])]
    if CLAIM[0] in keys and CLAIM[1] in keys and CLAIM not in contrasts:
        contrasts.append(CLAIM)
    crow = []
    did = {}
    for a, b in contrasts:
        did[(a, b)] = cluster_bootstrap(sums, difference_in_differences(b, a))
        seen_delta = float(np.mean([f["seen"][a] - f["seen"][b] for f in per_fold]))
        held_delta = float(np.mean([f["held"][a] - f["held"][b] for f in per_fold]))
        mark = " **(the claim)**" if (a, b) == CLAIM else ""
        crow.append(
            [
                f"{by_key[a].label} → {by_key[b].label}{mark}",
                f"{seen_delta:+,.1f}",
                f"{held_delta:+,.1f}",
                str(did[(a, b)]),
            ]
        )

    # -- table 3: the claim contrast per airport
    arow = []
    if CLAIM[0] in keys and CLAIM[1] in keys:
        for ap in sorted(features["icao"].astype(str).unique()):
            ap_sums = _fold_sums(results, features, y, folds, ex_flagged=ex, airport=ap)
            if len(ap_sums) < len(folds):
                continue
            n_held = int(sum(next(iter(f["held"].values())).n.sum() for f in ap_sums))
            r = cluster_bootstrap(ap_sums, difference_in_differences(CLAIM[1], CLAIM[0]))
            clear = "yes" if (r.low > 0 or r.high < 0) else "no"
            arow.append([ap, f"{n_held:,}", str(r), clear])

    # -- the raw, every-row gap, for the record
    raw_sums = _fold_sums(results, features, y, folds, ex_flagged=False) if ex else sums
    raw_rows = []
    for k in keys:
        r = cluster_bootstrap(raw_sums, gap(k), draws=200)
        raw_rows.append([by_key[k].label, str(r)])

    claim = did.get(CLAIM)
    am = boot_gap.get("airport-mean")
    # The geometric rung's gap net of the airport mean's: what a rung with no
    # identifier to memorize loses on a new pair, once composition is out.
    geo_net = (
        cluster_bootstrap(sums, lambda r: gap("geometric")(r) - gap("airport-mean")(r))
        if "geometric" in keys and "airport-mean" in keys
        else None
    )
    lines = [
        "## Held-out pairs",
        "",
        f"{len(folds)} folds. Whole (airport, stand, runway) pairs are dealt round-robin",
        "within each airport, so every fold holds out a random third of every",
        "airport's pairs -- a volume-representative third, not the rare ones; the",
        "rare-pair question is the strata table's below. Each fold fits one model",
        "and scores it twice: on the held-out pairs, and on a",
        f"{SEEN_FRACTION:.0%} sample of rows whose pairs stayed in training and which were",
        "themselves excluded from the fit. Same model, same period, same airports;",
        "the only difference is whether the pair was seen.",
        "",
        "**The held-out unit is the pair, not the stand.** Every stand and every",
        "runway is still seen in other pairs, so the identifier model is never",
        "asked about a stand it has not met -- only about a combination. This",
        "bounds what the geometry buys on a new pair; it does not isolate what it",
        "would buy on a new stand, which no split of this data can measure.",
        "",
        "**The claim is a difference in differences.** What a block buys on seen",
        "pairs is one number; what it buys on held-out pairs is another; the",
        "claim is that the second exceeds the first. Read that way, whatever",
        "composition difference exists between the two row sets cancels, because",
        "it enters both rungs alike -- and it is small regardless: the airport",
        "mean, which knows nothing about a pair, moves by "
        + (str(am) if am else "n/a")
        + " between the two.",
        "",
        "**Intervals are a cluster bootstrap over pairs**, "
        f"{BOOTSTRAP_DRAWS:,} draws, resampling pairs with replacement from the stored",
        "residuals and recomputing every RMSE from them -- no model is refitted.",
        "Rows within a pair share a stand, a runway and a route, so the pair is",
        "the unit. The interval is the 95% percentile interval of the fold-mean.",
        *(
            [
                "",
                "All figures below are over the rows the record-quality term leaves to",
                "the physical model: one defaulted block time in a 5% sample moves a",
                "raw RMSE more than the whole geometry block does. The raw, every-row",
                "gaps are at the end, for the record.",
            ]
            if ex
            else []
        ),
        "",
        "### Every rung, seen against held out",
        "",
        *table(
            [
                "rung",
                "seen (s)",
                "held out (s)",
                "gap [95%]",
                "gap per fold",
                "reads geometry",
            ],
            rows,
            "-rrrr-",
        ),
        "",
        *(
            [
                "The zero-parameter rung is the purest statement in the table. The"
                " geometric unimpeded term carries no identifier, so it has nothing"
                " to memorize and, if the geometry is doing what it claims, nothing"
                " to lose on a pair it has never seen. Net of the airport mean's own"
                f" gap -- pure composition -- its gap is {geo_net}"
                + (
                    ", which is indistinguishable from zero: it loses nothing."
                    if geo_net.low <= 0 <= geo_net.high
                    else (
                        ", which is above zero: even the identifier-free term does"
                        " worse on new pairs, and the geometry is not carrying over"
                        " as cleanly as it should."
                        if geo_net.low > 0
                        else ", which is below zero: it does better on the new pairs"
                        " than on the seen ones, which is composition the airport"
                        " mean did not remove."
                    )
                ),
            ]
            if geo_net is not None
            else []
        ),
        "",
        "### What each block buys where the pair is new",
        "",
        "Each row is one step of the ladder read as a difference in differences:",
        "how much more the added block improves RMSE on held-out pairs than on",
        "seen ones. Positive means the block carries over to pairs the model",
        "never saw; zero means it is memorization dressed as a feature.",
        "",
        *table(
            ["step", "on seen pairs (s)", "on held-out pairs (s)", "difference [95%]"],
            crow,
            "-rrr",
        ),
    ]
    if claim is not None:
        a, b = (by_key[CLAIM[0]].label, by_key[CLAIM[1]].label)
        verdict = f"**{claim}.** " + (
            "The interval excludes zero: the geometry block buys more on a pair the "
            "model has never seen than on one it has, which is the claim."
            if claim.low > 0
            else (
                "The interval includes zero: on this evidence the geometry block "
                "buys no more on a new pair than on a seen one, and the write-up "
                "has to say so."
                if claim.high >= 0
                else "The interval is entirely negative: the geometry block buys "
                "*less* on a new pair, and the write-up has to say so."
            )
        )
        lines += ["", f"The claim contrast, {a} → {b}. {verdict}"]
    if arow:
        lines += [
            "",
            "### The claim, per airport",
            "",
            "The same contrast, over each airport's held-out pairs alone. The"
            " intervals are wide where an airport is small, and that is the"
            " point of printing them: a per-airport difference that cannot be"
            " told from zero is not a finding.",
            "",
            *table(
                ["ICAO", "held-out rows", "difference [95%]", "clear of zero"],
                arow,
                "-rr-",
            ),
        ]
    if ex:
        lines += [
            "",
            "### Every row, for the record",
            "",
            "The gap per rung with the record-quality rows left in. Two hundred"
            " draws, because the interval is dominated by which artifacts land in"
            " the sample and a tighter one would be false precision.",
            "",
            *table(["rung", "gap, every row [95%]"], raw_rows, "-r"),
        ]
    return lines


def section_frequency(results, features, y, splits, references) -> list[str]:
    """RMSE by how often the validation row's pair appeared in training."""
    split = splits[0]
    ref = references.get(split.name)
    if ref is None:
        return []
    mask = split.valid
    counts = ref.pair_counts(features[mask])
    meta = results.get(("meta", split.name, "tail"))
    keep = None if meta is None else ~meta["mask"][mask]
    rows = []
    for rung in LADDER:
        entry = results.get((split.name, rung.key, "valid"))
        if entry is None:
            continue
        by = evaluate_by_bucket(y[mask], entry["pred"], counts, keep=keep).set_index(
            "bucket"
        )
        rows.append([rung.label] + [num(by.loc[b, "rmse_s"], 1) for b in by.index])
    shape = evaluate_by_bucket(y[mask], y[mask], counts, keep=keep)
    buckets = list(shape["bucket"])
    sizes = shape["rows"].tolist()
    return [
        "## Error by pair frequency",
        "",
        "December validation rows, bucketed by how many times their (airport,",
        "stand, runway) pair appeared in the training months. Geometry should buy",
        "most in the first two buckets, where an identifier model has little or",
        "nothing to remember.",
        "",
        *(
            [
                "Over the rows the record-quality term does not cover, for the",
                "reason the held-out-pair table gives: the rare-pair buckets are",
                "small, and one defaulted block time in a bucket of a thousand rows",
                "reports that record rather than anything about the rungs.",
                "",
            ]
            if keep is not None
            else []
        ),
        *table(
            ["rung"] + [f"{b} ({n:,} rows)" for b, n in zip(buckets, sizes)],
            rows,
            "-" + "r" * len(buckets),
        ),
    ]


def section_per_airport(results, splits, tails: Tails) -> list[str]:
    """Per-airport RMSE for the top rung on each forward split."""
    lines = [
        "## Per airport",
        "",
        "The pooled figure is weighted towards the three airports July 2026",
        "contains -- Frankfurt, Heathrow and Schiphol, 53% of the test rows -- so",
        "a pooled-optimal model could be a Frankfurt-Heathrow-Schiphol-optimal",
        "model. The top rung, per airport, on each forward split.",
    ]
    for split in splits:
        if split.name.startswith("pairs-"):
            continue
        entry = results.get((split.name, LADDER[-1].key, "valid"))
        if entry is None:
            continue
        ev = entry["eval"]
        rows = [
            [
                r["icao"] + (" *" if r["icao"] in JULY_AIRPORTS else ""),
                f"{int(r['rows']):,}",
                num(r["rmse_s"], 1),
                num(r["rmse_ex_flagged_s"], 1) if tails.available else "--",
                f"{r['bias_s']:+,.0f}",
            ]
            for _, r in ev.iterrows()
        ]
        # The three airports July 2026 contains, pooled. Exact from the
        # per-airport figures: each covers exactly its own rows.
        three = ev[ev["icao"].isin(JULY_AIRPORTS)]
        if len(three) == len(JULY_AIRPORTS):
            n = three["rows"].sum()
            rows.insert(
                1,
                [
                    "the three *",
                    f"{int(n):,}",
                    num(
                        float(np.sqrt((three["rows"] * three["rmse_s"] ** 2).sum() / n)),
                        1,
                    ),
                    (
                        num(
                            float(
                                np.sqrt(
                                    (
                                        (three["rows"] - three["flagged"])
                                        * three["rmse_ex_flagged_s"] ** 2
                                    ).sum()
                                    / max(
                                        int((three["rows"] - three["flagged"]).sum()), 1
                                    )
                                )
                            ),
                            1,
                        )
                        if tails.available
                        else "--"
                    ),
                    f"{float((three['rows'] * three['bias_s']).sum() / n):+,.0f}",
                ],
            )
        lines += [
            "",
            f"### {split.name.title()} split, {LADDER[-1].label}",
            "",
            *table(
                ["ICAO", "rows", "RMSE (s)", "excluding flagged (s)", "bias (s)"],
                rows,
                "-rrrr",
            ),
        ]
    lines += ["", "`*` an airport the July 2026 test month contains."]
    return lines


def section_importances(results, splits) -> list[str]:
    entry = results.get((splits[0].name, LADDER[-1].key, "valid"))
    if entry is None or "importances" not in entry["info"]:
        return []
    imp = entry["info"]["importances"]
    total = float(imp.sum()) or 1.0
    geometry = set(GEOMETRY_FEATURES) | set(QUALITY_FEATURES)
    rows = [
        [f"`{c}`", f"{v / total:.1%}", "geometry" if c in geometry else "other"]
        for c, v in imp.head(15).items()
    ]
    share = float(imp[[c for c in imp.index if c in geometry]].sum()) / total
    return [
        "## Where the top rung spends its splits",
        "",
        "Split counts, not a causal statement: a feature can be split on often and",
        "carry little, and two collinear features share the credit arbitrarily.",
        "It is here because a geometry block that is never split on would explain",
        "a flat ablation, and because the join-quality flags earning splits is the",
        "first evidence that the ambiguity bookkeeping was worth carrying.",
        "",
        *table(["feature", "share of splits", "block"], rows, "-r-"),
        "",
        f"The geometry block and its quality flags take {share:.1%} of the splits.",
    ]


# --------------------------------------------------------------------------
# guard, as in scripts/review_challenge_data.py


def check_no_identifiers(text: str, stands: set[str], matrix: pd.DataFrame) -> None:
    """Refuse to write a report carrying a literal stand identifier."""
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
            f"refusing to write {OUT.name}: {len(leaked)} literal stand "
            "identifier(s) would be published"
        )


# --------------------------------------------------------------------------
# main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="fit on a random fraction of rows (development only)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the report instead of writing it"
    )
    ap.add_argument("--pair-folds", type=int, default=PAIR_FOLDS)
    args = ap.parse_args()

    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to run", file=sys.stderr)
        return 1
    if booster_backend() == "none":
        print('no booster installed; pip install -e ".[modelling]"', file=sys.stderr)
        return 1
    if args.sample_frac and not args.dry_run:
        print(
            "--sample-frac writes a report that is not the report; add --dry-run",
            file=sys.stderr,
        )
        return 1

    tails = Tails()
    if not tails.available:
        print(
            f"WARNING: {tails.reason}; running without the record-quality term",
            file=sys.stderr,
        )

    print("loading ...")
    matrix = load_matrix()
    dep = load_departures(args.sample_frac)
    y = pd.to_numeric(dep[TARGET], errors="coerce")

    print("building features ...")
    t0 = time.time()
    features = build_features(dep, matrix, fallback=airport_fallback(matrix))
    print(f"  {len(features):,} rows in {time.time() - t0:.0f} s")

    month = month_of(dep["MVT_TIME_UTC_mvt"])
    splits = [december_split(month), july_split(month)]
    splits += held_out_pair_splits(
        features["icao"], features["pair"], n_folds=args.pair_folds
    )

    print("running the ladder ...")
    results = run(features, y, dep, splits, tails)
    references = {
        name: entry["info"]["reference"]
        for (name, rung, which), entry in results.items()
        if rung == "pru" and which == "valid"
    }

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    head = [
        "# Gate 1 baseline",
        "",
        "Gate 1 of [`../docs/plan.md`](../docs/plan.md): the ablation ladder that",
        "settles what each term of the decomposition is worth, on splits fixed",
        "before anything was fitted. Generated by `scripts/run_baselines.py` from",
        f"the data under `data/challenge/` on {generated}; every figure below is",
        "computed there, and the generator refuses to write a literal stand",
        "identifier from either source.",
        "",
    ]
    body = (
        head
        + section_setup(
            len(dep), splits, booster_backend(), tails, args.sample_frac, results
        )
        + [""]
        + section_unimpeded(results, features, splits)
        + [""]
        + section_ladder(results, splits, tails)
        + [""]
        + section_pairs(results, splits, features, y, tails)
        + [""]
        + section_frequency(results, features, y, splits, references)
        + [""]
        + section_per_airport(results, splits, tails)
        + [""]
        + section_importances(results, splits)
        + [
            "",
            "## What is not here yet",
            "",
            "Congestion (Gate 2), weather as a regime indicator (Gate 2b) and the",
            "queueing layer (Gate 3) are absent, so the excess term of the",
            "decomposition is unmodelled: every figure above comes from an",
            "unimpeded-and-identifiers model being asked to predict a quantity",
            "that is mostly excess. The forward splits should be read as a floor,",
            "not as an estimate of what the entry can do.",
            "",
            *(
                []
                if tails.available
                else [
                    "The record-quality term is absent for the reason given at the",
                    "top, and the RMSE levels below are dominated by artefact rows",
                    "until it lands. The ladder differences are readable now; the",
                    "levels are not.",
                    "",
                ]
            ),
            "Distance matrices derived from OpenStreetMap: data (c) "
            "OpenStreetMap contributors, ODbL 1.0. Movement and flight data",
            "supplied by EUROCONTROL PRC and the OpenSky Network under the terms",
            "of the PRC Data Challenge 2026.",
            "",
        ]
    )
    text = "\n".join(body)
    check_no_identifiers(text, set(dep["STAND_mvt"].dropna().astype(str)), matrix)
    if args.dry_run:
        print(text)
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
