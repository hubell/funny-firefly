#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The candidates after the board, C1 to C5, C7 and C8, under the rule written before them.

The candidates are section 3 of ``reports/preregistered-2026-10-04.md`` and
the two its amendment of the same day adds (C7 and C8); the rule that adopts
them is that document's section 4, which keeps the rule of
``reports/preregistered-2026-10-03.md`` unchanged in its legs, its reference,
its rows and its floors. Each candidate is read against the submission rung's
own fit at the ladder's seed, alone; the order of the stages changes no
figure.

This script is ``scripts/candidates.py`` in another directory: it imports
that harness and reuses its machinery (the rule's constants and readings, the
floors read from the reports that print them and compared before anything is
fitted, the reproduction checks, the fingerprints of everything it may not
write, the stage records), and writes only under
``data/processed/candidates_board/``, which git ignores, and to the one report
it generates, ``reports/candidates-2026-10-05.md``.

The candidates, and what is passed to the fitting code:

- **C1**, ``taxiout.aobt3``'s block appended to the rung's features;
- **C2a**, ``taxiout.winter``'s layer on the rung's predictions, no fit;
- **C2b**, only if C2a is adopted (not in this script until it is);
- **C3a**, a LightGBM member under the deviance the modified Park test
  chooses, every other parameter ``BOOSTER_PARAMS``, the target floored at
  1 s, read as the half-and-half blend with the rung at the same seed;
- **C3b**, a CatBoost member through ``taxiout.models.Booster``'s CatBoost
  backend, read the same way;
- **C4**, ``taxiout.flight_history``'s two columns appended, recomputed per
  split from that split's fitting rows;
- **C5**, a per-airport least-squares recalibration of the rung's
  predictions, no fit;
- **C7**, ``taxiout.thunder``'s block appended;
- **C8**, the booster refitted with ``taxiout.regime_weights``'s weights.

**Every fit's predictions are kept per row** in the form
``data/processed/candidates/`` keeps them, and so are the layers' applied
predictions and the blends, each with a JSON record carrying its
``pred_sha256``.

Usage
-----
    python scripts/candidates_board.py --amendment PATH --fit reference
    python scripts/candidates_board.py --amendment PATH --stage 1   # C2a, C5
    python scripts/candidates_board.py --amendment PATH --stage 2   # C1, C3a, C4
    python scripts/candidates_board.py --amendment PATH --stage 4   # C7, C8
    python scripts/candidates_board.py --amendment PATH --stage 5   # C3b
    python scripts/candidates_board.py --amendment PATH --stage 6   # section 4's combination
    python scripts/candidates_board.py --amendment PATH --stage 7   # the amendment's
    python scripts/candidates_board.py --amendment PATH --report

``--amendment`` is the working note of the amendment, which this repository
does not yet carry; its digest is checked before anything is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import candidates as cb  # noqa: E402
import run_baselines as rb  # noqa: E402
from taxiout.aobt3 import AOBT3_FEATURES, aobt3_features  # noqa: E402
from taxiout.aobt3 import REQUIRED_COLUMNS as AOBT3_COLUMNS  # noqa: E402
from taxiout.challenge import load_movements, to_pandas  # noqa: E402
from taxiout.flight_history import FLIGHT, FLIGHT_HISTORY_FEATURES, flight_history  # noqa: E402
from taxiout.metar import load_metar  # noqa: E402
from taxiout.models import LADDER_BY_KEY, SUBMISSION_KEY, Booster  # noqa: E402
from taxiout.provenance import FROZEN_DOCUMENTS, digest, report_inputs_footer  # noqa: E402
from taxiout.regime_weights import regime_weights  # noqa: E402
from taxiout.tails import plan_shift_threshold_s  # noqa: E402
from taxiout.thunder import THUNDER_FEATURES, thunder_by_hour, thunder_features  # noqa: E402
from taxiout.winter import WinterLayer, classes_by_hour, winter_class  # noqa: E402

WORK = REPO / "data" / "processed" / "candidates_board"
REPORT = REPO / "reports" / "candidates-2026-10-05.md"

FORBIDDEN = cb.FORBIDDEN + (REPO / "data" / "processed" / "candidates",)


def point_harness() -> None:
    """Point the harness this one is at this script's records.

    Its record directory and its forbidden list, so that every function of it
    that reads or fingerprints reads these records and guards the earlier
    harness's as well. Done when the script runs, not when it is imported, so
    that importing it (the tests do) leaves ``candidates`` as it was.
    """
    cb.WORK = WORK
    cb.FORBIDDEN = FORBIDDEN


BOARD_TEXT = "reports/preregistered-2026-10-04.md"
SEEDS_REPORT = "reports/monthly_seeds.md"


# --------------------------------------------------------------------------
# the rule's numbers that the rule of 3 October does not already hold


@dataclass(frozen=True)
class BoardRule:
    """Section 3's and the amendment's numbers, each in the words it is fixed in.

    ``tests/test_candidates_board.py`` checks each against the published copy
    of section 3 and against the amendment where it is on hand.
    """

    second_seed: int
    park_gamma_at: float
    tweedie_low: float
    tweedie_high: float
    tweedie_decimals: int
    target_floor_s: float
    catboost: dict
    catboost_gpu_after_s: float
    deadline_utc: str
    board_sha256: str
    amendment_sha256: str
    blend_weight: float


def board_rule() -> BoardRule:
    """The numbers, held in a function for the reason ``cb.the_rule`` gives.

    - "refitted at the second seed": seed 1, the ladder's seed plus one, as
      the rule of 3 October's seed-1 check reads it;
    - "`gamma` if lambda >= 1.75, else `tweedie` with variance power lambda
      clipped to [1.1, 1.9] and rounded to one decimal; the target is floored
      at 1 s for this member";
    - "CatBoost on CPU at eight threads: loss RMSE, 3,000 iterations, learning
      rate 0.08, depth 8, L2 3, 254 borders";
    - "If one fit on the December leg takes more than 45 minutes";
    - "The verdict is written by 7 October 2026, 12:00 UTC";
    - "two blend members at equal weights", "the half-and-half blend".
    """
    return BoardRule(
        second_seed=1,
        park_gamma_at=1.75,
        tweedie_low=1.1,
        tweedie_high=1.9,
        tweedie_decimals=1,
        target_floor_s=1.0,
        catboost={
            "loss_function": "RMSE",
            "iterations": 3000,
            "learning_rate": 0.08,
            "depth": 8,
            "l2_leaf_reg": 3,
            "border_count": 254,
            "thread_count": 8,
        },
        catboost_gpu_after_s=45 * 60.0,
        deadline_utc="2026-10-07T12:00:00+00:00",
        board_sha256="568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990",
        amendment_sha256="4ac17be6925d03255a2e52b1d1ee307e297e913ee42f33c063a03d8e36fbe8cd",
        blend_weight=0.5,
    )


# --------------------------------------------------------------------------
# the candidates


@dataclass(frozen=True)
class Member:
    """A fitted variation of the submission rung."""

    key: str
    blocks: tuple[str, ...] = ()
    objective: str | None = None
    tweedie_power: float | None = None
    # ``None`` for "no", so that no module-level literal reaches the register's scanner
    floor_target: bool | None = None
    backend: str = "lightgbm"
    weighted: bool | None = None
    gpu: bool | None = None


@dataclass(frozen=True)
class Candidate:
    """How a candidate's predictions are formed from fits and layers."""

    key: str
    label: str
    kind: str  # "member", "blend", "layer" or "combination"
    member: Member | None = None
    layer: str | None = None  # "winter" or "recal"
    layers: tuple[str, ...] = ()  # a combination's, in section 3's order
    members: tuple[Member, ...] = ()  # a combination's blend members


BLOCK_COLUMNS = {
    "aobt3": AOBT3_FEATURES,
    "flight": FLIGHT_HISTORY_FEATURES,
    "thunder": THUNDER_FEATURES,
}


def candidates(c3a: Member | None = None) -> dict[str, Candidate]:
    """The candidates in section 4's order of cost, the amendment's two before C3b.

    ``c3a`` is the C3a member once the Park test has chosen its objective;
    without it the candidate is listed with no member and cannot be fitted.
    """
    return {
        "c2a": Candidate("c2a", "C2a, the winter layer", "layer", layer="winter"),
        "c5": Candidate("c5", "C5, per-airport recalibration", "layer", layer="recal"),
        "c1": Candidate(
            "c1", "C1, `AOBT_3_flt` and its queue", "member", Member("c1", ("aobt3",))
        ),
        "c3a": Candidate(
            "c3a", "C3a, the variance-matched LightGBM member", "blend", c3a
        ),
        "c4": Candidate(
            "c4", "C4, the flight's own history", "member", Member("c4", ("flight",))
        ),
        "c7": Candidate(
            "c7", "C7, thunderstorms from the METAR", "member", Member("c7", ("thunder",))
        ),
        "c8": Candidate(
            "c8",
            "C8, the training rows weighted toward the target month",
            "member",
            Member("c8", weighted=True),
        ),
        "c3b": Candidate(
            "c3b",
            "C3b, the CatBoost member",
            "blend",
            Member("c3b_gpu", backend="catboost", gpu=True)
            if c3b_path().get("path") == "gpu"
            else Member("c3b_member", backend="catboost"),
        ),
    }


def stages() -> dict[int, tuple[str, ...]]:
    """Section 4's order by cost, the amendment's two placed before the costliest.

    A function and not a module-level table for the reason ``board_rule`` is.
    """
    return {
        1: ("c2a", "c5"),
        2: ("c1", "c3a", "c4"),
        4: ("c7", "c8"),
        5: ("c3b",),
    }


# --------------------------------------------------------------------------
# the frame, with the candidate blocks


class BoardFrame(cb.Frame):
    """``candidates.Frame`` with the columns the candidates read beside it.

    The departures' extra columns (``AOBT_3_flt`` for C1's module alone, the
    planning times it is differenced against, ``FLIGHT_mvt`` for C4) are read
    from the challenge files in the order ``load_departures`` reads them and
    checked against it by movement identifier before anything uses them.
    """

    def __init__(self) -> None:
        super().__init__()
        t0 = time.monotonic()
        wanted = sorted(set(AOBT3_COLUMNS) | {FLIGHT, "MVT_ID_mvt"})
        extra = to_pandas(load_movements(rb.DATA, wanted, phase="DEP")).reset_index(
            drop=True
        )
        same = len(extra) == len(self.dep) and np.array_equal(
            extra["MVT_ID_mvt"].to_numpy(),
            self.dep["MVT_ID_mvt"].to_numpy(),
            equal_nan=True,
        )
        if not same:
            raise RuntimeError(
                "the extra columns are not of the departures' rows in order"
            )
        self.extra = extra
        metar = load_metar(rb.METAR)
        self.metar_rows = len(metar)
        self.winter_hourly = classes_by_hour(metar)
        self.thunder_hourly = thunder_by_hour(metar)
        self.blocks: dict[str, pd.DataFrame] = {}
        self.extra_seconds = time.monotonic() - t0
        print(f"extra columns and METAR tables in {self.extra_seconds:.0f} s", flush=True)

    def block(self, name: str, prep: dict | None = None) -> pd.DataFrame:
        """One candidate block for every departure; the flight history is per split."""
        if name == "flight":
            if prep is None:
                raise ValueError("the flight history is computed per split")
            if "flight" not in prep:
                t0 = time.monotonic()
                prep["flight"] = flight_history(
                    prep["frame"], self.extra[FLIGHT], self.y, prep["rows"]
                )
                missing = int(prep["flight"]["fn_mean"].isna().sum())
                print(
                    f"  flight history in {time.monotonic() - t0:.0f} s,"
                    f" {missing:,} rows without a value",
                    flush=True,
                )
            return prep["flight"]
        if name not in self.blocks:
            t0 = time.monotonic()
            if name == "aobt3":
                got = aobt3_features(self.extra)
            elif name == "thunder":
                got = thunder_features(self.dep, self.thunder_hourly)
            else:
                raise KeyError(name)
            got.index = self.features.index
            self.blocks[name] = got
            print(f"  block {name} in {time.monotonic() - t0:.0f} s", flush=True)
        return self.blocks[name]


def classes_of(kept: pd.DataFrame, hourly: pd.DataFrame) -> np.ndarray:
    """The winter class of each kept row, from its airport and take-off."""
    frame = pd.DataFrame(
        {"ADEP_mvt": kept["icao"].to_numpy(), "MVT_TIME_UTC_mvt": kept["takeoff"]}
    )
    return winter_class(frame, hourly).to_numpy()


# --------------------------------------------------------------------------
# one fit


def member_columns(member: Member) -> tuple[str, ...]:
    base = tuple(LADDER_BY_KEY[SUBMISSION_KEY].features)
    extra: tuple[str, ...] = ()
    for b in member.blocks:
        extra += BLOCK_COLUMNS[b]
    overlap = sorted(set(extra) & set(base))
    if overlap:
        raise RuntimeError(f"the rung already reads {overlap}")
    return base + extra


def member_params(member: Member, seed: int) -> dict:
    params = cb.params_of(cb.Arm(member.key), seed)
    if member.objective == "gamma":
        params["objective"] = "gamma"
    elif member.objective == "tweedie":
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = float(member.tweedie_power)
    elif member.objective is not None:
        raise ValueError(member.objective)
    if member.backend == "catboost":
        c = board_rule().catboost
        params["n_estimators"] = int(c["iterations"])
        params["learning_rate"] = float(c["learning_rate"])
    return params


def kept_frame(frame: BoardFrame, prep: dict, pred: np.ndarray) -> pd.DataFrame:
    mask = prep["valid"]
    raw = frame.dep.loc[mask]
    return pd.DataFrame(
        {
            "mvt_id": raw["MVT_ID_mvt"].to_numpy(),
            "icao": raw["ADEP_mvt"].astype(str).to_numpy(),
            "runway": raw["RUNWAY_mvt"].astype(str).to_numpy(),
            "takeoff": pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).to_numpy(),
            "term": np.asarray(prep["flagged"][mask], dtype=bool),
            "shift": np.asarray(prep["shift"][mask], dtype=bool),
            "y": np.asarray(frame.y[mask], dtype=float),
            "pred": pred,
            "with_term": cb.combine(pred, prep["tail"][mask]).to_numpy(dtype=float),
        }
    )


def fit_member(
    frame: BoardFrame, prep: dict, member: Member, seed: int
) -> tuple[dict, pd.DataFrame]:
    """One fit of the submission rung under one member, scored on the split's month."""
    split = prep["split"]
    columns = member_columns(member)
    x = prep["frame"]
    if member.blocks:
        x = x.copy(deep=False)
        for b in member.blocks:
            got = frame.block(b, prep)
            for c in got.columns:
                x[c] = got[c]
    missing = [c for c in columns if c not in x.columns]
    if missing:
        raise RuntimeError(f"the frame lacks {missing}")
    params = member_params(member, seed)
    rows = prep["rows"]
    y_fit = frame.y[rows]
    if member.floor_target:
        y_fit = np.maximum(
            pd.to_numeric(y_fit, errors="coerce"), board_rule().target_floor_s
        )
    weight, weights = None, None
    if member.weighted:
        t0 = time.monotonic()
        got = regime_weights(x, frame.dep["MVT_TIME_UTC_mvt"], rows, prep["valid"])
        weight = got.weights
        weights = {**got.distribution(), "classifier_s": time.monotonic() - t0}
        print(
            "  weights before the booster: min {min:.3f}, quartiles {q1:.3f} / {median:.3f} /"
            " {q3:.3f}, max {max:.3f}, clipped {share_clipped:.4%} ({clipped_low:,} low,"
            " {clipped_high:,} high), {classifier_s:.0f} s".format(**weights),
            flush=True,
        )
    model = Booster(columns, params)
    backend_params = None
    t0 = time.monotonic()
    if member.backend == "catboost":
        model.backend = "catboost"
        with pinned_catboost(member.gpu):
            model.fit(x.loc[rows], y_fit, sample_weight=weight)
        trained = model.model.get_all_params()
        backend_params = {k: trained.get(k) for k in board_rule().catboost}
        backend_params["thread_count"] = model.model.get_params().get("thread_count")
    else:
        model.fit(x.loc[rows], y_fit, sample_weight=weight)
    fit_s = time.monotonic() - t0
    t1 = time.monotonic()
    pred = np.asarray(model.predict(x.loc[prep["valid"]]), dtype=float)
    predict_s = time.monotonic() - t1
    kept = kept_frame(frame, prep, pred)
    record = {
        "name": cb.record_name(split.name, member.key, seed),
        "split": split.name,
        "train_months": list(split.train_months),
        "valid_month": split.valid_month,
        "arm": {"key": member.key},
        "member": asdict(member),
        "seed": int(seed),
        "random_state": int(params["random_state"]),
        "n_estimators": int(params["n_estimators"]),
        "objective": str(params["objective"]) if member.backend == "lightgbm" else "RMSE",
        "backend": member.backend,
        "backend_params": backend_params,
        "columns": len(columns),
        "fit_rows": int(rows.sum()),
        "weight_sum": None if weight is None else float(np.sum(weight)),
        "weights": weights,
        "valid_rows": int(prep["valid"].sum()),
        "prepare_s": float(prep["seconds"]),
        "fit_s": fit_s,
        "predict_s": predict_s,
        "peak_rss_gib": cb.peak_gib(),
        "pred_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    del model
    return record, kept


def same_parameter(got, want) -> bool:
    """Equal, or for a float equal at the single precision CatBoost stores it in."""
    if isinstance(want, float) and isinstance(got, float):
        return abs(got - want) <= 1e-6 * abs(want)
    return got == want


class pinned_catboost:  # noqa: N801
    """Section 3's L2, borders and threads, pinned on CatBoost's constructor for one fit.

    ``taxiout.models.Booster``'s CatBoost backend builds the regressor with
    the loss, depth, iterations, learning rate and seed and leaves the rest at
    CatBoost's defaults. Section 3 names three of those defaults explicitly,
    so for the duration of a C3b fit the constructor the backend calls is
    wrapped to pass them, and the backend's own code runs unchanged.
    """

    def __init__(self, gpu: bool | None = None) -> None:
        self.gpu = bool(gpu)

    def __enter__(self):
        import catboost

        self.module = catboost
        self.original = catboost.CatBoostRegressor
        pinned = {
            k: v
            for k, v in board_rule().catboost.items()
            if k in ("l2_leaf_reg", "border_count", "thread_count")
        }
        if self.gpu:
            pinned.update(task_type="GPU", devices="0")
        original = self.original

        class Pinned(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **{**kwargs, **pinned})

        catboost.CatBoostRegressor = Pinned
        return self

    def __exit__(self, *exc):
        self.module.CatBoostRegressor = self.original
        return False


def fit_and_record(
    frame: BoardFrame, prep: dict, member: Member, seed: int, stage: dict
) -> None:
    name = cb.record_name(prep["split"].name, member.key, seed)
    if cb.record_path(name).exists():
        print(f"  {name}: on disk, not refitted", flush=True)
        return
    print(f"\nfitting {name} ({datetime.now(timezone.utc):%H:%M:%S}Z)", flush=True)
    record, kept = fit_member(frame, prep, member, seed)
    if member.backend == "catboost":
        want = board_rule().catboost
        got = record["backend_params"]
        if any(not same_parameter(got.get(k), v) for k, v in want.items()):
            raise SystemExit(f"STOPPED: the CatBoost fit ran with {got}, not {want}")
    cb.write_record(record, kept)
    stage["fits"].append(name)
    rows = cb.booster_rows(kept)
    print(
        f"  fit {record['fit_s']:.0f} s, predict {record['predict_s']:.0f} s, "
        f"{rows.sum():,} rows read at {cb.rmse((kept['y'] - kept['pred']).to_numpy()[rows]):.2f} s, "
        f"peak {record['peak_rss_gib']:.1f} GiB",
        flush=True,
    )


def write_derived(name: str, kept: pd.DataFrame, derived: dict) -> None:
    """A prediction formed from stored ones (a layer, a blend, a stored base), as a record."""
    pred = kept["pred"].to_numpy(dtype=float)
    split, arm, seed = name.split("__")
    record = {
        "name": name,
        "split": split,
        "arm": {"key": arm},
        "seed": int(seed.removeprefix("seed")),
        "derived": derived,
        "valid_rows": int(len(kept)),
        "pred_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cb.write_record(record, kept)


def replace_pred(base: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """``base`` with another prediction; the term's rows keep the term's value."""
    out = base.copy()
    out["pred"] = np.asarray(pred, dtype=float)
    term = out["term"].to_numpy(dtype=bool)
    out["with_term"] = np.where(
        term, base["with_term"].to_numpy(dtype=float), out["pred"]
    )
    return out


# --------------------------------------------------------------------------
# the stored out-of-fold predictions of the submission rung


def base_name(month: str) -> str:
    return cb.record_name(f"heldout-{month}", "base", 0)


def write_bases(frame: BoardFrame) -> None:
    """Each held-out month's stored seed-0 fit as a kept frame, the plan-shift rule beside it.

    The stored files hold ``y - pred`` and the term's flag in the order of the
    departures; the plan-shift rule is fitted here on the split's training
    rows exactly as ``candidates.coverage`` fits it, and the term it fits must
    flag the rows the stored file flags.
    """
    for split in cb.month_plan():
        name = base_name(split.valid_month)
        if cb.record_path(name).exists():
            continue
        train = frame.month.isin(split.train_months)
        valid = frame.month == split.valid_month
        covered = cb.coverage(frame.dep, train, valid)
        stored = pd.read_parquet(
            REPO / cb.MONTHLY / f"heldout-{split.valid_month}_{SUBMISSION_KEY}.parquet"
        )
        raw = frame.dep.loc[valid]
        same = (
            len(stored) == len(raw)
            and np.array_equal(stored["mvt_id"].to_numpy(), raw["MVT_ID_mvt"].to_numpy())
            and np.array_equal(
                stored["flagged"].to_numpy(dtype=bool),
                covered["flagged"][valid].to_numpy(dtype=bool),
            )
        )
        if not same:
            raise SystemExit(
                f"STOPPED: {split.name}: the stored file is not of the same rows and term"
            )
        y = np.asarray(frame.y[valid], dtype=float)
        kept = pd.DataFrame(
            {
                "mvt_id": raw["MVT_ID_mvt"].to_numpy(),
                "icao": raw["ADEP_mvt"].astype(str).to_numpy(),
                "runway": raw["RUNWAY_mvt"].astype(str).to_numpy(),
                "takeoff": pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).to_numpy(),
                "term": covered["flagged"][valid].to_numpy(dtype=bool),
                "shift": covered["shift"][valid].to_numpy(dtype=bool),
                "y": y,
                "pred": y - stored["residual"].to_numpy(dtype=float),
                "with_term": y - stored["residual_with_term"].to_numpy(dtype=float),
            }
        )
        write_derived(
            name,
            kept,
            {
                "from": f"{cb.MONTHLY}/heldout-{split.valid_month}_{SUBMISSION_KEY}.parquet"
            },
        )
        print(
            f"  {name}: {len(kept):,} rows, {int(kept['shift'].sum())} plan-shift",
            flush=True,
        )


def oof_rows(kept: pd.DataFrame) -> np.ndarray:
    """The rows a layer or the Park test is fitted on: neither rule's, 4 h or less."""
    rows = cb.booster_rows(kept)
    y = kept["y"].to_numpy(dtype=float)
    return rows & np.isfinite(y) & (y <= plan_shift_threshold_s())


def oof(months: list[str]) -> pd.DataFrame:
    """The stored out-of-fold rows of ``months``, pooled, restricted to ``oof_rows``."""
    parts = []
    for m in months:
        kept = cb.read_kept(base_name(m))
        parts.append(kept.loc[oof_rows(kept)])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# the two layers and the Park test


def fit_layer(kind: str, rows: pd.DataFrame, hourly: pd.DataFrame):
    residual = (rows["y"] - rows["pred"]).to_numpy(dtype=float)
    if kind == "winter":
        return WinterLayer().fit(
            rows["icao"].to_numpy(), classes_of(rows, hourly), residual
        )
    if kind == "recal":
        out = {}
        for a, g in rows.groupby("icao", sort=True):
            p = g["pred"].to_numpy(dtype=float)
            y = g["y"].to_numpy(dtype=float)
            beta = float(
                np.sum((p - p.mean()) * (y - y.mean())) / np.sum((p - p.mean()) ** 2)
            )
            out[a] = (float(y.mean() - beta * p.mean()), beta)
        return out
    raise ValueError(kind)


def apply_layer(kind: str, layer, kept: pd.DataFrame, hourly: pd.DataFrame) -> np.ndarray:
    pred = kept["pred"].to_numpy(dtype=float)
    if kind == "winter":
        return layer.apply(pred, kept["icao"].to_numpy(), classes_of(kept, hourly))
    if kind == "recal":
        icao = kept["icao"].to_numpy()
        alpha = np.array([layer.get(a, (0.0, 1.0))[0] for a in icao])
        beta = np.array([layer.get(a, (0.0, 1.0))[1] for a in icao])
        return alpha + beta * pred
    raise ValueError(kind)


def layer_summary(kind: str, layer) -> dict:
    if kind == "winter":
        return {
            "by_class": layer.by_class,
            "cells": {f"{a}|{c}": v for (a, c), v in layer.by_cell.items()},
        }
    return {a: {"alpha": v[0], "beta": v[1]} for a, v in layer.items()}


def months_before(leg: cb.Split) -> list[str]:
    """The held-out months strictly before the leg's month."""
    return [m for m in cb.MONTHS if m < leg.valid_month]


def park_test() -> dict:
    """Section 3's modified Park test over the rung's twelve out-of-fold months.

    The slope of log squared residual on log prediction, by least squares,
    over the rows neither rule covers, of 4 h or less, with a prediction above
    zero. A residual of exactly zero has no logarithm, and is counted.
    """
    rows = oof(list(cb.MONTHS))
    rows = rows.loc[rows["pred"].to_numpy(dtype=float) > 0]
    r = (rows["y"] - rows["pred"]).to_numpy(dtype=float)
    zero = int((r == 0).sum())
    if zero:
        raise SystemExit(
            f"STOPPED: {zero} residuals of exactly zero in the Park test's rows"
        )
    lx = np.log(rows["pred"].to_numpy(dtype=float))
    ly = np.log(r * r)
    lam = float(
        np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2)
    )
    objective, power = choose_objective(lam)
    return {"rows": int(len(rows)), "lambda": lam, "objective": objective, "power": power}


def choose_objective(lam: float) -> tuple[str, float | None]:
    """``gamma`` at or above the threshold, else ``tweedie`` at lambda clipped and rounded."""
    rule = board_rule()
    if lam >= rule.park_gamma_at:
        return "gamma", None
    power = min(max(lam, rule.tweedie_low), rule.tweedie_high)
    return "tweedie", float(f"{power:.{rule.tweedie_decimals}f}")


def c3b_path() -> dict:
    """Which path C3b took, once its December fit on CPU has been timed."""
    path = WORK / "c3b_path.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def decide_c3b_path(frame: BoardFrame, stage: dict) -> dict:
    """Section 3: the December leg on CPU first, timed; over 45 minutes, the GPU path.

    The GPU path is taken only if two GPU fits of that leg give bit-identical
    predictions; otherwise C3b stops, and the record says why.
    """
    got = c3b_path()
    if got:
        return got
    leg = cb.leg_plan()[0]
    prep = frame.prepare(leg)
    cpu = Member("c3b_member", backend="catboost")
    fit_and_record(frame, prep, cpu, 0, stage)
    rec = json.loads(cb.record_path(cb.record_name(leg.name, cpu.key, 0)).read_text())
    limit = board_rule().catboost_gpu_after_s
    out = {"cpu_december_fit_s": rec["fit_s"], "limit_s": limit}
    if rec["fit_s"] <= limit:
        out.update(path="cpu", why="the December fit on CPU took at most 45 minutes")
    else:
        names = []
        try:
            for key in ("c3b_gpu", "c3b_gpu_check"):
                fit_and_record(
                    frame, prep, Member(key, backend="catboost", gpu=True), 0, stage
                )
                names.append(cb.record_name(leg.name, key, 0))
            shas = [
                json.loads(cb.record_path(n).read_text())["pred_sha256"] for n in names
            ]
            out["gpu_sha256"] = shas
            if shas[0] == shas[1]:
                out.update(
                    path="gpu", why="over 45 minutes on CPU; two GPU fits bit-identical"
                )
            else:
                out.update(
                    path="stopped",
                    why="over 45 minutes on CPU, and two GPU fits of December differ",
                )
        except Exception as exc:  # the GPU path failing is a reason to stop, recorded
            out.update(
                path="stopped", why=f"over 45 minutes on CPU; the GPU fit failed: {exc!r}"
            )
    del prep
    (WORK / "c3b_path.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"C3b path: {out}", flush=True)
    return out


def c3a_member() -> Member | None:
    path = WORK / "c3a_park.json"
    if not path.exists():
        return None
    got = json.loads(path.read_text(encoding="utf-8"))
    return Member(
        "c3a_member",
        objective=got["objective"],
        tweedie_power=got["power"],
        floor_target=True,
    )


# --------------------------------------------------------------------------
# forming each candidate's predictions on a split


def reference_kept(split: str, seed: int) -> pd.DataFrame | None:
    """The rung's own predictions on ``split``: a fitted leg, or a stored held-out month."""
    if split.startswith("heldout-"):
        name = base_name(split.removeprefix("heldout-")) if seed == 0 else None
    else:
        name = cb.record_name(split, "reference", seed)
    if name is None or not cb.record_path(name).exists():
        return None
    return cb.read_kept(name)


def form(c: Candidate, split: str, seed: int, hourly: pd.DataFrame) -> bool:
    """Write the candidate's prediction on one split from what is on disk; whether it could."""
    name = cb.record_name(split, c.key, seed)
    if cb.record_path(name).exists():
        return True
    ref = reference_kept(split, seed)
    if ref is None:
        return False
    if c.kind == "member":
        return False  # the fit itself is the record, under the candidate's own name
    if c.kind == "combination":
        return form_combination(c, split, seed, ref, hourly)
    if c.kind == "blend":
        mname = cb.record_name(split, c.member.key, seed)
        if not cb.record_path(mname).exists():
            return False
        member = cb.read_kept(mname)
        cb.aligned(member, ref)
        w = board_rule().blend_weight
        pred = w * ref["pred"].to_numpy(dtype=float) + (1.0 - w) * member[
            "pred"
        ].to_numpy(dtype=float)
        write_derived(
            name,
            replace_pred(ref, pred),
            {
                "blend": [
                    cb.record_name(split, "reference", seed)
                    if not split.startswith("heldout-")
                    else base_name(split.removeprefix("heldout-")),
                    mname,
                ],
                "weights": [w, 1.0 - w],
            },
        )
        return True
    # a layer: on a held-out month, fitted on the other eleven; on a leg, on the
    # held-out months strictly before the leg's month, at either seed
    if split.startswith("heldout-"):
        month = split.removeprefix("heldout-")
        fitted_on = [m for m in cb.MONTHS if m != month]
    else:
        leg = next(s for s in cb.leg_plan() if s.name == split)
        fitted_on = months_before(leg)
    layer = fit_layer(c.layer, oof(fitted_on), hourly)
    pred = apply_layer(c.layer, layer, ref, hourly)
    write_derived(
        name,
        replace_pred(ref, pred),
        {
            "layer": c.layer,
            "applied_to": cb.record_name(split, "reference", seed)
            if not split.startswith("heldout-")
            else base_name(split.removeprefix("heldout-")),
            "fitted_on_months": fitted_on,
            "fitted_rows": int(len(oof(fitted_on))),
            "fitted": layer_summary(c.layer, layer),
        },
    )
    return True


def layer_months(split: str) -> list[str]:
    """The held-out months a layer is fitted on for ``split``, as section 4 reads them."""
    if split.startswith("heldout-"):
        month = split.removeprefix("heldout-")
        return [m for m in cb.MONTHS if m != month]
    leg = next(s for s in cb.leg_plan() if s.name == split)
    return months_before(leg)


def form_combination(
    c: Candidate, split: str, seed: int, ref: pd.DataFrame, hourly: pd.DataFrame
) -> bool:
    """Section 4's combination: the features together, then the layers, then the blend.

    The booster is the combination's own fit where it adds features or
    weights, else the rung itself; each layer is fitted as section 3 defines
    it, on the rung's out-of-fold rows, and applied in section 3's order to
    the booster's prediction; the blend then gives the booster and every
    member equal weights.
    """
    name = cb.record_name(split, c.key, seed)
    parts = {}
    if c.member is not None:
        bname = cb.record_name(split, c.member.key, seed)
        if not cb.record_path(bname).exists():
            return False
        booster = cb.read_kept(bname)
        cb.aligned(booster, ref)
        parts["booster"] = bname
    else:
        booster = ref
        parts["booster"] = "the rung"
    pred = booster["pred"].to_numpy(dtype=float)
    applied = []
    for kind in c.layers:
        months = layer_months(split)
        layer = fit_layer(kind, oof(months), hourly)
        pred = apply_layer(kind, layer, replace_pred(booster, pred), hourly)
        applied.append({"layer": kind, "fitted_on_months": months})
    members = [pred]
    names = []
    for m in c.members:
        mname = cb.record_name(split, m.key, seed)
        if not cb.record_path(mname).exists():
            return False
        kept = cb.read_kept(mname)
        cb.aligned(kept, ref)
        members.append(kept["pred"].to_numpy(dtype=float))
        names.append(mname)
    pred = np.mean(np.vstack(members), axis=0)
    write_derived(
        name,
        replace_pred(ref, pred),
        {**parts, "layers": applied, "members": names, "weights": "equal"},
    )
    return True


def combination(keys: list[str], cands: dict, key: str, label: str) -> Candidate:
    """The candidates ``keys`` combined as section 4 combines them."""
    blocks: tuple[str, ...] = ()
    weighted = False
    layers, members = [], []
    for k in keys:
        c = cands[k]
        if c.kind == "member":
            blocks += c.member.blocks
            weighted = weighted or bool(c.member.weighted)
        elif c.kind == "layer":
            layers.append(c.layer)
        elif c.kind == "blend":
            members.append(c.member)
    order = {"winter": 0, "recal": 1}
    booster = (
        Member(f"{key}_booster", blocks=blocks, weighted=True if weighted else None)
        if blocks or weighted
        else None
    )
    return Candidate(
        key,
        label,
        "combination",
        booster,
        layers=tuple(sorted(layers, key=order.get)),
        members=tuple(members),
    )


# --------------------------------------------------------------------------
# the readings


def arm_kept(c: Candidate, split: str, seed: int) -> pd.DataFrame | None:
    name = cb.record_name(split, c.key, seed)
    return cb.read_kept(name) if cb.record_path(name).exists() else None


def measure_candidate(c: Candidate) -> dict:
    """Every contrast the rule reads for one candidate, from the kept predictions."""
    out: dict = {}
    for seed in (0, board_rule().second_seed):
        for leg in cb.leg_plan():
            ref = reference_kept(leg.name, seed)
            arm = arm_kept(c, leg.name, seed)
            if ref is None or arm is None:
                continue
            slot = out.setdefault(f"seed{seed}", {})
            slot[f"{leg.name}:all"] = cb.reading(arm, ref, "all")
            slot[f"{leg.name}:three"] = cb.reading(arm, ref, "three")
    arms, bases, months = [], [], {}
    for split in cb.month_plan():
        arm = arm_kept(c, split.name, 0)
        if arm is None:
            return out
        base = reference_kept(split.name, 0)
        months[split.valid_month] = cb.reading(arm, base, "all")
        arms.append(arm)
        bases.append(base)
    out.setdefault("seed0", {})["year:all"] = {
        **cb.reading(
            pd.concat(arms, ignore_index=True), pd.concat(bases, ignore_index=True), "all"
        ),
        "months": months,
    }
    return out


def verdict_of(m: dict) -> dict:
    """The five conditions at seed 0, the sign check at the second seed, and the status."""
    rule = cb.the_rule()
    v = cb.condition_verdicts(m, rule)
    s = f"seed{board_rule().second_seed}"
    signs = {}
    for leg in ("december", "july"):
        r0 = m.get("seed0", {}).get(f"{leg}:all")
        r1 = m.get(s, {}).get(f"{leg}:all")
        if r0 is not None and r1 is not None:
            signs[leg] = cb.same_sign(
                r0["rows"]["contrast"], r1["rows"]["contrast"], rule.contrast_decimals
            )
    if any(v[k] is None for k in cb.FOUR):
        status = "legs not measured"
    elif not cb.clears(v, cb.FOUR):
        status = "fails the four"
    elif v["year"] is None:
        status = "months not measured"
    elif not v["year"]:
        status = "fails the fifth"
    elif len(signs) < 2:
        status = "second seed not measured"
    elif not all(signs.values()):
        status = "fails the sign check"
    else:
        status = "clears"
    return {"conditions": v, "signs": signs, "status": status}


# --------------------------------------------------------------------------
# the stages


def preflight(checks: cb.Checks, amendment: Path) -> dict:
    """The copies and the amendment at their digests, the floors, the directory."""
    rule, board = cb.the_rule(), board_rule()
    print("\nbefore anything is fitted", flush=True)
    cb.frozen_copies(checks)
    held = {f.path: f.sha256 for f in FROZEN_DOCUMENTS}
    got = digest(REPO / BOARD_TEXT)
    checks.add(
        f"`{BOARD_TEXT}` at the digest `taxiout.provenance.FROZEN_DOCUMENTS` holds",
        board.board_sha256,
        got,
        got == board.board_sha256 and held.get(BOARD_TEXT) == board.board_sha256,
    )
    got = digest(amendment)
    checks.add(
        "the amendment's working note at the digest it was fixed by",
        board.amendment_sha256,
        got,
        got == board.amendment_sha256,
    )
    floors = cb.floor_checks(checks, rule, REPO / SEEDS_REPORT)
    checks.add(
        "the records' directory is ignored by git",
        "ignored",
        "ignored" if cb.ignored(WORK) else "not ignored",
        cb.ignored(WORK),
    )
    return floors


def begin(name: str, amendment: Path):
    before = cb.fingerprint()
    stage = cb.stage_start(name)
    checks = cb.Checks()
    floors = preflight(checks, amendment)
    return before, stage, checks, floors


def run_reference(amendment: Path) -> int:
    """Step 3: the reference reproduces what the ladder stored, before anything is read."""
    before, stage, checks, floors = begin("reference", amendment)
    if checks.failed:
        return cb.stage_end(stage, checks, before, {"floors": floors})
    frame = BoardFrame()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "stands.json").write_text(json.dumps(frame.stands()), encoding="utf-8")
    ref = Member("reference")
    counts = {}
    for leg in cb.leg_plan():
        prep = frame.prepare(leg)
        counts[leg.name] = frame.counts(prep)
        fit_and_record(frame, prep, ref, 0, stage)
        del prep
    if not cb.reproduction_checks(
        checks, cb.read_records(), cb.the_rule(), floors["october_printed"]
    ):
        return cb.stage_end(stage, checks, before, {"floors": floors, "counts": counts})
    month = next(s for s in cb.month_plan() if s.valid_month == cb.check_month())
    prep = frame.prepare(month)
    fit_and_record(frame, prep, ref, 0, stage)
    del prep
    if not cb.month_check(checks, cb.read_records()):
        return cb.stage_end(stage, checks, before, {"floors": floors, "counts": counts})
    print("\nthe stored held-out months, as kept frames", flush=True)
    write_bases(frame)
    seed = board_rule().second_seed
    for leg in cb.leg_plan()[:2]:
        prep = frame.prepare(leg)
        fit_and_record(frame, prep, ref, seed, stage)
        del prep
        if leg.name == "december":
            cb.seed1_check(checks, cb.read_records())
    return cb.stage_end(
        stage,
        checks,
        before,
        {"floors": floors, "counts": counts, "frame_build_s": frame.build_seconds},
    )


def reference_passed() -> bool:
    got = cb.read_stage("reference")
    if got is None:
        return False
    latest = {k: v for k, v in got.items() if k != "previous"}
    return (
        bool(latest.get("checks"))
        and all(r["ok"] for r in latest["checks"])
        and not latest["fingerprint"]["touched"]
    )


def run_stage(number: int, amendment: Path) -> int:
    """One of section 4's stages for the candidates it names, as far as the rule reaches."""
    if not reference_passed():
        raise SystemExit("the reference stage has not passed its checks; nothing is read")
    if number in (3, 5, 6, 7) and not on_time(number):
        return 0
    before, stage, checks, floors = begin(f"stage{number}", amendment)
    if checks.failed:
        return cb.stage_end(stage, checks, before, {"floors": floors})
    frame = BoardFrame() if number != 1 else None
    hourly = frame.winter_hourly if frame is not None else winter_hourly()
    if number == 2 and c3a_member() is None:
        park = park_test()
        (WORK / "c3a_park.json").write_text(json.dumps(park, indent=1), encoding="utf-8")
        print(
            f"\nC3a: the modified Park test over {park['rows']:,} rows gives lambda"
            f" {park['lambda']:.4f}, so the objective is {park['objective']}"
            + (
                f" with variance power {park['power']}"
                if park["power"] is not None
                else ""
            ),
            flush=True,
        )
    if number == 5 and decide_c3b_path(frame, stage)["path"] == "stopped":
        return cb.stage_end(stage, checks, before, {"floors": floors, "c3b": c3b_path()})
    cands = candidates(c3a_member())
    keys = stages()[number]
    seed = board_rule().second_seed
    extra: dict = {"floors": floors, "candidates": list(keys)}
    # the legs at the ladder's seed
    fits = [cands[k] for k in keys if cands[k].kind != "layer"]
    if fits:
        for leg in cb.leg_plan():
            prep = frame.prepare(leg)
            for c in fits:
                fit_and_record(frame, prep, c.member, 0, stage)
            del prep
    for c in (cands[k] for k in keys):
        for leg in cb.leg_plan():
            form(c, leg.name, 0, hourly)
    # the twelve held-out months, for any that holds the four
    verdicts = {k: verdict_of(measure_candidate(cands[k])) for k in keys}
    four = [k for k in keys if cb.clears(verdicts[k]["conditions"], cb.FOUR)]
    print(f"\nholding the four: {four or 'none'}", flush=True)
    for split in cb.month_plan():
        need = [cands[k] for k in four if cands[k].kind != "layer"]
        if need and not all(
            cb.record_path(cb.record_name(split.name, c.member.key, 0)).exists()
            for c in need
        ):
            prep = frame.prepare(split)
            for c in need:
                fit_and_record(frame, prep, c.member, 0, stage)
            del prep
        for k in four:
            form(cands[k], split.name, 0, hourly)
    # the second seed, for any that holds all five
    verdicts = {k: verdict_of(measure_candidate(cands[k])) for k in keys}
    five = [k for k in keys if cb.clears(verdicts[k]["conditions"], cb.FIVE)]
    print(f"holding all five: {five or 'none'}", flush=True)
    for leg in cb.leg_plan()[:2]:
        need = [cands[k] for k in five if cands[k].kind != "layer"]
        if need and not all(
            cb.record_path(cb.record_name(leg.name, c.member.key, seed)).exists()
            for c in need
        ):
            prep = frame.prepare(leg)
            for c in need:
                fit_and_record(frame, prep, c.member, seed, stage)
            del prep
        for k in five:
            form(cands[k], leg.name, seed, hourly)
    verdicts = {k: verdict_of(measure_candidate(cands[k])) for k in keys}
    for k in keys:
        print(f"{cands[k].label}: {verdicts[k]['status']}", flush=True)
    extra["verdicts"] = verdicts
    return cb.stage_end(stage, checks, before, extra)


SECTION4 = ("c1", "c2a", "c3a", "c4", "c5", "c3b")
AMENDMENT = ("c7", "c8")


def december_gain(m: dict) -> float:
    return m["seed0"]["december:all"]["rows"]["contrast"]


def five_hold(m: dict) -> bool:
    return cb.clears(verdict_of(m)["conditions"], cb.FIVE)


def settle(stage: int) -> dict:
    """What section 4 (stage 6) or the amendment (stage 7) adopts, from what is on disk.

    ``need`` names a combination still to be measured; ``config`` is the list
    of candidate keys adopted, ``combined`` whether they are read as one
    combination, and ``why`` the rule's words for how it was reached.
    """
    cands = candidates(c3a_member())
    measured = {k: measure_candidate(c) for k, c in cands.items()}
    status = {k: verdict_of(m)["status"] for k, m in measured.items()}
    clear4 = [k for k in SECTION4 if status.get(k) == "clears"]
    out: dict = {"status": status, "clear4": clear4}
    if len(clear4) < 2:
        out["section4"] = {
            "config": clear4,
            "combined": False,
            "why": "no candidate cleared" if not clear4 else "one candidate cleared",
        }
    else:
        combo = combination(clear4, cands, "combo4", "section 4's combination")
        m = measure_candidate(combo)
        if any(verdict_of(m)["conditions"][k] is None for k in cb.FIVE):
            out.update(need=combo)
            return out
        if five_hold(m):
            out["section4"] = {
                "config": clear4,
                "combined": True,
                "why": "the combination holds the five",
            }
        else:
            best = min(clear4, key=lambda k: december_gain(measured[k]))
            out["section4"] = {
                "config": [best],
                "combined": False,
                "why": "the combination fails; the clearing candidate with the largest December gain",
            }
        out["combo4"] = m
    if stage == 6:
        return out
    x = out["section4"]["config"]
    clear_am = [k for k in AMENDMENT if status.get(k) == "clears"]
    out["clear_amendment"] = clear_am
    if x and not clear_am:
        out["final"] = {
            "config": x,
            "combined": out["section4"]["combined"],
            "why": "neither C7 nor C8 cleared",
        }
        return out
    if not x and len(clear_am) < 2:
        out["final"] = {
            "config": clear_am,
            "combined": False,
            "why": "nothing from section 4; "
            + ("one of C7 and C8 cleared" if clear_am else "neither C7 nor C8 cleared"),
        }
        return out
    keys = list(x) + clear_am
    combo = combination(keys, cands, "combo7", "the amendment's combination")
    m = measure_candidate(combo)
    v = verdict_of(m)["conditions"]
    if any(v[k] is None for k in cb.FIVE):
        out.update(need=combo)
        return out
    out["combo7"] = m
    if five_hold(m):
        out["final"] = {
            "config": keys,
            "combined": True,
            "why": "the amendment's combination holds the five",
        }
    elif x:
        out["final"] = {
            "config": x,
            "combined": out["section4"]["combined"],
            "why": "the amendment's combination fails; section 4's configuration stands alone",
        }
    else:
        best = min(clear_am, key=lambda k: december_gain(measured[k]))
        out["final"] = {
            "config": [best],
            "combined": False,
            "why": "C7 and C8 together fail; the one with the larger December gain",
        }
    return out


def run_combination(number: int, amendment: Path) -> int:
    """Stage 6 or 7: fit and form the combination the rule asks for, if it asks for one."""
    if not reference_passed():
        raise SystemExit("the reference stage has not passed its checks; nothing is read")
    got = settle(number)
    combo = got.get("need")
    if combo is None:
        print(
            f"stage {number}: no combination to measure: {got.get('final') or got.get('section4')}",
            flush=True,
        )
        return 0
    if not on_time(number):
        return 0
    before, stage, checks, floors = begin(f"stage{number}", amendment)
    if checks.failed:
        return cb.stage_end(stage, checks, before, {"floors": floors})
    frame = BoardFrame() if combo.member is not None else None
    hourly = frame.winter_hourly if frame is not None else winter_hourly()
    for split in cb.leg_plan() + cb.month_plan():
        if (
            combo.member is not None
            and not cb.record_path(
                cb.record_name(split.name, combo.member.key, 0)
            ).exists()
        ):
            prep = frame.prepare(split)
            fit_and_record(frame, prep, combo.member, 0, stage)
            del prep
        if not form(combo, split.name, 0, hourly):
            raise SystemExit(
                f"STOPPED: the combination could not be formed on {split.name}"
            )
    after = settle(number)
    return cb.stage_end(
        stage,
        checks,
        before,
        {
            "floors": floors,
            "combination": asdict(combo),
            "settled": {
                k: v
                for k, v in after.items()
                if k in ("section4", "final", "clear4", "clear_amendment")
            },
        },
    )


def winter_hourly() -> pd.DataFrame:
    return classes_by_hour(load_metar(rb.METAR))


def mean_fit_s(backend: str = "lightgbm") -> dict[str, float]:
    """Measured seconds per fit (prepare, fit and predict), by leg and for a held-out month."""
    by: dict[str, list[float]] = {}
    for r in cb.read_records().values():
        if "fit_s" not in r or r.get("backend", "lightgbm") != backend:
            continue
        key = "month" if r["split"].startswith("heldout-") else r["split"]
        by.setdefault(key, []).append(r["fit_s"] + r["predict_s"] + r["prepare_s"])
    return {k: float(np.mean(v)) for k, v in by.items()}


def projected_finish(number: int) -> tuple[datetime, dict]:
    """When stage 7 would finish if this stage and every later one ran in full.

    Each candidate or combination still to run is costed at its worst: three
    legs, twelve held-out months and two second-seed legs, at the measured
    mean of each; C3b's at its own measured December fit, or at the 45-minute
    ceiling until one exists.
    """
    lgb = mean_fit_s()
    leg = sum(
        lgb.get(k, lgb.get("december", 0.0)) for k in ("december", "july", "october")
    )
    month = lgb.get("month", lgb.get("december", 0.0))
    s1 = lgb.get("december", 0.0) + lgb.get("july", 0.0)
    one = leg + 12 * month + s1
    cat = mean_fit_s("catboost").get("december", board_rule().catboost_gpu_after_s)
    ratio = cat / max(lgb.get("december", 1.0), 1.0)
    remaining = {}
    if number <= 3:
        remaining["stage3"] = one
    if number <= 5:
        remaining["stage5"] = one * ratio
    if number <= 6:
        remaining["stage6"] = one
    remaining["stage7"] = one
    finish = datetime.now(timezone.utc) + pd.Timedelta(seconds=sum(remaining.values()))
    return finish, {
        "seconds": remaining,
        "lightgbm_mean_s": lgb,
        "catboost_december_s": cat,
    }


def on_time(number: int) -> bool:
    """Whether the stage can start without pushing stage 7 past the deadline."""
    finish, parts = projected_finish(number)
    deadline = datetime.fromisoformat(board_rule().deadline_utc)
    ok = finish <= deadline
    line = (
        f"stage {number}: projected finish of stage 7 {finish:%Y-%m-%d %H:%M}Z against the"
        f" deadline {deadline:%Y-%m-%d %H:%M}Z: {'starts' if ok else 'NOT STARTED'}"
    )
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"clock_stage{number}.json"
    path.write_text(
        json.dumps(
            {"finish_utc": finish.isoformat(timespec="seconds"), "ok": ok, **parts},
            indent=1,
        ),
        encoding="utf-8",
    )
    return ok


def projection() -> str:
    records = cb.read_records()
    fits = [r for r in records.values() if "fit_s" in r]
    if not fits:
        return "no fit measured yet"
    by = {}
    for r in fits:
        by.setdefault(
            r["split"] if not r["split"].startswith("heldout-") else "month", []
        ).append(r["fit_s"] + r["predict_s"] + r["prepare_s"])
    return "measured seconds per fit, mean: " + ", ".join(
        f"{k} {np.mean(v):.0f} ({len(v)})" for k, v in sorted(by.items())
    )


# --------------------------------------------------------------------------
# the report


STAGE_NAMES = (
    "reference",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
)


def what_is_passed(c: Candidate) -> str:
    if c.key == "c2a":
        return (
            "no fit: `taxiout.winter`'s layer added to the rung's predictions, fitted on the"
            " rung's stored out-of-fold rows (neither rule's, 4 h or less)"
        )
    if c.key == "c5":
        return (
            "no fit: recorded = alpha(a) + beta(a) x predicted, least squares per airport over"
            " the same rows, applied to every row of the airport"
        )
    if c.kind == "blend":
        m = c.member
        if m is None:
            return "a LightGBM member under the objective the Park test chooses, not yet chosen"
        if m.backend == "catboost":
            what = "a CatBoost member through `Booster`'s CatBoost backend, section 3's parameters"
        elif m.objective == "gamma":
            what = "a LightGBM member under the gamma deviance, the target floored at 1 s"
        else:
            what = (
                f"a LightGBM member under the Tweedie deviance, variance power {m.tweedie_power},"
                " the target floored at 1 s"
            )
        return what + ", read as the half-and-half blend with the rung at the same seed"
    m = c.member
    if m.weighted:
        return "the rung refitted with `taxiout.regime_weights`'s row weights toward the target month"
    cols = ", ".join(f"`{x}`" for b in m.blocks for x in BLOCK_COLUMNS[b])
    return f"the rung's features with {cols} appended"


def condition_rows(c: Candidate, m: dict) -> list[list[str]]:
    rule = cb.the_rule()
    d = rule.contrast_decimals
    rows = []
    by = {(k.split, k.rows): k for k in cb.conditions(rule)}
    for split, which in (
        ("december", "all"),
        ("december", "three"),
        ("july", "all"),
        ("july", "three"),
        ("october", "all"),
        ("year", "all"),
    ):
        r = m.get("seed0", {}).get(f"{split}:{which}")
        cond = by.get((split, which))
        label = (
            "the twelve held-out months"
            if split == "year"
            else f"{split.capitalize()}, {'all ten' if which == 'all' else 'the three'}"
        )
        if r is None:
            if cond is not None:
                rows.append(
                    [
                        label,
                        "",
                        "",
                        "",
                        "",
                        f"{cond.kind}, {cb.fmt(cond.floor, d)}",
                        "not reached",
                        "",
                        "",
                    ]
                )
            continue
        verdict = (
            "beside the rule"
            if cond is None
            else cb.mark(cb.holds(cond.kind, r["rows"]["contrast"], cond.floor, d))
        )
        unrounded = (
            ""
            if cond is None
            else f"{r['rows']['contrast']:+.6f}, {'holds' if cb.holds_unrounded(cond.kind, r['rows']['contrast'], cond.floor) else 'fails'}"
        )
        rows.append(
            [
                label,
                cb.fmt(r["rows"]["reference"], d),
                cb.fmt(r["rows"]["arm"], d),
                cb.signed(r["rows"]["contrast"], d),
                cb.interval(r["rows"], d),
                "" if cond is None else f"{cond.kind}, {cb.fmt(cond.floor, d)}",
                verdict,
                unrounded,
                f"{cb.fmt(r['short']['reference'], d)} / {cb.fmt(r['short']['arm'], d)} /"
                f" {cb.signed(r['short']['contrast'], d)}",
            ]
        )
    return rows


def seed_rows(m: dict) -> list[list[str]]:
    d = cb.the_rule().contrast_decimals
    s = f"seed{board_rule().second_seed}"
    out = []
    for leg in ("december", "july"):
        r0 = m.get("seed0", {}).get(f"{leg}:all")
        r1 = m.get(s, {}).get(f"{leg}:all")
        if r0 is None or r1 is None:
            continue
        same = cb.same_sign(r0["rows"]["contrast"], r1["rows"]["contrast"], d)
        out.append(
            [
                leg.capitalize(),
                cb.fmt(r1["rows"]["reference"], d),
                cb.fmt(r1["rows"]["arm"], d),
                cb.signed(r0["rows"]["contrast"], d),
                cb.signed(r1["rows"]["contrast"], d),
                cb.interval(r1["rows"], d),
                "same sign" if same else "**not the same sign**",
            ]
        )
    return out


def adoption(statuses: dict) -> dict:
    """Section 4 and the amendment, from each candidate's status alone.

    Combinations are reported by their own stages; until one is measured the
    verdict names what has cleared and what the combination would hold.
    """
    clear_4 = [
        k for k in ("c1", "c2a", "c3a", "c3b", "c4", "c5") if statuses.get(k) == "clears"
    ]
    clear_am = [k for k in ("c7", "c8") if statuses.get(k) == "clears"]
    return {"section4": clear_4, "amendment": clear_am}


def write_report(amendment: Path, generated: str) -> int:
    records = cb.read_records()
    checks = cb.Checks()
    floors = preflight(checks, amendment)
    cb.reproduction_checks(checks, records, cb.the_rule(), floors["october_printed"])
    cb.month_check(checks, records)
    cb.seed1_check(checks, records)
    if not (WORK / "stands.json").exists():
        raise SystemExit(
            "no reference stage has passed its checks; there is nothing to report"
        )
    stands = set(json.loads((WORK / "stands.json").read_text(encoding="utf-8")))
    movements: set[str] = set()
    for name in records:
        ids = pd.read_parquet(cb.kept_path(name), columns=["mvt_id"])["mvt_id"].dropna()
        movements |= {f"{int(v)}" for v in ids.unique()}
    before = cb.fingerprint(exempt=REPORT)
    text = build_report(records, checks, floors, generated)
    cb.guard(text, stands, movements)
    REPORT.write_text(text, encoding="utf-8")
    moved = cb.touched(before, cb.fingerprint(exempt=REPORT))
    if moved:
        raise SystemExit(
            "writing the report touched a file it must not: " + ", ".join(moved)
        )
    print(f"wrote {REPORT.relative_to(REPO)}")
    return 1 if checks.failed else 0


def build_report(records: dict, checks: cb.Checks, floors: dict, generated: str) -> str:
    rule = cb.the_rule()
    d = rule.contrast_decimals
    cands = candidates(c3a_member())
    measured = {k: measure_candidate(c) for k, c in cands.items()}
    verdicts = {k: verdict_of(m) for k, m in measured.items()}
    statuses = {k: v["status"] for k, v in verdicts.items()}
    statuses.update(not_run())
    stages = {s: cb.read_stage(s) for s in STAGE_NAMES}
    ad = adoption(statuses)
    stopped = bool(checks.failed)
    lines = [
        "# The candidates after the board, under the rule of 4 October",
        "",
        f"Generated by `scripts/candidates_board.py` on {generated} from the fits and the"
        " predictions it recorded under `data/processed/candidates_board/`; nothing is refitted"
        " to write it, every figure and every verdict is computed from those records, and the"
        " generator refuses to write a movement identifier or a literal stand identifier.",
        "",
    ]
    if stopped:
        lines += [cb.stop_line(checks.failed), ""]
    else:
        verdict = [
            x for x in adoption_lines(ad, statuses, cands) if x.startswith("**ADOPTED")
        ]
        if verdict:
            lines += [verdict[0], ""]
    lines += [
        "## What was measured",
        "",
        "The five candidates of section 3 of `reports/preregistered-2026-10-04.md` and the two"
        " its amendment of the same day adds, each fitted or formed on top of `+ plan"
        " differences` (`SUBMISSION_KEY`) and read against that rung's own fit at the ladder's"
        " seed, alone, under section 4's rule, which keeps the legs, the reference, the rows"
        " and the floors of `reports/preregistered-2026-10-03.md`. C2b is measured only if C2a"
        " is adopted.",
        "",
    ]
    lines += cb.table(
        ["candidate", "what is passed to the fitting code", "status"],
        [[c.label, what_is_passed(c), statuses[k]] for k, c in cands.items()],
        "---",
    )
    lines += [""]
    counts = ((stages.get("reference") or {}).get("counts")) or {}
    if counts:
        lines += ["### The rows, per leg", ""]
        lines += cb.table(
            [
                "leg",
                "validation rows",
                "the term's",
                "the plan-shift rule's",
                "read",
                "of them 4 h or less",
            ],
            [
                [
                    leg.capitalize(),
                    f"{counts[leg]['valid']:,}",
                    f"{counts[leg]['term']:,}",
                    f"{counts[leg]['plan_shift']:,}",
                    f"{counts[leg]['booster']:,}",
                    f"{counts[leg]['four_hours_or_less']:,}",
                ]
                for leg in counts
            ],
            "-rrrrr",
        )
        lines += [""]
    lines += [
        f"**How a condition is read.** Every contrast is the candidate's RMSE minus the"
        f" reference's over the same rows of the same split, both at the same seed, so negative"
        f" is better, read at {d} decimals against the floor printed at the same precision. The"
        " interval beside each is the day-clustered percentile interval of"
        f" `taxiout.validation.cluster_bootstrap`, {cb.BOOTSTRAP_DRAWS:,} draws; it decides"
        " nothing. The unrounded contrast and its reading are printed beside the rule's. For the"
        " two layers the twelve months are read leave one month out and each forward leg's"
        " layer is fitted on the held-out months strictly before the leg's month.",
        "",
        "## Before anything was read",
        "",
    ]
    lines += cb.table(
        ["check", "expected", "got", ""],
        [
            [r["what"], r["expected"], r["got"], "ok" if r["ok"] else "**stop**"]
            for r in checks.rows
        ],
        "---r",
    )
    lines += [""]
    park = WORK / "c3a_park.json"
    if park.exists():
        got = json.loads(park.read_text(encoding="utf-8"))
        lines += [
            "## C3a's objective, chosen before any C3a fit",
            "",
            f"The modified Park test over {got['rows']:,} out-of-fold rows of the rung (the twelve"
            f" held-out months, neither rule's, 4 h or less, prediction above zero): lambda ="
            f" {got['lambda']:.4f}, so the objective is `{got['objective']}`"
            + (f" with variance power {got['power']}" if got["power"] is not None else "")
            + ".",
            "",
        ]
    if not stopped:
        for k, c in cands.items():
            m = measured[k]
            if not m:
                continue
            lines += [f"## {c.label}", ""]
            lines += cb.table(
                [
                    "read on",
                    "reference (s)",
                    "candidate (s)",
                    "contrast (s)",
                    "interval (s)",
                    "condition, floor (s)",
                    "verdict",
                    "unrounded contrast, its reading",
                    "4 h or less: reference / candidate / contrast (s), diagnostic",
                ],
                condition_rows(c, m),
                "-rrrr---r",
            )
            lines += [""]
            year = m.get("seed0", {}).get("year:all")
            if year is not None:
                mrows = [
                    [
                        month,
                        f"{r['rows']['rows']:,}",
                        cb.fmt(r["rows"]["reference"], d),
                        cb.fmt(r["rows"]["arm"], d),
                        cb.signed(r["rows"]["contrast"], d),
                    ]
                    for month, r in year["months"].items()
                ]
                lines += cb.table(
                    ["held out", "rows", "the rung (s)", "candidate (s)", "contrast (s)"],
                    mrows,
                    "-rrrr",
                )
                lines += [""]
            sr = seed_rows(m)
            if sr:
                lines += cb.table(
                    [
                        "leg, all ten",
                        "reference at seed 1 (s)",
                        "candidate at seed 1 (s)",
                        "contrast at seed 0 (s)",
                        "contrast at seed 1 (s)",
                        "interval at seed 1 (s)",
                        "the check",
                    ],
                    sr,
                    "-rrrrr-",
                )
                lines += [""]
            lines += [f"**{c.label}: {statuses[k]}.**", ""]
    weighted = [r for r in records.values() if r.get("weights")]
    if weighted:
        lines += ["## C8's weights, printed before each booster was fitted", ""]
        lines += cb.table(
            [
                "fit",
                "rows",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "clipped low",
                "clipped high",
                "share clipped",
            ],
            [
                [
                    f"`{r['name']}`",
                    f"{w['rows']:,}",
                    f"{w['min']:.3f}",
                    f"{w['q1']:.3f}",
                    f"{w['median']:.3f}",
                    f"{w['q3']:.3f}",
                    f"{w['max']:.3f}",
                    f"{w['clipped_low']:,}",
                    f"{w['clipped_high']:,}",
                    f"{w['share_clipped']:.4%}",
                ]
                for r in sorted(weighted, key=lambda r: r["written_utc"])
                for w in [r["weights"]]
            ],
            "-rrrrrrrrr",
        )
        lines += [""]
    path = c3b_path()
    if path:
        lines += [
            "## C3b's timing and path",
            "",
            f"The December leg on CPU, eight threads, fitted in {path['cpu_december_fit_s']:,.0f} s"
            f" against the {path['limit_s'] / 60:.0f}-minute limit: {path['why']}.",
            "",
        ]
    lines += ["## What the rule adopts", ""]
    lines += adoption_lines(ad, statuses, cands)
    lines += fits_section(records, stages)
    lines += cb.section_fingerprint({k: v for k, v in stages.items()})
    env = cb.environment()
    lines += [
        f"Run under {env['python']}, {env['lightgbm']}, pandas {env['pandas']}, numpy {env['numpy']}.",
        "",
    ]
    lines += report_inputs_footer(REPO, report_inputs(records))
    lines += ["", cb.ATTRIBUTION, ""]
    return "\n".join(lines)


def not_run() -> dict[str, str]:
    """Stages refused by the clock, and C3b stopped on its own rule, with the reason."""
    out = {}
    for number, keys in stages().items():
        clock = WORK / f"clock_stage{number}.json"
        if clock.exists() and not json.loads(clock.read_text(encoding="utf-8"))["ok"]:
            for k in keys:
                out[k] = "not run: the clock projected the last stage past the deadline"
    path = c3b_path()
    if path.get("path") == "stopped":
        out["c3b"] = "not run: " + path["why"]
    return out


def adoption_lines(ad: dict, statuses: dict, cands: dict) -> list[str]:
    pending = [
        cands[k].label
        for k, s in statuses.items()
        if s
        not in ("clears", "fails the four", "fails the fifth", "fails the sign check")
        and not s.startswith("not run")
    ]
    if pending:
        return [
            "**No verdict yet.** Still to be measured: " + "; ".join(pending) + ".",
            "",
        ]
    got = settle(7)
    if "need" in got:
        return [f"**No verdict yet.** {got['need'].label} is still to be measured.", ""]
    lines = []
    s4 = got["section4"]
    lines += [
        "Section 4 combines the candidates among C1 to C5 that cleared alone ("
        + (", ".join(cands[k].label for k in got["clear4"]) or "none")
        + f"): {s4['why']}.",
        "",
    ]
    for key, label in (
        ("combo4", "section 4's combination"),
        ("combo7", "the amendment's combination"),
    ):
        m = got.get(key)
        if m is None:
            continue
        combo = cands_combination(key, got, cands)
        lines += [f"### {label.capitalize()}", "", what_combined(combo, cands), ""]
        lines += cb.table(
            [
                "read on",
                "reference (s)",
                "combination (s)",
                "contrast (s)",
                "interval (s)",
                "condition, floor (s)",
                "verdict",
                "unrounded contrast, its reading",
                "4 h or less: reference / combination / contrast (s), diagnostic",
            ],
            condition_rows(combo, m),
            "-rrrr---r",
        )
        lines += [""]
    final = got["final"]
    lines += [
        "The amendment adds C7 and C8 only where they cleared ("
        + (
            ", ".join(cands[k].label for k in got.get("clear_amendment", []))
            or "neither did"
        )
        + f"): {final['why']}.",
        "",
    ]
    keys = final["config"]
    verdict = "ADOPTED: " + (", ".join("C" + k[1:] for k in keys) if keys else "none")
    lines += [f"**{verdict}.**", ""]
    if keys:
        lines += [
            "v9 is this configuration"
            + (", read as one combination" if final["combined"] and len(keys) > 1 else "")
            + ", five seeds, both tail rules off, as section 4 and the amendment make it.",
            "",
        ]
    if "c1" in keys:
        lines += [
            "C1 reads `AOBT_3_flt`, which stays excluded by decision everywhere but its own"
            " module. Section 3 makes its admission need, beside this verdict, the owner's"
            " originality decision in writing; v9 waits for it.",
            "",
        ]
    return lines


def cands_combination(key: str, got: dict, cands: dict) -> Candidate:
    keys = (
        got["clear4"]
        if key == "combo4"
        else got["section4"]["config"] + got["clear_amendment"]
    )
    label = (
        "section 4's combination" if key == "combo4" else "the amendment's combination"
    )
    return combination(keys, cands, key, label)


def what_combined(c: Candidate, cands: dict) -> str:
    parts = []
    if c.member is not None:
        cols = ", ".join(f"`{x}`" for b in c.member.blocks for x in BLOCK_COLUMNS[b])
        parts.append(
            "one booster with the rung's features and "
            + (cols or "no added column")
            + (", fitted with the regime weights" if c.member.weighted else "")
        )
    else:
        parts.append("the rung as the booster")
    if c.layers:
        parts.append("then the layers " + ", ".join(c.layers))
    if c.members:
        parts.append(
            "then the blend in which the booster and "
            + ", ".join(f"`{m.key}`" for m in c.members)
            + " take equal weights"
        )
    return "What is combined: " + "; ".join(parts) + "."


def fits_section(records: dict, stages: dict) -> list[str]:
    fitted = {k: v for k, v in records.items() if "fit_s" in v}
    lines = ["## Every fit", ""]
    lines += [
        "Each fit's own time on the monotonic clock, which a suspended machine does not"
        " advance, and each stage's whole, with the time the machine spent suspended inside it.",
        "",
    ]
    lines += cb.table(
        [
            "fit",
            "backend",
            "trees",
            "columns",
            "weighted",
            "fit rows",
            "fit (s)",
            "predict (s)",
            "peak (GiB)",
        ],
        [
            [
                f"`{n}`",
                r.get("backend", "lightgbm"),
                f"{r['n_estimators']:,}",
                f"{r['columns']}",
                "yes" if r["weight_sum"] is not None else "no",
                f"{r['fit_rows']:,}",
                f"{r['fit_s']:,.0f}",
                f"{r['predict_s']:,.0f}",
                f"{r['peak_rss_gib']:,.1f}",
            ]
            for n, r in sorted(fitted.items(), key=lambda kv: kv[1]["written_utc"])
        ],
        "--rr-rrrr",
    )
    lines += [""]
    srows = []
    for name, s in stages.items():
        for run in cb.runs(s):
            srows.append(
                [
                    f"{name}, {run['started_utc']}",
                    f"{len(run['fits'])}",
                    f"{run['wall_s']:,.0f}",
                    f"{run['suspended_s']:,.0f}",
                    f"{run['peak_rss_gib']:,.1f}",
                ]
            )
    lines += cb.table(
        ["stage", "fits", "wall (s)", "suspended (s)", "peak (GiB)"], srows, "-rrrr"
    )
    lines += [""]
    counted = {n for s in stages.values() for run in cb.runs(s) for n in run["fits"]}
    orphans = sorted(n for n in fitted if n not in counted)
    if orphans:
        lines += [
            f"{len(orphans)} fit(s) were written by a run that ended without writing its"
            " stage record, because the machine lost power during the next fit; the run"
            " that resumed the stage found them on disk, did not refit them, and refitted"
            " only the fit that was in progress. Their own times are in the table above;"
            " the stage's row covers the resumed run alone: "
            + ", ".join(f"`{n}`" for n in orphans)
            + ".",
            "",
        ]
    return lines


def report_inputs(records: dict) -> list[Path]:
    paths = [
        REPO / BOARD_TEXT,
        REPO / cb.BASELINE,
        REPO / cb.CANDIDATES_TEXT,
        REPO / cb.RULE_TEXT,
        REPO / SEEDS_REPORT,
    ]
    paths += [
        REPO / cb.LADDER_RESIDUALS / f"{leg}_{SUBMISSION_KEY}.parquet"
        for leg in ("december", "july")
    ]
    paths += [
        REPO / cb.MONTHLY / f"heldout-{m}_{SUBMISSION_KEY}.parquet" for m in cb.MONTHS
    ]
    seeds = REPO / cb.MONTHLY_SEEDS / f"{max(cb.MONTHS)}_seed1.parquet"
    if seeds.exists():
        paths.append(seeds)
    paths += sorted(rb.METAR.glob("*.csv"))
    paths += [WORK / "stands.json"]
    paths += sorted(WORK.glob("stage_*.json")) + sorted(WORK.glob("c3a_park.json"))
    for name in sorted(records):
        paths += [cb.record_path(name), cb.kept_path(name)]
    return paths


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--amendment", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", choices=("reference",))
    mode.add_argument("--stage", type=int, choices=sorted(stages()) + [6, 7])
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = ap.parse_args(argv)
    point_harness()
    if args.fit == "reference":
        return run_reference(args.amendment)
    if args.stage in (6, 7):
        return run_combination(args.stage, args.amendment)
    if args.stage is not None:
        return run_stage(args.stage, args.amendment)
    if args.status:
        cands = candidates(c3a_member())
        for c in cands.values():
            print(c.key, verdict_of(measure_candidate(c)))
        print(projection())
        return 0
    return write_report(args.amendment, args.date)


if __name__ == "__main__":
    raise SystemExit(main())
