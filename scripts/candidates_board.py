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

And the one candidate the amendment of 6 October 2026 adds, read under the
same rule with its own stages and its own report,
``reports/candidates-2026-10-07.md``:

- **C8c**, C8 with each training row's odds multiplied by n0 / n1 before the
  clip (``regime_weights``'s ``class_ratio``); if it clears, the configuration
  adopted on 6 October with C8c's weights on its booster's fits.

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

    python scripts/candidates_board.py --amendment PATH --prior PATH --c8c check
    python scripts/candidates_board.py --amendment PATH --prior PATH --c8c alone
    python scripts/candidates_board.py --amendment PATH --prior PATH --c8c combination
    python scripts/candidates_board.py --amendment PATH --prior PATH --c8c report

``--amendment`` is the working note of the amendment, which this repository
does not yet carry; its digest is checked before anything is read. ``--prior``
is the working note of the amendment of 6 October, checked the same way.

**Group B of the last week**, for v11 (§3 of
``reports/preregistered-2026-10-06-last-week.md`` as
``reports/preregistered-2026-10-06-airports.md`` amends it), is formed here
too, from ``taxiout.group_b``'s blends and selector: W, C2a's winter layer on
the base's out-of-fold residuals; C10, the blend weight, chosen among four
(a selection, and disclosed as one); C11, C3b's CatBoost member with C1's and
C4's eleven columns; C12, the configuration's booster at 9,000 trees; C13,
one booster per airport; C14, the airport-by-airport selector over the eight
menu items, whose forward legs let later months inform their choice; and the
combination. ``fit_group_b`` fits one part as the overnight run of 6 October
fitted it, ``GroupB`` forms and checks every derived prediction, and
``group_b_readings`` reads every rule. ``scripts/report_group_b.py`` writes
``reports/group-b.md`` from the records.
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
from taxiout import group_b  # noqa: E402
from taxiout.aobt3 import AOBT3_FEATURES, aobt3_features  # noqa: E402
from taxiout.aobt3 import REQUIRED_COLUMNS as AOBT3_COLUMNS  # noqa: E402
from taxiout.challenge import load_movements, to_pandas  # noqa: E402
from taxiout.flight_history import FLIGHT, FLIGHT_HISTORY_FEATURES, flight_history  # noqa: E402
from taxiout.metar import load_metar  # noqa: E402
from taxiout.models import LADDER_BY_KEY, SUBMISSION_KEY, Booster  # noqa: E402
from taxiout.provenance import (  # noqa: E402
    FROZEN_DOCUMENTS,
    digest,
    parse_report_inputs,
    report_inputs_footer,
)
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
    # C8c's factor on the weights' odds; ``None`` for C8's weights or none
    class_ratio: bool | None = None


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
    if member.key == "c12_booster":
        params["n_estimators"] = int(group_b_rule().c12_trees)
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
        got = regime_weights(
            x,
            frame.dep["MVT_TIME_UTC_mvt"],
            rows,
            prep["valid"],
            class_ratio=bool(member.class_ratio),
        )
        weight = got.weights
        weights = {**got.distribution(), "classifier_s": time.monotonic() - t0}
        for f in weights.get("folds", []):
            print(
                "  the classifier fitted on the {fitted_on}: n0 {n0:,}, n1 {n1:,}, its odds"
                " times n0/n1 = {factor:.6f} on the {scored:,} rows it scored".format(
                    **f
                ),
                flush=True,
            )
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
    class_ratio = False
    layers, members = [], []
    for k in keys:
        c = cands[k]
        if c.kind == "member":
            blocks += c.member.blocks
            weighted = weighted or bool(c.member.weighted)
            class_ratio = class_ratio or bool(c.member.class_ratio)
        elif c.kind == "layer":
            layers.append(c.layer)
        elif c.kind == "blend":
            members.append(c.member)
    order = {"winter": 0, "recal": 1}
    booster = (
        Member(
            f"{key}_booster",
            blocks=blocks,
            weighted=True if weighted else None,
            class_ratio=True if class_ratio else None,
        )
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
    extra: dict = {"floors": floors, "candidates": list(keys)}
    extra["verdicts"] = read_through(frame, hourly, stage, cands, keys)
    return cb.stage_end(stage, checks, before, extra)


def read_through(
    frame: BoardFrame | None,
    hourly: pd.DataFrame,
    stage: dict,
    cands: dict[str, Candidate],
    keys: tuple[str, ...],
) -> dict:
    """Section 4's reading of ``keys``, as far as the rule reaches; their verdicts.

    The three legs at the ladder's seed; the twelve held-out months for any
    that holds the four leg conditions; the second seed on December and July
    for any that holds all five.
    """
    seed = board_rule().second_seed
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
    return verdicts


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
# C8c, the amendment of 6 October 2026


REPORT_C8C = REPO / "reports" / "candidates-2026-10-07.md"


@dataclass(frozen=True)
class PriorRule:
    """The amendment of 6 October's numbers, each in the words it is fixed in.

    ``tests/test_candidates_board_c8c.py`` checks them against the working
    note where it is on hand.
    """

    prior_sha256: str
    deadline_utc: str
    adopted: tuple[str, ...]


def prior_rule() -> PriorRule:
    """The numbers, held in a function for the reason ``board_rule`` gives.

    - "The verdict is written by 7 October 2026, 18:00 UTC, or nothing is
      adopted";
    - the configuration adopted on 6 October, "C1 and C4 in one booster,
      blended at equal weights with C3b's CatBoost member", which the records
      of 6 October must settle to before anything is fitted.
    """
    return PriorRule(
        prior_sha256="81871eeb4f43d82b6a0aca1b5f3302d40a65cd5446efc05cf7289cb3a6fc1e98",
        deadline_utc="2026-10-07T18:00:00+00:00",
        adopted=("c1", "c4", "c3b"),
    )


# The arms this amendment writes, and its stages, in the order they run.
C8C_ARMS = ("reference_refit", "c8c", "combo8c_booster", "combo8c")
C8C_STAGES = ("c8c_check", "c8c_alone", "c8c_combination")


def c8c() -> Candidate:
    return Candidate(
        "c8c",
        "C8c, C8's weights corrected for the class ratio",
        "member",
        Member("c8c", weighted=True, class_ratio=True),
    )


def c8c_combination() -> Candidate:
    """The configuration of 6 October with C8c's weights on its booster's fits.

    C1's and C4's features in one booster, fitted at the ladder's seed with
    C8c's weights, blended at equal weights with C3b's member, which is read
    from its stored predictions and stays unweighted.
    """
    cands = {**candidates(c3a_member()), "c8c": c8c()}
    return combination(
        [*prior_rule().adopted, "c8c"],
        cands,
        "combo8c",
        "the configuration with C8c's weights",
    )


def ours(path: Path) -> bool:
    """Whether a file under the records' directory is this amendment's to write."""
    name = path.name
    if name.startswith(("stage_c8c_", "clock_c8c")):
        return True
    parts = name.split("__")
    return len(parts) == 3 and parts[1] in C8C_ARMS


def inherited_now() -> dict[str, str]:
    """Every file under the records' directory that is not this amendment's, by sha256."""
    if not WORK.exists():
        return {}
    return {
        str(p.relative_to(REPO)): digest(p)
        for p in sorted(WORK.iterdir())
        if p.is_file() and not ours(p)
    }


def inherited_check(checks: cb.Checks, manifest: dict[str, str] | None) -> dict[str, str]:
    """The records of 5 and 6 October, read and never written here.

    Against the digests the report of 5 October declares for every one of
    them it read, and, once the first check has run, against the manifest it
    took of every file under the directory that is not this amendment's.
    """
    now = inherited_now()
    prefix = str(WORK.relative_to(REPO)) + "/"
    declared = {
        d.path: d.digest
        for d in parse_report_inputs(REPORT.read_text(encoding="utf-8"))
        if d.path.startswith(prefix)
    }
    same = sum(now.get(p) == sha for p, sha in declared.items())
    checks.add(
        f"the records of 5 and 6 October at the digests `{REPORT.relative_to(REPO)}` declares",
        f"{len(declared):,} identical",
        f"{same:,} identical",
        bool(declared) and same == len(declared),
    )
    if manifest is not None:
        moved = sorted(
            (set(manifest) ^ set(now))
            | {p for p in manifest if now.get(p) != manifest[p]}
        )
        checks.add(
            "every other file under the records' directory as the first check found it",
            f"{len(manifest):,} unchanged, none added",
            "unchanged" if not moved else "moved: " + ", ".join(moved),
            not moved,
        )
    return now


def stage_passed(name: str) -> dict | None:
    """The latest run of a stage, if every check held and nothing it may not write moved."""
    got = cb.read_stage(name)
    if got is None:
        return None
    latest = {k: v for k, v in got.items() if k != "previous"}
    ok = (
        bool(latest.get("checks"))
        and all(r["ok"] for r in latest["checks"])
        and not latest["fingerprint"]["touched"]
    )
    return latest if ok else None


def begin_c8c(name: str, amendment: Path, prior: Path, manifest: dict | None):
    before, stage, checks, floors = begin(name, amendment)
    rule = prior_rule()
    got = digest(prior)
    checks.add(
        "the working note of the amendment of 6 October at the digest it was fixed by",
        rule.prior_sha256,
        got,
        got == rule.prior_sha256,
    )
    inherited = inherited_check(checks, manifest)
    return before, stage, checks, floors, inherited


def configuration_words(final: dict) -> str:
    keys = final.get("config") or []
    return (", ".join("C" + k[1:] for k in keys) or "none") + (
        ", read as one combination" if final.get("combined") and len(keys) > 1 else ""
    )


WEIGHT_FIGURES = (
    "rows",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "clipped_low",
    "clipped_high",
    "share_clipped",
    "raw_mean",
)


def run_c8c_check(amendment: Path, prior: Path) -> int:
    """Before C8c is fitted: what was adopted on 6 October, C8's weights, the reference.

    The configuration the records of 6 October settle to; C8's December
    weights recomputed with the option off against the stored fit's, every
    figure of their distribution and their sum; then the December reference
    refitted, against the ladder's stored residuals and the stored reference
    fit. A disagreement is a stop.
    """
    if not reference_passed():
        raise SystemExit("the reference stage has not passed its checks; nothing is read")
    first = stage_passed("c8c_check")
    before, stage, checks, floors, inherited = begin_c8c(
        "c8c_check", amendment, prior, first["inherited"] if first else None
    )
    if first:
        inherited = first["inherited"]
    extra: dict = {"floors": floors, "inherited": inherited}
    if checks.failed:
        return cb.stage_end(stage, checks, before, extra)
    final = settle(7).get("final") or {}
    want = {"config": list(prior_rule().adopted), "combined": True}
    extra["adopted_6_october"] = final
    checks.add(
        "the configuration the records of 6 October settle to",
        configuration_words(want),
        configuration_words(final),
        final.get("config") == want["config"] and bool(final.get("combined")),
    )
    if checks.failed:
        return cb.stage_end(stage, checks, before, extra)
    frame = BoardFrame()
    extra["frame_build_s"] = frame.build_seconds + frame.extra_seconds
    leg = cb.leg_plan()[0]
    prep = frame.prepare(leg)
    stored = json.loads(
        cb.record_path(cb.record_name(leg.name, "c8", 0)).read_text(encoding="utf-8")
    )
    print("\nC8's December weights, recomputed with the option off", flush=True)
    t0 = time.monotonic()
    got = regime_weights(
        prep["frame"], frame.dep["MVT_TIME_UTC_mvt"], prep["rows"], prep["valid"]
    )
    again = {**got.distribution(), "classifier_s": time.monotonic() - t0}
    weight_sum = float(np.sum(got.weights))
    print(
        "  weights: min {min:.3f}, quartiles {q1:.3f} / {median:.3f} / {q3:.3f},"
        " max {max:.3f}, clipped {share_clipped:.4%} ({clipped_low:,} low,"
        " {clipped_high:,} high), {classifier_s:.0f} s".format(**again),
        flush=True,
    )
    extra["c8_december_weights"] = {
        "stored": stored["weights"],
        "recomputed": again,
        "stored_weight_sum": stored["weight_sum"],
        "recomputed_weight_sum": weight_sum,
    }
    w = stored["weights"]
    differ = [k for k in WEIGHT_FIGURES if again[k] != w[k]]
    if weight_sum != stored["weight_sum"]:
        differ.append("weight_sum")
    checks.add(
        "C8's December weights recomputed with the option off, against the stored fit's",
        "every figure of the distribution and the weight sum identical to the stored"
        f" ({w['clipped_low'] / w['rows']:.1%} of rows clipped low, the largest weight"
        f" {w['max']:.1f} after scaling)",
        "identical" if not differ else "differ in " + ", ".join(differ),
        not differ,
    )
    if not checks.failed:
        print("\nthe December reference, refitted", flush=True)
        refit = Member("reference_refit")
        fit_and_record(frame, prep, refit, 0, stage)
        name = cb.record_name(leg.name, refit.key, 0)
        got_s, ok = cb.reproduces(
            cb.read_kept(name),
            REPO / cb.LADDER_RESIDUALS / f"{leg.name}_{SUBMISSION_KEY}.parquet",
        )
        checks.add(
            "the December reference refitted, against the ladder's"
            f" `{leg.name}_{SUBMISSION_KEY}.parquet`",
            "a largest absolute difference of 0 on every row",
            got_s,
            ok,
        )
        new = json.loads(cb.record_path(name).read_text(encoding="utf-8"))["pred_sha256"]
        old = json.loads(
            cb.record_path(cb.record_name(leg.name, "reference", 0)).read_text(
                encoding="utf-8"
            )
        )["pred_sha256"]
        checks.add(
            "the December reference refitted, against the stored reference fit's pred_sha256",
            old,
            new,
            new == old,
        )
    del prep
    inherited_check(checks, inherited)
    return cb.stage_end(stage, checks, before, extra)


def run_c8c_alone(amendment: Path, prior: Path) -> int:
    """C8c read against the rung as section 4 reads a candidate, alone."""
    check = stage_passed("c8c_check")
    if check is None:
        raise SystemExit(
            "the check of 6 October's amendment has not passed; nothing is fitted"
        )
    before, stage, checks, floors, _ = begin_c8c(
        "c8c_alone", amendment, prior, check["inherited"]
    )
    extra: dict = {"floors": floors, "candidates": ["c8c"]}
    if checks.failed:
        return cb.stage_end(stage, checks, before, extra)
    frame = BoardFrame()
    extra["verdicts"] = read_through(
        frame, frame.winter_hourly, stage, {"c8c": c8c()}, ("c8c",)
    )
    inherited_check(checks, check["inherited"])
    return cb.stage_end(stage, checks, before, extra)


def projected_finish_c8c() -> tuple[datetime, dict]:
    """When the combination would finish, from fit times measured on the same splits.

    Each split still to fit costs the configuration's own booster on that
    split (prepare, fit and predict, as recorded on 6 October) plus C8c's
    classifiers on it, and the frame is built once, at the check's measured
    time.
    """
    records = cb.read_records()
    seconds = {}
    for split in cb.leg_plan() + cb.month_plan():
        if cb.record_name(split.name, "combo8c_booster", 0) in records:
            continue
        b = records[cb.record_name(split.name, "combo4_booster", 0)]
        w = records[cb.record_name(split.name, "c8c", 0)]
        seconds[split.name] = (
            b["prepare_s"] + b["fit_s"] + b["predict_s"] + w["weights"]["classifier_s"]
        )
    frame_s = float((stage_passed("c8c_check") or {}).get("frame_build_s", 0.0))
    finish = datetime.now(timezone.utc) + pd.Timedelta(
        seconds=sum(seconds.values()) + frame_s
    )
    return finish, {"seconds": seconds, "frame_s": frame_s}


def on_time_c8c() -> bool:
    """Whether the combination can start and finish by the amendment's deadline."""
    finish, parts = projected_finish_c8c()
    deadline = datetime.fromisoformat(prior_rule().deadline_utc)
    ok = finish <= deadline
    print(
        f"the combination: projected finish {finish:%Y-%m-%d %H:%M}Z against the deadline"
        f" {deadline:%Y-%m-%d %H:%M}Z: {'starts' if ok else 'NOT STARTED'}",
        flush=True,
    )
    (WORK / "clock_c8c.json").write_text(
        json.dumps(
            {
                "finish_utc": finish.isoformat(timespec="seconds"),
                "deadline_utc": deadline.isoformat(timespec="seconds"),
                "ok": ok,
                **parts,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return ok


def same_weights(split: str) -> tuple[str, bool]:
    """The combination's booster weighted exactly as C8c on the same split.

    The classifier reads the airport and the weather and congestion blocks
    alone, so C1's and C4's columns beside them change no weight.
    """
    a, b = (
        json.loads(
            cb.record_path(cb.record_name(split, arm, 0)).read_text(encoding="utf-8")
        )
        for arm in ("combo8c_booster", "c8c")
    )
    keys = [*WEIGHT_FIGURES, "folds"]
    differ = [k for k in keys if a["weights"].get(k) != b["weights"].get(k)]
    if a["weight_sum"] != b["weight_sum"]:
        differ.append("weight_sum")
    return ("identical" if not differ else "differ in " + ", ".join(differ)), not differ


def run_c8c_combination(amendment: Path, prior: Path) -> int:
    """The amendment's combination, only if C8c holds all five and its second-seed check."""
    check = stage_passed("c8c_check")
    if check is None:
        raise SystemExit(
            "the check of 6 October's amendment has not passed; nothing is fitted"
        )
    status = verdict_of(measure_candidate(c8c()))["status"]
    if status != "clears":
        print(
            f"C8c: {status}; no combination to measure, the configuration of 6 October"
            " stands",
            flush=True,
        )
        return 0
    if not on_time_c8c():
        return 0
    before, stage, checks, floors, _ = begin_c8c(
        "c8c_combination", amendment, prior, check["inherited"]
    )
    combo = c8c_combination()
    extra: dict = {"floors": floors, "combination": asdict(combo)}
    if checks.failed:
        return cb.stage_end(stage, checks, before, extra)
    frame = BoardFrame()
    for split in cb.leg_plan() + cb.month_plan():
        if not cb.record_path(cb.record_name(split.name, combo.member.key, 0)).exists():
            prep = frame.prepare(split)
            fit_and_record(frame, prep, combo.member, 0, stage)
            del prep
        got, ok = same_weights(split.name)
        if not checks.add(
            f"{split.name}: the booster's weights against C8c's own on the split",
            "identical",
            got,
            ok,
        ):
            break
        if not form(combo, split.name, 0, frame.winter_hourly):
            raise SystemExit(
                f"STOPPED: the combination could not be formed on {split.name}"
            )
    inherited_check(checks, check["inherited"])
    if not checks.failed:
        extra["settled"] = {
            k: v for k, v in settle_c8c().items() if k in ("status", "before", "final")
        }
    return cb.stage_end(stage, checks, before, extra)


def settle_c8c() -> dict:
    """What the amendment of 6 October adopts, from what is on disk.

    ``need`` names what is still to be measured; ``final`` carries the
    verdict line, the configuration v9 is, and the rule's words for how it
    was reached.
    """
    before = {"config": list(prior_rule().adopted), "combined": True}
    m = measure_candidate(c8c())
    status = verdict_of(m)["status"]
    out: dict = {"c8c": m, "status": status, "before": before}
    stands = {**before, "verdict": "C8c: not adopted"}
    if status in ("fails the four", "fails the fifth", "fails the sign check"):
        out["final"] = {
            **stands,
            "why": f"C8c {status}; the configuration of 6 October stands",
        }
        return out
    if status != "clears":
        out["need"] = "C8c"
        return out
    mc = measure_candidate(c8c_combination())
    out["combo8c"] = mc
    if any(verdict_of(mc)["conditions"][k] is None for k in cb.FIVE):
        clock = WORK / "clock_c8c.json"
        if clock.exists() and not json.loads(clock.read_text(encoding="utf-8"))["ok"]:
            out["final"] = {
                **stands,
                "why": "the clock projected the combination past the deadline, so it was"
                " not started; the configuration of 6 October stands",
            }
        else:
            out["need"] = "the configuration with C8c's weights"
        return out
    if five_hold(mc):
        out["final"] = {
            "config": [*before["config"], "c8c"],
            "combined": True,
            "verdict": "C8c: ADOPTED, combined",
            "why": "the configuration with C8c's weights holds the five, and is adopted in"
            " place of the configuration of 6 October",
        }
    else:
        out["final"] = {
            **stands,
            "why": "C8c clears, the configuration with C8c's weights fails the five; the"
            " configuration of 6 October stands",
        }
    return out


# --------------------------------------------------------------------------
# C8c's report


def c8c_records(records: dict) -> dict:
    """This amendment's records, and the stored ones its readings read."""
    read = {"reference", "base", "c8", "combo4", "combo4_booster"}
    read.add(candidates(c3a_member())["c3b"].member.key)
    return {n: r for n, r in records.items() if n.split("__")[1] in (*C8C_ARMS, *read)}


def write_report_c8c(amendment: Path, prior: Path, generated: str) -> int:
    check = stage_passed("c8c_check")
    if check is None:
        raise SystemExit(
            "the check of 6 October's amendment has not passed; there is nothing to report"
        )
    records = c8c_records(cb.read_records())
    checks = cb.Checks()
    preflight(checks, amendment)
    got = digest(prior)
    checks.add(
        "the working note of the amendment of 6 October at the digest it was fixed by",
        prior_rule().prior_sha256,
        got,
        got == prior_rule().prior_sha256,
    )
    inherited_check(checks, check["inherited"])
    twice_check(checks, records)
    stands = set(json.loads((WORK / "stands.json").read_text(encoding="utf-8")))
    movements: set[str] = set()
    for name in records:
        ids = pd.read_parquet(cb.kept_path(name), columns=["mvt_id"])["mvt_id"].dropna()
        movements |= {f"{int(v)}" for v in ids.unique()}
    before = cb.fingerprint(exempt=REPORT_C8C)
    text = build_report_c8c(records, checks, check, generated)
    cb.guard(text, stands, movements)
    REPORT_C8C.write_text(text, encoding="utf-8")
    moved = cb.touched(before, cb.fingerprint(exempt=REPORT_C8C))
    if moved:
        raise SystemExit(
            "writing the report touched a file it must not: " + ", ".join(moved)
        )
    print(f"wrote {REPORT_C8C.relative_to(REPO)}")
    return 1 if checks.failed else 0


def twice_check(checks: cb.Checks, records: dict) -> None:
    """The December leg and held-out December are one split, fitted twice: one digest."""
    for arm in ("c8c", "combo8c_booster"):
        a = records.get(cb.record_name("december", arm, 0))
        b = records.get(cb.record_name("heldout-2025-12", arm, 0))
        if a is None or b is None:
            continue
        checks.add(
            f"`{arm}`: the December leg and held-out December, the same split fitted twice",
            a["pred_sha256"],
            b["pred_sha256"],
            a["pred_sha256"] == b["pred_sha256"],
        )


def combination_status(m: dict) -> str:
    v = verdict_of(m)["conditions"]
    if any(v[k] is None for k in cb.FIVE):
        return "not measured"
    return "holds the five" if cb.clears(v, cb.FIVE) else "fails the five"


def beside_rows(arm: str, base: str, splits: list[tuple[str, str]]) -> list[list[str]]:
    """``arm`` against ``base`` on stored predictions, a reading beside the rule."""
    d = cb.the_rule().contrast_decimals
    rows = []
    for split, which in splits:
        if split == "year":
            parts = [
                (arm_kept_named(s.name, arm), arm_kept_named(s.name, base))
                for s in cb.month_plan()
            ]
            if any(a is None or b is None for a, b in parts):
                continue
            a = pd.concat([p[0] for p in parts], ignore_index=True)
            b = pd.concat([p[1] for p in parts], ignore_index=True)
            label = "the twelve held-out months"
        else:
            a, b = arm_kept_named(split, arm), arm_kept_named(split, base)
            if a is None or b is None:
                continue
            label = (
                f"{split.capitalize()}, {'all ten' if which == 'all' else 'the three'}"
            )
        r = cb.reading(a, b, which)["rows"]
        rows.append(
            [
                label,
                cb.fmt(r["arm"], d),
                cb.fmt(r["reference"], d),
                cb.signed(r["contrast"], d),
                cb.interval(r, d),
            ]
        )
    return rows


def arm_kept_named(split: str, arm: str) -> pd.DataFrame | None:
    name = cb.record_name(split, arm, 0)
    return cb.read_kept(name) if cb.record_path(name).exists() else None


LEGS_AND_YEAR = (
    ("december", "all"),
    ("july", "all"),
    ("july", "three"),
    ("october", "all"),
    ("year", "all"),
)


def months_rows(m: dict) -> list[list[str]]:
    d = cb.the_rule().contrast_decimals
    year = m.get("seed0", {}).get("year:all")
    if year is None:
        return []
    return [
        [
            month,
            f"{r['rows']['rows']:,}",
            cb.fmt(r["rows"]["reference"], d),
            cb.fmt(r["rows"]["arm"], d),
            cb.signed(r["rows"]["contrast"], d),
        ]
        for month, r in year["months"].items()
    ]


def weights_rows(records: dict) -> list[list[str]]:
    """Every weighted fit's distribution, C8's stored three first."""
    out = []
    fits = sorted(
        (r for r in records.values() if r.get("weights")),
        key=lambda r: (r["arm"]["key"] != "c8", r["written_utc"]),
    )
    for r in fits:
        w = r["weights"]
        folds = {f["fitted_on"]: f for f in w.get("folds", [])}
        ratio = [
            f"{folds[k]['n0']:,} / {folds[k]['n1']:,} = {folds[k]['factor']:.4f}"
            if k in folds
            else "none, C8"
            for k in ("even days", "odd days")
        ]
        out.append(
            [
                f"`{r['name']}`",
                f"{w['rows']:,}",
                *ratio,
                f"{w['min']:.3f}",
                f"{w['q1']:.3f}",
                f"{w['median']:.3f}",
                f"{w['q3']:.3f}",
                f"{w['max']:.3f}",
                f"{w['clipped_low']:,}",
                f"{w['clipped_high']:,}",
                f"{w['share_clipped']:.4%}",
            ]
        )
    return out


def build_report_c8c(
    records: dict, checks: cb.Checks, check: dict, generated: str
) -> str:
    rule = cb.the_rule()
    d = rule.contrast_decimals
    got = settle_c8c()
    final = got.get("final")
    stages = {s: cb.read_stage(s) for s in C8C_STAGES}
    stopped = bool(checks.failed)
    cands = {**candidates(c3a_member()), "c8c": c8c()}
    c, combo = c8c(), c8c_combination()
    m, mc = got["c8c"], got.get("combo8c") or {}
    status = got["status"]
    lines = [
        "# C8c, C8's weights corrected for the class ratio, under the rule of 4 October",
        "",
        f"Generated by `scripts/candidates_board.py` on {generated} from the fits and the"
        " predictions it recorded under `data/processed/candidates_board/`, beside the"
        " records of 5 and 6 October there, which it reads and never writes; nothing is"
        " refitted to write it, every figure and every verdict is computed from those"
        " records, and the generator refuses to write a movement identifier or a literal"
        " stand identifier. Nothing ships from it.",
        "",
    ]
    if stopped:
        lines += [cb.stop_line(checks.failed), ""]
    elif final:
        lines += [f"**{final['verdict']}.**", ""]
    else:
        lines += [f"**No verdict yet.** {got['need']} is still to be measured.", ""]
    lines += [
        "## What was measured",
        "",
        "C8, the training rows weighted toward the target month, is one of the two"
        " candidates the amendment of 4 October adds to section 3 of"
        " `reports/preregistered-2026-10-04.md`. It failed section 4's four leg"
        " conditions, and its weights were set by the clip rather than by its classifier"
        " (`reports/candidates-2026-10-05.md`): a classifier fitted on n1 target rows"
        " against n0 training rows estimates its odds with the prior n1/n0 built in, so"
        " with one month against eleven most training rows' odds fell under the clip's"
        " floor whatever their regime.",
        "",
        "The amendment of 6 October 2026, fixed by its sha256"
        f" (`{prior_rule().prior_sha256}`) before any fit of the candidate it defines and"
        " stated there as a choice made after C8's figures, adds one candidate, C8c: C8"
        " exactly, with each training row's odds multiplied, before the clip, by n0/n1,"
        " the counts of label 0 and label 1 among the rows the classifier that scored"
        " that row was fitted on. Section 4's rule reads it against the submission rung's"
        " own fit at the ladder's seed, unchanged in its legs, its reference, its rows,"
        " its floors and its second-seed check. If C8c holds all five conditions and its"
        " second-seed check, the configuration adopted on 6 October (C1's and C4's"
        " features in one booster, blended at equal weights with C3b's CatBoost member) is"
        " refitted with C8c's weights on its booster's fits at the ladder's seed, C3b's"
        " member read from its stored predictions and unweighted, and held to the same"
        " five conditions; if it holds them it is adopted in place of that configuration,"
        " and otherwise that configuration stands. The verdict is due by 7 October 2026,"
        " 18:00 UTC.",
        "",
    ]
    lines += cb.table(
        ["candidate", "what is passed to the fitting code", "status"],
        [
            [
                c.label,
                "the rung refitted with `taxiout.regime_weights`'s row weights toward the"
                " target month, each row's odds multiplied by n0/n1 of the fold whose"
                " classifier scored it (`class_ratio`)",
                status,
            ],
            [
                combo.label[0].upper() + combo.label[1:],
                what_combined(combo, cands),
                combination_status(mc) if mc else "not run",
            ],
        ],
        "---",
    )
    lines += [""]
    counts = ((cb.read_stage("reference") or {}).get("counts")) or {}
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
        f" reference's over the same rows of the same split, both at the same seed, so"
        f" negative is better, read at {d} decimals against the floor printed at the same"
        " precision. The interval beside each is the day-clustered percentile interval of"
        f" `taxiout.validation.cluster_bootstrap`, {cb.BOOTSTRAP_DRAWS:,} draws; it decides"
        " nothing. The unrounded contrast and its reading are printed beside the rule's."
        " The second seed moves the booster's seed alone: the classifier's seed is fixed,"
        " so a split's weights are the same at both seeds. The combination is held to the"
        " five conditions and no second seed, as section 4 holds a combination.",
        "",
        "## Before anything was read",
        "",
        "Checked again as this report was written:",
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
    lines += ["", f"Checked before C8c was fitted, on {check['started_utc']}:", ""]
    lines += cb.table(
        ["check", "expected", "got", ""],
        [
            [r["what"], r["expected"], r["got"], "ok" if r["ok"] else "**stop**"]
            for r in check["checks"]
        ],
        "---r",
    )
    lines += [""]
    c8w = check.get("c8_december_weights")
    if c8w:
        lines += [
            "## C8's December weights, recomputed with the option off",
            "",
            "C8's path through `taxiout.regime_weights` with `class_ratio` off, on the"
            " December leg's rows, against the distribution its fit of 5 October stored.",
            "",
        ]
        rows = [
            [
                k,
                f"{c8w['stored'][k]!r}",
                f"{c8w['recomputed'][k]!r}",
                "identical"
                if c8w["stored"][k] == c8w["recomputed"][k]
                else "**differs**",
            ]
            for k in WEIGHT_FIGURES
        ]
        rows.append(
            [
                "weight sum",
                f"{c8w['stored_weight_sum']!r}",
                f"{c8w['recomputed_weight_sum']!r}",
                "identical"
                if c8w["stored_weight_sum"] == c8w["recomputed_weight_sum"]
                else "**differs**",
            ]
        )
        lines += cb.table(["figure", "stored", "recomputed", ""], rows, "-rr-")
        lines += [""]
    if not stopped and m:
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
        mrows = months_rows(m)
        if mrows:
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
        br = beside_rows("c8c", "c8", list(LEGS_AND_YEAR[:4]))
        if br:
            lines += [
                "Beside the rule, C8c against C8 as specified, on the same legs at the"
                " ladder's seed (negative: C8c better):",
                "",
            ]
            lines += cb.table(
                ["read on", "C8c (s)", "C8 (s)", "contrast (s)", "interval (s)"],
                br,
                "-rrrr",
            )
            lines += [""]
        lines += [f"**{c.label}: {status}.**", ""]
    wrows = weights_rows(records)
    if wrows:
        lines += [
            "## The weights, printed before each booster was fitted",
            "",
            "Every weighted fit's distribution after scaling, C8's three of 5 October first"
            " for comparison; n0 / n1 is each fold's count of training rows against target"
            " rows, named by the parity its classifier was fitted on.",
            "",
        ]
        lines += cb.table(
            [
                "fit",
                "rows",
                "n0 / n1, fitted on even days",
                "n0 / n1, fitted on odd days",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "clipped low",
                "clipped high",
                "share clipped",
            ],
            wrows,
            "-r--rrrrrrrr",
        )
        lines += [""]
    if not stopped and mc:
        lines += [
            f"## {combo.label[0].upper() + combo.label[1:]}",
            "",
            what_combined(combo, cands),
            "",
        ]
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
            condition_rows(combo, mc),
            "-rrrr---r",
        )
        lines += [""]
        mrows = months_rows(mc)
        if mrows:
            lines += cb.table(
                ["held out", "rows", "the rung (s)", "combination (s)", "contrast (s)"],
                mrows,
                "-rrrr",
            )
            lines += [""]
        br = beside_rows("combo8c", "combo4", list(LEGS_AND_YEAR))
        if br:
            lines += [
                "Beside the rule, against the configuration of 6 October, read on the same"
                " rows (negative: the configuration with C8c's weights better); the rule"
                " adopts on the five conditions alone:",
                "",
            ]
            lines += cb.table(
                [
                    "read on",
                    "with C8c's weights (s)",
                    "of 6 October (s)",
                    "contrast (s)",
                    "interval (s)",
                ],
                br,
                "-rrrr",
            )
            lines += [""]
        lines += [
            f"**{combo.label[0].upper() + combo.label[1:]}: {combination_status(mc)}.**",
            "",
        ]
    lines += ["## What the rule adopts", ""]
    if final and not stopped:
        lines += [f"{final['why'][0].upper() + final['why'][1:]}.", ""]
        lines += [f"**{final['verdict']}.**", ""]
        lines += [
            f"v9 is {configuration_words(final)}, five seeds, both tail rules off. Its"
            " predictions at the ladder's seed, per row:",
            "",
        ]
        arm = "combo8c" if "c8c" in final["config"] else "combo4"
        prow = []
        for split in cb.leg_plan() + cb.month_plan():
            r = records.get(cb.record_name(split.name, arm, 0))
            if r is not None:
                prow.append(
                    [f"`{r['name']}`", f"{r['valid_rows']:,}", f"`{r['pred_sha256']}`"]
                )
        lines += cb.table(["record", "rows", "pred_sha256"], prow, "-r-")
        lines += [""]
    else:
        lines += ["No verdict is written here.", ""]
    lines += fits_section_c8c(records, stages)
    lines += fingerprint_section_c8c(stages)
    env = cb.environment()
    lines += [
        f"Run under {env['python']}, {env['lightgbm']}, pandas {env['pandas']}, numpy {env['numpy']}.",
        "",
    ]
    lines += report_inputs_footer(REPO, report_inputs_c8c(records))
    lines += ["", cb.ATTRIBUTION, ""]
    return "\n".join(lines)


def fits_section_c8c(records: dict, stages: dict) -> list[str]:
    fitted = {
        k: v for k, v in records.items() if "fit_s" in v and k.split("__")[1] in C8C_ARMS
    }
    lines = ["## Every fit", ""]
    lines += [
        "Each fit of this amendment's stages, its own time on the monotonic clock, which a"
        " suspended machine does not advance, and each stage's whole, with the time the"
        " machine spent suspended inside it. The classifiers' time is apart from the"
        " booster's.",
        "",
    ]
    lines += cb.table(
        [
            "fit",
            "trees",
            "columns",
            "weighted",
            "fit rows",
            "classifiers (s)",
            "prepare (s)",
            "fit (s)",
            "predict (s)",
            "peak (GiB)",
        ],
        [
            [
                f"`{n}`",
                f"{r['n_estimators']:,}",
                f"{r['columns']}",
                "C8c's" if r.get("weights") else "no",
                f"{r['fit_rows']:,}",
                f"{r['weights']['classifier_s']:,.0f}" if r.get("weights") else "",
                f"{r['prepare_s']:,.0f}",
                f"{r['fit_s']:,.0f}",
                f"{r['predict_s']:,.0f}",
                f"{r['peak_rss_gib']:,.1f}",
            ]
            for n, r in sorted(fitted.items(), key=lambda kv: kv[1]["written_utc"])
        ],
        "-rr-rrrrrr",
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
    return lines


def fingerprint_section_c8c(stages: dict) -> list[str]:
    every = [(name, run) for name, s in stages.items() for run in cb.runs(s)]
    moved = [name for name, run in every if run["fingerprint"]["touched"]]
    lines = ["## What each run was allowed to write", ""]
    lines += [
        "Every file under `reports/`, the stored residuals and records of the ladder, the"
        " held-out months and the earlier experiments, and the submissions, by size,"
        " modification time and sha256, before each stage and after it; and every record"
        " of 5 and 6 October under the records' directory, by sha256, at the start and the"
        " end of each stage. "
        + (
            "No such file changed in any run; each wrote only its own records under"
            " `data/processed/candidates_board/`, which git ignores."
            if not moved
            else "Files changed while these ran: " + ", ".join(sorted(set(moved))) + "."
        ),
        "",
    ]
    rows = []
    for name, run in every:
        inherited = [
            r for r in run.get("checks", []) if "records of 5 and 6 October" in r["what"]
        ]
        rows.append(
            [
                f"{name}, {run['started_utc']}",
                f"{run['fingerprint']['files']:,}",
                ", ".join(run["fingerprint"]["touched"]) or "none",
                ", ".join(r["got"] for r in inherited) or "not read",
            ]
        )
    lines += cb.table(
        ["stage", "files checked", "touched", "records of 5 and 6 October"], rows, "-r--"
    )
    lines.append("")
    return lines


def report_inputs_c8c(records: dict) -> list[Path]:
    paths = [
        REPO / BOARD_TEXT,
        REPO / cb.BASELINE,
        REPO / cb.CANDIDATES_TEXT,
        REPO / cb.RULE_TEXT,
        REPO / SEEDS_REPORT,
        REPORT,
        REPO / cb.LADDER_RESIDUALS / f"december_{SUBMISSION_KEY}.parquet",
    ]
    paths += [
        WORK / n
        for n in ("stands.json", "c3a_park.json", "c3b_path.json", "stage_reference.json")
    ]
    paths += [cb.stage_path(s) for s in C8C_STAGES if cb.stage_path(s).exists()]
    paths += sorted(WORK.glob("clock_c8c.json"))
    for name in sorted(records):
        paths += [cb.record_path(name), cb.kept_path(name)]
    return paths


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# group B of the last week, for v11


GROUP_B_TEXT = "reports/preregistered-2026-10-06-last-week.md"
AIRPORTS_TEXT = "reports/preregistered-2026-10-06-airports.md"


@dataclass(frozen=True)
class GroupBRule:
    """Group B's numbers, each in the words it is fixed in.

    ``tests/test_group_b.py`` checks each against the published copies of the
    last-week note's §3 and of the airports amendment.
    """

    grid: tuple[float, ...]
    c12_trees: int
    w_winter: tuple[tuple[str, str | None, str, float, str], ...]
    w_legs: tuple[tuple[str, str, str, float], ...]
    month_floors: dict[str, float]
    short_floors: dict[str, float]
    deadline_utc: str
    last_week_sha256: str
    airports_sha256: str


def group_b_rule() -> GroupBRule:
    """The numbers, held in a function for the reason ``board_rule`` gives.

    - C10, "w in {0.3, 0.4, 0.6, 0.7}";
    - C12, "`combo4_booster` refitted at `n_estimators` 9,000";
    - W, "better on January by more than 0.47 s; not worse on February by
      more than 1.59 s or on December by more than 0.23 s; not worse over the
      twelve held-out months' short rows by more than 0.15 s", and on the
      forward legs "not worse than the leg floors (December 0.23, July 0.54,
      the three July airports 1.02, October 0.21 s)";
    - the floors of the last-week note's §1, "over core rows Jan 0.33, Feb
      1.56, Mar 0.36, Apr 0.07, May 0.13, Jun 0.21, Jul 0.58, Aug 0.26, Sep
      0.21, Oct 0.29, Nov 0.39, Dec 0.23, year 0.16 s; over short rows Jan
      0.47, Feb 1.59, Dec 0.23, year 0.15 s", which the report checks against
      ``reports/monthly_seeds.md``;
    - the deadline, "Verdicts by 9 October 2026, 18:00 UTC";
    - the working notes' digests, as the copies' closing sections name them.
    """
    return GroupBRule(
        grid=(0.3, 0.4, 0.6, 0.7),
        c12_trees=9000,
        w_winter=(
            ("january", "2025-01", "better", 0.47, "short"),
            ("february", "2025-02", "not worse", 1.59, "short"),
            ("december", "2025-12", "not worse", 0.23, "short"),
            ("year", None, "not worse", 0.15, "short"),
        ),
        w_legs=(
            ("leg_december", "december", "all", 0.23),
            ("leg_july", "july", "all", 0.54),
            ("leg_july_three", "july", "three", 1.02),
            ("leg_october", "october", "all", 0.21),
        ),
        month_floors={
            "Jan": 0.33,
            "Feb": 1.56,
            "Mar": 0.36,
            "Apr": 0.07,
            "May": 0.13,
            "Jun": 0.21,
            "Jul": 0.58,
            "Aug": 0.26,
            "Sep": 0.21,
            "Oct": 0.29,
            "Nov": 0.39,
            "Dec": 0.23,
            "year": 0.16,
        },
        short_floors={"Jan": 0.47, "Feb": 1.59, "Dec": 0.23, "year": 0.15},
        deadline_utc="2026-10-09T18:00:00+00:00",
        last_week_sha256="eb35ecd3eb802b077307d5dba6ab07ca0cc5277b6fc7d5522d1b65a86bc9f1d5",
        airports_sha256="9ef5691f36365e2691204380b5f4e984d7049a0e2a8922f80943ed5309724a4a",
    )


def group_b_members() -> dict[str, Member]:
    """The fitted parts of group B, as the overnight run of 6 October fitted them.

    The configuration's booster (``combo4_booster``: the rung, C1's nine and
    C4's two columns), C11's member (C3b's CatBoost member with those eleven
    columns added), C12's booster (``combo4_booster`` at C12's trees) and C13's
    (one booster per airport on ``combo4_booster``'s columns).
    """
    blocks = ("aobt3", "flight")
    return {
        "combo4_booster": Member("combo4_booster", blocks=blocks),
        "c11_member": Member("c11_member", blocks=blocks, backend="catboost"),
        "c12_booster": Member("c12_booster", blocks=blocks),
        "c13_booster": Member("c13_booster", blocks=blocks),
    }


def fit_c13(
    frame: BoardFrame, prep: dict, member: Member, seed: int
) -> tuple[dict, pd.DataFrame]:
    """Ten boosters, one per airport, each on its airport's fitting rows only.

    The airport column stays in the frame and is constant within a fit; the
    ten predictions are joined in row order, every validation row once.
    """
    split = prep["split"]
    columns = member_columns(member)
    x = prep["frame"].copy(deep=False)
    for b in member.blocks:
        got = frame.block(b, prep)
        for c in got.columns:
            x[c] = got[c]
    missing = [c for c in columns if c not in x.columns]
    if missing:
        raise RuntimeError(f"the frame lacks {missing}")
    params = member_params(member, seed)
    rows, valid = prep["rows"], prep["valid"]
    icao = frame.dep["ADEP_mvt"].astype(str)
    airports = sorted(set(icao[rows]))
    valid_icao = icao[valid].to_numpy()
    if tuple(airports) != fusion_airports() or set(valid_icao) - set(airports):
        raise RuntimeError(
            f"{split.name}: fitting airports {airports}, validation {sorted(set(valid_icao))}"
        )
    parts, per = {}, {}
    fit_s = predict_s = 0.0
    for a in airports:
        ra = rows & (icao == a)
        va = valid & (icao == a)
        t0 = time.monotonic()
        model = Booster(columns, params).fit(x.loc[ra], frame.y[ra])
        f = time.monotonic() - t0
        t1 = time.monotonic()
        p = np.asarray(model.predict(x.loc[va]), dtype=float)
        q = time.monotonic() - t1
        del model
        parts[a] = p
        per[a] = {
            "fit_rows": int(ra.sum()),
            "valid_rows": int(va.sum()),
            "fit_s": f,
            "predict_s": q,
            "pred_sha256": group_b.pred_sha256(p),
        }
        fit_s += f
        predict_s += q
        print(
            f"    {a}: {int(ra.sum()):,} fit rows, {int(va.sum()):,} validation rows,"
            f" fit {f:.0f} s",
            flush=True,
        )
    pred = group_b.join_by_airport(valid_icao, parts)
    kept = kept_frame(frame, prep, pred)
    record = {
        "name": cb.record_name(split.name, member.key, seed),
        "split": split.name,
        "train_months": list(split.train_months),
        "valid_month": split.valid_month,
        "arm": {"key": member.key},
        "member": {**asdict(member), "per_airport": True},
        "seed": int(seed),
        "random_state": int(params["random_state"]),
        "n_estimators": int(params["n_estimators"]),
        "objective": str(params["objective"]),
        "backend": "lightgbm",
        "backend_params": None,
        "columns": len(columns),
        "fit_rows": int(rows.sum()),
        "valid_rows": int(valid.sum()),
        "airports": per,
        "prepare_s": float(prep["seconds"]),
        "fit_s": fit_s,
        "predict_s": predict_s,
        "peak_rss_gib": cb.peak_gib(),
        "pred_sha256": group_b.pred_sha256(pred),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return record, kept


def fusion_airports() -> tuple[str, ...]:
    from taxiout.fusion import AIRPORTS as airports

    return tuple(airports)


def fit_group_b(
    frame: BoardFrame, prep: dict, member: Member, seed: int
) -> tuple[dict, pd.DataFrame]:
    """One fit of a group B part, checked as the overnight run checked it."""
    name = cb.record_name(prep["split"].name, member.key, seed)
    s0 = cb.suspended_s()
    if member.key == "c13_booster":
        record, kept = fit_c13(frame, prep, member, seed)
    else:
        record, kept = fit_member(frame, prep, member, seed)
    if member.backend == "catboost":
        want = board_rule().catboost
        got = record["backend_params"]
        bad = {
            k: got.get(k) for k, v in want.items() if not same_parameter(got.get(k), v)
        }
        record["backend_params_checked"] = not bad
        if bad:
            raise RuntimeError(f"{name}: the CatBoost fit ran with {got}, not {want}")
    if member.key == "c12_booster" and record["n_estimators"] != group_b_rule().c12_trees:
        raise RuntimeError(f"{name}: {record['n_estimators']} trees")
    record["suspended_during_s"] = cb.suspended_s() - s0
    return record, kept


def group_b_plan() -> list[tuple[str, int]]:
    """The seventeen fits of a fitted candidate: the legs, the twelve months, the second seed."""
    second = board_rule().second_seed
    out = [(s.name, 0) for s in cb.leg_plan()]
    out += [(s.name, 0) for s in cb.month_plan()]
    out += [(s.name, second) for s in cb.leg_plan()[:2]]
    return out


class GroupB:
    """Group B's predictions and readings, from the records under ``cb.WORK``.

    Every fitted prediction is read only after it hashes to its record's
    ``pred_sha256``. Every derived one (a blend, the winter layer, C14's
    selection) is formed here; where a record of it exists, the formed
    prediction must hash to that record, and where none does it is written.
    """

    def __init__(self) -> None:
        self.cache: dict[str, pd.DataFrame] = {}
        self.derived: dict[str, str] = {}
        self.checked: dict[str, str] = {}
        self._hourly = None

    # the predictions

    def read(self, name: str) -> pd.DataFrame | None:
        if name in self.cache:
            return self.cache[name]
        if not cb.record_path(name).exists():
            return None
        rec = json.loads(cb.record_path(name).read_text(encoding="utf-8"))
        kept = cb.read_kept(name)
        got = group_b.pred_sha256(kept["pred"].to_numpy(dtype=float))
        if got != rec["pred_sha256"]:
            raise RuntimeError(f"{name}: prediction {got} against its record")
        self.cache[name] = kept
        self.checked[name] = got
        return kept

    def kept(self, key: str, split: str, seed: int) -> pd.DataFrame | None:
        return self.read(cb.record_name(split, key, seed))

    def derive(self, name: str, frame: pd.DataFrame, how: dict) -> pd.DataFrame:
        got = group_b.pred_sha256(frame["pred"].to_numpy(dtype=float))
        if cb.record_path(name).exists():
            rec = json.loads(cb.record_path(name).read_text(encoding="utf-8"))
            if rec["pred_sha256"] != got:
                raise RuntimeError(
                    f"{name}: formed again to {got}, its record {rec['pred_sha256']}"
                )
        else:
            write_derived(name, frame, how)
        self.derived[name] = got
        self.cache[name] = frame
        return frame

    def blended(
        self, key: str, split: str, seed: int, bkey: str, mkey: str, w: float
    ) -> pd.DataFrame | None:
        name = cb.record_name(split, key, seed)
        if name in self.cache:
            return self.cache[name]
        b, m = self.kept(bkey, split, seed), self.kept(mkey, split, seed)
        if b is None or m is None:
            return None
        cb.aligned(b, m)
        frame = replace_pred(
            b, group_b.blend(b["pred"].to_numpy(), m["pred"].to_numpy(), w)
        )
        how = {
            "blend": [
                cb.record_name(split, bkey, seed),
                cb.record_name(split, mkey, seed),
            ],
            "weights": [w, 1.0 - w],
        }
        return self.derive(name, frame, how)

    def base(self, split: str, seed: int) -> pd.DataFrame | None:
        """v9's configuration: stored at seed 0, the second-seed members' half-and-half otherwise."""
        if seed == 0:
            return self.kept("combo4", split, 0)
        half = board_rule().blend_weight
        return self.blended("combo4", split, seed, "combo4_booster", "c3b_member", half)

    def item(self, it: str):
        """A menu item's prediction for a split and seed, as a function."""
        if it == "base":
            return self.base
        booster, member, w = group_b.item_blend(it)
        key = {"c11": "c11", "c12": "c12", "c13": "c13"}.get(it)
        if key is None:
            key = f"c10_w{int(round(w * 10)):02d}"

        def of(split: str, seed: int):
            return self.blended(key, split, seed, booster, member, w)

        return of

    def menu(self) -> dict:
        """The eight items, in the amendment's order."""
        items = ["base"] + [group_b.weight_item(w) for w in group_b_rule().grid]
        return {it: self.item(it) for it in items + ["c11", "c12", "c13"]}

    # the readings

    def per_airport(self, arm: pd.DataFrame, ref: pd.DataFrame) -> dict:
        rows = cb.booster_rows(arm)
        icao = arm["icao"].to_numpy()
        return {a: cb.paired(arm, ref, rows & (icao == a)) for a in fusion_airports()}

    def measure(self, arm_of, ref_of, seeds=(0, 1)) -> dict:
        """Every contrast §4 reads, in ``measure_candidate``'s shape, airports beside."""
        out: dict = {}
        beside: dict = {}
        for seed in seeds:
            for leg in cb.leg_plan():
                a, r = arm_of(leg.name, seed), ref_of(leg.name, seed)
                if a is None or r is None:
                    continue
                slot = out.setdefault(f"seed{seed}", {})
                slot[f"{leg.name}:all"] = cb.reading(a, r, "all")
                slot[f"{leg.name}:three"] = cb.reading(a, r, "three")
                if seed == 0 and leg.name in ("december", "july"):
                    beside[leg.name] = self.per_airport(a, r)
        arms, refs, months = [], [], {}
        for split in cb.month_plan():
            a, r = arm_of(split.name, 0), ref_of(split.name, 0)
            if a is None or r is None:
                out["beside_airports"] = beside
                return out
            months[split.valid_month] = cb.reading(a, r, "all")
            arms.append(a)
            refs.append(r)
        ay, ry = pd.concat(arms, ignore_index=True), pd.concat(refs, ignore_index=True)
        out.setdefault("seed0", {})["year:all"] = {
            **cb.reading(ay, ry, "all"),
            "months": months,
        }
        beside["year"] = self.per_airport(ay, ry)
        out["beside_airports"] = beside
        return out

    def verdict(self, m: dict) -> dict:
        v = verdict_of(m)
        rule = cb.the_rule()
        other = {}
        for c in cb.conditions(rule):
            got = m.get("seed0", {}).get(f"{c.split}:{c.rows}")
            if got is None:
                continue
            other[c.key] = {
                r: cb.holds_read(r, c.kind, got["rows"], c.floor, rule.contrast_decimals)
                for r in ("rule", *cb.OTHER_READINGS)
            }
        v["other_readings"] = other
        return v

    # W

    def hourly(self) -> pd.DataFrame:
        if self._hourly is None:
            self._hourly = winter_hourly()
        return self._hourly

    def w_layer(self, of, split: str, seed: int, key: str) -> pd.DataFrame | None:
        """W applied to ``of(split, seed)``, its means from ``of``'s held-out months at seed 0."""
        name = cb.record_name(split, key, seed)
        if name in self.cache:
            return self.cache[name]
        target = of(split, seed)
        if target is None:
            return None
        months = w_fit_months(split)
        parts = []
        for mm in months:
            k = of(f"heldout-{mm}", 0)
            parts.append(k.loc[oof_rows(k)])
        rows = pd.concat(parts, ignore_index=True)
        layer = fit_layer("winter", rows, self.hourly())
        pred = apply_layer("winter", layer, target, self.hourly())
        how = {
            "layer": "winter",
            "fitted_on_months": months,
            "fitted_rows": int(len(rows)),
            "fitted": layer_summary("winter", layer),
        }
        return self.derive(name, replace_pred(target, pred), how)

    def w_rule(self, m: dict) -> dict:
        d = cb.the_rule().contrast_decimals
        rule = group_b_rule()
        out = {}
        year = m.get("seed0", {}).get("year:all")
        for key, month, kind, floor, rows in rule.w_winter:
            r = year[rows] if month is None else year["months"][month][rows]
            out[key] = {
                "kind": kind,
                "floor": floor,
                "rows": rows,
                "contrast": r["contrast"],
                "holds": cb.holds(kind, r["contrast"], floor, d),
                "other": {
                    x: cb.holds_read(x, kind, r, floor, d) for x in cb.OTHER_READINGS
                },
            }
        for key, leg, rows, floor in rule.w_legs:
            r = m["seed0"][f"{leg}:{rows}"]["rows"]
            out[key] = {
                "kind": "not worse",
                "floor": floor,
                "rows": f"core, {rows}",
                "contrast": r["contrast"],
                "holds": cb.holds("not worse", r["contrast"], floor, d),
                "other": {
                    x: cb.holds_read(x, "not worse", r, floor, d)
                    for x in cb.OTHER_READINGS
                },
            }
        return out

    # C14

    def item_sse(self, of) -> dict:
        """Per held-out month and airport, an item's squared error and count over core rows."""
        out = {}
        for mm in cb.MONTHS:
            k = of(f"heldout-{mm}", 0)
            rows = cb.booster_rows(k)
            r = (k["y"] - k["pred"]).to_numpy(dtype=float)
            icao = k["icao"].to_numpy()
            out[mm] = {}
            for a in fusion_airports():
                sel = rows & (icao == a) & np.isfinite(r)
                out[mm][a] = (float(np.sum(r[sel] ** 2)), int(sel.sum()))
        return out

    def c14_choices(self) -> dict:
        items = self.menu()
        sse = {it: self.item_sse(of) for it, of in items.items()}
        names = list(items)
        table, scores = {}, {}
        for mm in cb.MONTHS:
            table[mm], scores[mm] = {}, {}
            for a in fusion_airports():
                table[mm][a], scores[mm][a] = group_b.choose(sse, names, a, mm)
        ranking, ranking_scores = {}, {}
        for a in fusion_airports():
            ranking[a], ranking_scores[a] = group_b.choose(sse, names, a, None)
        return {
            "items": names,
            "table": table,
            "scores": scores,
            "ranking": ranking,
            "ranking_scores": ranking_scores,
            "sse": sse,
        }

    def c14_of(self, choices: dict):
        items = self.menu()
        leg_month = {s.name: s.valid_month for s in cb.leg_plan()}

        def of(split: str, seed: int):
            name = cb.record_name(split, "c14", seed)
            if name in self.cache:
                return self.cache[name]
            month = (
                split.removeprefix("heldout-")
                if split.startswith("heldout-")
                else leg_month[split]
            )
            choice = choices["table"][month]
            frames = {}
            for it in set(choice.values()) | {"base"}:
                f = items[it](split, seed)
                if f is None:
                    return None
                frames[it] = f
            b = frames["base"]
            for f in frames.values():
                cb.aligned(f, b)
            pred = group_b.apply_choice(
                b["icao"].to_numpy(),
                b["pred"].to_numpy(dtype=float),
                {it: f["pred"].to_numpy(dtype=float) for it, f in frames.items()},
                choice,
            )
            how = {"selector": choice, "choice_month": month}
            return self.derive(name, replace_pred(b, pred), how)

        return of


def w_fit_months(split: str) -> list[str]:
    """W's months: leave one out on a held-out month; strictly before on a forward leg."""
    if split.startswith("heldout-"):
        month = split.removeprefix("heldout-")
        return [m for m in cb.MONTHS if m != month]
    leg = next(s for s in cb.leg_plan() if s.name == split)
    return months_before(leg)


def group_b_why_not(v: dict) -> str:
    failed = [k for k, ok in v["conditions"].items() if ok is False]
    unsigned = [k for k, ok in v["signs"].items() if ok is False]
    parts = [v["status"]]
    if failed:
        parts.append("fails " + ", ".join(failed))
    if unsigned:
        parts.append("sign changes on " + ", ".join(unsigned))
    return "; ".join(parts)


def group_b_line(label: str, v: dict, extra: str = "") -> str:
    if v["status"] == "clears":
        return f"{label}: clears{extra}"
    return f"{label}: does not clear{extra} ({group_b_why_not(v)})"


def group_b_readings(g: GroupB | None = None) -> dict:
    """Every reading group B's rules make, in the order the overnight run of 6 October made them.

    W; C10's grid and its chosen weight; C11, C12 and C13 alone; C14's choice
    table, then C14; the combination. Each verdict line is the rule's.
    """
    g = GroupB() if g is None else g
    rule = group_b_rule()
    half = board_rule().blend_weight
    out: dict = {}
    # the base, as the mean of its members at seed 0, on every split
    base_checks = {}
    for s in cb.leg_plan() + cb.month_plan():
        c4 = g.kept("combo4", s.name, 0)
        b, m = g.kept("combo4_booster", s.name, 0), g.kept("c3b_member", s.name, 0)
        cb.aligned(b, c4)
        cb.aligned(m, c4)
        mean = np.mean(
            np.vstack([b["pred"].to_numpy(dtype=float), m["pred"].to_numpy(dtype=float)]),
            axis=0,
        )
        base_checks[s.name] = {
            "stored": group_b.pred_sha256(c4["pred"].to_numpy(dtype=float)),
            "mean": group_b.pred_sha256(mean),
            "half": group_b.pred_sha256(
                group_b.blend(b["pred"].to_numpy(), m["pred"].to_numpy(), half)
            ),
        }
    year = pd.concat([g.base(s.name, 0) for s in cb.month_plan()], ignore_index=True)
    rows = cb.booster_rows(year)
    out["base"] = {
        "checks": base_checks,
        "rmse": cb.rmse((year["y"] - year["pred"]).to_numpy(dtype=float)[rows]),
        "core_rows": int(rows.sum()),
    }

    def w_of(split, seed):
        return g.w_layer(g.base, split, seed, "w")

    m = g.measure(w_of, g.base, seeds=(0,))
    wr = g.w_rule(m)
    ok = all(r["holds"] for r in wr.values())
    out["w"] = {
        "measured": m,
        "rule": wr,
        "clears": ok,
        "line": "W: clears; W2 is due"
        if ok
        else "W: does not clear (fails "
        + ", ".join(k for k, r in wr.items() if not r["holds"])
        + ")",
    }
    grid = {}
    for w in rule.grid:
        of = g.item(group_b.weight_item(w))
        frames = [of(s.name, 0) for s in cb.month_plan()]
        for s in cb.leg_plan():
            of(s.name, 0)
        y = pd.concat(frames, ignore_index=True)
        r = cb.booster_rows(y)
        grid[f"{w:.1f}"] = cb.rmse((y["y"] - y["pred"]).to_numpy(dtype=float)[r])
    chosen = min(rule.grid, key=lambda w: (grid[f"{w:.1f}"], rule.grid.index(w)))
    m10 = g.measure(g.item(group_b.weight_item(chosen)), g.base)
    v10 = g.verdict(m10)
    out["c10"] = {
        "grid_rmse": grid,
        "base_rmse": out["base"]["rmse"],
        "chosen": chosen,
        "measured": m10,
        "verdict": v10,
        "line": group_b_line(
            "C10", v10, f" (w = {chosen:.1f}, chosen among four weights)"
        ),
    }
    out["alone"] = {}
    for key in ("c11", "c12", "c13"):
        ma = g.measure(g.item(key), g.base)
        va = g.verdict(ma)
        out["alone"][key] = {
            "measured": ma,
            "verdict": va,
            "line": group_b_line(key.upper(), va),
        }
    choices = g.c14_choices()
    out["c14_choices"] = choices
    m14 = g.measure(g.c14_of(choices), g.base)
    v14 = g.verdict(m14)
    out["c14"] = {"measured": m14, "verdict": v14, "line": group_b_line("C14", v14)}
    out["combination"] = group_b_combination(g, out)
    out["derived"] = dict(g.derived)
    out["checked"] = dict(g.checked)
    return out


def group_b_combination(g: GroupB, got: dict) -> dict:
    """§4 of the airports amendment: C14's choice if it holds, else the last-week combination."""
    half = board_rule().blend_weight
    clears = {
        "c10": got["c10"]["verdict"]["status"] == "clears",
        "c11": got["alone"]["c11"]["verdict"]["status"] == "clears",
        "c12": got["alone"]["c12"]["verdict"]["status"] == "clears",
        "c13": got["alone"]["c13"]["verdict"]["status"] == "clears",
        "c14": got["c14"]["verdict"]["status"] == "clears",
        "w": bool(got["w"]["clears"]),
    }
    if clears["c14"]:
        new_of, words = g.c14_of(got["c14_choices"]), ["C14's choice per airport"]
    else:
        bkey = "c12_booster" if clears["c12"] else "combo4_booster"
        mkey = "c11_member" if clears["c11"] else "c3b_member"
        wt = float(got["c10"]["chosen"]) if clears["c10"] else half
        if not (clears["c10"] or clears["c11"] or clears["c12"]):
            new_of, words = g.base, []
        else:

            def new_of(s, seed, bkey=bkey, mkey=mkey, wt=wt):
                return g.blended("nb", s, seed, bkey, mkey, wt)

            words = [
                f"booster {'C12' if clears['c12'] else '`combo4_booster`'}",
                f"member {'C11' if clears['c11'] else '`c3b_member`'}",
                f"w = {wt:.1f}{' (C10)' if clears['c10'] else ''}",
            ]
    if clears["w"]:

        def comb_of(s, seed):
            return g.w_layer(new_of, s, seed, "groupb")

        words.append("W on top, its means from the new base's out-of-fold residuals")
    else:
        comb_of = new_of
    out = {"clears": clears, "configuration": words}
    if not words:
        out["line"] = "GROUP B: nothing, no v11"
        out["why"] = "nothing in group B cleared; the configuration is v9's"
        return out
    m = g.measure(comb_of, g.base)
    v = g.verdict(m)
    five = cb.clears(v["conditions"], cb.FIVE)
    winter = g.w_rule(m) if clears["w"] else None
    winter_ok = winter is None or all(
        winter[k]["holds"] for k in ("january", "february", "december", "year")
    )
    config = "; ".join(words)
    out.update(
        measured=m,
        verdict=v,
        five=five,
        winter=winter,
        winter_ok=winter_ok,
        line=f"GROUP B: {config}" if five and winter_ok else "GROUP B: nothing, no v11",
    )
    if not (five and winter_ok):
        out["why"] = f"the combination ({config}) fails: " + group_b_why_not(v)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--amendment", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", choices=("reference",))
    mode.add_argument("--stage", type=int, choices=sorted(stages()) + [6, 7])
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--c8c", choices=("check", "alone", "combination", "report"))
    ap.add_argument("--prior", type=Path)
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = ap.parse_args(argv)
    point_harness()
    if args.c8c is not None:
        if args.prior is None:
            ap.error(
                "--c8c needs --prior, the working note of the amendment of 6 October"
            )
        if args.c8c == "report":
            return write_report_c8c(args.amendment, args.prior, args.date)
        run = {
            "check": run_c8c_check,
            "alone": run_c8c_alone,
            "combination": run_c8c_combination,
        }[args.c8c]
        return run(args.amendment, args.prior)
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
