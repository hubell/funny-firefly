# SPDX-License-Identifier: GPL-3.0-only
"""Training rows weighted toward the target month's regime.

Candidate C8 of the amendment of 4 October 2026 to the board pre-registration
(``preregistration_2026-10-04_c7_c8.md``). For a fit whose target rows are one
month's:

- a classifier separates the target rows (label 1) from that fit's training
  rows (label 0): LightGBM, objective binary, 200 trees, learning rate 0.05,
  31 leaves, minimum child 500, deterministic, column-wise, eight threads,
  seed 20260902; its features the airport, ``WEATHER_FEATURES`` and
  ``CONGESTION_FEATURES`` as the rung builds them, and no calendar or date
  field;
- each training row's probability p comes from two-fold cross-fitting, the
  folds being the even and the odd calendar days (of the take-off, UTC): the
  classifier fitted on one parity's rows, both labels, predicts the other
  parity's training rows;
- its weight is p / (1 - p), clipped to [0.1, 10], then scaled to a mean of 1
  over the training rows.

The booster is then refitted with those weights under ``BOOSTER_PARAMS``; that
is the caller's (``Booster.fit``'s ``sample_weight``). "That fit's training
rows" are the rows the booster is fitted on, ``taxiout.models.fit_rows``.

Candidate C8c of the amendment of 6 October 2026
(``preregistration_2026-10-06_c8_prior.md``) is C8 with one change, the
``class_ratio`` option, off by default: each training row's odds are
multiplied, before the clip, by n0 / n1, where n0 and n1 are the counts of
label 0 and label 1 among the rows the classifier that scored that row was
fitted on. A classifier fitted on n1 target rows against n0 training rows
estimates its odds with the prior n1 / n0 built in; the factor takes it out,
which leaves the density ratio of the target month's regime to the training
rows'. So the weight is (p / (1 - p)) x (n0 / n1), clipped to [0.1, 10], then
scaled to a mean of 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxiout.congestion import CONGESTION_FEATURES
from taxiout.metar import WEATHER_FEATURES

AIRPORT = "icao"
CLASSIFIER_FEATURES: tuple[str, ...] = (AIRPORT,) + WEATHER_FEATURES + CONGESTION_FEATURES


def classifier_params() -> dict:
    """The amendment's classifier, in its words, as LightGBM parameters.

    Held in a function so that a new module carries no module-level number
    before the commit that registers it.
    """
    return {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 500,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 8,
        "random_state": 20260902,
        "verbose": -1,
    }


def clip_bounds() -> tuple[float, float]:
    """ "its weight p/(1 - p), clipped to [0.1, 10]"."""
    return 0.1, 10.0


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[list(CLASSIFIER_FEATURES)].copy()
    out[AIRPORT] = out[AIRPORT].astype(str).astype("category")
    return out


@dataclass
class RegimeWeights:
    """The weights of one fit, and what they were built from."""

    weights: np.ndarray
    p: np.ndarray
    clipped_low: int
    clipped_high: int
    raw_mean: float
    # C8c's: whether the odds were multiplied by n0 / n1, and each fold's counts
    class_ratio: bool | None = None
    folds: tuple[dict, ...] = ()

    def distribution(self) -> dict:
        """Minimum, quartiles, maximum and the share clipped, after scaling.

        With the class-ratio option, each fold's counts and factor beside them;
        without it, exactly C8's figures.
        """
        w = self.weights
        q = np.quantile(w, [0.0, 0.25, 0.5, 0.75, 1.0])
        n = len(w)
        extra = (
            {"class_ratio": True, "folds": [dict(f) for f in self.folds]}
            if self.class_ratio
            else {}
        )
        return {
            "rows": int(n),
            "min": float(q[0]),
            "q1": float(q[1]),
            "median": float(q[2]),
            "q3": float(q[3]),
            "max": float(q[4]),
            "clipped_low": int(self.clipped_low),
            "clipped_high": int(self.clipped_high),
            "share_clipped": float((self.clipped_low + self.clipped_high) / n)
            if n
            else 0.0,
            "raw_mean": float(self.raw_mean),
            **extra,
        }


def regime_weights(
    frame: pd.DataFrame,
    takeoff: pd.Series,
    train_rows: pd.Series,
    target_rows: pd.Series,
    class_ratio: bool = False,
) -> RegimeWeights:
    """Weights for the ``train_rows`` of ``frame``, toward ``target_rows``.

    ``frame`` carries ``CLASSIFIER_FEATURES``; ``takeoff`` is the take-off
    time, read for the day's parity and nothing else. The weights come back
    in the order of ``frame.loc[train_rows]``. ``class_ratio`` is C8c's
    factor; off, the weights are C8's bit for bit.
    """
    import lightgbm as lgb

    train = train_rows.to_numpy(dtype=bool)
    target = target_rows.to_numpy(dtype=bool)
    if (train & target).any():
        raise ValueError("a row is both a training row and a target row")
    day = pd.to_datetime(takeoff, utc=True).dt.day.to_numpy()
    odd = (day % 2).astype(bool)
    use = train | target
    X = _design(frame)
    label = target.astype(int)
    p = np.full(len(frame), np.nan)
    factor = np.full(len(frame), np.nan)
    folds = []
    for fit_parity in (False, True):
        fit_on = use & (odd == fit_parity)
        score = train & (odd != fit_parity)
        n1 = int(label[fit_on].sum())
        n0 = int(fit_on.sum()) - n1
        if class_ratio and (n0 == 0 or n1 == 0):
            raise ValueError("a fold's classifier would be fitted on one label alone")
        model = lgb.LGBMClassifier(**classifier_params())
        model.fit(X.loc[fit_on], label[fit_on], categorical_feature=[AIRPORT])
        p[score] = model.predict_proba(X.loc[score])[:, 1]
        factor[score] = n0 / n1 if n1 else np.nan
        folds.append(
            {
                "fitted_on": "odd days" if fit_parity else "even days",
                "n0": n0,
                "n1": n1,
                "factor": n0 / n1 if n1 else None,
                "scored": int(score.sum()),
            }
        )
    p_train = p[train]
    if not np.isfinite(p_train).all():
        raise RuntimeError("a training row has no cross-fitted probability")
    low, high = clip_bounds()
    with np.errstate(divide="ignore"):
        raw = p_train / (1.0 - p_train)
    if class_ratio:
        raw = raw * factor[train]
    clipped = np.clip(raw, low, high)
    scaled = clipped / clipped.mean()
    return RegimeWeights(
        weights=scaled,
        p=p_train,
        clipped_low=int((raw < low).sum()),
        clipped_high=int((raw > high).sum()),
        raw_mean=float(clipped.mean()),
        class_ratio=True if class_ratio else None,
        folds=tuple(folds),
    )


__all__ = [
    "CLASSIFIER_FEATURES",
    "RegimeWeights",
    "classifier_params",
    "clip_bounds",
    "regime_weights",
]
