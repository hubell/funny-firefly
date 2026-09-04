#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The one parameter sweep: fit January to September 2025, score October.

Permitted once, on October 2025 alone, and never reported: the ladder's
splits -- December, July, the held-out pairs -- are the sets the write-up
reads, so nothing may be chosen on them, and the leaderboard is not a
validation set. October is a month the ladder never scores, so a number
read from it chooses parameters without touching what is reported.

One grid, over the booster's trees, learning rate, leaves and minimum child
samples, on the feature block of ``SWEEP_RUNG`` -- the saturation rung, which
was the top of the ladder when the sweep ran; each cell fitted on the unflagged
rows of January to September, as the ladder fits, and scored by the RMSE
over October's rows the record-quality term leaves to the physical model.
The number of trees is read from one fit per cell of the other three: the
first ``k`` trees of a boosted model are the same model whatever the total,
so a 1,000-tree fit scores 300, 600 and 1,000 at once.

The chosen parameters are written into ``BOOSTER_PARAMS`` in
``taxiout.models`` by hand, with the October number that chose them, and
the ladder is run once more. The grid's own numbers stay under
``data/processed/``, which is gitignored: they are not a result, they are
how the parameters were chosen, and the README says so.

Usage
-----
    python scripts/sweep_params.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_baselines import (  # noqa: E402
    Tails,
    load_all_movements,
    load_departures,
    load_edge_sets,
    load_matrix,
    load_weather,
)
from taxiout.features import build_features  # noqa: E402
from taxiout.models import BOOSTER_PARAMS, LADDER_BY_KEY, Booster  # noqa: E402
from taxiout.features import airport_fallback  # noqa: E402
from taxiout.validation import month_of, rmse  # noqa: E402

TARGET = "TAXITIME_SEC_mvt"
OUT = REPO / "data" / "processed" / "sweep_october.csv"

# ``month_of`` yields ``YYYY-MM``.
TRAIN_MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 10))
SCORE_MONTH = "2025-10"

# The rung the sweep is fitted on, pinned by name rather than read from
# ``TOP_RUNG``. The sweep ran once, on the saturation rung, when that was the
# top of the ladder; the saturation block has since been dropped from the
# submission and ``TOP_RUNG`` has moved down to the weather rung. Following
# ``TOP_RUNG`` here would silently re-run the sweep on a different feature
# block and stop reproducing the October numbers recorded in
# ``taxiout.models``, which is the one place the choice of parameters is
# accounted for.
SWEEP_RUNG = LADDER_BY_KEY["saturation"]

GRID = {
    "learning_rate": (0.03, 0.06, 0.1),
    "num_leaves": (31, 63, 127),
    "min_child_samples": (50, 200, 500),
}
TREES = (300, 600, 1000)


def _score_at(booster: Booster, X: pd.DataFrame, y: np.ndarray, trees: int) -> float:
    """RMSE of the first ``trees`` trees, with the wrapper's own categorical coding."""
    design = X[list(booster.features)].copy()
    for c in booster.categorical:
        levels = booster._levels.get(c)
        if levels is not None:
            design[c] = pd.Categorical(design[c].astype(str), categories=levels)
        else:
            design[c] = design[c].astype("category")
    pred = booster.model.predict(design, num_iteration=trees)
    return rmse(y, pred)


def main() -> int:
    print("loading ...")
    matrix = load_matrix()
    dep = load_departures(None)
    y = pd.to_numeric(dep[TARGET], errors="coerce")
    movements = load_all_movements()
    features = build_features(
        dep,
        matrix,
        fallback=airport_fallback(matrix),
        movements=movements,
        edges=load_edge_sets(),
        weather=load_weather(),
    )
    del movements
    month = month_of(dep["MVT_TIME_UTC_mvt"])
    train = month.isin(list(TRAIN_MONTHS))
    score = month == SCORE_MONTH
    print(
        f"  {int(train.sum()):,} training rows (Jan-Sep), {int(score.sum()):,} to score (Oct)"
    )

    tails = Tails()
    tail_model = tails.fit(dep.loc[train]) if tails.available else None
    if tail_model is not None:
        flagged = tail_model.predict(dep).notna()
    else:
        flagged = pd.Series(False, index=dep.index)
    fit_rows = train & ~flagged
    score_rows = score & ~flagged
    print(
        f"  fitted on {int(fit_rows.sum()):,} unflagged rows; scored on {int(score_rows.sum()):,}"
    )
    if fit_rows.sum() == 0 or score_rows.sum() == 0:
        print(
            "no rows to fit or to score; check TRAIN_MONTHS and SCORE_MONTH",
            file=sys.stderr,
        )
        return 1
    X_fit, y_fit = features.loc[fit_rows], y[fit_rows]
    X_sc, y_sc = features.loc[score_rows], y[score_rows].to_numpy(dtype=float)

    if booster_backend_is_not_lightgbm():
        print("the sweep reads trees from one fit and needs lightgbm", file=sys.stderr)
        return 1

    rows = []
    cells = list(itertools.product(*GRID.values()))
    for i, values in enumerate(cells, 1):
        params = dict(BOOSTER_PARAMS)
        params.update(dict(zip(GRID.keys(), values)))
        params["n_estimators"] = max(TREES)
        t0 = time.time()
        b = Booster(SWEEP_RUNG.features, params).fit(X_fit, y_fit)
        for trees in TREES:
            r = _score_at(b, X_sc, y_sc, trees)
            rows.append(
                {**dict(zip(GRID.keys(), values)), "n_estimators": trees, "rmse_oct_s": r}
            )
        print(
            f"  [{i:2d}/{len(cells)}] "
            + ", ".join(f"{k}={v}" for k, v in zip(GRID.keys(), values))
            + " -> "
            + ", ".join(
                f"{t}: {r['rmse_oct_s']:.1f}" for t, r in zip(TREES, rows[-len(TREES) :])
            )
            + f"  ({time.time() - t0:.0f} s)"
        )
    out = pd.DataFrame(rows).sort_values("rmse_oct_s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    current = out[
        (out["learning_rate"] == BOOSTER_PARAMS["learning_rate"])
        & (out["num_leaves"] == BOOSTER_PARAMS["num_leaves"])
        & (out["min_child_samples"] == BOOSTER_PARAMS["min_child_samples"])
        & (out["n_estimators"] == BOOSTER_PARAMS["n_estimators"])
    ]
    print()
    print(out.head(8).to_string(index=False))
    if len(current):
        print(
            f"\ncurrent BOOSTER_PARAMS on October: {current['rmse_oct_s'].iloc[0]:.1f} s"
        )
    best = out.iloc[0]
    print(
        "\nbest: "
        + ", ".join(f"{k}={best[k]}" for k in (*GRID.keys(), "n_estimators"))
        + f" -> {best['rmse_oct_s']:.1f} s on October"
    )
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


def booster_backend_is_not_lightgbm() -> bool:
    from taxiout.models import booster_backend

    return booster_backend() != "lightgbm"


if __name__ == "__main__":
    raise SystemExit(main())
