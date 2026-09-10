#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Leave one airport out, all ten, and write ``reports/transfer.md``.

Ten fits. Each trains on nine airports and predicts the tenth, and nothing
about the tenth that a model could have memorised is allowed in: no airport,
stand, runway or pair identifier, and no airport-level quantity fitted on the
held-out airport's own rows. What is allowed is the tenth airport's **own
geometry** -- the distance matrix built from OpenStreetMap, which exists for an
airport whether or not anybody has ever flown out of it -- and the exogenous
state of a movement: the clock, the aircraft, the plan lags, the traffic in the
window, the weather. Everything fitted is fitted across the other nine.

The baseline is the grand mean of the other nine, because that is what a model
without geometry has at an airport it has never seen. It is not a straw man; it
is the whole of what an identifier-based entry can say there. That is the point
of this report: **no identifier-based entry can produce this table at all.** At
an unseen airport there is no stand to encode, no pair to look up and no
per-airport constant to apply, and the question the challenge's test set cannot
pose -- what happens at the eleventh airport -- is the operational question.

Three predictors, in increasing order of what they read:

1. **the grand mean** of the other nine airports' taxi-out;
2. **the geometric unimpeded term**, fitted across the other nine with the
   pooled intercept and pooled speed applied to the tenth, plus the pooled
   additive offset that puts an unimpeded estimate on the scale of an observed
   taxi-out. Two numbers and five shared modulators, and not one of them fitted
   on the held-out airport;
3. **the transfer model**: the submission rung's feature block with every
   identifier removed (``taxiout.models.TRANSFER_FEATURES``), boosted, fitted
   across the other nine.

The number worth the most here is not an RMSE. It is the *speed*: the pooled
metres-per-second the geometric fit recovers, once per fold, from nine
airports that change every time. If that number is stable across ten fits, the
distance-to-time relation is a transferable physical constant rather than a
per-airport fudge, and the whole premise of the entry holds outside the ten
airports it was built on.

Needs the challenge data under ``data/challenge/`` and the matrices under
``data/processed/taxi_distance/`` (``make reports``), plus the modelling
extras. Neither dataset is committed. Takes an hour or more: ten boosted fits
on nine tenths of the training year each.

Usage
-----
    python scripts/report_transfer.py
    python scripts/report_transfer.py --dry-run --sample-frac 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from taxiout.features import TARGET  # noqa: E402
from taxiout.models import (  # noqa: E402
    TRANSFERABLE_EXCLUDED,
    TRANSFER_FEATURES,
    Booster,
    booster_backend,
    fit_rows as fitting_rows,
)
from taxiout.unimpeded import (  # noqa: E402
    fit_geometric,
    fit_pru_reference,
    lower_tail,
)
from taxiout.validation import evaluate, rmse  # noqa: E402

from run_baselines import (  # noqa: E402
    DATA,
    Tails,
    airport_fallback,
    build_features,
    check_no_identifiers,
    load_all_movements,
    load_departures,
    load_edge_sets,
    load_matrix,
    load_airport_table,
    load_weather,
    num,
    pct,
    table,
)

OUT = REPO / "reports" / "transfer.md"


def leave_one_out(features, y, unflagged, icao, airport) -> dict:
    """Fit the three predictors across every airport but ``airport``.

    ``unflagged`` is the record-quality mask, used identically for every
    predictor and for the baseline, so it cannot favour one over another. It is
    an annotation on the data and not an input to any fit: the fits exclude the
    flagged rows, and the evaluation reports the RMSE with and without them.
    """
    # The same fit-row rule the ladder and the submission use, so this report
    # and reports/baseline.md are describing one model and not two.
    train = fitting_rows(icao != airport, ~unflagged, y)
    test = icao == airport
    out: dict = {"airport": airport, "train_rows": int(train.sum())}

    grand = float(pd.to_numeric(y[train], errors="coerce").mean())
    out["grand_mean_s"] = grand
    out["pred_mean"] = pd.Series(grand, index=y.index[test], dtype=float)

    ref = fit_pru_reference(features, y, mask=train)
    geo = fit_geometric(
        features, ref, lower_tail(features, y, ref, mask=train), class_column=None
    )
    raw_train = geo.predict(features.loc[train])
    offset = float((pd.to_numeric(y[train], errors="coerce") - raw_train).mean())
    out["speed_ms"] = float(geo.default_speed)
    out["intercept_s"] = float(geo.default_intercept)
    out["offset_s"] = offset
    out["n_fit_geo"] = int(geo.n_fit)
    out["pooled_airports"] = geo.pooled_airports
    out["pred_geo"] = geo.predict(features.loc[test]) + offset

    model = Booster(TRANSFER_FEATURES)
    model.fit(features.loc[train], y[train])
    out["pred_model"] = model.predict(features.loc[test])
    out["importances"] = model.importances()
    out["test"] = test
    return out


def _scores(y, test, unflagged, preds: dict) -> dict:
    keep = test & unflagged
    return {
        name: {
            "rmse": rmse(y[test], p),
            "rmse_ex": rmse(y[keep], p.loc[y.index[keep]]),
        }
        for name, p in preds.items()
    }


def section_speeds(folds: list[dict]) -> list[str]:
    rows = [
        [
            f["airport"],
            f"{f['train_rows']:,}",
            f"{f['speed_ms']:.2f}",
            f"{3.6 * f['speed_ms']:.1f}",
            num(f["intercept_s"]),
            num(f["offset_s"]),
            f"{f['n_fit_geo']:,}",
        ]
        for f in folds
    ]
    speeds = np.array([f["speed_ms"] for f in folds])
    return [
        "## The speed, ten times over",
        "",
        "**This is the result.** Each row is a fit that has never seen the airport",
        "named in it. The speed is the pooled inverse-speed of the geometric",
        "unimpeded term -- one number for the whole fit, the answer it gives an",
        "airport it has no coefficient for -- and the intercept is the pushback,",
        "start-up and line-up component that no distance explains. The offset is the",
        "constant that carries an *unimpeded* estimate onto the scale of an observed",
        "taxi-out, and is the excess term held fixed.",
        "",
        *table(
            [
                "held out",
                "train rows",
                "m/s",
                "km/h",
                "intercept (s)",
                "offset (s)",
                "lower-tail rows",
            ],
            rows,
            "-rrrrrr",
        ),
        "",
        f"The speed runs from **{speeds.min():.1f} to {speeds.max():.1f} m/s** across the ten fits, a spread of",
        f"{100 * (speeds.max() - speeds.min()) / speeds.mean():.0f}% about a mean of {speeds.mean():.2f} m/s"
        f" ({3.6 * speeds.mean():.0f} km/h) -- and every fit is on a",
        "different nine airports. A quantity that moves this little when the",
        "training set is replaced is a physical constant of taxiing, not a",
        "coefficient fitted to a place. That is the claim the geometry rests on, and",
        "this is the only table in the repository that tests it by removing an",
        "airport rather than a pair.",
    ]


def section_table(folds: list[dict], scores: dict, ex: bool) -> list[str]:
    rows = []
    for f in folds:
        a = f["airport"]
        s = scores[a]
        key = "rmse_ex" if ex else "rmse"
        base = s["grand mean"][key]
        # The count the RMSE beside it was computed over. When ``ex``, that is
        # ``test & unflagged`` and not the fold's whole test set: printing
        # ``test_rows`` in both tables put LIRF at the same 160,704 rows either
        # side of a grand mean that moved from 613 s to 1,350 s, which cannot
        # both be true and read as a finding rather than as the bug it was.
        n = f["test_rows"] - f["flagged_rows"] if ex else f["test_rows"]
        rows.append(
            [
                a,
                f"{int(n):,}",
                num(base, 1),
                num(s["geometric"][key], 1),
                pct((base - s["geometric"][key]) / base),
                num(s["transfer model"][key], 1),
                pct((base - s["transfer model"][key]) / base),
                "yes" if s["transfer model"][key] < base else "**no**",
            ]
        )
    key = "rmse_ex" if ex else "rmse"
    beats = sum(
        1
        for f in folds
        if scores[f["airport"]]["transfer model"][key]
        < scores[f["airport"]]["grand mean"][key]
    )
    geo_beats = sum(
        1
        for f in folds
        if scores[f["airport"]]["geometric"][key]
        < scores[f["airport"]]["grand mean"][key]
    )
    gains = np.array(
        [
            (
                scores[f["airport"]]["grand mean"][key]
                - scores[f["airport"]]["transfer model"][key]
            )
            / scores[f["airport"]]["grand mean"][key]
            for f in folds
        ]
    )
    best = folds[int(np.argmax(gains))]["airport"]
    # Where the mask actually bites. Printed rather than left to a reader,
    # because nine of the ten rows are identical between the two tables and a
    # reader cannot otherwise tell whether that is a finding or a bug.
    hit = [(f["airport"], f["flagged_rows"]) for f in folds if f["flagged_rows"]]
    return [
        *table(
            [
                "held out",
                "kept rows" if ex else "rows",
                "grand mean (s)",
                "geometric (s)",
                "vs mean",
                "transfer model (s)",
                "vs mean",
                "beats the mean",
            ],
            rows,
            "-rrrrrr-",
        ),
        "",
        f"The transfer model beats the grand mean at **{beats} of {len(folds)}** airports, median",
        f"**{pct(float(np.median(gains)))}**, best {pct(float(gains.max()))} at {best}. The geometric term alone --"
        f" two fitted numbers and five shared modulators -- beats it at {geo_beats} of {len(folds)}.",
        *(
            [
                "",
                "The record-quality mask removes "
                + ", ".join(f"{n:,} rows at {a}" for a, n in hit)
                + f" and none at the other {len(folds) - len(hit)} on this training"
                " year, so every other row of this table is identical to the"
                " all-rows table below **by construction** rather than by finding."
                " The term is fitted per (airport, cell) with no pooled fallback and"
                " covers only cells that showed the mechanism in 2025"
                " (`tails.md`), which on this data is one airport; that is a"
                " property of the fit and not a guarantee about 2026.",
            ]
            if ex and hit
            else []
        ),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to run", file=sys.stderr)
        return 1
    if booster_backend() == "none":
        print('no booster installed; pip install -e ".[modelling]"', file=sys.stderr)
        return 1
    if args.sample_frac and not args.dry_run:
        print("--sample-frac writes a report that is not the report", file=sys.stderr)
        return 1

    print("loading ...")
    matrix = load_matrix()
    dep = load_departures(args.sample_frac)
    y = pd.to_numeric(dep[TARGET], errors="coerce")

    print("building features ...")
    t0 = time.time()
    movements = load_all_movements()
    weather = load_weather()
    features = build_features(
        dep,
        matrix,
        fallback=airport_fallback(matrix),
        movements=movements,
        edges=load_edge_sets(),
        weather=weather,
        airports=load_airport_table(),
    )
    del movements
    print(f"  {len(features):,} rows in {time.time() - t0:.0f} s")

    # The record-quality mask, for the fits to exclude and the report to
    # subtract. Fitted over the whole year: it is an annotation on the data --
    # which rows carry a defaulted block time -- and not a predictor. It enters
    # the baseline and the two models identically, so it cannot favour either.
    tails = Tails()
    if tails.available:
        flagged = tails.fit(dep).predict(dep).notna()
    else:
        flagged = pd.Series(False, index=dep.index)
        print("WARNING: no record-quality layer; every RMSE carries the artefacts")
    unflagged = ~flagged

    icao = features["icao"].astype(str)
    airports = sorted(icao.unique())
    print(f"leaving out one of {len(airports)} airports at a time ...")

    folds, scores = [], {}
    for a in airports:
        t0 = time.time()
        f = leave_one_out(features, y, unflagged, icao, a)
        test = f["test"]
        f["test_rows"] = int(test.sum())
        f["flagged_rows"] = int((test & flagged).sum())
        scores[a] = _scores(
            y,
            test,
            unflagged,
            {
                "grand mean": f["pred_mean"],
                "geometric": f["pred_geo"],
                "transfer model": f["pred_model"],
            },
        )
        f["eval"] = evaluate(
            y[test], f["pred_model"], icao=icao[test], flagged=flagged[test]
        )
        for k in ("pred_mean", "pred_geo", "pred_model", "test"):
            f.pop(k)
        folds.append(f)
        print(
            f"  {a}  mean {scores[a]['grand mean']['rmse_ex']:7.1f}  "
            f"geo {scores[a]['geometric']['rmse_ex']:7.1f}  "
            f"model {scores[a]['transfer model']['rmse_ex']:7.1f} s "
            f"({time.time() - t0:.0f} s)"
        )

    imp = folds[0]["importances"]
    total = float(imp.sum()) or 1.0
    top = [[f"`{c}`", f"{v / total:.1%}"] for c, v in imp.head(12).items()]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    head = [
        "# The eleventh airport",
        "",
        "Leave one airport out, ten times. Train on nine, predict the tenth, and",
        "let nothing about the tenth into the fit but its own taxiway geometry.",
        "Generated by `scripts/report_transfer.py` from the data under",
        f"`data/challenge/` on {generated}; every figure below is computed there,",
        "and the generator refuses to write a literal stand identifier.",
        "",
        (
            f"**Development run: a {args.sample_frac:.0%} sample. Not the report.**"
            if args.sample_frac
            else f"{len(dep):,} training departures: the 2025 movement files in full."
        ),
        "",
        "## The question this answers",
        "",
        "The challenge scores January and July 2026 at ten airports the training",
        "year already contains. Every split in `baseline.md` is therefore a split",
        "*within* those ten: the held-out-pair folds take away a pair and keep the",
        "airport, which bounds what the geometry buys on a new stand but says",
        "nothing about a new *field*. The operational question is the other one.",
        "An air navigation service provider with a taxi-out model wants to know",
        "what it does at the airport it has not instrumented yet, and no split of",
        "this data as the challenge poses it can answer that.",
        "",
        "This one can, because the geometry is not fitted from the movements. The",
        "distance matrix for an airport is built from OpenStreetMap, and it exists",
        "for an airport nobody has ever flown a recorded departure from. So an",
        "airport can be removed from the training set entirely and still be",
        "predicted -- through its own taxiways, by coefficients fitted somewhere",
        "else.",
        "",
        "**An identifier-based entry cannot produce this table at all.** Not worse:",
        "at all. At an unseen airport there is no stand to look up, no pair to",
        "encode, no per-airport constant to apply, and nothing left but the pooled",
        "mean -- which is the baseline column below.",
        "",
        "## What each predictor is allowed to read",
        "",
        "| | reads | fitted on |",
        "| --- | --- | --- |",
        "| grand mean | nothing | the other nine airports' taxi-out |",
        "| geometric | the held-out airport's own matrix cell | the other nine |",
        "| transfer model | the same, plus the clock, the aircraft, the plan lags, "
        "the traffic in the window and the weather | the other nine |",
        "",
        "No identifier of any kind reaches the last two: "
        + ", ".join(f"`{c}`" for c in TRANSFERABLE_EXCLUDED)
        + " are removed from the feature block, and the geometric term answers an",
        "unknown airport with its pooled intercept and pooled speed because it has",
        "no coefficient for that airport to answer with. The record-quality mask is",
        "applied identically to the baseline and to both models, and is an",
        "annotation on the data rather than an input to any fit.",
        "",
        f"The transfer model reads {len(TRANSFER_FEATURES)} columns.",
        "",
    ]

    body = (
        head
        + section_speeds(folds)
        + [
            "",
            "## RMSE, per held-out airport",
            "",
            "Over the rows the record-quality term does not cover. At an unseen",
            "airport that term has no cells of its own -- it is fitted per airport --",
            "so it can predict nothing there, and the honest reading is the one with",
            "its rows subtracted from every column alike. The all-rows table follows.",
            "",
        ]
        + section_table(folds, scores, ex=True)
        + [
            "",
            "### Every row, for the record",
            "",
            "The same table with the record-quality rows left in. Fiumicino's",
            "defaulted block times dominate it, which is what `reports/tails.md`",
            "measured and why the table above is the one to read.",
            "",
        ]
        + section_table(folds, scores, ex=False)
        + [
            "",
            "## Where the transfer model spends its splits",
            "",
            "The first fold's model, as a check that the geometry is what is carrying",
            "it rather than the clock. Split counts, not a causal statement.",
            "",
            *table(["feature", "share of splits"], top, "-r"),
            "",
            "## What this does not show",
            "",
            "Three limits, stated rather than left to a reader.",
            "",
            "The ten airports are all large European hubs with mapped taxiways. A",
            "regional field with a single runway and a hundred movements a day is",
            "not represented here, and the speed's stability across these ten is",
            "evidence about airports like these.",
            "",
            "The held-out airport's *own* movements are still read by the congestion,",
            "corridor and saturation blocks -- they count the traffic around a",
            "departure, which is exogenous state and not a fitted quantity, but it is",
            "state the model would need at the eleventh airport too. The geometric",
            "row is free of even that: it reads the matrix cell and nothing else, and",
            "it is the row to quote when the question is what pure geometry",
            "transfers.",
            "",
            "The weather block reads that airport's own METAR archive. Again",
            "exogenous, again something the eleventh airport would have, and again",
            "absent from the geometric row.",
            "",
            # One line, deliberately. The attribution check matches the literal
            # string "OpenStreetMap contributors", and wrapping it across a
            # newline is exactly the way this file failed the gate the first
            # time it was generated.
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
