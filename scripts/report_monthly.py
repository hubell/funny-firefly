#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Every month of 2025, held out in turn, and write ``reports/monthly.md``.

The paper quotes two validation months, December and July, because the design
of 2 September fixed two forward splits as the proxies for the two ranking
months. This report reads every month of the training year, per airport, in
seconds. A forward fit cannot validate January, the training year's first
month, and a forward fit of February or March would train on one or two months
and mean little. So each month is held out in turn and the submitted rung is
trained on the other eleven: the same rows every month, a design that is **not
forward in time** and is said to be so wherever its figures appear.

**It changes nothing that ships, chooses nothing, and decides nothing.** No
parameter, no feature and no rule in ``taxiout.models`` is changed by this
script. It passes nothing to the fitting code but a list of training months and
a validation month, as ``scripts/preregistered.py`` does, and it is read for one
thing: how the submitted rung's error is distributed over airports and months.

Fourteen legs, each one fit of ``SUBMISSION_KEY`` at the ladder's seed under
``BOOSTER_PARAMS``. Inside every leg, exactly as ``run_baselines.run`` does
inside its split loop: the record-quality model of ``taxiout.tails`` on the
leg's training months, ``fit_rows`` over those months (unflagged, the
recorded-taxi tail over ``TAIL_EXCLUSION_ADOPTED_H`` out of the fit, fit side
only), and the shrunk encoding recomputed from those rows and overlaid.

The two forward legs, December and July, run first, and a failure stops the
run: their fits must reproduce the ladder's stored residuals with a largest
absolute difference of zero, joined on the movement identifier, which is never
printed. That is what makes the twelve held-out fits the ladder's own fitting
path and not a lookalike.

Three modes:

``--dry-run``
    the fourteen legs with their training months and row counts; fits nothing.
``--fit``
    builds the frame once, fits the fourteen legs and stores one parquet of
    validation residuals per leg under ``data/processed/monthly/``, gitignored
    because it carries the target. 1 h 13 min 45 s on eight cores, measured on
    1 October 2026.
``--report``
    reads the stored residuals and writes the report; refits nothing.

The report carries aggregates only, and the generator refuses to write a
movement identifier or a literal stand identifier. ``reports/`` and
``data/processed/residuals/`` are fingerprinted before every mode and checked
after it.

Usage
-----
    python scripts/report_monthly.py --dry-run
    python scripts/report_monthly.py --fit
    python scripts/report_monthly.py --report --out reports/monthly.md
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_baselines as rb  # noqa: E402
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.encoding import encode_split, with_encoding  # noqa: E402
from taxiout.features import TARGET, airport_fallback, build_features  # noqa: E402
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER_BY_KEY,
    SUBMISSION_KEY,
    TAIL_EXCLUSION_ADOPTED_H,
    Booster,
    fit_rows as fitting_rows,
)
from taxiout.provenance import report_inputs_footer  # noqa: E402
from taxiout.validation import (  # noqa: E402
    DECEMBER_TRAIN,
    DECEMBER_VALID,
    JULY_TRAIN,
    JULY_VALID,
    month_of,
)

WORK = REPO / "data" / "processed" / "monthly"
RUN_RECORD = "run.json"
DEFAULT_OUT = REPO / "reports" / "monthly.md"

# The two directories this script may never write, checked rather than
# promised, as ``preregistered.py`` checks them.
FORBIDDEN = (REPO / "reports", REPO / "data" / "processed" / "residuals")

# The twelve months of the training year, each held out once.
MONTHS: tuple[str, ...] = tuple(f"2025-{m:02d}" for m in range(1, 13))

# The ladder's seed: the offset added to ``BOOSTER_PARAMS["random_state"]``,
# 0 being the model every rung of the ladder is fitted with.
SEED = 0

# The forward legs' figures ``reports/baseline.md`` prints for
# ``+ plan differences`` (the ladder tables, lines 234 and 270): excluding
# flagged rows, every row with the term, every row without it.
REF_FORWARD: dict[str, dict[str, float]] = {
    "december": {"ex_flagged": 214.7, "with_term": 234.0, "booster": 364.8},
    "july": {"ex_flagged": 322.4, "with_term": 368.2, "booster": 620.5},
}
PRINTED_PRECISION_S = rb.PRINTED_PRECISION_S

# The airport whose rows the record-quality term covers. Every covered cell of
# ``reports/tails.md`` is LIRF; a term row anywhere else is a check failure.
TERM_AIRPORT = "LIRF"

# The stored files' columns: ``run_baselines.write_residuals``'s six, plus the
# residual with the record-quality term substituted on the rows it covers,
# which the ladder keeps in memory and this report needs on disk because
# ``--report`` refits nothing.
RESIDUAL_COLUMNS: tuple[str, ...] = (
    "mvt_id",
    "icao",
    "runway",
    "takeoff",
    "residual",
    "flagged",
)
TERM_COLUMN = "residual_with_term"


# --------------------------------------------------------------------------
# the legs


@dataclass
class Leg:
    name: str
    kind: str  # "forward" or "held-out"
    train_months: tuple[str, ...]
    valid_month: str
    description: str
    train: pd.Series | None = field(default=None, repr=False)
    valid: pd.Series | None = field(default=None, repr=False)

    @property
    def stem(self) -> str:
        return f"{self.name}_{SUBMISSION_KEY}"


def leg_plan() -> list[Leg]:
    """The fourteen legs, the two forward ones first. Months only, no rows."""
    legs = [
        Leg(
            "forward-december",
            "forward",
            tuple(DECEMBER_TRAIN),
            DECEMBER_VALID[0],
            "the ladder's December split: train January to November, validate December",
        ),
        Leg(
            "forward-july",
            "forward",
            tuple(JULY_TRAIN),
            JULY_VALID[0],
            "the ladder's July split: train January to June, validate July",
        ),
    ]
    for month in MONTHS:
        legs.append(
            Leg(
                f"heldout-{month}",
                "held-out",
                tuple(m for m in MONTHS if m != month),
                month,
                f"train the other eleven months, validate {month}",
            )
        )
    for leg in legs:
        if leg.valid_month in leg.train_months:
            raise RuntimeError(f"{leg.name} trains on the month it validates")
    return legs


def build_legs(month: pd.Series) -> list[Leg]:
    """The fourteen legs with their row masks over ``month``."""
    legs = leg_plan()
    for leg in legs:
        leg.train = month.isin(leg.train_months)
        leg.valid = month == leg.valid_month
        if bool((leg.train & leg.valid).any()):
            raise RuntimeError(f"{leg.name}: a row is both training and validation")
    return legs


# --------------------------------------------------------------------------
# what this script may not write


def fingerprint() -> dict[str, tuple[int, int]]:
    """Size and modification time of every file this script must not write."""
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


def touched(before: dict, after: dict) -> list[str]:
    return sorted(
        {k for k in before if after.get(k) != before[k]} | (set(after) - set(before))
    )


# --------------------------------------------------------------------------
# the report refuses an identifier


def check_no_identifiers(text: str, stands: set[str], movements: set[str]) -> None:
    """Refuse to write a report carrying a stand or a movement identifier.

    The two rules of ``preregistered.check_no_identifiers``, both with a zero
    threshold: a stand identifier with a letter, a digit and at least three
    characters, ICAO codes exempt; and any movement identifier this run
    touched, as a whole word.
    """
    exempt = {a.icao for a in AIRPORTS}
    leaked = sorted(
        v
        for v in stands - exempt
        if len(v) >= 3
        and re.search(r"[A-Za-z]", v)
        and re.search(r"[0-9]", v)
        and v in text
        and re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", text)
    )
    if leaked:
        raise SystemExit(
            f"refusing to write the report: {len(leaked)} literal stand "
            "identifier(s) would be published"
        )
    ids = sorted(set(re.findall(r"(?<![\w])\d+(?![\w])", text)) & movements)
    if ids:
        raise SystemExit(
            f"refusing to write the report: {len(ids)} movement identifier(s) "
            "would be published"
        )


# --------------------------------------------------------------------------
# the frame, built once


def peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


class Frame:
    """The feature frame and the target, built as ``preregistered.Frame`` is."""

    def __init__(self) -> None:
        t0 = time.monotonic()
        self.tails = rb.Tails()
        if not self.tails.available:
            raise SystemExit(f"the record-quality term is required: {self.tails.reason}")
        matrix = rb.load_matrix()
        self.dep = rb.load_departures(None)
        self.y = pd.to_numeric(self.dep[TARGET], errors="coerce")
        movements = rb.load_all_movements()
        self.features = build_features(
            self.dep,
            matrix,
            fallback=airport_fallback(matrix),
            movements=movements,
            edges=rb.load_edge_sets(),
            weather=rb.load_weather(),
            airports=rb.load_airport_table(),
        )
        del movements
        self.build_seconds = time.monotonic() - t0
        self.month = month_of(self.dep["MVT_TIME_UTC_mvt"])
        self.legs = build_legs(self.month)
        print(
            f"frame: {len(self.features):,} rows in {self.build_seconds:.0f} s, "
            f"peak {peak_gib():.1f} GiB",
            flush=True,
        )


def fit_leg(frame: Frame, leg: Leg) -> tuple[pd.DataFrame, dict]:
    """One leg: the term, the fit rows, the encoding, one fit, one prediction."""
    t0 = time.monotonic()
    tail_model = frame.tails.fit(frame.dep.loc[leg.train])
    tail = tail_model.predict(frame.dep)
    flagged = tail.notna()
    rows = fitting_rows(leg.train, flagged, frame.y)
    encoding = encode_split(frame.features, frame.y, rows)
    encoded = with_encoding(frame.features, encoding)
    prepare_s = time.monotonic() - t0

    rung = LADDER_BY_KEY[SUBMISSION_KEY]
    params = dict(BOOSTER_PARAMS)
    params["random_state"] = int(BOOSTER_PARAMS["random_state"]) + SEED
    t1 = time.monotonic()
    model = Booster(tuple(rung.features), params).fit(encoded.loc[rows], frame.y[rows])
    fit_s = time.monotonic() - t1

    mask = leg.valid
    t2 = time.monotonic()
    pred = np.asarray(model.predict(encoded.loc[mask]), dtype=float)
    predict_s = time.monotonic() - t2
    y = np.asarray(frame.y[mask], dtype=float)
    with_term = frame.tails.combine(pred, tail[mask]).to_numpy(dtype=float)
    raw = frame.dep.loc[mask]
    out = pd.DataFrame(
        {
            "mvt_id": raw["MVT_ID_mvt"].to_numpy(),
            "icao": raw["ADEP_mvt"].astype(str).to_numpy(),
            "runway": raw["RUNWAY_mvt"].astype(str).to_numpy(),
            "takeoff": pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).to_numpy(),
            "residual": y - pred,
            "flagged": np.asarray(flagged[mask], dtype=bool),
            TERM_COLUMN: y - with_term,
        }
    )
    record = {
        "leg": leg.name,
        "kind": leg.kind,
        "train_months": list(leg.train_months),
        "valid_month": leg.valid_month,
        "train_rows": int(leg.train.sum()),
        "valid_rows": int(mask.sum()),
        "fit_rows": int(rows.sum()),
        "covered_cells": int(tail_model.cells["covered"].sum()),
        "valid_term_rows": int(flagged[mask].sum()),
        "prepare_s": prepare_s,
        "fit_s": fit_s,
        "predict_s": predict_s,
        "peak_rss_gib": peak_gib(),
    }
    del encoded, encoding, model
    return out, record


# --------------------------------------------------------------------------
# reading what was stored


def rmse(residual) -> float:
    r = np.asarray(residual, dtype=float)
    return float(np.sqrt(np.mean(r**2))) if len(r) else float("nan")


def printed(x: float) -> str:
    return f"{x:.1f}"


def reproduces(mine: pd.DataFrame, stored: Path) -> tuple[str, bool]:
    """Largest absolute difference against a stored residual file.

    ``preregistered.reproduces`` on two residual frames: joined on the movement
    identifier, which is read and never printed; a row count or a join that
    disagrees is itself a failure.
    """
    if not stored.exists():
        return (f"{stored.name} is not on disk", False)
    left = pd.read_parquet(stored, columns=["mvt_id", "residual"])
    right = mine[["mvt_id", "residual"]].rename(columns={"residual": "mine"})
    if len(left) != len(right):
        return (f"{len(right):,} rows against {len(left):,} stored", False)
    merged = left.merge(right, on="mvt_id", how="inner", validate="one_to_one")
    if len(merged) != len(left):
        return (f"{len(merged):,} of {len(left):,} identifiers joined", False)
    worst = float(
        np.max(np.abs(merged["residual"].to_numpy() - merged["mine"].to_numpy()))
    )
    return (f"{worst:.6g} s over {len(merged):,} rows", worst == 0.0)


def forward_figures(frame: pd.DataFrame) -> dict[str, float]:
    keep = ~frame["flagged"].to_numpy(dtype=bool)
    return {
        "ex_flagged": rmse(frame["residual"].to_numpy()[keep]),
        "with_term": rmse(frame[TERM_COLUMN]),
        "booster": rmse(frame["residual"]),
    }


def cell(frame: pd.DataFrame) -> dict[str, float]:
    """The figures one airport-month, or one month pooled, is read for."""
    keep = ~frame["flagged"].to_numpy(dtype=bool)
    residual = frame["residual"].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "term_rows": int((~keep).sum()),
        "ex_flagged": rmse(residual[keep]),
        "with_term": rmse(frame[TERM_COLUMN]),
        "booster": rmse(residual),
        # Prediction minus target over every row, as ``validation.evaluate``.
        "bias": float(-residual.mean()) if len(residual) else float("nan"),
    }


def stored_path(leg: Leg) -> Path:
    return WORK / f"{leg.stem}.parquet"


def read_stored(legs: list[Leg]) -> dict[str, pd.DataFrame]:
    missing = [stored_path(leg).name for leg in legs if not stored_path(leg).exists()]
    if missing:
        raise SystemExit(f"not on disk under {WORK}: {', '.join(missing)}; run --fit")
    return {leg.name: pd.read_parquet(stored_path(leg)) for leg in legs}


def movements_on_disk(stored: dict[str, pd.DataFrame]) -> set[str]:
    out: set[str] = set()
    for frame in stored.values():
        out |= {f"{int(v)}" for v in frame["mvt_id"].dropna().unique()}
    return out


# --------------------------------------------------------------------------
# the report


def table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    return rb.table(header, rows, align)


def fmt(x: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:,.{digits}f}"


def signed(x: float, digits: int = 0) -> str:
    if not np.isfinite(x):
        return "n/a"
    # A value that rounds to zero prints as +0, never -0.
    return f"{round(x, digits) + 0.0:+,.{digits}f}"


def month_label(month: str) -> str:
    return datetime.strptime(month, "%Y-%m").strftime("%b")


def build_report(stored: dict[str, pd.DataFrame], legs: list[Leg], run: dict) -> str:
    by_name = {leg.name: leg for leg in legs}
    held = [leg for leg in legs if leg.kind == "held-out"]
    airports = sorted(
        {a.icao for a in AIRPORTS}
        & set().union(*(set(stored[leg.name]["icao"].unique()) for leg in held))
    )
    rung = LADDER_BY_KEY[SUBMISSION_KEY]
    lines: list[str] = []
    add = lines.append

    add("# Every month, held out in turn")
    add("")
    add(
        "Generated by `scripts/report_monthly.py` with `--report` from the residuals "
        "`--fit` stored under `data/processed/monthly/`; every figure below is "
        "computed there, and the generator refuses to write a movement identifier "
        "or a literal stand identifier."
    )
    add("")

    # 1 ------------------------------------------------------------------
    add("## What this is")
    add("")
    add(
        "The paper quotes two validation months because the design of 2 September "
        "fixed two forward splits, December and July, as the proxies for the two "
        "ranking months. A forward fit cannot validate January, the training "
        "year's first month, and a forward fit of February or March would train on "
        "one or two months and mean little. So here each month of 2025 is held out "
        "in turn and the model is trained on the other eleven: the same rows every "
        "month, a design that is **not forward in time**. Every held-out figure "
        "below was fitted on months that come after the month it scores as well as "
        "before it."
    )
    add("")
    add(
        f"The model is the submitted rung, `{rung.label}` (`SUBMISSION_KEY`), one "
        f"fit per leg at the ladder's seed (offset {SEED} on "
        '`BOOSTER_PARAMS["random_state"]`), under `taxiout.models.BOOSTER_PARAMS`. '
        "Inside every leg, as in the ladder: the record-quality term of "
        "`taxiout.tails` is fitted on the training months, the rung is fitted on "
        "`taxiout.models.fit_rows` (training, unflagged, and the recorded-taxi "
        f"tail over {TAIL_EXCLUSION_ADOPTED_H:g} h, "
        "`TAIL_EXCLUSION_ADOPTED_H`, out of the fit on the fitting side only), and "
        "the shrunk encoding is recomputed from those rows. The submission itself "
        "is a seed average fitted on all twelve months; this report fits single "
        "seeds."
    )
    add("")
    add(
        "**This report decides nothing.** It changes no parameter, no feature and "
        "no rule, and nothing that ships is fitted from it. It is read for one "
        "thing: how the submitted rung's error is distributed over airports and "
        "months."
    )
    add("")
    add(
        "The constants the script holds: the twelve months (`MONTHS`), the seed "
        "offset (`SEED`), the six forward figures it checks against "
        "`reports/baseline.md` (`REF_FORWARD`), and the airport the term is "
        f"expected at (`TERM_AIRPORT`, {TERM_AIRPORT})."
    )
    add("")

    # 2 ------------------------------------------------------------------
    add("## The harness is the ladder's")
    add("")
    add(
        "The two forward legs are the ladder's December and July splits, fitted "
        "here by this script before any held-out leg. Their residuals are joined "
        "to the ladder's stored ones on the movement identifier, every row, and "
        "their pooled figures set beside `reports/baseline.md`'s for "
        "`+ plan differences`."
    )
    add("")
    rows = []
    all_ok = True
    for name, ladder_name, key in (
        ("forward-december", "december", "december"),
        ("forward-july", "july", "july"),
    ):
        got, ok = reproduces(
            stored[name], rb.RESIDUALS / f"{ladder_name}_{SUBMISSION_KEY}.parquet"
        )
        figures = forward_figures(stored[name])
        agree = all(
            printed(figures[k]) == printed(REF_FORWARD[key][k]) for k in REF_FORWARD[key]
        )
        all_ok &= ok and agree
        rows.append(
            [
                month_label(by_name[name].valid_month) + " (forward)",
                f"{len(stored[name]):,}",
                got,
                f"{printed(figures['ex_flagged'])} / {printed(REF_FORWARD[key]['ex_flagged'])}",
                f"{printed(figures['with_term'])} / {printed(REF_FORWARD[key]['with_term'])}",
                f"{printed(figures['booster'])} / {printed(REF_FORWARD[key]['booster'])}",
                "yes" if ok and agree else "**no**",
            ]
        )
    lines += table(
        [
            "leg",
            "rows",
            "largest absolute difference",
            "excluding flagged (s), here / baseline.md",
            "with the term, every row (s)",
            "booster only, every row (s)",
            "reproduces",
        ],
        rows,
        "lrlrrrl",
    )
    add("")
    add(
        "Both forward legs reproduce the ladder to the bit, so the held-out legs "
        "below are fitted by the ladder's own path."
        if all_ok
        else "**A forward leg did not reproduce the ladder; no figure below should be read.**"
    )
    add("")

    # 3 ------------------------------------------------------------------
    pooled = {leg.valid_month: cell(stored[leg.name]) for leg in held}
    add("## Every month, pooled")
    add("")
    add(
        f"One held-out fit per month, all {len(airports)} airports. *Excluding "
        "flagged* drops the "
        "rows the record-quality term covers; *with the term* scores every row with "
        "the term substituted on those rows, which is what the submission does; "
        "*booster only* scores every row with those rows left to the booster. Bias "
        "is prediction minus recorded taxi time over every row, booster only."
    )
    add("")
    rows = []
    for leg in held:
        c = pooled[leg.valid_month]
        rows.append(
            [
                leg.valid_month,
                f"{c['rows']:,}",
                f"{c['term_rows']:,}",
                fmt(c["ex_flagged"]),
                fmt(c["with_term"]),
                fmt(c["booster"]),
                signed(c["bias"]),
            ]
        )
    every = pd.concat([stored[leg.name] for leg in held], ignore_index=True)
    c = cell(every)
    rows.append(
        [
            "the year",
            f"{c['rows']:,}",
            f"{c['term_rows']:,}",
            fmt(c["ex_flagged"]),
            fmt(c["with_term"]),
            fmt(c["booster"]),
            signed(c["bias"]),
        ]
    )
    lines += table(
        [
            "held-out month",
            "rows",
            "term rows",
            "excluding flagged (s)",
            "with the term (s)",
            "booster only (s)",
            "bias (s)",
        ],
        rows,
        "lrrrrrr",
    )
    add("")
    ex = {m: pooled[m]["ex_flagged"] for m in pooled}
    best = min(ex, key=ex.get)
    worst = max(ex, key=ex.get)
    add(
        f"Excluding flagged rows, the held-out months run from {fmt(ex[best])} s "
        f"({month_label(best)}) to {fmt(ex[worst])} s ({month_label(worst)})."
    )
    add("")
    sentences = []
    for name in ("forward-december", "forward-july"):
        month = by_name[name].valid_month
        fwd_leg, held_leg = by_name[name], by_name["heldout-" + month]
        fwd = forward_figures(stored[name])["ex_flagged"]
        hold = pooled[month]["ex_flagged"]
        if fwd_leg.train_months == held_leg.train_months:
            same, _ = reproduces(stored[held_leg.name], stored_path(fwd_leg))
            sentences.append(
                f"{month_label(month)} held out trains on the same "
                f"{len(fwd_leg.train_months)} months as the forward leg, since nothing "
                f"in the year follows it, and reads {fmt(hold)} s against the forward "
                f"{fmt(fwd)} s (the two fits differ by {same})"
            )
            continue
        diff = hold - fwd
        direction = "lower" if diff < 0 else "higher" if diff > 0 else "equal"
        sentences.append(
            f"{month_label(month)} held out reads {fmt(hold)} s against the forward "
            f"{fmt(fwd)} s, {fmt(abs(diff))} s {direction}, trained on "
            f"{len(held_leg.train_months)} months against "
            f"{len(fwd_leg.train_months)}"
        )
    add(
        "Set beside the forward figures the paper quotes: "
        + "; ".join(sentences)
        + ". The other held-out fits train on months after the one they score, so "
        "they are not a substitute for the forward figures, which remain the ones "
        "quoted for the two test months."
    )
    add("")

    # 4 ------------------------------------------------------------------
    grid: dict[tuple[str, str], dict] = {}
    for leg in held:
        frame = stored[leg.name]
        for icao in airports:
            grid[(icao, leg.valid_month)] = cell(frame.loc[frame["icao"] == icao])
    by_airport = {icao: cell(every.loc[every["icao"] == icao]) for icao in airports}
    year = cell(every)
    add("## Every airport, every month")
    add("")
    add(
        "The same twelve held-out fits, split by departure airport. The margins are "
        "pooled over the rows they cover, not means of the cells: each airport's "
        f"year over its rows in the {len(held)} months, each month over its rows at "
        f"the {len(airports)} airports, and the corner over every row of the year. "
        "Beside each margin is the count of rows it reads: the rows the term leaves "
        "in the first table, every row in the second. A pooled figure weights each "
        "cell by its rows."
    )
    for key, title, render, count in (
        (
            "ex_flagged",
            "Excluding flagged rows (s)",
            fmt,
            lambda x: x["rows"] - x["term_rows"],
        ),
        ("bias", "Bias, every row, booster only (s)", signed, lambda x: x["rows"]),
    ):
        add("")
        add(f"### {title}")
        add("")
        rows = []
        for icao in airports:
            values = [grid[(icao, leg.valid_month)][key] for leg in held]
            rows.append(
                [icao]
                + [render(v) for v in values]
                + [render(by_airport[icao][key]), f"{count(by_airport[icao]):,}"]
            )
        rows.append(
            [f"all {len(airports)}, pooled"]
            + [render(pooled[leg.valid_month][key]) for leg in held]
            + [render(year[key]), f"{count(year):,}"]
        )
        rows.append(
            ["rows"]
            + [f"{count(pooled[leg.valid_month]):,}" for leg in held]
            + [f"{count(year):,}", ""]
        )
        lines += table(
            ["ICAO"]
            + [month_label(leg.valid_month) for leg in held]
            + ["year, pooled", "rows"],
            rows,
            "l" + "r" * (len(held) + 2),
        )
    add("")
    ex_cells = {k: v["ex_flagged"] for k, v in grid.items()}
    lo = min(ex_cells, key=ex_cells.get)
    hi = max(ex_cells, key=ex_cells.get)
    add(
        f"Excluding flagged rows, the best airport-month is {lo[0]} in "
        f"{month_label(lo[1])}, {fmt(ex_cells[lo])} s, and the worst {hi[0]} in "
        f"{month_label(hi[1])}, {fmt(ex_cells[hi])} s."
    )
    add("")

    # 5 ------------------------------------------------------------------
    add("## Where the term acts")
    add("")
    term_elsewhere = sum(
        grid[(icao, leg.valid_month)]["term_rows"]
        for icao in airports
        if icao != TERM_AIRPORT
        for leg in held
    )
    add(
        "Rows the record-quality term covers, per airport and held-out month. "
        f"{'Every one is at ' + TERM_AIRPORT + '.' if term_elsewhere == 0 else f'**{term_elsewhere:,} are outside {TERM_AIRPORT}.**'}"
    )
    add("")
    rows = []
    for icao in airports:
        values = [grid[(icao, leg.valid_month)]["term_rows"] for leg in held]
        if sum(values) or icao == TERM_AIRPORT:
            rows.append([icao] + [f"{v:,}" for v in values] + [f"{sum(values):,}"])
    lines += table(
        ["ICAO"] + [month_label(leg.valid_month) for leg in held] + ["year"],
        rows,
        "l" + "r" * (len(held) + 1),
    )
    add("")
    add(f"{TERM_AIRPORT} alone, every row, with the term and with the booster only:")
    add("")
    rows = []
    for leg in held:
        c = grid[(TERM_AIRPORT, leg.valid_month)]
        rows.append(
            [
                leg.valid_month,
                f"{c['rows']:,}",
                f"{c['term_rows']:,}",
                fmt(c["with_term"]),
                fmt(c["booster"]),
                signed(c["with_term"] - c["booster"], 1),
            ]
        )
    lines += table(
        [
            "held-out month",
            "rows",
            "term rows",
            "with the term (s)",
            "booster only (s)",
            "difference (s)",
        ],
        rows,
        "lrrrrr",
    )
    add("")

    # timings, from the fit run's own record
    if run.get("fits"):
        add("## What the fits cost")
        add("")
        add(
            f"The frame was built once in {run['frame_build_s'] / 60:.1f} min. Per leg, "
            "monotonic clock, so a suspend would not count:"
        )
        add("")
        rows = []
        for r in run["fits"]:
            rows.append(
                [
                    r["leg"],
                    f"{r['train_rows']:,}",
                    f"{r['fit_rows']:,}",
                    f"{r['valid_rows']:,}",
                    f"{r['prepare_s']:.0f}",
                    f"{r['fit_s']:.0f}",
                    f"{r['peak_rss_gib']:.1f}",
                ]
            )
        lines += table(
            [
                "leg",
                "train rows",
                "fit rows",
                "validation rows",
                "term + encoding (s)",
                "fit (s)",
                "peak RSS (GiB)",
            ],
            rows,
            "lrrrrrr",
        )
        add("")

    # 6 ------------------------------------------------------------------
    inputs = rb._inputs() + sorted(
        [stored_path(leg) for leg in legs]
        + [
            rb.RESIDUALS / f"december_{SUBMISSION_KEY}.parquet",
            rb.RESIDUALS / f"july_{SUBMISSION_KEY}.parquet",
        ]
    )
    lines += report_inputs_footer(REPO, inputs)
    add("")
    add(
        "Distance matrices derived from OpenStreetMap: data (c) "
        "OpenStreetMap contributors, ODbL 1.0. Movement and flight data"
    )
    add("supplied by EUROCONTROL PRC and the OpenSky Network under the terms")
    add("of the PRC Data Challenge 2026.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# the modes


def dry_run() -> int:
    before = fingerprint()
    dep = rb.load_departures(None)
    month = month_of(dep["MVT_TIME_UTC_mvt"])
    legs = build_legs(month)
    outside = int((~month.isin(MONTHS)).sum())
    print(f"{len(dep):,} departures; {outside:,} outside the twelve months")
    print("\nthe legs, and nothing fitted:")
    for leg in legs:
        print(
            f"  {leg.name:>18s}  {leg.kind:<8s}  train {leg.train_months[0]}..{leg.train_months[-1]}"
            f" ({len(leg.train_months):>2d} months, {int(leg.train.sum()):>9,} rows)"
            f"  validate {leg.valid_month} ({int(leg.valid.sum()):>7,} rows)"
        )
    held = [leg for leg in legs if leg.kind == "held-out"]
    total = sum(int(leg.valid.sum()) for leg in held)
    print(f"\nheld-out validation rows summed: {total:,} of {len(dep):,}")
    moved = touched(before, fingerprint())
    print("\n" + ("nothing was touched" if not moved else f"TOUCHED: {moved}"))
    return 0 if not moved and total + outside == len(dep) else 1


def fit() -> int:
    before = fingerprint()
    print(
        f"writing only to {WORK.relative_to(REPO)}; "
        + " and ".join(str(p.relative_to(REPO)) for p in FORBIDDEN)
        + f" are checked untouched ({len(before)} files)",
        flush=True,
    )
    t0 = time.monotonic()
    frame = Frame()
    WORK.mkdir(parents=True, exist_ok=True)
    run = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame_build_s": frame.build_seconds,
        "frame_rows": int(len(frame.features)),
        "fits": [],
        "stands": sorted(
            {str(v).strip() for v in frame.dep["STAND_mvt"].dropna().astype(str).unique()}
        ),
    }
    for leg in frame.legs:
        out, record = fit_leg(frame, leg)
        out.to_parquet(stored_path(leg), index=False)
        run["fits"].append(record)
        (WORK / RUN_RECORD).write_text(json.dumps(run, indent=1), encoding="utf-8")
        print(
            f"  {leg.name:>18s}: {record['fit_rows']:,} fit rows, "
            f"prepare {record['prepare_s']:.0f} s, fit {record['fit_s']:.0f} s, "
            f"ex-flagged {fmt(forward_figures(out)['ex_flagged'])} s, "
            f"peak {record['peak_rss_gib']:.1f} GiB",
            flush=True,
        )
        if leg.kind == "forward":
            ladder = leg.name.split("-", 1)[1]
            got, ok = reproduces(out, rb.RESIDUALS / f"{ladder}_{SUBMISSION_KEY}.parquet")
            figures = forward_figures(out)
            agree = {
                k: printed(figures[k]) == printed(REF_FORWARD[ladder][k])
                for k in REF_FORWARD[ladder]
            }
            print(
                f"    [{'ok ' if ok else 'STOP'}] against {ladder}_{SUBMISSION_KEY}.parquet: {got}",
                flush=True,
            )
            for k, v in agree.items():
                print(
                    f"    [{'ok ' if v else 'STOP'}] {k}: {printed(figures[k])} s, "
                    f"baseline.md {printed(REF_FORWARD[ladder][k])} s",
                    flush=True,
                )
            if not (ok and all(agree.values())):
                print("the harness is not the ladder's; stopping", file=sys.stderr)
                return 1
    run["total_s"] = time.monotonic() - t0
    run["peak_rss_gib"] = peak_gib()
    (WORK / RUN_RECORD).write_text(json.dumps(run, indent=1), encoding="utf-8")
    moved = touched(before, fingerprint())
    if moved:
        print("TOUCHED: " + ", ".join(moved), file=sys.stderr)
        return 1
    print(f"\n{len(run['fits'])} fits in {run['total_s'] / 60:.1f} min; nothing touched")
    print("DONE", flush=True)
    return 0


def report(out: Path) -> int:
    before = fingerprint()
    legs = leg_plan()
    stored = read_stored(legs)
    record = WORK / RUN_RECORD
    run = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    text = build_report(stored, legs, run)
    check_no_identifiers(text, set(run.get("stands", [])), movements_on_disk(stored))
    out.parent.mkdir(parents=True, exist_ok=True)
    moved = touched(before, fingerprint())
    if moved:
        print("TOUCHED: " + ", ".join(moved), file=sys.stderr)
        return 1
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="the legs; fit nothing")
    mode.add_argument("--fit", action="store_true", help="fit the fourteen legs")
    mode.add_argument(
        "--report", action="store_true", help="write the report; fit nothing"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the report's path")
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.fit:
        return fit()
    return report(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
