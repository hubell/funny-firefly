#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fit the ladder's rungs at a chosen thread count, and check them against it.

The instrument for one question: what the booster's thread count changes about
a fit. It fits any set of the ladder's rungs at a chosen ``n_jobs``, on either
forward split, and ``--compare`` subtracts each fit from the rung's stored
residuals in ``data/processed/residuals/`` and prints the largest difference.
``--report`` turns every run it has recorded into a generated table.

It was written as a fast lane -- eight rungs at one thread each, finishing in
the time three eight-thread fits take -- and its own first run refuted both
halves of that premise. Both are recorded here because they are the reason it
is kept:

* **The parallel mode is worth 1.08x, not the ~3x predicted.** December's 25
  rungs took 1,409 s of wall clock as eight one-thread fits at a time against
  1,528 s fitted one at a time on eight threads. The premise was that a fit
  given one thread would cost about what a fit given eight costs, so eight at
  once would be nearly eight times the work. It is not: on the two rungs
  measured both ways alone, one thread costs 4.0x and 4.1x what eight cost,
  so the booster already takes most of what the eight threads are worth.
  Running eight fits beside each other then costs another 1.37x and 1.23x on
  those same rungs. What is left over is the 1.08x. Why the booster scales
  about fourfold across eight threads rather than eightfold is not measured
  here and no reason is offered for it.
* **The fits at one thread are not the ladder's on every rung.** Of those 25,
  nine are bit-identical to the stored residuals, thirteen differ in the last
  bits of a sum taken in another order, and three grow different trees:
  ``weather-nolag`` by 158.3 s, ``throughput`` by 40.3 s and ``corridor-nolag``
  by 4.97 s on a flight. No printed RMSE moves at the one decimal the reports
  carry, so the difference is invisible in a report and real in the model.

So this is an instrument and not a fast lane. The parallel mode stays, because
fitting a whole ladder at one thread is still cheapest that way, but a run made
to answer a question about the thread count should use ``--workers 1`` so that
nothing contends with anything.

**Nothing here may produce a reported figure** except its own report. It writes
under ``data/processed/fastlane/`` and, in ``--report`` mode only, the one file
named by ``REPORT``; every other file in ``reports/`` and every stored residual
is fingerprinted before the run and verified untouched after it. A fast lane
that wrote a residual file would put an unreported fit where a reported one
belongs, and nothing downstream would say so.

Workers are forked, so the feature frame is inherited rather than pickled: one
copy, shared until written to. Each worker reports its own peak memory, because
the frame is about a gigabyte and the machine has no swap.

Usage
-----
    python scripts/fast_lane.py --split december --workers 1 --threads 8 --compare
    python scripts/fast_lane.py --rungs throughput weather-nolag --lgbm-verbose 2
    taskset -c 0-3 python scripts/fast_lane.py --threads 8 --workers 1 --compare
    python scripts/fast_lane.py --report
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_baselines as rb  # noqa: E402
from taxiout.encoding import ENCODING_FEATURES, encode_split, with_encoding  # noqa: E402
from taxiout.features import TARGET, airport_fallback, build_features  # noqa: E402
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER,
    LADDER_BY_KEY,
    Booster,
    fit_rows as fitting_rows,
)
from taxiout.validation import december_split, july_split, month_of, rmse  # noqa: E402

WORK = REPO / "data" / "processed" / "fastlane"
# The two directories this script may never write. Checked rather than
# promised: the whole reason it exists is that its numbers are not the
# report's, and a fast lane that wrote a residual file would put an unreported
# fit where a reported one belongs.
FORBIDDEN = (REPO / "reports", REPO / "data" / "processed" / "residuals")

# Set once in the parent, inherited by every forked worker. A module global
# because that is what fork shares for free; passing it as an argument would
# pickle a gigabyte per rung.
STATE: dict[str, object] = {}


def _fingerprint(exempt: Path | None = None) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file this script must not write.

    ``exempt`` is the one file ``--report`` is allowed to write, left out by
    name rather than by leaving its directory out, which would hand the fast
    lane the report's privileges back.
    """
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != exempt:
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


class _Log:
    """Collects LightGBM's native log lines, which ``verbose: -1`` suppresses.

    The same shape ``check_determinism.py`` registers: LightGBM calls ``info``
    and ``warning`` on whatever object it is given.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(str(msg))

    def warning(self, msg: str) -> None:
        self.lines.append(str(msg))


# The dataset-construction lines that decide whether a rung's design carries a
# multi-value bin group. LightGBM prints them at ``verbose`` 2 and above, on
# the way to choosing how to sum the histograms; ``Using Sparse Multi-Val Bin``
# and ``Using Dense Multi-Val Bin`` are the two it can print, and a design with
# no such group prints neither.
MULTI_VAL = re.compile(r"using (sparse|dense) multi-val bin", re.IGNORECASE)
BUILT = re.compile(r"GetMultiBinFrom(Sparse|All)Features")
BIN_LINES = re.compile(
    r"multi-val|sparse rate|num_groups|total bins|number of used features"
    r"|col-wise|row-wise",
    re.IGNORECASE,
)


def _multi_value_bin(lines: list[str]) -> tuple[str, list[str]]:
    """What the fit did about a multi-value bin group, and the lines that say so.

    Two different things are logged and they must not be run together.
    ``Dataset::GetMultiBinFrom{Sparse,All}Features`` says a multi-value bin was
    **built**; ``Using {Sparse,Dense} Multi-Val Bin`` says one was **used** to
    sum the histograms. Under ``force_col_wise`` the first happens and the
    second does not, so a verdict of "none" for a fit that built one would read
    as evidence it never constructed one, which is the opposite of the truth.

    ``"not logged"`` is a third answer again: the run kept no log, and nothing
    about the fit can be read from that.
    """
    kept = [
        part.strip()
        for line in lines
        for part in str(line).splitlines()
        if BIN_LINES.search(part)
    ]
    text = " ".join(str(line) for line in lines)
    built = m.group(1).lower() if (m := BUILT.search(text)) else None
    used = m.group(1).lower() if (m := MULTI_VAL.search(text)) else None
    if built and used:
        return (f"built from {built} features, used {used}", kept)
    if built:
        return (f"built from {built} features, not used", kept)
    if used:
        return (f"used {used}", kept)
    return ("none logged" if kept else "not logged", kept)


def _fit_one(key: str) -> dict:
    """One rung, in a worker: fit, predict, and report what it cost."""
    rung = LADDER_BY_KEY[key]
    frame = (
        STATE["encoded"]
        if set(rung.features) & set(ENCODING_FEATURES)
        else STATE["features"]
    )
    y, rows, valid = STATE["y"], STATE["rows"], STATE["valid"]
    params = dict(BOOSTER_PARAMS)
    params["n_jobs"] = int(STATE["threads"])
    # ``verbose`` is a logging setting and not a model one, which is the whole
    # point of raising it here: the digest below must still be the ladder's.
    # ``--compare`` is what checks that rather than this comment.
    log = _Log()
    if int(STATE["lgbm_verbose"]) > 0:
        import lightgbm

        params["verbose"] = int(STATE["lgbm_verbose"])
        lightgbm.register_logger(log)
    t0 = time.perf_counter()
    model = Booster(rung.features, params).fit(frame.loc[rows], y[rows])
    seconds = time.perf_counter() - t0
    pred = model.predict(frame.loc[valid])
    bin_kind, bin_lines = _multi_value_bin(log.lines)
    return {
        "rung": key,
        "seconds": seconds,
        "multi_value_bin": bin_kind,
        "construction_log": bin_lines,
        "rmse": rmse(y[valid], pred),
        "sha256": hashlib.sha256(
            np.asarray(pred, dtype=np.float64).tobytes()
        ).hexdigest(),
        "residual": np.asarray(y[valid], dtype=np.float64)
        - np.asarray(pred, dtype=np.float64),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "pid": os.getpid(),
    }


def _against_the_ladder(split_name: str, key: str, residual: np.ndarray) -> float:
    """Largest difference from the rung's stored residuals, or NaN without them."""
    path = rb.RESIDUALS / f"{split_name}_{key}.parquet"
    if not path.exists():
        return float("nan")
    stored = pd.read_parquet(path, columns=["residual"])["residual"].to_numpy(dtype=float)
    if len(stored) != len(residual):
        return float("inf")
    finite = np.isfinite(stored) & np.isfinite(residual)
    return float(np.max(np.abs(stored[finite] - residual[finite]), initial=0.0))


# ---------------------------------------------------------------------------
# --report: every recorded run, in one generated table

# The file this script is allowed to write, and the only one: named here so the
# forbidden-directory check below can exempt it by name rather than by
# exempting the directory, which would give the fast lane the report's
# privileges back.
REPORT = REPO / "reports" / "threads-2026-09-12.md"

# Above this a difference is a different set of trees; at or below it, the last
# bits of a sum that was added in another order. The threshold and both names
# are ``reports/determinism.md``'s, so that the two reports read alike.
TREES_DIFFER = 1e-6

# The condition every reported figure in the repository was fitted under: the
# booster given the thread count ``BOOSTER_PARAMS`` pins, one fit at a time, on
# the whole machine. Read from the parameters rather than typed, so that
# changing them cannot leave this report calling the old count "the reference".
REFERENCE = (int(BOOSTER_PARAMS["n_jobs"]), 1, None)

ATTRIBUTION = "Distance matrices derived from OpenStreetMap"


def _records() -> list[dict]:
    """Every run this script has recorded, oldest first.

    A record whose run is still going has claimed its name and holds nothing
    yet; it is skipped rather than crashing the report.
    """
    out = []
    for path in sorted(WORK.glob("fastlane_*.json")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        record = json.loads(text)
        record["file"] = path.name
        out.append(record)
    return out


def _condition(record: dict) -> tuple[int, int, int | None]:
    """The condition a run's fits were made under.

    Three things tell one run from another and all three can move a number in
    the tables: the booster's thread count, how many fits ran beside each
    other, and how many CPUs the run was allowed. The last is ``None`` where
    the run had the machine and a count where ``taskset`` narrowed it, which is
    the whole point of those runs and why they are not pooled with the
    unrestricted eight-thread ones.
    """
    cpus = record.get("cpus_available")
    total = record.get("cpu_count")
    narrowed = len(cpus) if cpus and total and len(cpus) < total else None
    return (int(record["threads"]), int(record["workers"]), narrowed)


def _label(condition: tuple[int, int, int | None]) -> str:
    threads, workers, cpus = condition
    text = f"{threads} thread" + ("s" if threads != 1 else "")
    if workers > 1:
        text += f", {workers} at once"
    if cpus is not None:
        text += f", {cpus} CPUs"
    return text


def _fmt_delta(x: float | None) -> str:
    """A difference in seconds, in the vocabulary ``determinism.md`` uses."""
    if x is None:
        return "--"
    if np.isinf(x):
        # Not a large difference: the two are not the same shape and were never
        # comparable. It must not print as "--", which is what a rung nobody
        # fitted prints.
        return "**not comparable**"
    if not np.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    if x <= TREES_DIFFER:
        return f"{x:.1e}"
    return f"**{x:.4g}**"


def _order(rungs) -> list[str]:
    """Ladder order where the ladder knows the rung, then anything else."""
    ladder = [r.key for r in LADDER]
    return [k for k in ladder if k in rungs] + sorted(set(rungs) - set(ladder))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
        "",
    ]


def _by_rung(records: list[dict], split: str) -> dict[str, dict[tuple, list[dict]]]:
    """``{rung: {condition: [fit, ...]}}`` for one split."""
    out: dict[str, dict[tuple, list[dict]]] = {}
    for record in records:
        if record["split"] != split:
            continue
        condition = _condition(record)
        for fit in record["rungs"]:
            fit = dict(fit, lgbm_verbose=int(record.get("lgbm_verbose") or 0))
            out.setdefault(fit["rung"], {}).setdefault(condition, []).append(fit)
    return out


def _bins(records: list[dict]) -> dict[tuple[str, str], dict]:
    """What a logged run saw while it built each rung's dataset.

    Only runs made with ``--lgbm-verbose`` carry this: the reference
    configuration's ``verbose: -1`` prints none of it.
    """
    out: dict[tuple[str, str], dict] = {}
    for record in records:
        if not record.get("lgbm_verbose"):
            continue
        for fit in record["rungs"]:
            lines = fit.get("construction_log") or []
            groups = rate = bins = features = None
            # First match wins throughout, which is what ``_multi_value_bin``
            # reads too: LightGBM can log a second construction line for the
            # dense path, and letting the last one win would report a rate that
            # belongs to a bin the fit did not build from.
            for line in lines:
                if groups is None and (m := re.search(r"num_groups:\s*(\d+)", line)):
                    groups = int(m.group(1))
                if rate is None and (
                    m := re.search(r"sparse rate\s+([0-9.eE+-]+)", line)
                ):
                    rate = float(m.group(1))
                if bins is None and (m := re.search(r"Total Bins\s+(\d+)", line)):
                    bins = int(m.group(1))
                if features is None and (
                    m := re.search(r"number of used features:\s*(\d+)", line)
                ):
                    features = int(m.group(1))
            kind, _ = _multi_value_bin(lines)
            out[(record["split"], fit["rung"])] = {
                "kind": kind,
                "built": kind.startswith("built") or kind.startswith("used"),
                "groups": groups,
                "sparse_rate": rate,
                "total_bins": bins,
                "used_features": features,
            }
    return out


def _worst(fits: list[dict]) -> float | None:
    """The largest difference from the ladder over these fits.

    ``None`` means no fit here was compared at all, which is not a result.
    Infinity is a result -- ``_against_the_ladder`` returns it when the stored
    residuals and the fit are not even the same length -- so it is kept rather
    than filtered out with the missing values, which would print it as though
    the rung had never been fitted.
    """
    seen = [f["vs_the_ladder"] for f in fits if f.get("vs_the_ladder") is not None]
    seen = [v for v in seen if not (isinstance(v, float) and np.isnan(v))]
    return max(seen) if seen else None


def _moved(by_rung, condition) -> list[tuple[str, float]]:
    """Rungs whose fits at this condition grew different trees, worst first."""
    out = []
    for rung, per in by_rung.items():
        worst = _worst(per.get(condition, []))
        if worst is not None and worst > TREES_DIFFER:
            out.append((rung, worst))
    return sorted(out, key=lambda kv: -kv[1])


def _split_sections(split: str, by_rung, bins) -> list[str]:
    """The three tables for one split: difference, RMSE, fit time."""
    conditions = sorted(
        {c for per in by_rung.values() for c in per},
        key=lambda c: (c[0], c[1], c[2] if c[2] is not None else 99),
    )
    rungs = _order(by_rung)
    labels = [_label(c) for c in conditions]
    name = split.capitalize()

    difference = []
    for rung in rungs:
        row = [f"`{rung}`", bins.get((split, rung), {}).get("kind", "--")]
        row += [_fmt_delta(_worst(by_rung[rung].get(c, []))) for c in conditions]
        difference.append(row)

    def joined(rung, condition, fmt) -> str:
        """Every distinct value at this condition, in numeric order."""
        fits = by_rung[rung].get(condition, [])
        if not fits:
            return "--"
        seen = {fmt(f) for f in fits}
        return " / ".join(sorted(seen, key=lambda s: (float(s), s)))

    rmse = [
        [f"`{rung}`"]
        + [joined(rung, c, lambda f: f"{f['rmse']:.1f}") for c in conditions]
        for rung in rungs
    ]
    seconds = [
        [f"`{rung}`"]
        + [joined(rung, c, lambda f: f"{f['seconds']:.0f}") for c in conditions]
        for rung in rungs
    ]
    return [
        f"## {name}: largest difference from the ladder (s)",
        "",
        *_table(["rung", "multi-value bin", *labels], difference),
        f"## {name}: RMSE (s), at the one decimal the reports print",
        "",
        *_table(["rung", *labels], rmse),
        f"## {name}: fit time (s)",
        "",
        *_table(["rung", *labels], seconds),
    ]


def _findings(records, by_split, bins) -> list[str]:
    """The load-bearing sentences, every one computed from the tables above."""
    out = ["## What these runs establish", ""]

    for split, by_rung in by_split.items():
        ref = by_rung and {
            rung: _worst(per.get(REFERENCE, [])) for rung, per in by_rung.items()
        }
        at_reference = {r: v for r, v in ref.items() if v is not None}
        exact = sorted(r for r, v in at_reference.items() if v == 0.0)
        if at_reference:
            out.append(
                f"- **{split.capitalize()} at the reference condition "
                f"({_label(REFERENCE)}).** {len(exact)} of {len(at_reference)} rungs "
                "reproduce the ladder's stored residuals exactly"
                + (
                    "."
                    if len(exact) == len(at_reference)
                    else ", and "
                    + ", ".join(
                        f"`{r}` by {v:.3g} s"
                        for r, v in sorted(at_reference.items(), key=lambda kv: -kv[1])
                        if v != 0.0
                    )
                    + "."
                )
            )

        for condition in sorted(
            {c for per in by_rung.values() for c in per if c != REFERENCE},
            key=lambda c: (c[0], c[1], c[2] if c[2] is not None else 99),
        ):
            # Fitted AND compared: a run made without --compare carries no
            # difference, and counting those rungs in the denominator would
            # make the three tallies below fail to add up to it.
            fitted = [
                r
                for r, per in by_rung.items()
                if _worst(per.get(condition, [])) is not None
            ]
            moved = _moved(by_rung, condition)
            last_bits = [
                r
                for r in fitted
                if (w := _worst(by_rung[r].get(condition, []))) is not None
                and 0.0 < w <= TREES_DIFFER
            ]
            same = [r for r in fitted if _worst(by_rung[r].get(condition, [])) == 0.0]
            if not fitted:
                # Fitted, but never compared: the run was made without
                # ``--compare``, so there is no difference to report. Said
                # rather than skipped, so that a condition cannot disappear
                # from the findings without the reader being told why.
                ran = [r for r, per in by_rung.items() if per.get(condition)]
                if ran:
                    out.append(
                        f"- **{split.capitalize()} at {_label(condition)}** "
                        f"({len(ran)} rung" + ("s" if len(ran) != 1 else "") + "). "
                        "Fitted but never compared against the stored residuals, "
                        "so nothing here says whether these are the ladder's fits."
                    )
                continue
            sentence = (
                f"- **{split.capitalize()} at {_label(condition)}"
                f"** ({len(fitted)} rung" + ("s" if len(fitted) != 1 else "") + "). "
            )
            if moved:
                sentence += (
                    f"{len(moved)} grew different trees: "
                    + ", ".join(f"`{r}` by {v:.4g} s" for r, v in moved)
                    + f"; {len(last_bits)} differ in the last bits only and "
                    f"{len(same)} are bit-identical."
                )
            else:
                sentence += (
                    f"None grew different trees: {len(same)} bit-identical, "
                    f"{len(last_bits)} differing in the last bits only."
                )
            out.append(sentence)

        # Does any printed RMSE move between conditions?
        movers = []
        for rung, per in by_rung.items():
            printed = {f"{f['rmse']:.1f}" for fits in per.values() for f in fits}
            if len(printed) > 1:
                movers.append((rung, sorted(printed)))
        out.append(
            f"- **{split.capitalize()}: what a reader would see.** "
            + (
                "No rung's RMSE moves at the one decimal the reports print, over "
                f"every condition run ({len(by_rung)} rungs)."
                if not movers
                else "The printed RMSE moves on "
                + ", ".join(f"`{r}` ({' / '.join(v)})" for r, v in movers)
                + "."
            )
        )
        out.append("")
    return out


def _digests(by_split) -> list[str]:
    """Whether a condition ever gave two different fits of the same rung.

    Runs made with ``--lgbm-verbose`` pool into the same condition as the
    quiet ones, so this is also what checks that raising LightGBM's verbosity
    left the model alone: if it had not, the logged run's digest would differ
    from the quiet run's and the rung would be named here.
    """
    out = ["## One condition, one fit", ""]
    disagreed = []
    pooled = 0
    mixed = 0
    for split, by_rung in by_split.items():
        for rung, per in by_rung.items():
            for condition, fits in per.items():
                if len(fits) < 2:
                    continue
                pooled += 1
                # Both a logged fit and a quiet one under the same condition:
                # the only arrangement that says anything about what raising
                # LightGBM's verbosity does to the model.
                loud = {f.get("lgbm_verbose", 0) > 0 for f in fits}
                if loud == {True, False}:
                    mixed += 1
                if len({f["sha256"] for f in fits}) > 1:
                    disagreed.append((split, rung, _label(condition), len(fits)))
    if not pooled:
        out.append(
            "No rung was fitted twice under one condition, so there is "
            "nothing here to check."
        )
    elif not disagreed:
        out.append(
            f"{pooled} rung-and-condition pairs were fitted more than once, and "
            "each gave one digest every time."
            + (
                f" {mixed} of those pairs hold both a fit made with "
                "`--lgbm-verbose` and a quiet one, which is what says that "
                "raising LightGBM's verbosity to log the dataset construction "
                "left the model alone: a `verbose` that changed the fit would "
                "have shown up as a disagreement here."
                if mixed
                else " None of them holds both a logged fit and a quiet one, so "
                "nothing here says what raising LightGBM's verbosity does to a "
                "fit."
            )
        )
    else:
        out.append(
            "**One condition gave more than one fit**, which the reference "
            "configuration is supposed to make impossible: "
            + ", ".join(f"`{r}` on {s} at {c} ({n} fits)" for s, r, c, n in disagreed)
            + "."
        )
    return out + [""]


def _cpus(by_split) -> list[str]:
    """Is the row block cut by the thread count, or by the CPUs on offer?"""
    out = ["## The parameter, or the cores", ""]
    rows, verdicts = [], []
    for split, by_rung in by_split.items():
        narrowed = sorted(
            {c for per in by_rung.values() for c in per if c[2] is not None},
            key=lambda c: -c[2],
        )
        for condition in narrowed:
            for rung in _order(by_rung):
                here = by_rung[rung].get(condition, [])
                there = by_rung[rung].get(REFERENCE, [])
                if not here or not there:
                    continue
                mine = {f["sha256"] for f in here}
                ladder = {f["sha256"] for f in there}
                agree = mine == ladder and len(mine) == 1
                verdicts.append(agree)
                rows.append(
                    [
                        split,
                        f"`{rung}`",
                        _label(condition),
                        sorted(mine)[0][:10],
                        sorted(ladder)[0][:10],
                        "identical" if agree else "**differs**",
                        _fmt_delta(_worst(here)),
                    ]
                )
    if not rows:
        return out + [
            "No run has been made on a narrowed set of CPUs, so whether the "
            "row blocks follow `n_jobs` or the machine's core count is "
            "unmeasured.",
            "",
        ]
    out += _table(
        [
            "split",
            "rung",
            "condition",
            "digest",
            "the ladder's",
            "result",
            "vs the ladder",
        ],
        rows,
    )
    if all(verdicts):
        out.append(
            f"Every one of the {len(rows)} fits made on a narrowed set of CPUs "
            "gives the ladder's own digest. The row blocks the histograms are "
            "summed over are therefore cut by `n_jobs` and not by the number of "
            "CPUs the process can see, so the reproduction claim is a claim "
            "about a fixed thread count rather than about an eight-core machine."
        )
    else:
        out.append(
            f"{sum(1 for v in verdicts if not v)} of {len(rows)} fits made on a "
            "narrowed set of CPUs differ from the ladder's, so the number of "
            "CPUs on offer is part of the result as well as `n_jobs`, and the "
            "reproduction claim stays at eight cores."
        )
    return out + [""]


def _concurrency(by_split) -> list[str]:
    """Whether fitting several rungs at once changes any of them.

    A whole ladder at one thread is cheapest fitted several rungs at a time,
    and most of the one-thread evidence was gathered that way. That is only
    admissible if running beside other fits changes nothing, which is a
    control rather than an assumption: any rung fitted at the same thread
    count both alone and alongside others is compared here.
    """
    out = ["## Concurrency is not the cause", ""]
    rows, agree = [], []
    for split, by_rung in by_split.items():
        for rung in _order(by_rung):
            per = by_rung[rung]
            for alone in [c for c in per if c[1] == 1]:
                for beside in [
                    c for c in per if c[1] > 1 and c[0] == alone[0] and c[2] == alone[2]
                ]:
                    a = {f["sha256"] for f in per[alone]}
                    b = {f["sha256"] for f in per[beside]}
                    same = a == b and len(a) == 1
                    agree.append(same)
                    rows.append(
                        [
                            split,
                            f"`{rung}`",
                            _label(alone),
                            _label(beside),
                            sorted(a)[0][:10],
                            sorted(b)[0][:10],
                            "identical" if same else "**differs**",
                        ]
                    )
    if not rows:
        return out + [
            "No rung has been fitted at the same thread count both alone and "
            "beside others, so nothing here separates the thread count from the "
            "concurrency.",
            "",
        ]
    out += _table(
        ["split", "rung", "alone", "beside others", "digest", "digest", "result"],
        rows,
    )
    out.append(
        (
            f"All {len(rows)} of these pairs agree, so a rung fitted beside others "
            "is the same fit as one fitted alone and the thread count is what the "
            "tables above are measuring."
            if all(agree)
            else f"{sum(1 for a in agree if not a)} of {len(rows)} pairs disagree, so "
            "fitting several rungs at once is itself part of the result and the "
            "one-thread figures gathered that way cannot be read as thread-count "
            "effects alone."
        )
    )
    return out + [""]


def _mechanism(by_split, bins) -> list[str]:
    """Do the rungs that move carry a multi-value bin group, and the others not?"""
    out = ["## The multi-value bin", ""]
    rows = [
        [
            split,
            f"`{rung}`",
            seen["kind"],
            "--" if seen["sparse_rate"] is None else f"{seen['sparse_rate']:.6g}",
            "--" if seen["total_bins"] is None else f"{seen['total_bins']:,}",
            "--" if seen["used_features"] is None else str(seen["used_features"]),
            "--" if seen["groups"] is None else str(seen["groups"]),
        ]
        for (split, rung), seen in sorted(bins.items())
    ]
    if not rows:
        return out + [
            "No run has been made with `--lgbm-verbose`, so nothing here says what "
            "any rung's dataset does about a multi-value bin group.",
            "",
        ]
    out += _table(
        [
            "split",
            "rung",
            "multi-value bin",
            "sparse rate",
            "total bins",
            "used features",
            "feature groups",
        ],
        rows,
    )
    if all(seen["groups"] is None for seen in bins.values()):
        out += [
            "The feature-group count is empty because this LightGBM does not print "
            "it on this path at the verbosity these runs used. The column is kept "
            "rather than dropped so that its absence is on the record: it was asked "
            "for, and it is not there to be had from the log.",
            "",
        ]

    # The comment in ``taxiout.models`` proposes a mechanism: a dataset with a
    # multi-value bin group sums its histograms over row blocks cut by the
    # thread count, so the rungs that move at another thread count should be
    # exactly the rungs that carry one. That is testable here.
    carries, moves, both = set(), set(), set()
    for split, by_rung in by_split.items():
        for rung in by_rung:
            seen = bins.get((split, rung))
            if seen is None:
                continue
            both.add((split, rung))
            if seen["built"]:
                carries.add((split, rung))
            if any(
                (w := _worst(fits)) is not None and w > TREES_DIFFER
                for condition, fits in by_rung[rung].items()
                if condition != REFERENCE and condition[2] is None
            ):
                moves.add((split, rung))
    if not both:
        return out + [""]

    def names(pairs):
        return ", ".join(f"`{r}`" for _, r in sorted(pairs))

    if carries == moves:
        out.append(
            f"Over the {len(both)} rungs fitted with the log kept, the rungs whose "
            "dataset carries a multi-value bin group are exactly the rungs that "
            "grow different trees at a thread count other than the reference's"
            + (f" ({names(carries)})." if carries else ", and neither set has a member.")
            + " The mechanism the comment in `taxiout.models` proposes is that far "
            "measured."
        )
    else:
        out.append(
            f"**The mechanism is not what separates these rungs.** Of the "
            f"{len(both)} rungs fitted with the log kept, {len(carries)} build a "
            f"multi-value bin group and {len(moves)} grow different trees at "
            "another thread count, and the two sets are not the same"
            + (
                f". Building one without moving: {names(carries - moves)}"
                if carries - moves
                else ""
            )
            + (
                f". Moving without building one: {names(moves - carries)}"
                if moves - carries
                else ""
            )
            + ". "
            + _rates(bins, moves, both)
            + _built_but_unused(bins, both)
            + " So which rungs move at another thread count is **unexplained**: "
            "the measurement in the tables above stands on its own, and no "
            "mechanism is claimed for it."
        )
    return out + [""]


def _rates(bins, moves, both) -> str:
    """Whether the sparse rate tells the rungs that move from the rungs that do not."""
    moving = {bins[k]["sparse_rate"] for k in moves if bins.get(k, {}).get("sparse_rate")}
    still = {
        bins[k]["sparse_rate"]
        for k in (both - moves)
        if bins.get(k, {}).get("sparse_rate")
    }
    if not moving or not still:
        return "Too few rates were logged to say whether the sparse rate separates them."
    shared = moving & still
    if shared:
        return (
            "The sparse rate does not separate them either: "
            + ", ".join(f"{r:.6g}" for r in sorted(shared))
            + (" is" if len(shared) == 1 else " are each")
            + " reported by a rung that moves and by one that does not."
        )
    return (
        "The sparse rates of the two sets do not overlap ("
        + ", ".join(f"{r:.6g}" for r in sorted(moving))
        + " against "
        + ", ".join(f"{r:.6g}" for r in sorted(still))
        + "), which is worth another look but is not yet a mechanism."
    )


def _built_but_unused(bins, both) -> str:
    """The state every logged fit was actually in, counted rather than asserted."""
    kinds = [bins[k]["kind"] for k in both if k in bins]
    built = sum(1 for k in kinds if k.startswith("built"))
    unused = sum(1 for k in kinds if "not used" in k)
    if built == len(kinds) and unused == len(kinds):
        return (
            f" All {len(kinds)} of them built such a group and then used none of "
            "it, which is what the *not used* in the table means and what "
            "`force_col_wise` asks for."
        )
    return (
        f" {built} of {len(kinds)} built such a group and {unused} then used none of it."
    )


def _environment() -> dict:
    """The versions a run was made with, recorded so the report need not guess."""
    import platform

    import lightgbm

    return {
        # The architecture and not the CPU model: what a reader needs in order
        # to know whether these numbers could be theirs is the instruction set
        # and the wheels, and the model would only name the machine.
        "architecture": platform.machine(),
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
    }


def _provenance(records: list[dict]) -> list[str]:
    stamps = sorted(
        m.group(1)
        for r in records
        if (m := re.search(r"_(\d{8}T\d{6})\.json$", r["file"]))
    )
    days = sorted({f"{s[:4]}-{s[4:6]}-{s[6:8]}" for s in stamps})
    span = (
        "an unrecorded date"
        if not days
        else days[0]
        if len(days) == 1
        else f"{days[0]} to {days[-1]}"
    )
    fits = sum(len(r["rungs"]) for r in records)
    known = [r["environment"] for r in records if r.get("environment")]
    versions = sorted({e["lightgbm"] for e in known})
    said = (
        f" LightGBM {' and '.join(versions)}, from the {len(known)} of them that "
        "record it"
        if versions
        else " No run records the versions it was made with"
    )
    return [
        f"Generated by `scripts/fast_lane.py --report` from the {len(records)} runs "
        f"recorded under `data/processed/fastlane/`, which are gitignored: {fits} "
        f"fits in all, made on {span}, over "
        + " and ".join(f"the {s} split" for s in sorted({r["split"] for r in records}))
        + f".{said}. Every figure below is computed from those records.",
        "",
    ]


def report() -> int:
    records = _records()
    if not records:
        print(f"no runs recorded under {WORK.relative_to(REPO)}", file=sys.stderr)
        return 1
    bins = _bins(records)
    by_split = {
        split: _by_rung(records, split) for split in sorted({r["split"] for r in records})
    }

    text = [
        "# The booster's thread count, and which rungs it changes",
        "",
        *_provenance(records),
        "The ladder's stored residuals are the reference throughout. Each fit is "
        "subtracted from `data/processed/residuals/<split>_<rung>.parquet` and the "
        "largest absolute difference over the validation rows is what the tables "
        f"give, in seconds of taxi-out time. At or below {TREES_DIFFER:g} s that "
        "is the last bits of a sum taken in another order; above it the fit grew a "
        "different set of trees. The reference condition is `BOOSTER_PARAMS` with "
        "`n_jobs` 8, one fit at a time on the whole machine, which is what every "
        "reported figure in this repository was fitted under.",
        "",
        "Nothing here is a reported figure. `scripts/fast_lane.py` writes only "
        "under `data/processed/fastlane/` and this one file, and checks at the end "
        "of every run that it wrote nowhere else.",
        "",
    ]
    for split, by_rung in by_split.items():
        text += _split_sections(split, by_rung, bins)
    text += _findings(records, by_split, bins)
    text += _concurrency(by_split)
    text += _mechanism(by_split, bins)
    text += _cpus(by_split)
    text += _digests(by_split)
    text += [
        "## What this does not establish",
        "",
        "- Any LightGBM other than the one these runs were made with. The "
        "library's own documentation says `deterministic` \"should ensure the "
        "stable results when using the same data and the same parameters (and "
        'different `num_threads`)"; the tables above are what this version did.',
        "- Any rung not fitted here, and any split other than the ones named.",
        "- Any machine other than the reference one. Where a run was narrowed "
        "with `taskset` it was narrowed on that same machine, which is a "
        "different thing from another host.",
        "- A GPU. Nothing in this repository has been fitted on one.",
        "",
        "Distance matrices derived from OpenStreetMap: data (c)",
        "OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied by",
        "EUROCONTROL PRC and the OpenSky Network under the terms of the PRC Data",
        "Challenge 2026.",
        "",
    ]
    body = "\n".join(text)
    if ATTRIBUTION not in body:
        raise RuntimeError("the report lost its OpenStreetMap attribution")
    REPORT.write_text(body, encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO)} from {len(records)} runs")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("december", "july"), default="december")
    ap.add_argument(
        "--rungs",
        nargs="*",
        default=None,
        help=(
            "rung keys (default: every non-diagnostic booster rung the ladder "
            "carries, which includes the no-lag chain and so is wider than the "
            "ladder proper)"
        ),
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--threads", type=int, default=1, help="booster threads per worker")
    ap.add_argument(
        "--lgbm-verbose",
        type=int,
        default=0,
        metavar="N",
        help=(
            "LightGBM's own verbosity (2 logs dataset construction, which is "
            "where a multi-value bin group is named); 0 leaves it at the "
            "reference configuration's -1"
        ),
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="read each rung's stored residuals back and print the largest difference",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help=(
            "fit nothing; write the generated table of every run recorded so far "
            f"to {REPORT.relative_to(REPO)}"
        ),
    )
    args = ap.parse_args()

    if args.report:
        # The one file this script may write outside its own directory, and the
        # only mode in which it may: everything else in reports/ and every
        # stored residual is still checked untouched around the write.
        before = _fingerprint(exempt=REPORT)
        status = report()
        after = _fingerprint(exempt=REPORT)
        if after != before:
            print("--report wrote where it must not", file=sys.stderr)
            return 1
        return status

    # The promise that this writes no report and no stored residual, checked
    # rather than stated: every file in both directories is fingerprinted here
    # and verified at the end. A fast lane that wrote one would put an
    # unreported fit where a reported one belongs, and nothing downstream
    # would say so.
    before = _fingerprint()
    print(
        f"writing only to {WORK.relative_to(REPO)}; "
        + " and ".join(str(p.relative_to(REPO)) for p in FORBIDDEN)
        + f" are checked untouched ({len(before)} files)"
    )

    keys = args.rungs or [
        r.key for r in LADDER if r.kind == "booster" and not r.diagnostic
    ]
    unknown = [k for k in keys if k not in LADDER_BY_KEY]
    if unknown:
        print(f"not rungs: {unknown}", file=sys.stderr)
        return 1

    t0 = time.time()
    tails = rb.Tails()
    matrix = rb.load_matrix()
    dep = rb.load_departures(None)
    y = pd.to_numeric(dep[TARGET], errors="coerce")
    movements = rb.load_all_movements()
    features = build_features(
        dep,
        matrix,
        fallback=airport_fallback(matrix),
        movements=movements,
        edges=rb.load_edge_sets(),
        weather=rb.load_weather(),
        airports=rb.load_airport_table(),
    )
    del movements
    build_seconds = time.time() - t0
    print(f"frame: {len(features):,} rows in {build_seconds:.0f} s")

    month = month_of(dep["MVT_TIME_UTC_mvt"])
    split = december_split(month) if args.split == "december" else july_split(month)
    tail_model = tails.fit(dep.loc[split.train])
    flagged = (
        tail_model.predict(dep).notna()
        if tail_model is not None
        else pd.Series(False, index=dep.index)
    )
    unflagged = ~flagged
    rows = fitting_rows(split.train, ~unflagged, y)
    encoding = encode_split(features, y, rows)

    STATE.update(
        features=features,
        encoded=with_encoding(features, encoding),
        y=y,
        rows=rows,
        valid=split.valid,
        threads=args.threads,
        lgbm_verbose=args.lgbm_verbose,
    )

    print(
        f"{len(keys)} rungs on {args.split}, {args.workers} workers x "
        f"{args.threads} booster threads"
    )
    t0 = time.time()
    results = []
    with cf.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("fork")
    ) as pool:
        for got in pool.map(_fit_one, keys):
            residual = got.pop("residual")
            if args.compare:
                got["vs_the_ladder"] = _against_the_ladder(
                    args.split, got["rung"], residual
                )
            results.append(got)
            line = (
                f"  {got['rung']:<28s} RMSE {got['rmse']:7.1f} s "
                f"({got['seconds']:5.1f} s, peak {got['peak_rss_mib'] / 1024:.1f} GiB)"
            )
            if args.compare:
                line += f"  vs the ladder: {got['vs_the_ladder']:.3g}"
            print(line)
    wall = time.time() - t0

    fits = len(results)
    serial = sum(r["seconds"] for r in results)
    print(
        f"{fits} fits in {wall:.0f} s wall, {serial:.0f} s of fitting: "
        f"{serial / wall:.2f} fits in the time one takes"
    )
    if args.compare:
        worst = max(
            (r for r in results if np.isfinite(r.get("vs_the_ladder", float("nan")))),
            key=lambda r: r["vs_the_ladder"],
            default=None,
        )
        if worst is None:
            print("no stored residuals to compare against")
        elif worst["vs_the_ladder"] == 0.0:
            print(
                f"every rung matches the ladder's stored residuals exactly ({fits} compared)"
            )
        else:
            print(
                f"largest difference from the ladder: {worst['vs_the_ladder']:.3g} s "
                f"on {worst['rung']}"
            )

    WORK.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    record = {
        "split": args.split,
        "workers": args.workers,
        "threads": args.threads,
        "lgbm_verbose": args.lgbm_verbose,
        "environment": _environment(),
        # What the run was actually allowed to use, not what the machine has:
        # under ``taskset`` these differ, and that difference is the
        # measurement.
        "cpus_available": sorted(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "frame_seconds": build_seconds,
        "wall_seconds": wall,
        "fitting_seconds": serial,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0,
        "rungs": results,
    }
    out = WORK / f"fastlane_{args.split}_w{args.workers}t{args.threads}_{stamp}.json"
    out.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")

    after = _fingerprint()
    touched = sorted(
        {k for k in before if after.get(k) != before[k]} | (set(after) - set(before))
    )
    if touched:
        print(
            "this run wrote where it must not, which makes its numbers a report's:\n  "
            + "\n  ".join(str(Path(t).relative_to(REPO)) for t in touched),
            file=sys.stderr,
        )
        return 1
    print(f"no report and no stored residual was touched ({len(after)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
