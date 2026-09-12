#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Compare two runs of the pipeline, file by file, and say what differs.

A run here is a directory holding what the pipeline wrote: ``reports/``, with
the generated reports, and ``residuals/``, with the validation residual files
``scripts/run_baselines.py`` keeps. Two runs of a pipeline that repeats agree
on every byte, and this is the check that says whether they did rather than
the claim that they should.

For a residual file that differs it prints the largest difference in any
residual and both files' RMSE, says whether the difference is in the last few
bits of the same trees or is a split that went the other way, and names any
other column that differs. For a report that differs it counts the lines, and
says so separately when the only lines that moved are the date the report was
generated on -- the one thing in a report that comes from the clock rather than
from the computation. It prints aggregates only: never a movement identifier
or any other row of the residual files.

``--expect`` names reports both runs must hold. A report the pipeline failed to
write keeps its previous copy, so its absence from a run is the only sign that
survives into a snapshot; one expected and missing counts as a failure.

Usage
-----
    python scripts/compare_runs.py RUN_A RUN_B
    python scripts/compare_runs.py RUN_A RUN_B --names col-a col-b > out.md
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The same line ``scripts/check_determinism.py`` draws: at or below it two
# predictions come from the same trees; above it a split went the other way.
SAME_TREES_S = 1e-6

# The reports ``make baseline residual occupancy congestion transfer`` write.
PIPELINE_REPORTS = (
    "baseline.md",
    "residual.md",
    "occupancy.md",
    "congestion.md",
    "transfer.md",
)

DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(n: int) -> str:
    return f"{n} line" if n == 1 else f"{n} lines"


def compare_report(a: Path, b: Path) -> tuple[str, str]:
    """``(kind, text)``, kind being ``date`` or ``differ``."""
    left = a.read_text(encoding="utf-8").splitlines()
    right = b.read_text(encoding="utf-8").splitlines()
    blocks = [
        (left[i1:i2], right[j1:j2])
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, left, right, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]
    if not blocks:
        return "differ", "**bytes differ, no line does (line endings or final newline)**"
    # A block counts as many lines as its longer side, so a paragraph that
    # grew or shrank is not undercounted.
    n = sum(max(len(x), len(y)) for x, y in blocks)
    if all(
        len(x) == len(y) and all(DATE.sub("", p) == DATE.sub("", q) for p, q in zip(x, y))
        for x, y in blocks
    ):
        return "date", f"differs only in its generation date ({_lines(n)})"
    return "differ", f"**{_lines(n)} differ**"


def compare_residuals(a: Path, b: Path) -> tuple[str, str]:
    left = pd.read_parquet(a)
    right = pd.read_parquet(b)
    if list(left.columns) != list(right.columns):
        return "differ", "**columns differ**"
    if len(left) != len(right) or not left["mvt_id"].equals(right["mvt_id"]):
        return "differ", "**rows differ**"
    x = left["residual"].to_numpy(dtype=float)
    y = right["residual"].to_numpy(dtype=float)
    others = [c for c in left.columns if c != "residual" and not left[c].equals(right[c])]
    also = f"; {', '.join(others)} also differ" if others else ""
    missing = int((np.isfinite(x) != np.isfinite(y)).sum())
    if missing:
        return "differ", f"**{missing} residuals missing in one run only**{also}"
    finite = np.isfinite(x)
    delta = float(np.max(np.abs(x[finite] - y[finite]), initial=0.0))
    if delta == 0.0:
        if others:
            return "differ", f"**residuals equal; {', '.join(others)} differ**"
        return "differ", "**every value equal; the files differ in encoding or metadata**"
    kind = "last bits" if delta <= SAME_TREES_S else "trees differ"
    rmse_x = float(np.sqrt(np.mean(x[finite] ** 2)))
    rmse_y = float(np.sqrt(np.mean(y[finite] ** 2)))
    return (
        "differ",
        f"**max delta {delta:.3g} s ({kind})**; RMSE {rmse_x:.4f} -> {rmse_y:.4f} s"
        + also,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--names", nargs=2, default=None)
    ap.add_argument(
        "--expect",
        nargs="*",
        default=list(PIPELINE_REPORTS),
        help="reports both runs must hold (default: the five the pipeline writes)",
    )
    args = ap.parse_args()
    name_a, name_b = args.names or (args.run_a.name, args.run_b.name)
    for run in (args.run_a, args.run_b):
        for sub in ("reports", "residuals"):
            if not (run / sub).is_dir():
                print(f"missing directory: {run / sub}", file=sys.stderr)
                return 2

    out = [f"# `{name_a}` against `{name_b}`", ""]
    totals = {"identical": 0, "date": 0, "differ": 0, "missing": 0}
    for sub, pattern, compare in (
        ("reports", "*.md", compare_report),
        ("reports", "*.svg", None),
        ("residuals", "*.parquet", compare_residuals),
    ):
        names = sorted(
            {p.name for p in (args.run_a / sub).glob(pattern)}
            | {p.name for p in (args.run_b / sub).glob(pattern)}
            | (set(args.expect) if pattern == "*.md" else set())
        )
        if not names:
            if pattern == "*.parquet":
                print(f"no {sub}/{pattern} in either run", file=sys.stderr)
                return 2
            continue
        out += [f"## {sub}/{pattern}", "", "| file | result |", "| --- | --- |"]
        for name in names:
            a, b = args.run_a / sub / name, args.run_b / sub / name
            if not (a.exists() and b.exists()):
                where = name_a if a.exists() else name_b if b.exists() else None
                result = f"**only in `{where}`**" if where else "**in neither run**"
                totals["missing"] += 1
            elif sha256(a) == sha256(b):
                result = "identical"
                totals["identical"] += 1
            elif compare is None:
                result = "**differs**"
                totals["differ"] += 1
            else:
                kind, result = compare(a, b)
                totals[kind] += 1
            out.append(f"| `{name}` | {result} |")
        out.append("")
    out.insert(
        2,
        f"{totals['identical']} files byte-identical, {totals['date']} differing only "
        f"in their generation date, {totals['differ']} differing otherwise, "
        f"{totals['missing']} missing from a run.\n",
    )
    out += [
        "Distance matrices derived from OpenStreetMap: data (c)",
        "OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied by",
        "EUROCONTROL PRC and the OpenSky Network under the terms of the PRC Data",
        "Challenge 2026.",
    ]
    sys.stdout.write("\n".join(out) + "\n")
    return 0 if totals["differ"] == totals["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
