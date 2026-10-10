#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""W, the flight-plan window, measured on the training year: ``reports/window.md``.

Section 1 of ``reports/preregistered-2026-10-09-records.md`` defines the
measurement and its two conditions, fixed before any count existed:

- over every 2025 departure with ``LOBT_flt`` present, the off-block time less
  the plan, ``BLOCK_TIME_UTC_mvt - LOBT_flt`` in seconds; W* is its largest
  absolute value. Printed per airport, with the counts beyond 3,600 and
  3,900 s, the three plan-shift class rows' differences, and the histogram in
  30 s bins over [-4,200, +4,200];
- **valid** only if both hold: W* is at most 3,900 s, and the rows whose
  absolute difference lies in (W* - 300, W*] number at least half of those in
  (W* - 600, W* - 300], a wall as a join condition would leave rather than a
  tail thinning into the edge.

The submission's window, ``taxiout.window``, holds W* as ``W_STAR_S``; this
script measures it again and refuses to write if the two differ or if the
measurement is not valid, so the constant cannot drift from the data it was
read on. The plan-shift class is the record-quality term and the rule fitted
on the whole training year, as the submission fits them. No fit beyond those
two, no ranking file, no truth beyond the training year's own.

Usage::

    python scripts/report_window.py            # writes reports/window.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import monthly as c  # noqa: E402
from taxiout import tails, window  # noqa: E402
from taxiout.challenge import movement_files, read_table, to_pandas  # noqa: E402
from taxiout.provenance import FROZEN_DOCUMENTS, digest  # noqa: E402

DATA = REPO / "data" / "challenge"
RULE = "reports/preregistered-2026-10-09-records.md"
DEFAULT_OUT = REPO / "reports" / "window.md"
COLUMNS = [
    "MVT_ID_mvt",
    "ADEP_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "EOBT_1_flt",
    "BLOCK_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt",
    "STAND_mvt",
]


def rule_numbers() -> dict:
    """The section's numbers in its words, held in a function so that the
    provenance register has no module-level literal to categorise:

    - "W* <= 3,900 s";
    - "the rows with |Δ| in (W* - 300, W*] number at least half of those in
      (W* - 600, W* - 300]";
    - "the counts beyond 3,600 and 3,900 s";
    - "the histogram of Δ in 30 s bins over [-4,200, +4,200]".
    """
    return {
        "cap_s": 3900.0,
        "wall_s": 300.0,
        "half": 0.5,
        "beyond_s": (3600.0, 3900.0),
        "histogram": (-4200.0, 4200.0, 30.0),
    }


def rule_copy_checked() -> Path:
    path = REPO / RULE
    held = {f.path: f.sha256 for f in FROZEN_DOCUMENTS}
    if RULE not in held or digest(path) != held[RULE]:
        raise SystemExit(f"{RULE} is not at the digest the register holds")
    return path


def training() -> pd.DataFrame:
    """The 2025 departures, each with the month of the file it came from."""
    parts = []
    for f in movement_files(DATA):
        t = to_pandas(read_table(f, COLUMNS, phase="DEP"))
        t["month"] = int(f.stem.split("_")[1][5:7])
        parts.append(t)
    frame = pd.concat(parts, ignore_index=True)
    frame["ADEP_mvt"] = frame["ADEP_mvt"].astype(object)
    return frame


def seconds(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (
        (pd.to_datetime(a, utc=True) - pd.to_datetime(b, utc=True))
        .dt.total_seconds()
        .to_numpy(dtype=float)
    )


def measure(frame: pd.DataFrame) -> dict:
    """The section's measurement, its verdict and what it prints beside."""
    n = rule_numbers()
    has = frame["LOBT_flt"].notna().to_numpy()
    d = frame[has]
    delta = seconds(d["BLOCK_TIME_UTC_mvt"], d["LOBT_flt"])
    if np.isnan(delta).any():
        raise SystemExit("a departure with LOBT_flt has no block time")
    absd = np.abs(delta)
    w_star = float(absd.max())
    near = int(((absd > w_star - n["wall_s"]) & (absd <= w_star)).sum())
    inner = int(
        ((absd > w_star - 2 * n["wall_s"]) & (absd <= w_star - n["wall_s"])).sum()
    )
    cap = w_star <= n["cap_s"]
    wall = near >= n["half"] * inner
    out = {
        "rows": int(len(d)),
        "departures": int(len(frame)),
        "w_star": w_star,
        "near": near,
        "inner": inner,
        "cap": cap,
        "wall": wall,
        "valid": cap and wall,
        "beyond": {b: int((absd > b).sum()) for b in n["beyond_s"]},
    }
    icao = d["ADEP_mvt"].astype(str).to_numpy()
    out["per_airport"] = {
        a: {
            "rows": int((icao == a).sum()),
            "w_star": float(absd[icao == a].max()),
            "min": float(delta[icao == a].min()),
            "max": float(delta[icao == a].max()),
            **{b: int((absd[icao == a] > b).sum()) for b in n["beyond_s"]},
        }
        for a in sorted(set(icao))
    }
    lo, hi, step = n["histogram"]
    edges = np.arange(lo, hi + step, step)
    counts, _ = np.histogram(delta, bins=edges)
    out["histogram"] = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "rows": counts})
    out["outside"] = int(((delta < lo) | (delta >= hi)).sum())
    term = tails.TailModel().fit(frame)
    cls = tails.PlanShiftModel(term).fit(frame).in_class(frame).to_numpy(dtype=bool)
    k = frame[cls]
    out["plan_shift"] = [
        {
            "airport": str(r.ADEP_mvt),
            "month": int(r.month),
            "delta_s": float((r.BLOCK_TIME_UTC_mvt - r.LOBT_flt).total_seconds()),
            "lag_s": float((r.MVT_TIME_UTC_mvt - r.LOBT_flt).total_seconds()),
            "y": float(r.TAXITIME_SEC_mvt),
        }
        for r in k.itertuples()
    ]
    return out


MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def build_report(m: dict, inputs: list[Path]) -> str:
    n = rule_numbers()
    w = m["w_star"]
    lo_b, hi_b = n["beyond_s"]
    L: list[str] = []
    add = L.append
    add("# W: the flight-plan window")
    add("")
    add(
        "Generated by `scripts/report_window.py` from the twelve 2025 training files,"
        f" under §1 of `{RULE}`, whose digest is checked against the register before"
        " anything is read. The difference is `BLOCK_TIME_UTC_mvt - LOBT_flt` in"
        " seconds over every 2025 departure with `LOBT_flt` present; W* is its"
        " largest absolute value. The two conditions and every threshold below are"
        " the section's, fixed before any count existed. Public discussion of the"
        " challenge prompted the test, and the section says so; the measurement is"
        " this entry's own, on its own data."
    )
    add("")
    add("## The verdict")
    add("")
    add("```")
    add(f"W: {'valid' if m['valid'] else 'not valid'}")
    add("```")
    add("")
    add(
        f"- W* = **{w:,.0f} s**, at most {n['cap_s']:,.0f} s: {'yes' if m['cap'] else 'no'}."
    )
    add(
        f"- The wall: {m['near']:,} rows with an absolute difference in"
        f" ({w - n['wall_s']:,.0f}, {w:,.0f}] against {m['inner']:,} in"
        f" ({w - 2 * n['wall_s']:,.0f}, {w - n['wall_s']:,.0f}]; half of the latter is"
        f" {n['half'] * m['inner']:,.1f}: {'yes' if m['wall'] else 'no'}."
    )
    add(
        f"- `taxiout.window` holds W* as `W_STAR_S` = {window.W_STAR_S:,} s and applies"
        f" W' = W* + {window.REDRAW_S} s = {window.w_prime_s():,} s to the ranking rows,"
        " whose take-off times the organizers redraw within"
        f" {window.REDRAW_S} s either way."
    )
    add("")
    add("## The difference over the training year")
    add("")
    add(
        f"{m['rows']:,} of the {m['departures']:,} departures of 2025 carry `LOBT_flt`."
        f" Beyond {lo_b:,.0f} s: {m['beyond'][lo_b]:,} rows; beyond {hi_b:,.0f} s:"
        f" {m['beyond'][hi_b]:,}. Outside the histogram's range: {m['outside']:,} rows."
    )
    add("")
    rows = [
        [
            a,
            f"{v['rows']:,}",
            f"{v['w_star']:,.0f}",
            f"{v['min']:,.0f}",
            f"{v['max']:,.0f}",
            f"{v[lo_b]:,}",
            f"{v[hi_b]:,}",
        ]
        for a, v in m["per_airport"].items()
    ]
    L.extend(
        c.table(
            [
                "airport",
                "rows",
                "W* (s)",
                "smallest (s)",
                "largest (s)",
                f"beyond {lo_b:,.0f} s",
                f"beyond {hi_b:,.0f} s",
            ],
            rows,
            "lrrrrrr",
        )
    )
    add("")
    add("## The plan-shift class")
    add("")
    add(
        "The record-quality term and the plan-shifted departures' rule fitted on the"
        " whole training year, as the submission fits them; the class's departures,"
        " one row per departure, without identifiers:"
    )
    add("")
    L.extend(
        c.table(
            [
                "airport",
                "month",
                "difference (s)",
                "take-off less plan (s)",
                "recorded (s)",
            ],
            [
                [
                    r["airport"],
                    MONTHS[r["month"] - 1],
                    f"{r['delta_s']:,.0f}",
                    f"{r['lag_s']:,.0f}",
                    f"{r['y']:,.0f}",
                ]
                for r in m["plan_shift"]
            ],
            "llrrr",
        )
    )
    add("")
    add("## The histogram at its edges")
    add("")
    h = m["histogram"]
    edge = h[(h["hi"] <= -w + 2 * n["wall_s"]) | (h["lo"] >= w - 2 * n["wall_s"])]
    shown = edge[edge["rows"] > 0]
    empty = edge[edge["rows"] == 0]
    add(
        f"The {n['histogram'][2]:,.0f} s bins of the difference within"
        f" {2 * n['wall_s']:,.0f} s of the edges and beyond them; every bin not shown"
        f" there holds no row ({len(empty):,} bins)."
    )
    add("")
    L.extend(
        c.table(
            ["bin (s)", "rows"],
            [
                [f"[{r.lo:,.0f}, {r.hi:,.0f})", f"{int(r.rows):,}"]
                for r in shown.itertuples()
            ],
            "lr",
        )
    )
    add("")
    L += c.inputs_footer(inputs)
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    before = c.fingerprint()
    copy = rule_copy_checked()
    frame = training()
    m = measure(frame)
    print(
        f"W: {'valid' if m['valid'] else 'not valid'}; W* {m['w_star']:,.0f} s;"
        f" wall {m['near']:,} against {m['inner']:,}"
    )
    if not m["valid"]:
        raise SystemExit("W is not valid on the training year; nothing written")
    if m["w_star"] != window.W_STAR_S:
        raise SystemExit(
            f"W* measures {m['w_star']:,.0f} s and taxiout.window holds"
            f" {window.W_STAR_S:,} s; nothing written"
        )
    inputs = list(movement_files(DATA)) + [copy]
    text = build_report(m, inputs)
    c.write(
        args.out, text, c.stands_of(frame), c.movement_ids(frame["MVT_ID_mvt"]), before
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
