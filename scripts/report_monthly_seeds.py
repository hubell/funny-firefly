#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Five seeds per held-out month: ``reports/monthly_seeds.md``.

Reads ``scripts/report_monthly.py``'s held-out files (seed offset 0) and the
four further offsets ``scripts/fit_monthly_seeds.py`` stored under
``data/processed/monthly_seeds/``, and ``scripts/report_monthly_ladder.py``'s
residuals for the re-reading of its *In how many months each block helps*.
Nothing is refitted.

The five-seed average is the model that ships: ``taxiout.models.SeedAverage``
fits ``BAGGING_SEEDS`` boosters whose seeds run consecutively from the ladder's
and averages their predictions. Its residual here is the mean of the five
residuals, which is the recorded taxi-out minus the mean of the five
predictions.

Usage::

    python scripts/report_monthly_seeds.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import monthly as c  # noqa: E402
from monthly import rb, rm  # noqa: E402
import fit_monthly_seeds as fs  # noqa: E402
from taxiout.models import (  # noqa: E402
    BAGGING_SEEDS,
    LADDER,
    LADDER_BY_KEY,
    SEED_AVERAGE_ADOPTED,
    SUBMISSION_KEY,
    base_of,
)

DEFAULT_OUT = c.REPO / "reports" / "monthly_seeds.md"
OFFSETS = (0,) + fs.OFFSETS
FLOOR_FACTOR = float(rb.NOISE_FLOOR_FACTOR)
# Held-out December trains on January to November, which is the ladder's
# December split, and its five offsets are the seeds the seed-averaging section
# of reports/baseline.md fitted on that split. So its figures must be those.
LADDER_MONTH = "2025-12"
KEYS = tuple(r.key for r in LADDER)


def offset_path(month: str, offset: int) -> Path:
    if offset == 0:
        return c.MONTHLY / f"heldout-{month}_{SUBMISSION_KEY}.parquet"
    return fs.seed_path(month, offset)


class Seeds:
    """The five offsets' residuals per held-out month, joined to the target."""

    def __init__(self, dep: pd.DataFrame) -> None:
        dep_ids = dep["MVT_ID_mvt"].to_numpy()
        y_all = c.target(dep).to_numpy()
        self.files: list[Path] = []
        self.months = {}
        for m in c.MONTHS:
            if not fs.month_done(m):
                raise SystemExit(
                    f"{m}: the seed fits are not on disk; run scripts/fit_monthly_seeds.py --fit"
                )
            frames = []
            for o in OFFSETS:
                p = offset_path(m, o)
                self.files.append(p)
                frames.append(pd.read_parquet(p))
            first = frames[0]
            for o, f in zip(OFFSETS, frames):
                for col in ("mvt_id", "icao", "flagged"):
                    if not np.array_equal(f[col].to_numpy(), first[col].to_numpy()):
                        raise SystemExit(f"{m} offset {o}: {col} differs from offset 0")
                if not np.array_equal(
                    f[rm.TERM_COLUMN].to_numpy()[first["flagged"].to_numpy()],
                    first[rm.TERM_COLUMN].to_numpy()[first["flagged"].to_numpy()],
                ):
                    raise SystemExit(
                        f"{m} offset {o}: the term's residual differs from offset 0"
                    )
            ids = first["mvt_id"].to_numpy()
            y = y_all[c.position(dep_ids, ids)]
            R = np.column_stack([f["residual"].to_numpy(dtype=float) for f in frames])
            flagged = first["flagged"].to_numpy(dtype=bool)
            if not (np.isfinite(y).all() and np.isfinite(R).all()):
                raise SystemExit(f"{m}: a target or a stored residual is not finite")
            self.months[m] = {
                "ids": ids,
                "icao": first["icao"].astype(str).to_numpy(),
                "y": y,
                "R": R,
                "avg": R.mean(axis=1),
                "flagged": flagged,
                "term_resid": first[rm.TERM_COLUMN].to_numpy(dtype=float),
                "run": json.loads(fs.run_path(m).read_text(encoding="utf-8")),
            }


def readings(d: dict, mask: np.ndarray | None = None) -> dict:
    """Every reading over one set of rows: per offset and for the average."""
    keep = np.ones(len(d["y"]), bool) if mask is None else mask
    ex = keep & ~d["flagged"]
    d4 = ex & (d["y"] <= c.DAY_SHIFT_S)
    out = {
        "n": int(keep.sum()),
        "n_ex": int(ex.sum()),
        "n_d4": int(d4.sum()),
        "n_term": int((keep & d["flagged"]).sum()),
    }
    out["sse_ex"] = [float(np.sum(d["R"][ex, j] ** 2)) for j in range(len(OFFSETS))]
    out["sse_d4"] = [float(np.sum(d["R"][d4, j] ** 2)) for j in range(len(OFFSETS))]
    out["avg_sse_ex"] = float(np.sum(d["avg"][ex] ** 2))
    out["avg_sse_d4"] = float(np.sum(d["avg"][d4] ** 2))
    out["avg_s1_ex"] = float(np.sum(d["avg"][ex]))
    out["avg_s1_d4"] = float(np.sum(d["avg"][d4]))
    wt = np.where(d["flagged"], d["term_resid"], d["avg"])
    out["avg_sse_term"] = float(np.sum(wt[keep] ** 2))
    out["avg_sse_all"] = float(np.sum(d["avg"][keep] ** 2))
    return out


def add_readings(parts: list[dict]) -> dict:
    out = {}
    for k in parts[0]:
        if isinstance(parts[0][k], list):
            out[k] = [sum(p[k][j] for p in parts) for j in range(len(parts[0][k]))]
        else:
            out[k] = sum(p[k] for p in parts)
    return out


def root(s: float, n: int) -> float:
    return float(np.sqrt(s / n)) if n else float("nan")


def figures(r: dict) -> dict:
    singles = [root(s, r["n_ex"]) for s in r["sse_ex"]]
    singles4 = [root(s, r["n_d4"]) for s in r["sse_d4"]]
    sd = float(np.std(singles, ddof=1))
    sd4 = float(np.std(singles4, ddof=1))
    return {
        "singles": singles,
        "singles4": singles4,
        "sd": sd,
        "sd4": sd4,
        "floor": FLOOR_FACTOR * sd,
        "floor4": FLOOR_FACTOR * sd4,
        "avg": root(r["avg_sse_ex"], r["n_ex"]),
        "avg4": root(r["avg_sse_d4"], r["n_d4"]),
        "term": root(r["avg_sse_term"], r["n"]),
        "booster": root(r["avg_sse_all"], r["n"]),
        "bias": -r["avg_s1_ex"] / r["n_ex"] if r["n_ex"] else float("nan"),
        "bias4": -r["avg_s1_d4"] / r["n_d4"] if r["n_d4"] else float("nan"),
        "n": r["n"],
        "n_ex": r["n_ex"],
        "n_d4": r["n_d4"],
        "n_term": r["n_term"],
    }


# --------------------------------------------------------------------------
# The held-out-month ladder, read back for its per-month changes


def ladder_sums(dep: pd.DataFrame) -> tuple[dict, list[Path]]:
    """Per (month, rung): rows and squared-error sums, excluding flagged rows
    and over the rows of 4 h or less."""
    dep_ids = dep["MVT_ID_mvt"].to_numpy()
    y_all = c.target(dep).to_numpy()
    out, files = {}, []
    for m in c.MONTHS:
        first = None
        for key in KEYS:
            p = c.MONTHLY_LADDER / f"{m}_{key}.parquet"
            files.append(p)
            f = pd.read_parquet(p, columns=["mvt_id", "residual", "flagged"])
            if first is None:
                first = f
                y = y_all[c.position(dep_ids, f["mvt_id"].to_numpy())]
                ex = ~f["flagged"].to_numpy(dtype=bool)
                d4 = ex & (y <= c.DAY_SHIFT_S)
            elif not np.array_equal(f["mvt_id"].to_numpy(), first["mvt_id"].to_numpy()):
                raise SystemExit(f"{m}_{key}: rows differ")
            r = f["residual"].to_numpy(dtype=float)
            out[(m, key)] = {
                "n_ex": int(ex.sum()),
                "n_d4": int(d4.sum()),
                "sse_ex": float(np.sum(r[ex] ** 2)),
                "sse_d4": float(np.sum(r[d4] ** 2)),
            }
    return out, files


def printed_changes() -> dict[str, list[str]]:
    """``reports/monthly_ladder.md``'s *Changes against the rung each is read
    against* table, as printed."""
    text = (
        (c.REPO / "reports" / "monthly_ladder.md")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    at = text.index("### Changes against the rung each is read against (s)")
    rows = {}
    for line in text[at + 1 :]:
        if line.startswith("#"):
            break
        if line.startswith("| `"):
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


# --------------------------------------------------------------------------
# the report


def baseline_december() -> dict:
    """``reports/baseline.md``'s December seed figures, as printed."""
    text = (c.REPO / "reports" / "baseline.md").read_text(encoding="utf-8").splitlines()
    out = {}
    for line in text:
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("| December |") and len(cells) == 5:
            out["one"], out["avg"], out["change"] = cells[1], cells[2], cells[3]
        if line.startswith("| December, all ten |") and len(cells) == 4:
            out["range"], out["sd"], out["floor"] = cells[1], cells[2], cells[3]
    return out


def build(
    st: Seeds, ladder: dict, inputs: list[Path], check: dict, log: dict
) -> tuple[str, dict]:
    L: list[str] = []
    add = L.append
    per_month = {m: readings(d) for m, d in st.months.items()}
    fig = {m: figures(r) for m, r in per_month.items()}
    fig["year"] = figures(add_readings(list(per_month.values())))
    cols = [c.month_label(m) for m in c.MONTHS] + ["year"]
    keys = list(c.MONTHS) + ["year"]

    add("# Five seeds per held-out month")
    add("")
    add(
        "Generated by `scripts/report_monthly_seeds.py` from the residuals of "
        "`scripts/fit_monthly_seeds.py` under `data/processed/monthly_seeds/`, the held-out-month "
        "fits' files under `data/processed/monthly/` and the held-out-month ladder's under "
        "`data/processed/monthly_ladder/`; nothing is refitted, every figure is computed there, and the generator "
        "refuses to write a movement identifier or a literal stand identifier. **It decides nothing.**"
    )
    add("")
    add("## What this is")
    add("")
    add(
        f"`{LADDER_BY_KEY[SUBMISSION_KEY].label}` (`SUBMISSION_KEY`) on each of the twelve held-out-month splits of "
        "`reports/monthly.md` and `reports/monthly_ladder.md` (one month of 2025 held out, the other eleven training, so not forward in time), under "
        f'{len(OFFSETS)} seed offsets on `BOOSTER_PARAMS["random_state"]`, {", ".join(map(str, OFFSETS))}, as '
        "`run_baselines`' seed-averaging section fits them. Offset 0 is the held-out-month fit's file (`reports/monthly.md`) "
        "and was not refitted; offsets 1 to 4 were fitted by `scripts/report_monthly.py`'s own "
        "`fit_leg` with only the offset moved. Everything else, the "
        "record-quality term on the training months, `fit_rows` with the 4 h exclusion, the shrunk encoding and "
        "`BOOSTER_PARAMS`, is the ladder's."
    )
    add("")
    add(
        f"**The five-seed average is the model that ships.** `taxiout.models.SEED_AVERAGE_ADOPTED` is "
        f"`{SEED_AVERAGE_ADOPTED}` and the submission is the mean of `BAGGING_SEEDS` = {BAGGING_SEEDS} boosters "
        "whose seeds run consecutively from the ladder's. Its residual here is the mean of the five residuals, the "
        "recorded taxi-out minus the mean of the five predictions. `reports/monthly.md` and `reports/monthly_ladder.md` scored one seed."
    )
    add("")
    add(
        "The floor is the standard deviation of the five single-seed figures times the square root of two "
        "(`run_baselines.NOISE_FLOOR_FACTOR`), as `reports/baseline.md`'s *The noise floor* defines it: the spread "
        "of a difference between two single-seed fits whose seed noise is independent. Every margin is pooled from "
        "the rows it names; nothing is a mean of RMSEs. Beside every scored figure, in brackets, the same reading "
        "over the rows whose recorded taxi-out is 4 h or less, a diagnostic, with the count it leaves out."
    )
    add("")

    # checks
    add("## The seed path, and December against the ladder")
    add("")
    add(
        "Before the 48 fits, offset 0 was refitted for January by the same path and compared with the held-out-month "
        f"fits' January file: {check['summary']}. So the path the other offsets took is "
        "the held-out-month fits'."
    )
    add("")
    bd = baseline_december()
    dec = fig[LADDER_MONTH]
    mine = {
        "range": f"{min(dec['singles']):.2f} to {max(dec['singles']):.2f}",
        "sd": f"{dec['sd']:.2f}",
        "floor": f"{dec['floor']:.2f}",
        "one": f"{dec['singles'][0]:.1f}",
        "avg": f"{dec['avg']:.1f}",
        "change": f"{dec['avg'] - dec['singles'][0]:+.2f}".replace("+-", "-"),
    }
    rows = []
    agree = True
    for k, name in (
        ("one", "the ladder's seed (s)"),
        ("avg", "five seeds averaged (s)"),
        ("change", "change (s)"),
        ("range", "range of the five single seeds (s)"),
        ("sd", "standard deviation (s)"),
        ("floor", "floor (s)"),
    ):
        same = mine[k] == bd.get(k)
        agree &= same
        rows.append([name, mine[k], bd.get(k, "missing"), "yes" if same else "**no**"])
    add(
        "Held-out December trains on January to November, which is the ladder's December split, and its five "
        "offsets are the five seeds `reports/baseline.md`'s *Seed averaging* and *The noise floor* fitted there. "
        "So its figures must be those, at the precision printed:"
    )
    add("")
    L += c.table(["December", "here", "reports/baseline.md", "same"], rows, "lrrl")
    add("")
    log["december_agrees"] = agree

    # 1
    add("## Per month and over the year")
    add("")
    add(
        "Excluding the rows the record-quality term covers, as the scorer's rule implies (s):"
    )
    add("")
    rows = []
    for k, col in zip(keys, cols):
        f = fig[k]
        left = f["n_ex"] - f["n_d4"]
        rows.append(
            [col, f"{f['n_ex']:,}"]
            + [f"{x:.2f}" for x in f["singles"]]
            + [
                f"{f['sd']:.2f}",
                f"{f['floor']:.2f}",
                f"**{f['avg']:.2f}**",
                f"{f['avg'] - f['singles'][0]:+.2f}"
                + (f" [{f['avg4'] - f['singles4'][0]:+.2f}]" if left else ""),
                (f"{f['avg4']:.2f} ({left:,} out)" if left else f"{f['avg4']:.2f}"),
            ]
        )
    L += c.table(
        ["month", "rows"]
        + [f"seed {o}" for o in OFFSETS]
        + [
            "standard deviation",
            "floor",
            "five-seed average",
            "average against seed 0",
            "five-seed average, 4 h or less (diagnostic)",
        ],
        rows,
        "lr" + "r" * (len(OFFSETS) + 5),
    )
    add("")
    add(
        "Where a month holds rows over 4 h, the change against seed 0 carries in brackets the same change over the "
        "rows of 4 h or less, a diagnostic; the next table has that reading for every column."
    )
    add("")
    add("The same over the rows of 4 h or less, a diagnostic (s):")
    add("")
    rows = []
    for k, col in zip(keys, cols):
        f = fig[k]
        rows.append(
            [col, f"{f['n_d4']:,}"]
            + [f"{x:.2f}" for x in f["singles4"]]
            + [f"{f['sd4']:.2f}", f"{f['floor4']:.2f}", f"{f['avg4']:.2f}"]
        )
    L += c.table(
        ["month", "rows"]
        + [f"seed {o}" for o in OFFSETS]
        + ["standard deviation", "floor", "five-seed average"],
        rows,
        "lr" + "r" * (len(OFFSETS) + 3),
    )
    add("")
    add(
        "Every row: with the term substituted on the rows it covers, which is what the submission does, and with "
        "those rows left to the five-seed average (s):"
    )
    add("")
    rows = []
    for k, col in zip(keys, cols):
        f = fig[k]
        left = f["n_ex"] - f["n_d4"]
        rows.append(
            [
                col,
                f"{f['n']:,}",
                f"{f['n_term']:,}",
                f"{f['term']:.1f}",
                f"{f['booster']:.1f}",
                c.cell_text(f["avg"], f["avg4"], f["n_ex"], left),
                c.cell_text(f["singles"][0], f["singles4"][0], f["n_ex"], left),
            ]
        )
    L += c.table(
        [
            "month",
            "rows",
            "term rows",
            "every row, with the term",
            "every row, booster only",
            "excluding flagged, five-seed average [4 h or less, diagnostic]",
            "excluding flagged, seed 0 (`reports/monthly.md`) [4 h or less, diagnostic]",
        ],
        rows,
        "lrrrrrr",
    )
    add("")
    y = fig["year"]
    beats = sum(fig[m]["avg"] < min(fig[m]["singles"]) for m in c.MONTHS)
    beats4 = sum(fig[m]["avg4"] < min(fig[m]["singles4"]) for m in c.MONTHS)
    lo_m = min(c.MONTHS, key=lambda m: fig[m]["floor"])
    hi_m = max(c.MONTHS, key=lambda m: fig[m]["floor"])
    add(
        f"Pooled over the year, the five-seed average reads {y['avg']:.1f} s excluding flagged rows "
        f"({y['avg4']:.1f} s over the rows of 4 h or less, a diagnostic), {y['term']:.1f} s every row with the term, "
        f"against {y['singles'][0]:.1f} s for the ladder's seed alone ({y['singles4'][0]:.1f} s over the rows of "
        f"4 h or less, a diagnostic). The year's floor is {y['floor']:.2f} s ({y['floor4']:.2f} s, diagnostic). The "
        f"monthly floors run from {fig[lo_m]['floor']:.2f} s ({c.month_label(lo_m)}; {fig[lo_m]['floor4']:.2f} s "
        f"diagnostic) to {fig[hi_m]['floor']:.2f} s ({c.month_label(hi_m)}; {fig[hi_m]['floor4']:.2f} s diagnostic). "
        f"The average beats every one of its five single seeds in {beats} of 12 months ({beats4} of 12 over the rows "
        "of 4 h or less, a diagnostic)."
    )
    add("")
    log["fig"] = {k: {kk: vv for kk, vv in v.items()} for k, v in fig.items()}

    # 2
    add("## Per airport and month: the five-seed average")
    add("")
    add(
        "RMSE excluding flagged rows, then bias (prediction minus recorded taxi-out over the same rows). The "
        "margins pool the rows they cover: the *year* column an airport's twelve months, the *all 10* row a "
        "month's ten airports. A cell under 30 rows prints its count only."
    )
    grid = {}
    # the airports with departures; LTAI is in the airport table and has none
    airports = sorted(
        set(c.ICAOS) & set(np.concatenate([d["icao"] for d in st.months.values()]))
    )
    for m, d in st.months.items():
        for ap in airports:
            grid[(ap, m)] = readings(d, d["icao"] == ap)
    for ap in airports:
        grid[(ap, "year")] = add_readings([grid[(ap, m)] for m in c.MONTHS])
    for m in c.MONTHS:
        grid[("all 10", m)] = per_month[m]
    grid[("all 10", "year")] = add_readings(list(per_month.values()))
    for what, title in (
        ("rmse", "RMSE, five-seed average (s)"),
        ("bias", "Bias, five-seed average (s)"),
    ):
        add("")
        add(f"### {title}")
        add("")
        rows = []
        for ap in list(airports) + ["all 10"]:
            row = [ap]
            for k in keys:
                g = grid[(ap, k)]
                f = figures(g) if g["n_ex"] > 1 else None
                n_ex, left = g["n_ex"], g["n_ex"] - g["n_d4"]
                if what == "rmse":
                    a, b = root(g["avg_sse_ex"], n_ex), root(g["avg_sse_d4"], g["n_d4"])
                    row.append(c.cell_text(a, b, n_ex, left))
                else:
                    a = -g["avg_s1_ex"] / n_ex if n_ex else float("nan")
                    b = -g["avg_s1_d4"] / g["n_d4"] if g["n_d4"] else float("nan")
                    row.append(c.cell_text(a, b, n_ex, left, lambda x: c.signed(x, 0)))
                del f
            rows.append(row)
        L += c.table(["ICAO"] + cols, rows, "l" + "r" * len(cols))
    add("")

    # 3
    add("## Each block's months, against each month's floor")
    add("")
    add(
        "`reports/monthly_ladder.md` counted a block as helping in a month when its change, excluding flagged rows, printed below "
        "zero. Here each change is set against that month's floor from the table above, on the unrounded figures: "
        "a change more negative than minus the floor helps by more than the floor, one more positive than the "
        "floor hurts by more than it, and anything between sits inside it, which is within what changing the seed "
        "alone produces. **The floor is measured on the shipped rung and applied to every block's change, which is "
        "the same approximation `reports/baseline.md` makes** (its *The noise floor*: the spread is measured on one "
        "rung and assumed for the others; the reference rungs draw no seed, so where one is on a side the floor "
        "overstates the seed's share). With "
        f"{BAGGING_SEEDS} seeds the standard deviation is itself uncertain by about "
        f"{1.0 / np.sqrt(2.0 * (BAGGING_SEEDS - 1)):.0%}, as that section computes it. "
        "The changes are recomputed from the held-out-month ladder's residuals, and each is checked "
        "against the figure `reports/monthly_ladder.md` printed."
    )
    add("")
    floors = {k: fig[k]["floor"] for k in keys}
    floors4 = {k: fig[k]["floor4"] for k in keys}
    add(
        "The floors (s), excluding flagged rows [over the rows of 4 h or less, a diagnostic]:"
    )
    add("")
    L += c.table(
        ["", *cols],
        [["floor"] + [f"{floors[k]:.2f} [{floors4[k]:.2f}]" for k in keys]],
        "l" + "r" * len(cols),
    )
    add("")
    printed = printed_changes()
    pairs = []
    for rung in LADDER:
        base = base_of(rung)
        if base is not None:
            pairs.append((rung, base))
    rows, counts, mismatches = [], [], []
    for rung, base in pairs:
        name = (
            f"`{rung.label}`"
            + (" (diagnostic)" if rung.diagnostic else "")
            + f" against `{base.label}`"
        )
        row = [name]
        tally = {"helps": [], "hurts": [], "inside": []}
        tally4 = {"helps": [], "hurts": [], "inside": []}
        want = printed.get(name)
        for j, k in enumerate(keys):
            if k == "year":
                a = {
                    x: sum(ladder[(m, rung.key)][x] for m in c.MONTHS)
                    for x in ("n_ex", "n_d4", "sse_ex", "sse_d4")
                }
                b = {
                    x: sum(ladder[(m, base.key)][x] for m in c.MONTHS)
                    for x in ("n_ex", "n_d4", "sse_ex", "sse_d4")
                }
            else:
                a, b = ladder[(k, rung.key)], ladder[(k, base.key)]
            ch = root(a["sse_ex"], a["n_ex"]) - root(b["sse_ex"], b["n_ex"])
            ch4 = root(a["sse_d4"], a["n_d4"]) - root(b["sse_d4"], b["n_d4"])
            left = a["n_ex"] - a["n_d4"]
            mine = c.signed1(ch) + (f" [{c.signed1(ch4)}; {left:,} out]" if left else "")
            if want is None or j >= len(want) or want[j] != mine:
                mismatches.append(
                    f"{name}, {cols[j]}: here {mine}, printed {want[j] if want and j < len(want) else 'missing'}"
                )
            verdict = (
                "helps" if ch < -floors[k] else "hurts" if ch > floors[k] else "inside"
            )
            verdict4 = (
                "helps"
                if ch4 < -floors4[k]
                else "hurts"
                if ch4 > floors4[k]
                else "inside"
            )
            if k != "year":
                tally[verdict].append(cols[j])
                tally4[verdict4].append(cols[j])
            text = c.signed1(ch)
            cell = f"**{text}**" if verdict == "inside" else text
            if left:
                t4 = c.signed1(ch4)
                cell += (
                    f" [{'**' + t4 + '**' if verdict4 == 'inside' else t4}; {left:,} out]"
                )
            row.append(cell)
        rows.append(row)
        counts.append(
            [
                name,
                f"{len(tally['helps'])} of 12",
                f"{len(tally['hurts'])}",
                ", ".join(tally["hurts"]) or "none",
                f"{len(tally['inside'])}",
                ", ".join(tally["inside"]) or "none",
                f"{len(tally4['helps'])} / {len(tally4['hurts'])} / {len(tally4['inside'])}",
            ]
        )
        log.setdefault("blocks", {})[rung.key] = {
            "helps": tally["helps"],
            "hurts": tally["hurts"],
            "inside": tally["inside"],
            "helps4": tally4["helps"],
            "hurts4": tally4["hurts"],
            "inside4": tally4["inside"],
        }
    log["mismatches"] = mismatches
    add(
        "Each block's change against the rung it is read against, excluding flagged rows (s). **A change in bold "
        "sits inside that month's floor**, the convention of `reports/baseline.md`'s *The noise floor*; the year "
        "column is read against the year's floor. Where a cell holds rows over 4 h, the bracket is the same change "
        "over the rows of 4 h or less, a diagnostic, with the count it leaves out, read against the floor measured "
        "over those rows and bold on the same rule."
    )
    add("")
    L += c.table(["block", *cols], rows, "l" + "r" * len(cols))
    add("")
    add(
        f"Recomputed from the held-out-month ladder's residuals, every cell of that table, with its "
        f"bracketed reading, equals the one `reports/monthly_ladder.md` printed: {'yes' if not mismatches else f'**no, {len(mismatches)} cells differ**'}."
    )
    add("")
    add(
        "Per block, the months it helps or hurts by more than the floor, and the months inside it:"
    )
    add("")
    L += c.table(
        [
            "block",
            "helps by more than the floor",
            "hurts by more than the floor",
            "months",
            "inside the floor",
            "months",
            "4 h or less: helps / hurts / inside (diagnostic)",
        ],
        counts,
        "lrrlrlr",
    )
    add("")
    chain = [
        r
        for r, b in pairs
        if not r.diagnostic and r.kind == "booster" and b.kind == "booster"
    ]
    most = [r for r in chain if len(log["blocks"][r.key]["helps"]) >= 7]
    add(
        "Of the boosted blocks, each read against the rung `reports/baseline.md` reads it against, those that "
        "still help by more than the floor in most months (7 or more of 12), with the count: "
        + (
            ", ".join(f"`{r.label}` ({len(log['blocks'][r.key]['helps'])})" for r in most)
            or "none"
        )
        + "; the rest: "
        + ", ".join(
            f"`{r.label}` ({len(log['blocks'][r.key]['helps'])})"
            for r in chain
            if r not in most
        )
        + "."
    )
    add("")

    # timings
    add("## What the fits cost")
    add("")
    add(
        "Monotonic clock, from each month's run record; the 48 fits ran in one process under a sleep inhibitor "
        "after the January check."
    )
    add("")
    rows = []
    for m in c.MONTHS:
        run = st.months[m]["run"]
        f = run["fits"]
        rows.append(
            [c.month_label(m), run["started_utc"].replace("+00:00", "Z")]
            + [f"{x['fit_s']:.0f}" for x in f]
            + [
                f"{sum(x['prepare_s'] + x['predict_s'] for x in f):.0f}",
                f"{run['month_s']:.0f}",
                f"{max(x['peak_rss_gib'] for x in f):.1f}",
            ]
        )
    L += c.table(
        ["month", "started (UTC)"]
        + [f"offset {o} fit (s)" for o in fs.OFFSETS]
        + ["term, encoding, predict (s)", "month (s)", "peak RSS (GiB)"],
        rows,
        "ll" + "r" * 7,
    )
    add("")
    L += c.inputs_footer(
        rb._inputs()
        + inputs
        + [c.REPO / "reports" / "baseline.md", c.REPO / "reports" / "monthly_ladder.md"]
        + [fs.run_path(m) for m in c.MONTHS]
    )
    return "\n".join(L) + "\n", fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    before = c.fingerprint()
    dep = c.departures()
    st = Seeds(dep)
    ladder, ladder_files = ladder_sums(dep)
    chk = json.loads(fs.CHECK_RECORD.read_text(encoding="utf-8"))
    check = {"summary": "; ".join(chk["lines"]), "ok": chk["ok"]}
    if not chk["ok"]:
        raise SystemExit("the seed-path check did not pass")
    log: dict = {}
    text, fig = build(st, ladder, st.files + ladder_files + [fs.CHECK_RECORD], check, log)
    c.write(args.out, text, c.stands_of(dep), c.movement_ids(dep["MVT_ID_mvt"]), before)
    bad = []
    if not log["december_agrees"]:
        bad.append("December does not equal reports/baseline.md's seed figures")
    if log["mismatches"]:
        bad.append(
            f"{len(log['mismatches'])} of the ladder report's printed changes were not reproduced"
        )
    for b in bad:
        print("DISAGREES: " + b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
