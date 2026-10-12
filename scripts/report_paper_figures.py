#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Draw the paper's data figures from the committed reports.

Every plotted value is parsed from a table in ``reports/``; this script holds
only the drawing (sizes, colors, labels). Run it from a checkout:

    python scripts/report_paper_figures.py          write docs/figures/*.svg
    python scripts/report_paper_figures.py --check  compare against the files

Needs no network and no challenge data. The output is plain SVG written by
hand, so it is byte-stable across machines.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
OUT = REPO / "docs" / "figures"

INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"
BLUE, RUST, BAND = "#3b6ea5", "#b5482f", "#e6edf5"
BAND2, EDGE2 = "#c3d4ea", "#8fb0d6"
MINUS = "−"


# ---------------------------------------------------------------- parsing


def number(cell: str) -> float:
    """A report cell such as ``+1.5``, ``-0.69``, ``1.78%`` or ``**+2.5 s**``."""
    s = cell.replace("*", "").replace(MINUS, "-").replace(",", "")
    m = re.search(r"[+-]?\d+(?:\.\d+)?", s)
    if m is None:
        raise ValueError(f"no number in {cell!r}")
    return float(m.group(0))


def interval(cell: str) -> tuple[float, float, float]:
    """``+1.5 s [+0.5, +2.5]`` as (estimate, low, high)."""
    m = re.search(r"\[([^,\]]+),\s*([^\]]+)\]", cell)
    if m is None:
        raise ValueError(f"no interval in {cell!r}")
    return number(cell.split("[")[0]), number(m.group(1)), number(m.group(2))


def tables_after(text: str, heading: str) -> list[list[list[str]]]:
    """The tables under ``heading``, each as a list of rows of cells."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    level = len(heading) - len(heading.lstrip("#"))
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for ln in lines[start + 1 :]:
        if ln.startswith("#") and len(ln) - len(ln.lstrip("#")) <= level:
            break
        if ln.lstrip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def row(table: list[list[str]], first_cell: str) -> dict[str, str]:
    header = table[0]
    for r in table[1:]:
        if r[0].replace("*", "") == first_cell:
            return dict(zip(header, r, strict=False))
    raise KeyError(first_cell)


def read(name: str) -> str:
    return (REPORTS / name).read_text(encoding="utf-8")


def table_headed(text: str, first_header: str) -> list[list[str]]:
    """The first table in ``text`` whose header row starts with ``first_header``."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if ln.lstrip().startswith("|") and cells[0] == first_header:
            table = []
            for later in lines[i:]:
                if not later.lstrip().startswith("|"):
                    break
                row_cells = [c.strip() for c in later.strip().strip("|").split("|")]
                if not set("".join(row_cells)) <= set("-: "):
                    table.append(row_cells)
            return table
    raise KeyError(first_header)


# ---------------------------------------------------------------- drawing


def quartiles(values: list[float]) -> tuple[float, float]:
    """Q1 and Q3 by linear interpolation between order statistics."""
    v = sorted(values)

    def at(q: float) -> float:
        pos = (len(v) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return at(0.25), at(0.75)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v: float, places: int = 1, sign: bool = False) -> str:
    s = f"{abs(v):.{places}f}"
    if v < 0 and float(s) != 0:
        return MINUS + s
    return ("+" + s) if sign else s


class Canvas:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.w, self.h, self.title = width, height, title
        self.items: list[str] = []

    def text(self, x, y, s, size=12, anchor="start", fill=INK, weight="normal"):
        size = max(size, 12.5)
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"'
            f' fill="{fill}" font-weight="{weight}">{esc(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="{stroke}" stroke-width="{width}"{d}/>'
        )

    def rect(self, x, y, w, h, fill, stroke="none"):
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
            f' fill="{fill}" stroke="{stroke}"/>'
        )

    def dot(self, x, y, r=4.5, fill=BLUE, stroke="none"):
        self.items.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}"/>'
        )

    def svg(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}"'
            f' width="{self.w}" height="{self.h}" font-family="sans-serif">\n'
            f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>\n'
            f"<title>{esc(self.title)}</title>\n"
        )
        return head + "\n".join(self.items) + "\n</svg>\n"


class Axis:
    """A linear axis mapped onto pixels."""

    def __init__(self, lo, hi, p0, p1) -> None:
        self.lo, self.hi, self.p0, self.p1 = lo, hi, p0, p1

    def __call__(self, v: float) -> float:
        return self.p0 + (v - self.lo) / (self.hi - self.lo) * (self.p1 - self.p0)


def data_lines(c_width: int, text: str) -> list[str]:
    return textwrap.wrap("Data: " + text, int((c_width - 40) / 6.7))


def data_note(c: Canvas, lines: list[str], y0: float) -> None:
    c.line(20, y0 - 4, c.w - 20, y0 - 4, GRID, 1)
    for i, ln in enumerate(lines):
        c.text(20, y0 + 14 + 18 * i, ln, 12.5, fill=INK)


def ticks(lo: float, hi: float, step: float) -> list[float]:
    out, v = [], lo
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


def legend(c, x, y, items):
    """A row of mark keys: (kind, color, text) with kind dot, whisker or bar."""
    for kind, color, label in items:
        if kind == "dot":
            c.dot(x + 6, y - 4, 4.5, color)
            x += 18
        elif kind == "whisker":
            c.line(x, y - 4, x + 26, y - 4, color, 2)
            c.line(x, y - 9, x, y + 1, color, 2)
            c.line(x + 26, y - 9, x + 26, y + 1, color, 2)
            x += 36
        elif kind == "bar":
            c.rect(x, y - 10, 26, 12, color)
            x += 36
        elif kind == "bar2":
            c.rect(x, y - 10, 26, 12, BAND2, EDGE2)
            x += 36
        elif kind == "band":
            c.rect(x, y - 12, 26, 16, color, GRID)
            x += 36
        c.text(x, y, label, 12.5, fill=MUTED)
        x += 8 + 6.4 * len(label) + 22


def horizontal_axis(c, ax, y, lo, hi, step, label, places=0, grid_to=None):
    c.line(ax(lo), y, ax(hi), y, MUTED)
    for t in ticks(lo, hi, step):
        c.line(ax(t), y, ax(t), y + 4, MUTED)
        c.text(ax(t), y + 17, fmt(t, places), 11, "middle", MUTED)
        if grid_to is not None:
            c.line(ax(t), grid_to, ax(t), y, GRID, 0.7)
    c.text((ax(lo) + ax(hi)) / 2, y + 36, label, 11.5, "middle", INK)


# ---------------------------------------------------------------- figures

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
FULL = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

STEPS = (
    "The booster, with the two feature groups",
    "The refitted CatBoost learner, blended in",
    "The per-airport choice of blend",
    "The ground-movement correction, fitted on 174 days",
    "The same correction, fitted on 348 days",
    "The flight-plan window",
)


def title(c: Canvas, head: str, sub: str) -> None:
    c.text(20, 28, head, 15, weight="bold")
    c.text(20, 50, sub, 12.5, fill=MUTED)


def ladder_tables() -> tuple[dict[str, str], list[tuple[str, dict[str, str]]]]:
    text = read("v12-ladder.md")
    levels = tables_after(text, "### Every row: RMSE (s)")[0]
    changes = tables_after(
        text, "### Every row: each step less the one before (s), 95% interval over days"
    )[0]
    return row(levels, "the four"), [(m, row(changes, m)) for m in (*MONTHS, "the four")]


def figure_ladder_level() -> str:
    lv, _ = ladder_tables()
    level = [number(lv[k]) for k in list(lv)[4:10]]
    n_rows = int(number(lv["rows"]))
    note = data_lines(
        820,
        "2025 flights at all ten airports. January, February, June and July, each held "
        f"out in turn and predicted from the other eleven months ({n_rows:,} rows over "
        f"{int(number(lv['days']))} days). RMSE over every row, including those the "
        "record-quality term covers.",
    )
    c = Canvas(
        820, 420 + 18 * len(note) + 14, "The error on the four months, after each step"
    )
    title(
        c,
        "Figure 3a. The error after each step of the final configuration",
        f"RMSE in seconds, pooled over the four months read ({n_rows:,} rows).",
    )
    top, step = 112, 44
    ax = Axis(292, 308, 400, 780)
    for i, name in enumerate(STEPS):
        c.text(20, top + i * step + 4, f"{i + 1}. {name}", 13)
    base = level[0]
    horizontal_axis(c, ax, top + 5 * step + 30, 292, 308, 4, "RMSE (s)", 0, top - 24)
    c.line(ax(base), top - 24, ax(base), top + 5 * step + 30, MUTED, 1, "5 4")
    c.text(ax(base), top - 32, f"the booster alone, {fmt(base)}", 12.5, "middle", MUTED)
    for i, v in enumerate(level):
        y = top + i * step
        c.dot(ax(v), y, 5.5)
        c.text(ax(v) + 12, y + 4, fmt(v), 13, weight="bold")
    c.text(
        20,
        404,
        "Point estimates. Figure 3b gives what each step changed, with its interval.",
        12.5,
        fill=MUTED,
    )
    data_note(c, note, 420)
    return c.svg()


def figure_ladder_change() -> str:
    _, per_month = ladder_tables()
    four = dict(per_month)["the four"]
    keys = list(four)[1:6]
    est = [interval(four[k]) for k in keys]
    monthly = [[interval(r[k])[0] for m, r in per_month if m in MONTHS] for k in keys]
    iqr = [quartiles(v) for v in monthly]
    lv4 = ladder_tables()[0]
    note = data_lines(
        900,
        "2025 flights at all ten airports. January, February, June and July, each held "
        f"out in turn and predicted from the other eleven months ({int(number(lv4['rows'])):,} "
        f"rows over {int(number(lv4['days']))} days). RMSE over every row, including those "
        "the record-quality term covers. The bars use all twelve held-out months.",
    )
    c = Canvas(
        900, 490 + 18 * len(note) + 14, "What each step changed, with its interval"
    )
    title(
        c,
        "Figure 3b. What each step changed, on the four months",
        "Change in RMSE from the step before, in seconds. Below zero lowers the error.",
    )
    top, step = 118, 50
    ax = Axis(-17, 2, 400, 740)
    for i in range(1, 6):
        c.text(20, top + (i - 1) * step + 4, f"{i + 1}. {STEPS[i]}", 13)
    horizontal_axis(
        c, ax, top + 4 * step + 34, -16, 0, 4, "Change in RMSE (s)", 0, top - 26
    )
    c.line(ax(0), top - 26, ax(0), top + 4 * step + 34, INK, 1.2)
    c.text(770, top - 32, "Change [95% interval]", 12.5, fill=MUTED)
    for i, ((e, lo, hi), (q1, q3)) in enumerate(zip(est, iqr, strict=True)):
        y = top + i * step
        col = BLUE if hi < 0 else RUST
        c.rect(ax(q1), y - 9, max(ax(q3) - ax(q1), 3), 18, BAND2, EDGE2)
        c.line(ax(lo), y, ax(hi), y, col, 2.2)
        c.dot(ax(e), y, 5, col)
        c.text(
            770,
            y + 4,
            f"{fmt(e, 2, True)} [{fmt(lo, 2, True)}, {fmt(hi, 2, True)}]",
            12.5,
            fill=col,
        )
    legend(
        c,
        20,
        436,
        [
            ("dot", BLUE, "estimate, four months"),
            ("whisker", BLUE, "95% interval over days"),
            ("bar2", BAND2, "middle half of the twelve monthly changes"),
        ],
    )
    last = monthly[-1]
    big = max(range(12), key=lambda j: abs(last[j]))
    c.text(
        20,
        464,
        "Blue: the interval excludes zero. Rust: it reaches zero. The window's "
        f"largest monthly change is {FULL[big]}, {fmt(last[big], 2)} s.",
        12.5,
        fill=MUTED,
    )
    data_note(c, note, 490)
    return c.svg()


def geometry_inputs():
    text = read("baseline.md")
    claim = row(
        tables_after(text, "### What each block buys where the pair is new")[0],
        "identifiers + time \u2192 identifiers + time + geometry (the claim)",
    )
    floors = tables_after(text, "## The noise floor")
    return claim, floors[0], row(floors[1], "`identifiers + time + geometry`")


def figure_geometry_forward() -> str:
    _, floor_table, cmp_row = geometry_inputs()
    rows = []
    for name, col, key in (
        ("December, all ten airports", "December, all ten", "December, all ten"),
        ("July, all ten airports", "July, all ten", "July, all ten"),
    ):
        rows.append(
            (name, -number(cmp_row[key]), number(row(floor_table, col)["floor (s)"]))
        )
    legs = table_headed(read("baseline.md"), "split")
    dec = dict(
        zip(legs[0], next(x for x in legs[1:] if x[0] == "`december`"), strict=True)
    )
    jul = dict(zip(legs[0], next(x for x in legs[1:] if x[0] == "`july`"), strict=True))
    note = data_lines(
        820,
        "2025 flights at all ten airports. December: trained on January to November "
        f"({int(number(dec['train rows'])):,} rows), scored on December "
        f"({int(number(dec['valid rows'])):,} rows). July: trained on January to June "
        f"({int(number(jul['train rows'])):,} rows), scored on July "
        f"({int(number(jul['valid rows'])):,} rows). RMSE over the rows the "
        "record-quality term leaves to the model, one seed.",
    )
    c = Canvas(
        820, 380 + 18 * len(note) + 14, "The geometry block on the two forward splits"
    )
    title(
        c,
        "Figure 4a. What the geometry block adds on the forward splits",
        "Gain in RMSE in seconds; positive lowers the error. Single-seed changes.",
    )
    ax = Axis(-1, 3, 330, 780)
    for k, (name, gain, floor) in enumerate(rows):
        y = 130 + k * 80
        c.text(20, y + 4, name, 13)
        c.text(20, y + 24, f"seed floor {fmt(floor, 2)} s", 12.5, fill=MUTED)
        c.rect(ax(-floor), y - 22, ax(floor) - ax(-floor), 44, BAND, GRID)
        col = BLUE if abs(gain) > floor else RUST
        c.dot(ax(gain), y, 6, col)
        c.text(ax(gain), y - 30, fmt(gain, 2, True), 13, "middle", col, "bold")
    c.line(ax(0), 96, ax(0), 276, INK, 1.2)
    horizontal_axis(c, ax, 286, -1, 3, 1, "Gain from the geometry block (s)", 0, 96)
    legend(c, 20, 344, [("dot", BLUE, "estimate"), ("band", BAND, "seed floor")])
    c.text(
        20,
        368,
        "No interval exists for a single-seed change; a change inside the band "
        "cannot be told from the random seed.",
        12.5,
        fill=MUTED,
    )
    data_note(c, note, 380)
    return c.svg()


def figure_geometry_heldout() -> str:
    claim, _, _ = geometry_inputs()
    seen = number(claim["on seen pairs (s)"])
    unseen = number(claim["on held-out pairs (s)"])
    est, lo, hi = interval(claim["difference [95%]"])
    year_row = row(
        tables_after(read("monthly_pairs.md"), "## Per month of 2025")[0], "the year"
    )
    held = int(number(year_row["held-out rows, 3 folds"]))
    note = data_lines(
        820,
        "2025 flights at all ten airports. Three folds, each holding out a random third "
        "of every airport's (stand, runway) pairs and scoring them against a sample of "
        f"rows whose pairs were seen ({held:,} held-out rows). RMSE over the rows the "
        "record-quality term leaves to the model.",
    )
    c = Canvas(820, 380 + 18 * len(note) + 14, "The geometry block on held-out pairs")
    title(
        c,
        "Figure 4b. What the geometry block adds on pairs the model has not seen",
        "Gain in RMSE in seconds from adding the block; positive lowers the error.",
    )
    ax = Axis(-2, 24, 300, 780)
    for name, v, y in (("On seen pairs", seen, 120), ("On held-out pairs", unseen, 170)):
        c.text(20, y + 4, name, 13)
        c.dot(ax(v), y, 5.5, MUTED)
        c.text(ax(v) + 12, y + 4, fmt(v, 1, True), 13, "start", MUTED, "bold")
    c.text(20, 234, "Difference", 13, weight="bold")
    c.text(20, 252, "(the claim)", 12.5, fill=MUTED)
    c.line(ax(lo), 230, ax(hi), 230, BLUE, 3)
    c.line(ax(lo), 223, ax(lo), 237, BLUE, 3)
    c.line(ax(hi), 223, ax(hi), 237, BLUE, 3)
    c.dot(ax(est), 230, 6.5, BLUE)
    c.text(
        ax(est),
        206,
        f"{fmt(est, 1, True)} [{fmt(lo, 1, True)}, {fmt(hi, 1, True)}]",
        13,
        "middle",
        BLUE,
        "bold",
    )
    c.line(ax(0), 96, ax(0), 270, INK, 1.2)
    horizontal_axis(c, ax, 280, 0, 24, 4, "Gain from the geometry block (s)", 0, 96)
    legend(
        c,
        20,
        338,
        [
            ("dot", MUTED, "mean gain on each kind of pair"),
            ("dot", BLUE, "difference"),
            ("whisker", BLUE, "95% interval over pairs"),
        ],
    )
    c.text(
        20,
        364,
        "The difference in differences is the block's gain on held-out pairs "
        "less its gain on seen pairs.",
        12.5,
        fill=MUTED,
    )
    data_note(c, note, 380)
    return c.svg()


def figure_months() -> str:
    text = read("monthly_pairs.md")
    table = tables_after(text, "## Per month of 2025")[0]
    key = "the claim [95%]"
    per = [interval(row(table, m)[key]) for m in MONTHS]
    year = interval(row(table, "the year")[key])
    q1, q3 = quartiles([e for e, _l, _h in per])
    held = int(number(row(table, "the year")["held-out rows, 3 folds"]))
    note = data_lines(
        900,
        "2025 flights at all ten airports. Three folds, each holding out a random third "
        "of every airport's (stand, runway) pairs and scoring them against a sample of "
        f"rows whose pairs were seen ({held:,} held-out rows over the year). The "
        "interval is clustered by pair.",
    )
    c = Canvas(
        900, 410 + 18 * len(note) + 14, "The held-out-pair difference, month by month"
    )
    title(
        c,
        "Figure 2. The geometry claim in each month of the held-out year",
        "Difference in differences in seconds; positive means the block helps more on "
        "unseen pairs than on seen ones.",
    )
    ax = Axis(0, 12, 70, 800)
    yax = Axis(26, 10, 100, 310)
    est, lo, hi = year
    c.rect(ax(0), yax(hi), ax(12) - ax(0), yax(lo) - yax(hi), BAND)
    c.line(ax(0), yax(est), ax(12), yax(est), BLUE, 1, "5 4")
    c.rect(826, yax(q3), 14, yax(q1) - yax(q3), BAND2, EDGE2)
    c.text(846, yax(q3) - 22, "middle half", 12.5, "middle", MUTED)
    c.text(846, yax(q3) - 8, "of months", 12.5, "middle", MUTED)
    for t in range(10, 27, 4):
        c.line(70, yax(t), 800, yax(t), GRID, 0.7)
        c.text(62, yax(t) + 4, str(t), 12.5, "end", MUTED)
    c.text(
        76,
        yax(11.2),
        f"The year {fmt(est, 1, True)} [{fmt(lo, 1, True)}, {fmt(hi, 1, True)}]",
        12.5,
        "start",
        BLUE,
        "bold",
    )
    for i, (m, (e, lw, hg)) in enumerate(zip(MONTHS, per, strict=True)):
        x = ax(i + 0.5)
        c.line(x, yax(lw), x, yax(hg), BLUE, 2)
        c.line(x - 4, yax(lw), x + 4, yax(lw), BLUE, 2)
        c.line(x - 4, yax(hg), x + 4, yax(hg), BLUE, 2)
        c.dot(x, yax(e), 4.8)
        c.text(x, yax(hg) - 8, fmt(e, 1), 12.5, "middle")
        c.text(x, 334, m, 13, "middle")
    c.line(70, 310, 800, 310, MUTED)
    legend(
        c,
        70,
        366,
        [
            ("dot", BLUE, "month's estimate"),
            ("whisker", BLUE, "95% interval over pairs"),
            ("band", BAND, "the year's interval"),
            ("bar2", BAND2, "middle half of the twelve months"),
        ],
    )
    c.text(
        70,
        394,
        f"The middle half of the twelve months runs {fmt(q1, 1)} to "
        f"{fmt(q3, 1)} s. Every interval excludes zero; the axis starts at 10 s.",
        12.5,
        fill=MUTED,
    )
    data_note(c, note, 410)
    return c.svg()


def figure_tail() -> str:
    text = read("residual.md")
    table = tables_after(text, "## The shape of the tail")[0]
    legs = (
        ("December, all ten airports", "over 600", "December, residual over 600 s"),
        ("December, all ten airports", "over 1,200", "December, residual over 1,200 s"),
        ("July, the three", "over 600", "July, three airports, over 600 s"),
        ("July, the three", "over 1,200", "July, three airports, over 1,200 s"),
    )
    found = []
    for leg, thr, label in legs:
        r = next(x for x in table[1:] if x[0] == leg and x[1] == thr)
        found.append((label, number(r[3]), number(r[4]), number(r[5]), number(r[2])))
    note = data_lines(
        900,
        "Validation residuals of the base configuration's rung, one seed, over the rows "
        "the record-quality term leaves to the model. December 2025: all ten airports. "
        "July 2025: Frankfurt, Heathrow and Schiphol, the three airports the July test "
        "set contains.",
    )
    c = Canvas(900, 410 + 18 * len(note) + 14, "The residual tail")
    title(
        c,
        "Figure 5. A thin tail of rows carries much of the squared error",
        "Rows whose residual exceeds the threshold: their share of all rows, and "
        "their share of the squared error (SSE).",
    )
    ax = Axis(0, 100, 300, 700)
    top, step = 96, 60
    for t in range(0, 101, 20):
        c.line(ax(t), top - 14, ax(t), top + 4 * step - 14, GRID, 0.7)
        c.text(ax(t), top + 4 * step + 6, f"{t}%", 12.5, "middle", MUTED)
    for i, (label, rows_pct, sse_pct, under, n) in enumerate(found):
        y = top + i * step
        c.text(20, y + 12, label, 13)
        c.rect(ax(0), y, ax(rows_pct) - ax(0), 15, MUTED)
        c.text(
            ax(rows_pct) + 6,
            y + 12,
            f"{fmt(rows_pct, 2)}% of rows ({int(n):,})",
            12.5,
            fill=MUTED,
        )
        c.rect(ax(0), y + 20, ax(sse_pct) - ax(0), 15, BLUE)
        c.text(
            ax(sse_pct) + 6,
            y + 32,
            f"{fmt(sse_pct, 1)}% of SSE",
            12.5,
            fill=BLUE,
            weight="bold",
        )
        c.text(880, y + 32, f"{int(under)}% under-predicted", 12.5, "end", MUTED)
    legend(c, 20, 368, [("bar", MUTED, "share of rows"), ("bar", BLUE, "share of SSE")])
    c.text(
        20,
        394,
        "The tail is one-sided: most of these rows are under-predicted.",
        12.5,
        fill=MUTED,
    )
    data_note(c, note, 410)
    return c.svg()


FIGURES = {
    "ladder_level.svg": figure_ladder_level,
    "ladder_change.svg": figure_ladder_change,
    "geometry_forward.svg": figure_geometry_forward,
    "geometry_heldout.svg": figure_geometry_heldout,
    "heldout_pairs_by_month.svg": figure_months,
    "residual_tail.svg": figure_tail,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare, write nothing")
    args = ap.parse_args()
    bad = 0
    for name, build in FIGURES.items():
        svg, path = build(), OUT / name
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == svg
            print(("ok  " if same else "DIFF"), path.relative_to(REPO))
            bad += not same
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(svg, encoding="utf-8")
            print("wrote", path.relative_to(REPO))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
