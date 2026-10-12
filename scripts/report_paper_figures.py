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
import math
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
OUT = REPO / "docs" / "figures"

# One visual language for every figure. Each color pair differs in lightness
# as well as hue, so the figures still read when printed in grayscale:
# estimates are dark blue and filled, estimates whose interval (or seed floor)
# reaches zero are rust and open, context marks are light gray.
INK, MUTED, GRID = "#1c1c1c", "#595959", "#e0e0e0"
BLUE, RUST, LIGHT = "#1f4f8a", "#b4532a", "#a3a3a3"
BAND, BAND_EDGE = "#dde6f1", "#a8bdd8"
BAND2, EDGE2 = "#b7cbe6", "#6f93c4"
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
    """A figure 800 px wide; its height is set by the footer, which comes last."""

    def __init__(self, title: str) -> None:
        self.w, self.h, self.title = 800, 0, title
        self.items: list[str] = []

    def text(
        self, x, y, s, size=13, anchor="start", fill=INK, weight="normal", rotate=False
    ):
        size = max(size, 12.5)
        turn = f' transform="rotate(-90 {x:.1f} {y:.1f})"' if rotate else ""
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"'
            f' fill="{fill}" font-weight="{weight}"{turn}>{esc(s)}</text>'
        )

    def lead_text(self, x, y, lead, rest, size, fill=INK):
        """One line whose first word is bold."""
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}">'
            f'<tspan font-weight="bold">{esc(lead)}</tspan>{esc(rest)}</text>'
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

    def dot(self, x, y, r=5, fill=BLUE, stroke="none", width=0):
        w = f' stroke-width="{width}"' if width else ""
        self.items.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}"{w}/>'
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


def wrap(text: str, size: float) -> list[str]:
    """Lines that fit between the margins, at about half an em a character."""
    return textwrap.wrap(text, int(752 / (size * 0.5)))


def paragraph(c: Canvas, y: float, text: str) -> float:
    """A muted note under the plot; returns the baseline of its last line."""
    lines = wrap(text, 13)
    for i, ln in enumerate(lines):
        c.text(24, y + 18 * i, ln, 13, fill=MUTED)
    return y + 18 * (len(lines) - 1)


def data_note(c: Canvas, text: str, y0: float) -> None:
    """The footer every figure ends on: a rule, then the sample, "Data:" in bold."""
    lines = wrap("Data: " + text, 12.5)
    c.line(24, y0, 776, y0, GRID, 1)
    for i, ln in enumerate(lines):
        y = y0 + 20 + 17 * i
        if i == 0:
            c.lead_text(24, y, "Data:", ln[len("Data:") :], 12.5)
        else:
            c.text(24, y, ln, 12.5)
    c.h = round(y0 + 20 + 17 * (len(lines) - 1) + 14)


def ticks(lo: float, hi: float, step: float) -> list[float]:
    out, v = [], lo
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


def estimate(c: Canvas, x: float, y: float, clear: bool) -> None:
    """Filled blue when the interval (or seed floor) clears zero, open rust if not."""
    if clear:
        c.dot(x, y, 5, BLUE)
    else:
        c.dot(x, y, 4.2, "#ffffff", RUST, 2)


def hwhisker(c: Canvas, x0: float, x1: float, y: float, color: str) -> None:
    c.line(x0, y, x1, y, color, 2)
    c.line(x0, y - 5, x0, y + 5, color, 2)
    c.line(x1, y - 5, x1, y + 5, color, 2)


def vwhisker(c: Canvas, x: float, y0: float, y1: float, color: str) -> None:
    c.line(x, y0, x, y1, color, 2)
    c.line(x - 5, y0, x + 5, y0, color, 2)
    c.line(x - 5, y1, x + 5, y1, color, 2)


def legend(c, x, y, items):
    """A row of mark keys: (kind, color, text)."""
    for kind, color, label in items:
        if kind == "dot":
            c.dot(x + 6, y - 4, 5, color)
            x += 18
        elif kind == "open":
            c.dot(x + 6, y - 4, 4.2, "#ffffff", RUST, 2)
            x += 18
        elif kind == "whisker":
            hwhisker(c, x, x + 26, y - 4, color)
            x += 36
        elif kind == "bar":
            c.rect(x, y - 10, 26, 12, color)
            x += 36
        elif kind == "bar2":
            c.rect(x, y - 10, 26, 12, BAND2, EDGE2)
            x += 36
        elif kind == "band":
            c.rect(x, y - 12, 26, 16, BAND, BAND_EDGE)
            x += 36
        c.text(x, y, label, 13, fill=MUTED)
        x += 6.6 * len(label) + 26


def horizontal_axis(c, ax, y, lo, hi, step, label, places=0, grid_to=None, unit=""):
    if grid_to is not None:
        for t in ticks(lo, hi, step):
            c.line(ax(t), grid_to, ax(t), y, GRID, 0.7)
    c.line(ax(lo), y, ax(hi), y, MUTED)
    for t in ticks(lo, hi, step):
        c.line(ax(t), y, ax(t), y + 4, MUTED)
        c.text(ax(t), y + 19, fmt(t, places) + unit, 13, "middle", MUTED)
    c.text((ax(lo) + ax(hi)) / 2, y + 39, label, 13, "middle", INK)


def column_head(c: Canvas, x: float, y: float, s: str, anchor: str = "start") -> None:
    c.text(x, y, s, 13, anchor, MUTED)


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
    c.text(24, 30, head, 16, weight="bold")
    c.text(24, 52, sub, 13, fill=MUTED)


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
    c = Canvas("The error on the four months, after each step")
    title(
        c,
        "Figure 3a. The error after each step of the final configuration",
        f"RMSE in seconds, pooled over the four months read ({n_rows:,} rows).",
    )
    top, step, col = 126, 40, 664
    ax = Axis(292, 308, 392, 632)
    axis_y = top + 5 * step + 28
    horizontal_axis(c, ax, axis_y, 292, 308, 4, "RMSE (s)", 0, top - 22)
    base = level[0]
    c.line(ax(base), top - 22, ax(base), axis_y, MUTED, 1.2, "5 4")
    c.text(ax(base) + 4, top - 30, f"the booster alone, {fmt(base)}", 13, "end", MUTED)
    column_head(c, col, top - 30, "RMSE (s)")
    for i, (name, v) in enumerate(zip(STEPS, level, strict=True)):
        y = top + i * step
        c.text(24, y + 5, f"{i + 1}. {name}", 14)
        c.dot(ax(v), y, 5, BLUE)
        c.text(col, y + 5, fmt(v), 14, fill=BLUE, weight="bold")
    end = paragraph(
        c,
        axis_y + 70,
        "Point estimates. Figure 3b gives what each step changed, with its interval.",
    )
    data_note(
        c,
        "2025 flights at all ten airports. January, February, June and July, each held "
        f"out in turn and predicted from the other eleven months ({n_rows:,} rows over "
        f"{int(number(lv['days']))} days). RMSE over every row, including those the "
        "record-quality term covers.",
        end + 22,
    )
    return c.svg()


def figure_ladder_change() -> str:
    _, per_month = ladder_tables()
    four = dict(per_month)["the four"]
    keys = list(four)[1:6]
    est = [interval(four[k]) for k in keys]
    monthly = [[interval(r[k])[0] for m, r in per_month if m in MONTHS] for k in keys]
    iqr = [quartiles(v) for v in monthly]
    lv4 = ladder_tables()[0]
    c = Canvas("What each step changed, with its interval")
    title(
        c,
        "Figure 3b. What each step changed, on the four months",
        "Change in RMSE from the step before, in seconds. Below zero lowers the error.",
    )
    top, step, col = 126, 46, 608
    ax = Axis(-17, 2, 384, 592)
    axis_y = top + 4 * step + 30
    horizontal_axis(c, ax, axis_y, -16, 0, 4, "Change in RMSE (s)", 0, top - 22)
    column_head(c, col, top - 30, "Change [95% interval]")
    for i, (q1, q3) in enumerate(iqr):
        y = top + i * step
        c.text(24, y + 5, f"{i + 2}. {STEPS[i + 1]}", 14)
        c.rect(ax(q1), y - 8, max(ax(q3) - ax(q1), 3), 16, BAND2, EDGE2)
    c.line(ax(0), top - 22, ax(0), axis_y, INK, 1.2)
    for i, (e, lo, hi) in enumerate(est):
        y = top + i * step
        clear = hi < 0 or lo > 0
        color = BLUE if clear else RUST
        hwhisker(c, ax(lo), ax(hi), y, color)
        estimate(c, ax(e), y, clear)
        c.text(
            col,
            y + 5,
            f"{fmt(e, 2, True)} [{fmt(lo, 2, True)}, {fmt(hi, 2, True)}]",
            13,
            fill=color,
        )
    legend(
        c,
        24,
        axis_y + 72,
        [
            ("dot", BLUE, "estimate, four months: interval excludes zero"),
            ("open", RUST, "interval reaches zero"),
        ],
    )
    legend(
        c,
        24,
        axis_y + 98,
        [
            ("whisker", BLUE, "95% interval over days"),
            ("bar2", BAND2, "middle half of the twelve monthly changes"),
        ],
    )
    last = monthly[-1]
    big = max(range(12), key=lambda j: abs(last[j]))
    end = paragraph(
        c,
        axis_y + 128,
        f"The window's largest monthly change is {FULL[big]}, {fmt(last[big], 2)} s.",
    )
    data_note(
        c,
        "2025 flights at all ten airports. January, February, June and July, each held "
        f"out in turn and predicted from the other eleven months ({int(number(lv4['rows'])):,} "
        f"rows over {int(number(lv4['days']))} days). RMSE over every row, including those "
        "the record-quality term covers. The bars use all twelve held-out months.",
        end + 22,
    )
    return c.svg()


def geometry_inputs():
    text = read("baseline.md")
    claim = row(
        tables_after(text, "### What each block buys where the pair is new")[0],
        "identifiers + time → identifiers + time + geometry (the claim)",
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
    c = Canvas("The geometry block on the two forward splits")
    title(
        c,
        "Figure 4a. What the geometry block adds on the forward splits",
        "Gain in RMSE in seconds; positive lowers the error. Single-seed changes.",
    )
    top, step, col = 132, 72, 648
    ax = Axis(-1, 3, 300, 620)
    axis_y = top + step + 44
    horizontal_axis(
        c, ax, axis_y, -1, 3, 1, "Gain from the geometry block (s)", 0, top - 30
    )
    column_head(c, col, top - 38, "Gain (s)")
    for k, (name, _gain, floor) in enumerate(rows):
        y = top + k * step
        c.text(24, y - 2, name, 14)
        c.text(24, y + 17, f"seed floor {fmt(floor, 2)} s", 13, fill=MUTED)
        c.rect(ax(-floor), y - 18, ax(floor) - ax(-floor), 36, BAND, BAND_EDGE)
    c.line(ax(0), top - 30, ax(0), axis_y, INK, 1.2)
    for k, (_name, gain, floor) in enumerate(rows):
        y = top + k * step
        clear = abs(gain) > floor
        estimate(c, ax(gain), y, clear)
        c.text(
            col,
            y + 5,
            fmt(gain, 2, True),
            14,
            fill=BLUE if clear else RUST,
            weight="bold",
        )
    legend(
        c,
        24,
        axis_y + 72,
        [
            ("dot", BLUE, "estimate outside the seed floor"),
            ("open", RUST, "inside the seed floor"),
            ("band", BAND, "seed floor"),
        ],
    )
    end = paragraph(
        c,
        axis_y + 100,
        "No interval exists for a single-seed change; a change inside the band "
        "cannot be told from the random seed.",
    )
    data_note(
        c,
        "2025 flights at all ten airports. December: trained on January to November "
        f"({int(number(dec['train rows'])):,} rows), scored on December "
        f"({int(number(dec['valid rows'])):,} rows). July: trained on January to June "
        f"({int(number(jul['train rows'])):,} rows), scored on July "
        f"({int(number(jul['valid rows'])):,} rows). RMSE over the rows the "
        "record-quality term leaves to the model, one seed.",
        end + 22,
    )
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
    c = Canvas("The geometry block on held-out pairs")
    title(
        c,
        "Figure 4b. What the geometry block adds on pairs the model has not seen",
        "Gain in RMSE in seconds from adding the block; positive lowers the error.",
    )
    top, col = 128, 600
    ax = Axis(-2, 24, 230, 576)
    axis_y = top + 136
    horizontal_axis(
        c, ax, axis_y, 0, 24, 4, "Gain from the geometry block (s)", 0, top - 30
    )
    column_head(c, col, top - 38, "Gain (s)")
    c.line(ax(0), top - 30, ax(0), axis_y, INK, 1.2)
    for name, v, y in (
        ("On seen pairs", seen, top),
        ("On held-out pairs", unseen, top + 44),
    ):
        c.text(24, y + 5, name, 14)
        c.dot(ax(v), y, 5, LIGHT)
        c.text(col, y + 5, fmt(v, 1, True), 14, fill=MUTED, weight="bold")
    y = top + 98
    c.text(24, y - 1, "Difference", 14, weight="bold")
    c.text(24, y + 17, "(the claim)", 13, fill=MUTED)
    clear = hi < 0 or lo > 0
    color = BLUE if clear else RUST
    hwhisker(c, ax(lo), ax(hi), y, color)
    estimate(c, ax(est), y, clear)
    c.text(
        col,
        y + 5,
        f"{fmt(est, 1, True)} [{fmt(lo, 1, True)}, {fmt(hi, 1, True)}]",
        14,
        fill=color,
        weight="bold",
    )
    legend(
        c,
        24,
        axis_y + 72,
        [
            ("dot", LIGHT, "mean gain on each kind of pair"),
            ("dot", BLUE, "difference"),
            ("whisker", BLUE, "95% interval over pairs"),
        ],
    )
    end = paragraph(
        c,
        axis_y + 100,
        "The difference in differences is the block's gain on held-out pairs "
        "less its gain on seen pairs.",
    )
    data_note(
        c,
        "2025 flights at all ten airports. Three folds, each holding out a random third "
        "of every airport's (stand, runway) pairs and scoring them against a sample of "
        f"rows whose pairs were seen ({held:,} held-out rows). RMSE over the rows the "
        "record-quality term leaves to the model.",
        end + 22,
    )
    return c.svg()


def figure_months() -> str:
    text = read("monthly_pairs.md")
    table = tables_after(text, "## Per month of 2025")[0]
    key = "the claim [95%]"
    per = [interval(row(table, m)[key]) for m in MONTHS]
    year = interval(row(table, "the year")[key])
    q1, q3 = quartiles([e for e, _l, _h in per])
    held = int(number(row(table, "the year")["held-out rows, 3 folds"]))
    c = Canvas("The held-out-pair difference, month by month")
    title(
        c,
        "Figure 2. The geometry claim in each month of the held-out year",
        "Difference in differences in seconds; positive means the block helps more on "
        "unseen pairs than on seen ones.",
    )
    ax = Axis(0, 12, 72, 676)
    yax = Axis(26, 10, 116, 316)
    est, lo, hi = year
    c.text(
        ax(0),
        92,
        f"The year {fmt(est, 1, True)} [{fmt(lo, 1, True)}, {fmt(hi, 1, True)}]",
        14,
        "start",
        BLUE,
        "bold",
    )
    c.text(
        34,
        (yax(26) + yax(10)) / 2,
        "Difference in differences (s)",
        13,
        "middle",
        INK,
        rotate=True,
    )
    for t in range(10, 27, 4):
        c.line(ax(0), yax(t), ax(12), yax(t), GRID, 0.7)
        c.text(64, yax(t) + 4, str(t), 13, "end", MUTED)
    c.rect(ax(0), yax(hi), ax(12) - ax(0), yax(lo) - yax(hi), BAND, BAND_EDGE)
    c.line(ax(0), yax(est), ax(12), yax(est), MUTED, 1.2, "5 4")
    c.line(ax(0), yax(10), ax(12), yax(10), MUTED)
    mid = (yax(q3) + yax(q1)) / 2
    c.rect(690, yax(q3), 12, yax(q1) - yax(q3), BAND2, EDGE2)
    for k, word in enumerate(("middle", "half of", "months")):
        c.text(710, mid - 12 + 16 * k, word, 13, fill=MUTED)
    for i, (m, (e, lw, hg)) in enumerate(zip(MONTHS, per, strict=True)):
        x = ax(i + 0.5)
        clear = hg < 0 or lw > 0
        color = BLUE if clear else RUST
        vwhisker(c, x, yax(lw), yax(hg), color)
        estimate(c, x, yax(e), clear)
        c.text(x, 338, m, 13, "middle")
        c.text(x, 356, fmt(e, 1), 13, "middle", color)
    legend(
        c,
        ax(0),
        392,
        [
            ("dot", BLUE, "month's estimate (value under the month)"),
            ("whisker", BLUE, "95% interval over pairs"),
        ],
    )
    legend(
        c,
        ax(0),
        418,
        [
            ("band", BAND, "the year's interval"),
            ("bar2", BAND2, "middle half of the twelve months"),
        ],
    )
    end = paragraph(
        c,
        446,
        f"The middle half of the twelve months runs {fmt(q1, 1)} to "
        f"{fmt(q3, 1)} s. Every interval excludes zero; the axis starts at 10 s.",
    )
    data_note(
        c,
        "2025 flights at all ten airports. Three folds, each holding out a random third "
        "of every airport's (stand, runway) pairs and scoring them against a sample of "
        f"rows whose pairs were seen ({held:,} held-out rows over the year). The "
        "interval is clustered by pair.",
        end + 22,
    )
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
    c = Canvas("The residual tail")
    title(
        c,
        "Figure 5. A thin tail of rows carries much of the squared error",
        "Rows whose residual exceeds the threshold: their share of all rows, and "
        "their share of the squared error (SSE).",
    )
    end_pct = 20 * math.ceil(max(f[2] for f in found) / 20)
    ax = Axis(0, end_pct, 272, 472)
    top, step = 124, 56
    rows_col, sse_col = 594, 668
    axis_y = top + 4 * step - 12
    horizontal_axis(c, ax, axis_y, 0, end_pct, 20, "Share (%)", 0, top - 16, "%")
    column_head(c, rows_col, top - 30, "% of rows (n)", "end")
    column_head(c, sse_col, top - 30, "% of SSE", "end")
    column_head(c, 776, top - 30, "under-predicted", "end")
    for i, (label, rows_pct, sse_pct, under, n) in enumerate(found):
        y = top + i * step
        c.text(24, y + 19, label, 14)
        c.rect(ax(0), y, ax(rows_pct) - ax(0), 12, LIGHT)
        c.rect(ax(0), y + 16, ax(sse_pct) - ax(0), 12, BLUE)
        c.text(rows_col, y + 19, f"{fmt(rows_pct, 2)}% ({int(n):,})", 13, "end", MUTED)
        c.text(sse_col, y + 19, f"{fmt(sse_pct, 1)}%", 13, "end", BLUE, "bold")
        c.text(776, y + 19, f"{int(under)}%", 13, "end", MUTED)
    legend(
        c,
        24,
        axis_y + 72,
        [("bar", LIGHT, "share of rows"), ("bar", BLUE, "share of SSE")],
    )
    end = paragraph(
        c,
        axis_y + 100,
        "The tail is one-sided: most of these rows are under-predicted.",
    )
    data_note(
        c,
        "Validation residuals of the base configuration's rung, one seed, over the rows "
        "the record-quality term leaves to the model. December 2025: all ten airports. "
        "July 2025: Frankfurt, Heathrow and Schiphol, the three airports the July test "
        "set contains.",
        end + 22,
    )
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
