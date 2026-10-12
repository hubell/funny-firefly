# SPDX-License-Identifier: GPL-3.0-only
"""The paper's figures are drawn from the committed reports and nothing else.

These tests read `reports/` and `docs/figures/` only, so they need neither the
challenge data nor the network.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import report_paper_figures as rpf  # noqa: E402


def test_number_and_interval_parse_a_report_cell():
    assert rpf.number("**+18.5 s**") == 18.5
    assert rpf.number("−0.69") == -0.69
    assert rpf.number("2,950") == 2950.0
    assert rpf.interval("+18.5 s [+16.7, +20.5]") == (18.5, 16.7, 20.5)
    assert rpf.interval("-0.69 [-1.35, -0.05]") == (-0.69, -1.35, -0.05)


def test_quartiles_interpolate_between_order_statistics():
    assert rpf.quartiles([1.0, 2.0, 3.0, 4.0, 5.0]) == (2.0, 4.0)
    assert rpf.quartiles([0.0, 10.0]) == (2.5, 7.5)


def test_tables_after_reads_the_table_under_its_heading():
    text = "## A\n\n| x | y |\n| --- | ---: |\n| a | 1 |\n\n## B\n\n| z |\n| - |\n| q |\n"
    table = rpf.tables_after(text, "## A")[0]
    assert table == [["x", "y"], ["a", "1"]]
    assert rpf.row(table, "a") == {"x": "a", "y": "1"}


def test_every_figure_is_well_formed_and_matches_the_committed_file():
    for name, build in rpf.FIGURES.items():
        svg = build()
        ET.fromstring(svg)
        assert (rpf.OUT / name).read_text(encoding="utf-8") == svg, name


def test_the_plotted_estimates_are_the_reports_estimates():
    """The headline value of each figure is in the report it is drawn from."""
    v12 = (REPO / "reports" / "v12-ladder.md").read_text(encoding="utf-8")
    lv, _ = rpf.ladder_tables()
    assert [rpf.fmt(rpf.number(lv[k])) for k in list(lv)[4:10]] == [
        "306.0",
        "305.3",
        "305.3",
        "299.6",
        "299.3",
        "294.4",
    ]
    assert "306.0" in v12 and "294.4" in v12
    claim, _, cmp_row = rpf.geometry_inputs()
    assert rpf.interval(claim["difference [95%]"]) == (18.5, 16.7, 20.5)
    assert rpf.number(cmp_row["December, all ten"]) == -2.27


def test_the_drawing_script_holds_no_result_literal():
    """A figure's numbers come from a report: no plotted value is typed in."""
    src = (REPO / "scripts" / "report_paper_figures.py").read_text(encoding="utf-8")
    for literal in ("18.5", "16.7", "20.5", "2.27", "294.4", "306.0", "17.23"):
        assert literal not in src, literal
    dashes = "[" + chr(0x2013) + chr(0x2014) + "]"
    assert not re.search(dashes, src)
