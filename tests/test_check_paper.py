# SPDX-License-Identifier: GPL-3.0-only
"""The paper's own checker, on constructed papers.

Every test but the last two builds a small repository under ``tmp_path``, a
paper and the reports it cites, and points ``scripts/check_paper.py`` at it,
so each check is seen failing on the case it exists for rather than trusted
because the real paper happens to trip it. The last two read the real tree:
the checker must hold no constant and no prose figure of its own, and
``--strict`` must pass on the paper as it stands, since the figure pass of
25 September, and the checks below are what that pass means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import check_paper as chk  # noqa: E402
import check_provenance as cp  # noqa: E402
from taxiout.provenance import Frozen, PaperFigure, PaperSection  # noqa: E402

LIVE = """# A live report

## Held-out pairs

The claim contrast is +14.72 s [-2.9, +31.0], over 215,876 rows.
The share is 53.2% of the test rows, and the step is 12.4 s.

### Per airport

The largest is +31.6 s.

## The ceiling: what any queue model could still add

The ceiling reads 9.46 s, and the speed 0.25 m/s.

## Every step, per airport

| step | December |
| --- | ---: |
| one | -5.26 |
| two | 8.81 |

## The ceiling, again

Repeated: 9.46 s once more, and 7.7 s.
"""

FROZEN = """# A dated record

The seed rule read 0.52 against a floor of 0.23 on the day.
"""

DRAFTING = "Drafting notes, not part of the paper"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with one live report, one frozen record and a README."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "reports" / "live.md").write_text(LIVE, encoding="utf-8")
    (tmp_path / "reports" / "frozen-2026-01-01.md").write_text(FROZEN, encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Readme\n\n## Reproducing\n\nRun it.\n", "utf-8"
    )
    monkeypatch.setattr(chk, "REPO", tmp_path)
    monkeypatch.setattr(
        chk,
        "FROZEN_DOCUMENTS",
        (Frozen("reports/frozen-2026-01-01.md", "0" * 64, "a dated record"),),
    )
    monkeypatch.setattr(chk, "PAPER_NO_SOURCE_SECTIONS", ())
    monkeypatch.setattr(
        chk, "PAPER_EXCLUDED_SECTIONS", (PaperSection(DRAFTING, "not the paper"),)
    )
    monkeypatch.setattr(chk, "PAPER_HISTORICAL_FIGURES", ())
    monkeypatch.setattr(chk, "PAPER_FIGURE_EXEMPTIONS", ())
    return tmp_path


def paper(repo: Path, body: str, drafting: bool = True) -> None:
    """Write a paper, with the drafting notes the fixture excludes by default."""
    notes = f"\n## {DRAFTING}\n\n- The old gain was 99.9 s.\n" if drafting else ""
    (repo / "docs" / "paper.md").write_text(f"# A paper\n\n{body}{notes}", "utf-8")


def statuses(result) -> dict[tuple[str, str], str]:
    return {
        (r.section.key, v.number.written): v.status
        for r in result.sections
        for v in r.verdicts
    }


def outcome(result, number: int, strict: bool = True):
    return chk.run(result, strict)[number - 1]


# --------------------------------------------------------------------------
# the match rule


def test_a_figure_present_in_its_cited_source_is_matched(repo):
    paper(repo, "## 1. One\n\nThe gain is +14.72 s.\n\nSource: `reports/live.md`.\n")
    result = chk.check()
    assert statuses(result) == {("§1", "+14.72"): chk.MATCHED}
    [v] = result.sections[1].verdicts
    assert (v.source, v.match.line, v.match.written) == ("reports/live.md", 5, "+14.72")
    assert outcome(result, 1).ok
    assert chk.main(["--strict"]) == 0


def test_a_figure_absent_from_its_cited_source_is_unmatched_and_fails_strict(
    repo, capsys
):
    paper(repo, "## 1. One\n\nThe gain is 15.1 s.\n\nSource: `reports/live.md`.\n")
    result = chk.check()
    assert statuses(result) == {("§1", "15.1"): chk.UNMATCHED}
    assert outcome(result, 1, strict=False).ok, "outside --strict it is a finding"
    assert not outcome(result, 1).ok
    assert chk.main([]) == 0
    assert chk.main(["--strict"]) == 1
    assert "15.1 is in no cited source" in capsys.readouterr().out


def test_a_figure_matches_at_its_own_precision_and_no_finer(repo):
    """The source's 14.72 is 14.7 at one decimal; it is not 14.8, nor 14.720.

    A figure written with more decimals than the source carries claims a
    precision the source cannot confirm, so it does not match.
    """
    paper(
        repo,
        "## 1. One\n\nRead as 14.7, as 14.8 and as 14.720.\n\n"
        "Source: `reports/live.md`.\n",
    )
    assert statuses(chk.check()) == {
        ("§1", "14.7"): chk.MATCHED,
        ("§1", "14.8"): chk.UNMATCHED,
        ("§1", "14.720"): chk.UNMATCHED,
    }


def test_a_half_rounds_either_way_and_the_report_prints_what_matched(repo):
    """0.25 is 0.3 half up and 0.2 half to even; the rule accepts both."""
    paper(repo, "## 1. One\n\nIt is 0.3, or 0.2.\n\nSource: `reports/live.md`.\n")
    result = chk.check()
    assert set(statuses(result).values()) == {chk.MATCHED}
    assert {v.match.written for v in result.sections[1].verdicts} == {"0.25"}


def test_a_percentage_matches_only_a_percentage(repo):
    """53.2% rounds to 53%; 12.4 s is not 12%, however it rounds."""
    paper(repo, "## 1. One\n\nThat is 53% and 12%.\n\nSource: `reports/live.md`.\n")
    assert statuses(chk.check()) == {
        ("§1", "53%"): chk.MATCHED,
        ("§1", "12%"): chk.UNMATCHED,
    }


def test_a_written_sign_must_agree_and_a_bare_figure_is_a_magnitude(repo):
    """The paper's U+2212 minus matches a report's hyphen, and only a minus.

    A bare figure is a magnitude, the way the paper says a block is worth so
    many seconds where the report prints the step as a negative change; a plus
    claims a number that is not negative.
    """
    paper(
        repo,
        "## 1. One\n\nFrom −2.9 to +31.0; a step of 2.9, not +2.9 nor −31.0.\n\n"
        "Source: `reports/live.md`.\n",
    )
    assert statuses(chk.check()) == {
        ("§1", "−2.9"): chk.MATCHED,
        ("§1", "+31.0"): chk.MATCHED,
        ("§1", "2.9"): chk.MATCHED,
        ("§1", "+2.9"): chk.UNMATCHED,
        ("§1", "−31.0"): chk.UNMATCHED,
    }


def test_an_integer_needs_four_digits_and_a_year_is_not_a_figure(repo):
    paper(
        repo,
        "## 1. One\n\nOf 107 rows, 215,876 in all, over 1000 draws, in 2025.\n\n"
        "Source: `reports/live.md`.\n",
    )
    result = chk.check()
    assert statuses(result) == {
        ("§1", "215,876"): chk.MATCHED,
        ("§1", "1000"): chk.UNMATCHED,
    }
    assert [(n.figure, p) for n, p in result.sections[1].by_pattern] == [("2025", "year")]


def test_section_numbers_headings_and_versions_are_exempt_by_pattern(repo):
    paper(
        repo,
        "### 7.1 One\n\nAs §7.2 says, under ODbL 1.0 and Python 3.13.5.\n\n"
        "Source: `reports/live.md`.\n",
    )
    result = chk.check()
    assert statuses(result) == {}
    assert [(n.figure, p) for n, p in result.sections[1].by_pattern] == [
        ("7.1", "heading number"),
        ("7.2", "section reference"),
        ("1.0", "version string"),
    ]


# --------------------------------------------------------------------------
# the frozen rule


def test_a_figure_found_only_in_a_frozen_record_fails_unless_listed(repo, monkeypatch):
    """The case the rule exists for: a seed rule quoted from a dated record.

    The record will carry 0.52 and 0.23 forever, so a check that accepted a
    match there would certify the figures long after the model moved.
    """
    paper(
        repo,
        "## 8. Results\n\nThe seed rule cleared, 0.52 against 0.23, and the claim"
        " is +14.72.\n\nSource: `reports/live.md`; `reports/frozen-2026-01-01.md`.\n",
    )
    result = chk.check()
    assert statuses(result) == {
        ("§8", "0.52"): chk.HISTORICAL,
        ("§8", "0.23"): chk.HISTORICAL,
        ("§8", "+14.72"): chk.MATCHED,
    }
    assert outcome(result, 2, strict=False).ok
    assert not outcome(result, 2).ok
    assert (
        "matches only in reports/frozen-2026-01-01.md" in outcome(result, 2).findings[0]
    )

    monkeypatch.setattr(
        chk,
        "PAPER_HISTORICAL_FIGURES",
        (
            PaperFigure("§8", "0.52", "quoted as the reading of its day"),
            PaperFigure("§8", "0.23", "quoted as the floor of its day"),
        ),
    )
    result = chk.check()
    assert all(
        v.listed for v in result.sections[1].verdicts if v.status == chk.HISTORICAL
    )
    assert outcome(result, 2).ok
    assert chk.main(["--strict"]) == 0


def test_a_live_match_outranks_a_frozen_one(repo):
    """A figure the live report still carries is matched, whatever the record says."""
    (repo / "reports" / "frozen-2026-01-01.md").write_text(FROZEN + "\nAnd +14.72.\n")
    paper(
        repo,
        "## 1. One\n\nIt is +14.72.\n\n"
        "Source: `reports/frozen-2026-01-01.md`; `reports/live.md`.\n",
    )
    [v] = chk.check().sections[1].verdicts
    assert (v.status, v.source) == (chk.MATCHED, "reports/live.md")


def test_a_register_entry_nothing_reaches_fails_in_every_mode(repo, monkeypatch):
    paper(repo, "## 1. One\n\nIt is +14.72.\n\nSource: `reports/live.md`.\n")
    monkeypatch.setattr(
        chk, "PAPER_HISTORICAL_FIGURES", (PaperFigure("§1", "9.9", "an old reading"),)
    )
    monkeypatch.setattr(
        chk, "PAPER_FIGURE_EXEMPTIONS", (PaperFigure("§1", "8.8", "a minus b"),)
    )
    result = chk.check()
    assert not outcome(result, 2, strict=False).ok
    assert not outcome(result, 3, strict=False).ok
    assert chk.main([]) == 1


def test_a_computed_figure_is_exempt_only_with_its_arithmetic_verified(repo, monkeypatch):
    """The arithmetic is checked, not trusted: operands in scope, value rounding."""
    paper(
        repo,
        "## 1. One\n\nThe two differ by 5.26 s.\n\n"
        "Source: `reports/live.md`, *Held-out pairs* and *The ceiling*.\n",
    )
    assert statuses(chk.check()) == {("§1", "5.26"): chk.UNMATCHED}

    def listed(arithmetic: str, operands: tuple[str, ...], figure: str = "5.26"):
        monkeypatch.setattr(
            chk,
            "PAPER_FIGURE_EXEMPTIONS",
            (PaperFigure("§1", figure, "a minus b", arithmetic, operands),),
        )
        result = chk.check()
        return result, chk.run(result, strict=True)[2]

    result, third = listed("14.72 - 9.46", ("14.72", "9.46"))
    [v] = result.sections[1].verdicts
    assert (v.status, v.reason) == (chk.EXEMPT, "a minus b")
    assert v.arithmetic.startswith("= 14.72 - 9.46 = 5.26; operands 14.72 at")
    assert "`reports/live.md`:5 `+14.72`" in v.arithmetic
    assert "`reports/live.md`:14 `9.46`" in v.arithmetic
    assert third.ok and all(o.ok for o in chk.run(result, strict=True))

    # each failure is a broken entry in both modes, and the figure is then read
    # as if it were not listed
    for arithmetic, operands, why in (
        ("", ("14.72", "9.46"), "carries no arithmetic"),
        ("14.72 - 9.4", ("14.72", "9.4"), "operand 9.4 is in no cited source"),
        ("14.72 - 9.46", ("14.72", "0.25"), "operand 0.25 is not in '14.72 - 9.46'"),
        ("14.72 - 9.46", (), "names no operand"),
        ("14.72 - 9.40", ("14.72",), "does not round to 5.26"),
        ("abs(14.72 - 9.46)", ("14.72", "9.46"), "is not arithmetic"),
    ):
        result, third = listed(arithmetic, operands)
        assert not third.ok, arithmetic
        assert any(why in b for b in third.broken), (arithmetic, third.broken)
        assert statuses(result) == {("§1", "5.26"): chk.UNMATCHED}
    assert chk.main([]) == 1, "a broken entry fails outside --strict too"

    # a figure written without a sign is a magnitude, so a negative value
    # verifies it; one written with a sign has to agree
    result, third = listed("9.46 - 14.72", ("14.72", "9.46"))
    assert third.ok and statuses(result) == {("§1", "5.26"): chk.EXEMPT}
    paper(
        repo,
        "## 1. One\n\nThe two differ by +5.26 s.\n\n"
        "Source: `reports/live.md`, *Held-out pairs* and *The ceiling*.\n",
    )
    result, third = listed("9.46 - 14.72", ("14.72", "9.46"), "+5.26")
    assert not third.ok
    assert any("the figure is written +5.26" in b for b in third.broken)
    assert statuses(result) == {("§1", "+5.26"): chk.UNMATCHED}


def test_a_constant_of_the_arithmetic_is_printed_and_not_looked_for(repo, monkeypatch):
    """A number in the expression that is no operand is the entry's own."""
    paper(repo, "## 1. One\n\nAbout 10.4 s.\n\nSource: `reports/live.md`.\n")
    monkeypatch.setattr(
        chk,
        "PAPER_FIGURE_EXEMPTIONS",
        (
            PaperFigure(
                "§1", "10.4", "a times root two", "14.72 / 2 ** (1 / 2)", ("14.72",)
            ),
        ),
    )
    result = chk.check()
    [v] = result.sections[1].verdicts
    assert v.status == chk.EXEMPT
    assert v.arithmetic.endswith("constants of the arithmetic: 1, 2")
    assert all(o.ok for o in chk.run(result, strict=True))


def test_a_verified_exemption_outranks_a_chance_match_and_both_are_printed(
    repo, monkeypatch
):
    """The case the precedence exists for: 5.26 is also a step in a cited table."""
    paper(
        repo,
        "## 1. One\n\nThe two differ by 5.26 s, and the step is −5.26.\n\n"
        "Source: `reports/live.md`.\n",
    )
    result = chk.check()
    assert statuses(result) == {
        ("§1", "5.26"): chk.MATCHED,
        ("§1", "−5.26"): chk.MATCHED,
    }, "read whole, the file carries -5.26, and a magnitude matches it"
    monkeypatch.setattr(
        chk,
        "PAPER_FIGURE_EXEMPTIONS",
        (PaperFigure("§1", "5.26", "a minus b", "14.72 - 9.46", ("14.72", "9.46")),),
    )
    result = chk.check()
    [computed, quoted] = result.sections[1].verdicts
    assert (computed.status, computed.source, computed.match.written) == (
        chk.EXEMPT,
        "reports/live.md",
        "-5.26",
    )
    assert quoted.status == chk.MATCHED, "the key carries the sign: −5.26 is not 5.26"
    assert all(o.ok for o in chk.run(result, strict=True))
    rows = [
        line
        for line in chk.report(result, chk.run(result, True), True)
        if "| 5.26 |" in line
    ]
    assert "also `reports/live.md`:20 `-5.26`, by chance" in rows[0]


# --------------------------------------------------------------------------
# sections, sources and headings


def test_a_section_with_figures_and_no_source_fails_unless_listed(repo, monkeypatch):
    """Listed, it is read against every live report, and never a frozen one."""
    paper(repo, "## Abstract [settled]\n\nA claim of +14.72 s, and of 0.52.\n")
    result = chk.check()
    assert statuses(result) == {
        ("Abstract", "+14.72"): chk.UNMATCHED,
        ("Abstract", "0.52"): chk.UNMATCHED,
    }
    assert outcome(result, 4, strict=False).ok
    assert not outcome(result, 4).ok
    assert outcome(result, 4).findings == [
        "Abstract carries 2 figure(s) and no source line"
    ]

    monkeypatch.setattr(
        chk, "PAPER_NO_SOURCE_SECTIONS", (PaperSection("Abstract", "it summarizes"),)
    )
    result = chk.check()
    assert result.sections[1].scope == chk.EVERY_LIVE_REPORT
    assert statuses(result) == {
        ("Abstract", "+14.72"): chk.MATCHED,
        ("Abstract", "0.52"): chk.UNMATCHED,
    }
    assert outcome(result, 4).ok


def test_a_section_without_figures_needs_no_source_line(repo):
    """What check 4 asks about is figures; a heading's own number is not one."""
    paper(repo, "### 9.2 Two\n\nIt touches 107 rows.\n")
    result = chk.check()
    assert statuses(result) == {}
    assert outcome(result, 4).ok


def test_a_cited_heading_that_does_not_exist_fails(repo):
    """A name resolves when a heading is it, or begins with it at a word boundary."""
    paper(
        repo,
        "## 1. One\n\nIt is +14.72.\n\nSources: `reports/live.md`, *Held-out pairs*,"
        " *The\nceiling* and *The regime shift*; README, *Reproducing*.\n",
    )
    result = chk.check()
    unresolved = [n for c in result.sections[1].citations for n in c.unresolved]
    assert unresolved == ["The regime shift"]
    assert not outcome(result, 5).ok
    assert outcome(result, 5).findings == [
        "§1: 'The regime shift' is not a heading in reports/live.md"
    ]
    assert outcome(result, 5, strict=False).ok


def test_a_cited_section_name_confines_the_search_to_that_section(repo):
    """The case the confinement exists for: 9.46 is in *The ceiling*, not here."""
    paper(
        repo,
        "## 1. One\n\nIt is +14.72, and the ceiling 9.46.\n\n"
        "Source: `reports/live.md`, *Held-out pairs*.\n",
    )
    result = chk.check()
    assert statuses(result) == {
        ("§1", "+14.72"): chk.MATCHED,
        ("§1", "9.46"): chk.UNMATCHED,
    }
    [_hit, miss] = result.sections[1].verdicts
    assert miss.outside == ("`reports/live.md`:14 `9.46`",)
    assert miss.elsewhere == ()
    assert "outside the cited sections" in outcome(result, 1).findings[0]
    [cite] = result.sections[1].citations
    assert cite.spans == (("Held-out pairs", 3, 11),), "the subsection is inside"

    paper(
        repo,
        "## 1. One\n\nIt is +14.72, and the ceiling 9.46.\n\n"
        "Source: `reports/live.md`, *Held-out pairs* and *The ceiling*.\n",
    )
    result = chk.check()
    assert set(statuses(result).values()) == {chk.MATCHED}
    [cite] = result.sections[1].citations
    assert cite.spans == (
        ("Held-out pairs", 3, 11),
        ("The ceiling", 12, 15),
        ("The ceiling", 23, 25),
    ), "a name that is a heading twice reads both, by its short form"


def test_a_subsection_is_inside_its_section_and_a_file_cited_whole_is_read_whole(
    repo,
):
    paper(
        repo,
        "## 1. One\n\nThe largest is +31.6 s, and a step 8.81.\n\n"
        "Source: `reports/live.md`, *Held-out pairs*.\n",
    )
    assert statuses(chk.check()) == {
        ("§1", "+31.6"): chk.MATCHED,
        ("§1", "8.81"): chk.UNMATCHED,
    }
    paper(
        repo,
        "## 1. One\n\nThe largest is +31.6 s, and a step 8.81.\n\n"
        "Source: `reports/live.md`.\n",
    )
    assert set(statuses(chk.check()).values()) == {chk.MATCHED}


def test_a_name_that_is_no_heading_confines_to_nothing(repo):
    paper(
        repo,
        "## 1. One\n\nIt is +14.72.\n\nSource: `reports/live.md`, *The regime shift*.\n",
    )
    result = chk.check()
    assert statuses(result) == {("§1", "+14.72"): chk.UNMATCHED}
    assert not outcome(result, 5).ok
    assert not outcome(result, 1).ok


def test_the_report_prints_every_match_in_scope_after_the_likeliest(repo):
    """A reader who knows the row can see whether the check found it."""
    paper(repo, "## 1. One\n\nThe ceiling is 9.46.\n\nSource: `reports/live.md`.\n")
    result = chk.check()
    [v] = result.sections[1].verdicts
    assert (v.match.line, v.alternates) == (14, ("`reports/live.md`:25 `9.46`",))
    rows = [line for line in chk.report(result, [], False) if "| 9.46 |" in line]
    assert "`reports/live.md`:14 `9.46` (also `reports/live.md`:25 `9.46`)" in rows[0]


def test_the_register_key_carries_the_sign(repo, monkeypatch):
    """One entry must not reach a magnitude and a signed bound that share digits."""
    paper(
        repo,
        "## 8. Results\n\nIt read 0.52, and −0.52 elsewhere.\n\n"
        "Source: `reports/frozen-2026-01-01.md`.\n",
    )
    monkeypatch.setattr(
        chk, "PAPER_HISTORICAL_FIGURES", (PaperFigure("§8", "0.52", "of its day"),)
    )
    result = chk.check()
    [plain, signed] = result.sections[1].verdicts
    assert (plain.status, plain.listed) == (chk.HISTORICAL, True)
    assert (signed.status, signed.listed) == (chk.UNMATCHED, False)
    assert chk.keyed(signed.number) == "-0.52", "a minus reads as a hyphen"


def test_the_register_is_never_a_source_for_its_own_entries(repo, monkeypatch):
    """A figure listed in the register is written there as a string."""
    register = repo / "src" / "taxiout"
    register.mkdir(parents=True)
    (register / "provenance.py").write_text(
        'X = ("a basis that quotes 0.63 s on its day",)\n'
        "_TEXT = (\n    'shared by two entries, 3.14'\n)\n"
        "PAPER_HISTORICAL_FIGURES = (\n    ('§1', '2.72', 'a reason'),\n)\n"
        "_ROWS = (\n    (1.62,),\n)\n",
        "utf-8",
    )
    monkeypatch.setattr(chk, "REGISTER_MODULE", "src/taxiout/provenance.py")
    paper(
        repo,
        "## 1. One\n\nIt read 0.63, 2.72, 3.14 and 1.62.\n\n"
        "Source: `src/taxiout/provenance.py`.\n",
    )
    assert statuses(chk.check()) == {
        ("§1", "0.63"): chk.MATCHED,
        ("§1", "2.72"): chk.UNMATCHED,
        ("§1", "3.14"): chk.UNMATCHED,
        ("§1", "1.62"): chk.MATCHED,
    }, "a basis is a source; a paper list and its shared text are not"


def test_a_cited_file_that_does_not_exist_fails(repo):
    paper(
        repo,
        "## 1. One\n\nNo figure.\n\nSource: `reports/gone.md`; the rationale page.\n",
    )
    result = chk.check()
    assert outcome(result, 5).findings == ["§1: reports/gone.md does not exist"]
    [gone, page] = result.sections[1].citations
    assert (gone.path, gone.exists) == ("reports/gone.md", False)
    assert (page.path, page.names) == (None, ()), "not a file, and not checked"


def test_a_section_excluded_by_name_is_skipped_and_reported(repo):
    """The mechanism the drafting notes used, kept although the list is empty."""
    paper(repo, "## 1. One\n\nIt is +14.72.\n\nSource: `reports/live.md`.\n")
    result = chk.check()
    [notes] = [r for r in result.sections if r.section.key == DRAFTING]
    assert notes.scope == chk.EXCLUDED
    assert notes.verdicts == ()
    assert [n.figure for n in notes.unchecked] == ["99.9"]
    assert (DRAFTING, "99.9") not in statuses(result)
    lines = outcome(result, 6).lines
    assert any("excluded by name, 1 figure(s) not checked" in line for line in lines)
    assert any("### Drafting notes" in line for line in chk.report(result, [], False))

    paper(repo, "## 1. One\n\nIt is +14.72.\n\nSource: `reports/live.md`.\n", False)
    assert not outcome(chk.check(), 6, strict=False).ok, "an exclusion nothing reaches"


def test_the_report_gives_every_figure_a_row_and_computes_the_totals(repo):
    paper(
        repo,
        "## 1. One\n\nIt is +14.72, then 15.1, then 53%.\n\nSource: `reports/live.md`.\n",
    )
    result = chk.check()
    lines = chk.report(result, chk.run(result, False), False)
    rows = [line for line in lines if line.startswith("| ") and "*" in line.split("|")[3]]
    assert len(rows) == 3
    assert "| *matched* | 2 |" in lines
    assert "| *unmatched* | 1 |" in lines
    assert "| figures in the checked sections | 3 |" in lines


def test_every_check_the_paper_checker_defines_is_wired_into_its_run(repo, monkeypatch):
    """The guard ``check_provenance.py`` carries, held here for the six."""
    paper(repo, "## 1. One\n\nIt is +14.72.\n\nSource: `reports/live.md`.\n")
    result = chk.check()
    titles = [o.title for o in chk.run(result, False)]
    assert [int(t.split(" -- ")[0]) for t in titles] == [1, 2, 3, 4, 5, 6]
    monkeypatch.setattr(
        chk, "check_forgotten", lambda result: ([], [], []), raising=False
    )
    with pytest.raises(SystemExit, match="7 checks defined and 6 wired"):
        chk.run(result, False)


# --------------------------------------------------------------------------
# the real tree


def test_the_checker_holds_no_constant_and_no_prose_figure():
    """Named explicitly, so this holds before the file is tracked as well as after.

    A numeric constant here would need a register entry, and an entry for a
    module check 1 cannot see until it is committed would fail the gate for as
    long as the file is new. A decimal figure in its prose would fail check 7
    the moment it is tracked.
    """
    module = "scripts/check_paper.py"
    assert cp.module_constants([module]) == ()
    ok, lines = cp.check_prose(cp.string_literals([module], prose_only=True))
    assert ok, lines


def test_strict_passes_on_the_paper_as_it_stands(capsys):
    """The freeze condition, held from the figure pass on.

    Before 25 September this test asserted the opposite: the paper was stale
    and a checker that passed it could not see. The figure pass brought every
    figure to the reports, and every one is now matched in the section its
    source line names, listed as history with a reason, or computed by an
    arithmetic the check verifies. A regeneration that moves a figure the
    paper quotes fails this test, which is what it is for.

    It did so once. On 28 September every report that reads the ranking set
    was regenerated against the files the organizers re-issued on 4 September
    (`reports/ranking-refresh-2026-09-28.md`), and figures the paper quoted
    from the set of 2 September moved: 20 unmatched and one figure matching
    only in a frozen record without being listed (0.19, in §7.4). From that
    day until the paper pass of 29 September this test was
    test_strict_fails_on_the_paper_until_the_paper_pass and pinned that debt
    at exactly those counts; the pass brought both to nothing and flipped it
    back to this form.
    """
    assert chk.main(["--strict"]) == 0
    assert chk.main([]) == 0
    ok, _lines = cp.check_paper_figures(strict=True)
    assert ok
    result = chk.check()
    counts = {s: 0 for s in chk.STATUSES}
    for _r, v in chk._verdicts(result):
        counts[v.status] += 1
    assert counts[chk.UNMATCHED] == 0
    assert all(v.listed for _r, v in chk._verdicts(result, chk.HISTORICAL))
    assert not any(r.broken for r in result.sections)
    capsys.readouterr()


def test_the_two_subsections_of_the_design_stand_alone_and_other_fours_do_not():
    """7.1.1 and 7.1.2 sit at four hashes in the paper and are still read as
    sections of their own; any other four-hash heading stays inside the
    section above it."""
    text = "\n".join(
        [
            "# Title",
            "## 7. Validation",
            "### 7.1 The design",
            "#### 7.1.1 What the forward legs cannot see",
            "#### 7.1.2 The rules under the parameters that ship",
            "### 8.7 After the first score",
            "#### 8.7.1 From the base configuration to the final one",
        ]
    )
    keys = [s.key for s in chk.sections(text)]
    assert "§7.1.1" in keys and "§7.1.2" in keys
    assert not any(k.startswith("§8.7.") for k in keys)
