#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Check every figure in ``docs/paper.md`` against the files its sections cite.

Each section of the paper names the files its figures come from, in a
``Source:`` or ``Sources:`` line. Nothing tied a quoted figure to that file
until this script, so a regeneration that moved a figure left the paper quoting
the old one, and the only instrument that could notice was a reader. The paper
is also the one document no later block reviews, so this is the instrument the
figure pass runs under, not a report written after it.

What a figure is
----------------
A decimal, with or without thousands separators; an integer of at least
``taxiout.provenance.PAPER_INTEGER_MIN_DIGITS`` digits; or a percentage. A
smaller integer is not, and that is a stated gap rather than an oversight:
``taxiout.provenance.SCOPE`` records it. A token shaped like a figure that
measures nothing, a section number, a year or a version, is exempt by a pattern
in ``PAPER_EXEMPT_PATTERNS`` with its reason, and is counted under the
pattern's name rather than dropped.

The match rule
--------------
A figure written with ``k`` decimals matches a file if the file carries a
number, written with at least ``k`` decimals, that rounds to it at ``k``
decimals, half up or half to even. A percentage matches only a percentage. A
figure written with a minus matches only a negative number, and one written
with a plus only a number that is not negative. A figure written without a sign
is a magnitude and matches either: the paper says a block is worth so many
seconds where a report prints the step as a negative change. The paper sets a
minus as U+2212 and the reports as a hyphen, and both read as minus. **The
report prints the number that matched, with its file and line**, so a
coincidental match, which a short figure invites, is visible rather than
silent.

Where a figure is looked for
----------------------------
In the files its section's source line cites, where they exist, **and where
the source line names a section of a file, in that section of it.** A section
name is set in italics after the file it belongs to, and it confines the search
to the lines from every heading it names, in that file, to the next heading of
the same or a higher level: a subsection is inside its section, and a name that
is a heading twice reads both. A name that is no heading in its file confines
to nothing, so every figure read against it is unmatched there and check 5
names the citation. A file cited with no section name is read whole. A section
on ``PAPER_NO_SOURCE_SECTIONS`` is read against every live report instead:
every report under ``reports/`` that is not a frozen dated record. A section
with figures and no source line that is not on that list has nowhere to look,
so every figure in it is unmatched and check 4 names it.

The reason for the confinement is in the work list that preceded it: a figure
of one or two significant digits matches a long report somewhere by chance,
and a checker that reads the whole file certifies a stale figure against a
number in a table its section never cites. Read against the section the paper
names, the same figure has to be there.

The frozen rule
---------------
A figure that matches in no live file it is read against, and does match in a
cited frozen dated record from ``FROZEN_DOCUMENTS``, is *historical*: it was
true of a model that has since moved, and it will match forever because the
record cannot change. It passes only on ``PAPER_HISTORICAL_FIGURES``, with its
section and a reason. Without the rule the check would certify a seed-averaging
gain quoted from ``reports/preregistered-2026-09-20.md`` long after the shipped
model read something else.

The computed rule
-----------------
A figure the paper computes from report figures rather than quotes, a
difference stated in words, is on ``PAPER_FIGURE_EXEMPTIONS`` with its
arithmetic written as an expression and its operands named, and **the
arithmetic is verified, not trusted**: every named operand has to be a number
in the expression and has to match in the section's own sources, and the
expression's value has to round to the figure at the figure's precision with
the sign the figure carries. A number in the expression that is not a named
operand is a constant of the arithmetic, and the report prints it as one. An
entry that fails any part of that is broken in both modes and certifies
nothing: the figure is then read as if it were not listed.

So a figure is, in this order of precedence: **exempt** on a verified entry,
**matched** in a live source, **historical** in a frozen one, or
**unmatched**. The verified entry comes first because a figure computed from
two report figures can also equal a number its section's sources carry, by
chance; the entry says what the figure is, and the chance match is printed
beside it so a reader can see both.

The checks
----------
1. every figure appears in a cited source
2. a figure found only in a frozen record is listed as historical
3. a figure computed from report figures carries its arithmetic, verified
4. a section with figures and no source line is listed
5. every cited file exists, and every cited section name is a heading in it
6. a section excluded by name is reported, and reaches a section

Outside ``--strict`` a finding is printed and passes, which is how
``check_provenance.py`` treats its pending list: a gate that fails today cannot
be wired into a commit path. ``--strict`` fails on every finding and is the
paper's freeze condition. **A register entry that reaches nothing, or that
fails its own verification, fails in both modes**, as an unreached citation
exemption does in check 11: an allowance nobody needs is a stale record of its
own. ``--report`` prints every figure with its status, section by section, then
the totals: the figure pass's work list.

Usage
-----
    python scripts/check_paper.py
    python scripts/check_paper.py --report
    python scripts/check_paper.py --strict
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import NamedTuple

# The repository root, resolved without a subscript so that this module holds no
# numeric constant. A register entry for one would name a constant check 1
# cannot see while the file is untracked, and the gate would fail until the
# commit that adds the file.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from taxiout.provenance import (  # noqa: E402
    FROZEN_DOCUMENTS,
    PAPER,
    PAPER_EXCLUDED_SECTIONS,
    PAPER_EXEMPT_PATTERNS,
    PAPER_FIGURE_EXEMPTIONS,
    PAPER_HISTORICAL_FIGURES,
    PAPER_INTEGER_MIN_DIGITS,
    PAPER_NO_SOURCE_SECTIONS,
    REGISTER_MODULE,
)

MATCHED = "matched"
HISTORICAL = "historical"
EXEMPT = "exempt"
UNMATCHED = "unmatched"
STATUSES = (MATCHED, HISTORICAL, EXEMPT, UNMATCHED)

# Where a section's figures are looked for.
CITED = "its cited files"
EVERY_LIVE_REPORT = "every live report"
NOWHERE = "nowhere"
EXCLUDED = "excluded"

# The lines before the first section: the title, the status note and the team.
# Kept as a section of their own, so a figure there cannot fall outside every
# check.
PREAMBLE = "(title and preamble)"

# A number as prose or a table writes it: an integer part, bare or with
# thousands separators, an optional decimal part and an optional percent sign.
# Never inside a word, a digest, a path component or a larger number: a digit
# after a letter, a point or a comma does not start one, and nothing that would
# continue a number may follow it. A dotted number with three or more parts, a
# version, yields nothing at all.
NUMBER = re.compile(
    r"(?<![\w.,])(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?"
    r"(?![\w]|\.\d|,\d{3})(?P<pct>%?)"
)

# A sign written against a number. The paper sets a minus as U+2212 and the
# reports as a hyphen; both read as minus.
SIGNS = {"+": "+", "-": "-", "−": "-"}

SECTION_HEADING = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
STATUS_TAG = re.compile(r"\s*\[[^\]]*\]\s*$")
SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
SOURCE_LINE = re.compile(r"^Sources?:\s*")
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# What a source line names. A file is a path in backticks with an extension;
# the paper also names the README by the bare word. A section name is set in
# italics after the file it belongs to.
FILE_REF = re.compile(
    r"`(?P<path>[^`\s]+\.(?:md|py|txt|svg|csv|json|toml))`"
    r"|(?<![`/\w])(?P<readme>README)(?![\w`]|\.md)"
)
ITALIC = re.compile(r"(?<![*\w])\*(?!\*)(?P<name>[^*]+?)(?<!\s)\*(?![*\w])")

# The arithmetic an exemption may use: the four operations, a power, a sign
# and parentheses over numbers, and nothing that names or calls anything.
OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}


class Number(NamedTuple):
    """One number as a file writes it."""

    line: int
    start: int
    end: int
    figure: str  # as written, without its sign
    sign: str  # "+", "-" or ""
    written: str  # as written, with the sign character the file used
    value: Decimal  # unsigned
    decimals: int
    percent: bool
    digits: int  # in the integer part


class Section(NamedTuple):
    """One section of the paper, from its heading to the next."""

    key: str
    heading: str
    start: int
    end: int
    lines: tuple[tuple[int, str], ...]
    source: str  # the source paragraphs, joined; empty if there are none
    source_lines: frozenset[int]


class Heading(NamedTuple):
    """One heading of a cited file, with the lines its section runs over."""

    line: int
    level: int
    name: str  # as written, emphasis and hashes stripped
    end: int  # the last line before the next heading of its level or higher


class Citation(NamedTuple):
    """One clause of a source line: the file it names, if any, and its names."""

    path: str | None  # None for a clause that names no file in the repository
    names: tuple[str, ...]
    text: str
    exists: bool
    unresolved: tuple[str, ...]  # names that are no heading in the file
    spans: tuple[tuple[str, int, int], ...]  # (name, first line, last line)


class Source(NamedTuple):
    """A file figures are read against, with its numbers indexed by rounding.

    ``lines`` is the set of lines the section's citation confines the file to,
    or ``None`` for a file cited whole.
    """

    path: str
    frozen: bool
    index: dict
    lines: frozenset[int] | None


class Verdict(NamedTuple):
    number: Number
    status: str
    source: str  # the file it matched in, for matched and historical
    match: Number | None
    listed: bool  # a historical figure on PAPER_HISTORICAL_FIGURES
    reason: str  # the entry's reason, for an exempt figure
    arithmetic: str  # the verified arithmetic, printed, for an exempt figure
    elsewhere: tuple[str, ...]  # files carrying an unmatched figure, not sources
    outside: tuple[str, ...]  # matches in a cited file outside its cited sections
    alternates: tuple[str, ...]  # the other matches in scope, likeliest first


class SectionResult(NamedTuple):
    section: Section
    excluded: str  # the reason, if the section is excluded by name
    citations: tuple[Citation, ...]
    scope: str
    sources: tuple[str, ...]
    verdicts: tuple[Verdict, ...]
    by_pattern: tuple[tuple[Number, str], ...]
    unchecked: tuple[Number, ...]  # the figures of an excluded section
    broken: tuple[str, ...]  # exemption entries of this section that failed


class Result(NamedTuple):
    paper: str
    sha256: str
    sections: tuple[SectionResult, ...]
    live_reports: tuple[str, ...]


# --------------------------------------------------------------------------
# reading numbers


def _sign(line: str, start: int) -> str:
    """The sign written immediately before a number, or ``""``.

    A hyphen after a digit or a letter is a range, a date or a name and not a
    sign, so ``20-25`` and ``2026-09-20`` carry no negative number.
    """
    if start == 0 or line[start - 1] not in SIGNS:
        return ""
    before = line[start - 2] if start >= 2 else " "
    if before.isalnum() or before in "._":
        return ""
    return SIGNS[line[start - 1]]


def numbers(lines) -> list[Number]:
    """Every number in ``lines``, an iterable of (line number, text)."""
    out = []
    for i, line in lines:
        for m in NUMBER.finditer(line):
            whole = m.group("int").replace(",", "")
            dec = m.group("dec") or ""
            try:
                value = Decimal(f"{whole}.{dec}" if dec else whole)
            except InvalidOperation:
                continue
            sign = _sign(line, m.start())
            out.append(
                Number(
                    line=i,
                    start=m.start(),
                    end=m.end(),
                    figure=m.group(0),
                    sign=sign,
                    written=(line[m.start() - 1] if sign else "") + m.group(0),
                    value=value,
                    decimals=len(dec),
                    percent=bool(m.group("pct")),
                    digits=len(whole),
                )
            )
    return out


def is_figure(n: Number) -> bool:
    """A decimal, a percentage, or an integer long enough to be a quantity."""
    return bool(n.decimals or n.percent or n.digits >= PAPER_INTEGER_MIN_DIGITS)


def keyed(n: Number) -> str:
    """The figure as a register entry writes it: with its sign, a minus as a hyphen.

    The sign is part of the key because the same digits can be a computed
    magnitude in one sentence and a quoted negative bound in the next, and one
    entry must not reach both.
    """
    return n.sign + n.figure


def _rounded(value: Decimal, k: int) -> set[Decimal]:
    step = Decimal(f"1e-{k}")
    try:
        return {
            value.quantize(step, rounding=ROUND_HALF_UP),
            value.quantize(step, rounding=ROUND_HALF_EVEN),
        }
    except InvalidOperation:
        return set()


def index(found: list[Number]) -> dict:
    """Every number, under each precision it can be rounded to."""
    out: dict = defaultdict(list)
    for n in found:
        for k in range(n.decimals + 1):
            for r in _rounded(n.value, k):
                out[(n.percent, k, r)].append(n)
    return out


def _signs_agree(figure: Number, n: Number) -> bool:
    """Whether ``n`` carries the sign ``figure`` claims.

    A figure written without a sign is a magnitude: the paper says a block is
    worth so many seconds where the report prints the step as a negative
    change, so it agrees with either sign. A figure written with a minus
    claims a negative number and agrees only with one; a figure written with a
    plus agrees with a plus or with no sign, which is how a report writes a
    positive number.
    """
    if not figure.sign:
        return True
    return (n.sign == "-") == (figure.sign == "-")


def candidates(figure: Number, source: Source) -> list[Number]:
    """Every number in ``source``, inside its cited lines, that ``figure`` matches."""
    return [
        n
        for n in source.index.get((figure.percent, figure.decimals, figure.value), ())
        if _signs_agree(figure, n) and (source.lines is None or n.line in source.lines)
    ]


def find(figure: Number, source: Source) -> Number | None:
    """The number in ``source`` that ``figure`` matches, or ``None``."""
    found = best_match(figure, [source])
    return found[1] if found else None


def ranked(
    figure: Number, sources, companions: tuple[Number, ...] = ()
) -> list[tuple[Source, Number]]:
    """Every match for ``figure`` in ``sources``, likeliest first, with its file.

    ``companions`` are the other figures on the figure's own line of the
    paper. A match with more of them on its own line or the lines either side
    is preferred, since a sentence quoting an interval is quoting one row and a
    sentence quoting two thresholds may be quoting two rows of one table; then
    a number written at the figure's own precision over one that rounds to it;
    then one whose sign is written the way the figure's is; then the earlier
    source in the order given, and the earlier line. **The preference decides
    only which match is printed first, never whether there is one**, and the
    report prints the others after it, so a reader who knows which row the
    sentence quotes can see whether the check found that row or another.
    """
    found = []
    for order, source in enumerate(sources):
        near = [
            {m.line + d for m in candidates(c, source) for d in (-1, 0, 1)}
            for c in companions
        ]
        for n in candidates(figure, source):
            key = (
                -sum(n.line in lines for lines in near),
                n.decimals != figure.decimals,
                n.sign != figure.sign,
                order,
                n.line,
            )
            found.append((key, source, n))
    return [(s, n) for _k, s, n in sorted(found, key=lambda x: x[0])]


def best_match(
    figure: Number, sources, companions: tuple[Number, ...] = ()
) -> tuple[Source, Number] | None:
    """The likeliest match for ``figure`` in ``sources``, or ``None``."""
    found = ranked(figure, sources, companions)
    return found[0] if found else None


# --------------------------------------------------------------------------
# reading the paper


def section_key(heading: str) -> str:
    """``§`` and the number for a numbered section, else the bare heading.

    The bracketed status tag is dropped, so editing it does not orphan a
    register entry keyed on the section.
    """
    bare = STATUS_TAG.sub("", heading).strip()
    m = SECTION_NUMBER.match(bare + " ")
    return f"§{m.group(1)}" if m else bare


def _source_paragraphs(lines) -> tuple[str, frozenset[int]]:
    """The section's source line or lines, joined, and the lines they take.

    A source runs from a line starting ``Source:`` or ``Sources:`` to the end
    of its paragraph, so a source wrapped over several lines is read whole.
    """
    joined: list[str] = []
    taken: set[int] = set()
    block: list[tuple[int, str]] = []
    for i, text in [*lines, (None, "")]:
        if text.strip():
            block.append((i, text.strip()))
            continue
        at = next((j for j, (_i, t) in enumerate(block) if SOURCE_LINE.match(t)), None)
        if at is not None:
            joined.append(" ".join(t for _i, t in block[at:]))
            taken |= {i for i, _t in block[at:]}
        block = []
    return " ".join(joined), frozenset(taken)


def sections(text: str) -> list[Section]:
    """The paper cut at every heading of two or three hashes.

    What comes before the first is kept as a section of its own, so a figure
    in the title or the status note cannot fall outside every check.
    """
    rows = text.splitlines()
    heads = [
        (i, m) for i, line in enumerate(rows, 1) if (m := SECTION_HEADING.match(line))
    ]
    bounds = [(1, None), *heads]
    out = []
    for n, (start, m) in enumerate(bounds):
        end = bounds[n + 1][0] - 1 if n + 1 < len(bounds) else len(rows)
        if end < start:
            continue
        heading = m.group(2) if m else PREAMBLE
        lines = tuple((i, rows[i - 1]) for i in range(start, end + 1))
        source, taken = _source_paragraphs(lines)
        out.append(
            Section(
                key=section_key(heading) if m else PREAMBLE,
                heading=heading,
                start=start,
                end=end,
                lines=lines,
                source=source,
                source_lines=taken,
            )
        )
    return out


def figures(section: Section, patterns) -> tuple[list[Number], list[tuple[Number, str]]]:
    """The section's figures, and the tokens an exempt pattern covers.

    The source paragraphs are left out: they cite, they do not claim.
    """
    found: list[Number] = []
    exempt: list[tuple[Number, str]] = []
    for i, line in section.lines:
        if i in section.source_lines:
            continue
        spans = [
            (m.start(), m.end(), name) for name, rx in patterns for m in rx.finditer(line)
        ]
        for n in numbers([(i, line)]):
            if not is_figure(n):
                continue
            hit = next(
                (name for a, b, name in spans if a <= n.start and n.end <= b), None
            )
            if hit:
                exempt.append((n, hit))
            else:
                found.append(n)
    return found, exempt


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[`*_]", "", text).split()).casefold()


def headings(text: str) -> list[Heading]:
    """Every markdown heading in ``text``, with the lines its section runs over.

    A section runs from its heading to the line before the next heading of the
    same or a higher level, so a subsection is inside the section above it.
    """
    rows = text.splitlines()
    found = [
        (i, len(m.group(1)), m.group(2))
        for i, line in enumerate(rows, 1)
        if (m := MARKDOWN_HEADING.match(line))
    ]
    out = []
    for k, (line, level, name) in enumerate(found):
        end = next(
            (nl - 1 for nl, nlevel, _n in found[k + 1 :] if nlevel <= level), len(rows)
        )
        out.append(Heading(line, level, name, end))
    return out


def _names(name: str, heading: str) -> bool:
    """Whether a cited ``name`` names ``heading``.

    Case and emphasis aside, the heading is the name itself or begins with it
    at a word boundary, which is how a heading with a subtitle after a colon
    is cited by its short name.
    """
    want, have = _plain(name), _plain(heading)
    return have == want or (have.startswith(want) and not have[len(want)].isalnum())


def spans(name: str, text: str) -> tuple[tuple[int, int], ...]:
    """The line ranges of every section of ``text`` that ``name`` names."""
    return tuple((h.line, h.end) for h in headings(text) if _names(name, h.name))


def resolves(name: str, text: str) -> bool:
    """Whether ``name`` is a heading in the markdown ``text``."""
    return bool(spans(name, text))


def citations(source: str) -> list[tuple[str | None, tuple[str, ...], str]]:
    """What a source line names: (file, section names, the clause it is in).

    Clauses are separated by semicolons. A section name belongs to the nearest
    file before it in its clause, and a clause naming no file is kept, with any
    names in it, so a name with no file to be a heading in is still seen.
    """
    body = SOURCE_LINE.sub("", source, count=1)
    out: list[tuple[str | None, tuple[str, ...], str]] = []
    for clause in (c.strip() for c in body.split(";")):
        if not clause:
            continue
        files = [
            (m.start(), m.group("path") or "README.md") for m in FILE_REF.finditer(clause)
        ]
        names = [(m.start(), m.group("name").strip()) for m in ITALIC.finditer(clause)]
        if not files:
            out.append((None, tuple(n for _p, n in names), clause))
            continue
        for k, (at, path) in enumerate(files):
            until = files[k + 1][0] if k + 1 < len(files) else len(clause)
            out.append((path, tuple(n for p, n in names if at < p < until), clause))
        before = tuple(n for p, n in names if p < files[0][0])
        if before:
            out.append((None, before, clause))
    return out


# --------------------------------------------------------------------------
# the arithmetic of an exemption


def literals(expression: str) -> list[Decimal]:
    """Every number in ``expression``, as written, or a ``ValueError``.

    Parsed as Python and walked, so only numbers, the operators in
    ``OPERATORS``, a unary sign and parentheses are accepted; a name, a call
    or a subscript is refused, which keeps the arithmetic arithmetic.
    """
    tree = ast.parse(expression.strip(), mode="eval")
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError(f"{node.value!r} is not a number")
            out.append(Decimal(ast.get_source_segment(expression.strip(), node)))
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in OPERATORS:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        elif not isinstance(node, (ast.Expression, ast.operator, ast.unaryop)):
            raise ValueError(f"{type(node).__name__} is not arithmetic")
    return out


def evaluate(expression: str) -> Decimal:
    """The value of ``expression``, in decimal arithmetic."""
    text = expression.strip()

    def value(node) -> Decimal:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant):
            return Decimal(ast.get_source_segment(text, node))
        if isinstance(node, ast.UnaryOp):
            v = value(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](value(node.left), value(node.right))
        raise ValueError(f"{type(node).__name__} is not arithmetic")

    literals(text)
    return value(ast.parse(text, mode="eval"))


def _as_number(text: str) -> Number | None:
    """``text``, an operand as an entry writes it, read as one number."""
    found = numbers([(0, text.strip())])
    return found[0] if len(found) == 1 and found[0].written == text.strip() else None


def verify(entry, figure: Number, sources) -> tuple[str, list[str]]:
    """Check an exemption's arithmetic against ``sources``.

    Returns what the report prints for a verified entry, and the list of what
    failed, empty when nothing did. Every failure is a broken register entry.
    """
    problems: list[str] = []
    expression = entry.arithmetic.strip()
    if not expression:
        return "", [f"{entry.figure} in {entry.section} carries no arithmetic"]
    try:
        found = literals(expression)
        result = evaluate(expression)
    except (ValueError, SyntaxError, InvalidOperation, ZeroDivisionError) as err:
        return "", [f"{entry.figure} in {entry.section}: {expression!r} ({err})"]
    if not entry.operands:
        problems.append(f"{entry.figure} in {entry.section} names no operand")
    shown = []
    for text in entry.operands:
        operand = _as_number(text)
        if operand is None:
            problems.append(
                f"{entry.figure} in {entry.section}: operand {text!r} is not one number"
            )
            continue
        signed = -operand.value if operand.sign == "-" else operand.value
        if not any(lit == signed for lit in found):
            problems.append(
                f"{entry.figure} in {entry.section}: operand {text} is not in {expression!r}"
            )
        hit = best_match(operand, sources)
        if hit is None:
            problems.append(
                f"{entry.figure} in {entry.section}: operand {text} is in no cited source"
            )
            continue
        shown.append(f"{text} at `{hit[0].path}`:{hit[1].line} `{hit[1].written}`")
    named = {
        -o.value if o.sign == "-" else o.value
        for o in map(_as_number, entry.operands)
        if o
    }
    constants = sorted({str(lit) for lit in found if lit not in named})
    if figure.sign == "-" and result >= 0 or figure.sign == "+" and result < 0:
        problems.append(
            f"{entry.figure} in {entry.section}: {expression!r} is {result}, and the"
            f" figure is written {figure.written}"
        )
    elif figure.value not in _rounded(abs(result), figure.decimals):
        problems.append(
            f"{entry.figure} in {entry.section}: {expression!r} is {result}, which does"
            f" not round to {figure.figure}"
        )
    printed = f"= {expression} = {result.normalize()}"
    if shown:
        printed += "; operands " + ", ".join(shown)
    if constants:
        printed += "; constants of the arithmetic: " + ", ".join(constants)
    return printed, problems


# --------------------------------------------------------------------------
# the check


def own_entries(register: str) -> frozenset[int]:
    """The lines of the register that hold its lists for the paper.

    A section may cite the register, for a basis that quotes a dated reading,
    and the register also holds the entries this check reads: a figure listed
    as historical or computed is written in it as a string. Read as a source,
    those lines would match a figure against its own entry, so every
    module-level assignment whose name starts with ``PAPER_`` is left out of
    the register when it is read as a source, and so is every text those
    entries share, which is a private assignment whose value is one string.
    The register's bases stay: a section may quote a dated reading from one.
    """
    out: set[int] = set()
    for node in ast.parse(register).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        text = isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        if any(n.startswith("PAPER_") or (n.startswith("_") and text) for n in names):
            out |= set(range(node.lineno, node.end_lineno + 1))
    return frozenset(out)


def check() -> Result:
    """Every figure in the paper, with its status and where it was found."""
    paper = REPO / PAPER
    text = paper.read_text(encoding="utf-8")
    frozen = {f.path for f in FROZEN_DOCUMENTS}
    live = tuple(
        sorted(
            f"reports/{p.name}"
            for p in (REPO / "reports").glob("*.md")
            if f"reports/{p.name}" not in frozen
        )
    )
    # Where an unmatched figure is looked for as a lead: every live report and
    # every frozen record.
    anywhere = (*live, *sorted(p for p in frozen if (REPO / p).is_file()))
    texts: dict[str, str | None] = {}
    indexed: dict[str, dict | None] = {}

    def body(rel: str) -> str | None:
        if rel not in texts:
            path = REPO / rel
            texts[rel] = (
                path.read_text(encoding="utf-8", errors="replace")
                if path.is_file()
                else None
            )
        return texts[rel]

    def source(rel: str, lines: frozenset[int] | None = None) -> Source | None:
        if rel not in indexed:
            got = body(rel)
            if got is None:
                indexed[rel] = None
            else:
                skip = own_entries(got) if rel == REGISTER_MODULE else frozenset()
                found = [
                    n
                    for n in numbers(enumerate(got.splitlines(), 1))
                    if n.line not in skip
                ]
                indexed[rel] = index(found)
        got = indexed[rel]
        return None if got is None else Source(rel, rel in frozen, got, lines)

    excluded = {e.section: e.why for e in PAPER_EXCLUDED_SECTIONS}
    no_source = {e.section for e in PAPER_NO_SOURCE_SECTIONS}
    historical = {(e.section, e.figure) for e in PAPER_HISTORICAL_FIGURES}
    computed = {(e.section, e.figure): e for e in PAPER_FIGURE_EXEMPTIONS}
    patterns = [(p.name, re.compile(p.pattern)) for p in PAPER_EXEMPT_PATTERNS]

    out = []
    for s in sections(text):
        found, by_pattern = figures(s, patterns)
        if s.key in excluded:
            out.append(
                SectionResult(
                    s,
                    excluded[s.key],
                    (),
                    EXCLUDED,
                    (),
                    (),
                    tuple(by_pattern),
                    tuple(found),
                    (),
                )
            )
            continue
        cites = []
        for path, names, clause in citations(s.source) if s.source else []:
            if path is None:
                cites.append(Citation(None, names, clause, False, names, ()))
                continue
            got = body(path)
            resolved = []
            unresolved = []
            for n in names:
                ranges = spans(n, got) if got is not None and path.endswith(".md") else ()
                if ranges:
                    resolved += [(n, a, b) for a, b in ranges]
                else:
                    unresolved.append(n)
            cites.append(
                Citation(
                    path,
                    names,
                    clause,
                    got is not None,
                    tuple(unresolved),
                    tuple(resolved),
                )
            )
        # The lines each cited file is confined to: ``None`` where any citation
        # of it names no section, else the union of the sections named.
        confined: dict[str, frozenset[int] | None] = {}
        for c in cites:
            if not c.path or not c.exists:
                continue
            if not c.names:
                confined[c.path] = None
            elif c.path not in confined or confined[c.path] is not None:
                have = confined.get(c.path) or frozenset()
                confined[c.path] = have | frozenset(
                    i for _n, a, b in c.spans for i in range(a, b + 1)
                )
        files = tuple(confined)
        if files:
            scope, pool = CITED, files
            sources = [x for x in (source(p, confined[p]) for p in pool) if x is not None]
        elif s.key in no_source:
            scope, pool = EVERY_LIVE_REPORT, live
            sources = [x for x in (source(p) for p in pool) if x is not None]
        else:
            scope, pool, sources = NOWHERE, (), []
        verdicts = []
        broken: list[str] = []

        def where(hits) -> tuple[str, ...]:
            return tuple(f"`{x.path}`:{n.line} `{n.written}`" for x, n in hits)

        for fig in found:
            companions = tuple(c for c in found if c.line == fig.line and c is not fig)
            key = (s.key, keyed(fig))
            entry = computed.get(key)
            if entry is not None:
                printed, problems = verify(entry, fig, sources)
                broken += problems
                if not problems:
                    also = ranked(fig, sources, companions)
                    verdicts.append(
                        Verdict(
                            fig,
                            EXEMPT,
                            also[0][0].path if also else "",
                            also[0][1] if also else None,
                            False,
                            entry.why,
                            printed,
                            (),
                            (),
                            where(also[1:]),
                        )
                    )
                    continue
            hits = ranked(fig, [x for x in sources if not x.frozen], companions)
            if hits:
                (src, n), *rest = hits
                verdicts.append(
                    Verdict(fig, MATCHED, src.path, n, False, "", "", (), (), where(rest))
                )
                continue
            hits = ranked(fig, [x for x in sources if x.frozen], companions)
            if hits:
                (src, n), *rest = hits
                verdicts.append(
                    Verdict(
                        fig,
                        HISTORICAL,
                        src.path,
                        n,
                        key in historical,
                        "",
                        "",
                        (),
                        (),
                        where(rest),
                    )
                )
                continue
            outside = tuple(
                f"`{x.path}`:{n.line} `{n.written}`"
                for x in sources
                if x.lines is not None and (n := find(fig, x._replace(lines=None)))
            )
            elsewhere = tuple(
                p
                for p in anywhere
                if p not in pool and (x := source(p)) is not None and find(fig, x)
            )
            verdicts.append(
                Verdict(fig, UNMATCHED, "", None, False, "", "", elsewhere, outside, ())
            )
        out.append(
            SectionResult(
                s,
                "",
                tuple(cites),
                scope,
                pool,
                tuple(verdicts),
                tuple(by_pattern),
                (),
                tuple(broken),
            )
        )
    digest = hashlib.sha256(paper.read_bytes()).hexdigest()
    return Result(PAPER, digest, tuple(out), live)


# --------------------------------------------------------------------------
# the six checks
#
# Each returns (findings, broken, lines). A finding fails under --strict only;
# a broken register entry fails in both modes; the lines are what is printed.


def _checked(result: Result) -> list[SectionResult]:
    return [r for r in result.sections if r.scope != EXCLUDED]


def _verdicts(result: Result, status: str | None = None):
    return [
        (r, v)
        for r in _checked(result)
        for v in r.verdicts
        if status is None or v.status == status
    ]


def _counted(keys) -> str:
    counts = Counter(keys)
    return ", ".join(f"{k} {n}" for k, n in counts.items()) or "none"


def check_cited(result: Result):
    """Check 1: every figure appears in a cited source."""
    every = _verdicts(result)
    counts = Counter(v.status for _r, v in every)
    unmatched = _verdicts(result, UNMATCHED)
    headed = [r for r in _checked(result) if r.section.key != PREAMBLE]
    lines = [
        f"{len(every)} figures in the {len(headed)} sections checked and the title"
        " and preamble: " + ", ".join(f"{counts[s]} {s}" for s in STATUSES),
        "matched, by the file they matched in: "
        + _counted(sorted(v.source for _r, v in _verdicts(result, MATCHED))),
        "unmatched, by section: " + _counted(r.section.key for r, _v in unmatched),
        "unmatched but in a cited file outside the section it cites: "
        + str(sum(1 for _r, v in unmatched if v.outside)),
    ]
    findings = [
        f"{r.section.key} line {v.number.line}: {v.number.written} is in no cited source"
        + (
            f" (in {', '.join(v.outside)}, outside the cited sections)"
            if v.outside
            else ""
        )
        for r, v in unmatched
    ]
    return findings, [], lines


def check_frozen_rule(result: Result):
    """Check 2: a figure found only in a frozen record is listed as historical."""
    hist = _verdicts(result, HISTORICAL)
    unlisted = [(r, v) for r, v in hist if not v.listed]
    reached = {(r.section.key, keyed(v.number)) for r, v in hist}
    lines = [
        f"{len(hist)} figure(s) match only in a frozen dated record,"
        f" {len(hist) - len(unlisted)} of them on PAPER_HISTORICAL_FIGURES",
        "by record: " + _counted(sorted(v.source for _r, v in hist)),
        "by section: " + _counted(r.section.key for r, _v in hist),
    ]
    findings = [
        f"{r.section.key} line {v.number.line}: {v.number.written} matches only in"
        f" {v.source}, a frozen record, and is not listed"
        for r, v in unlisted
    ]
    broken = [
        f"PAPER_HISTORICAL_FIGURES lists {e.figure} in {e.section}, and no figure there"
        " matches only in a frozen record: an entry nothing reaches is stale"
        for e in PAPER_HISTORICAL_FIGURES
        if (e.section, e.figure) not in reached
    ]
    return findings, broken, lines


def check_computed(result: Result):
    """Check 3: a figure computed from report figures carries its arithmetic, verified."""
    exempt = _verdicts(result, EXEMPT)
    present = {
        (r.section.key, keyed(v.number)) for r in _checked(result) for v in r.verdicts
    }
    broken = [
        f"PAPER_FIGURE_EXEMPTIONS lists {e.figure} in {e.section}, which carries no"
        " such figure: an exemption nothing reaches is stale"
        for e in PAPER_FIGURE_EXEMPTIONS
        if (e.section, e.figure) not in present
    ] + [f"PAPER_FIGURE_EXEMPTIONS: {b}" for r in _checked(result) for b in r.broken]
    chance = [(r, v) for r, v in exempt if v.match is not None]
    lines = [
        f"{len(exempt)} figure(s) exempt as computed from report figures, each with its"
        " arithmetic verified: every operand in a cited source, the value rounding to"
        " the figure",
        f"{len(chance)} of them also equal a number in a cited source, by chance;"
        " the report prints both",
    ]
    return [], broken, lines


def check_no_source(result: Result):
    """Check 4: a section with figures and no source line is listed."""
    listed = {e.section for e in PAPER_NO_SOURCE_SECTIONS}
    keys = {r.section.key: r for r in result.sections}
    bare = [r for r in _checked(result) if not r.section.source and r.verdicts]
    failing = [r for r in bare if r.section.key not in listed]
    broken = []
    for e in PAPER_NO_SOURCE_SECTIONS:
        r = keys.get(e.section)
        if r is None:
            why = "which the paper does not have"
        elif r.scope == EXCLUDED:
            why = "which is excluded by name"
        elif r.section.source:
            why = "which has a source line now: the entry is stale"
        elif not r.verdicts:
            why = "which carries no figure: the entry is stale"
        else:
            continue
        broken.append(f"PAPER_NO_SOURCE_SECTIONS names {e.section}, {why}")
    lines = [
        f"{len(bare)} section(s) carry figures and no source line: "
        + (", ".join(r.section.key for r in bare) or "none"),
        f"{len(bare) - len(failing)} listed in PAPER_NO_SOURCE_SECTIONS and read against"
        f" every live report, {len(failing)} not listed: "
        + (", ".join(r.section.key for r in failing) or "none"),
    ]
    findings = [
        f"{r.section.key} carries {len(r.verdicts)} figure(s) and no source line"
        for r in failing
    ]
    return findings, broken, lines


def check_citations(result: Result):
    """Check 5: every cited file exists, and every cited name is a heading in it."""
    cites = [(r, c) for r in _checked(result) for c in r.citations]
    files = [(r, c) for r, c in cites if c.path]
    missing = [(r, c) for r, c in files if not c.exists]
    unresolved = [(r, c, n) for r, c in cites for n in c.unresolved]
    outside = [(r, c) for r, c in cites if not c.path and not c.names]
    named = [(r, c) for r, c in files if c.names and c.exists]
    lines = [
        f"{len(files)} file citation(s) in {len({r.section.key for r, _c in files})}"
        f" section(s), {len(missing)} naming a file that does not exist",
        f"{sum(len(c.names) for _r, c in cites)} section name(s) cited,"
        f" {len(unresolved)} not a heading in the file they are cited from",
        f"{len(named)} citation(s) confined to the sections they name, reading"
        f" {sum(b - a + 1 for _r, c in named for _n, a, b in c.spans)} lines in all",
        f"{len(outside)} clause(s) name no repository file and are not checked",
    ]
    findings = [f"{r.section.key}: {c.path} does not exist" for r, c in missing] + [
        f"{r.section.key}: '{n}' is not a heading in {c.path or 'any file'}"
        for r, c, n in unresolved
    ]
    return findings, [], lines


def check_excluded(result: Result):
    """Check 6: a section excluded by name is reported, and reaches a section."""
    keys = {r.section.key: r for r in result.sections}
    lines, broken = [], []
    for e in PAPER_EXCLUDED_SECTIONS:
        r = keys.get(e.section)
        if r is None:
            broken.append(
                f"PAPER_EXCLUDED_SECTIONS names '{e.section}', which the paper does not"
                " have: an exclusion nothing reaches is stale"
            )
            continue
        lines.append(
            f"'{e.section}', lines {r.section.start} to {r.section.end}: excluded by"
            f" name, {len(r.unchecked)} figure(s) not checked, and it ships in {PAPER}"
        )
    if not PAPER_EXCLUDED_SECTIONS:
        lines.append("no section is excluded by name: every section is checked")
    return [], broken, lines


CHECKS = (
    ("1 -- every figure appears in a cited source", check_cited),
    (
        "2 -- a figure found only in a frozen record is listed as historical",
        check_frozen_rule,
    ),
    (
        "3 -- a figure computed from report figures carries its arithmetic, verified",
        check_computed,
    ),
    ("4 -- a section with figures and no source line is listed", check_no_source),
    ("5 -- every cited file and section name resolves", check_citations),
    (
        "6 -- a section excluded by name is reported, and reaches a section",
        check_excluded,
    ),
)


class Outcome(NamedTuple):
    title: str
    ok: bool
    lines: list[str]
    findings: list[str]
    broken: list[str]


def run(result: Result, strict: bool) -> list[Outcome]:
    """Every check, in order, with its verdict in this mode.

    Every check this module defines has to be in ``CHECKS``. One that is
    written and not wired in cannot fail, and its silence reads as a pass,
    which is how check 10 of ``check_provenance.py`` went unrun in its first
    version; that gate refuses to run on the same condition.
    """
    defined = {
        name
        for name, fn in globals().items()
        if name.startswith("check_") and callable(fn)
    }
    wired = {fn.__name__ for _title, fn in CHECKS}
    if defined != wired:
        raise SystemExit(
            f"{len(defined)} checks defined and {len(wired)} wired into the run"
            f" ({', '.join(sorted(defined ^ wired))}): a check that is not called"
            " cannot fail, and its silence reads as a pass"
        )
    out = []
    for title, fn in CHECKS:
        findings, broken, lines = fn(result)
        out.append(
            Outcome(
                title, not broken and not (strict and findings), lines, findings, broken
            )
        )
    return out


def structure(result: Result) -> str:
    """The shape of the paper, in one line."""
    headed = [r for r in result.sections if r.section.key != PREAMBLE]
    return (
        f"{result.paper}: {len(headed)} sections at ## and ###,"
        f" {sum(1 for r in headed if r.section.source)} with a source line,"
        f" {sum(1 for r in headed if r.scope == EXCLUDED)} excluded by name"
    )


def gate(strict: bool) -> tuple[bool, list[str]]:
    """Check 14 of ``scripts/check_provenance.py``: the verdict and its lines."""
    result = check()
    outcomes = run(result, strict)
    counts = Counter(v.status for _r, v in _verdicts(result))
    lines = [
        f"      {structure(result)}",
        f"      {sum(counts.values())} figures checked: "
        + ", ".join(f"{counts[s]} {s}" for s in STATUSES),
    ]
    for o in outcomes:
        state = "passes" if o.ok else "fails"
        lines.append(f"        {state}, {len(o.findings)} finding(s): {o.title}")
        lines += [f"          BROKEN  {b}" for b in o.broken]
    if not strict and any(o.findings for o in outcomes):
        lines.append(
            "      findings are not a failure outside --strict, which is the paper's"
            " freeze condition; `scripts/check_paper.py --report` prints every one"
        )
    return all(o.ok for o in outcomes), lines


# --------------------------------------------------------------------------
# what is printed


def plain(result: Result, outcomes: list[Outcome], strict: bool) -> list[str]:
    """The default output: each check with its verdict, as the gate prints."""
    out = [f"checking {structure(result)}", ""]
    for o in outcomes:
        out.append(f"  {'ok  ' if o.ok else 'FAIL'}  {o.title}")
        out += [f"      {line}" for line in o.lines]
        out += [f"      BROKEN  {b}" for b in o.broken]
        if o.findings:
            shown = o.findings if strict else []
            out += [f"        {f}" for f in shown]
            if not strict:
                out.append(
                    f"      {len(o.findings)} finding(s), not a failure outside"
                    " --strict: --strict is the paper's freeze condition"
                )
        out.append("")
    failed = [o.title for o in outcomes if not o.ok]
    if failed:
        out.append(f"FAIL  {len(failed)} check(s) failed:")
        out += [f"        {t}" for t in failed]
    else:
        out.append(
            "paper clean: every figure is matched, listed or exempt"
            if not any(o.findings for o in outcomes)
            else "paper check passes outside --strict, with findings for the figure pass"
        )
    return out


def _table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    sep = ["---:" if a == "r" else "---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _kind(n: Number) -> str:
    if n.percent:
        return "percentages"
    if n.decimals:
        return "decimals"
    return "integers"


def _title(r: SectionResult) -> str:
    s = r.section
    if s.key == PREAMBLE:
        return PREAMBLE
    bare = STATUS_TAG.sub("", s.heading).strip()
    return f"§{bare}" if s.key.startswith("§") else bare


def report(result: Result, outcomes: list[Outcome], strict: bool) -> list[str]:
    """The work list: every figure with its status, section by section."""
    every = _verdicts(result)
    counts = Counter(v.status for _r, v in every)
    headed = [r for r in result.sections if r.section.key != PREAMBLE]
    excluded = [r for r in result.sections if r.scope == EXCLUDED]
    patterned = Counter(p for r in result.sections for _n, p in r.by_pattern)
    order = {r.section.key: k for k, r in enumerate(result.sections)}
    confined = [
        c
        for r in _checked(result)
        for c in r.citations
        if c.path and c.names and c.exists
    ]
    totals = [
        ["sections at `##` and `###`", str(len(headed))],
        ["of them with a source line", str(sum(1 for r in headed if r.section.source))],
        ["excluded by name", str(len(excluded))],
        ["file citations confined to the sections they name", str(len(confined))],
        ["figures in the checked sections", str(len(every))],
        *[
            [f"of them {kind}", str(n)]
            for kind, n in sorted(Counter(_kind(v.number) for _r, v in every).items())
        ],
        *[[f"*{s}*", str(counts[s])] for s in STATUSES],
        ["of the historical, listed", str(sum(1 for _r, v in every if v.listed))],
        [
            "of the exempt, also equal to a cited number by chance",
            str(sum(1 for _r, v in every if v.status == EXEMPT and v.match)),
        ],
        [
            "of the unmatched, in a cited file outside the cited sections",
            str(sum(1 for _r, v in every if v.status == UNMATCHED and v.outside)),
        ],
        [
            "figures in the excluded sections, not checked",
            str(sum(len(r.unchecked) for r in excluded)),
        ],
        *[[f"exempt by pattern: {p}", str(n)] for p, n in sorted(patterned.items())],
    ]
    out = [
        "# The paper's figures, section by section",
        "",
        f"Generated by `python scripts/check_paper.py --report` from `{result.paper}`,",
        f"sha256 `{result.sha256}`. Every count here is computed by the check.",
        "",
        "**How to read a row.** A figure with `k` decimals is *matched* when a file",
        "its section cites carries a number that rounds to it at `k` decimals, inside",
        "the section of that file the source line names where it names one; the",
        "file, the line and that number are printed, so a coincidental match can be",
        "seen. *historical* means it matched only in a frozen dated record, and it",
        "passes only if it is listed with a reason. *exempt* means it is computed",
        "from report figures and listed with its arithmetic, which is verified and",
        "printed with the line each operand was read from. *unmatched* is work.",
        "**Outside** names a cited file that carries an unmatched figure outside the",
        "sections the source line names, the usual sign of a citation pointing at",
        "the wrong section. **Also in** names files that carry it without being its",
        "section's sources: a lead for the figure pass and never a source, since a",
        "short figure collides by chance.",
        "",
        "## Totals",
        "",
        *_table(["", "count"], totals, "-r"),
        "",
        "**Matched, by the file they matched in.**",
        "",
    ]
    matched = Counter(v.source for _r, v in every if v.status == MATCHED)
    out += _table(
        ["file", "figures"],
        [[f"`{f}`", str(n)] for f, n in sorted(matched.items())],
        "-r",
    )
    out += ["", "**Historical, by section and record.**", ""]
    hist = Counter((r.section.key, v.source) for r, v in every if v.status == HISTORICAL)
    out += (
        _table(
            ["section", "record", "figures"],
            [
                [k, f"`{f}`", str(n)]
                for (k, f), n in sorted(hist.items(), key=lambda x: order[x[0][0]])
            ],
            "--r",
        )
        if hist
        else ["None."]
    )
    out += ["", "**Unmatched, by section.**", ""]
    unmatched = Counter(r.section.key for r, v in every if v.status == UNMATCHED)
    out += (
        _table(
            ["section", "figures"],
            [
                [k, str(n)]
                for k, n in sorted(unmatched.items(), key=lambda x: order[x[0]])
            ],
            "-r",
        )
        if unmatched
        else ["None."]
    )
    out += ["", "**What `--strict` fails on.**", ""]
    for o in outcomes:
        out.append(f"- check {o.title}: {len(o.findings)} finding(s)")
        out += [f"  - BROKEN in every mode: {b}" for b in o.broken]
    out += ["", "## Section by section", ""]
    for r in result.sections:
        out += _section(r)
    out += [
        f"## The checks, as `python scripts/check_paper.py{' --strict' if strict else ''}`"
        " prints them",
        "",
        "```",
        *plain(result, outcomes, strict),
        "```",
    ]
    return out


def _leads(files: tuple[str, ...]) -> str:
    """Where else an unmatched figure appears, the first few by name.

    A figure that turns up in many files is a short one colliding by chance,
    and the count says so without a row the width of the page.
    """
    if not files:
        return ""
    shown = ", ".join(f"`{p}`" for p in files[:4])
    more = f", and {len(files) - 4} more" if len(files) > 4 else ""
    return f"also in {len(files)} file(s): {shown}{more}"


def _others(v: Verdict) -> str:
    """The other numbers in scope a figure matches, after the one printed first.

    A reader who knows which row the sentence quotes can see whether the
    check found that row or another; the first few are named and the rest
    counted.
    """
    if not v.alternates:
        return ""
    shown = ", ".join(v.alternates[:3])
    more = f", and {len(v.alternates) - 3} more" if len(v.alternates) > 3 else ""
    return f" (also {shown}{more})"


def _section(r: SectionResult) -> list[str]:
    s = r.section
    out = [f"### {_title(r)}", "", f"Lines {s.start} to {s.end}."]
    if r.scope == EXCLUDED:
        return out + [
            "",
            f"**Excluded by name**: {r.excluded}. {len(r.unchecked)} figure(s) here are"
            f" not checked, and the section ships in `{PAPER}`.",
            "",
        ]
    if s.source:
        parts = []
        for c in r.citations:
            if c.path is None:
                parts.append(
                    "*" + "*, *".join(c.names) + "* with no file before it, NOT RESOLVED"
                    if c.names
                    else f'not a repository file, not checked: "{_cell(c.text.rstrip("."))}"'
                )
                continue
            names = []
            for n in c.names:
                if n in c.unresolved:
                    names.append(f"*{n}* NOT A HEADING THERE")
                    continue
                ranges = ", ".join(f"{a} to {b}" for m, a, b in c.spans if m == n)
                names.append(f"*{n}* lines {ranges}")
            parts.append(
                f"`{c.path}`"
                + ("" if c.exists else " DOES NOT EXIST")
                + (f" ({'; '.join(names)})" if names else " whole")
            )
        out += ["", "Sources: " + "; ".join(parts) + "."]
    elif r.scope == EVERY_LIVE_REPORT:
        why = next(e.why for e in PAPER_NO_SOURCE_SECTIONS if e.section == s.key)
        out += ["", f"No source line. Listed, and read against every live report: {why}."]
    elif r.verdicts:
        out += [
            "",
            "**No source line, and not listed**: nothing to read its figures against,"
            " so every one is unmatched and check 4 names the section.",
        ]
    else:
        out += ["", "No source line."]
    if r.by_pattern:
        out += [
            "",
            "Exempt by pattern: "
            + ", ".join(f"{n.figure} ({p})" for n, p in r.by_pattern)
            + ".",
        ]
    if r.broken:
        out += ["", "**Broken exemption entries**: " + "; ".join(r.broken) + "."]
    if not r.verdicts:
        return out + ["", "No figures.", ""]
    rows = []
    for v in r.verdicts:
        if v.status in (MATCHED, HISTORICAL):
            where = f"`{v.source}`:{v.match.line} `{v.match.written}`" + _others(v)
            if v.status == HISTORICAL:
                where += ", listed" if v.listed else ", **not listed**"
        elif v.status == EXEMPT:
            where = v.arithmetic
            if v.match is not None:
                where += (
                    f"; also `{v.source}`:{v.match.line} `{v.match.written}`"
                    + _others(v)
                    + ", by chance"
                )
        else:
            where = "; ".join(
                p
                for p in (
                    "outside the cited sections: " + ", ".join(v.outside)
                    if v.outside
                    else "",
                    _leads(v.elsewhere),
                )
                if p
            )
        rows.append([str(v.number.line), v.number.written, f"*{v.status}*", _cell(where)])
    return out + ["", *_table(["line", "figure", "status", "where"], rows, "rr--"), ""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        action="store_true",
        help="print every figure with its status, section by section, and the"
        " totals: the figure pass's work list",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="the paper's freeze condition: an unmatched figure, an unlisted"
        " historical one, an unlisted section with figures and no source line, or"
        " a citation that does not resolve fails",
    )
    args = ap.parse_args(argv)
    result = check()
    outcomes = run(result, args.strict)
    lines = (
        report(result, outcomes, args.strict)
        if args.report
        else plain(result, outcomes, args.strict)
    )
    print("\n".join(lines))
    return 1 if any(not o.ok for o in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
