#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prove that every number in this repository is categorised.

The standing rule is that every number reaching a report, a note or the paper
is computed by the run that prints it, quoted from a named source with that
source beside it, or marked an estimate with an instrument and an owner. There
is no fourth kind. ``taxiout.provenance`` is the register; this is the gate that
makes it true rather than aspirational.

**The property being bought is narrow and worth stating exactly.** It is not
that every number is right. It is that **no number is uncategorised**: adding a
constant without classifying it fails this check, and that is the property the
repository did not have when three provenance defects reached a committed tree.

The scanner, and why it is an ``ast`` walk
------------------------------------------
``module_constants``, ``string_literals`` and ``comments`` are exported and are
used by **both** the checks and ``--census``, so two counts of one quantity
cannot disagree. That is the same reason ``scripts/preregistered.py`` sums its
substitution counts from its own manifest.

A line-anchored regex was tried first, in a census outside the repository, and
it is wrong in five ways that an ``ast`` walk closes: it reads only the first
line of a value, so numbers in a dict or tuple spanning lines are unseen; it
misses constants with lowercase names; it misses numeric defaults in dataclass
fields; it counts an assignment once however many literals it holds; and it
counts assignments whose value is a string that merely contains a digit. On this
tree the regex matched 255 assignments and this walk finds 275, adding 51 and
over-counting 31. ``--census`` prints both.

Comments are not in the ``ast`` at all, so they come from ``tokenize``, which is
exact where a line-starts-with-hash rule would catch hashes inside strings.
Prose figures come from ``ast.Constant`` of type ``str`` and from the literal
parts of an ``ast.JoinedStr``, so a format field is excluded for free rather
than by stripping braces with a regex.

**The scanner enumerates; it does not classify.** A number is ``mechanism``,
``chosen``, ``derived`` or ``quoted`` because a person said so in the register.
A class guessed from a name or a comment would be exactly the unearned
provenance this gate exists to remove.

Scope, stated so silence is not read as a pass
----------------------------------------------
``module_constants`` walks ``Module.body`` and the bodies of module-level
classes, which is what puts a dataclass field default in scope. It does not walk
function signature defaults: those are parameters of a call rather than
constants of a module. ``--census`` counts them anyway and prints the count.
``taxiout.provenance.SCOPE`` carries the whole of this and check 0 prints it.

The checks
----------
0. what the register does and does not reach, printed rather than assumed
1. every constant the scanner finds is categorised
2. every registered value still matches the code
3. every ``chosen`` constant has a basis
4. the authored artifact stays out of every model figure
5. the frozen documents are unchanged
6. ``PROVENANCE_PENDING``, printed in full; ``--strict`` fails on a non-empty one
7. no decimal figure is hardcoded in a generator's prose
8. every declared report input still has the digest it was read at
9. a report claims every generator that writes into it
10. the generated document matches the manifests
11. a figure quoted beside a report's name is in that report
12. every line a basis cites says what the basis says
13. no generator of a report imports a fetcher
14. every figure in the paper is in a file its section cites, by
    ``scripts/check_paper.py``

``--strict`` is the freeze condition and Block 11 runs it. The ordinary run
prints check 6's open items, check 8's unverifiable inputs and check 14's
findings on the paper, and passes, because a gate that fails today cannot be
wired into a commit path.

Usage
-----
    python scripts/check_provenance.py
    python scripts/check_provenance.py --census
    python scripts/check_provenance.py --strict
    make provenance
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import check_paper as paper  # noqa: E402
from taxiout.provenance import (  # noqa: E402
    AUTHORED_ARTIFACTS,
    REGISTER_MODULE,
    CHOSEN,
    DERIVED,
    MECHANISM,
    QUOTED,
    CLASSES,
    CONSTANTS,
    FROZEN_DOCUMENTS,
    MODEL_FIGURE_GENERATORS,
    NO_BASIS,
    OSM_PINS,
    OSM_PINS_BASIS,
    PAPER,
    PAPER_EXCLUDED_SECTIONS,
    PAPER_EXEMPT_PATTERNS,
    PAPER_FIGURE_EXEMPTIONS,
    PAPER_HISTORICAL_FIGURES,
    PAPER_INTEGER_MIN_DIGITS,
    PAPER_NO_SOURCE_SECTIONS,
    PROVENANCE_PENDING,
    REBUILD_PINS,
    REPORT_CITATION_EXEMPTIONS,
    REPORT_INPUTS_DECLARED,
    REPORTS_WITHOUT_FOOTER,
    SCOPE,
    VALUE_NOT_JUSTIFIED,
    digest,
    parse_report_inputs,
)

# Where the register's remit runs. Anything else is out of scope by
# ``SCOPE`` and not by omission.
SCANNED_ROOTS: tuple[str, ...] = ("src/", "scripts/")

# The one module registered whole rather than constant by constant.
AUTHORED_MODULES: frozenset[str] = frozenset(a.module for a in AUTHORED_ARTIFACTS)

# Modules check 1 does not require an entry for, and the two reasons are
# different. The authored artifact is registered whole because there is no
# provenance question to answer about an invented stand's coordinates. The
# register is excluded because it cannot be an entry in itself: its aggregate
# assignments hold every registered literal in the repository. Both are named in
# ``SCOPE`` and check 0 prints them.
UNREGISTERED_MODULES: frozenset[str] = AUTHORED_MODULES | {REGISTER_MODULE}

# A decimal figure in prose: a digit, a point, a digit. Deliberately not bare
# integers -- a report says "the three airports" and "five folds" in words a
# gate cannot tell from a measurement, and a check that cried wolf on those
# would be turned off. A measurement carries a decimal point.
PROSE_FIGURE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})*\.\d+(?![\w.])")

# Prose figures that are definitions rather than measurements, each with its
# reason. A band label is a definition: nobody measured 1.5 km, it is where the
# band edge was put, and the constant it comes from is in the register.
PROSE_EXEMPT: tuple[tuple[str, str, str], ...] = (
    # A whole module, where every prose figure in it is the same kind of thing.
    # ``"*"`` as the figure means the module; anything narrower would be a list
    # that grew by one every time the authored field gained a worked route.
    (
        "src/taxiout/synthetic.py",
        "*",
        "the authored field's own geometry, hand-derived and registered whole in"
        " `taxiout.provenance.AUTHORED_ARTIFACTS`. These figures ARE the"
        " definition of the fixture, there is no run that could produce them,"
        " and check 4 proves no model figure reads them",
    ),
    (
        "scripts/report_synthetic_airport.py",
        "*",
        "the authored field's report, printing the same hand-derived geometry for"
        " the same reason",
    ),
    (
        "scripts/preregistered.py",
        "*",
        "its report is `reports/preregistered-2026-09-20.md`, a FROZEN dated"
        " record in `taxiout.provenance.FROZEN_DOCUMENTS`, and its prose figures"
        " are the pre-registration's own references and floors, each registered"
        " as a `quoted` constant in the same module."
        " `preregistered.substitution_audit` aligns the carried text against the"
        " dated note line by line and rebuilds it byte for byte, which is a"
        " stronger instrument than this check",
    ),
    (
        "src/taxiout/provenance.py",
        "*",
        "the register. Every figure in it is a basis, and a basis names its"
        ' source: "scored 214.90 s on October against 217.28 s" IS the'
        " provenance of `BOOSTER_PARAMS`, not an unsourced figure in a report."
        " The module writes no report, so no figure in it can reach one, which"
        " is the same reason docstrings and comments are out of this check",
    ),
    (
        "scripts/check_provenance.py",
        "*",
        "this gate. Its prose figures are the entries of the exemption table"
        " below, which is a list of figures by construction, and the two"
        " PENDING reasons that quote the prose they are about. It writes no"
        " report either",
    ),
    # Single figures, each a definition rather than a measurement.
    (
        "scripts/report_residual.py",
        "1.5",
        "a visibility band label in kilometres: the definition"
        " `taxiout.metar.VIS_LOW_M` fixes, which is registered, and not a"
        " measurement",
    ),
    (
        "scripts/review_challenge_data.py",
        "1.5",
        "the same visibility band label",
    ),
    (
        "scripts/report_congestion.py",
        "1.5",
        "the same visibility band label, in that report's weather tables",
    ),
    (
        "scripts/report_tails.py",
        "1.0",
        "the ODbL version in the data credit, not a measurement",
    ),
    (
        "src/taxiout/osm.py",
        "0.1",
        "this client's version in its user-agent string, not a measurement",
    ),
    (
        "scripts/check_provenance.py",
        "4.345",
        "this gate's own comment names the missing audit reading it looks for",
    ),
    # Real prose measurements, unfixed because their reports cost a
    # regeneration this block does not spend. Each is on PROVENANCE_PENDING,
    # the same arrangement check 3 uses for an unrecorded basis: recorded as an
    # excursion rather than passed off as fine.
    (
        "scripts/build_taxi_graph.py",
        "0.1",
        'PENDING: "within 0.1 deg of the same physical axis" at LTFM is a'
        " measurement stated in prose. Computable from the runway geometry the"
        " same run holds; fixing it regenerates `reports/taxi_routing.md`",
    ),
    (
        "scripts/review_challenge_data.py",
        "0.02",
        'PENDING: "the same rate to within 0.02 points" restates a threshold'
        " the code writes as `0.0002` inside a function, so the figure and the"
        " threshold are two copies of one number. The fix promotes the threshold"
        " to a registered constant and formats the prose from it, and"
        " regenerates `reports/data_review.md`",
    ),
)


# String-valued assignments exempted whole from check 7, each with a reason.
# A figure inside one of these is not a figure a run produced: it is text this
# repository carries deliberately and checks by another instrument.
PROSE_EXEMPT_ASSIGNMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "scripts/preregistered.py",
        "PREREGISTRATION",
        "the pre-registration's own carried text, quoted from a dated note."
        " Its figures are the record and must not move, and"
        " `preregistered.substitution_audit` already aligns it against the note"
        " line by line and rebuilds it byte for byte, which is a stronger check"
        " than this one",
    ),
)


class Assignment(NamedTuple):
    """One module-level numeric assignment the scanner found."""

    module: str
    line: int
    name: str
    literals: tuple[float, ...]


class Literal(NamedTuple):
    """One string literal or comment the scanner found."""

    module: str
    line: int
    text: str


# --------------------------------------------------------------------------
# the scanner


def tracked_files() -> list[str]:
    """Every file git tracks, repo-relative.

    From ``git ls-files`` the way ``check_licence_hygiene.py`` gets its list:
    untracked scratch must not pollute a count, and nothing tracked may be
    missed. NUL-delimited, so a path with a space stays one path.
    """
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"GIT_OPTIONAL_LOCKS": "0", "PATH": "/usr/bin:/bin"},
    )
    if proc.returncode != 0:
        raise SystemExit("git ls-files failed; this gate reads tracked files only")
    return sorted(p for p in proc.stdout.split("\0") if p)


def scanned_modules() -> list[str]:
    """The tracked Python files the register covers."""
    return [
        f
        for f in tracked_files()
        if f.endswith(".py") and any(f.startswith(r) for r in SCANNED_ROOTS)
    ]


def _numeric(node: ast.AST) -> tuple[float, ...]:
    """Every int or float literal anywhere in ``node``, in source order.

    Walks the whole value, so a number inside a tuple, list, dict, set or call
    argument is found. ``ast.walk`` is breadth-first, so the nodes are sorted by
    position to make the order the source's and not the traversal's.
    """
    found = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float)
    ]
    found.sort(key=lambda n: (n.lineno, n.col_offset))
    return tuple(n.value for n in found)


def _targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    if isinstance(node, ast.Assign):
        return [t.id if isinstance(t, ast.Name) else ast.unparse(t) for t in node.targets]
    target = node.target
    return [target.id if isinstance(target, ast.Name) else ast.unparse(target)]


def module_constants(modules: list[str] | None = None) -> tuple[Assignment, ...]:
    """Every module-level numeric assignment in the scanned tree.

    ``Module.body`` and the bodies of module-level classes, the second so that a
    dataclass field default is in scope. Function signature defaults are not:
    see the module docstring and ``taxiout.provenance.SCOPE``.
    """
    out: list[Assignment] = []
    for module in modules if modules is not None else scanned_modules():
        tree = ast.parse((REPO / module).read_text(encoding="utf-8"))

        def take(body, prefix: str = "", module=module) -> None:
            for node in body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    if node.value is None:
                        continue
                    lits = _numeric(node.value)
                    if lits:
                        for name in _targets(node):
                            out.append(
                                Assignment(module, node.lineno, prefix + name, lits)
                            )
                elif isinstance(node, ast.ClassDef):
                    take(node.body, prefix=f"{node.name}.")

        take(tree.body)
    return tuple(sorted(out, key=lambda a: (a.module, a.line, a.name)))


def function_defaults(modules: list[str] | None = None) -> tuple[Assignment, ...]:
    """Numeric defaults in function signatures: counted, not registered.

    Out of the register's scope by ``SCOPE``, and counted here so that the
    exclusion is a stated number rather than a silence.
    """
    out: list[Assignment] = []
    for module in modules if modules is not None else scanned_modules():
        tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default in defaults:
                lits = _numeric(default)
                if lits:
                    out.append(Assignment(module, node.lineno, node.name, lits))
    return tuple(out)


def _docstring_ids(tree: ast.AST) -> set[int]:
    """The ids of every docstring node in ``tree``.

    A docstring is prose ABOUT the code and cannot reach a report, which is the
    same reason ``SCOPE`` gives for leaving comments to the census: a stale
    figure in one misleads a reader of the source and nothing else. Identified
    structurally, as the first statement of a module, class or function, rather
    than by guessing from position.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ids.add(id(body[0].value))
    return ids


def _exempt_assignment_ids(tree: ast.AST, module: str) -> set[int]:
    """The ids of literals belonging to an assignment exempted whole."""
    names = {n for m, n, _why in PROSE_EXEMPT_ASSIGNMENTS if m == module}
    if not names:
        return set()
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            if set(_targets(node)) & names:
                ids |= {
                    id(n)
                    for n in ast.walk(node.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                }
    return ids


def string_literals(
    modules: list[str] | None = None, *, prose_only: bool = False
) -> tuple[Literal, ...]:
    """Every string literal in the scanned tree, format fields excluded.

    An ``ast.JoinedStr`` contributes only its literal ``Constant`` parts, so a
    figure arriving through a format field is excluded for free rather than by
    stripping braces with a regex afterwards. Implicitly concatenated and
    multi-line strings come out whole, which is what a report's prose is.

    ``prose_only`` drops docstrings and the literals of assignments exempted
    whole in ``PROSE_EXEMPT_ASSIGNMENTS``, which is the surface check 7 reads:
    prose a run can write into a report. The census reads the whole set.
    """
    out: list[Literal] = []
    for module in modules if modules is not None else scanned_modules():
        tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
        skip: set[int] = set()
        if prose_only:
            skip = _docstring_ids(tree) | _exempt_assignment_ids(tree, module)
        # An f-string's literal parts are ``Constant`` nodes too, so they are
        # collected once through the ``JoinedStr`` and skipped by identity in
        # the general branch. Counting them twice would inflate every figure
        # this gate reports about itself.
        inside = {
            id(part)
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            for part in node.values
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if (
                        isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                        and id(part) not in skip
                    ):
                        out.append(Literal(module, node.lineno, part.value))
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in inside
                and id(node) not in skip
            ):
                out.append(Literal(module, node.lineno, node.value))
    return tuple(out)


def comments(modules: list[str] | None = None) -> tuple[Literal, ...]:
    """Every comment in the scanned tree, by ``tokenize``.

    Comments are not in the ``ast``. ``tokenize`` is stdlib and exact, where a
    rule matching lines that start with a hash would catch a hash inside a
    string literal and miss a trailing comment.
    """
    out: list[Literal] = []
    for module in modules if modules is not None else scanned_modules():
        text = (REPO / module).read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                out.append(Literal(module, token.start[0], token.string))
    return tuple(out)


# --------------------------------------------------------------------------
# the checks


def _key(module: str, name: str) -> tuple[str, str]:
    return (module, name)


def check_scope() -> list[str]:
    """Check 0: what the register reaches, and what it does not."""
    return [f"  --  {what}: {why}" for what, why in SCOPE]


def check_categorised(found: tuple[Assignment, ...]) -> tuple[bool, list[str]]:
    """Check 1: every constant the scanner finds is in exactly one state."""
    registered = {_key(c.module, c.name) for c in CONSTANTS}
    missing = [
        a
        for a in found
        if a.module not in UNREGISTERED_MODULES
        and _key(a.module, a.name) not in registered
    ]
    scanned = {_key(a.module, a.name) for a in found}
    stale = sorted(k for k in registered - scanned)
    counts = Counter(c.cls for c in CONSTANTS)
    lines = [
        f"      {len(found)} assignments found, "
        f"{len([a for a in found if a.module in AUTHORED_MODULES])} in the authored"
        " artifact, "
        f"{len([a for a in found if a.module == REGISTER_MODULE])} in the register"
        " itself, "
        f"{len(CONSTANTS)} registered",
        "      by class: " + ", ".join(f"{cls} {counts.get(cls, 0)}" for cls in CLASSES),
    ]
    if missing:
        lines.append(f"      {len(missing)} constant(s) are in no state at all:")
        lines += [f"        {a.module}:{a.line} {a.name} = {a.literals}" for a in missing]
    if stale:
        lines.append(f"      {len(stale)} register entr(ies) name no constant:")
        lines += [f"        {m} {n}" for m, n in stale]
    return (not missing and not stale), lines


def check_values(found: tuple[Assignment, ...]) -> tuple[bool, list[str]]:
    """Check 2: every registered value still matches the code."""
    actual = {_key(a.module, a.name): a for a in found}
    drifted = []
    for c in CONSTANTS:
        got = actual.get(_key(c.module, c.name))
        if got is None:
            continue
        if tuple(got.literals) != tuple(c.literals):
            drifted.append((c, got))
    lines = [f"      {len(CONSTANTS) - len(drifted)} of {len(CONSTANTS)} match the code"]
    for c, got in drifted:
        lines += [
            f"        {c.module} {c.name}",
            f"          register: {tuple(c.literals)}",
            f"          code:     {tuple(got.literals)}",
        ]
    return (not drifted), lines


def check_basis() -> tuple[bool, list[str]]:
    """Check 3: every ``chosen`` constant has a basis."""
    chosen = [c for c in CONSTANTS if c.cls == CHOSEN]
    empty = [c for c in chosen if not c.basis.strip()]
    unrecorded = [c for c in chosen if c.basis.startswith(NO_BASIS)]
    unjustified = [c for c in chosen if c.basis.startswith(VALUE_NOT_JUSTIFIED)]
    lines = [
        f"      {len(chosen)} chosen constants, {len(empty)} with an empty basis",
        f"      {len(unrecorded)} record no reason at all,"
        f" {len(unjustified)} record the method or a neighbouring use and not"
        f" the value, {len(chosen) - len(unrecorded) - len(unjustified)} carry"
        " an ordinary basis",
    ]
    for c in empty:
        lines.append(f"        EMPTY  {c.module} {c.name}")
    if unrecorded:
        lines.append(
            "      no reason recorded anywhere in the code, the docs, the"
            " reports or the notes. **This count is meant to be non-zero.** A"
            " constant whose basis says what it does and that its reason was"
            " never written down is registered, which is one of the four"
            " states; driving it to zero would mean writing plausible reasons"
            " after the fact for thresholds nobody argued, which is the failure"
            " this register exists to prevent:"
        )
        lines += [f"        {c.module} {c.name}" for c in unrecorded]
    return (not empty), lines


def check_authored() -> tuple[bool, list[str]]:
    """Check 4: the authored artifact stays out of every model figure.

    An import fails it. A prose citation or a comment does not: naming
    ``reports/synthetic_airport.md`` in a sentence is how the verification
    fixture earns its place, and a comment recording what a probe measured is
    not a data path.
    """
    lines = []
    bad = []
    for artifact in AUTHORED_ARTIFACTS:
        module_name = (
            artifact.module.removeprefix("src/").removesuffix(".py").replace("/", ".")
        )
        for generator in MODEL_FIGURE_GENERATORS:
            path = REPO / generator
            if not path.exists():
                bad.append(f"{generator} does not exist; this list is stale")
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(
                    n == module_name or n.startswith(module_name + ".") for n in names
                ):
                    bad.append(
                        f"{generator}:{node.lineno} imports {module_name}"
                        f" -- {artifact.condition}"
                    )
        lines.append(
            f"      {artifact.module}: {len(MODEL_FIGURE_GENERATORS)} model-figure"
            f" generators checked, condition is that {artifact.condition}"
        )
    lines += [f"        {b}" for b in bad]
    return (not bad), lines


def check_frozen() -> tuple[bool, list[str]]:
    """Check 5: the frozen documents are unchanged."""
    lines = []
    bad = []
    for f in FROZEN_DOCUMENTS:
        got = digest(REPO / f.path)
        if got == "absent":
            bad.append(f"{f.path} is missing, and it is a dated record")
        elif got != f.sha256:
            bad.append(
                f"{f.path} has changed. It is a DATED RECORD: its value is in its"
                " date, additions to it are additive and dated, and nothing"
                " already in it is revised. If the change is an addition, update"
                " the digest in taxiout.provenance.FROZEN_DOCUMENTS in the same"
                f" commit and say so in the message. ({f.why})"
            )
    lines.append(f"      {len(FROZEN_DOCUMENTS)} dated records, {len(bad)} changed")
    lines += [f"        {b}" for b in bad]
    return (not bad), lines


def check_pending(strict: bool) -> tuple[bool, list[str]]:
    """Check 6: what is still open. ``--strict`` fails on a non-empty list."""
    unrecorded = sum(1 for c in CONSTANTS if c.basis.startswith(NO_BASIS))
    lines = [f"      {len(PROVENANCE_PENDING)} open item(s)"]
    for p in PROVENANCE_PENDING:
        extra = (
            f" [{unrecorded} constants]" if "basis is recorded nowhere" in p.what else ""
        )
        lines += [f"        - {p.what}{extra}", f"          owner: {p.owner}"]
    if PROVENANCE_PENDING and not strict:
        lines.append(
            "      not a failure outside --strict: --strict is the freeze"
            " condition, and a gate that fails today cannot gate a commit"
        )
    return (not (strict and PROVENANCE_PENDING)), lines


def check_prose(literals: tuple[Literal, ...]) -> tuple[bool, list[str]]:
    """Check 7: no decimal figure is hardcoded in a generator's prose."""
    exempt = {(m, fig) for m, fig, _why in PROSE_EXEMPT}
    exempt_modules = {m for m, fig, _why in PROSE_EXEMPT if fig == "*"}
    hits = []
    for lit in literals:
        if lit.module in exempt_modules:
            continue
        for m in PROSE_FIGURE.finditer(lit.text):
            if (lit.module, m.group(0)) in exempt:
                continue
            hits.append((lit.module, lit.line, m.group(0), lit.text.strip()[:70]))
    lines = [
        f"      {len(literals)} prose literals scanned (docstrings and"
        f" {len(PROSE_EXEMPT_ASSIGNMENTS)} exempted assignment(s) excluded),"
        f" {len(PROSE_EXEMPT)} figure(s) exempt with a reason,"
        f" {len(hits)} hardcoded figure(s)"
    ]
    for module, line, fig, text in hits:
        lines.append(f"        {module}:{line} {fig!r} in {text!r}")
    return (not hits), lines


def check_report_inputs(strict: bool) -> tuple[bool, list[str]]:
    """Check 8: every declared report input still has the digest it was read at.

    Three outcomes, never collapsed into two. **verified**: present and
    matching. **stale**: present and different, which fails. **unverifiable**:
    absent, the ordinary case on a clean checkout because the data this
    repository does not redistribute is not there, printed as unverifiable and
    never passed off as verified, and failed under ``--strict``.
    """
    lines = []
    stale: list[str] = []
    unverifiable: list[str] = []
    declared_reports = {r for _g, r in REPORT_INPUTS_DECLARED}
    listed = {u.report for u in REPORTS_WITHOUT_FOOTER}
    frozen = {f.path for f in FROZEN_DOCUMENTS}
    silent = sorted(set(report_writers()) - declared_reports - listed - frozen)
    for generator, report in REPORT_INPUTS_DECLARED:
        path = REPO / report
        if not path.exists():
            unverifiable.append(f"{report} is not built here")
            lines.append(f"      {report}: unverifiable, the report is not built")
            continue
        declared = parse_report_inputs(path.read_text(encoding="utf-8"))
        if not declared:
            stale.append(
                f"{report} declares no inputs, and {generator} should write them"
            )
            continue
        verified = 0
        for rel, was in declared:
            now = digest(REPO / rel)
            if now == "absent":
                unverifiable.append(f"{report}: {rel} is absent")
            elif was == "absent":
                unverifiable.append(
                    f"{report}: {rel} was absent when the report was written"
                )
            elif now == was:
                verified += 1
            else:
                stale.append(f"{report}: {rel} has changed since the report was written")
        lines.append(
            f"      {report}: {len(declared)} declared input(s),"
            f" {verified} verified,"
            f" {len([u for u in unverifiable if u.startswith(report)])} unverifiable,"
            f" {len([s for s in stale if s.startswith(report)])} stale"
        )
    lines += [f"        STALE  {s}" for s in stale]
    lines += [f"        unverifiable: {u}" for u in unverifiable]
    lines.append(
        f"      {len(declared_reports)} report(s) declare their inputs,"
        f" {len(listed)} are listed as carrying no footer with a cost and an"
        f" owner, {len(silent)} are neither"
    )
    lines += [
        f"        {u.report}: no footer -- {u.cost} ({u.owner})"
        for u in REPORTS_WITHOUT_FOOTER
    ]
    lines += [
        f"        SILENT  {s} has no footer and is on no list: a report skipped"
        " without a reason is indistinguishable from one that was forgotten"
        for s in silent
    ]
    if unverifiable and not strict:
        lines.append(
            "      unverifiable is not a failure outside --strict: data/ is"
            " gitignored, so a clean checkout cannot verify a data digest"
        )
    return (not stale and not silent and not (strict and unverifiable)), lines


def report_writers() -> dict[str, set[str]]:
    """Which generators write into which report, structurally.

    **Not "which script mentions the file".** A generator cites other reports in
    prose all the time, and a check built on mentions would flag every citation
    and miss nothing useful -- the failure mode the cross-report tier is being
    deferred to avoid. So: a module-level assignment whose value is a path
    expression ending in a ``"<name>.md"`` literal with a ``"reports"``
    component, in a module that calls ``write_text`` somewhere. That is what
    writing a report looks like in this repository, in every generator.
    """
    writers: dict[str, set[str]] = {}
    for module in scanned_modules():
        tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
        writes = any(
            isinstance(n, ast.Attribute) and n.attr == "write_text"
            for n in ast.walk(tree)
        )
        if not writes:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            parts = [
                n.value
                for n in ast.walk(node.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            ]
            if "reports" in parts:
                for part in parts:
                    if part.endswith(".md"):
                        writers.setdefault(f"reports/{part}", set()).add(module)
    return writers


def check_one_generator_per_claim() -> tuple[bool, list[str]]:
    """Check 9: a report claims every generator that writes into it.

    ``reports/tails.md`` claimed one generator and "every figure below is
    computed there" while carrying a section written by another, which was false
    from the commit that gave it a second generator and which no figure moved to
    announce. A report that names one generator must not carry a section from
    another.
    """
    writers = report_writers()
    bad = []
    checked = 0
    for report, wrote in sorted(writers.items()):
        path = REPO / report
        if not path.exists():
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        claimed = set(re.findall(r"`(scripts/[a-z_0-9]+\.py)`", text))
        unclaimed = sorted(wrote - claimed)
        if unclaimed:
            bad.append(
                f"{report} does not name {', '.join(unclaimed)}, which write(s) into it"
            )
    lines = [
        f"      {checked} report(s) checked against the generators that write"
        f" them, {sum(len(w) for w in writers.values())} (report, generator)"
        f" pair(s), {len(bad)} mismatched"
    ]
    for report, wrote in sorted(writers.items()):
        if len(wrote) > 1:
            lines.append(
                f"      {report} has {len(wrote)} generators: " + ", ".join(sorted(wrote))
            )
    lines += [f"        {b}" for b in bad]
    return (not bad), lines


# --------------------------------------------------------------------------
# checks 11, 12 and 13


# A report cites another by filename. The citation is scoped to the sentence it
# sits in, because a paragraph drags in figures that belong to other sentences
# and a whole file drags in everything: the first instrument written for check 9
# in this file's own history flagged eight reports and meant nothing, and the
# lesson was that the scope has to be the unit a person would call the claim.
_SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z*`"\[(])')
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d+\.)\s")
# A decimal figure, and only a decimal figure. An integer in prose is a year, a
# count, a section number or an airport's runway far more often than it is a
# measurement, and the class this check exists for is the one that bit: the
# oracle ceiling at Heathrow on July moved 9.9 to 9.5 under a regeneration.
_DECIMAL = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})*\.\d+|\d+\.\d+)(?![\w.])")


def _units(text: str):
    """(line number, text) for every unit a citation can be scoped to.

    A table row scopes to its own cell, a heading and a list item to
    themselves, and ordinary prose to the sentence. Anything coarser makes the
    check flag figures no one claimed came from the cited file.
    """
    para: list[str] = []
    start = 0

    def flush():
        nonlocal para
        if not para:
            return []
        out = [(start, s) for s in _SENTENCE.split(" ".join(para))]
        para = []
        return out

    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("|") or s.startswith("#") or _LIST_ITEM.match(s):
            yield from flush()
            if s.startswith("|"):
                for cell in s.strip("|").split("|"):
                    yield i, cell.strip()
            elif s.startswith("#"):
                yield i, s
            elif _LIST_ITEM.match(s):
                start = i
                para = [s]
            continue
        if not para:
            start = i
        para.append(s)
    yield from flush()


def citing_documents() -> list[str]:
    """The documents whose citations of a report are checked.

    Every report that is not a frozen dated record, and ``README.md``. **A
    frozen record is excluded and the exclusion is the point**: its figures were
    true on its own date and the file it cites has moved on since, which is what
    makes it a record rather than a claim about the tree, and check 5 holds it
    to its digest instead. ``docs/paper.md`` is not here either: it names its
    sources once per section rather than in the sentence, and check 14 reads it
    that way.
    """
    frozen = {f.path for f in FROZEN_DOCUMENTS}
    reports = sorted(
        p.as_posix()
        for p in (REPO / "reports").glob("*.md")
        if f"reports/{p.name}" not in frozen
    )
    return [Path(p).relative_to(REPO).as_posix() for p in reports] + ["README.md"]


def check_report_citations() -> tuple[bool, list[str]]:
    """Check 11: a figure quoted beside a report's name is in that report.

    Where a document cites a report by filename, every decimal figure in the
    citation's own sentence must appear in the cited file. Exemptions are in
    ``REPORT_CITATION_EXEMPTIONS`` with a reason each and are counted, never
    skipped silently.
    """
    names = sorted(p.name for p in (REPO / "reports").glob("*.md"))
    cite = re.compile(rf"(?:reports/)?({'|'.join(re.escape(n) for n in names)})")
    exempt = {
        (e.source, e.cited, f): e.why
        for e in REPORT_CITATION_EXEMPTIONS
        for f in e.figures
    }
    citations = with_figure = verified = 0
    reached: set[tuple[str, str, str]] = set()
    stale: list[str] = []
    for doc in citing_documents():
        path = REPO / doc
        own = Path(doc).name
        for lineno, unit in _units(path.read_text(encoding="utf-8")):
            cited = sorted({m.group(1) for m in cite.finditer(unit)} - {own})
            if not cited:
                continue
            citations += 1
            figures = _DECIMAL.findall(unit)
            if not figures:
                continue
            with_figure += 1
            for target in cited:
                body = (REPO / "reports" / target).read_text(encoding="utf-8")
                for fig in figures:
                    key = (doc, f"reports/{target}", fig)
                    if re.search(rf"(?<![\w.]){re.escape(fig)}(?![\w.])", body):
                        verified += 1
                    elif key in exempt:
                        reached.add(key)
                    else:
                        stale.append(
                            f"{doc}:{lineno} quotes {fig} beside"
                            f" `{target}`, which does not carry it"
                        )
    unused = sorted(f"{s} -> {c} {g}" for (s, c, g) in set(exempt) - reached)
    lines = [
        f"      {len(citing_documents())} citing documents,"
        f" {citations} citations of a report by name,"
        f" {with_figure} carrying a decimal figure",
        f"      {verified} figure(s) found in the cited report,"
        f" {len(reached)} exempt with a reason, {len(stale)} unresolved",
    ]
    lines += [f"        STALE  {s}" for s in stale]
    lines += [
        f"        UNREACHED exemption  {u}: an exemption for a citation that no"
        " longer exists is a stale record of its own"
        for u in unused
    ]
    return (not stale and not unused), lines


def check_register_citations() -> tuple[bool, list[str]]:
    """Check 12: every line a basis cites says what the basis says it says.

    A basis that names a line is a citation, and a citation drifts the moment
    anything above it is edited. **This is not hypothetical**: the commit that
    created this register edited two of the generators it registers and left
    sixteen of its own line citations behind, pointing at blank lines and at
    unrelated code, and nothing noticed because nothing looked.

    The rule: the cited span either covers the assignment, or is a comment whose
    block reaches the assignment with no nearer comment block in between. A
    citation into another file is checked for existing and for not being blank,
    which is all a line number in prose can be checked for.
    """
    cite = re.compile(r"\blines? (\d+)(?: to (\d+))?")
    fileref = re.compile(r"`([^`]+\.(?:py|md))`")
    same = cross = 0
    broken: list[str] = []
    cache: dict[str, list[str]] = {}

    def lines_of(rel: str) -> list[str] | None:
        if rel not in cache:
            p = REPO / rel
            if not p.exists():
                return None
            cache[rel] = p.read_text(encoding="utf-8").splitlines()
        return cache[rel]

    def is_comment(ls: list[str], n: int) -> bool:
        return 1 <= n <= len(ls) and ls[n - 1].lstrip().startswith("#")

    for c in CONSTANTS:
        for m in cite.finditer(c.basis):
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            refs = fileref.findall(c.basis[: m.start()])
            target = refs[-1] if refs else c.module
            if target.startswith("taxiout."):
                target = "src/" + target.replace(".", "/") + ".py"
            ls = lines_of(target)
            if ls is None:
                broken.append(f"{c.module} {c.name}: cites {target}, which is not a file")
                continue
            if hi > len(ls):
                broken.append(
                    f"{c.module} {c.name}: cites {target} line {hi}, past its"
                    f" {len(ls)} lines"
                )
                continue
            if target != c.module:
                cross += 1
                if not any(ls[n - 1].strip() for n in range(lo, hi + 1)):
                    broken.append(
                        f"{c.module} {c.name}: cites {target} line {lo}, which is blank"
                    )
                continue
            same += 1
            assign = re.compile(rf"^{re.escape(c.name)}\s*[:=]")
            at = next((k for k, s in enumerate(ls, 1) if assign.match(s)), None)
            if at is None:
                broken.append(
                    f"{c.module} {c.name}: no assignment found to anchor the citation"
                )
                continue
            if lo <= at <= hi:
                continue
            if not is_comment(ls, lo):
                broken.append(
                    f"{c.module} {c.name}: cites line {lo}, which is not a comment"
                    f" ({ls[lo - 1].strip()[:44]!r}); the assignment is at line {at}"
                )
                continue
            end = hi
            while is_comment(ls, end + 1):
                end += 1
            if end >= at or any(is_comment(ls, n) for n in range(end + 1, at)):
                broken.append(
                    f"{c.module} {c.name}: cites lines {lo}-{end}, but that is not"
                    f" the comment block attached to the assignment at line {at}"
                )
    lines = [
        f"      {same} citations into the constant's own module, {cross} into"
        f" another file, {len(broken)} broken"
    ]
    lines += [f"        BROKEN  {b}" for b in broken]
    return (not broken), lines


def check_fetchers_reach_no_report() -> tuple[bool, list[str]]:
    """Check 13: no generator of a report imports a fetcher.

    The six constants in ``scripts/fetch_metar.py`` and
    ``scripts/fetch_ourairports.py`` are registered on the ground that no
    reported figure depends on their values, only on whether the fetch
    succeeded. **That is a claim about the import graph, so it is proved here
    rather than asserted in the basis.** Naming a fetcher in a sentence -- "run
    `python scripts/fetch_metar.py`" is how five reports tell a reader what is
    missing -- is not a data path and does not fail this.
    """
    fetchers = ("fetch_metar", "fetch_ourairports")
    writers = report_writers()
    generators = sorted({g for gs in writers.values() for g in gs})
    offending: list[str] = []
    for gen in generators:
        tree = ast.parse((REPO / gen).read_text(encoding="utf-8"), filename=gen)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hit = [a.name for a in node.names if a.name.split(".")[0] in fetchers]
            elif isinstance(node, ast.ImportFrom):
                hit = (
                    [node.module] if (node.module or "").split(".")[0] in fetchers else []
                )
            else:
                continue
            offending += [f"{gen} imports {h}" for h in hit]
    lines = [
        f"      {len(generators)} report generators, {len(fetchers)} fetchers,"
        f" {len(offending)} import(s) of a fetcher by a generator",
        "      so the six pacing and robustness constants reach no reported"
        " figure, which is what their basis claims",
    ]
    lines += [f"        {o}" for o in offending]
    return (not offending), lines


def check_paper_figures(strict: bool) -> tuple[bool, list[str]]:
    """Check 14: every figure in the paper is in a file its section cites.

    ``scripts/check_paper.py`` is the instrument, and this puts it inside the
    one gate, so a run covers the repository and the paper together. Its
    findings pass outside ``--strict``, the arrangement check 6 uses for the
    pending list, so that a regeneration which moves a figure the paper quotes
    is reported by every ordinary run and fails the freeze condition rather
    than the commit path; ``--strict`` fails on them. A register entry of the
    paper's that reaches nothing, or whose arithmetic does not verify, fails in
    both modes.
    """
    return paper.gate(strict)


# --------------------------------------------------------------------------
# the generated document

DOC = REPO / "docs" / "provenance.md"


def document() -> str:
    """``docs/provenance.md``, built from the manifests.

    **Every count here is summed from the lists and none is written down**,
    which is the arrangement ``scripts/preregistered.py`` uses for its own
    substitution counts and for the same reason: two counts of one quantity
    cannot then disagree in the third digit. Check 10 rebuilds this and
    requires the committed file to match, so an entry added to the register
    without regenerating the document fails the gate.
    """
    counts = Counter(c.cls for c in CONSTANTS)
    unrecorded = [c for c in CONSTANTS if c.basis.startswith(NO_BASIS)]
    unjustified = [c for c in CONSTANTS if c.basis.startswith(VALUE_NOT_JUSTIFIED)]
    found = module_constants()
    authored = [a for a in found if a.module in AUTHORED_MODULES]
    lines = [
        "# Provenance: where every number in this repository comes from",
        "",
        "Generated by `scripts/check_provenance.py --write-doc` from",
        "`taxiout.provenance`. Every count below is summed from those manifests",
        "rather than written down, and check 10 of that script rebuilds this file",
        "and fails if the committed copy differs, so an entry added to the register",
        "without regenerating this document does not get committed.",
        "",
        "The standing rule: every number reaching a report, a note or the paper is",
        "computed by the run that prints it, quoted from a named source with that",
        "source beside it, or marked an estimate with an instrument and an owner.",
        "There is no fourth kind. This file is how that becomes checkable rather",
        "than conventional.",
        "",
        "## The four states",
        "",
        "Every number in a covered surface is in exactly one of four states, and",
        "the gate's job is to prove there is no fifth.",
        "",
        "1. **Generated** by the run that prints it. Needs no entry.",
        "2. **Checked** against a named source, so drift fails a gate.",
        "3. **Registered** with a one-line basis: quoted measurements, chosen",
        "   thresholds, frozen records.",
        "4. **Exempt**, on an explicit list with a reason: mechanism and definition.",
        "",
        "## The constants",
        "",
        f"`taxiout.provenance.CONSTANTS` carries **{len(CONSTANTS)}** entries, one",
        "per module-level numeric assignment the scanner finds outside the",
        f"authored artifact. The scanner finds **{len(found)}** in all, of which",
        f"**{len(authored)}** are in `src/taxiout/synthetic.py` and are covered by",
        "`AUTHORED_ARTIFACTS` as a module rather than one by one.",
        "",
        *table(
            ["class", "entries", "what it means"],
            [
                [
                    f"`{MECHANISM}`",
                    str(counts.get(MECHANISM, 0)),
                    "definitional or a unit conversion; nobody chose it",
                ],
                [
                    f"`{CHOSEN}`",
                    str(counts.get(CHOSEN, 0)),
                    "somebody picked it, and the basis says what against",
                ],
                [
                    f"`{DERIVED}`",
                    str(counts.get(DERIVED, 0)),
                    "computed from the data and frozen into the source",
                ],
                [
                    f"`{QUOTED}`",
                    str(counts.get(QUOTED, 0)),
                    "a measurement from a named run or document",
                ],
            ],
            "-r-",
        ),
        "",
        f"**Of the {counts.get(CHOSEN, 0)} chosen constants, {len(unrecorded)} record",
        "no reason at all, and a further",
        f"{len(unjustified)} record the method, the convention or a neighbouring",
        "use and not the value.** That is the finding this register was built to",
        "produce: `HEADROOM_TRIGGER_S` was chosen with a stated reason and no",
        "recorded basis, and turned out to be an absolute line on a quantity",
        "`docs/review_headroom.md` §7 had already characterised as a floor a refit",
        "exceeds. Each is marked in its own entry and the counts are summed from",
        "those marks.",
        "",
        f"**The first count is meant to be non-zero, and {len(unrecorded)} is the",
        "finished state, not a backlog.** A constant whose basis says what it does",
        "and says plainly that its reason was never written down is *registered*,",
        "which is one of the four states above. Driving the count to zero would",
        "mean writing a plausible reason after the fact for a threshold nobody",
        "argued, which is undetectable afterwards and is precisely the failure this",
        "register exists to prevent. Check 3 prints the count so it cannot drift in",
        "either direction without somebody seeing it.",
        "",
        *table(
            ["module", "constant", "state"],
            [
                [f"`{c.module}`", f"`{c.name}`", state]
                for group, state in (
                    (unrecorded, "no reason recorded"),
                    (unjustified, "method recorded, value not argued"),
                )
                for c in group
            ],
            "---",
        ),
        "",
        "## What a report declares it was generated from",
        "",
        f"**{len(REPORT_INPUTS_DECLARED)}** generators write a footer naming every",
        "path the run read, with a sha256 each; check 8 recomputes them and calls",
        "the report verified, stale or unverifiable. The rule for who carries one",
        "is the Makefile's own measured time: under five minutes and the generator",
        "writes a footer.",
        "",
        f"**{len(REPORTS_WITHOUT_FOOTER)}** reports carry none, and are listed here",
        "rather than skipped: check 8 fails on a report with no footer that is on",
        "no list, so the remainder is a decision and not an oversight.",
        "",
        *table(
            ["report", "what it costs to rebuild", "owner"],
            [[f"`{u.report}`", u.cost, u.owner] for u in REPORTS_WITHOUT_FOOTER],
            "---",
        ),
        "",
        "## What a report quotes from another report",
        "",
        "Check 11 reads every citation of a report by filename and requires every",
        "decimal figure in the citation's own sentence to appear in the cited file.",
        "The scope is the sentence: a paragraph drags in figures that belong to",
        "other sentences, and a checker that flags what nobody claimed is a checker",
        "that gets turned off.",
        "",
        f"**{sum(len(e.figures) for e in REPORT_CITATION_EXEMPTIONS)}** figures are",
        "exempt, each with a reason, and check 11 fails on an exemption it cannot",
        "reach as well as on a figure it cannot find.",
        "",
        *table(
            ["in", "citing", "figures", "why"],
            [
                [f"`{e.source}`", f"`{e.cited}`", ", ".join(e.figures), e.why]
                for e in REPORT_CITATION_EXEMPTIONS
            ],
            "----",
        ),
        "",
        "## What the paper quotes",
        "",
        f"Check 14 runs `scripts/check_paper.py` over `{PAPER}`. A figure there is",
        "a decimal, a percentage, or an integer of at least",
        f"**{PAPER_INTEGER_MIN_DIGITS}** digits, and every figure in a section must",
        "appear in a file the section's `Source:` line cites and, where that line",
        "names a section of the file in italics, in that section of it, as a",
        "number that rounds to it at the figure's own precision and carries its",
        "sign where the paper writes one. A figure that matches only in a frozen",
        "dated record is *historical*, and passes only if it is listed below with a",
        "reason. A figure the paper computes from report figures is listed below",
        "with its arithmetic, which the check verifies: every named operand has to",
        "be in the cited sections and the value has to round to the figure. The",
        "register's own lists below are never read as a source. Outside `--strict`",
        "the check prints its findings and passes; `--strict` is the paper's freeze",
        "condition, and `--report` prints every figure with its status, section by",
        "section, with every number in scope it matches, so a chance match can be",
        "told from the row a sentence quotes.",
        "",
        f"**{len(PAPER_EXEMPT_PATTERNS)}** patterns mark a token shaped like a",
        "figure as something else. A token they cover is counted under the",
        "pattern's name, not dropped.",
        "",
        *table(
            ["pattern", "regular expression", "why"],
            [
                [p.name, "`" + p.pattern.replace("|", "\\|") + "`", p.why]
                for p in PAPER_EXEMPT_PATTERNS
            ],
            "---",
        ),
        "",
        f"**{len(PAPER_NO_SOURCE_SECTIONS)}** sections carry figures and no source",
        "line by design, and are read against every live report instead. What an",
        "entry excuses is the missing line, not the figures.",
        "",
        *table(
            ["section", "why"],
            [[e.section, e.why] for e in PAPER_NO_SOURCE_SECTIONS],
            "--",
        ),
        "",
        f"**{len(PAPER_EXCLUDED_SECTIONS)}** section(s) excluded by name, and",
        "reported with their figure count rather than skipped.",
        "",
        *table(
            ["section", "why"],
            [[e.section, e.why] for e in PAPER_EXCLUDED_SECTIONS],
            "--",
        ),
        "",
        f"**{len(PAPER_HISTORICAL_FIGURES)}** historical figures stand, each with",
        "its reason.",
        "",
        *(
            table(
                ["section", "figure", "why"],
                [[e.section, e.figure, e.why] for e in PAPER_HISTORICAL_FIGURES],
                "---",
            )
            if PAPER_HISTORICAL_FIGURES
            else [
                "None. A figure that matches only in a frozen record fails `--strict`",
                "until it is listed here or brought up to date.",
            ]
        ),
        "",
        f"**{len(PAPER_FIGURE_EXEMPTIONS)}** figures are computed from report",
        "figures rather than quoted, each with the arithmetic the check verifies,",
        "the operands it reads from the cited sections, and the reason.",
        "",
        *(
            table(
                ["section", "figure", "arithmetic", "operands", "why"],
                [
                    [
                        e.section,
                        e.figure,
                        f"`{e.arithmetic}`",
                        ", ".join(e.operands),
                        e.why,
                    ]
                    for e in PAPER_FIGURE_EXEMPTIONS
                ],
                "-----",
            )
            if PAPER_FIGURE_EXEMPTIONS
            else [
                "None. A computed figure that is not listed here is unmatched, and",
                "fails `--strict`.",
            ]
        ),
        "",
        "## Authored artifacts",
        "",
        *[
            line
            for a in AUTHORED_ARTIFACTS
            for line in (
                f"**`{a.module}`** -- {a.what}.",
                "",
                f"The condition that makes it safe: **{a.condition}**. Check 4 proves",
                f"it against {len(MODEL_FIGURE_GENERATORS)} named generators rather",
                "than this record claiming it.",
                "",
            )
        ],
        "## Frozen documents",
        "",
        f"**{len(FROZEN_DOCUMENTS)}** dated records, each with a sha256. These need",
        "the opposite gate from every other file here: the risk is not that they go",
        "stale, it is that somebody edits them. A currency check over one would",
        "demand destroying the record.",
        "",
        *table(
            ["document", "why it is frozen"],
            [[f"`{f.path}`", f.why] for f in FROZEN_DOCUMENTS],
            "--",
        ),
        "",
        "## What a rebuild of an external input is pinned to",
        "",
        f"**{len(REBUILD_PINS)}** external input(s) carry a pin. A pin is a claim",
        "about what a rebuild would return. Exercising one costs the network and a",
        "server willing to answer, so what is written here is how the pin is",
        "applied, how a rebuild proves it was, and what would make it fail.",
        "",
        *[
            line
            for k in REBUILD_PINS
            for line in (
                f"**{k.what}**",
                "",
                f"Rebuilt by {k.how}.",
                "",
                f"Recoverable {k.recoverable}.",
                "",
                *[f"- {c}" for c in k.caveats],
                "",
            )
        ],
        "### The OSM pins, one per airport",
        "",
        f"**{len(OSM_PINS)}** airports at",
        f"**{len({p.timestamp_osm_base for p in OSM_PINS})}** distinct moments, in",
        "`taxiout.provenance.OSM_PINS`, which is what `--pin-from-register` reads.",
        f"The basis, the same for every entry: {OSM_PINS_BASIS}.",
        "",
        *table(
            ["airport", "`timestamp_osm_base`"],
            [[p.icao, p.timestamp_osm_base] for p in OSM_PINS],
            "--",
        ),
        "",
        "## What is still open",
        "",
        f"**{len(PROVENANCE_PENDING)}** items. `--strict` fails while this is not",
        "empty and is the freeze condition; the ordinary run prints it and passes,",
        "because a gate that fails today cannot be wired into a commit path. Emptied",
        "by the work and never by editing the tuple.",
        "",
        *table(
            ["what", "owner"],
            [[p.what, p.owner] for p in PROVENANCE_PENDING],
            "--",
        ),
        "",
        "## What this does not reach",
        "",
        "Stated so its silence is not read as a pass.",
        "",
        *table(
            ["surface", "treatment"],
            [[what, why] for what, why in SCOPE],
            "--",
        ),
        "",
    ]
    return "\n".join(lines)


def check_document() -> tuple[bool, list[str]]:
    """Check 10: the generated document matches the manifests."""
    want = document()
    if not DOC.exists():
        return False, [f"      {DOC.relative_to(REPO)} is missing; --write-doc writes it"]
    got = DOC.read_text(encoding="utf-8")
    if got != want:
        return False, [
            f"      {DOC.relative_to(REPO)} differs from the manifests;"
            " regenerate it with --write-doc",
            f"      committed {len(got)} bytes, generated {len(want)} bytes",
        ]
    return True, [f"      {DOC.relative_to(REPO)} is what the manifests generate"]


def table(
    header: list[str], rows: list[list[str]], align: str | None = None
) -> list[str]:
    """A markdown table, as every report generator here writes one."""
    align = align or ("-" * len(header))
    sep = ["---:" if a == "r" else "---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


# --------------------------------------------------------------------------
# the census


def census() -> int:
    """Print the whole enumeration, with every count computed."""
    modules = scanned_modules()
    found = module_constants(modules)
    defaults = function_defaults(modules)
    literals = string_literals(modules)
    notes = comments(modules)

    print(f"tracked Python files under {', '.join(SCANNED_ROOTS)}: {len(modules)}")
    print()
    authored = [a for a in found if a.module in AUTHORED_MODULES]
    print("module-level numeric assignments, by ast:")
    print(f"  total                       {len(found)}")
    print(f"  in the authored artifact    {len(authored)}")
    print(f"  registered one by one       {len(found) - len(authored)}")
    print(f"  numeric literals in them    {sum(len(a.literals) for a in found)}")
    print()
    print("what a line-anchored regex would have said, for comparison:")
    rx = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*[:=]")
    matched = []
    for module in modules:
        for i, line in enumerate(
            (REPO / module).read_text(encoding="utf-8").splitlines(), 1
        ):
            m = rx.match(line)
            if m and re.search(r"\d", line):
                matched.append((module, m.group(1)))
    ast_keys = {_key(a.module, a.name) for a in found}
    rx_keys = set(matched)
    print(f"  matched                     {len(matched)}")
    print(f"  the ast walk adds           {len(ast_keys - rx_keys)}")
    print(f"  the regex over-counted      {len(rx_keys - ast_keys)}")
    print()
    print("out of the register's scope, counted so the exclusion is a number:")
    print(f"  numeric defaults in function signatures  {len(defaults)}")
    print()
    print(f"string literals (f-string fields excluded): {len(literals)}")
    print(f"comments, by tokenize:                      {len(notes)}")
    figures = sum(len(PROSE_FIGURE.findall(lit.text)) for lit in literals)
    in_comments = sum(len(PROSE_FIGURE.findall(n.text)) for n in notes)
    print(f"  decimal figures in string literals        {figures}")
    print(f"  decimal figures in comments               {in_comments}")
    print()
    counts = Counter(c.cls for c in CONSTANTS)
    print("the register, by class:")
    for cls in CLASSES:
        print(f"  {cls:10s} {counts.get(cls, 0)}")
    print(
        f"  of the {counts.get(CHOSEN, 0)} chosen,"
        f" {sum(1 for c in CONSTANTS if c.basis.startswith(NO_BASIS))} have no basis"
        " recorded in the code"
    )
    print()
    print("per file:")
    per = Counter(a.module for a in found)
    for module, n in sorted(per.items()):
        print(f"  {n:4d}  {module}")
    print()
    print("every assignment, in full:")
    for a in found:
        where = "authored" if a.module in AUTHORED_MODULES else ""
        print(f"  {a.module}:{a.line} {a.name} = {a.literals} {where}")
    return 0


# --------------------------------------------------------------------------
# main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--census", action="store_true", help="print the whole enumeration and stop"
    )
    ap.add_argument(
        "--write-doc",
        action="store_true",
        help="regenerate docs/provenance.md from the manifests and stop",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="the freeze condition: a non-empty pending list or an unverifiable"
        " report input fails",
    )
    args = ap.parse_args()
    if args.census:
        return census()
    if args.write_doc:
        DOC.write_text(document(), encoding="utf-8")
        print(f"wrote {DOC.relative_to(REPO)}")
        return 0

    modules = scanned_modules()
    found = module_constants(modules)
    prose = string_literals(modules, prose_only=True)

    print(f"checking provenance over {len(modules)} tracked Python files\n")
    print("  0 -- what the register reaches, and what it does not")
    for line in check_scope():
        print(line)
    print()

    checks = (
        ("1 -- every constant is categorised", check_categorised(found)),
        ("2 -- every registered value matches the code", check_values(found)),
        ("3 -- every chosen constant has a basis", check_basis()),
        ("4 -- the authored artifact stays out of the model", check_authored()),
        ("5 -- the frozen documents are unchanged", check_frozen()),
        ("6 -- what is still open", check_pending(args.strict)),
        ("7 -- no hardcoded figure in generator prose", check_prose(prose)),
        (
            "8 -- reports against the inputs they declare",
            check_report_inputs(args.strict),
        ),
        (
            "9 -- one report, every generator that writes it",
            check_one_generator_per_claim(),
        ),
        ("10 -- the generated document matches the manifests", check_document()),
        (
            "11 -- a figure quoted beside a report's name is in that report",
            check_report_citations(),
        ),
        (
            "12 -- every line a basis cites says what the basis says",
            check_register_citations(),
        ),
        (
            "13 -- no generator of a report imports a fetcher",
            check_fetchers_reach_no_report(),
        ),
        (
            "14 -- every figure in the paper is in a file its section cites",
            check_paper_figures(args.strict),
        ),
    )
    # Every check this module defines has to be in the tuple above. A check that
    # is written and not wired in is worse than no check: its silence reads as a
    # pass. That happened to check 10 in this file's first version, so it is
    # asserted here as well as in tests/test_provenance.py.
    wired = {title.split(" -- ")[0] for title, _ in checks}
    defined = {
        name
        for name in globals()
        if name.startswith("check_") and callable(globals()[name])
    } - {"check_scope"}
    if len(defined) != len(wired):
        raise SystemExit(
            f"{len(defined)} checks defined and {len(wired)} wired into the run:"
            " a check that is not called cannot fail, and its silence reads as a"
            " pass"
        )

    failures = []
    for title, (ok, lines) in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {title}")
        for line in lines:
            print(line)
        if not ok:
            failures.append(title)
        print()

    if failures:
        print(f"FAIL  {len(failures)} check(s) failed:")
        for f in failures:
            print(f"        {f}")
        return 1
    print(
        "provenance clean: every number in the scanned surface is categorised"
        + (", and nothing is pending" if not PROVENANCE_PENDING else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
