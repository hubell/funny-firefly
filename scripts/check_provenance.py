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

``--strict`` is the freeze condition and Block 11 runs it. The ordinary run
prints checks 6 and 8's unverifiable inputs and passes, because a gate that
fails today cannot be wired into a commit path.

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

from taxiout.provenance import (  # noqa: E402
    AUTHORED_ARTIFACTS,
    CHOSEN,
    DERIVED,
    MECHANISM,
    QUOTED,
    CLASSES,
    CONSTANTS,
    FROZEN_DOCUMENTS,
    MODEL_FIGURE_GENERATORS,
    NO_BASIS,
    PROVENANCE_PENDING,
    REPORT_INPUTS_DECLARED,
    SCOPE,
    digest,
    parse_report_inputs,
)

# Where the register's remit runs. Anything else is out of scope by
# ``SCOPE`` and not by omission.
SCANNED_ROOTS: tuple[str, ...] = ("src/", "scripts/")

# The one module registered whole rather than constant by constant.
AUTHORED_MODULES: frozenset[str] = frozenset(a.module for a in AUTHORED_ARTIFACTS)

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
        if a.module not in AUTHORED_MODULES and _key(a.module, a.name) not in registered
    ]
    scanned = {_key(a.module, a.name) for a in found}
    stale = sorted(k for k in registered - scanned)
    counts = Counter(c.cls for c in CONSTANTS)
    lines = [
        f"      {len(found)} assignments found, "
        f"{len([a for a in found if a.module in AUTHORED_MODULES])} in the authored"
        " artifact, "
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
    covered = any("basis is recorded nowhere" in p.what for p in PROVENANCE_PENDING)
    lines = [
        f"      {len(chosen)} chosen constants, {len(empty)} with an empty basis,"
        f" {len(unrecorded)} whose basis the code does not record",
    ]
    for c in empty:
        lines.append(f"        EMPTY  {c.module} {c.name}")
    if unrecorded:
        lines.append(
            "      not recorded in the code, and on the pending list as a class:"
            if covered
            else "      not recorded in the code, and NOT on the pending list:"
        )
        lines += [f"        {c.module} {c.name}" for c in unrecorded]
    return (not empty and (covered or not unrecorded)), lines


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
    if unverifiable and not strict:
        lines.append(
            "      unverifiable is not a failure outside --strict: data/ is"
            " gitignored, so a clean checkout cannot verify a data digest"
        )
    return (not stale and not (strict and unverifiable)), lines


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
        f"**Of the {counts.get(CHOSEN, 0)} chosen constants, {len(unrecorded)} have",
        "no basis recorded anywhere in the code.** That is the finding this register",
        "was built to produce, and it is the half that failed: `HEADROOM_TRIGGER_S`",
        "was chosen with a stated reason and no recorded basis, and turned out to be",
        "an absolute line on a quantity `docs/review_headroom.md` §7 had already",
        "characterised as a floor a refit exceeds. Each is marked in its own entry,",
        "the count above is summed from those marks, and `PROVENANCE_PENDING` owns",
        "the class.",
        "",
        *table(
            ["module", "constant"],
            [[f"`{c.module}`", f"`{c.name}`"] for c in unrecorded],
            "--",
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
