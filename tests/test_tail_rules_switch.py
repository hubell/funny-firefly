# SPDX-License-Identifier: GPL-3.0-only
"""Whether the file substitutes the two tail rules: one recorded decision.

``taxiout.models.TAIL_RULES_SUBSTITUTED`` records §1 of
``reports/preregistered-2026-10-04.md`` (False, which built v8) and its
withdrawal by §1 of ``reports/preregistered-2026-10-04-tails.md`` (True again
from v9), and ``scripts/make_submission.py`` reads it rather than deciding for
itself, the way it reads ``SEED_AVERAGE_ADOPTED``. Both files are read structurally, with the standard
library alone, so this runs where neither the model stack nor the challenge
data is installed. What the written column is under each setting is
``tests/test_tails.py``'s, through the script's ``submitted``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "src" / "taxiout" / "models.py"
SCRIPT = REPO / "scripts" / "make_submission.py"
COPY = "reports/preregistered-2026-10-04.md"
COPY_TAILS = "reports/preregistered-2026-10-04-tails.md"

# Each decision the constant has recorded, with the copy that fixed it and the
# words of that copy its comment quotes. The constant holds the last of them.
DECISIONS = {
    "v8, the rules off": (
        COPY,
        ("are still fitted, still exclude", "but their predictions are not substituted"),
    ),
    "v9, the rules restored": (
        COPY_TAILS,
        (
            "Every file uploaded from now (v9, v10) substitutes the record-quality term",
            "C6 replaces the term on the term's rows and on nothing else",
        ),
    ),
}


def _assignment(tree: ast.Module, name: str) -> ast.AnnAssign | ast.Assign:
    found = [
        node
        for node in tree.body
        if (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == name for t in node.targets)
        )
    ]
    assert len(found) == 1, name
    return found[0]


def _comment_above(source: str, line: int) -> str:
    """The comment block that ends on the line before ``line``."""
    lines = source.splitlines()[: line - 1]
    block: list[str] = []
    for text in reversed(lines):
        if not text.startswith("#"):
            break
        block.append(text.lstrip("# ").rstrip())
    return " ".join(reversed(block))


@pytest.mark.parametrize("decision", sorted(DECISIONS))
def test_the_rules_are_restored_and_each_decision_names_its_basis(decision):
    source = MODELS.read_text(encoding="utf-8")
    node = _assignment(ast.parse(source), "TAIL_RULES_SUBSTITUTED")
    assert isinstance(node, ast.AnnAssign) and ast.unparse(node.annotation) == "bool"
    assert isinstance(node.value, ast.Constant) and node.value.value is True
    basis = " ".join(_comment_above(source, node.lineno).split())
    copy, words = DECISIONS[decision]
    assert copy in basis
    for w in words:
        assert w in basis, w
    assert (REPO / copy).exists()
    # and the copy says what the comment quotes it as saying
    text = " ".join((REPO / copy).read_text(encoding="utf-8").split())
    for w in words:
        assert w in text, w


def test_the_term_rows_take_c6b_and_the_decision_names_its_basis():
    source = MODELS.read_text(encoding="utf-8")
    node = _assignment(ast.parse(source), "RECORD_QUALITY_FROM_C6B")
    assert isinstance(node, ast.AnnAssign) and ast.unparse(node.annotation) == "bool"
    assert isinstance(node.value, ast.Constant) and node.value.value is True
    basis = " ".join(_comment_above(source, node.lineno).split())
    assert COPY_TAILS in basis and "reports/c6.md" in basis
    assert '"C6: ADOPTED C6b"' in basis
    assert "C6: ADOPTED C6b" in (REPO / "reports" / "c6.md").read_text(encoding="utf-8")


def test_the_script_reads_the_flag_and_decides_nothing_itself():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "taxiout.models"
        for alias in node.names
    }
    assert "TAIL_RULES_SUBSTITUTED" in imported
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "submitted"
    ]
    assert len(calls) == 1, "one place writes the predicted column"
    assert ast.unparse(calls[0].args[-1]) == "TAIL_RULES_SUBSTITUTED"
    # and nothing else in the script calls combine on the way to the file
    combines = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("combine")
    ]
    inside = {
        id(n)
        for f in tree.body
        if isinstance(f, ast.FunctionDef) and f.name == "submitted"
        for n in ast.walk(f)
    }
    assert combines and all(id(c) in inside for c in combines)


def test_the_manifest_records_the_flag_and_the_rows_each_rule_would_take():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fields = {
        key.value: ast.unparse(value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert fields["tail_rules_substituted"] == "TAIL_RULES_SUBSTITUTED"
    would = fields["rows_each_rule_would_take"]
    assert "'record_quality': int(tail.notna().sum())" in would
    assert "'plan_shift': int(shifted.sum())" in would
    # the existing counts stay beside it, unchanged
    assert fields["record_quality_rows"] == "int(tail.notna().sum())"
    assert "shifted.groupby" in fields["plan_shift_rows"]
    # and, from v9, which predictor the term's rows took and the rows each took
    assert fields["record_quality_from_c6b"] == "RECORD_QUALITY_FROM_C6B"
    took = fields["rows_each_rule_took"]
    assert "int(record_quality.notna().sum()) if TAIL_RULES_SUBSTITUTED else 0" in took
    assert "int(shifted.sum()) if TAIL_RULES_SUBSTITUTED else 0" in took
    assert fields["booster_file_sha256"] == "sha256(booster_file)"
    assert fields["file_sha256"] == "sha256(written)"


def test_the_script_substitutes_the_record_quality_rows_it_was_told_to():
    """The term's rows take C6b where ``RECORD_QUALITY_FROM_C6B`` says so, and
    that is the series the one ``submitted`` call is given. Its base is the
    configuration's prediction, ``physical``, unless group B's new base takes
    its place (``taxiout.models.GROUP_B_APPLIED``)."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    chosen = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "record_quality" for t in node.targets)
    ]
    assert len(chosen) == 1
    assert ast.unparse(chosen[0].value) == (
        "got['c6'] if RECORD_QUALITY_FROM_C6B else tail"
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "submitted"
    ]
    assert [ast.unparse(a) for a in calls[0].args] == [
        "base",
        "record_quality",
        "plan_shift",
        "TAIL_RULES_SUBSTITUTED",
    ]
    (main,) = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    bases = [
        ast.unparse(node.value)
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "base" for t in node.targets)
    ]
    assert bases == ["physical", "grouped['base']"]


def test_the_register_holds_both_decisions_because_the_scanner_reads_bools():
    """A bool constant is a recorded decision, so the provenance scanner finds
    it and the register carries it with a basis, like any chosen number."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    from check_provenance import module_constants
    from taxiout.provenance import CHOSEN, CONSTANTS

    found = {a.name: a.literals for a in module_constants(["src/taxiout/models.py"])}
    assert found["TAIL_RULES_SUBSTITUTED"] == (True,)
    assert found["RECORD_QUALITY_FROM_C6B"] == (True,)
    assert found["SEED_AVERAGE_ADOPTED"] == (True,)
    held = {c.name: c for c in CONSTANTS if c.module == "src/taxiout/models.py"}
    assert held["TAIL_RULES_SUBSTITUTED"].cls == CHOSEN
    assert COPY in held["TAIL_RULES_SUBSTITUTED"].basis
    assert COPY_TAILS in held["TAIL_RULES_SUBSTITUTED"].basis
    assert held["RECORD_QUALITY_FROM_C6B"].cls == CHOSEN
    assert COPY_TAILS in held["RECORD_QUALITY_FROM_C6B"].basis
    assert "reports/c6.md" in held["RECORD_QUALITY_FROM_C6B"].basis
    assert held["SEED_AVERAGE_ADOPTED"].cls == CHOSEN
