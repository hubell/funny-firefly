# SPDX-License-Identifier: GPL-3.0-only
"""Whether the file substitutes the two tail rules: one recorded decision.

``taxiout.models.TAIL_RULES_SUBSTITUTED`` records §1 of
``reports/preregistered-2026-10-04.md``, and ``scripts/make_submission.py``
reads it rather than deciding for itself, the way it reads
``SEED_AVERAGE_ADOPTED``. Both files are read structurally, with the standard
library alone, so this runs where neither the model stack nor the challenge
data is installed. What the written column is under each setting is
``tests/test_tails.py``'s, through the script's ``submitted``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "src" / "taxiout" / "models.py"
SCRIPT = REPO / "scripts" / "make_submission.py"
COPY = "reports/preregistered-2026-10-04.md"


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


def test_the_rules_are_off_and_the_decision_names_its_basis():
    source = MODELS.read_text(encoding="utf-8")
    node = _assignment(ast.parse(source), "TAIL_RULES_SUBSTITUTED")
    assert isinstance(node, ast.AnnAssign) and ast.unparse(node.annotation) == "bool"
    assert isinstance(node.value, ast.Constant) and node.value.value is False
    basis = _comment_above(source, node.lineno)
    assert COPY in basis
    assert "are still fitted, still exclude" in basis
    assert "but their predictions are not substituted" in basis
    assert (REPO / COPY).exists()


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


def test_the_register_holds_both_decisions_because_the_scanner_reads_bools():
    """A bool constant is a recorded decision, so the provenance scanner finds
    it and the register carries it with a basis, like any chosen number."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    from check_provenance import module_constants
    from taxiout.provenance import CHOSEN, CONSTANTS

    found = {a.name: a.literals for a in module_constants(["src/taxiout/models.py"])}
    assert found["TAIL_RULES_SUBSTITUTED"] == (False,)
    assert found["SEED_AVERAGE_ADOPTED"] == (True,)
    held = {c.name: c for c in CONSTANTS if c.module == "src/taxiout/models.py"}
    assert held["TAIL_RULES_SUBSTITUTED"].cls == CHOSEN
    assert COPY in held["TAIL_RULES_SUBSTITUTED"].basis
    assert held["SEED_AVERAGE_ADOPTED"].cls == CHOSEN
