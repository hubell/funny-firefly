# SPDX-License-Identifier: GPL-3.0-only
"""The provenance register and its gate, on constructed trees.

Two halves. The first holds the **scanner** to what it claims, because every
count in the register and in `docs/provenance.md` comes out of it and a scanner
that misses a constant makes the whole gate a formality: a number inside a
multi-line dict, a lowercase name, a dataclass field default, a string that
merely contains a digit, a format field, a hash inside a string literal, and an
untracked file.

The second holds each **check** to failing on the case it exists for, built as a
small tree under ``tmp_path`` rather than by trusting the real one: a check
verified only against a tree that passes is a check nobody has seen fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import check_provenance as cp  # noqa: E402
from taxiout.provenance import (  # noqa: E402
    CHOSEN,
    REGISTER_MODULE,
    SCOPE,
    CLASSES,
    CONSTANTS,
    FROZEN_DOCUMENTS,
    MODEL_FIGURE_GENERATORS,
    NO_BASIS,
    PROVENANCE_PENDING,
    Constant,
    Frozen,
    Pending,
    digest,
    parse_report_inputs,
    report_inputs_footer,
)

SOURCE = '''
# SPDX-License-Identifier: GPL-3.0-only
"""A module docstring mentioning 1.25 seconds, which is not report prose."""

from dataclasses import dataclass

HOUR = 3600
lowercase_floor = 7

# A value spanning lines, which a line-anchored regex reads only the first of.
GRID = {
    "learning_rate": (0.015, 0.03),
    "leaves": 255,
}

VERSION = "release 2.4"  # a string that merely contains a digit


@dataclass
class Thing:
    count: int = 11
    label: str = "x"


def helper(digits: int = 41) -> str:
    # a comment carrying 9.99
    total = 3
    return f"{total} rows at 0.5 each"


PROSE = "the cut is at 6.75 s"
'''


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A one-module tracked tree the scanner reads instead of the real one."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "src" / "untracked.py").write_text("SECRET = 99\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "tracked_files", lambda: ["src/mod.py"])
    return tmp_path


# --------------------------------------------------------------------------
# the scanner


def test_a_constant_inside_a_multi_line_value_is_found(tree):
    found = {a.name: a for a in cp.module_constants()}
    assert "GRID" in found
    # Every literal in the value, not just the first line's.
    assert found["GRID"].literals == (0.015, 0.03, 255)


def test_a_lowercase_module_constant_is_found(tree):
    assert "lowercase_floor" in {a.name for a in cp.module_constants()}


def test_a_dataclass_field_default_is_found(tree):
    found = {a.name: a.literals for a in cp.module_constants()}
    assert found["Thing.count"] == (11,)
    # And the string-valued field is not a numeric constant.
    assert "Thing.label" not in found


def test_a_string_containing_a_digit_is_not_a_numeric_constant(tree):
    names = {a.name for a in cp.module_constants()}
    assert "VERSION" not in names
    assert "PROSE" not in names


def test_a_function_signature_default_is_counted_but_not_registered(tree):
    """Out of the register's scope by SCOPE, and counted so it is not a silence."""
    assert (41,) in {a.literals for a in cp.function_defaults()}
    assert "helper" not in {a.name for a in cp.module_constants()}


def test_a_format_field_is_not_reported_as_prose(tree):
    """The f-string's literal parts come through; the substituted name does not."""
    prose = [lit.text for lit in cp.string_literals(prose_only=True)]
    assert any("rows at 0.5 each" in p for p in prose)
    assert not any("total" in p for p in prose)


def test_a_docstring_is_not_report_prose(tree):
    """A docstring cannot reach a report, which is why check 7 does not read it."""
    everything = [lit.text for lit in cp.string_literals()]
    prose = [lit.text for lit in cp.string_literals(prose_only=True)]
    assert any("1.25 seconds" in p for p in everything)
    assert not any("1.25 seconds" in p for p in prose)


def test_a_hash_inside_a_string_is_not_a_comment(tree):
    got = [c.text for c in cp.comments()]
    assert any("9.99" in c for c in got)
    assert all(c.startswith("#") for c in got)
    # The SPDX line is a comment; the docstring's text is not.
    assert not any("module docstring" in c for c in got)


def test_an_untracked_file_is_not_scanned(tree):
    assert "SECRET" not in {a.name for a in cp.module_constants()}
    assert (tree / "src" / "untracked.py").exists()


def test_the_scanner_is_the_one_instrument_both_callers_use():
    """One implementation, so two counts of one quantity cannot disagree."""
    found = cp.module_constants()
    unregistered = [a for a in found if a.module in cp.UNREGISTERED_MODULES]
    assert len(found) - len(unregistered) == len(CONSTANTS)


def test_the_gate_and_the_register_are_in_their_own_scope():
    """The failure this test exists for, and it happened.

    The register and the gate were validated while both were still untracked,
    so the scanner -- which reads ``git ls-files`` and is right to -- had never
    seen either of them. Committing them put four assignments and twenty-nine
    prose figures in scope at once and the commit's own tree failed its tests.
    Nothing about that was subtle; it was simply never checked. So it is checked.
    """
    modules = cp.scanned_modules()
    assert "scripts/check_provenance.py" in modules, "the gate must scan itself"
    assert REGISTER_MODULE in modules, "the register must be scanned too"
    # And being in scope, each is in exactly one state: the gate's own constant
    # is registered like every other script's, and the register is excluded by
    # name with its reason in SCOPE.
    assert any(c.module == "scripts/check_provenance.py" for c in CONSTANTS), (
        "the gate's own constants are registered"
    )
    assert not any(c.module == REGISTER_MODULE for c in CONSTANTS), (
        "the register cannot be an entry in itself"
    )
    assert REGISTER_MODULE in cp.UNREGISTERED_MODULES
    assert any(REGISTER_MODULE in what for what, _why in SCOPE)
    ok, lines = cp.check_categorised(cp.module_constants())
    assert ok, lines


# --------------------------------------------------------------------------
# the checks


def test_an_unregistered_constant_fails_check_1(monkeypatch):
    extra = cp.Assignment("src/taxiout/nowhere.py", 4, "NEW_FLOOR", (12,))
    ok, lines = cp.check_categorised(cp.module_constants() + (extra,))
    assert not ok
    assert any("NEW_FLOOR" in line for line in lines)


def test_a_register_entry_naming_no_constant_fails_check_1(monkeypatch):
    monkeypatch.setattr(
        cp,
        "CONSTANTS",
        CONSTANTS + (Constant("GHOST", "src/taxiout/models.py", (1,), CHOSEN, "x"),),
    )
    ok, lines = cp.check_categorised(cp.module_constants())
    assert not ok
    assert any("GHOST" in line for line in lines)


def test_a_drifted_value_fails_check_2(monkeypatch):
    drifted = tuple(
        c._replace(literals=(999.0,)) if c.name == "HEADROOM_TRIGGER_S" else c
        for c in CONSTANTS
    )
    monkeypatch.setattr(cp, "CONSTANTS", drifted)
    ok, lines = cp.check_values(cp.module_constants())
    assert not ok
    assert any("HEADROOM_TRIGGER_S" in line for line in lines)


def test_an_empty_basis_on_a_chosen_constant_fails_check_3(monkeypatch):
    monkeypatch.setattr(
        cp,
        "CONSTANTS",
        CONSTANTS + (Constant("BARE", "src/taxiout/models.py", (3,), CHOSEN, "  "),),
    )
    ok, lines = cp.check_basis()
    assert not ok
    assert any("BARE" in line for line in lines)


def test_an_unrecorded_basis_passes_only_while_the_class_is_pending(monkeypatch):
    """The arrangement: say the basis is unrecorded, and own it on the list."""
    ok, _lines = cp.check_basis()
    assert ok, "the tree's unrecorded bases are covered by the pending entry"
    monkeypatch.setattr(
        cp,
        "PROVENANCE_PENDING",
        tuple(p for p in PROVENANCE_PENDING if "basis is recorded nowhere" not in p.what),
    )
    ok, lines = cp.check_basis()
    assert not ok
    assert any("NOT on the pending list" in line for line in lines)


def test_every_chosen_constant_says_what_it_was_picked_against():
    """A basis that is not recorded says so, in the marker the count reads."""
    for c in CONSTANTS:
        if c.cls == CHOSEN:
            assert c.basis.strip(), f"{c.module} {c.name} has no basis at all"
    unrecorded = [c for c in CONSTANTS if c.basis.startswith(NO_BASIS)]
    assert unrecorded, "the finding this register exists to produce is not empty"
    assert all(c.cls == CHOSEN for c in unrecorded)


def test_a_module_importing_the_authored_field_fails_check_4(monkeypatch, tmp_path):
    bad = tmp_path / "scripts"
    bad.mkdir()
    (bad / "run_baselines.py").write_text(
        "from taxiout.synthetic import field\n", encoding="utf-8"
    )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "MODEL_FIGURE_GENERATORS", ("scripts/run_baselines.py",))
    ok, lines = cp.check_authored()
    assert not ok
    assert any("imports taxiout.synthetic" in line for line in lines)


def test_a_prose_citation_of_the_authored_field_does_not_fail_check_4(
    monkeypatch, tmp_path
):
    """Naming its report in a sentence is how the fixture earns its place."""
    good = tmp_path / "scripts"
    good.mkdir()
    (good / "run_baselines.py").write_text(
        'BODY = "see reports/synthetic_airport.md"  # taxiout.synthetic\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "MODEL_FIGURE_GENERATORS", ("scripts/run_baselines.py",))
    ok, _lines = cp.check_authored()
    assert ok


def test_the_authored_field_is_out_of_every_model_figure_generator_today():
    ok, lines = cp.check_authored()
    assert ok, lines
    assert len(MODEL_FIGURE_GENERATORS) >= 10


def test_an_edited_frozen_document_fails_check_5(monkeypatch, tmp_path):
    doc = tmp_path / "docs"
    doc.mkdir()
    (doc / "plan.md").write_text("edited\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(
        cp, "FROZEN_DOCUMENTS", (Frozen("docs/plan.md", "0" * 64, "a dated record"),)
    )
    ok, lines = cp.check_frozen()
    assert not ok
    assert any("DATED RECORD" in line for line in lines)
    assert any("additive and dated" in line for line in lines)


def test_a_missing_frozen_document_fails_check_5(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(
        cp, "FROZEN_DOCUMENTS", (Frozen("docs/gone.md", "0" * 64, "a dated record"),)
    )
    ok, lines = cp.check_frozen()
    assert not ok
    assert any("missing" in line for line in lines)


def test_the_frozen_digests_are_the_documents_on_disk():
    for f in FROZEN_DOCUMENTS:
        assert digest(REPO / f.path) == f.sha256, f.path


def test_strict_fails_on_a_non_empty_pending_list_and_passes_on_an_empty_one(
    monkeypatch,
):
    ok, lines = cp.check_pending(strict=False)
    assert ok, "the ordinary run prints the list and passes"
    assert any("not a failure outside --strict" in line for line in lines)
    ok, _lines = cp.check_pending(strict=True)
    assert not ok, "--strict is the freeze condition"
    monkeypatch.setattr(cp, "PROVENANCE_PENDING", ())
    ok, _lines = cp.check_pending(strict=True)
    assert ok


def test_every_pending_item_has_a_reason_and_an_owner():
    for p in PROVENANCE_PENDING:
        assert p.what.strip() and p.why.strip() and p.owner.strip()
    assert any("code_09e" in p.owner for p in PROVENANCE_PENDING), (
        "cross-report citation is deferred, and the deferral is on the list"
    )


def test_a_hardcoded_figure_in_prose_fails_check_7():
    literal = cp.Literal("scripts/report_x.py", 9, "the step was 12.5 s")
    ok, lines = cp.check_prose((literal,))
    assert not ok
    assert any("12.5" in line for line in lines)
    # A whole number in words is not a measurement and must not cry wolf.
    ok, _lines = cp.check_prose((cp.Literal("scripts/report_x.py", 9, "the three"),))
    assert ok


def test_the_tree_has_no_hardcoded_figure_left():
    ok, lines = cp.check_prose(cp.string_literals(prose_only=True))
    assert ok, lines


def test_a_report_whose_declared_input_changed_fails_check_8(monkeypatch, tmp_path):
    """The Block 09b case, constructed: the ladder rewrote a residual file."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "data").mkdir()
    residual = tmp_path / "data" / "december_encoding-for-stand.parquet"
    residual.write_bytes(b"the residuals as the report read them")
    footer = report_inputs_footer(tmp_path, [residual])
    (tmp_path / "reports" / "r.md").write_text("\n".join(footer), encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "REPORT_INPUTS_DECLARED", (("scripts/g.py", "reports/r.md"),))

    ok, lines = cp.check_report_inputs(strict=False)
    assert ok, lines

    # Now the ladder regenerates the residuals, as Block 09 did.
    residual.write_bytes(b"the residuals after the wider sweep")
    ok, lines = cp.check_report_inputs(strict=False)
    assert not ok
    assert any("STALE" in line for line in lines)
    assert any("has changed since the report was written" in line for line in lines)


def test_an_absent_input_reads_unverifiable_and_never_verified(monkeypatch, tmp_path):
    """The clean-checkout case: data/ is gitignored, so a digest cannot be had."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "data").mkdir()
    residual = tmp_path / "data" / "gone.parquet"
    residual.write_bytes(b"present when the report was written")
    footer = report_inputs_footer(tmp_path, [residual])
    (tmp_path / "reports" / "r.md").write_text("\n".join(footer), encoding="utf-8")
    residual.unlink()
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "REPORT_INPUTS_DECLARED", (("scripts/g.py", "reports/r.md"),))

    ok, lines = cp.check_report_inputs(strict=False)
    assert ok, "unverifiable is not a failure outside --strict"
    assert any("unverifiable" in line for line in lines)
    assert not any("verified, 1 unverifiable, 0 stale" in line for line in lines) or any(
        "0 verified" in line for line in lines
    )
    ok, _lines = cp.check_report_inputs(strict=True)
    assert not ok, "--strict refuses to call an absent input verified"


def test_a_report_declaring_no_inputs_fails_check_8(monkeypatch, tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.md").write_text("# a report\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "REPORT_INPUTS_DECLARED", (("scripts/g.py", "reports/r.md"),))
    ok, lines = cp.check_report_inputs(strict=False)
    assert not ok
    assert any("declares no inputs" in line for line in lines)


def test_the_footer_round_trips():
    footer = "\n".join(report_inputs_footer(REPO, [REPO / "README.md"]))
    got = parse_report_inputs(footer)
    assert got == (("README.md", digest(REPO / "README.md")),)
    assert parse_report_inputs("# a report with no footer") == ()


def test_a_report_carrying_a_section_from_an_unnamed_generator_fails_check_9(
    monkeypatch, tmp_path
):
    """The tails.md case: one header, two generators, and no figure to say so."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "reports" / "t.md").write_text(
        "Generated by `scripts/first.py`; every figure below is computed there.\n",
        encoding="utf-8",
    )
    for name in ("first.py", "second.py"):
        (tmp_path / "scripts" / name).write_text(
            'OUT = REPO / "reports" / "t.md"\ndef main():\n    OUT.write_text("x")\n',
            encoding="utf-8",
        )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(
        cp, "scanned_modules", lambda: ["scripts/first.py", "scripts/second.py"]
    )
    ok, lines = cp.check_one_generator_per_claim()
    assert not ok
    assert any("scripts/second.py" in line for line in lines)


def test_a_cited_report_is_not_a_written_one(monkeypatch, tmp_path):
    """A generator citing another report in prose is not one of its writers."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "reports" / "t.md").write_text("Generated by `scripts/first.py`.\n")
    (tmp_path / "scripts" / "first.py").write_text(
        'OUT = REPO / "reports" / "t.md"\ndef main():\n    OUT.write_text("x")\n',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "other.py").write_text(
        'BODY = "see reports/t.md for the rest"\n'
        'OUT = REPO / "reports" / "o.md"\n'
        'def main():\n    OUT.write_text("x")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(
        cp, "scanned_modules", lambda: ["scripts/first.py", "scripts/other.py"]
    )
    writers = cp.report_writers()
    assert writers["reports/t.md"] == {"scripts/first.py"}
    ok, _lines = cp.check_one_generator_per_claim()
    assert ok


def test_tails_md_now_names_both_of_its_generators():
    """The defect item 9 was about, on the published file."""
    text = (REPO / "reports" / "tails.md").read_text(encoding="utf-8")
    assert "`scripts/report_tails.py`" in text
    assert "`scripts/pooled_tails.py`" in text
    assert "two generators" in text
    # And the claim that used to be false is gone.
    assert "every figure below is computed there. Aggregates only" not in text


def test_the_generated_document_carries_counts_summed_from_the_manifests():
    ok, lines = cp.check_document()
    assert ok, lines
    text = cp.DOC.read_text(encoding="utf-8")
    assert f"**{len(CONSTANTS)}** entries" in text
    unrecorded = sum(1 for c in CONSTANTS if c.basis.startswith(NO_BASIS))
    assert f"{unrecorded} have" in text
    for cls in CLASSES:
        assert f"| `{cls}` |" in text


def test_a_manifest_entry_drifted_from_the_document_is_caught(monkeypatch):
    monkeypatch.setattr(
        cp,
        "PROVENANCE_PENDING",
        PROVENANCE_PENDING + (Pending("a new excursion", "because", "somebody"),),
    )
    ok, lines = cp.check_document()
    assert not ok
    assert any("differs from the manifests" in line for line in lines)


def test_every_check_the_gate_defines_is_wired_into_its_run():
    """The other defect this block made, and the reason it is worth a test.

    ``check_document`` was written, tested directly, and never added to the
    tuple ``main`` iterates, so the gate reported nine checks and called itself
    clean. A check that is not called cannot fail, and its silence reads as a
    pass -- which is the precise failure this whole block exists to remove, made
    by the block itself. ``main`` now refuses to run if the two disagree, and
    this holds the count from the outside.
    """
    import inspect

    source = inspect.getsource(cp.main)
    defined = {
        name
        for name in dir(cp)
        if name.startswith("check_") and callable(getattr(cp, name))
    } - {"check_scope"}
    for name in defined:
        assert f"{name}(" in source, f"{name} is defined but never called by main"
    # And the run prints one line per check, numbered from 1 with no gaps.
    numbers = sorted(int(n) for n in re.findall(r'"(\d+) -- ', source))
    assert numbers == list(range(1, len(defined) + 1)), numbers


def test_the_classes_are_the_four_and_no_fifth():
    assert set(c.cls for c in CONSTANTS) <= set(CLASSES)
    assert len(CLASSES) == 4
