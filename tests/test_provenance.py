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


def test_an_unrecorded_basis_is_registered_and_an_empty_one_is_not(monkeypatch):
    """The arrangement after the basis pass, and it changed.

    While the basis pass was outstanding, an unrecorded basis passed check 3
    only because ``PROVENANCE_PENDING`` owned the class. The pass is done: a
    constant whose basis says what it does and says its reason was never
    written down is **registered**, one of the four states, and needs no
    pending entry. What still fails is a basis that says nothing at all, which
    is the difference between a recorded absence and an omission.
    """
    ok, lines = cp.check_basis()
    assert ok, lines
    assert any("meant to be non-zero" in line for line in lines)

    monkeypatch.setattr(
        cp,
        "CONSTANTS",
        (Constant("SOMETHING_S", "src/m.py", (5.0,), CHOSEN, "   "),),
    )
    ok, lines = cp.check_basis()
    assert not ok
    assert any("EMPTY" in line for line in lines)


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
    """On a constructed item, since the real list is empty from the freeze on."""
    monkeypatch.setattr(
        cp,
        "PROVENANCE_PENDING",
        (Pending("an excursion", "the reason it is open", "the block that owns it"),),
    )
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
    assert not any("code_09e" in p.owner for p in PROVENANCE_PENDING), (
        "cross-report citation is checked now; nothing is owed to this block"
    )
    assert not any("basis is recorded nowhere" in p.what for p in PROVENANCE_PENDING), (
        "the basis pass is done, and its entry is emptied by the work"
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
    # check 8 also lists the reports carrying no footer, which reads the tracked
    # tree; this constructed one is not a repository, so the listing is empty.
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})
    # check 8 also lists the reports carrying no footer, which reads the tracked
    # tree; this constructed one is not a repository, so the listing is empty.
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})
    # check 8 also lists the reports carrying no footer, which reads the tracked
    # tree; this constructed one is not a repository, so the listing is empty.
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})

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
    # check 8 also lists the reports carrying no footer, which reads the
    # tracked tree; this constructed one is not a repository.
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})

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
    # check 8 also lists the reports carrying no footer, which reads the
    # tracked tree; this constructed one is not a repository.
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})
    ok, lines = cp.check_report_inputs(strict=False)
    assert not ok
    assert any("declares no inputs" in line for line in lines)


def test_the_footer_round_trips():
    footer = "\n".join(report_inputs_footer(REPO, [REPO / "README.md"]))
    got = parse_report_inputs(footer)
    assert got == (("README.md", digest(REPO / "README.md"), False),)
    assert parse_report_inputs("# a report with no footer") == ()
    assert "parsed elements" not in footer, "no cached answer, no mark and no note"


# --------------------------------------------------------------------------
# a cached Overpass answer, digested by its elements: Block 11a


def _overpass(elements, *, stamp, fetched=None):
    """An invented Overpass answer as the client stores it."""
    payload = {
        "version": 0.6,
        "generator": "Overpass API, invented",
        "osm3s": {"timestamp_osm_base": stamp, "copyright": "invented"},
        "elements": elements,
    }
    if fetched:
        payload["taxiout:fetch"] = {
            "mirror": "https://instance.invalid/api/interpreter",
            "date_pin": "2026-09-01T13:32:15Z",
            "fetched_utc": fetched,
            "timestamp_osm_base": stamp,
        }
    return payload


def _elements_of_an_airport():
    return [
        {"type": "node", "id": 7, "lat": 47.45, "lon": 8.56, "tags": {"ref": "A"}},
        {"type": "way", "id": 70, "nodes": [7, 8], "tags": {"aeroway": "taxiway"}},
        {"type": "node", "id": 8, "lat": 47.46, "lon": 8.57},
    ]


def _store(path, payload):
    import gzip
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def test_two_answers_differing_only_in_header_and_fetch_record_digest_the_same(
    tmp_path,
):
    """The header and the fetch record are left out; so are order and key order."""
    from taxiout.provenance import elements_digest

    first = _store(
        tmp_path / "a.json.gz",
        _overpass(_elements_of_an_airport(), stamp="2026-09-01T13:32:15Z"),
    )
    again = _store(
        tmp_path / "b.json.gz",
        _overpass(
            _elements_of_an_airport(),
            stamp="2026-09-26T07:36:11Z",
            fetched="2026-09-26T07:36:22Z",
        ),
    )
    assert digest(first) != digest(again), "the bytes differ, as every refetch's do"
    assert elements_digest(first) == elements_digest(again)

    shuffled = [dict(reversed(list(e.items()))) for e in _elements_of_an_airport()[::-1]]
    third = _store(
        tmp_path / "c.json.gz", _overpass(shuffled, stamp="2020-01-01T00:00:00Z")
    )
    assert elements_digest(third) == elements_digest(first)
    assert elements_digest(tmp_path / "gone.json.gz") == "absent"


def test_a_single_changed_tag_digests_differently(tmp_path):
    """One `alt_name` gone is what LSZH's May answer looked like, thirteen times."""
    from taxiout.provenance import elements_digest

    base = _store(
        tmp_path / "a.json.gz",
        _overpass(_elements_of_an_airport(), stamp="2026-09-01T13:32:15Z"),
    )
    elements = _elements_of_an_airport()
    elements[1]["tags"] = {"aeroway": "taxiway", "alt_name": "B"}
    moved = _store(
        tmp_path / "b.json.gz", _overpass(elements, stamp="2026-09-01T13:32:15Z")
    )
    assert elements_digest(moved) != elements_digest(base)


def _constructed_repo(monkeypatch, tmp_path, footer_lines):
    """A repository of one report reading one cached answer, for check 8."""
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "r.md").write_text("\n".join(footer_lines), encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "REPORT_INPUTS_DECLARED", (("scripts/g.py", "reports/r.md"),))
    # check 8 also lists the reports carrying no footer, which reads the
    # tracked tree; this constructed one is not a repository.
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})


def test_a_refetched_answer_verifies_against_a_footer_written_from_the_other(
    monkeypatch, tmp_path
):
    """The same elements under another header verify, both ways; a moved tag does not.

    This is what a clean checkout that rebuilt the cache sees: every byte of
    every answer differs from the one the footer was written from, and every
    element is the same.
    """
    answer = tmp_path / "data" / "raw" / "LSZH.json.gz"
    first = _overpass(_elements_of_an_airport(), stamp="2026-09-01T13:32:15Z")
    again = _overpass(
        _elements_of_an_airport(),
        stamp="2026-09-26T07:36:11Z",
        fetched="2026-09-26T07:36:22Z",
    )
    for written, found in ((first, again), (again, first)):
        _store(answer, written)
        footer = report_inputs_footer(tmp_path, [answer])
        assert any("`data/raw/LSZH.json.gz` (parsed elements)" in row for row in footer)
        assert any("is a cached Overpass answer" in row for row in footer)
        _constructed_repo(monkeypatch, tmp_path, footer)
        (declared,) = parse_report_inputs("\n".join(footer))
        assert declared.path == "data/raw/LSZH.json.gz" and declared.by_elements

        _store(answer, found)
        ok, lines = cp.check_report_inputs(strict=True)
        assert ok, lines
        assert any(
            "1 verified" in line and "by their parsed elements" in line for line in lines
        )

    elements = _elements_of_an_airport()
    del elements[0]["tags"]
    _store(answer, _overpass(elements, stamp="2026-09-26T07:36:11Z"))
    ok, lines = cp.check_report_inputs(strict=False)
    assert not ok
    assert any("STALE" in line and "LSZH.json.gz" in line for line in lines)


def test_a_byte_digest_still_verifies_the_file_it_was_taken_from(monkeypatch, tmp_path):
    """A footer written before the mark: the byte digest of a cached answer.

    Identical bytes are identical elements, so it verifies the file it was
    taken from, and a footer of either era reads. After a refetch its bytes
    are gone, and it reads stale until the report is regenerated.
    """
    answer = _store(
        tmp_path / "data" / "raw" / "LSZH.json.gz",
        _overpass(_elements_of_an_airport(), stamp="2026-09-01T13:32:15Z"),
    )
    footer = [
        "## What this report was generated from",
        "",
        "| input | sha256 |",
        "| --- | --- |",
        f"| `data/raw/LSZH.json.gz` | `{digest(answer)}` |",
    ]
    (declared,) = parse_report_inputs("\n".join(footer))
    assert not declared.by_elements and declared.digest == digest(answer)
    _constructed_repo(monkeypatch, tmp_path, footer)
    ok, lines = cp.check_report_inputs(strict=True)
    assert ok, lines
    assert not any("by their parsed elements" in line for line in lines)

    _store(
        answer,
        _overpass(
            _elements_of_an_airport(),
            stamp="2026-09-26T07:36:11Z",
            fetched="2026-09-26T07:36:22Z",
        ),
    )
    ok, lines = cp.check_report_inputs(strict=False)
    assert not ok
    assert any("STALE" in line for line in lines)


def test_the_last_two_footers_parse_and_declare_the_cache_by_its_elements():
    """The routing report and the penalty sensitivity, footered by the rebuild.

    Both read the eleven cached answers and nothing else, so each declares
    eleven inputs, every one a cached answer in the marked form. Read from the
    committed reports, so it holds in a clean checkout too, where the cache is
    absent and check 8 can only call the inputs unverifiable.
    """
    from taxiout.airports import AIRPORTS
    from taxiout.provenance import (
        REPORT_INPUTS_DECLARED,
        REPORTS_WITHOUT_FOOTER,
        is_overpass_answer,
    )

    declared = {report: generator for generator, report in REPORT_INPUTS_DECLARED}
    for report, generator in (
        ("reports/taxi_routing.md", "scripts/build_taxi_graph.py"),
        ("reports/penalty_sensitivity.md", "scripts/check_penalty_sensitivity.py"),
    ):
        assert declared.get(report) == generator, report
        rows = parse_report_inputs((REPO / report).read_text(encoding="utf-8"))
        assert [r.path for r in rows] == [f"data/raw/{a.icao}.json.gz" for a in AIRPORTS]
        assert all(r.by_elements and is_overpass_answer(r.path) for r in rows), report
        assert all(re.fullmatch(r"[0-9a-f]{64}", r.digest) for r in rows), report
    assert REPORTS_WITHOUT_FOOTER == ()


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
    assert f"{unrecorded} record" in text
    assert f"{unrecorded} is the" in text, "the document says the count is finished"
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


# --------------------------------------------------------------------------
# the bases, the citation checker and the footers: Block 09e


def test_the_geometry_report_cannot_disagree_with_the_graph_about_a_snap():
    """The one constant the basis pass removed instead of annotating.

    ``scripts/report_geometry_gaps.py`` carried its own ``DIRECT_SNAP_M =
    60.0``, a copy of ``taxiout.graph.DIRECT_SNAP_M``, which is the one with a
    recorded reason. A copy cannot be given a basis; it can only be removed, so
    it was, and this is the assertion that keeps it removed. **Identity and not
    equality**: two modules agreeing on 60.0 today is what the copy already
    did.
    """
    import report_geometry_gaps as rgg
    from taxiout import graph

    assert rgg.DIRECT_SNAP_M is graph.DIRECT_SNAP_M
    source = (REPO / "scripts" / "report_geometry_gaps.py").read_text(encoding="utf-8")
    assert not re.search(r"^DIRECT_SNAP_M\s*=", source, re.M), (
        "the copy is back; it must be imported from taxiout.graph"
    )
    assert not any(
        c.module == "scripts/report_geometry_gaps.py" and c.name == "DIRECT_SNAP_M"
        for c in CONSTANTS
    ), "an imported name is not a module constant and needs no entry"


def test_no_generator_of_a_report_imports_a_fetcher():
    """The six fetcher constants' basis, as an assertion rather than a promise.

    Their basis is that no reported figure depends on their values, only on
    whether the fetch succeeded. That is a claim about the import graph, so the
    gate proves it. Five reports name a fetcher in a sentence -- "run `python
    scripts/fetch_metar.py`" is how they tell a reader what is missing -- and a
    sentence is not a data path.
    """
    ok, lines = cp.check_fetchers_reach_no_report()
    assert ok, lines
    assert any("0 import(s) of a fetcher" in line for line in lines)


def test_a_generator_that_imports_a_fetcher_fails_check_13(monkeypatch, tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "scripts" / "g.py").write_text(
        "from pathlib import Path\n"
        "import fetch_metar\n"
        'OUT = Path("reports") / "r.md"\n'
        'OUT.write_text("x")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})
    ok, lines = cp.check_fetchers_reach_no_report()
    assert not ok
    assert any("imports fetch_metar" in line for line in lines)


def test_the_count_of_constants_with_no_recorded_reason_is_pinned():
    """Fourteen, and the number is pinned from the outside in both directions.

    **This is the assertion most easily got wrong either way.** A constant
    whose basis says what it does and says its reason was never written down is
    *registered*, so the pending entry that owned the class is resolved -- at
    fourteen and not at zero. Driving it to zero would mean writing plausible
    reasons after the fact for thresholds nobody argued, which is undetectable
    afterwards; letting it grow would mean a new constant arrived unargued. So
    the test fails on a change in either direction and the number is written
    here, deliberately, in the one place a written-down count belongs.
    """
    unrecorded = [
        c for c in CONSTANTS if c.cls == CHOSEN and c.basis.startswith(NO_BASIS)
    ]
    assert len(unrecorded) == 14, sorted(f"{c.module} {c.name}" for c in unrecorded)
    assert {c.module for c in unrecorded} == {
        "src/taxiout/ceiling.py",
        "src/taxiout/congestion.py",
        "src/taxiout/exposure.py",
        "src/taxiout/planlags.py",
        "src/taxiout/runways.py",
        "src/taxiout/saturation.py",
        "scripts/report_geometry_gaps.py",
        "scripts/report_tails.py",
        "scripts/run_baselines.py",
    }
    # And every one says what the constant does, not merely that nothing is known.
    for c in unrecorded:
        assert len(c.basis) > len(NO_BASIS) + 30, f"{c.name}'s basis says too little"


def test_a_marked_basis_is_one_of_the_two_states_and_never_both():
    from taxiout.provenance import VALUE_NOT_JUSTIFIED

    for c in CONSTANTS:
        assert not (c.basis.startswith(NO_BASIS) and VALUE_NOT_JUSTIFIED in c.basis), (
            f"{c.name} is marked as both states"
        )
    unjustified = [c for c in CONSTANTS if c.basis.startswith(VALUE_NOT_JUSTIFIED)]
    assert unjustified, "the second marker is defined and used by nothing"
    assert all(c.cls == CHOSEN for c in unjustified)


def test_a_figure_quoted_beside_a_report_that_lacks_it_fails_check_11(
    monkeypatch, tmp_path
):
    """The class that bit, constructed: 9.9 became 9.5 under a regeneration."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "congestion.md").write_text(
        "# Congestion\n\nThe oracle ceiling at Heathrow on July is 9.9 s.\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "paper_draft.md").write_text(
        "# Draft\n\nThe ceiling is 9.9 s, from `congestion.md`.\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# nothing\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "FROZEN_DOCUMENTS", ())
    monkeypatch.setattr(cp, "REPORT_CITATION_EXEMPTIONS", ())

    ok, lines = cp.check_report_citations()
    assert ok, lines

    # The ladder regenerates congestion.md and the ceiling moves.
    (tmp_path / "reports" / "congestion.md").write_text(
        "# Congestion\n\nThe oracle ceiling at Heathrow on July is 9.5 s.\n",
        encoding="utf-8",
    )
    ok, lines = cp.check_report_citations()
    assert not ok
    assert any("STALE" in line and "9.9" in line for line in lines)


def test_an_exemption_no_citation_reaches_fails_check_11(monkeypatch, tmp_path):
    """An exemption is a record, and a record nothing reaches is stale too."""
    from taxiout.provenance import Uncited

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "a.md").write_text("# A\n\nnothing cites b here.\n", "utf-8")
    (tmp_path / "reports" / "b.md").write_text("# B\n", "utf-8")
    (tmp_path / "README.md").write_text("# nothing\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "FROZEN_DOCUMENTS", ())
    monkeypatch.setattr(
        cp,
        "REPORT_CITATION_EXEMPTIONS",
        (Uncited("reports/a.md", "reports/b.md", ("1.5",), "a reason"),),
    )
    ok, lines = cp.check_report_citations()
    assert not ok
    assert any("UNREACHED exemption" in line for line in lines)


def test_the_citation_is_scoped_to_its_sentence_and_not_its_paragraph(
    monkeypatch, tmp_path
):
    """The scope is the whole difference between a useful check and noise.

    A figure two sentences away from the citation was never claimed to come
    from the cited file, and flagging it is the failure check 9's first
    instrument made in the other direction.
    """
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "b.md").write_text("# B\n", "utf-8")
    (tmp_path / "reports" / "a.md").write_text(
        "# A\n\nThis paragraph measures 3.75 s here. The method is in `b.md`.\n",
        "utf-8",
    )
    (tmp_path / "README.md").write_text("# nothing\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "FROZEN_DOCUMENTS", ())
    monkeypatch.setattr(cp, "REPORT_CITATION_EXEMPTIONS", ())
    ok, lines = cp.check_report_citations()
    assert ok, lines
    assert any("0 carrying a decimal figure" in line for line in lines)


def test_a_frozen_record_is_not_read_as_a_stale_citation(monkeypatch, tmp_path):
    """A dated record's figures were true on its date; that is what it is for."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "b.md").write_text("# B\n\nnow 9.5 s.\n", "utf-8")
    (tmp_path / "reports" / "old-2026-01-01.md").write_text(
        "# Old\n\nIt was 9.9 s, from `b.md`.\n", "utf-8"
    )
    (tmp_path / "README.md").write_text("# nothing\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "REPORT_CITATION_EXEMPTIONS", ())
    monkeypatch.setattr(
        cp, "FROZEN_DOCUMENTS", (Frozen("reports/old-2026-01-01.md", "x" * 64, "dated"),)
    )
    ok, lines = cp.check_report_citations()
    assert ok, lines
    assert "reports/old-2026-01-01.md" not in cp.citing_documents()


def test_a_basis_citing_the_wrong_line_fails_check_12(monkeypatch, tmp_path):
    """09d's own defect, constructed: it edited a generator and left the line.

    The commit that created this register moved ``report_residual.py`` down by
    eighteen lines and left six of its own citations pointing at what used to
    be there. Nothing noticed, because nothing looked.
    """
    (tmp_path / "src").mkdir()
    module = tmp_path / "src" / "m.py"
    module.write_text(
        "# A comment that is the basis.\nTHRESHOLD = 30\n", encoding="utf-8"
    )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(
        cp,
        "CONSTANTS",
        (Constant("THRESHOLD", "src/m.py", (30,), CHOSEN, "line 1: because"),),
    )
    ok, lines = cp.check_register_citations()
    assert ok, lines

    # Something is inserted above it, as 09d inserted an import.
    module.write_text(
        "import os\n\n# A comment that is the basis.\nTHRESHOLD = 30\n", encoding="utf-8"
    )
    ok, lines = cp.check_register_citations()
    assert not ok
    assert any("BROKEN" in line and "not a comment" in line for line in lines)


def test_a_group_comment_covering_two_constants_is_not_a_broken_citation(
    monkeypatch, tmp_path
):
    """The convention the register uses, held so check 12 cannot over-flag."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(
        "# One comment covering both of the two below.\nA_S = 60.0\nB_S = 90.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(
        cp,
        "CONSTANTS",
        (
            Constant("A_S", "src/m.py", (60.0,), CHOSEN, "line 1: the reason"),
            Constant("B_S", "src/m.py", (90.0,), CHOSEN, "line 1, covering this too"),
        ),
    )
    ok, lines = cp.check_register_citations()
    assert ok, lines


def test_the_tree_has_no_broken_line_citation_left():
    ok, lines = cp.check_register_citations()
    assert ok, lines


def test_a_report_with_no_footer_is_listed_and_never_silently_skipped(
    monkeypatch, tmp_path
):
    """A report skipped without a reason reads exactly like one forgotten."""
    from taxiout.provenance import Unfootered

    (tmp_path / "reports").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "reports" / "r.md").write_text("# R\n\nno footer here.\n", "utf-8")
    (tmp_path / "scripts" / "g.py").write_text(
        'from pathlib import Path\nOUT = Path("reports") / "r.md"\nOUT.write_text("x")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "REPORT_INPUTS_DECLARED", ())
    monkeypatch.setattr(cp, "REPORTS_WITHOUT_FOOTER", ())
    monkeypatch.setattr(cp, "FROZEN_DOCUMENTS", ())
    monkeypatch.setattr(cp, "report_writers", lambda: {"reports/r.md": {"scripts/g.py"}})

    ok, lines = cp.check_report_inputs(strict=False)
    assert not ok
    assert any("SILENT" in line and "reports/r.md" in line for line in lines)

    # Listed with a cost and an owner, it passes -- and the listing is printed.
    monkeypatch.setattr(
        cp,
        "REPORTS_WITHOUT_FOOTER",
        (
            Unfootered(
                "reports/r.md", "scripts/g.py", "seven hours", "the next ladder run"
            ),
        ),
    )
    ok, lines = cp.check_report_inputs(strict=False)
    assert ok, lines
    assert any("no footer -- seven hours" in line for line in lines)


def test_every_unfootered_report_names_a_cost_and_an_owner():
    from taxiout.provenance import REPORTS_WITHOUT_FOOTER, REPORT_INPUTS_DECLARED

    declared = {r for _g, r in REPORT_INPUTS_DECLARED}
    for u in REPORTS_WITHOUT_FOOTER:
        assert (REPO / u.report).exists(), u.report
        assert u.report not in declared, f"{u.report} both declares and does not"
        assert len(u.cost) > 10 and len(u.owner) > 10, u.report


# --------------------------------------------------------------------------
# the OSM pins, recorded where a rebuild cannot destroy them: Block 09i


# The seven moments Block 09e counted across the eleven cache files on
# 22 September, with how many airports carry each. Written out here as a second
# transcription, independent of the register's, so an edit to either shows.
BLOCK_09E_STAMPS = {
    "2026-09-01T13:23:05Z": 3,
    "2026-09-01T13:24:06Z": 1,
    "2026-09-01T13:26:06Z": 1,
    "2026-09-01T13:27:05Z": 2,
    "2026-09-01T13:29:10Z": 1,
    "2026-09-01T13:30:11Z": 2,
    "2026-09-01T13:32:15Z": 1,
}


def test_the_register_carries_one_osm_pin_per_airport_at_the_seven_stamps():
    """Eleven entries, eleven distinct airports, the seven moments 09e counted.

    Nothing else in the repository carries the per-airport values: the cache
    that did is gitignored, and a pinned rebuild re-stamps it with the rebuild
    day. So the register's copy is held here against a second transcription.
    """
    from collections import Counter

    from taxiout.airports import AIRPORTS
    from taxiout.provenance import OSM_PINS

    icaos = [p.icao for p in OSM_PINS]
    assert len(OSM_PINS) == 11
    assert len(set(icaos)) == 11
    assert icaos == [a.icao for a in AIRPORTS], "one per airport, in fetch order"
    stamps = [p.timestamp_osm_base for p in OSM_PINS]
    assert len(set(stamps)) == 7
    assert Counter(stamps) == BLOCK_09E_STAMPS
    # Fetched one after another against a moving database, so never backwards.
    assert stamps == sorted(stamps)


def test_the_osm_pins_hold_for_whatever_cache_this_checkout_has():
    """The register against the cache, by the evidence that kind of cache carries.

    Three branches, each for its own reason. **An original cache**, fetched
    unpinned on 1 September, carries in each header the state it was served
    from, so each header must equal its pin. **A rebuilt cache** carries the
    rebuild day in every header, because Overpass stamps an answer with its own
    database state and never with the pin, so the header proves nothing there;
    the sidecar ``scripts/fetch_osm.py`` writes must record each airport as
    fetched at its pin, and each answer's own fetch record must equal its entry.
    **No cache**, which is every clean checkout because ``data/raw/`` is
    gitignored, is skipped: there the register is the only record of the pins.
    """
    from taxiout.osm import pin_agreement
    from taxiout.provenance import OSM_PINS

    got = pin_agreement(
        REPO / "data" / "raw", {p.icao: p.timestamp_osm_base for p in OSM_PINS}
    )
    if not got:
        pytest.skip(
            "no OSM cache in this checkout: data/raw/ is gitignored, so a clean"
            " checkout has none, and there the register is the only record of"
            " the pins"
        )
    bad = [f"{c.icao}, {c.branch}: {c.problem}" for c in got if c.problem]
    assert not bad, (
        "a cached answer does not show it was fetched at its pin in OSM_PINS:"
        f" {bad}. An original answer is read by its header, a rebuilt one by the"
        " sidecar data/raw/PINS.json, which scripts/fetch_osm.py writes on every"
        " fetch"
    )


def _cached(cache, icao, *, stamp, pin=None, mirror=None, fetched=None):
    """An invented cached answer: a header stamp and, if fetched here, a record."""
    import gzip
    import json

    payload = {"osm3s": {"timestamp_osm_base": stamp}, "elements": [{"id": 1}]}
    if mirror:
        payload["taxiout:fetch"] = {
            "mirror": mirror,
            "date_pin": pin,
            "fetched_utc": fetched,
            "timestamp_osm_base": stamp,
        }
    cache.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache / f"{icao}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return payload


def test_the_pin_check_reads_each_kind_of_cache_by_its_own_evidence(tmp_path):
    """All three branches, on caches constructed under ``tmp_path``.

    Each branch passes on the evidence it should read and fails on a planted
    disagreement, so a rebuilt cache cannot pass on its header and an original
    one cannot pass on a sidecar it does not have.
    """
    import json

    from taxiout.osm import PINS_SIDECAR, pin_agreement, read_sidecar, record_fetch

    pins = {"EDDF": "2026-09-01T13:23:05Z", "LIRF": "2026-09-01T13:29:10Z"}
    rebuild_day = "2026-09-26T08:00:00Z"
    mirror = "https://instance.invalid/api/interpreter"

    # No cache: nothing is read, and the caller skips.
    assert pin_agreement(tmp_path / "absent", pins) == []

    # An original cache: read by its headers.
    original = tmp_path / "original"
    for icao, pin in pins.items():
        _cached(original, icao, stamp=pin)
    got = pin_agreement(original, pins)
    assert [(c.icao, c.branch, c.problem) for c in got] == [
        ("EDDF", "original", None),
        ("LIRF", "original", None),
    ]
    _cached(original, "LIRF", stamp=rebuild_day)
    (lirf,) = [c for c in pin_agreement(original, pins) if c.problem]
    assert lirf.icao == "LIRF" and "no sidecar entry" in lirf.problem

    # A rebuilt cache: every header carries the rebuild day, and the sidecar
    # carries the pins.
    rebuilt = tmp_path / "rebuilt"
    for icao, pin in pins.items():
        answer = _cached(
            rebuilt, icao, stamp=rebuild_day, pin=pin, mirror=mirror, fetched=rebuild_day
        )
        record_fetch(rebuilt, icao, answer)
    got = pin_agreement(rebuilt, pins)
    assert [(c.icao, c.branch, c.problem) for c in got] == [
        ("EDDF", "rebuilt", None),
        ("LIRF", "rebuilt", None),
    ]
    written = json.loads((rebuilt / PINS_SIDECAR).read_text(encoding="utf-8"))
    assert sorted(written["airports"]) == ["EDDF", "LIRF"]
    assert read_sidecar(rebuilt)["LIRF"] == {
        "date_pin": pins["LIRF"],
        "mirror": mirror,
        "fetched_utc": rebuild_day,
        "timestamp_osm_base": rebuild_day,
    }

    # A sidecar entry at another pin fails, and so does an answer a later fetch
    # replaced behind its entry's back.
    stale = _cached(
        rebuilt, "EDDF", stamp=rebuild_day, pin=None, mirror=mirror, fetched=rebuild_day
    )
    record_fetch(rebuilt, "EDDF", stale)
    (eddf,) = [c for c in pin_agreement(rebuilt, pins) if c.problem]
    assert eddf.icao == "EDDF" and "fetched at None" in eddf.problem
    _cached(
        rebuilt,
        "LIRF",
        stamp=rebuild_day,
        pin=pins["LIRF"],
        mirror=mirror,
        fetched="2026-09-26T09:00:00Z",
    )
    lirf = [c for c in pin_agreement(rebuilt, pins) if c.icao == "LIRF"][0]
    assert lirf.branch == "rebuilt" and "another fetch" in lirf.problem

    # Requested at the pin and answered from a database that predates it, as
    # a fallback mirror answered LSZH on 26 September: the request is right,
    # the entry agrees with its file, and the stamp alone fails it.
    may = "2026-05-31T22:37:44Z"
    served = _cached(
        rebuilt, "LIRF", stamp=may, pin=pins["LIRF"], mirror=mirror, fetched=rebuild_day
    )
    record_fetch(rebuilt, "LIRF", served)
    lirf = [c for c in pin_agreement(rebuilt, pins) if c.icao == "LIRF"][0]
    assert lirf.branch == "rebuilt"
    assert may in lirf.problem and pins["LIRF"] in lirf.problem
    assert "cannot have been served at the pin" in lirf.problem

    # An entry written before the sidecar carried a stamp says nothing about
    # what the answer could have been, and fails for that.
    sidecar = json.loads((rebuilt / PINS_SIDECAR).read_text(encoding="utf-8"))
    for airport in ("EDDF", "LIRF"):
        answer = _cached(
            rebuilt,
            airport,
            stamp=rebuild_day,
            pin=pins[airport],
            mirror=mirror,
            fetched=rebuild_day,
        )
        del answer["taxiout:fetch"]["timestamp_osm_base"]
        _write_payload(rebuilt / f"{airport}.json.gz", answer)
        sidecar["airports"][airport] = dict(answer["taxiout:fetch"])
    (rebuilt / PINS_SIDECAR).write_text(json.dumps(sidecar), encoding="utf-8")
    got = pin_agreement(rebuilt, pins)
    assert [c.icao for c in got if c.problem] == ["EDDF", "LIRF"]
    assert all("records no database state" in c.problem for c in got)


def _write_payload(path, payload):
    import gzip
    import json

    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def test_the_recorded_rebuild_reads_the_register_and_never_the_cache():
    """The recorded command is the one that survives a rebuild.

    The one recorded before read each pin out of the cache header, which a
    rebuild overwrites with the rebuild day, so its second run would have
    pinned every airport to that day with no error.
    """
    from taxiout.provenance import REBUILD_PINS

    for k in REBUILD_PINS:
        assert "--pin-from-register --force" in k.how
        assert "`OSM_PINS`" in k.how
        assert "--pin-from-cache" not in k.how
        # and it names what a rebuilt cache carries in place of the header
        assert "`data/raw/PINS.json`" in k.how
        assert "pin_agreement" in k.recoverable


def test_the_pending_list_is_empty_at_the_freeze():
    """Emptied by the work of the freeze, and by nothing else.

    The three items it owned closed on what its run established on 26 September
    2026: the paper's by `check_paper.py --strict` against every regenerated
    report; the footers' by the two reports the pinned rebuild regenerated with
    theirs; and the pin's by eleven airports fetched at their pins, compared
    element by element with the cache the matrices were built from, one control
    and the matrices compared after. Empty is the freeze condition, so anything
    added fails `--strict` until the work that closes it is done.
    """
    from taxiout.provenance import REPORTS_WITHOUT_FOOTER

    assert PROVENANCE_PENDING == ()
    assert REPORTS_WITHOUT_FOOTER == ()
    ok, lines = cp.check_pending(strict=True)
    assert ok, lines
