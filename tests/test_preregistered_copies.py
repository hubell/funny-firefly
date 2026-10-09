# SPDX-License-Identifier: GPL-3.0-only
"""The published copies of the pre-registrations after the freeze.

``reports/preregistered-2026-09-28.md`` and ``reports/preregistered-2026-10-03.md``
are the working notes the rule was fixed in, with the places their closing
sections list changed and nothing else. ``reports/preregistered-2026-10-04.md``
is its note byte for byte: written to need no change, it has no closing section
of differences, and its digest is the note's. The notes are not published; where one
is on hand, its digest is checked first and the copy is rebuilt from it, in the
form ``tests/test_tails.py`` checks the copy of 2 October in. Each substitution
is held by a pattern for the note's side, so this file names no work block
either, and by the copy's words for the other side.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# The working notes are on hand where the environment variable names their
# directory; there is no default, and without it each copy is checked alone.
NOTES = os.environ.get("PREREGISTRATION_NOTES")
PUBLISHED_TAIL = "\n## What differs from the note\n"
# A numbered work block, with or without a letter after its number; and a dated
# working note's file name, ``preregistered.NOTE_REFERENCE_TOKEN`` held here so
# that this file needs the standard library alone, as the pytest-only run does.
WORK_BLOCK_TOKEN = r"\bBlock\s+\d+[A-Za-z]*\b"
NOTE_REFERENCE_TOKEN = r"\b[A-Za-z0-9_]+_\d{4}-\d{2}-\d{2}\.md\b"

NOTE_0928 = "eee5b947e4c449eb7df5765ca62d25a3ca954f201ecefa23311eec8918813e60"
NOTE_1003 = "91e3306ef2b5e0efd6ab8d0a5c0ed9e2fc29ed0b0015bb4af8e96c6a2a601662"
NOTE_1004 = "568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990"

# Copies equal to their note, byte for byte: path to (note, the note's digest).
IDENTICAL = {
    "reports/preregistered-2026-10-04.md": (
        "preregistration_2026-10-04_board.md",
        NOTE_1004,
    ),
}

COPIES = {
    "reports/preregistered-2026-09-28.md": (
        "preregistration_2026-09-28.md",
        NOTE_0928,
        (
            (
                WORK_BLOCK_TOKEN + r" copies this file, unchanged, into\n"
                r"`reports/preregistered-2026-09-28\.md` and freezes it under its"
                r" digest; the\n"
                r"digest recorded in the cha[t] before the upload is the proof of the"
                r" order\.",
                "The working note this is copied from was digested when\n"
                "it was written, sha256\n"
                f"`{NOTE_0928}`, before\n"
                "any of the figures it governs existed; that digest is the proof of the"
                " order.\n"
                "This copy differs from the note only in the place listed at its end.",
            ),
        ),
    ),
    "reports/preregistered-2026-10-03.md": (
        "preregistration_2026-10-03_candidates.md",
        NOTE_1003,
        (
            (
                r"\(`workingdir/block\d+[a-z]*/seeds\.md`,\n  the year row",
                "(a working file outside\n  this repository, the year row",
            ),
        ),
    ),
}


def _published(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


@pytest.mark.parametrize("path", sorted(COPIES))
def test_the_note_with_the_listed_substitutions_is_the_published_copy(path):
    """Where the note is on hand: its digest first, then each listed
    substitution applied once, and the result is the published copy up to its
    closing section byte for byte, which is all the copy adds after it."""
    name, note_sha, substitutions = COPIES[path]
    note = Path(NOTES or ".") / name
    if not NOTES or not note.exists():
        pytest.skip(
            "the working note is not on hand; the copy is checked alone"
            " (PREREGISTRATION_NOTES names the notes' directory)"
        )
    raw = note.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == note_sha
    text = raw.decode("utf-8")
    for pattern, words in substitutions:
        text, n = re.subn(pattern, lambda _m, w=words: w, text)
        assert n == 1, pattern
    copy = _published(path)
    assert copy.count(PUBLISHED_TAIL) == 1
    assert copy[: copy.index(PUBLISHED_TAIL)] == text


@pytest.mark.parametrize("path", sorted(COPIES))
def test_the_published_copy_names_no_work_block_and_credits_openstreetmap(path):
    sys.path.insert(0, str(REPO / "scripts"))
    import check_licence_hygiene as hygiene

    copy = _published(path)
    assert not re.search(WORK_BLOCK_TOKEN, copy)
    assert not re.search(r"\bblock\d+[a-z]*\b", copy)
    assert not re.search(r"\bcha[t]\b|\bsessio[n]\b", copy, re.IGNORECASE)
    assert not re.search(NOTE_REFERENCE_TOKEN, copy)
    assert not hygiene.attribution_in(copy)
    assert copy.count(hygiene.ATTRIBUTION) == 1
    # and the note's digest, which fixed the order, is the one it names in full
    assert set(re.findall(r"`([0-9a-f]{64})`", copy)) == {COPIES[path][1]}


@pytest.mark.parametrize("path", sorted(COPIES))
def test_the_copy_is_frozen_at_its_digest_and_its_record_names_the_note(path):
    sys.path.insert(0, str(REPO / "src"))
    from taxiout.provenance import FROZEN_DOCUMENTS, digest

    held = {f.path: f for f in FROZEN_DOCUMENTS}
    assert path in held
    assert digest(REPO / path) == held[path].sha256
    assert COPIES[path][1] in held[path].why


@pytest.mark.parametrize("path", sorted(IDENTICAL))
def test_the_identical_copy_is_its_note_byte_for_byte(path):
    """Where the note is on hand, the copy is the note; and either way the
    copy's own digest is the note's, which is what fixed the order."""
    name, note_sha = IDENTICAL[path]
    raw = (REPO / path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == note_sha
    note = Path(NOTES or ".") / name
    if not NOTES or not note.exists():
        pytest.skip(
            "the working note is not on hand; the copy is checked alone"
            " (PREREGISTRATION_NOTES names the notes' directory)"
        )
    assert note.read_bytes() == raw


@pytest.mark.parametrize("path", sorted(IDENTICAL))
def test_the_identical_copy_names_no_work_block_and_credits_openstreetmap(path):
    sys.path.insert(0, str(REPO / "scripts"))
    import check_licence_hygiene as hygiene

    copy = _published(path)
    assert not re.search(WORK_BLOCK_TOKEN, copy)
    assert not re.search(r"\bblock\d+[a-z]*\b", copy)
    assert not re.search(r"\bcha[t]\b|\bsessio[n]\b", copy, re.IGNORECASE)
    assert not re.search(NOTE_REFERENCE_TOKEN, copy)
    assert not hygiene.attribution_in(copy)
    assert copy.count(hygiene.ATTRIBUTION) == 1
    assert "ODbL" in copy
    assert PUBLISHED_TAIL not in copy


@pytest.mark.parametrize("path", sorted(IDENTICAL))
def test_the_identical_copy_is_frozen_at_the_notes_digest(path):
    sys.path.insert(0, str(REPO / "src"))
    from taxiout.provenance import FROZEN_DOCUMENTS, digest

    held = {f.path: f for f in FROZEN_DOCUMENTS}
    assert path in held
    assert digest(REPO / path) == held[path].sha256 == IDENTICAL[path][1]


# Copies whose note ends with the credit paragraph. The note's substitutions
# are applied as above, its credit paragraph is moved after the closing
# section so that it stays the last paragraph, and nothing else changes. The
# patterns name no note: a dated working note's file name and a numbered work
# block are matched by form, as ``WORK_BLOCK_TOKEN`` matches one.
NOTE_1004_TAILS = "4d388f1d65a56ee06a66c45598a3231bbac63e125dc0b229e931cddf6641efeb"
NOTE_1004_C7C8 = "4ac17be6925d03255a2e52b1d1ee307e297e913ee42f33c063a03d8e36fbe8cd"
_NOTE_FILE = r"[A-Za-z0-9_]+_\d{4}-\d{2}-\d{2}\.md"
_WORK_BLOCK = r"Block\s+\d+[A-Za-z]*"
_UNPUBLISHED = "a working note this repository does not publish"
_MEASUREMENT = "the measurement of section 3's candidates"
CREDIT = "\n\nMovement and flight data supplied by EUROCONTROL PRC"

COPIES_WITH_CREDIT = {
    "reports/preregistered-2026-10-04-tails.md": (
        "preregistration_2026-10-04_tails_restored.md",
        NOTE_1004_TAILS,
        (
            (
                r"recorded in\n`" + _NOTE_FILE + r"`\.",
                "recorded in\nhis statements, " + _UNPUBLISHED + ".",
            ),
            (
                r"\(`workingdir/block\d+[a-z]*/[A-Za-z0-9_]+\.txt`\)",
                "(a working file outside this repository)",
            ),
        ),
        0,
    ),
    "reports/preregistered-2026-10-04-c7c8.md": (
        "preregistration_2026-10-04_c7_c8.md",
        NOTE_1004_C7C8,
        (
            (r"\(`" + _NOTE_FILE + r"`, item 26\)", "(his statements, item 26)"),
            (
                r"recorded in `" + _NOTE_FILE + r"`\.",
                "recorded in his statements, " + _UNPUBLISHED + ".",
            ),
            (
                _WORK_BLOCK
                + r"'s record commit \(`93732cb`\) is HEAD; "
                + _WORK_BLOCK
                + r" has not started\.",
                "The record commit of the second upload (`93732cb`) is HEAD; the"
                " measurement of\nsection 3's candidates has not started.",
            ),
            (r"\(`" + _NOTE_FILE + r"`\) lists", "(" + _UNPUBLISHED + ") lists"),
            # the three inside the combination rule
            (
                r"If " + _WORK_BLOCK + r" adopts a configuration",
                "If " + _MEASUREMENT + " adopts a configuration",
            ),
            (
                _WORK_BLOCK + r"'s configuration stands alone",
                "the configuration of " + _MEASUREMENT + " stands alone",
            ),
            (
                r"If " + _WORK_BLOCK + r"\nadopts nothing",
                "If " + _MEASUREMENT + "\nadopts nothing",
            ),
        ),
        3,
    ),
}


def _split_credit(text: str) -> tuple[str, str]:
    """The text up to its credit paragraph, and the credit paragraph."""
    i = text.rindex(CREDIT)
    return text[: i + 1], text[i + 2 :]


@pytest.mark.parametrize("path", sorted(COPIES_WITH_CREDIT))
def test_the_note_with_its_substitutions_and_its_credit_last_is_the_copy(path):
    """Where the note is on hand: its digest, each listed substitution applied
    once, and the result is the copy up to its closing section, and the note's
    credit paragraph is the copy's last paragraph, byte for byte."""
    name, note_sha, substitutions, _ = COPIES_WITH_CREDIT[path]
    note = Path(NOTES or ".") / name
    if not NOTES or not note.exists():
        pytest.skip(
            "the working note is not on hand; the copy is checked alone"
            " (PREREGISTRATION_NOTES names the notes' directory)"
        )
    raw = note.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == note_sha
    text = raw.decode("utf-8")
    for pattern, words in substitutions:
        text, n = re.subn(pattern, lambda _m, w=words: w, text)
        assert n == 1, pattern
    body, credit = _split_credit(text)
    copy = _published(path)
    assert copy.count(PUBLISHED_TAIL) == 1
    assert copy[: copy.index(PUBLISHED_TAIL)] == body
    assert copy.endswith("\n\n" + credit)


@pytest.mark.parametrize("path", sorted(COPIES_WITH_CREDIT))
def test_the_copy_with_its_credit_names_no_work_block_and_credits_once(path):
    sys.path.insert(0, str(REPO / "scripts"))
    import check_licence_hygiene as hygiene

    copy = _published(path)
    assert not re.search(WORK_BLOCK_TOKEN, copy)
    assert not re.search(r"\bblock\d+[a-z]*\b", copy)
    assert not re.search(r"\bcha[t]\b|\bsessio[n]\b", copy, re.IGNORECASE)
    assert not re.search(NOTE_REFERENCE_TOKEN, copy)
    assert not hygiene.attribution_in(copy)
    assert copy.count(hygiene.ATTRIBUTION) == 1
    # the credit is the last paragraph, after the closing section
    assert copy.rindex(CREDIT) > copy.index(PUBLISHED_TAIL)
    # the note's digest is named in full; the only other is the board's, which
    # the note itself names
    assert set(re.findall(r"`([0-9a-f]{64})`", copy)) == {
        COPIES_WITH_CREDIT[path][1],
        NOTE_1004,
    }


@pytest.mark.parametrize("path", sorted(COPIES_WITH_CREDIT))
def test_the_closing_section_lists_every_place_and_those_inside_the_rule(path):
    _, _, substitutions, inside = COPIES_WITH_CREDIT[path]
    closing = _published(path).split(PUBLISHED_TAIL, 1)[1]
    rows = [r for r in closing.split("\n") if r.startswith("| ") and "---" not in r]
    assert len(rows) == 1 + len(substitutions)
    assert sum("(inside the rule)" in r for r in rows) == inside
    for _, words in substitutions:
        assert " ".join(words.split()) in " ".join(closing.split())


@pytest.mark.parametrize("path", sorted(COPIES_WITH_CREDIT))
def test_the_copy_with_its_credit_is_frozen_at_its_digest(path):
    sys.path.insert(0, str(REPO / "src"))
    from taxiout.provenance import FROZEN_DOCUMENTS, digest

    held = {f.path: f for f in FROZEN_DOCUMENTS}
    assert path in held
    assert digest(REPO / path) == held[path].sha256
    assert COPIES_WITH_CREDIT[path][1] in held[path].why


# The copies of 7 October 2026: four amendments of the board pre-registration
# and the two notes of 6 October. Made as the two above are, with the owner's
# decision of 7 October 2026 for the places inside a rule or a definition: the
# work that a rule reads is named in words. Each note names other digests than
# its own (the documents it amends, the files it reads), and the copy names
# exactly those and its note's.
_REPORT = "a working report this repository does not publish"
_OUTSIDE = "a working file outside this repository"
_PILOT_REPORT = "(the pilot's report, "
_FETCH = "the OPDI fetch of 4 October"
_FUSION = "the fusion model's measurement"
_LANDING = "the landing of v10"
_REPORT_FILE = r"\(`" + _NOTE_FILE + r"`\)"

COPIES_OF_7_OCTOBER = {
    "reports/preregistered-2026-10-04-adsb-leg.md": (
        "preregistration_2026-10-04_adsb_leg.md",
        "6cdc0f76d32ea314e5df05216a56a8b63963538b4fe60b11a54ed3be2d7bc126",
        (
            (
                r"after "
                + _WORK_BLOCK
                + r" stopped at the end of its Step 3\n"
                + _REPORT_FILE,
                "after the pilot stopped at the end of its Step 3\n(" + _REPORT + ")",
            ),
            (
                r"recorded in\n`" + _NOTE_FILE + r"`\.",
                "recorded in\nhis statements, " + _UNPUBLISHED + ".",
            ),
            (
                r"\(" + _WORK_BLOCK + r", sections 2 and 3\)",
                _PILOT_REPORT + "sections 2 and 3)",
            ),
            (r"\(" + _WORK_BLOCK + r", section 3\)", _PILOT_REPORT + "section 3)"),
            (r"The " + _WORK_BLOCK + r" prompt's", "The pilot's instruction"),
            (
                r"\(" + _WORK_BLOCK + r", section 4\.2, the sample D-AILL\)",
                _PILOT_REPORT + "section 4.2, the sample D-AILL)",
            ),
            (
                r"as " + _WORK_BLOCK + r" describes it",
                "as the pilot's report describes it",
            ),
            (_WORK_BLOCK + r" read the first motion", "The pilot read the first motion"),
        ),
        5,
    ),
    "reports/preregistered-2026-10-04-opdi.md": (
        "preregistration_2026-10-04_opdi.md",
        "3b22e7b49038d4d1216b59af7f9027f1521796a966236165278e034b0bea95c5",
        (
            (
                r"\(`" + _NOTE_FILE + r"`, items 24 and 25\)",
                "(his statements, items 24 and 25)",
            ),
            (
                r"recorded in `" + _NOTE_FILE + r"`\.",
                "recorded in his statements, " + _UNPUBLISHED + ".",
            ),
            (
                r"\(OurAirports, as " + _WORK_BLOCK + r" uses them\)",
                "(OurAirports, as the pilot uses them)",
            ),
        ),
        1,
    ),
    "reports/preregistered-2026-10-04-adsb-after-gate.md": (
        "preregistration_2026-10-04_adsb_after_gate.md",
        "9b08c30924d70a65f842a78d96bb9241e76abe4d74914b4846548a95c30bde65",
        (
            (
                r"after " + _WORK_BLOCK + r"'s gate failed",
                "after the pilot's gate failed",
            ),
            (r"after [A-Z][a-z]+'s\nexploratory check", "after an\nexploratory check"),
            (
                r"recorded in `" + _NOTE_FILE + r"`, item 31\.",
                "recorded in his statements, item 31.",
            ),
            (r"\(`" + _NOTE_FILE + r"` §8\)", "(the pilot's report, §8)"),
            (
                r"the runner " + _WORK_BLOCK + r" built and tested\n"
                r"\(`workingdir/block\d+[a-z]*/portable/`\)",
                "the runner the pilot built and tested\n"
                "(a working directory outside this repository)",
            ),
            (_WORK_BLOCK + r" found 42\.0%", "The pilot found 42.0%"),
        ),
        2,
    ),
    "reports/preregistered-2026-10-04-opdi-tie.md": (
        "preregistration_2026-10-04_opdi_tie.md",
        "e3c8f7216722fed756371a4dd8f5d8b8058e86a6d580f1594bbf4155db43fcd3",
        (
            (
                r"after "
                + _WORK_BLOCK
                + r" stopped at the end of its Step 3\n"
                + _REPORT_FILE,
                "after "
                + _FETCH
                + " stopped at the end of its Step 3\n("
                + _REPORT
                + ")",
            ),
            (
                r"to be recorded in\n`" + _NOTE_FILE + r"`\.",
                "to be recorded in\nhis statements, " + _UNPUBLISHED + ".",
            ),
            (r"\(" + _WORK_BLOCK + r"\)", "(" + _FETCH + ")"),
        ),
        0,
    ),
    "reports/preregistered-2026-10-06-last-week.md": (
        "preregistration_2026-10-06_last_week.md",
        "eb35ecd3eb802b077307d5dba6ab07ca0cc5277b6fc7d5522d1b65a86bc9f1d5",
        (
            (r"item 40 of `" + _NOTE_FILE + r"`", "item 40 of his statements"),
            (r"others " + _WORK_BLOCK + r" checked", "others " + _FUSION + " checked"),
            (
                r"\*\*" + _WORK_BLOCK + r"'s fusion\*\* " + _REPORT_FILE,
                "**The fusion model's measurement** (" + _REPORT + ")",
            ),
            (
                r"`" + _NOTE_FILE + r"` computed",
                "A working note this repository does not publish computed",
            ),
            (
                _WORK_BLOCK + r"'s twelve leave-one-month-out fits "
                r"\(`workingdir/block\d+[a-z]*/fits/\nlomo_MM`\)",
                "The twelve leave-one-month-out fits of " + _FUSION + " (working\n"
                "files outside this repository)",
            ),
            (
                r"on " + _WORK_BLOCK + r"'s scored rows \(989,825\)",
                "on the scored rows of " + _FUSION + " (989,825)",
            ),
            (
                r"on " + _WORK_BLOCK + r"'s scored\nrows",
                "on the scored\nrows of " + _FUSION,
            ),
            (
                r"as §5 and " + _WORK_BLOCK + r"\ndefine it",
                "as §5 and " + _LANDING + "\ndefine it",
            ),
        ),
        4,
    ),
    "reports/preregistered-2026-10-06-g1n.md": (
        "preregistration_2026-10-06_g1n_c9b.md",
        "4e6958eb394e72d663234ce364ee93982447834909436f593c2416e7080805f4",
        (
            (
                r"after " + _WORK_BLOCK + r"'s verdicts",
                "after the verdicts of the measurement of group A",
            ),
            (
                r"\*\*" + _WORK_BLOCK + r"\*\* " + _REPORT_FILE,
                "**The measurement of group A** (" + _REPORT + ")",
            ),
            (
                r"`workingdir/block\d+[a-z]*/v10_candidate\.parquet`, sha256",
                "(" + _OUTSIDE + "), sha256",
            ),
            (
                r"\(`workingdir/block\d+[a-z]*/out/g1_rows\.parquet`, sha256",
                "(" + _OUTSIDE + ", sha256",
            ),
            (
                r"\(`workingdir/block\d+[a-z]*/out/c9_lirf_rows\.parquet`, sha256",
                "(" + _OUTSIDE + ", sha256",
            ),
            (
                _WORK_BLOCK + r" reproduces it from committed code",
                _LANDING + " reproduces it from committed code",
            ),
        ),
        3,
    ),
    # the airports amendment of 6 October, made the same way on 7 October
    "reports/preregistered-2026-10-06-airports.md": (
        "preregistration_2026-10-06_airports.md",
        "9ef5691f36365e2691204380b5f4e984d7049a0e2a8922f80943ed5309724a4a",
        (
            (
                r"after " + _WORK_BLOCK + r"'s verdicts and before " + _WORK_BLOCK + r",",
                "after the verdicts of the measurement of group A and before"
                " the measurement of G1n and C9b,",
            ),
            (r"\(" + _WORK_BLOCK + r"\)", "(the measurement of group A)"),
            (
                r"Per-fit times measured in " + _WORK_BLOCK + r" " + _REPORT_FILE,
                "Per-fit times measured in the reading of the candidates C1 to"
                " C5, C7 and C8 (" + _REPORT + ")",
            ),
        ),
        0,
    ),
    # the group C note of 7 October, made the same way on 8 October; its first
    # place names a range of work blocks in the plural
    "reports/preregistered-2026-10-07-group-c.md": (
        "preregistration_2026-10-07_group_c.md",
        "cb0ced3de2e6fb1a0a75bb8d9a4a41af29390ecee52356df25bbbf456d1be060",
        (
            (
                r"\(Blocks\s+\d+ to \d+\)",
                "(the work from the overnight run of 6 October to the landing of v11)",
            ),
            (
                _WORK_BLOCK + r"'s `heldout-2025-MM__c14__seed0`",
                "the overnight run's `heldout-2025-MM__c14__seed0`",
            ),
            (_REPORT_FILE, "(" + _REPORT + ")"),
            (
                r"\(" + _WORK_BLOCK + r"\): " + _WORK_BLOCK + r"'s stacker",
                "(the fusion's refit on group B's base): the stacker of " + _FUSION,
            ),
            (r"\(" + _WORK_BLOCK + r"'s records\)", "(the overnight run's records)"),
            # the three inside C16's definition and its rule
            (
                _WORK_BLOCK + r"'s stacker with 3,000 trees",
                "The stacker of " + _FUSION + " with 3,000 trees",
            ),
            (
                r"on\n" + _WORK_BLOCK + r"'s table",
                "on\nthe table of the fusion's refit on group B's base",
            ),
            (
                r"as " + _WORK_BLOCK + r" fitted it",
                "as the fusion's refit on group B's base fitted it",
            ),
        ),
        3,
    ),
}


def _applied(path: str) -> str | None:
    """The note with its substitutions, or ``None`` where it is not on hand."""
    name, note_sha, substitutions, _ = COPIES_OF_7_OCTOBER[path]
    note = Path(NOTES or ".") / name
    if not NOTES or not note.exists():
        return None
    raw = note.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == note_sha
    text = raw.decode("utf-8")
    for pattern, words in substitutions:
        text, n = re.subn(pattern, lambda _m, w=words: w, text)
        assert n == 1, pattern
    return text


@pytest.mark.parametrize("path", sorted(COPIES_OF_7_OCTOBER))
def test_the_note_of_7_october_with_its_substitutions_is_the_copy(path):
    """Where the note is on hand: its digest, each listed substitution applied
    once, and the result is the copy up to its closing section, and the note's
    credit paragraph is the copy's last paragraph, byte for byte."""
    text = _applied(path)
    if text is None:
        pytest.skip(
            "the working note is not on hand; the copy is checked alone"
            " (PREREGISTRATION_NOTES names the notes' directory)"
        )
    body, credit = _split_credit(text)
    copy = _published(path)
    assert copy.count(PUBLISHED_TAIL) == 1
    assert copy[: copy.index(PUBLISHED_TAIL)] == body
    assert copy.endswith("\n\n" + credit)


@pytest.mark.parametrize("path", sorted(COPIES_OF_7_OCTOBER))
def test_the_copy_of_7_october_names_no_work_block_and_credits_once(path):
    sys.path.insert(0, str(REPO / "scripts"))
    import check_licence_hygiene as hygiene

    copy = _published(path)
    assert not re.search(WORK_BLOCK_TOKEN, copy)
    assert not re.search(r"\bblock\d+[a-z]*\b", copy)
    assert not re.search(r"\bcha[t]\b|\bsessio[n]\b", copy, re.IGNORECASE)
    assert not re.search(NOTE_REFERENCE_TOKEN, copy)
    assert not hygiene.attribution_in(copy)
    assert copy.count(hygiene.ATTRIBUTION) == 1
    assert copy.rindex(CREDIT) > copy.index(PUBLISHED_TAIL)
    # its note's digest is named in the closing section; every other digest is
    # one the note itself names, above the closing section
    above, closing = copy.split(PUBLISHED_TAIL, 1)
    assert set(re.findall(r"`([0-9a-f]{64})`", closing)) == {COPIES_OF_7_OCTOBER[path][1]}
    text = _applied(path)
    if text is not None:
        assert set(re.findall(r"`([0-9a-f]{64})`", above)) == set(
            re.findall(r"`([0-9a-f]{64})`", _split_credit(text)[0])
        )


@pytest.mark.parametrize("path", sorted(COPIES_OF_7_OCTOBER))
def test_the_closing_of_7_october_lists_every_place_and_those_inside_a_rule(path):
    _, _, substitutions, inside = COPIES_OF_7_OCTOBER[path]
    closing = _published(path).split(PUBLISHED_TAIL, 1)[1]
    rows = [r for r in closing.split("\n") if r.startswith("| ") and "---" not in r]
    assert len(rows) == 1 + len(substitutions)
    assert sum("(inside the rule)" in r for r in rows) == inside
    for _, words in substitutions:
        assert " ".join(words.split()) in " ".join(closing.split())


@pytest.mark.parametrize("path", sorted(COPIES_OF_7_OCTOBER))
def test_the_copy_of_7_october_is_frozen_at_its_digest(path):
    sys.path.insert(0, str(REPO / "src"))
    from taxiout.provenance import FROZEN_DOCUMENTS, digest

    held = {f.path: f for f in FROZEN_DOCUMENTS}
    assert path in held
    assert digest(REPO / path) == held[path].sha256
    assert COPIES_OF_7_OCTOBER[path][1] in held[path].why


def test_the_g1n_copy_says_it_was_chosen_after_the_figures():
    """The disclosure the G1n amendment requires, in its own words."""
    copy = _published("reports/preregistered-2026-10-06-g1n.md")
    assert (
        "**Both candidates are chosen after 22d's numbers were seen, and the\n"
        "paper says so.**"
    ) in copy
