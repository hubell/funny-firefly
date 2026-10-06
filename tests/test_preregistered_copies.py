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
