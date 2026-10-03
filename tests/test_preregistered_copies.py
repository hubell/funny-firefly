# SPDX-License-Identifier: GPL-3.0-only
"""The published copies of the candidates' two pre-registrations.

``reports/preregistered-2026-09-28.md`` and ``reports/preregistered-2026-10-03.md``
are the working notes the rule was fixed in, with the places their closing
sections list changed and nothing else. The notes are not published; where one
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
