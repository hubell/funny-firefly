# SPDX-License-Identifier: GPL-3.0-only
"""The attribution check, and the property that lets it scan its own source.

Every word this file needs is assembled from fragments rather than written
out, for the same reason the check spells its pattern in bracket classes: a
test that names what it forbids becomes a file the check has to be told to
ignore, and an ignored file is where the next one hides.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_licence_hygiene.py"


def _module():
    spec = importlib.util.spec_from_file_location("_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hygiene = _module()

# Assembled, never spelled. The check would flag this file otherwise.
FORBIDDEN_WORDS = (
    "co" + "work",
    "cla" + "ude",
    "anthro" + "pic",
    "copi" + "lot",
    "chat" + "gpt",
    "open" + "ai",
    "co-authored-" + "by",
)


@pytest.mark.parametrize("word", FORBIDDEN_WORDS)
def test_the_pattern_matches_every_word_it_claims_to(word):
    import re

    rx = re.compile(hygiene.ATTRIBUTION_PATTERN, re.IGNORECASE)
    assert rx.search(word)
    assert rx.search(word.upper())
    assert rx.search(f"Generated with {word}, by hand")


def test_the_pattern_does_not_match_its_own_source():
    """The property the bracket classes exist for.

    Written plainly, the pattern matches the file that defines it, and then
    every scanner has to exclude itself and every other scanner by name --
    which is how one commit script came to fail a run merely because a newer
    one existed.
    """
    import re

    rx = re.compile(hygiene.ATTRIBUTION_PATTERN, re.IGNORECASE)
    assert not rx.search(hygiene.ATTRIBUTION_PATTERN)
    assert not rx.search(SCRIPT.read_text(encoding="utf-8"))
    assert not rx.search(Path(__file__).read_text(encoding="utf-8"))


def test_ordinary_prose_is_not_flagged():
    import re

    rx = re.compile(hygiene.ATTRIBUTION_PATTERN, re.IGNORECASE)
    for benign in (
        "a clause in the licence",
        "co-authored a paper with two colleagues",
        "the open air apron",
        "an anthropogenic effect",
    ):
        assert not rx.search(benign), benign


def test_the_substring_limit_is_known_rather_than_discovered():
    """The pattern matches inside words, and that is the accepted trade.

    A word-boundary pattern would miss the compound forms these names take,
    so the check errs towards a false positive that a human reads once. The
    known cases are pinned here so the behaviour is a decision rather than a
    surprise on some future report.
    """
    import re

    rx = re.compile(hygiene.ATTRIBUTION_PATTERN, re.IGNORECASE)
    assert rx.search("the " + "co" + "working" + " space")
    assert rx.search("re" + "open" + "ai" + "ded")


def test_the_repository_itself_is_clean():
    """The check over the real tree and the real history, as the gate runs it."""
    files = hygiene.tracked_files()
    assert files, "no tracked files found; the walk is broken"
    assert hygiene.attribution_hits(files) == {}


def test_a_planted_word_is_found(tmp_path, monkeypatch):
    planted = tmp_path / "docs"
    planted.mkdir()
    target = planted / "note.md"
    target.write_text(f"Drafted with {FORBIDDEN_WORDS[1]}.\n", encoding="utf-8")
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    hits = hygiene.attribution_hits(["docs/note.md"])
    assert "docs/note.md" in hits
    assert hits["docs/note.md"] == [FORBIDDEN_WORDS[1]]


def test_a_file_the_check_does_not_read_is_not_flagged(tmp_path, monkeypatch):
    """Only authored files are scanned; a data fixture is not prose."""
    (tmp_path / "data").mkdir()
    target = tmp_path / "data" / "fixture.json"
    target.write_text(f'{{"tool": "{FORBIDDEN_WORDS[1]}"}}\n', encoding="utf-8")
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    assert hygiene.attribution_hits(["data/fixture.json"]) == {}
