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


# --------------------------------------------------------------------------
# the forbidden-path rule
#
# The assistant-configuration paths are the case that motivated the directory
# rule. They are named through the same fragments the check itself uses: this
# file is scanned by check 4 like any other source, so spelling either name
# would make the suite fail on itself.

ASSISTANT_DIR = "." + FORBIDDEN_WORDS[1]
ASSISTANT_FILE = FORBIDDEN_WORDS[1].upper() + ".md"


@pytest.mark.parametrize(
    "path",
    [
        f"{ASSISTANT_DIR}/settings.local.json",
        f"{ASSISTANT_DIR}/settings.json",
        f"{ASSISTANT_DIR}/agents/reviewer.md",
        f"{ASSISTANT_DIR}/skills/deep/nested/file.txt",
        ASSISTANT_DIR,
        ASSISTANT_FILE,
        f"docs/{ASSISTANT_FILE}",
    ],
)
def test_assistant_configuration_is_forbidden_at_every_depth(path):
    """The hole this rule closes: the settings file is not an authored type.

    Check 4 scans prose and source, so a JSON settings file carrying a tool
    name would have passed it. The path rule is what catches that, and it has
    to hold at any depth, which is why it is a prefix test and not a glob.
    """
    assert hygiene.forbidden(path), path


@pytest.mark.parametrize(
    "path",
    [
        "src/taxiout/features.py",
        "reports/baseline.md",
        "docs/plan.md",
        "Makefile",
        ".gitignore",
        "data/challenge/README.md",
        f"src/{ASSISTANT_DIR}_notes.py",
    ],
)
def test_ordinary_paths_are_not_forbidden(path):
    """Including one that merely starts with the directory's name."""
    assert not hygiene.forbidden(path), path


def test_the_previously_forbidden_paths_still_are():
    for path in (
        "data/raw/EDDF.json.gz",
        "data/processed/taxi_distance/EDDF.csv",
        "data/challenge/ranking.parquet",
        "data/challenge/movements/training_2025_01.parquet",
        "notes.osm",
    ):
        assert hygiene.forbidden(path), path


def test_the_ignore_rule_and_the_gate_agree(tmp_path, monkeypatch):
    """Both guards, checked against each other rather than against habit.

    ``.gitignore`` keeps the paths out of a commit; ``FORBIDDEN`` fails the
    gate if that rule is ever removed. The failure mode is deleting one and
    trusting the other, so this asserts the ignore file still names both.
    """
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    entries = {line.strip() for line in ignore if line.strip()}
    assert f"{ASSISTANT_DIR}/" in entries
    assert ASSISTANT_FILE in entries


def test_a_tracked_assistant_file_fails_the_gate(tmp_path, monkeypatch):
    """End to end: if the ignore rule went away, check 1 would report it."""
    (tmp_path / ASSISTANT_DIR).mkdir()
    (tmp_path / ASSISTANT_DIR / "settings.local.json").write_text(
        "{}\n", encoding="utf-8"
    )
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    files = hygiene.tracked_files()
    assert f"{ASSISTANT_DIR}/settings.local.json" in files
    assert [f for f in files if hygiene.forbidden(f)]
