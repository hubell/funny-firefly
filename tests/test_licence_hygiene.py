# SPDX-License-Identifier: GPL-3.0-only
"""The licence gate's checks, and the property that lets it read its own source.

The words check 4 forbids are held by the gate as digests, and this file spells
none of them in any form, for the reason the gate's docstring gives: a test
that names what it forbids becomes a file the check has to be told to ignore,
and an ignored file is where the next one hides. Where a test needs a word to
find, it invents one and holds it the way the gate holds the real ones.
"""

from __future__ import annotations

import hashlib
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


def held(*words: str) -> tuple[tuple[str, int], ...]:
    """Invented words, held as the gate holds its own: digest and length."""
    return tuple((hashlib.sha256(w.encode()).hexdigest(), len(w)) for w in words)


INVENTED = "zorblax"
INVENTED_KEY = "made-with-zorblax"
WORDS = held(INVENTED, INVENTED_KEY)
DIGEST = WORDS[0][0]


# --------------------------------------------------------------------------
# the held words


def test_the_held_words_are_digests_with_their_lengths():
    assert len(hygiene.ATTRIBUTION_WORDS) == 7
    for digest, length in hygiene.ATTRIBUTION_WORDS:
        assert len(digest) == 64 and int(digest, 16) >= 0
        assert isinstance(length, int) and length > 0
    assert len({d for d, _ in hygiene.ATTRIBUTION_WORDS}) == 7


@pytest.mark.parametrize(
    "text",
    [
        f"Drafted with {INVENTED}.",
        f"DRAFTED WITH {INVENTED.upper()}",
        f"the {INVENTED}ing step",
        f"{INVENTED}-assisted",
        f"pre{INVENTED}4",
    ],
)
def test_a_held_word_is_found_alone_and_inside_a_compound(text):
    assert hygiene.attribution_in(text, WORDS) == [DIGEST]


def test_a_hyphenated_key_is_found_as_one_word():
    text = f"{INVENTED_KEY.title()}: someone"
    assert hygiene.attribution_in(text, WORDS) == sorted({WORDS[0][0], WORDS[1][0]})


def test_ordinary_prose_is_not_flagged():
    for benign in (
        "a clause in the licence",
        "co-authored a paper with two colleagues",
        "the open air apron",
        "an anthropogenic effect",
        "a copy of the pilot's log",
    ):
        assert hygiene.attribution_in(benign) == [], benign


def test_the_gate_and_this_file_hold_no_word_they_forbid():
    """The property the digests exist for: the gate reads its own source."""
    assert hygiene.attribution_in(SCRIPT.read_text(encoding="utf-8")) == []
    assert hygiene.attribution_in(Path(__file__).read_text(encoding="utf-8")) == []


def test_the_repository_itself_is_clean():
    """The tree, as the gate reads it: every tracked file that reads as text."""
    files = hygiene.tracked_files()
    assert files, "no tracked files found; the walk is broken"
    assert hygiene.attribution_hits(files) == {}


def test_the_history_holds_no_word_outside_the_accepted_commits():
    """Every commit the clone holds; a shallow one holds only the last."""
    history = hygiene.history_hits()
    if history is None:
        pytest.skip("git is not to hand")
    hits, commits = history
    assert commits >= 1
    accepted = {c for c, _ in hygiene.HISTORY_ACCEPTED}
    assert set(hits) <= accepted, sorted(set(hits) - accepted)


def test_a_planted_word_is_found_in_any_text_file(tmp_path, monkeypatch):
    """Every tracked file that reads as text, a data fixture included."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text(f"Drafted with {INVENTED}.\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fixture.json").write_text(f'{{"tool": "{INVENTED}"}}\n')
    (tmp_path / "data" / "blob.bin").write_bytes(b"\xff\xfe" + INVENTED.encode())
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    monkeypatch.setattr(hygiene, "ATTRIBUTION_WORDS", WORDS)
    hits = hygiene.attribution_hits(
        ["docs/note.md", "data/fixture.json", "data/blob.bin"]
    )
    assert hits == {"docs/note.md": [DIGEST], "data/fixture.json": [DIGEST]}


def test_the_history_scan_reads_messages_and_added_lines_only():
    """A removed line is the fix, and a context line or a hunk header is not
    the commit's own writing; an added line is read whatever it starts with,
    and so is the path a commit writes to."""
    head = "\0{}\n{}\n\n{} <x@x>\n{} <x@x>\n\x01\n"
    log = (
        head.format("a" * 40, "Add the step", "A", "A")
        + "diff --git a/f b/f\n--- a/f\n+++ b/f\n"
        + f"@@ -1,2 +1,2 @@ {INVENTED}\n {INVENTED}\n-{INVENTED}\n+plain\n"
        + head.format("b" * 40, f"Made with {INVENTED}", "B", "B")
        + head.format("c" * 40, "Plain", "C", "C")
        + "diff --git a/g b/g\n--- a/g\n+++ b/g\n@@ -0,0 +1 @@\n"
        + f"+uses {INVENTED}\n"
        + head.format(
            "d" * 40, "Quote a patch\n\ndiff --git a/x b/x", "D", f"{INVENTED} D"
        )
        + head.format("e" * 40, "Plain", "E", "E")
        + "diff --git a/h b/h\n--- a/h\n+++ b/h\n@@ -0,0 +1 @@\n"
        + f"+++{INVENTED}\n"
        + head.format("f" * 40, "Plain", "F", "F")
        + f"diff --git a/{INVENTED}.md b/{INVENTED}.md\nnew file mode 100644\n"
        + f"--- /dev/null\n+++ b/{INVENTED}.md\n"
    )
    hits, commits = hygiene.scan_log(log, WORDS)
    assert commits == 6
    assert hits == {c * 40: [DIGEST] for c in "bcdef"}


def test_every_accepted_commit_is_a_full_id_with_its_reason():
    for commit, why in hygiene.HISTORY_ACCEPTED:
        assert len(commit) == 40 and int(commit, 16) >= 0
        assert why


# --------------------------------------------------------------------------
# the forbidden-path rule
#
# The assistant-configuration names are held as digests too, so the rule is
# exercised here with invented ones held the same way.


def _invented_names(monkeypatch):
    names = frozenset(
        hashlib.sha256(n.encode()).hexdigest() for n in (".zorblax", "zorblax.md")
    )
    monkeypatch.setattr(hygiene, "FORBIDDEN_NAMES", names)


def test_the_two_names_are_held_as_digests():
    assert len(hygiene.FORBIDDEN_NAMES) == 2
    assert all(len(d) == 64 for d in hygiene.FORBIDDEN_NAMES)


@pytest.mark.parametrize(
    "path",
    [
        ".zorblax/settings.local.json",
        ".zorblax/agents/reviewer.md",
        ".zorblax/skills/deep/nested/file.txt",
        ".zorblax",
        "ZORBLAX.md",
        "docs/ZORBLAX.md",
        "src/.zorblax/notes.txt",
    ],
)
def test_a_held_name_is_forbidden_at_every_depth(path, monkeypatch):
    _invented_names(monkeypatch)
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
        "src/.zorblax_notes.py",
    ],
)
def test_ordinary_paths_are_not_forbidden(path, monkeypatch):
    """Including one that merely starts with a held name."""
    _invented_names(monkeypatch)
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


def test_the_ignore_file_spells_no_held_word():
    """The two paths are ignored through .git/info/exclude, which git never
    publishes; the published ignore file names neither."""
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert hygiene.attribution_in(text) == []


def _quiet(monkeypatch):
    """Every check but the one a test is about has nothing to read."""
    monkeypatch.setattr(hygiene, "osm_stand_identifiers", lambda: set())
    monkeypatch.setattr(hygiene, "challenge_stand_identifiers", lambda: set())
    monkeypatch.setattr(hygiene, "history_hits", lambda: None)


def test_a_tracked_file_under_a_held_name_fails_the_gate(tmp_path, monkeypatch, capsys):
    """End to end: if the ignore rule went away, check 1 would report it."""
    _invented_names(monkeypatch)
    _quiet(monkeypatch)
    (tmp_path / ".zorblax").mkdir()
    (tmp_path / ".zorblax" / "settings.local.json").write_text("{}\n")
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    files = hygiene.tracked_files()
    assert ".zorblax/settings.local.json" in files
    assert hygiene.main() == 1
    assert ".zorblax/settings.local.json" in capsys.readouterr().out


# --------------------------------------------------------------------------
# attribution, and the licence line


def test_the_credit_without_the_licence_named_fails(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "a.md").write_text("Data (c) OpenStreetMap contributors.\n")
    (tmp_path / "reports" / "b.md").write_text(
        "Data (c) OpenStreetMap contributors, ODbL 1.0.\n"
    )
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    _quiet(monkeypatch)
    monkeypatch.setattr(hygiene, "tracked_files", lambda: ["reports/b.md"])
    assert hygiene.main() == 0
    monkeypatch.setattr(
        hygiene, "tracked_files", lambda: ["reports/a.md", "reports/b.md"]
    )
    assert hygiene.main() == 1


def test_a_source_file_without_the_licence_line_is_named(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "ok.py").write_text(
        "#!/usr/bin/env python3\n# SPDX-License-Identifier: GPL-3.0-only\n"
    )
    (tmp_path / "scripts" / "late.py").write_text(
        '"""A docstring first."""\n\n# SPDX-License-Identifier: GPL-3.0-only\n'
    )
    (tmp_path / "Makefile").write_text("all:\n")
    (tmp_path / "notes.txt").write_text("no licence line needed\n")
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    files = ["scripts/ok.py", "scripts/late.py", "Makefile", "notes.txt"]
    assert hygiene.spdx_missing(files) == ["scripts/late.py", "Makefile"]
