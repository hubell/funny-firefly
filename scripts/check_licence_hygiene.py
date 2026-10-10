#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Verify that nothing OpenStreetMap-derived is committed to this repository.

This repository ships GPLv3 in totality. OpenStreetMap is ODbL, and its
share-alike obligation is not this project's to waive, so no OSM content may
appear in a tracked file -- and "no OSM content" has to be a checkable property
rather than a habit, because the failure mode is forgetting once, months from
now, in a hurry.

Five checks:

1. **No non-redistributable artifact is tracked.** Three classes, for three
   different reasons. OSM-derived artifacts -- the distance matrices, the raw
   Overpass cache, any feature frame built from them -- are Derivative
   Databases carrying ODbL. Challenge data is supplied under terms that permit
   no redistribution whatsoever, which makes it the stricter case and the one
   where a mistake cannot be undone by relicensing. The assistant-configuration
   directory and instruction file are the third: they carry no licence problem
   but they are exactly the trace check 4 exists to keep out of the published
   artifact. They are ignored through ``.git/info/exclude``, which git never
   publishes, and forbidden here by the digest of their lower-cased names, so
   the two guards fail independently and neither spells them.

2. **No list of literal stand identifiers appears in prose.** Stand
   identifiers are the ones that matter: unlike runway designators, which are
   published in AIPs and in the public-domain OurAirports data and so are
   nobody's to restrict, a stand identifier at these airports comes from OSM
   or from the challenge data, and neither may be excerpted here. Reports
   describe identifier *shapes* instead; a shape is a description, an
   identifier is an extract. The challenge side is checked only where the
   data is present locally, which CI is not.

   Four deliberate limits, stated rather than hidden, because a check that
   cries wolf gets ignored, and printed with the verdict so that it is not
   read as wider than it is:

   - Only identifiers with a letter *and* a digit and at least three characters
     are considered. Bare numerics are indistinguishable from ordinary numbers,
     and two-character tokens collide with everything.
   - Only prose is scanned -- the tracked Markdown files. Code fixtures are
     invented by hand, and short invented names collide with real identifiers
     by chance: an invented helipad ref and two invented stand names all
     tripped an earlier version of this check.
   - A file is flagged only at three or more distinct matches. One match is a
     coincidence; a list is a leak, and a list is what the reports would carry
     if this regressed.
   - A name that coincides with a stand identifier is exempt only in the one
     file and for the tokens ``IDENTIFIER_EXEMPTIONS`` names, with its reason.

   So this catches the realistic failure -- a report reverting to printing
   examples -- and not a single unlucky string in a test.

3. **Attribution is present** wherever OSM was used to produce a tracked file:
   every report and the paper carry the contributors' credit and name the
   licence beside it, the two things a notice for a Produced Work under ODbL
   1.0 §4.3 has to say.

4. **No assistant attribution anywhere.** The repository carries no trace of
   what wrote it -- no tool name in a file, a comment, a docstring or a commit
   message, and no co-authorship trailer naming one. This is a property of the
   published artifact, so it is checked over every tracked file that reads as
   text and over the whole history: each commit's message, author and
   committer, and the lines its patch adds. A line a patch removes is the fix
   and is not read. The commits made before the freeze that add one, which
   could be rewritten only at the cost of every later commit id, are accepted
   by id in ``HISTORY_ACCEPTED`` and listed with the verdict.

   **The words are held as the sha256 of their lower-cased forms**, beside
   their lengths, and spelled nowhere in any form: not plainly, not in bracket
   classes, not assembled from fragments. A check that spells what it forbids
   matches its own source, and a spelling split to dodge that is still a
   spelling any reader reassembles at once. Every run of letters, digits and
   hyphens in a text is lower-cased and cut into windows of each held length,
   and a window whose digest is held is a hit, so a word is found inside a
   compound as well as alone. A hit is reported by its digest.

5. **Every source file carries the licence.** Each tracked ``*.py``, ``*.sh``,
   ``*.yml``, ``Makefile`` and ``Containerfile`` holds
   ``SPDX-License-Identifier: GPL-3.0-only`` within its first two lines, so
   "GPLv3 in totality" is checked rather than kept by habit.

Exit code is non-zero if anything fails, so it can gate a commit or a push.

Usage
-----
    python scripts/check_licence_hygiene.py
    make licence-check
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.coverage import stand_identifier  # noqa: E402

CACHE = REPO / "data" / "raw"

# The assistant-configuration directory and instruction file, held as the
# sha256 of each one's lower-cased name, so that this file spells neither; the
# docstring says why. A path is forbidden when any of its components, lower-
# cased, has one of these digests.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "e026f56c08dcdea441a2aa620ab410ce91162062609b2430f5a13bc9b7cf93f1",
        "bccc86317c281a19e56f3f941077ea61b3fb37d5bd527c518b92b188c4461251",
    }
)

# Paths that must never be tracked, as patterns against repo-relative paths.
# Three distinct reasons, all fatal. The OSM-derived artifacts carry ODbL, which
# this GPLv3-in-totality repository cannot host. The challenge data carries the
# challenge's own terms, which permit no redistribution at all -- so it is the
# stricter of the two, and the one where a mistake is unrecoverable. The
# assistant-configuration paths, ``FORBIDDEN_NAMES`` above, carry neither
# problem and are forbidden for check 4's reason instead: they are a trace of
# what wrote the repository.
#
# A pattern ending in ``/`` is a directory rule and covers everything beneath
# it at any depth; every other pattern is a glob matched by ``Path.match``,
# which is anchored on the right and whose ``**`` is not recursive.
FORBIDDEN = (
    # OpenStreetMap-derived: ODbL
    "data/raw/*",
    "data/interim/*",
    "data/processed/taxi_distance/*.csv",
    "*.osm",
    "*.pbf",
    # Challenge data and anything built from it: not redistributable
    "data/challenge/*.parquet",
    "data/challenge/**/*.parquet",
    "challenge_data/*",
    "**/*.parquet",
)


def forbidden(path: str) -> bool:
    """Whether a repo-relative path matches a ``FORBIDDEN`` pattern.

    Directory rules are a prefix test rather than a glob because ``Path.match``
    cannot express "everything under here at any depth": its ``**`` matches one
    component, so a two-line glob would silently stop at the depth it was
    written for.
    """
    if any(
        hashlib.sha256(part.lower().encode()).hexdigest() in FORBIDDEN_NAMES
        for part in Path(path).parts
    ):
        return True
    for pat in FORBIDDEN:
        if pat.endswith("/"):
            if path == pat.rstrip("/") or path.startswith(pat):
                return True
        elif Path(path).match(pat):
            return True
    return False


# A tracked file that reports OSM-derived figures must attribute OSM: the
# contributors' credit, and the licence named beside it.
NEEDS_ATTRIBUTION = ("reports/*.md", "docs/paper.md")
ATTRIBUTION = "OpenStreetMap contributors"
LICENCE_NAMED = re.compile(r"ODbL|Open Database License")


def tracked_files() -> list[str]:
    """Every file git would ship, whatever the current commit state.

    Deliberately not ``git ls-files``: that reports what is committed *now*,
    and the interesting moment is just before a commit, when the answer would
    be the previous tree. Walking the filesystem and subtracting what
    ``.gitignore`` excludes gives what the next commit would contain.
    """
    candidates = [
        p.relative_to(REPO).as_posix()
        for p in REPO.rglob("*")
        if p.is_file()
        and ".git/" not in p.relative_to(REPO).as_posix()
        and "__pycache__" not in p.as_posix()
    ]
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-z", "--stdin"],
            cwd=REPO,
            input="\0".join(candidates),
            capture_output=True,
            text=True,
        )
        # NUL-delimited in both directions. The default line format splits a
        # path containing a space into fragments that match no candidate, and
        # C-quotes any path holding a non-ASCII byte; either leaves an ignored
        # file looking un-ignored, so it gets scanned and counted. That errs
        # conservatively -- more files checked, never fewer -- but it makes the
        # printed file count wrong, and a count nobody can trust is one nobody
        # reads. The output is NUL-*terminated*, hence the empty final field.
        ignored = set(proc.stdout.split("\0")) - {""}
    except FileNotFoundError:
        ignored = set()
    return sorted(f for f in candidates if f not in ignored)


# Files where a leak would realistically land: prose that might list examples.
PROSE = ("reports/*.md", "docs/*.md", "*.md")

# The words check 4 forbids: the sha256 of each one's lower-cased form, and its
# length, which the window scan cuts to. Seven: six tool names and the
# co-authorship trailer's key. Spelled nowhere; the docstring says why.
ATTRIBUTION_WORDS: tuple[tuple[str, int], ...] = (
    ("750bd39768d2b52a276a0f16e37e8d89cc51808ed2aee34292a613728c215aec", 6),
    ("c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d", 6),
    ("c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131", 9),
    ("3ea125d0bff386e6754b3782b300016fc79a9cf8f8669c0a5c3db64467ddb681", 7),
    ("60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c", 7),
    ("7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef", 6),
    ("7813670ff886c2f9397c151cf25b097663b3b1edda5c403ebf162982972b2c8e", 14),
)

# A run of the characters a held word is written in: letters, digits and the
# hyphen of the trailer's key.
_RUN = re.compile(r"[a-z0-9-]+")

# Commits made before the freeze that carry a held word in their patches.
# Accepted by the owner and listed by full id, because rewriting them would
# change every later commit id, the tags among them; every other commit, and
# every one from the freeze on, is held to the rule. The third of 2 September
# only removes the words the first two added, so it is not a hit; it is listed
# with them as the owner accepted the three.
HISTORY_ACCEPTED: tuple[tuple[str, str], ...] = (
    (
        "401f617b4e867b463b5c043d55206d3ac0ed1622",
        "2 September 2026: a grep line in a commit script of that day, on the"
        " remote since then and removed the same day",
    ),
    (
        "ad8406060c025c1a6cace7c264fc78f42cbf7412",
        "2 September 2026: the same grep line in a second commit script, on the"
        " remote since then and removed the same day",
    ),
    (
        "033215da67bfb7ee6545e850671074bb4af3279e",
        "2 September 2026: removes the two grep lines above, on the remote since"
        " then; it adds no held word",
    ),
    (
        "cf5e6be618a57a96bd1faeb007aab63e4037cd09",
        "5 September 2026: the two assistant-configuration paths written into"
        " .gitignore, inside the freeze tag; moved to .git/info/exclude later",
    ),
)

# Every tracked source file carries this within its first two lines.
SPDX_LINE = "SPDX-License-Identifier: GPL-3.0-only"
NEEDS_SPDX = ("*.py", "*.sh", "*.yml", "Makefile", "Containerfile")

# Below this many distinct matches in one file, treat it as coincidence.
LEAK_THRESHOLD = 3

# Tokens that are names in a named file and happen to be stand identifiers
# too, each file with its tokens and the reason. Narrow on purpose: a token is
# exempt in its file alone, and any other identifier in that file still counts.
# The owner's decisions of 7 October 2026, one for each file.
IDENTIFIER_EXEMPTIONS: dict[str, tuple[frozenset[str], str]] = {
    "reports/preregistered-2026-10-06-last-week.md": (
        frozenset({"C10", "C11", "C12"}),
        "the names of three candidates of group B, inside the copy's rules,"
        " which may not change a word; that stands carry the same names at"
        " some airports is a coincidence of form",
    ),
    "reports/preregistered-2026-10-06-airports.md": (
        frozenset({"C10", "C11", "C12", "C13", "C14"}),
        "the names of five candidates of group B, inside the copy's rules,"
        " which may not change a word; that stands carry the same names at"
        " some airports is a coincidence of form",
    ),
    "reports/group-b.md": (
        frozenset({"C10", "C11", "C12", "C13", "C14"}),
        "the same five candidate names in the report that reads them, whose"
        " verdict lines are the rule's words; its generator's guard exempts"
        " the same five and refuses every other identifier",
    ),
    "reports/preregistered-2026-10-07-group-c.md": (
        frozenset({"C10", "C11", "C14", "C15", "C16"}),
        "the names of group C's two candidates and of the three group B"
        " candidates its note names, which the copy keeps as the note words"
        " them, inside its rules and out; that stands carry the same names at"
        " some airports is a coincidence of form",
    ),
    "reports/preregistered-2026-10-09-jj-check.md": (
        frozenset({"C18"}),
        "the name of a candidate its note names, which the copy keeps as the"
        " note words it; that a stand carries the same name at some airport is"
        " a coincidence of form",
    ),
    "reports/preregistered-2026-10-09-records.md": (
        frozenset({"C18"}),
        "the name of a candidate its note names, which the copy keeps as the"
        " note words it; that a stand carries the same name at some airport is"
        " a coincidence of form",
    ),
}


def osm_stand_identifiers() -> set[str]:
    """Distinctive stand identifiers present in the local OSM cache.

    Distinctive means at least three characters with both a letter and a digit:
    short or purely alphabetic identifiers collide with ordinary prose.
    """
    out: set[str] = set()
    for ap in AIRPORTS:
        path = CACHE / f"{ap.icao}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            elements = json.load(fh)["elements"]
        for el in elements:
            tags = el.get("tags") or {}
            if tags.get("aeroway") != "parking_position":
                continue
            found = stand_identifier(tags)
            if not found:
                continue
            value = found[1].strip()
            if (
                len(value) >= 3
                and re.search(r"[A-Za-z]", value)
                and re.search(r"[0-9]", value)
            ):
                out.add(value)
    return out


def challenge_stand_identifiers() -> set[str]:
    """Distinctive ``STAND_mvt`` values from the local challenge data, if any.

    The challenge data is not redistributable at all, so its identifiers are
    held to the same rule as OSM's. Reads only the one column, and only when
    the data and ``pyarrow`` are both present; CI has neither and skips.
    """
    files = sorted((REPO / "data" / "challenge").rglob("*.parquet"))
    if not files:
        return set()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return set()
    out: set[str] = set()
    for path in files:
        if "STAND_mvt" not in pq.read_schema(path).names:
            continue
        column = pq.read_table(path, columns=["STAND_mvt"])["STAND_mvt"]
        for value in column.unique().to_pylist():
            if not value:
                continue
            value = str(value).strip()
            if (
                len(value) >= 3
                and re.search(r"[A-Za-z]", value)
                and re.search(r"[0-9]", value)
            ):
                out.add(value)
    return out


def attribution_in(text: str, words=None) -> list[str]:
    """The digests of the held words a text holds, sorted; empty is clean.

    Every run of ``_RUN`` characters, lower-cased, is cut into windows of each
    held length, and a window whose sha256 is held is a hit, so a word is
    found inside a compound as well as alone. That is the trade the check
    makes, and it is known rather than discovered: an ordinary English word
    that contains a held one is a hit too, and a person reads it once,
    where a pattern on word boundaries would miss the compound forms these
    names take. ``words`` defaults to ``ATTRIBUTION_WORDS``; the suite
    passes invented ones.
    """
    held = dict(words if words is not None else ATTRIBUTION_WORDS)
    lengths = sorted(set(held.values()))
    found: set[str] = set()
    for run in set(_RUN.findall(text.lower())):
        for n in lengths:
            for i in range(len(run) - n + 1):
                digest = hashlib.sha256(run[i : i + n].encode()).hexdigest()
                if digest in held:
                    found.add(digest)
    return sorted(found)


def text_of(f: str) -> str | None:
    """A tracked file's text, or None where it does not read as UTF-8."""
    try:
        return (REPO / f).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def attribution_hits(files: list[str]) -> dict[str, list[str]]:
    """Held words in the tracked files that read as text; empty is clean."""
    hits: dict[str, list[str]] = {}
    for f in files:
        text = text_of(f)
        if text is None:
            continue
        found = attribution_in(text)
        if found:
            hits[f] = found
    return hits


def scan_log(log: str, words=None) -> tuple[dict[str, list[str]], int]:
    """Held words per commit in ``git log -p`` output, and the commits read.

    The output is the one ``history_hits`` asks for: each commit opens with a
    NUL, then its full id, its message, its author and committer, a byte 0x01
    that closes them, and its patch. What is read is the message, the two
    names and every line the patch adds, the ``+++`` line among them, which
    carries the path a commit writes to. A removed line is the fix and is not
    read, nor is a context line or a hunk header. Two limits, stated: a merge
    commit's own resolution is not shown by ``git log -p`` and is not read,
    its parents' commits are; and a clone made at a depth reads only the
    commits it holds, which the verdict counts.
    """
    hits: dict[str, list[str]] = {}
    chunks = log.split("\0")[1:]
    for chunk in chunks:
        head, _, patch = chunk.partition("\x01")
        commit = head.strip().split("\n", 1)[0].strip()
        added = [line[1:] for line in patch.splitlines() if line.startswith("+")]
        found = attribution_in(head + "\n" + "\n".join(added), words)
        if found:
            hits[commit] = found
    return hits, len(chunks)


def history_hits() -> tuple[dict[str, list[str]], int] | None:
    """``scan_log`` over every commit, or None where git is not to hand."""
    try:
        proc = subprocess.run(
            [
                "git",
                "log",
                "-p",
                "--no-color",
                "--no-ext-diff",
                "--format=%x00%H%n%B%n%an <%ae>%n%cn <%ce>%n%x01",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return scan_log(proc.stdout)


def spdx_missing(files: list[str]) -> list[str]:
    """Tracked source files without ``SPDX_LINE`` in their first two lines."""
    out = []
    for f in files:
        if not any(Path(f).match(pat) for pat in NEEDS_SPDX):
            continue
        text = text_of(f) or ""
        if SPDX_LINE not in "\n".join(text.splitlines()[:2]):
            out.append(f)
    return out


def main() -> int:
    files = tracked_files()
    failures: list[str] = []
    print(f"checking {len(files)} tracked files\n")

    # 1 -- forbidden artifacts
    bad = sorted({f for f in files if forbidden(f)})
    if bad:
        failures.append(
            "non-redistributable or non-publishable paths are tracked:\n    "
            + "\n    ".join(bad)
        )
    else:
        print(
            "  ok  no OSM-derived, challenge or assistant-configuration file is tracked"
        )

    # 2 -- literal identifiers, from either non-redistributable source
    osm_ids = osm_stand_identifiers()
    challenge_ids = challenge_stand_identifiers()
    ids = osm_ids | challenge_ids
    if not ids:
        print("  --  no OSM cache or challenge data present; identifier check skipped")
    else:
        hits: dict[str, list[str]] = {}
        prose = [f for f in files if any(Path(f).match(pat) for pat in PROSE)]
        for f in prose:
            try:
                text = (REPO / f).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            # The substring test first: it decides nothing the pattern would
            # not, and spares the pattern on every identifier the file lacks.
            exempt = IDENTIFIER_EXEMPTIONS.get(f, (frozenset(), ""))[0]
            found = sorted(
                i
                for i in ids - exempt
                if i in text
                and re.search(r"(?<![\w])" + re.escape(i) + r"(?![\w])", text)
            )
            if len(found) >= LEAK_THRESHOLD:
                hits[f] = found
        if hits:
            failures.append(
                "literal stand identifiers appear in tracked files:\n    "
                + "\n    ".join(f"{f}: {v}" for f, v in hits.items())
            )
        else:
            sources = []
            if osm_ids:
                sources.append(f"{len(osm_ids)} OSM")
            if challenge_ids:
                sources.append(f"{len(challenge_ids)} challenge")
            print(
                f"  ok  no tracked Markdown file ({len(prose)} read) carries"
                f" {LEAK_THRESHOLD} or more of the "
                + " and ".join(sources)
                + " stand identifiers; code fixtures are invented and not read"
            )

    # 3 -- attribution
    missing = []
    for f in files:
        if not any(Path(f).match(pat) for pat in NEEDS_ATTRIBUTION):
            continue
        text = (REPO / f).read_text(encoding="utf-8")
        if ATTRIBUTION not in text or not LICENCE_NAMED.search(text):
            missing.append(f)
    if missing:
        failures.append(
            "files reporting OSM-derived figures without the contributors'"
            " credit and the licence named:\n    " + "\n    ".join(missing)
        )
    else:
        print(
            "  ok  every report and the paper credit the OpenStreetMap"
            " contributors and name the ODbL"
        )

    # 4 -- assistant attribution, in the tree and in the history
    attribution = attribution_hits(files)
    read = sum(1 for f in files if text_of(f) is not None)
    history = history_hits()
    accepted = {c for c, _ in HISTORY_ACCEPTED}
    if history is not None:
        commits, n_commits = history
        attribution.update(
            {f"commit {c}": v for c, v in commits.items() if c not in accepted}
        )
    if attribution:
        failures.append(
            "a held word appears in the repository (by digest):\n    "
            + "\n    ".join(f"{k}: {[d[:12] for d in v]}" for k, v in attribution.items())
        )
    else:
        seen = sorted(c[:7] for c in accepted if history and c in history[0])
        quiet = sorted(c[:7] for c in accepted if not (history and c in history[0]))
        print(
            f"  ok  no held word in the {read} tracked files that read as text"
            + (
                f", nor in the messages and added lines of the {n_commits} commit(s)"
                f" in the history; accepted by id: {', '.join(seen) or 'none'}"
                + (f" (and {', '.join(quiet)}, accepted, no hit here)" if quiet else "")
                if history is not None
                else "; no history to read here"
            )
        )

    # 5 -- the licence line in every source file
    unlicensed = spdx_missing(files)
    if unlicensed:
        failures.append(
            f"source files without {SPDX_LINE} in their first two lines:\n    "
            + "\n    ".join(unlicensed)
        )
    else:
        n = sum(1 for f in files if any(Path(f).match(p) for p in NEEDS_SPDX))
        print(f"  ok  {SPDX_LINE} in the first two lines of all {n} source files")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        print("\nSee data/processed/taxi_distance/ODbL-NOTICE.md for what may ship.")
        return 1
    print("licence hygiene clean: GPLv3 in totality, no OSM content tracked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
