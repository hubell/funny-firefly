#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit, after the freeze: the paper and the README say what the
# clean-checkout reproduction showed, the deposit has its DOI, and the one
# report a clean checkout cannot regenerate is a dated record.
#
# WHAT THE REPRODUCTION FOUND. On 27 September 2026 a fresh clone at the tag
# freeze-2026-10-04, with the challenge data and the two cached tables brought
# in and the geometry fetched at its pins, rebuilt the 35 matrix files and the
# 52 residual files byte for byte, regenerated fourteen reports in which no
# figure moved, and built the submission byte for byte. One report could not
# be regenerated: reports/determinism.md reads 90 records of a measurement
# campaign under data/processed/determinism/, made under the parameters of its
# day, holding arrays derived from the challenge data; nothing reruns the
# campaign and the records are not redistributed. And two sentences of the
# paper's §10 did not hold as written: the geometry reports need the fetched
# answers or the deposit, not "no network", and a reader needs the METAR and
# OurAirports caches as well as the challenge data.
#
# WHAT MOVES, IN ONE PLACE:
#   * reports/reproduction-2026-09-27.md, new: the dated record of that run,
#     every figure from its logs. Frozen under its digest.
#   * src/taxiout/provenance.py: reports/determinism.md and the new record join
#     FROZEN_DOCUMENTS; determinism.md leaves REPORT_INPUTS_DECLARED, so the
#     gate holds it to its digest and no longer asks a clean checkout for
#     records it cannot have. The file itself is unchanged since the freeze.
#   * docs/provenance.md, regenerated from the register.
#   * docs/paper.md: §10 says what a reader brings and what a clean checkout
#     reproduces, names the record, and states the mirror rule; §2.3 and the
#     preamble name the deposit and its DOI; the working-draft preamble and the
#     bracketed status tags are gone. The two [score] marks stay for the
#     upload's commit. No figure moves, and check_paper.py --strict passes.
#   * README.md: the Reproducing section gains what a reader brings and the
#     deposit paragraph, the Data sources section names the deposit, and the
#     48 em dashes it carried since before the paper pass are replaced.
#   * Makefile, scripts/check_penalty_sensitivity.py, docs/modelling_notes.md:
#     three stale statements replaced by measured ones, each naming the record
#     it comes from: 'make reports' and the penalty sweep take minutes, not an
#     hour; the ladder under the parameters that ship; the cache is pinned and
#     rebuilt, not "predates pinning".
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * it regenerates no report and moves no figure. Every report but the new
#     record is byte-identical to the tag's.
#   * it touches nothing under src/ but the register, and no test.
#   * it makes no tag: the freeze tag stays where it is, on the model.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

PY="${PYTHON:-$([ -x .venv/bin/python ] && echo "$PWD/.venv/bin/python" || echo python3)}"
RUFF="$([ -x .venv/bin/ruff ] && echo "$PWD/.venv/bin/ruff" || echo ruff)"
case "$PY" in /*) ;; */*) PY="$PWD/$PY" ;; esac

BASE="bfe43f1"
TAG="freeze-2026-10-04"
PATHS="Makefile
README.md
docs/modelling_notes.md
docs/paper.md
docs/provenance.md
reports/reproduction-2026-09-27.md
scripts/check_penalty_sensitivity.py
scripts/commit_docs.sh
src/taxiout/provenance.py"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say "this sits on the frozen commit, and the tag points at it"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "$BASE^{commit}")" ]; then
    echo "HEAD is not $BASE; this script expects to sit on the freeze commit"
    exit 1
fi
if [ "$(git rev-parse "$TAG^{commit}")" != "$(git rev-parse HEAD)" ]; then
    echo "the tag $TAG does not point at HEAD"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' HEAD), tagged $TAG"

say "the working tree holds these 9 paths and nothing else"
expected="$(printf '%s\n' "$PATHS" | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok  9 paths"

say "1/8  the DOI is written, the same in the README and the paper, and no mark is left"
doi_readme="$(grep -oE '10\.5281/zenodo\.[0-9]+' README.md | sort -u)"
doi_paper="$(grep -oE '10\.5281/zenodo\.[0-9]+' docs/paper.md | sort -u)"
if [ -z "$doi_readme" ] || [ -z "$doi_paper" ]; then
    echo "no Zenodo DOI in README.md or docs/paper.md; write it where [DOI] stands"
    exit 1
fi
if [ "$(printf '%s\n' "$doi_readme" | wc -l)" != "1" ] || [ "$doi_readme" != "$doi_paper" ]; then
    echo "the DOI is not one and the same in both files: README '$doi_readme', paper '$doi_paper'"
    exit 1
fi
if grep -qF '[DOI]' README.md docs/paper.md; then
    echo "a [DOI] mark is still in place"
    exit 1
fi
echo "ok  DOI $doi_readme"

say "2/8  the paper: no drafting apparatus, the score marks kept, no em dash"
if grep -qE '\[settled|\[pending|Working draft' docs/paper.md; then
    echo "docs/paper.md still carries drafting apparatus"
    exit 1
fi
n="$(grep -cF '**[score]**' docs/paper.md || true)"
if [ "$n" != "2" ]; then
    echo "expected the two [score] marks for the upload's commit and found $n"
    exit 1
fi
for f in docs/paper.md README.md reports/reproduction-2026-09-27.md Makefile \
         scripts/check_penalty_sensitivity.py; do
    if grep -q -- "$(printf '\xe2\x80\x94')" "$f"; then
        echo "$f carries an em dash"
        exit 1
    fi
done
# docs/modelling_notes.md keeps the em dashes of its earlier drafting outside
# the one paragraph this commit rewrites; that paragraph is checked by its text.
if sed -n '/^\*\*The committed matrices were built from an unpinned fetch/,/^$/p' docs/modelling_notes.md \
   | grep -q -- "$(printf '\xe2\x80\x94')"; then
    echo "the rewritten paragraph of docs/modelling_notes.md carries an em dash"
    exit 1
fi
echo "ok  no [settled], no [pending], 2 [score] marks, no em dash in the five prose files or the rewritten paragraph"

say "3/8  reports/ moves in the new record alone, and determinism.md is the tag's"
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- reports/ \
         | sed 's/^...//')"
if [ "$moved" != "reports/reproduction-2026-09-27.md" ]; then
    echo "reports/ should move in the new record alone; it moves in: ${moved:-nothing}"
    exit 1
fi
git show "HEAD:reports/determinism.md" > "$work/determinism.head.md"
if ! cmp -s "$work/determinism.head.md" reports/determinism.md; then
    echo "reports/determinism.md differs from the tag's; a dated record is not edited"
    exit 1
fi
echo "ok  one new record; determinism.md unchanged"

say "4/8  the frozen documents, the two new ones included, are at their digests"
"$PY" - <<'PY'
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "src")
from taxiout.provenance import FROZEN_DOCUMENTS, REPORT_INPUTS_DECLARED

paths = [f.path for f in FROZEN_DOCUMENTS]
for want in ("reports/determinism.md", "reports/reproduction-2026-09-27.md"):
    if want not in paths:
        raise SystemExit(f"{want} is not in FROZEN_DOCUMENTS")
for f in FROZEN_DOCUMENTS:
    have = hashlib.sha256(Path(f.path).read_bytes()).hexdigest()
    if have != f.sha256:
        raise SystemExit(f"{f.path} is not at its digest")
if any(r == "reports/determinism.md" for _g, r in REPORT_INPUTS_DECLARED):
    raise SystemExit("reports/determinism.md is still in REPORT_INPUTS_DECLARED")
print(f"  ok  {len(FROZEN_DOCUMENTS)} dated records at their digests; determinism.md is one")
PY

say "5/8  docs/provenance.md is what the register generates"
"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import check_provenance as cp

if cp.document() != Path("docs/provenance.md").read_text(encoding="utf-8"):
    raise SystemExit("docs/provenance.md is not what the register generates")
print("  ok")
PY

say "6/8  nothing else under src/, and models.py is the tag's"
if [ -n "$(git -c core.fileMode=false status --porcelain -- src/ | grep -v 'src/taxiout/provenance.py')" ]; then
    echo "src/ moves outside the register"
    exit 1
fi
echo "ok  $(git rev-parse --short HEAD:src/taxiout/models.py), unchanged"

say "7/8  the gates, and both freeze conditions"
"$RUFF" check .
"$RUFF" format --check .
"$PY" -m pytest -q
"$PY" scripts/check_licence_hygiene.py
"$PY" scripts/check_provenance.py > "$work/provenance.txt" || {
    cat "$work/provenance.txt"
    echo "the provenance gate fails; it cannot be committed failing"
    exit 1
}
n="$(grep -cE '^  ok  ' "$work/provenance.txt" || true)"
if [ "$n" != "14" ]; then
    echo "expected 14 verdicts and found $n; a check that is not called cannot fail"
    cat "$work/provenance.txt"
    exit 1
fi
"$PY" scripts/check_provenance.py --strict > "$work/strict.txt" || {
    cat "$work/strict.txt"
    echo "check_provenance.py --strict fails, and it held at the freeze"
    exit 1
}
echo "ok  14 provenance checks green, and --strict passes"
"$PY" scripts/check_paper.py --strict > "$work/paper_strict.txt" || {
    cat "$work/paper_strict.txt"
    echo "check_paper.py --strict fails"
    exit 1
}
summary="$(grep -m1 -E '^      [0-9]+ figures in the ' "$work/paper_strict.txt" | sed 's/^ *//')"
case "$summary" in
    *" 0 unmatched"*) ;;
    *) echo "the summary does not read 0 unmatched: $summary"; exit 1 ;;
esac
echo "ok  $summary"

say "8/8  the strings this commit is supposed to contain"
check() {
    if ! grep -qF -- "$2" "$1"; then
        echo "missing from $1: $2"
        exit 1
    fi
    echo "ok  $1: $2"
}
check docs/paper.md 'A reader brings four things to the repository at the tag'
check docs/paper.md 'reports/reproduction-2026-09-27.md'
check docs/paper.md 'froze at the commit tagged `freeze-2026-10-04`'
check docs/paper.md 'the Zenodo deposit'
check README.md '**What a reader brings, and what a clean checkout reproduces.**'
check README.md '**The deposit.**'
check reports/reproduction-2026-09-27.md 'no figure moved in any of the fourteen regenerated reports'
check src/taxiout/provenance.py '"reports/reproduction-2026-09-27.md"'
check docs/provenance.md 'reports/reproduction-2026-09-27.md'
check Makefile 'reports/reproduction-2026-09-27.md'
check scripts/check_penalty_sensitivity.py 'reports/reproduction-2026-09-27.md'
check docs/modelling_notes.md 'reports/reproduction-2026-09-27.md'
if grep -rqE -- 'Co-authored-b[y]' $PATHS; then
    echo "a committed path carries a co-authorship trailer"
    exit 1
fi
echo "ok  no trailer"

say "staging"
git add -- $PATHS
git -c core.fileMode=false status --short

git commit -q -F - <<MSG
Say what the clean checkout reproduced, name the deposit, and make the one report it cannot regenerate a dated record

On 27 September 2026 a fresh clone at freeze-2026-10-04, with the challenge
data and the two cached tables brought in and the geometry fetched at its
pins, rebuilt the 35 matrix files and the 52 residual files byte for byte,
regenerated fourteen reports in which no figure moved, passed the paper's own
check against them, and built the submission byte for byte. The pinned fetch
was answered by the primary instance for every airport after the client
refused three fallback answers whose databases predated the pin.
reports/reproduction-2026-09-27.md is the dated record of that run.

One report could not be regenerated. reports/determinism.md reads 90 records
of a measurement campaign under the parameters of its day; the records hold
arrays derived from the challenge data and are not redistributed, and no
target reruns the campaign. It joins the dated records under its digest and
leaves the declared-inputs check, which had no way to verify it anywhere but
on the machine it was measured on. The file is unchanged.

The paper's §10 now says what a reader brings and what a clean checkout
reproduces, in the words the run supports: the geometry reports need the
fetched answers or the deposit, the model reports need the METAR and
OurAirports caches as well as the challenge data, and the worked airport
needs nothing. §2.3 and the README name the Zenodo deposit, $doi_readme,
which carries the matrices and the answers under ODbL and the two
public-domain caches beside them. The working-draft preamble and the status
tags are gone; the two [score] marks stay for the upload's commit. The README
is in the house style, and three stale statements about cost and pinning in
the Makefile, the penalty script and the modelling notes now quote the record.

No report is regenerated and no figure moves. check_provenance.py --strict
and check_paper.py --strict both pass.
MSG

say "the commit"
git log -1 --stat --format='%h %s'

say "the commit's own tree, in a fresh worktree"
wt="$work/worktree"
git worktree add -q --detach "$wt" HEAD
(
    cd "$wt"
    "$PY" -m pytest -q -rs > "$work/wt_pytest.txt" 2>&1 || {
        tail -30 "$work/wt_pytest.txt"
        exit 1
    }
    tail -3 "$work/wt_pytest.txt"
    "$PY" scripts/check_provenance.py > "$work/wt_provenance.txt" 2>&1 || {
        cat "$work/wt_provenance.txt"
        exit 1
    }
    grep -E "^provenance" "$work/wt_provenance.txt"
    "$PY" scripts/check_paper.py --strict > "$work/wt_paper.txt" 2>&1 || {
        tail -5 "$work/wt_paper.txt"
        echo "check_paper.py --strict fails in the committed tree"
        exit 1
    }
    echo "ok  check_paper.py --strict passes in the committed tree"
) || {
    git worktree remove --force "$wt"
    echo "the committed tree does not pass its own gates."
    exit 1
}
git worktree remove --force "$wt"

say "done"
echo "Nothing is pushed. 'git show HEAD' to read it, 'git reset --hard HEAD^' to undo."
