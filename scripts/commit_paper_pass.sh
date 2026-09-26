#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit: the paper is brought to the reports it cites, and it gets the
# checker that holds it there.
#
# WHY ONE COMMIT AND NOT THREE. The checker was built first, on 25 September,
# so that the figure pass ran under an instrument rather than by eye, and its
# --report on the paper as it then stood was that pass's work list: 245
# figures, 111 of them unmatched, four headline figures three model changes
# old. The pass then rewrote the paper, and the same day the checker gained the
# mode the pass showed it needed. The three pieces do not stand apart in the
# working tree: the checker as first built reads whole files and certified six
# right figures against the wrong numbers, and the paper as rewritten fails
# under it until the register is filled. So they land together, and this
# header says what each is.
#
# WHAT MOVES, IN ONE PLACE:
#   * docs/paper.md. Every figure is now the reports' as regenerated on 23 and
#     24 September under the parameters that ship: the five-seed model at
#     213.55 s on December, 234.20 on the three airports July 2026 contains and
#     320.90 over all ten, and the ladder's own single seed beside it where the
#     text quotes one. Every section that carries a figure names its source and,
#     for the reports, the section of it. The drafting notes, which said in
#     their own heading that they were not part of the paper, are out of it.
#     Em dashes are gone; the house style is American English without them.
#   * README.md: the Gate 3 date, the two sweeps, the tails generator's two
#     writers and the second sweep's script, brought to the same state.
#   * scripts/check_paper.py, new. Every figure in a section must appear in a
#     file the section's Source: line cites and, where that line names a
#     section of the file in italics, in that section of it, as a number that
#     rounds to it at the figure's own precision. A figure that matches only in
#     a frozen dated record is historical and fails unless listed with a reason.
#     A figure the paper computes from report figures is exempt only on an
#     entry whose arithmetic the check verifies: operands in the cited
#     sections, value rounding to the figure. The report prints every number in
#     scope a figure matches, so a chance match can be told from the row a
#     sentence quotes. Outside --strict it reports and passes; --strict is the
#     paper's freeze condition, and it passes today.
#   * src/taxiout/provenance.py gains the paper's register: the patterns that
#     exempt a section number, a year or a version; the two sections read
#     against every live report; 26 historical entries, 23 of them for the
#     experiments of 20 September; 11 computed entries with their arithmetic;
#     no excluded section. SCOPE says what the check does not reach, which is
#     the correctness of a figure and integers under a thousand, and the paper's
#     pending entry now names what the freeze's regenerations owe it.
#   * scripts/check_provenance.py wires the checker in as check 14, so one gate
#     covers the repository and the paper, and the defined-and-wired guard
#     counts it.
#   * docs/provenance.md is regenerated from the register and moves with it.
#   * tests/test_check_paper.py, new, on constructed papers: the match rule,
#     the confinement, the frozen rule, the verified arithmetic, the signed
#     key, the register never read as a source for its own entries; and one
#     test that --strict passes on the paper as it stands.
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * it regenerates no report and moves no figure. reports/ is HEAD's tree,
#     and the paper quotes it.
#   * it moves neither BOOSTER_PARAMS nor SEED_AVERAGE_ADOPTED. The seed
#     comment in models.py that still says the average is not adopted is the
#     freeze's to move, together with the constant.
#   * it does not touch the README's em dashes, which predate the pass.

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

BASE="92ce7b5"
PATHS="README.md
docs/paper.md
docs/provenance.md
scripts/check_paper.py
scripts/check_provenance.py
scripts/commit_paper_pass.sh
src/taxiout/provenance.py
tests/test_check_paper.py"

# The paper and the README as reviewed, so that what is committed is what was
# read and checked, byte for byte, and not a later edit.
PAPER_SHA="4505485daff5cf6b8afb64dae595c76ffc5b739573359356a899bd98e6d09f15"
README_SHA="d2af5f70f60002e211ffabf5f840b677eae8b957f569d920befdbe6c19894907"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say "this sits on the commit it says it sits on"
history="$(git log --format=%H HEAD)"
if ! grep -qx "$(git rev-parse "$BASE^{commit}")" <<< "$history"; then
    echo "$BASE is not an ancestor of HEAD; this script expects to follow it"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' "$BASE")"

say "the working tree holds these 8 paths and nothing else"
expected="$(printf '%s\n' "$PATHS" | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok  8 paths"

say "1/7  the paper and the README are the reviewed ones, byte for byte"
if [ "$(sha256sum docs/paper.md | cut -d' ' -f1)" != "$PAPER_SHA" ]; then
    echo "docs/paper.md is not the file that was reviewed and checked"
    exit 1
fi
if [ "$(sha256sum README.md | cut -d' ' -f1)" != "$README_SHA" ]; then
    echo "README.md is not the file that was reviewed"
    exit 1
fi
if grep -q -- "$(printf '\xe2\x80\x94')" docs/paper.md; then
    echo "docs/paper.md carries an em dash"
    exit 1
fi
if grep -q 'Drafting notes, not part of the paper' docs/paper.md; then
    echo "the drafting notes are still in docs/paper.md"
    exit 1
fi
echo "ok  paper ${PAPER_SHA:0:16}, README ${README_SHA:0:16}, no em dash, no drafting notes"

say "2/7  the gates, first"
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
echo "ok  14 provenance checks green, the paper's among them"
if "$PY" scripts/check_provenance.py --strict > "$work/strict.txt" 2>&1; then
    echo "--strict passed, which would mean the freeze condition is met; it is not"
    exit 1
fi
echo "ok  --strict still fails, because the pending list is not empty"

say "3/7  the paper's checker passes --strict, which is the point"
"$PY" scripts/check_paper.py --strict > "$work/paper_strict.txt" || {
    cat "$work/paper_strict.txt"
    echo "check_paper.py --strict fails; the paper is not what this commit says"
    exit 1
}
summary="$(grep -m1 -E '^      [0-9]+ figures in the ' "$work/paper_strict.txt" | sed 's/^ *//')"
case "$summary" in
    *" 0 unmatched"*) ;;
    *) echo "the summary does not read 0 unmatched: $summary"; exit 1 ;;
esac
if ! grep -q 'paper clean: every figure is matched, listed or exempt' "$work/paper_strict.txt"; then
    echo "the paper is not clean:"
    cat "$work/paper_strict.txt"
    exit 1
fi
echo "ok  $summary"
"$PY" scripts/check_paper.py --report > "$work/paper_report.md"
for want in "| \*unmatched\* | 0 |" "| excluded by name | 0 |"; do
    if ! grep -q -- "$want" "$work/paper_report.md"; then
        echo "the report lacks: $want"
        exit 1
    fi
done
echo "ok  --report agrees: 0 unmatched, 0 excluded"

say "4/7  reports/ is HEAD's, as a whole tree"
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- reports/)"
if [ -n "$moved" ]; then
    echo "reports/ moved, and this commit regenerates no report:"
    echo "$moved"
    exit 1
fi
echo "ok  $(git rev-parse --short HEAD:reports), unchanged"

say "5/7  BOOSTER_PARAMS and SEED_AVERAGE_ADOPTED are HEAD's"
if [ -n "$(git -c core.fileMode=false status --porcelain -- src/taxiout/models.py)" ]; then
    echo "src/taxiout/models.py moved, and nothing in it is this commit's"
    exit 1
fi
git show "HEAD:src/taxiout/models.py" > "$work/models.head.py"
"$PY" - "$work/models.head.py" <<'PY'
import ast
import sys
from pathlib import Path

NAMES = ("BOOSTER_PARAMS", "SEED_AVERAGE_ADOPTED")


def read(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") in NAMES:
            found[node.target.id] = ast.literal_eval(node.value)
    if set(found) != set(NAMES):
        raise SystemExit(f"{path}: found {sorted(found)}, not {list(NAMES)}")
    return found


was, now = read(sys.argv[1]), read("src/taxiout/models.py")
if was != now:
    raise SystemExit(f"a frozen constant moved:\n  HEAD {was}\n  now  {now}")
if now["SEED_AVERAGE_ADOPTED"] is not False:
    raise SystemExit("SEED_AVERAGE_ADOPTED is not False; that decision is the freeze's")
cell = {k: now["BOOSTER_PARAMS"][k] for k in
        ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")}
print(f"  ok  BOOSTER_PARAMS unchanged, {cell}; SEED_AVERAGE_ADOPTED still False")
PY

say "6/7  docs/ moves in the paper and the generated register, together with it"
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- docs/ \
         | sed 's/^...//' | sort | tr '\n' ' ')"
if [ "$moved" != "docs/paper.md docs/provenance.md " ]; then
    echo "docs/ should move in docs/paper.md and docs/provenance.md; it moves in: ${moved:-nothing}"
    exit 1
fi
if [ -z "$(git -c core.fileMode=false status --porcelain -- src/taxiout/provenance.py)" ]; then
    echo "docs/provenance.md moved and the register it is generated from did not"
    exit 1
fi
"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import check_provenance as cp

if cp.document() != Path("docs/provenance.md").read_text(encoding="utf-8"):
    raise SystemExit("docs/provenance.md is not what the register generates")
print("  ok  docs/provenance.md is what the register generates")
PY

say "7/7  the register's paper lists are what the checker reaches, and no more"
"$PY" - <<'PY'
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import check_paper as chk
from taxiout.provenance import (
    PAPER_EXCLUDED_SECTIONS,
    PAPER_FIGURE_EXEMPTIONS,
    PAPER_HISTORICAL_FIGURES,
)

if PAPER_EXCLUDED_SECTIONS:
    raise SystemExit("PAPER_EXCLUDED_SECTIONS is not empty; the drafting notes are gone")
result = chk.check()
verdicts = chk._verdicts(result)
hist = {(r.section.key, chk.keyed(v.number)) for r, v in verdicts if v.status == chk.HISTORICAL}
listed = {(e.section, e.figure) for e in PAPER_HISTORICAL_FIGURES}
if hist != listed:
    raise SystemExit(f"historical entries and verdicts differ: {hist ^ listed}")
exempt = {(r.section.key, chk.keyed(v.number)) for r, v in verdicts if v.status == chk.EXEMPT}
entries = {(e.section, e.figure) for e in PAPER_FIGURE_EXEMPTIONS}
if exempt != entries:
    raise SystemExit(f"exemption entries and verdicts differ: {exempt ^ entries}")
if any(not e.arithmetic or not e.operands for e in PAPER_FIGURE_EXEMPTIONS):
    raise SystemExit("an exemption carries no arithmetic or no operand")
if any(r.broken for r in result.sections):
    raise SystemExit("a broken exemption entry")
n_hist = sum(1 for _r, v in verdicts if v.status == chk.HISTORICAL)
n_ex = sum(1 for _r, v in verdicts if v.status == chk.EXEMPT)
print(f"  ok  {len(listed)} historical entries reach {n_hist} figures,"
      f" {len(entries)} exemptions reach {n_ex}, every one verified")
PY

say "the strings this commit is supposed to contain"
check() {
    if ! grep -qF -- "$2" "$1"; then
        echo "missing from $1: $2"
        exit 1
    fi
    echo "ok  $1: $2"
}
check docs/paper.md '213.55 s'
check docs/paper.md '234.20'
check docs/paper.md '320.90'
check docs/paper.md 'The paper checks itself'
check docs/paper.md 'with every cell'
check docs/paper.md 'reads +10.00 s, and draws no'
check README.md 'Two parameter sweeps and one extension of the second'
check scripts/check_paper.py 'def own_entries('
check scripts/check_paper.py 'def verify('
check scripts/check_paper.py 'def keyed('
check scripts/check_paper.py 'def spans('
check src/taxiout/provenance.py 'PAPER_EXCLUDED_SECTIONS: tuple[PaperSection, ...] = ()'
check src/taxiout/provenance.py '    arithmetic: str = ""'
check src/taxiout/provenance.py '"0.19 / 2 ** (1 / 2) * 1.998"'
check src/taxiout/provenance.py 'the correctness of a figure in `docs/paper.md`'
check scripts/check_provenance.py 'def check_paper_figures('
check docs/provenance.md '## What the paper quotes'
check tests/test_check_paper.py 'def test_strict_passes_on_the_paper_as_it_stands('
check tests/test_check_paper.py 'def test_the_register_is_never_a_source_for_its_own_entries('
if grep -rqE -- 'Co-authored-b[y]' $PATHS; then
    echo "a committed path carries a co-authorship trailer"
    exit 1
fi
echo "ok  no trailer"

say "staging"
git add -- $PATHS
git -c core.fileMode=false status --short

git commit -q -F - <<'MSG'
Bring the paper to the reports it cites, and give it the checker that holds it there

docs/paper.md had not been touched since 20 September and quoted a submission
rung three model changes old: 217.6 and 325.8 on the two forward splits where
the model that ships reads 213.55 and 320.90. Nothing tied a figure in it to
the file its section named, so a regeneration that moved a figure left the
paper quoting the old one and only a reader could notice.

The checker came first, before a figure moved. scripts/check_paper.py reads
every decimal, percentage and long integer in the paper against the files its
section's Source: line cites, and its --report on the paper as it stood was
the figure pass's work list: 245 figures, 111 unmatched, 7 matching only a
dated record. Then the pass. Every figure is now the reports' as regenerated
on 23 and 24 September under the parameters that ship, every section that
carries a figure names its source and the section of it, the drafting notes
are out of the paper, and the house style holds.

Then the mode the pass showed the checker needed. Read whole, a cited report
certified six right figures against the wrong numbers: a per-airport interval
bound for a dated ceiling, a step in another table for a computed expectation.
So where a source line names a section of a file, the figure must be in that
section, from its heading to the next of the same or a higher level; a name
that is no heading confines to nothing and fails. A figure that matches only
in a frozen dated record is historical and passes only on a listed entry with
a reason: 26 entries, 23 of them for the experiments of 20 September that
§7.1.1 quotes as that day's. A figure the paper computes from report figures
passes only on an entry whose arithmetic the check verifies, operands in the
cited sections and value rounding to the figure: 11 entries, the ladder steps
stated as differences and the winner's-curse expectation. The key carries the
sign, because the same digits are a computed magnitude in one sentence of §6.3
and a quoted interval bound in the next. The register's own lists are never
read as a source. The report prints every number in scope a figure matches,
so a chance match can be told from the row the sentence quotes; the check
itself cannot tell them apart, and SCOPE says so.

check_paper.py --strict passes: 398 figures, 358 matched in their cited
sections, 29 historical and listed, 11 computed and verified, 0 unmatched. It
is wired into check_provenance.py as check 14, and the ordinary gate is
clean. The paper's pending entry now says what the freeze owes it: --strict
again after every report is regenerated, and the paper moved with any figure
that moves.

No report is regenerated and no figure moves; reports/ is the parent's tree.
BOOSTER_PARAMS and SEED_AVERAGE_ADOPTED are untouched.
MSG

say "the commit"
git log -1 --stat --format='%h %s'

say "reports/ is the parent's tree"
if [ "$(git rev-parse HEAD:reports)" != "$(git rev-parse HEAD^:reports)" ]; then
    echo "reports/ moved in this commit"
    exit 1
fi
echo "ok  $(git rev-parse --short HEAD:reports)"

say "the commit's own tree, in a worktree with no data/"
# The condition the checker is validated in: a tree that contains the new
# files as tracked files, so the scanner reads them, and no challenge data.
wt="$work/worktree"
git worktree add -q --detach "$wt" HEAD
(
    cd "$wt"
    rm -rf data
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
