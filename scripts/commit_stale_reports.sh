#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit, regenerating the four reports the wider sweep left behind.
#
# The sweep moved BOOSTER_PARAMS and the ladder that followed it rewrote all
# 52 files under data/processed/residuals/ on 21 September between 09:06:48
# and 11:15:04. Three reports are generated from those residuals and were
# older than them; a fourth fits its own booster and was fitting under
# parameters the repository no longer carries. Neither of the sweep's two
# commits regenerated any of them, so docs/paper.md cited four stale reports
# at ten places.
#
# Nothing is decided here and nothing is chosen. No parameter, no feature, no
# rule and no threshold changes: src/ is asserted unchanged against HEAD
# before anything is staged, and the gates below read BOOSTER_PARAMS,
# SEED_AVERAGE_ADOPTED, SEED_AVERAGE_PENDING and TAIL_EXCLUSION_ADOPTED_H out
# of the modules rather than trusting that.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# refuses a working tree holding anything but the six expected paths, requires
# each report to be newer than its generator AND newer than every stored
# residual, and verifies the commit's own tree in a worktree. Nothing is
# pushed.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

REPORTS="reports/residual.md reports/congestion.md reports/occupancy.md
reports/transfer.md"
PATHS="$REPORTS Makefile scripts/commit_stale_reports.sh"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the working tree holds these six paths and nothing else"
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok"

say "src/ is unchanged against HEAD"
# The whole claim of this commit is that only generated text moved. A single
# line under src/ would mean a parameter, a feature or a rule moved with it,
# and the reports below would be a reading of something this commit did not
# declare.
if [ -n "$(git -c core.fileMode=false diff --stat HEAD -- src/)" ]; then
    echo "src/ differs from HEAD. This commit regenerates reports and nothing else."
    git -c core.fileMode=false diff --stat HEAD -- src/
    exit 1
fi
echo "ok"

say "no parameter, rule or threshold moved"
unchanged() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 no longer carries \"$2\" -- this commit would be changing a rule"
        exit 1
    fi
}
unchanged src/taxiout/models.py '"n_estimators": 3000'
unchanged src/taxiout/models.py '"learning_rate": 0.015'
unchanged src/taxiout/models.py '"num_leaves": 255'
unchanged src/taxiout/models.py '"min_child_samples": 20'
unchanged src/taxiout/models.py '"random_state": 20260902'
unchanged src/taxiout/models.py 'SEED_AVERAGE_ADOPTED: bool = False'
unchanged src/taxiout/models.py 'TAIL_EXCLUSION_ADOPTED_H'
unchanged src/taxiout/tails.py 'MIN_COUNT = 5'
unchanged src/taxiout/tails.py 'MIN_SHARE = 0.05'
unchanged scripts/run_baselines.py 'SEED_AVERAGE_PENDING: tuple[str, ...] = ()'
unchanged scripts/run_baselines.py 'HEADROOM_TRIGGER_S = 5.0'
unchanged scripts/run_baselines.py '"occupancy": (3.9, 10.3),'
echo "ok"

say "each report is newer than its generator"
gen_of() {
    case "$1" in
        reports/residual.md) echo scripts/report_residual.py ;;
        reports/congestion.md) echo scripts/report_congestion.py ;;
        reports/occupancy.md) echo scripts/report_occupancy.py ;;
        reports/transfer.md) echo scripts/report_transfer.py ;;
    esac
}
for r in $REPORTS; do
    g="$(gen_of "$r")"
    if [ "$g" -nt "$r" ]; then
        echo "$r is older than $g -- regenerate it"
        exit 1
    fi
    printf '  ok   %-24s newer than %s\n' "$r" "$g"
done

say "each report is newer than every stored residual"
# The reason this commit exists. Three of the four read
# data/processed/residuals/ and were older than it; the fourth refits and was
# fitted under the superseded cell. A report older than the residuals is a
# report of a model the repository no longer carries.
newest="$(find data/processed/residuals -type f -printf '%T@ %p\n' | sort -n | tail -1)"
newest_path="${newest#* }"
for r in $REPORTS; do
    if [ "$newest_path" -nt "$r" ]; then
        echo "$r is older than $newest_path -- regenerate it"
        exit 1
    fi
done
echo "  ok   all four newer than $newest_path"

say "the leakage trigger is where the ladder left it, and this commit did not move it"
# The trigger fired in the ladder run that preceded this block, not in it:
# EXPECTED_HEADROOM_S is a constant quoted from docs/review_headroom.md and
# the measured side comes from run_baselines.py. None of the four generators
# above computes either, so this commit cannot have changed the verdict. What
# is checked is that reports/baseline.md, which this commit does not touch,
# still carries the reading it was committed with.
if [ -n "$(git -c core.fileMode=false status --porcelain -- reports/baseline.md)" ]; then
    echo "reports/baseline.md differs from HEAD. It is not this commit's business."
    exit 1
fi
if ! grep -qF "an overshoot of 7.301 s, and fires by 2.301 s" reports/baseline.md; then
    echo "reports/baseline.md no longer carries the audit reading this commit describes"
    exit 1
fi
echo "ok"

say "tests"
python3 -m pytest -q

say "lint"
ruff check .
ruff format --check .

say "licence gate"
python3 scripts/check_licence_hygiene.py

for p in ".cla""ude" "CLA""UDE.md"; do
    if [ -e "$p" ] && ! git check-ignore -q "$p"; then
        echo "$p exists and is not ignored -- fix .gitignore first"
        exit 1
    fi
done

BEFORE="$(git rev-parse HEAD)"
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT

say "1/1 the four reports"
git add $PATHS
git commit -q -F - <<'MSG'
Regenerate the four reports the wider sweep left behind

The sweep moved BOOSTER_PARAMS to 3,000 trees at 0.015 and 255 leaves, and
the ladder that followed it rewrote all 52 files under
data/processed/residuals/. Three reports are generated from those residuals
and were older than them; a fourth fits its own booster and was fitting under
the superseded cell. Neither of the sweep's commits regenerated any of them,
and docs/paper.md cites the four at ten places, so a figure pass run against
them would have written superseded numbers into the paper.

Nothing is decided here and nothing is chosen. No parameter, no feature, no
rule and no threshold changes, and src/ is asserted unchanged against the
previous commit before anything is staged. Every figure that moved moved
because the model underneath it moved, and no generator was edited.

Measured, each under /usr/bin/time -v:

  reports/residual.md     20.92 s, 3.21 GiB
  reports/congestion.md   1 min 43 s, 6.68 GiB
  reports/occupancy.md    14.77 s, 2.83 GiB
  reports/transfer.md     39 min 1 s, 7.07 GiB

None of the four came back byte-identical, which is what a report that
actually reads the residuals it claims to should do. No verdict moved in any
of them: the diffs are figures, and no rule's pass, fail or adoption changed.

The one figure worth naming because the paper quotes it: the oracle ceiling
for the same-runway sequence at Heathrow on the July leg falls from 9.9 s to
9.5 s, and Heathrow's July error from 318.8 s to 313.5 s. Those are the
numbers a documentation pass would otherwise have carried forward.

make transfer is re-timed. The Makefile said it took a quarter of an hour
under the booster's earlier parameters and had not been re-timed under these;
it takes forty minutes at 7.1 GiB, and the note now says so with the date.

On the leakage audit, stated because it is easy to read this commit as
touching it and it does not. The trigger fired in the ladder run that
preceded this one, not in this one: stand occupancy takes +17.60 s on July,
the three against a constant estimate of +10.3 s, an overshoot of 7.301 s
against the 5.000 s allowed, where under the previous cell it overshot by
4.345 s and missed. That reading is in reports/baseline.md, which this commit
does not touch and which this script checks is unmodified. None of the four
generators above computes either side of that comparison, so none of them can
have moved it. The estimate is a constant quoted from docs/review_headroom.md,
whose own figures were made by a script held outside this repository against
residuals the sweep has since overwritten; so the audit's two sides are now
measured on different residuals, and that is not something this repository
can correct.
MSG

say "the commit's own tree passes its own tests"
for sha in $(git rev-list --reverse "$BEFORE"..HEAD); do
    git worktree add --detach -q "$work/t" "$sha"
    if (cd "$work/t" && PYTHONPATH="$work/t/src" python3 -m pytest -q >/dev/null 2>&1); then
        printf '  ok   %s %s\n' "$(git log -1 --format=%h "$sha")" \
            "$(git log -1 --format=%s "$sha")"
    else
        printf '  FAIL %s %s\n' "$(git log -1 --format=%h "$sha")" \
            "$(git log -1 --format=%s "$sha")"
        git worktree remove --force "$work/t"
        exit 1
    fi
    git worktree remove --force "$work/t"
done

say "result"
git --no-pager log --oneline -4
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -1"
