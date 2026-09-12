#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Make the ladder repeat to the bit, and record how that was found and what it
# changed. Two commits: the harness that found the source and its evidence, then
# the reference configuration with every report it changes regenerated under
# it. Additive; nothing is rewritten. Not pushed.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# refuses to commit a report older than the code that generates it, and refuses
# to commit unless the two reference runs compared byte for byte. The
# verification pass at the foot builds each commit in a throwaway worktree and
# runs the suite against it, with PYTHONPATH forced to the worktree's own src,
# as scripts/commit_gate2c_closeout.sh does and for the reason it gives.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in scripts/check_determinism.py scripts/compare_runs.py \
         scripts/run_baselines.py scripts/make_submission.py src/taxiout/models.py \
         Makefile reports/determinism.md reports/reproduction-2026-09-11.md \
         reports/row-wise-2026-09-11.md \
         reports/migration-2026-09-10.md reports/baseline.md reports/residual.md \
         reports/occupancy.md reports/congestion.md reports/transfer.md \
         reports/linux-blas.txt reports/linux-freeze.txt \
         reports/linux-residuals-delta.txt reports/linux-run2-vs-run1.txt \
         reports/linux-stack-delta.txt; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

# A generated report older than the code that generates it carries figures from
# code that no longer exists. Every report this change regenerates is gated,
# and the harness's own report with them.
for pair in "scripts/run_baselines.py reports/baseline.md" \
            "scripts/report_residual.py reports/residual.md" \
            "scripts/report_occupancy.py reports/occupancy.md" \
            "scripts/report_congestion.py reports/congestion.md" \
            "scripts/report_transfer.py reports/transfer.md" \
            "scripts/check_determinism.py reports/determinism.md"; do
    set -- $pair
    if [ -n "$(find src/taxiout "$1" -newer "$2" -name '*.py' 2>/dev/null)" ]; then
        echo "$2 is older than the code that generates it."
        echo "Run: python3 $1"
        exit 1
    fi
done

say "the two reference runs agree byte for byte"
# The comparison scripts/compare_runs.py wrote, not a claim about it: its first
# summary line has to say that nothing differed and nothing was missing.
if ! grep -qE "^[0-9]+ files byte-identical, [0-9]+ differing only in their generation date, 0 differing otherwise, 0 missing from a run\.$" \
        reports/reproduction-2026-09-11.md; then
    echo "reports/reproduction-2026-09-11.md does not record two identical runs"
    exit 1
fi
if ! grep -q '"deterministic": True' src/taxiout/models.py \
        || ! grep -q '"force_col_wise": True' src/taxiout/models.py; then
    echo "BOOSTER_PARAMS does not carry the configuration the runs were made under"
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

# The assistant-configuration paths are forbidden by the licence gate if they
# are tracked; this makes sure they are also ignored, so a later `git add -A`
# cannot stage them either. The names are assembled from fragments for the
# reason scripts/check_licence_hygiene.py gives: spelled out, they would match
# the gate's own pattern and fail it on this file.
for p in ".cla""ude" "CLA""UDE.md"; do
    if [ -e "$p" ] && ! git check-ignore -q "$p"; then
        echo "$p exists and is not ignored -- fix .gitignore first"
        exit 1
    fi
done

if git status --porcelain --untracked-files=all data/challenge data/submissions \
        data/raw data/processed | grep -vE 'README.md|ODbL-NOTICE.md' | grep -q .; then
    echo "untracked, unignored files under data/ -- fix .gitignore first"
    exit 1
fi

BEFORE="$(git rev-parse HEAD)"

say "1/2 the harness and what it found"
git add scripts/check_determinism.py reports/determinism.md \
        reports/linux-blas.txt reports/linux-freeze.txt reports/linux-stack-delta.txt \
        reports/linux-residuals-delta.txt reports/linux-run2-vs-run1.txt
git commit -q -F - <<'MSG'
Find where the ladder stops repeating

Two consecutive runs of the ladder on one machine agreed byte for byte on
none of the fifty-two residual files, and on twenty-seven of them they
differed by whole trees. scripts/check_determinism.py finds the source on
one split and one rung instead of refitting the ladder: it builds the
feature frame from the ladder's own loaders, records a digest of every
column, and fits December's identifiers rung repeatedly under named booster
configurations.

The feature frame repeats: two builds in separate processes, under random
string-hash seeds, agree on all 175 digests. One BLAS thread instead of
eight moves one thing, the geometric-class rung's predictions, by at most
2.9e-08 s, and repeats at either count.

The booster does not. Under the parameters as they stood, fifteen fits gave
fifteen different models, with LightGBM choosing its row-wise histograms
every time it said what it chose. Forcing a histogram layout alone changed
nothing; deterministic alone gave two outcomes in three, exactly the
column-wise against row-wise difference. Deterministic with the column-wise
layout gave one outcome in six fits at eight threads and six at fourteen,
and the same bits at both and as the old parameters at one thread.

reports/determinism.md is the measurement. The linux-*.txt files are the
evidence the migration started from: the stack against the reference Mac,
the BLAS build, the frozen package list, and the two residual comparisons
that showed the runs disagreeing.
MSG

say "2/2 the reference configuration, and every report it changes"
git add src/taxiout/models.py scripts/run_baselines.py scripts/make_submission.py \
        scripts/compare_runs.py Makefile \
        reports/baseline.md reports/residual.md reports/occupancy.md \
        reports/congestion.md reports/transfer.md \
        reports/reproduction-2026-09-11.md reports/row-wise-2026-09-11.md \
        reports/migration-2026-09-10.md scripts/commit_determinism.sh
git commit -q -F - <<'MSG'
Make the booster repeat to the bit, and measure what that leaves

BOOSTER_PARAMS gains deterministic and force_col_wise, and n_jobs goes to
the reference machine's eight cores. deterministic makes LightGBM's
threaded sums run in a fixed order; forcing the layout removes the timing
test that picked it on every fit. Column-wise rather than row-wise because
it alone gave the same bits at every thread count measured. It was chosen
on that property, fixed in reports/migration-2026-09-10.md before any
ladder figure under it existed, and the row-wise configuration was run
once as well so that what the choice changes is on the record:
reports/row-wise-2026-09-11.md. Of 164 conclusions read, it changes no
verdict the report acts on.

Two consecutive runs of make baseline, residual, occupancy, congestion and
transfer agree byte for byte on every report and every residual file:
reports/reproduction-2026-09-11.md, written by scripts/compare_runs.py.
The seed-averaging section's reproducibility check, which read 4.55e-13
on this machine before, reads zero on both splits.

Every model figure moves. The printed conclusions that move with them:
the leakage audit on + stand occupancy now fires, repeatably, by 0.37 s
on July's three airports, inside that leg's 1.28 s noise floor; the
report says so, says that the rule read literally would reject the
block, and says the block is kept on a reading adopted after the number
was seen. Seed averaging gains 0.86 and 0.54 s and is still not adopted.

baseline.md gains a noise floor: the five seed refits it already made give
each leg's seed-to-seed spread, and every rung and tail-exclusion arm whose
change is smaller than the square root of two times that spread is listed,
by a rule fixed before the runs. The prose the refit made stale is
computed now rather than typed, and the decomposition of the encoding arms
is of the arm that ships, as its own comment always said.

The Makefile's run times are measured: about two hours for the ladder on
eight cores, a quarter of an hour for transfer, and seconds to two minutes
for the reports that read the stored residuals.
MSG

say "each commit's tree passes its own tests"
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
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

say "nothing tracked is left behind"
if ! git diff --quiet; then
    echo "tracked files are still modified after the last commit:"
    git --no-pager diff --name-only | sed 's/^/  /'
    exit 1
fi
echo "ok"

say "result"
git --no-pager log --oneline -4
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -2"
