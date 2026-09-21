#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Two commits, for the one wider parameter sweep and the ladder run that
# follows it.
#
# The sweep was permitted once on 19 September and pre-registered on 20
# September with its grid, its fit rules, its floor and its reading rule fixed
# in writing before any of its figures existed. Twenty-seven cells, one
# 3,000-tree fit each, read at six tree counts, on the submission rung under
# the fit rules that stand, training January to September 2025 and scoring
# October 2025 over the rows the record-quality term leaves to the physical
# model. October is a month the ladder never scores, so a number read from it
# chooses parameters without touching what is reported. Nothing was chosen on
# December, July or the held-out pairs.
#
# The first commit is the instruments and changes no figure in any report. The
# second is the result: the chosen cell in BOOSTER_PARAMS and the ladder
# re-read under it. They are in that order because an instrument that is
# committed after the number it produced cannot be checked against it.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# refuses a working tree holding anything but the expected paths, requires each
# regenerated report to be newer than its generator, and verifies each commit's
# own tree in a worktree before finishing. Nothing is pushed.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

INSTRUMENTS="scripts/sweep_params.py tests/test_sweep_params.py
scripts/run_baselines.py tests/test_seed_average_guard.py
scripts/commit_sweep.sh"
# The Makefile is here and not above because both of its changes are about
# the result: its note on what 'make baseline' costs is a measurement of the
# run this commit carries, and its description of the sweep target describes
# a sweep that had not yet run at the commit before.
RESULT="Makefile src/taxiout/models.py reports/baseline.md"
PATHS="$INSTRUMENTS $RESULT"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the working tree holds these paths and nothing else"
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok"

say "docs/plan.md and the pre-registered report are untouched"
# Neither is this block's business. The plan entry is the record the sweep's
# criteria were fixed in and is committed as f7cb53c; the pre-registered
# report is the run the gate below reproduced, and a sweep that edited the
# figures it was checked against would have checked nothing.
for p in docs/plan.md reports/preregistered-2026-09-20.md; do
    if [ -n "$(git -c core.fileMode=false status --porcelain -- "$p")" ]; then
        echo "$p differs from HEAD. It is not this commit's business."
        exit 1
    fi
done
echo "ok"

say "the chosen cell is the one the sweep's own record chose"
# Read out of the sweep's verdict record rather than written down here, so
# that the constant and the measurement that chose it cannot drift apart. The
# record is gitignored, as the grid's numbers are; this check is what carries
# it into the history.
VERDICT="data/processed/sweep/verdict.json"
if [ ! -f "$VERDICT" ]; then
    echo "$VERDICT is not there -- run: PYTHON=.venv/bin/python make sweep"
    exit 1
fi
python3 - "$VERDICT" <<'PY'
import json
import re
import sys
from pathlib import Path

verdict = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = verdict["chosen"]
source = Path("src/taxiout/models.py").read_text(encoding="utf-8")


def constant(name):
    found = re.search(rf'"{name}": ([0-9.]+),', source)
    if not found:
        raise SystemExit(f"BOOSTER_PARAMS carries no {name}")
    return float(found.group(1))


wanted = {
    "n_estimators": float(chosen["n_estimators"]),
    "learning_rate": float(chosen["learning_rate"]),
    "num_leaves": float(chosen["num_leaves"]),
    "min_child_samples": float(chosen["min_child_samples"]),
}
for name, value in wanted.items():
    got = constant(name)
    if got != value:
        raise SystemExit(
            f"BOOSTER_PARAMS has {name} = {got:g} where the sweep chose {value:g}"
        )
if verdict["rows"] != 162:
    raise SystemExit(f"the grid has {verdict['rows']} rows, not 162")
if not verdict["booster_params_moves"]:
    raise SystemExit(
        "the sweep kept the current cell, and this script commits a move; "
        "the message below would be wrong"
    )
difference = verdict["difference_s"]
floor = verdict["floor_s"]
if not (-difference > floor):
    raise SystemExit(
        f"the difference {difference:+.4f} s does not clear the floor {floor:.4f} s"
    )
print(
    f"  chosen: {verdict['chosen_label']} at {chosen['rmse_oct_s']:.4f} s,"
    f" {difference:+.4f} s against a floor of {floor:.4f} s"
)
print(f"  on an edge of: {', '.join(verdict['best_on_edges']) or 'no axis'}")
PY
echo "ok"

say "the gate reproduced the pre-registered run to the bit"
# The sweep's five single-seed fits of the cell that stood had to reproduce
# the October singles of the run recorded under data/processed/preregistered/,
# on the stored doubles and on the stored prediction digests, before any cell
# was read. A sweep whose gate did not hold is a sweep of a harness that fits
# something else.
python3 - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("data/processed/sweep/gate.json").read_text(encoding="utf-8"))
if gate["disagreed"]:
    raise SystemExit("the gate disagreed: " + "; ".join(gate["disagreed"]))
for row in gate["singles"]:
    if not (row["equal_double"] and row["equal_digest"] and row["same_row_set"]):
        raise SystemExit(f"seed +{row['seed']} did not reproduce")
tree = gate["tree_reading"]
if not (tree["equal_double"] and tree["equal_digest"]):
    raise SystemExit("the tree-reading check did not hold")
print(f"  5 singles reproduced to the bit; floor {gate['floor_s']:.6f} s")
print("  reading 1,000 trees of a 3,000-tree fit gives the 1,000-tree fit")
PY
echo "ok"

say "nothing else was adopted by this block"
unchanged() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 no longer carries \"$2\" -- this commit would be adopting something"
        exit 1
    fi
}
# The season columns stay in the frame: Experiment 3 adopted nothing in either
# direction. No recency schedule ships: Experiment 2 adopted neither. And the
# submission is still fitted from a single seed until the constant is moved,
# which is a decision this block does not take.
for c in tko_month tko_doy_sin tko_doy_cos; do
    unchanged src/taxiout/features.py "\"$c\","
done
unchanged src/taxiout/models.py "+ TIME_FEATURES"
unchanged src/taxiout/models.py 'SEED_AVERAGE_ADOPTED: bool = False'
unchanged src/taxiout/models.py '"random_state": 20260902'
unchanged src/taxiout/tails.py "MIN_COUNT = 5"
unchanged src/taxiout/tails.py "MIN_SHARE = 0.05"
if grep -n "sample_weight=" scripts/sweep_params.py | grep -qv "sample_weight=None"; then
    echo "the sweep passes a weight somewhere; no recency schedule was adopted"
    exit 1
fi
echo "ok"

say "the first sweep's record survives"
# The numbers in the comment beside BOOSTER_PARAMS are how the parameters were
# chosen the first time, and the second sweep does not get to delete them. The
# sentence that says the two are not comparable is checked by name, because it
# is the one a reader needs and the easiest to lose in an edit.
check() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 does not carry \"$2\""
        exit 1
    fi
}
check src/taxiout/models.py "235.0 s on October"
check src/taxiout/models.py "235.0 to 244.8 s"
check src/taxiout/models.py "top eight cells lie within 0.8 s"
check src/taxiout/models.py "THE TWO SWEEPS' OCTOBER FIGURES ARE NOT COMPARABLE"
check src/taxiout/models.py "WHY 235.0 AND 214.9 CANNOT BE SUBTRACTED"
check scripts/sweep_params.py "FIRST_SWEEP"
check scripts/sweep_params.py '"comparable_with_this_sweep": False'
echo "ok"

say "the seed section makes the two measurements the rule asks for"
check scripts/run_baselines.py "SEED_AVERAGE_PENDING: tuple[str, ...] = ()"
check scripts/run_baselines.py "def seed_average_guard"
check scripts/run_baselines.py "def _october_seed_average"
echo "ok"

say "the seed section printed both columns and the ladder read them"
# The rule of 19 September asks for the three and for October before the
# average may ship, and SEED_AVERAGE_PENDING is emptied on the strength of
# this run having made them. If the report does not carry both columns, the
# tuple was emptied against a run that did not measure what it named.
check reports/baseline.md "| the three |"
check reports/baseline.md "| October, all ten |"
check reports/baseline.md "the guard of 20 September"
echo "ok"

say "baseline.md is newer than its generator and than the parameters it was fitted under"
for generator in scripts/run_baselines.py src/taxiout/models.py; do
    if [ "$generator" -nt reports/baseline.md ]; then
        echo "reports/baseline.md is older than $generator -- regenerate it"
        exit 1
    fi
done
echo "ok"

say "no cell label in the tracked tree can collide with a stand identifier"
# The sweep prints twenty-seven cell labels. Any token of letters and digits
# three characters or longer can equal a stand identifier at one of the ten
# airports, and the report guards refuse a whole report over one, through a
# code span as readily as outside one. The labels are written out in full so
# that no such token exists; this is what says so rather than hoping.
python3 - <<'PY'
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import sweep_params as sw  # noqa: E402

RUN = re.compile(r"\w+")
bad = []
for cell in sw.CELLS:
    for trees in (None, *sw.TREES):
        label = sw.cell_label(sw.cell_values(cell), trees)
        for token in RUN.findall(label):
            if (
                len(token) >= 3
                and re.search(r"[A-Za-z]", token)
                and re.search(r"[0-9]", token)
            ):
                bad.append((label, token))
if bad:
    for label, token in bad:
        print(f"  {label} carries {token}", file=sys.stderr)
    raise SystemExit("a cell label could collide with a stand identifier")
print(f"  {len(sw.CELLS) * (len(sw.TREES) + 1)} labels, no colliding token")
PY
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

say "the grid's own numbers are not tracked"
for p in data/processed/sweep_october.csv data/processed/sweep; do
    if [ -e "$p" ] && ! git check-ignore -q "$p"; then
        echo "$p exists and is not ignored -- fix .gitignore first"
        exit 1
    fi
done
echo "ok"

BEFORE="$(git rev-parse HEAD)"
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT

say "1/2 the instruments"
git add $INSTRUMENTS
git commit -q -F - <<'MSG'
Bring the sweep to the current fit rules, and make the ladder read the seed rule's two columns

Two instruments, committed before the numbers they produce. Neither changes a
figure in any report; the sweep's result and the ladder re-read under it are
the commit after this one.

The sweep script ran the first sweep and could not run the second. Three
things in it were wrong for a sweep under today's rules and each was right in
its own context, so each is changed with the reason recorded rather than
quietly corrected. It pinned the saturation rung by name, which kept the first
sweep's October numbers reproducible once TOP_RUNG had moved past it; it now
follows SUBMISSION_KEY, and the first sweep's terms are kept in the script
under FIRST_SWEEP and in the comment beside BOOSTER_PARAMS. It fitted on
train & ~flagged with no four-hour exclusion and no shrunk encoding, which is
how the ladder fitted when it ran; it now imports scripts/preregistered.py and
uses that script's own Frame.prepare and the same construction its fit_once
uses, so the fit rules are the ones that stand and are not restated anywhere
they could drift. And it carried the first grid, which the pre-registration of
20 September replaced.

One consequence of the rung change that would not have announced itself: the
old reader recoded the categoricals by hand and skipped the rare-level pooling
Booster.predict applies. The saturation rung carries no pooled categorical, so
nothing came of it; the submission rung carries cs_prefix, so it would have.
The new reader mirrors Booster.predict step for step and adds num_iteration,
which is the only difference between reading a prefix of a fit and reading the
whole of it.

Cell labels are written out in full, learning_rate=0.015, num_leaves=255 and
so on, and never abbreviated. A short label of letters and digits can equal a
stand identifier at one of the ten airports, and the report guards refuse a
whole report over one such token, through a code span as readily as outside
one, as H12 already showed. Written out in full a label carries no token of
that shape at all, and a test and a gate in this script both say so over every
label the sweep can print.

The ladder's seed section now makes the two measurements the rule of 19
September asks for before the five-seed average may ship. It asked for the
three airports July 2026 contains and for October, and the generator carried
them in SEED_AVERAGE_PENDING as measurements nothing had made. The three came
out of the July leg's own five fits and were not printed; October is a leg no
rung of this ladder scores, so the section now fits the submission rung on it
five times, under the same rules as every other leg, for this table and for
nothing else. October is deliberately not added to the ladder's splits: that
would put an eleventh column in every table in the report and move what the
second forward leg means, which is July in every section that reads it. Its
noise floor is measured from its own five singles by the rule the report
already states, and it is printed in the noise floor table with a sentence
saying why it carries no comparison column.

The guard of 20 September is applied there too, and not only the rule. It was
fixed before either measurement existed: the average is not adopted if, on the
three or on October, it is worse than the ladder's seed by more than that
leg's floor. It sizes a regression and asks nothing of a gain, so a column
better by less than its floor is not a gain that failed a test. SEED_AVERAGE_-
PENDING is emptied because the section now makes the measurements itself, and
what stands in its place is computed from the figures rather than asserted: a
column this run did not make leaves the guard unset and blocks adoption
exactly as the unmade measurement did.

Fifteen tests on the guard and its boundary, and twenty-two on the sweep: the
grid is twenty-seven cells and six tree counts out of one fit, the reading
rule keeps the current cell on a difference equal to the floor and replaces it
on a larger one, an edge cell is detected on each of the four axes, the floor
is root two times the sample standard deviation at ddof=1, and a label that
would collide with a stand identifier is refused rather than written. None of
them assumes which cell BOOSTER_PARAMS holds, because the next commit moves it.

The Makefile's two comments about the sweep are in the next commit and not
this one. One of them is a measurement of the ladder run that commit carries,
and the other describes a sweep that has not run at this commit.
MSG

say "2/2 the result"
git add $RESULT
git commit -q -F - <<'MSG'
Take the wider sweep's cell, and re-read the ladder under it

The one wider parameter sweep, permitted once on 19 September and
pre-registered on 20 September with its grid, its fit rules, its floor and its
reading rule fixed in writing before any of its figures existed.

The grid, as fixed: learning_rate (0.015, 0.03, 0.06), num_leaves
(63, 127, 255), min_child_samples (20, 50, 200), twenty-seven cells, each
fitted once at 3,000 trees and read at 300, 600, 1,000, 1,500, 2,000 and
3,000 trees, which is 162 readings. Train January to September 2025, score
October 2025 over the rows the record-quality term leaves to the physical
model, on the submission rung with the season columns in, the four-hour
exclusion, the shrunk encoding on the fitting rows, the record-quality model
on the training months, and no recency schedule. Nothing was chosen on
December, July or the held-out pairs, and October is a month the ladder never
scores.

The gate ran before any cell was read, and it held. The parameters and the fit
rules had not moved since the pre-registered run, so that run's five October
single-seed fits had to come back identical. All five did, on the stored
double and on the sha256 of the prediction vector, over the same 185,674 rows
with the same 115 flagged and the same 1,570,127 fitting rows. The floor
measured from them is 0.189955 s, root two times their sample standard
deviation, which is the 0.19 s the pre-registered report prints. The
tree-reading shortcut the whole grid rests on was checked rather than assumed:
the current cell fitted once at 3,000 trees and read at 1,000 gives the
1,000-tree fit of the same cell and seed, to the bit. A disagreement in any of
the three would have stopped the block before a cell was read.

The reading rule, as fixed: the best cell replaces BOOSTER_PARAMS only if its
October figure is below the current cell's figure from the same run by more
than the floor. The current cell, learning_rate=0.03, num_leaves=127,
min_child_samples=50, n_estimators=1000, reads 217.2826 s in this run. The
best of the 162, learning_rate=0.015, num_leaves=255, min_child_samples=20,
n_estimators=3000, reads 214.9045 s. The difference is -2.3781 s against a
floor of 0.1900 s, so it clears and the cell moves. The grid ran from 214.9045
to 235.4421 s.

The chosen cell sits on an edge of this grid on all four axes: the lowest
learning rate, the most leaves, the fewest minimum child samples and the most
trees. That is recorded and the grid is not extended again, which the
pre-registration fixes whatever the grid shows. What it means is that the
sweep found a direction and not a floor, and the honest statement is that the
optimum is outside the grid and was not searched for.

The day-clustered interval on the chosen cell against the one it replaces, days
as the clusters and the same draw for both arms, is -2.38 s [-2.89, -1.89]. It
decides nothing: the reading rule is floor-only and stays floor-only. It is
printed because the sweep could have chosen a cell whose interval spanned zero
and that should not be met for the first time in the documentation pass.

The comment beside BOOSTER_PARAMS now carries both sweeps and says in terms
that their October figures cannot be subtracted. The first sweep scored the
saturation rung with no shrunk encoding and no four-hour exclusion and its
chosen cell read 235.0 s; the same parameter cell under today's rules reads
217.28 s. Between 235.0 and 217.3 lie the feature block and the fit rules. What
this sweep bought is 217.28 to 214.90 and nothing else here is a difference
between two comparable numbers.

reports/baseline.md is regenerated under the parameters that stand, in one
run of 7 h 11 min 17 s at a peak of 8.15 GiB. Every figure in it moves,
because every fit in it moves, and the rung chain, the acceptance verdicts,
the noise floors and the seed section are all re-read rather than edited.
Nothing was chosen on any of them. The submission rung comes back at 215.3 s
on December all ten against 217.6, 323.2 on July all ten against 325.8, 234.6
on July the three against 237.8, and 216.4 at the ranking mix against 218.6.
The four acceptance verdicts hold as they did before: stand occupancy,
callsign prefix, destination and holidays and plan differences all pass. The
recorded-taxi tail is still adopted at four hours and still refused at two.
The floors move with the fits: 0.26 on December, 0.61 on July all ten, 0.63 on
July the three and 0.18 on October.

One check in that run is worth naming, because it closes the loop on the
sweep by a route the sweep did not take. The ladder's own October seed 0
reads 214.90 s, and the cell the sweep chose read 214.9045 s. Those are the
same fit reached by two independent paths: the sweep fitting a grid cell at
3,000 trees and reading it through its own prefix reader, and the ladder
fitting BOOSTER_PARAMS and reading it through Booster.predict. The cell that
was chosen is demonstrably the cell that went into the module.

The seed section now prints the three July airports and October, which is
what the rule of 19 September asks for before the five-seed average ships,
and applies the guard of 20 September to both. The rule holds on both forward
legs: December gains 0.72 s against a floor of 0.26 and beats all five
singles, July gains 0.63 s against a floor of 0.61 and beats all five, which
on July is a margin of two hundredths of a second over the floor and is
stated rather than rounded away. The guard holds on both columns: the three
at -0.67 s against a floor of 0.63 and October at -0.25 s against 0.18, both
better rather than worse, so neither carries a regression of any size.
SEED_AVERAGE_PENDING is emptied by this run, which is where the
pre-registration puts it and not before.

taxiout.models.SEED_AVERAGE_ADOPTED is left at False, so the report also
prints its own standing notice that the constant disagrees with the rule.
That notice is the generator working as built. Moving the constant changes
what scripts/make_submission.py ships, which is a decision about the
submission and not about this sweep, and it is not taken here.

The Makefile's measured note said 'make baseline' takes about two hours. That
was true under the booster's earlier parameters and the sweep made it false,
so it is corrected with this run's measurement and says what changed and when.
The same 3.2 times the work per fit applies to every target that fits the
submission rung; make transfer is not re-timed here and the note says so
rather than scaling its figure.

The grid's own numbers stay under data/processed/, gitignored, as the first
sweep's did: they are not a result, they are how the parameters were chosen.
MSG

say "each commit's own tree passes its own tests"
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
git --no-pager log --oneline -6
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -2"
