#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit: the booster moves to the cell the extension of the second sweep
# left standing, and every report that cell changes is regenerated under it.
#
# WHY ONE COMMIT AND NOT THREE. The parameter move forces the ladder run; the
# ladder run rewrites reports/baseline.md and the residuals under it; the
# residuals make reports/determinism.md stale; the regenerations are what let
# four reports carry a provenance footer at last; and the footers move
# REPORT_INPUTS_DECLARED, REPORTS_WITHOUT_FOOTER and PROVENANCE_PENDING, which
# regenerate docs/provenance.md. Every split tried leaves an intermediate commit
# where check 2, check 8 or check 10 fails, because the register and the module
# it registers would be apart for one commit. The register is one file and this
# script stages whole files, so the honest arrangement is one commit that is
# green before and after.
#
# WHAT MOVED AND WHY, IN ONE PLACE:
#   * BOOSTER_PARAMS goes from (3,000, 0.015, 255, 20) to (6,000, 0.03, 127,
#     20). The account is in the comment above it: the trees axis bought about
#     0.59 s on three paired contrasts, and the leaf and rate setting was taken
#     on cost between two cells the instrument cannot separate.
#   * EVERY FIGURE IN reports/baseline.md MOVES, and that is the point of the
#     commit rather than a defect in it. The prompt this block ran from also
#     asked for an assertion that no figure moved except three; the two cannot
#     both hold, because run_baselines.py fits every rung with BOOSTER_PARAMS.
#     The owner chose the parameter move on 23 September and the assertions
#     below are what replaced that one.
#   * SEED_AVERAGE_ADOPTED does not move. A five-seed submission was timed for
#     the first time in this block, at 29 min 20 s against 6 min 55 s for one
#     seed, and the decision is the owner's.
#   * FOUR MORE REPORTS MOVE, and not because anything in them was edited. The
#     ladder rewrites data/processed/residuals/, every report that declares one
#     of those files goes stale by digest, and check 8 catches it: congestion,
#     headroom, occupancy and residual are regenerated for that reason alone.
#     reports/headroom.md also carries LADDER_STEP_QUOTED_S, a figure quoted
#     out of baseline.md, and its generator REFUSED TO WRITE until that quote
#     was moved from 17.60 to the 15.98 the new ladder prints. That refusal is
#     the register working as designed and is worth saying out loud.
#   * reports/stand_coverage.md was regenerated and came back byte for byte
#     identical, so it is not in this commit.
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * it does not fetch anything. The per-airport OSM pin is code and a register
#     entry; no query was sent, and Block 11 is where it is exercised.
#   * it does not touch docs/plan.md or docs/paper.md. The paper is stale by its
#     own pending entry and Block 10 owns it.
#   * it footers no report that no run in this tree regenerates. Two are left,
#     both geometry runs that rewrite inputs to other reports.

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

BASE="c42867e"
PATHS="Makefile
docs/provenance.md
reports/baseline.md
reports/congestion.md
reports/determinism.md
reports/geometry_gaps.md
reports/headroom.md
reports/occupancy.md
reports/residual.md
reports/transfer.md
scripts/check_determinism.py
scripts/check_provenance.py
scripts/commit_last_ladder.sh
scripts/fetch_osm.py
scripts/report_geometry_gaps.py
scripts/report_headroom.py
scripts/report_transfer.py
scripts/run_baselines.py
scripts/sweep_params.py
src/taxiout/models.py
src/taxiout/osm.py
src/taxiout/provenance.py
tests/test_sweep_params.py"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say "this sits on the commit it says it sits on"
if ! git merge-base --is-ancestor "$BASE" HEAD 2>/dev/null; then
    echo "$BASE is not an ancestor of HEAD; this script expects to follow it"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' "$BASE")"

say "the working tree holds these 23 paths and nothing else"
expected="$(printf '%s\n' "$PATHS" | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok  23 paths"

say "1/9  BOOSTER_PARAMS is exactly the adopted cell"
"$PY" - <<'PY'
import sys
sys.path.insert(0, "src")
from taxiout.models import BOOSTER_PARAMS, SEED_AVERAGE_ADOPTED

want = {"n_estimators": 6000, "learning_rate": 0.03, "num_leaves": 127,
        "min_child_samples": 20}
got = {k: BOOSTER_PARAMS[k] for k in want}
if got != want:
    raise SystemExit(f"BOOSTER_PARAMS is {got}, not the adopted cell {want}")
print(f"  ok  {got}")

# 2/9 in the same interpreter: the constant the owner decides, untouched.
if SEED_AVERAGE_ADOPTED is not False:
    raise SystemExit(
        "SEED_AVERAGE_ADOPTED is not False. This block measured what a five-seed"
        " submission costs and moved nothing; flipping it is the owner's, on the"
        " measurement."
    )
print("  ok  SEED_AVERAGE_ADOPTED is still False")
PY

say "3/9  reports/baseline.md was written by this cell, and says so"
"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from taxiout.models import BOOSTER_PARAMS

text = Path("reports/baseline.md").read_text(encoding="utf-8")
# The report prints BOOSTER_PARAMS in full. A report regenerated under the old
# cell would still carry the old numbers here, which is the failure this
# catches: a stale baseline.md beside a moved constant.
for key in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples"):
    want = f"`{key}={BOOSTER_PARAMS[key]}`"
    if want not in text:
        raise SystemExit(
            f"reports/baseline.md does not print {want}; it was not written by"
            " this cell"
        )
print("  ok  the report prints the cell it was fitted with")
PY

say "4/9  HEADROOM_TRIGGER_S is still 5.0 and EXPECTED_HEADROOM_S is untouched"
n="$(grep -c '^HEADROOM_TRIGGER_S = 5\.0$' scripts/run_baselines.py || true)"
if [ "$n" != "1" ]; then
    echo "expected exactly one 'HEADROOM_TRIGGER_S = 5.0'; found $n"
    exit 1
fi
git show "HEAD:scripts/run_baselines.py" > "$work/run_baselines.head.py"
"$PY" - "$work" <<'PY'
import ast
import sys
from pathlib import Path

work = Path(sys.argv[1])


def expected(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == (
            "EXPECTED_HEADROOM_S"
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("EXPECTED_HEADROOM_S not found")


was = expected(work / "run_baselines.head.py")
now = expected("scripts/run_baselines.py")
if was != now:
    raise SystemExit(f"EXPECTED_HEADROOM_S moved: {was} -> {now}")
if len(now) != 4:
    raise SystemExit(f"EXPECTED_HEADROOM_S has {len(now)} rows, not 4")
print(f"  ok  4 rows, unchanged: {now}")
PY

say "5/9  the headroom overshoot still fires, on the same step and estimate"
"$PY" - <<'PY'
import re
from pathlib import Path

text = Path("reports/baseline.md").read_text(encoding="utf-8")
para = next(
    (p for p in re.split(r"\n\s*\n", text) if p.startswith("**The headroom overshoot.")),
    None,
)
if para is None:
    raise SystemExit("reports/baseline.md carries no *The headroom overshoot* section")
if "The leakage audit, applied" in text:
    raise SystemExit("the old section name is still in reports/baseline.md")
if "which is more than the" not in para or " it fires by " not in para:
    raise SystemExit(f"the overshoot does not fire in this run:\n{para[:400]}")
# The estimate it is read against is the quoted one and has not moved.
if "+10.3 s" not in para:
    raise SystemExit("the section is not read against the +10.3 s estimate")
print("  ok  it fires, against the same +10.3 s estimate")
PY

say "6/9  the fourth reading is in, and the history reads fired/missed five times"
"$PY" - <<'PY'
import re
from pathlib import Path

text = Path("reports/baseline.md").read_text(encoding="utf-8")
para = next(
    (p for p in re.split(r"\n\s*\n", text) if p.startswith("**What this test has read")),
    None,
)
if para is None:
    raise SystemExit("the run-by-run paragraph is missing")
for fig in ("14.64", "4.345"):
    if fig not in para:
        raise SystemExit(f"the fourth reading's {fig} is not in the paragraph")
verdicts = re.findall(r"the trigger (fired|missed)|and (fires|misses) by", para)
flat = [a or b for a, b in verdicts]
want = ["fired", "missed", "fired", "missed", "fires"]
if flat != want:
    raise SystemExit(f"the history reads {flat}, not {want}")
if para.count("times the estimate") != 5:
    raise SystemExit(
        f"{para.count('times the estimate')} ratios printed, not 5"
    )
print("  ok  fired, missed, fired, missed, fires, each with its ratio")
PY

say "7/9  congestion.md keeps the name *The leakage audit*, and docs/ moves once"
if ! grep -q "The leakage audit" reports/congestion.md; then
    echo "reports/congestion.md lost the section the rename left it"
    exit 1
fi
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- docs/ \
         | sed 's/^...//' | grep -v '^docs/provenance.md$' || true)"
if [ -n "$moved" ]; then
    echo "docs/ moved somewhere other than the generated register: $moved"
    exit 1
fi
echo "ok  congestion.md keeps it; docs/ moves only in the generated register"

say "8/9  every declared report is newer than its generator"
"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from taxiout.provenance import REPORT_INPUTS_DECLARED

bad = []
for gen, report in REPORT_INPUTS_DECLARED:
    if not Path(report).exists():
        bad.append(f"{report} does not exist")
        continue
    if Path(report).stat().st_mtime < Path(gen).stat().st_mtime:
        bad.append(f"{report} is older than {gen}; regenerate it before committing")
    else:
        print(f"  ok  {report} newer than {gen}")
if bad:
    raise SystemExit("\n".join(bad))
PY

say "9/9  the gates"
"$RUFF" check .
"$RUFF" format --check .
"$PY" -m pytest -q
"$PY" scripts/check_licence_hygiene.py
"$PY" scripts/check_provenance.py

say "the strings this commit is supposed to contain"
check() {
    if ! grep -q "$2" "$1"; then
        echo "missing from $1: $2"
        exit 1
    fi
    echo "ok  $1: $2"
}
check src/taxiout/models.py '"n_estimators": 6000'
check src/taxiout/models.py 'SEED_AVERAGE_ADOPTED: bool = False'
check scripts/run_baselines.py 'HEADROOM_REFERENCE_READING = 2'
check scripts/fetch_osm.py 'pin-from-cache'
check src/taxiout/osm.py 'def cached_osm_base'
check src/taxiout/provenance.py 'REBUILD_PINS'
check reports/baseline.md 'What this report was generated from'
check reports/transfer.md 'What this report was generated from'
check reports/determinism.md 'What this report was generated from'
check reports/geometry_gaps.md 'What this report was generated from'

say "the commit"
git add -A -- $PATHS
git commit -q -F - <<'MSG'
Move the booster to the cell the extension left standing, and rebuild under it

The second sweep's cell sat on an edge on all four axes and the
pre-registration fixed that the grid would not be extended again. It was
reopened on 22 September anyway, with the edge recorded and its direction
known, and that is written into the comment above BOOSTER_PARAMS rather than
presented as the plan. Three axes were probed one at a time; only the trees
axis won.

Read as paired contrasts within a seed, 3,000 to 6,000 trees buys -0.6004,
-0.6019 and -0.5591 s across two seeds and two leaf settings, a spread of
0.0428 s against a 0.1815 s floor. The seed moves the level by 0.2872 s and the
contrast by less than two thousandths. The leaf and rate setting was taken on
cost between two cells this instrument cannot separate: (0.015, 255) at 6,000
is the better reading by 0.1350 s, which is 0.74 of the floor and less than
half that cell's own seed-to-seed spread, and it costs 1.35x the October fit
time. It is named beside the chosen cell so the choice is visible.

The adopted cell reads 214.4391 s on October against 214.9045 s, 2.56 floors,
with a day-clustered interval of [-0.9492, +0.0009] that spans zero by nine
ten-thousandths. Both instruments are printed and neither is suppressed.

Every figure in reports/baseline.md moves with the cell, and the ladder run
that moved them also carries two changes queued behind it since 22 September.
HEADROOM_READINGS gains the fourth reading the migration record holds, +14.64 s
at an overshoot of 4.345 s on 19 September, so the published history reads
fired, missed, fired, missed and then this run rather than fired, missed,
fired. And the section that fires is renamed from The leakage audit, applied to
The headroom overshoot, which is what it measures: a residual fit bounds a
refit from below, so the gap between them is interaction value, and the
estimate is not stale, having been recomputed to within five hundredths. The
name leakage audit stays with the source registry in congestion.md, which is a
leakage test and which passes. No threshold, no constant and no verdict moves:
the trigger is still 5.0 s, the estimates are untouched, and it still fires.

Four reports carry a provenance footer for the first time, because a run
regenerated each. reports/geometry_gaps.md needed its generator fixed first:
three cells printed -0 or +0 for a difference the printed precision has already
discarded, which made a report that had not moved look as though it had.
reports/determinism.md was stale against residuals the ladder rewrote and is
regenerated rather than repaired. Two reports still carry none and are listed
with what each costs.

scripts/fetch_osm.py gains --pin-from-cache, which reads each airport's own
timestamp_osm_base and pins that airport's query to it. The eleven cache files
carry seven distinct stamps spanning nine minutes, so one --date is not the
rebuild. The pin is registered as recoverable per airport and unverified: no
query was sent here.

Four further reports move because the ladder rewrote the residuals they
declare, and one of them, reports/headroom.md, quotes the ladder's step out of
baseline.md; its generator refused to write until that quote was moved from
17.60 to 15.98. The Makefile's timings are all re-measured, and its note that
'make reports' takes two to three minutes is corrected: that was one of its
three scripts, and the target is about an hour.

SEED_AVERAGE_ADOPTED does not move. A five-seed submission was timed for the
first time, at 29 min 20 s against 6 min 55 s for one seed, which was the last
unmeasured figure the recommendation rested on, and what is done with the
measurement is a separate decision.
MSG

say "done"
git log -1 --stat --format='%h %s'
