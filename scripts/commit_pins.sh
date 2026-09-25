#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit: the eleven per-airport OSM pins move out of the gitignored cache,
# where a rebuild destroys them, and into the register, and the fetcher reads
# them from there. The mode that read them from the cache is retired.
#
# WHY THE CACHE COULD NOT KEEP THEM. Overpass stamps every answer with the
# answering instance's own current database state and never with the [date:]
# pin, measured on 25 September on three answers carrying a 2026 pin, a 2020
# pin and none. So a pinned rebuild writes eleven cache files stamped with the
# rebuild day, and --pin-from-cache, which read each airport's pin out of its
# cache header, would have worked exactly once and then pinned every airport to
# the rebuild day on every later run, with no error. And data/raw/ is
# gitignored: a clean checkout had no cache, and so no pins, which is exactly
# where the reproduction has to start.
#
# WHAT MOVES, IN ONE PLACE:
#   * src/taxiout/provenance.py gains OSM_PINS, one entry per airport, read
#     from the cache files on 25 September and checked against the two earlier
#     readings of 22 September, Block 09e's report and the freeze-ordering note.
#     All four sources agree on all eleven. REBUILD_PINS now names
#     --pin-from-register and OSM_PINS. PROVENANCE_PENDING goes from four items
#     to three: the five-seed timing is closed, because Block 09g measured it at
#     29 min 20 s and the entry outlived the measurement, and the pin entry is
#     rewritten rather than closed.
#   * scripts/fetch_osm.py: --pin-from-register pins each airport to its entry;
#     --pin-from-cache refuses and says why; --compare checks a fetched answer
#     against a reference element by element; --control-date asks the instance
#     that answered for one airport at a date whose answer is known to differ.
#   * src/taxiout/osm.py: compare_elements and read_payload; fetch_airport
#     takes mirrors=; cached_osm_base says it is never a pin source. Two new
#     imports moved POLITE_DELAY_S's comment from line 47 to line 49, and its
#     register citation moves with it. Check 12 holds that.
#   * docs/provenance.md is regenerated from the register and moves with it.
#   * README.md's Reproducing section names the rebuild and how it is checked.
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * it sends no query. Nothing was fetched, and nothing under data/raw/ was
#     written. Block 11 runs the pinned rebuild, the comparison and the control.
#   * it regenerates no report and moves no figure. reports/ is HEAD's tree.
#   * it leaves the Makefile alone. `make data` still fetches today's map,
#     unpinned, and the README now says so beside the command that pins.
#   * it moves neither BOOSTER_PARAMS nor SEED_AVERAGE_ADOPTED.

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

BASE="208ccae"
PATHS="README.md
docs/provenance.md
scripts/check_provenance.py
scripts/commit_pins.sh
scripts/fetch_osm.py
src/taxiout/osm.py
src/taxiout/provenance.py
tests/test_fetch_osm.py
tests/test_provenance.py"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say "this sits on the commit it says it sits on"
# Read from the history rather than asked of merge-base, so every git command
# before the staging below is one of status, log, show and rev-parse. Captured
# first, because grep -q leaving a pipe early would fail it under pipefail.
history="$(git log --format=%H HEAD)"
if ! grep -qx "$(git rev-parse "$BASE^{commit}")" <<< "$history"; then
    echo "$BASE is not an ancestor of HEAD; this script expects to follow it"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' "$BASE")"

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

say "1/5  the gates, first"
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
if [ "$n" != "13" ]; then
    echo "expected 13 verdicts and found $n; a check that is not called cannot fail"
    cat "$work/provenance.txt"
    exit 1
fi
echo "ok  13 provenance checks green"
if "$PY" scripts/check_provenance.py --strict > "$work/strict.txt" 2>&1; then
    echo "--strict passed, which would mean the freeze condition is met; it is not"
    exit 1
fi
echo "ok  --strict still fails, because the pending list is not empty"

say "2/5  reports/ is HEAD's, as a whole tree"
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- reports/)"
if [ -n "$moved" ]; then
    echo "reports/ moved, and this commit regenerates no report:"
    echo "$moved"
    exit 1
fi
echo "ok  $(git rev-parse --short HEAD:reports), unchanged"

say "3/5  BOOSTER_PARAMS and SEED_AVERAGE_ADOPTED are HEAD's"
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
    raise SystemExit("SEED_AVERAGE_ADOPTED is not False; that decision is not this commit's")
cell = {k: now["BOOSTER_PARAMS"][k] for k in
        ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")}
print(f"  ok  BOOSTER_PARAMS unchanged, {cell}; SEED_AVERAGE_ADOPTED still False")
PY

say "4/5  docs/ moves in the generated register alone, and together with it"
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- docs/ \
         | sed 's/^...//')"
if [ "$moved" != "docs/provenance.md" ]; then
    echo "docs/ should move in docs/provenance.md alone; it moves in: ${moved:-nothing}"
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
print("  ok  docs/provenance.md is what the register generates, and nothing else in docs/ moved")
PY

say "5/5  eleven pins in the register, and the recorded rebuild reads them there"
"$PY" - <<'PY'
import sys
from collections import Counter

sys.path.insert(0, "src")
from taxiout.airports import AIRPORTS
from taxiout.provenance import OSM_PINS, REBUILD_PINS

icaos = [p.icao for p in OSM_PINS]
if icaos != [a.icao for a in AIRPORTS]:
    raise SystemExit(f"OSM_PINS is not one pin per airport in fetch order: {icaos}")
stamps = Counter(p.timestamp_osm_base for p in OSM_PINS)
first, last = min(stamps), max(stamps)
if (len(stamps), first, last) != (7, "2026-09-01T13:23:05Z", "2026-09-01T13:32:15Z"):
    raise SystemExit(f"OSM_PINS carries {len(stamps)} moments from {first} to {last}")
for k in REBUILD_PINS:
    if "--pin-from-register --force" not in k.how or "--pin-from-cache" in k.how:
        raise SystemExit(f"REBUILD_PINS records the wrong command: {k.how}")
print(f"  ok  {len(OSM_PINS)} pins at {len(stamps)} moments, {first} to {last}")
PY

say "the strings this commit is supposed to contain"
check() {
    if ! grep -qF -- "$2" "$1"; then
        echo "missing from $1: $2"
        exit 1
    fi
    echo "ok  $1: $2"
}
check src/taxiout/provenance.py 'OSM_PINS: tuple[OsmPin, ...] = ('
check src/taxiout/provenance.py '`python scripts/fetch_osm.py --pin-from-register --force`'
check src/taxiout/provenance.py '"line 49: the Overpass free tier'
check scripts/fetch_osm.py 'PIN_FROM_CACHE_RETIRED = ('
check scripts/fetch_osm.py '"--pin-from-register",'
check scripts/fetch_osm.py '"--compare",'
check scripts/fetch_osm.py '"--control-date",'
check src/taxiout/osm.py 'def compare_elements('
check tests/test_fetch_osm.py 'def test_pin_from_cache_refuses_and_says_why('
check tests/test_provenance.py 'def test_the_osm_pins_equal_the_cache_stamps_while_the_original_cache_exists('
check docs/provenance.md '### The OSM pins, one per airport'
check README.md 'python scripts/fetch_osm.py --pin-from-register --force'

say "staging"
git add -- $PATHS
git -c core.fileMode=false status --short

git commit -q -F - <<'MSG'
Record the eleven OSM pins where a rebuild cannot destroy them, and retire the mode that read them from the cache

Block 09h exercised the per-airport OSM pin on one airport and found two
things that change how the rebuild has to work.

Overpass stamps every answer with the answering instance's own current
database state and never with the [date:] pin, measured on three answers
carrying a 2026 pin, a 2020 pin and none. So a pinned rebuild writes eleven
cache files stamped with the rebuild day, and --pin-from-cache, which read
each airport's pin out of that stamp, would have worked exactly once and then
pinned every airport to the rebuild day on every later run, with no error and
a plausible result. And the eleven values existed only in data/raw/, which is
gitignored: a clean checkout had no cache and therefore no pins, so the
reproduction the paper promises could not start.

The pins are now in the register. OSM_PINS carries one entry per airport,
read from the cache files on 25 September and the same eleven values Block
09e read on 22 September: seven distinct moments from 2026-09-01T13:23:05Z to
13:32:15Z. A test holds them against the cache while the original cache
exists, and skips with that reason where it does not. It will fail after a
pinned rebuild, and that is correct: from then on the register is right and
the cache is not.

scripts/fetch_osm.py reads them from there.

  - --pin-from-register pins each airport to its own entry. It needs
    --force, and an airport without an entry fails rather than fetching
    unpinned. REBUILD_PINS and the README's Reproducing section name it.
  - --pin-from-cache refuses, and says why.
  - --compare FETCHED REFERENCE matches each airport's parsed elements on
    type and id, compares every field, and prints nodes, ways and relations
    on both sides. Run on Block 09h's own payloads it reproduces that
    block's comparisons by hand: 494 of 494 identical against the unpinned
    probe, and 86, 225 and 161 against the 2020 control.
  - --control-date asks the instance that answered for one airport at a
    date whose answer is known to differ, and exits non-zero if it does not
    differ. That is the proof a header cannot give.

PROVENANCE_PENDING goes from four items to three. The five-seed timing is
closed: Block 09g measured it at 29 min 20 s and the entry outlived the
measurement. The pin entry is rewritten and not closed: the pin has served on
one airport on one instance, and all eleven, with the comparison and the
control, are Block 11's.

No query was sent, nothing under data/raw/ was written, no report is
regenerated and no figure moves. BOOSTER_PARAMS and SEED_AVERAGE_ADOPTED are
untouched.
MSG

say "the commit"
git log -1 --stat --format='%h %s'

say "reports/ is the parent's tree"
if [ "$(git rev-parse HEAD:reports)" != "$(git rev-parse HEAD^:reports)" ]; then
    echo "reports/ moved in this commit"
    exit 1
fi
echo "ok  $(git rev-parse --short HEAD:reports)"

say "the commit's own tree, in a worktree with no OSM cache"
# The condition this commit exists for: the register carries the pins and the
# cache does not exist. The cache-agreement test skips there, with its reason.
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
) || {
    git worktree remove --force "$wt"
    echo "the committed tree does not pass its own gates."
    exit 1
}
git worktree remove --force "$wt"

say "done"
echo "Nothing is pushed. 'git show HEAD' to read it, 'git reset --hard HEAD^' to undo."
