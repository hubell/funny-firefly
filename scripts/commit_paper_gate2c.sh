#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit on top of the current history, additive, nothing rewritten:
# the paper's §6.5 and §9.5, written from the reports the Gate 2c follow-up
# regenerated, plus this script.
#
# The draft cites reports rather than restating them. §9.5 carries a handful
# of figures marked with a dagger that no report prints yet -- the shape of the
# residual tail, its enrichment classes, the error by plan lag, and the one row
# that is a third of July's pooled squared error -- computed for the draft from
# the stored residuals. The drafting notes say so and say which generator
# section replaces them; this script checks that the daggers and that note are
# both present, so the draft cannot claim a computed source it does not have.
#
# Runs the citation checks, the tests and the licence gate first. Does not push.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

PAPER="docs/paper.md scripts/commit_paper_gate2c.sh"
for f in $PAPER; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the two sections are written and the reports they cite carry their sections"
for pair in \
    "reports/occupancy.md|^## Where the gain lands" \
    "reports/baseline.md|^### Where the stand-occupancy gain lands" \
    "reports/baseline.md|^### The same block without the subject's take-off" \
    "reports/baseline.md|^## The recorded-taxi tail, out of the fit" \
    "reports/congestion.md|^## The ceiling"; do
    f="${pair%%|*}"; h="${pair#*|}"
    if ! grep -qE "$h" "$f"; then
        echo "$f has no section matching '$h'; regenerate it"
        exit 1
    fi
done
for h in "^### 6.5 .*\[settled" "^### 9.5 .*\[settled"; do
    if ! grep -qE "$h" docs/paper.md; then
        echo "docs/paper.md: no section matching '$h'"
        exit 1
    fi
done
echo "ok"

say "§9.5's dagger figures are declared as computed for the draft"
if ! grep -q "†" docs/paper.md; then
    echo "no dagger in the draft; either the generator section landed (drop this check) or the marks were lost"
    exit 1
fi
if ! grep -q "What the residual is" docs/paper.md; then
    echo "the draft does not name the generator section that replaces the dagger figures"
    exit 1
fi
echo "ok"

say "§8 names the rung the ladder chose"
CHOSEN=$(grep -oE '^\*\*Chosen: `[^`]+`' reports/baseline.md | sed -E 's/^\*\*Chosen: `//; s/`$//')
if [ -z "$CHOSEN" ]; then
    echo "reports/baseline.md names no chosen rung; run make baseline"
    exit 1
fi
if ! awk '/^## 8\. Results/{p=1} /^## 9\. Discussion/{p=0} p' docs/paper.md \
        | tr '\n' ' ' | grep -qF "$CHOSEN"; then
    echo "reports/baseline.md chose '$CHOSEN' but §8 of the draft does not name it"
    exit 1
fi
echo "ok: $CHOSEN"

say "every report, doc and test the draft cites exists"
MISSING=0
for f in $(grep -oE '(reports|docs|tests)/[a-z_]+\.(md|svg|py)' docs/paper.md | sort -u); do
    if [ ! -f "$f" ]; then
        echo "  missing: $f"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "the draft cites a file that does not exist; fix the citation or generate the report"
    exit 1
fi
echo "ok"

say "no literal stand identifier in the draft"
python3 - <<'EOF'
import re, sys
from pathlib import Path
sys.path.insert(0, "src")
text = Path("docs/paper.md").read_text(encoding="utf-8")
try:
    import pandas as pd  # noqa: F401
    from taxiout.challenge import load_movements, load_ranking, to_pandas
    root = Path("data/challenge")
    if not (root / "ranking.parquet").exists():
        raise FileNotFoundError
    ids = set()
    for table in (
        load_movements(root, ["STAND_mvt"]),
        load_ranking(root, ["STAND_mvt"]),
    ):
        ids |= set(to_pandas(table)["STAND_mvt"].dropna().astype(str))
except (ImportError, FileNotFoundError):
    print("  challenge data or pandas absent; the licence gate below is the check")
    sys.exit(0)
ids = {v.strip() for v in ids if len(v.strip()) >= 3
       and re.search(r"[A-Za-z]", v) and re.search(r"[0-9]", v)}
hits = [i for i in ids if re.search(r"(?<![\w])" + re.escape(i) + r"(?![\w])", text)]
if hits:
    print(f"  {len(hits)} literal stand identifier(s) in docs/paper.md")
    sys.exit(1)
print("ok")
EOF

say "tests"
python3 -m pytest -q

say "licence gate"
python3 scripts/check_licence_hygiene.py

say "1/1 the paper's two pending sections"
git add $PAPER
git commit -q -m "Write the stand section and the residual section of the paper

§6.5 is written from the regenerated occupancy and baseline reports, and
not from the bound. The block is the stand's recent history; the bound
lives inside it in one band, and the section gives the band's share of the
block on each leg, the type split inside it, the floor that loses, the
schedule-anchored diagnostic that keeps 42%, 34% and 4% of the block under
an operational rule, and the audit that missed its trigger by twenty-nine
thousandths of a second after firing on the fit before the tail exclusion.

§9.5 says what the residual is: a thin tail of individual holds, not
shared, enriched among heavies, departures with no Network Manager record
and re-occupied stands, growing with the plan lag as a conditional mean
between two modes must. It carries a handful of figures no report prints
yet, marked with a dagger and declared in the drafting notes as computed
for the draft from the stored residuals, to be replaced when the baseline
report gains the section that prints them. It also records which July
figure the paper quotes for the submitted model -- the three-airport one --
and why: one Fiumicino row carries a third of the pooled July squared
error and July 2026 has no Fiumicino rows.

§8's preamble names the rung the submission now fits from and says its
figures are pre-refit until the freeze pass; the drafting notes say what
is written, what waits, and that seed averaging stands declined."

say "result"
git --no-pager log --oneline -3
echo
git status --short

echo
echo "Not pushed. Review with: git log -p -1"
