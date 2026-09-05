#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Two commits on top of the current history, additive, nothing rewritten:
#
#   1. the thin-pair section of the geometry-gaps report and its generator
#      -- where in the ranking set the memorized pair runs out, which stands
#      are new outright, and what the pair mean is worth shrunk toward the
#      airport mean against shrunk toward the geometry;
#   2. the paper's working draft and this script.
#
# The draft cites reports rather than restating them, so it stays true as
# long as those reports regenerate. Sections whose figures are not final
# are marked in the draft itself; this script does not check that, because
# a draft is allowed to be incomplete -- it checks only that every report,
# doc and test the draft names exists, so a citation cannot rot silently.
#
# The two commits are separable because they belong on either side of the
# encoding batch (scripts/commit_encoding.sh): that batch's tests parse the
# thin-pair table commit 1 adds, and the paper quotes the ladder that batch
# regenerates. So:
#
#     scripts/commit_paper.sh report    # commit 1, before commit_encoding.sh
#     scripts/commit_paper.sh paper     # commit 2, after it
#     scripts/commit_paper.sh           # both, if the batch is already in
#
# Runs the tests and the licence gate first. Does not push.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-both}"
case "$MODE" in
    report|paper|both) ;;
    *) echo "usage: $0 [report|paper]"; exit 1 ;;
esac

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

GENERATOR="scripts/report_geometry_gaps.py reports/geometry_gaps.md"
PAPER="docs/paper.md scripts/commit_paper.sh"
for f in $GENERATOR $PAPER; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the report carries the section the paper cites"
if ! grep -q '^## Where memorization runs out' reports/geometry_gaps.md; then
    echo "reports/geometry_gaps.md has no thin-pair section; run make geometry-gaps"
    exit 1
fi
echo "ok"

if [ "$MODE" != "report" ]; then
    say "the paper's results section names the rung the ladder chose"
    CHOSEN=$(grep -oE '^\*\*Chosen: `[^`]+`' reports/baseline.md | sed -E 's/^\*\*Chosen: `//; s/`$//')
    if [ -z "$CHOSEN" ]; then
        echo "reports/baseline.md names no chosen arm; run make baseline"
        exit 1
    fi
    # The draft wraps the rung's name across a line; compare §8 on one line.
    if ! awk '/^## 8\. Results/{p=1} /^## 9\. Discussion/{p=0} p' docs/paper.md \
            | tr '\n' ' ' | grep -qF "$CHOSEN"; then
        echo "reports/baseline.md chose '$CHOSEN' but §8 of the draft does not name it; re-read §8"
        exit 1
    fi
    echo "ok: $CHOSEN"
fi

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
    import pandas as pd
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

if [ "$MODE" != "paper" ]; then
say "1/2 where memorization runs out"
git add $GENERATOR
git commit -q -m "Measure where the memorized pair runs out in the ranking set

The paper's first draft explained the gap between the geometry's 0.1 s
on a forward split and its +14.7 s on held-out pairs by asserting that
the ranking months reuse 2025's stands and runways. That was wrong, and
the figures that replaced it were typed from a scratch calculation --
two of them incorrectly: a network-wide count of new stands attributed
to one airport, and transfer figures from a stand-in rather than the
report. So the measurement moves into the generator.

The geometry-gaps report gains a section that counts, for every 2026
ranking departure, how many unflagged 2025 departures its raw (airport,
stand, runway) triple carries, in the baseline's own buckets, pooled,
per airport and per ranking month; lists the stands and designators the
training year never saw, per airport, with the shape of each identifier
and whether the OpenStreetMap join can place it; and, on the December
and July forward splits, scores the pair's own mean against that mean
shrunk toward the airport mean and toward the geometric prior, by
bucket, with the shrinkage weight estimated per airport by the encoding's
own construction. The prior is fitted by the same function the shrunk
encoding uses, on the same columns, so the two reports agree by
construction; the airport-mean strata reproduce the baseline's to the
tenth, which is the check.

Nine point eight percent of the ranking rows sit on a pair seen a
hundred times or fewer, 24 stands are new outright, and on pairs seen
one to ten times the raw pair mean is worse than the airport mean on
both splits while the mean shrunk toward the geometry beats the mean
shrunk toward the airport by about twenty seconds.

The identifier refusal now covers the ranking set's stands as well as
the training set's and the matrix's: a stand new in 2026 is in the
ranking set and nowhere else. Two copies of the full frame in the wake
section are cut to the columns they read, which brings the generator
under three gigabytes; no figure in the existing sections moves."
fi

if [ "$MODE" != "report" ]; then
say "2/2 the paper's working draft"
git add $PAPER
git commit -q -m "Draft the paper: the argument, the settled sections, and what waits

Gate 4 begins while Gate 2's numbers are still warm, because the plan
reserves the final week for the paper rather than squeezing it. Every
figure names the report that computes it, and the draft cites rather
than restates, so it stays true as long as the reports regenerate. This
script checks that every report, doc and test the draft names exists
and that no literal stand identifier reached the prose.

What the order of the argument fixes now, while it is cheap to change:
the tail analysis is its own early section rather than a footnote,
stated as the RMSE-optimal prediction under the scoring rule for rows
whose recorded value no aircraft produced, and nothing else. The worked
airport carries the correctness argument for the geometry, so a reader
need not take 24,000 rows on trust. The queue layer that was not built
gets a section of its own, because a measured stop is a result.

Three claims are stated with the split each was measured on, and the
one that distinguishes the entry -- that the geometry carries over to
stand-runway pairs never seen together -- is reported as a difference
in differences with a cluster-bootstrap interval, whichever way it
falls. Where memorization runs out is measured rather than asserted,
and the section that once asserted the opposite says so.

Two things are reported that a shorter paper would bury. The shrunk
identifier encoding, pre-registered against the thin pairs and guarded
against four leakage modes, bought most of its gain from the pairs it
was not aimed at, and is written up under that description. The
headroom ceiling it was built against was wrong twice -- once in
arithmetic, in the direction that made the work look worth doing, and
once in what it measured, since it looked only where the block was
aimed -- and both errors are stated in the discussion.

The results section is written against the submission rung as named in
taxiout.models, which is the encoding arm; two placeholders remain for
the organizers' figure, and the drafting notes say which rung each
figure is anchored to and what to re-read after the freeze-week rebuild."
fi

say "result"
git --no-pager log --oneline -3
echo
git status --short

echo
echo "Not pushed. Review with: git log -p -2"
