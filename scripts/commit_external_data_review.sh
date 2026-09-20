#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# The external datasets reviewed on 20 September 2026 and not taken, recorded
# where the earlier exclusions are: the README's data sources, the licence
# table in docs/prior_art.md, and the paper's external-data section. One
# commit. Additive; nothing is rewritten. Not pushed.
#
# Runs the licence gate first and stops if it fails; the change is prose in
# three documents no test reads, so the suite and the lint are not run here.
# Run scripts/commit_submission_rebuild.sh before this one: this script expects
# the tree to hold nothing but the three documents and itself.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

say "the three documents carry the review"
for pair in "README.md|OpenSky's published scientific datasets, ten of them" \
            "docs/prior_art.md|OpenSky scientific datasets (the ten listed" \
            "docs/paper.md|ten published scientific datasets were reviewed"; do
    f="${pair%%|*}"; needle="${pair#*|}"
    if ! grep -qF "$needle" "$f"; then
        echo "$f does not carry the review paragraph"
        exit 1
    fi
done
echo "ok"

say "only the three documents and this script are changed"
expected=" M README.md
 M docs/paper.md
 M docs/prior_art.md
?? scripts/commit_external_data_review.sh"
actual="$(git -c core.fileMode=false status --short)"
if [ "$actual" != "$expected" ]; then
    echo "unexpected working-tree state:"
    printf '%s\n' "$actual"
    exit 1
fi
echo "ok"

say "licence gate"
python3 scripts/check_licence_hygiene.py

say "commit"
git add README.md docs/paper.md docs/prior_art.md scripts/commit_external_data_review.sh
git commit -q -F - <<'MSG'
Record the OpenSky scientific datasets reviewed, and why none is used

Ten datasets are listed on the network's scientific-datasets page as of
20 September 2026. Each was read against the three tests every external
source meets here, in order: an open licence; ground-phase coverage of
2025 at the ten airports; and a mechanism the exposed fields lack, which
reports/residual.md narrows to separating a gate delay from a hold. None
passes all three. The weekly Monday state-vector snapshots are the only
set with surface positions and carry the network's data terms, which
this entry already reads as research use, so they stay in the role the
historical database has: a diagnostic for the paper at most. The 2024
challenge dataset holds 2022 taxi-out times with no stand or runway; the
other eight carry no ground operations.

The README's data sources, the licence table in docs/prior_art.md and
the paper's external-data section now say so, beside the PRU portal,
OPDI and ERA5 entries, so a reader sees what was ruled out and on which
test. No code, feature or report changes.
MSG

say "result"
git --no-pager log --oneline -3
echo
git -c core.fileMode=false status --short
echo
echo "Not pushed."
