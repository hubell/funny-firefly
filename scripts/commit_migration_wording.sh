#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One sentence in the migration record, corrected to what was measured. One
# commit. Additive; nothing is rewritten. Not pushed.
#
# Runs the licence gate first and stops if it fails; the change is prose in a
# report no test reads, so the suite and the lint are not run here.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

say "the sentence reads as measured"
if grep -q "column-wise or not" reports/migration-2026-09-10.md; then
    echo "reports/migration-2026-09-10.md still says 'column-wise or not'"
    exit 1
fi
if ! grep -q "with the column-wise setting every Phase 4 fit used" \
        reports/migration-2026-09-10.md; then
    echo "the corrected sentence is not in reports/migration-2026-09-10.md"
    exit 1
fi
echo "ok"

say "only the record and this script are changed"
changed="$(git -c core.fileMode=false status --short | grep -v '^?? scripts/commit_migration_wording.sh$' || true)"
if [ "$changed" != " M reports/migration-2026-09-10.md" ]; then
    echo "unexpected working-tree state:"
    git -c core.fileMode=false status --short
    exit 1
fi
echo "ok"

say "licence gate"
python3 scripts/check_licence_hygiene.py

say "commit"
git add reports/migration-2026-09-10.md scripts/commit_migration_wording.sh
git commit -q -F - <<'MSG'
Say only what was measured about the column-wise setting

The migration record's first pointer to its Phase 4 section said that
three of December's rungs grow different trees at one thread "column-wise
or not". Every Phase 4 fit ran with the column-wise setting, so the other
half of that phrase was never measured. The sentence now says which
setting the fits used and nothing about the one they did not.
MSG

say "result"
git --no-pager log --oneline -3
echo
git -c core.fileMode=false status --short
echo
echo "Not pushed."
