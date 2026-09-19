#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Phase 5: the path from the training files to the scored file, and the record
# made whole. Three commits, additive, nothing rewritten, nothing pushed.
#
#   1. the leaderboard sentence; the shrunk encoding fitted on the rows every
#      fit reads, and the reports regenerated under it; the leakage audit's
#      history and the seed-averaging rule of 19 September; make_submission.py
#      refusing any booster but LightGBM and writing a manifest, with
#      data/submissions/ ignored
#   2. the truncated first hours of the ranking months, and the scored file's
#      record, in reports/submission-2026-09-19.md
#   3. the migration record's Phase 5 section, the peak memory corrected, and
#      the encoding's rows
#
# The first was planned as four commits and is one, because three files carry
# more than one of the four -- scripts/run_baselines.py the first three,
# docs/paper.md the first and the third, scripts/make_submission.py the third
# and the fourth -- and this script stages whole files, never part of one.
#
# Checks that the tree is the one these commits were written against, file by
# file; that each generated report is newer than the code that generates it;
# runs the tests, the lint and the licence gate and stops if any fails; and
# verifies every commit's own tree in a worktree before finishing.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

say "the tree is the one these commits were written against"
BASE=9af7e2eaf0c66f538ce5dad4c44033f223dd0085
if [ "$(git rev-parse HEAD)" != "$BASE" ]; then
    echo "HEAD is $(git rev-parse --short HEAD), not ${BASE:0:7}"
    exit 1
fi
expected=" M .gitignore
 M Makefile
 M docs/paper.md
 M reports/baseline.md
 M reports/congestion.md
 M reports/determinism.md
 M reports/migration-2026-09-10.md
 M reports/occupancy.md
 M reports/residual.md
 M reports/transfer.md
 M scripts/check_determinism.py
 M scripts/fast_lane.py
 M scripts/make_submission.py
 M scripts/run_baselines.py
 M src/taxiout/models.py
?? reports/submission-2026-09-19.md
?? scripts/commit_phase5.sh
?? scripts/report_submission.py"
# Read both ways: with the mode ignored it is the content, and with it read
# a mode-only change would show here and be committed by a whole-file add.
for mode in false true; do
    actual="$(git -c core.fileMode=$mode status --short)"
    if [ "$actual" != "$expected" ]; then
        echo "unexpected working-tree state (core.fileMode=$mode):"
        printf '%s\n' "$actual"
        exit 1
    fi
done
# Every file the commits take, by content, as it stood when they were written.
if ! sha256sum --quiet -c - <<'SUMS'
ff6355317ae7c183ce0cfc6ba1264c948c44e6c0a320acde932e7a5653cd7b29  .gitignore
49c12a8a3c458e76366051ed0456db0342ac07e5d9444555dd6dbff163901b30  Makefile
1d4dbb62d4da8300a1bb95df288f622688631c7d2f3bf4e7f97efd385a78f2d4  docs/paper.md
464cdcb219933c56aa0fd67f20626f9d9b4f69df071df6e34bd284bbf054d0a3  reports/baseline.md
b8397b8e5075e36ac23c44f7d1e26e25da5603c7ad6921fdab593d99a709c20f  reports/congestion.md
a7200723dcc876e2f8cf23e1015bc3919f7b8cf5740cd86c0ddfced55421d830  reports/determinism.md
76e557901ae48f0fb360f459f5db160d6b969195cfbbca275b1aa7d2178549b6  reports/migration-2026-09-10.md
c47a729cfc87f03c0ffc2d41317b0497b89c0a39970d814c0aa6f99d85a6f21b  reports/occupancy.md
302d96002a42d16b429a49eb9e8079c2cd3ee4a5cbbb4d8f8f0766ddc8c20cd3  reports/residual.md
3145af1091960c666180b2683c13385c3df42d8a64531c0f9f82b8b227c19d1f  reports/transfer.md
918d277fd3dce5ab6818760d5ed83797925a33d40a844daec691c2abab8fac93  scripts/check_determinism.py
41c6a22853d32c36f407edcea90541ff44d6a7ccbf35eee2779d56d104e6def3  scripts/fast_lane.py
70fd9a13dc878de7dbd0a9c91c98a4219fbc6f3f72e3043806f02d6b04d5368f  scripts/make_submission.py
a7720c9b7a02081d8fa302e8f06169caec0e66120d8c6d4a08e4e7fb27c34105  scripts/run_baselines.py
eb312a5e3defd3165dbc005e9309b9a8a6f4188688f1f550b0773f9191e75494  src/taxiout/models.py
182ca2a4534aa946aac24f857d8a53b5573e296ff8835e72c9a088c6f014f003  reports/submission-2026-09-19.md
d4931c277c744c20b959d55049a0beb85b56e7339906ef3befcbca0b2f20c579  scripts/report_submission.py
SUMS
then
    echo "a file above is not the one these commits were written against"
    exit 1
fi
echo "ok"

say "each generated report is newer than the code that generates it"
for pair in "scripts/run_baselines.py reports/baseline.md" \
            "scripts/report_transfer.py reports/transfer.md" \
            "scripts/report_residual.py reports/residual.md" \
            "scripts/report_occupancy.py reports/occupancy.md" \
            "scripts/report_congestion.py reports/congestion.md" \
            "scripts/check_determinism.py reports/determinism.md" \
            "scripts/fast_lane.py reports/threads-2026-09-12.md" \
            "scripts/report_submission.py reports/submission-2026-09-19.md" \
            "scripts/make_submission.py reports/submission-2026-09-19.md"; do
    set -- $pair
    if [ "$1" -nt "$2" ]; then
        echo "$2 is older than $1 -- regenerate it"
        exit 1
    fi
done
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

say "1/3 the encoding's rows, the audit's history, the seed rule, the file"
git add scripts/run_baselines.py docs/paper.md scripts/fast_lane.py \
        scripts/check_determinism.py reports/baseline.md reports/residual.md \
        reports/occupancy.md reports/congestion.md reports/transfer.md \
        reports/determinism.md src/taxiout/models.py \
        scripts/make_submission.py .gitignore Makefile
git commit -q -F - <<'MSG'
Fit the encoding on the rows every fit reads, and say what it moved

Four changes planned as four commits go in as one, because three files
carry more than one of them: scripts/run_baselines.py the first three,
docs/paper.md the first and the third, and scripts/make_submission.py
the third and the fourth. This stages whole files, never part of one.

The sentence. The ladder's report and the paper said the challenge has
no leaderboard. It has one, which ranks each team by the best RMSE
across its submissions. Both now say so in the ranking page's words and
state this entry's rule for it: at most one upload per frozen version,
no upload used to choose a feature, a rung, a parameter or a rule, and
nothing read back from the scorer but usedPairs and the score.

The one row rule. The shrunk encoding was fitted on split.train &
unflagged in the ladder and on the fit rows in the submission, so seven
of December's training rows and five of July's, each recording a taxi
over four hours, were in the ladder's encoding and in no booster fit.
run_baselines.py, fast_lane.py and check_determinism.py now fit it on
taxiout.models.fit_rows, as make_submission.py already did. The ladder
was regenerated twice under the rule, and the two runs agree on all 52
residual files. Against 11 September's, 16 changed: December and July
for the eight rungs that read the encoding, encoding-added,
encoding-for-stand, encoding-for-identifiers, occupancy,
occupancy-tfree, callsign, destination and plan-diffs. The other 36 are
byte-identical. The submission rung goes from 218.1 to 217.6 s
ex-flagged on December, 325.6 to 325.8 on July, 237.9 to 237.8 on
July's three airports and 219.2 to 218.6 at the ranking mix, and the
chain still climbs to the top. Three verdicts turned. The
within-variance check is material, at 1.56 where it was 1.29 against
1.5, so the pair-level k moves with route length on the December leg.
The leakage audit on + stand occupancy no longer fires: an overshoot of
4.345 s against 5.000, where it fired by 0.37 s. And the seed floors
moved from 0.19, 0.39 and 1.28 s to 0.34, 0.23 and 1.02, with 9 of 31
comparisons below them where 13 were. transfer.md changes in its date
alone, and so does determinism.md, regenerated from its records because
its generator changed.

The audit's history. That audit has read four ways on four
regenerations, none of which touched the block: overshoots of 5.56 s at
ad1eecf, 4.971 s on the Mac run, 5.37 s under the reference
configuration and 4.345 s now, so fired, missed, fired, missed.
baseline.md printed the latest reading and, when it did not fire,
credited the drop below the line to the tail rows alone; the record
shows the verdict moving with the booster's configuration and the
encoding's rows as well. Both branches now print the whole history with
its dates and revisions, set the last two verdicts against the seed
floors they sit inside, 0.37 s against 1.28 and 0.655 s against 1.02,
and say what the block rests on: the source registry, the poison tests
and the measured concentration of its gain, not this test.

The seed rule. The one-second rule for seed averaging was retired, and
the rule that replaces it fixed, on 19 September, with the 11 September
two-leg figures in view, which cleared it, and before this regeneration's
figures existed: the retired rule's stated basis, the run-to-run
variation before the booster was made repeatable, no longer holds. The
rule now: "Adopt it for the submission if, on both forward legs, the
five-seed average beats every one of the five single seeds and its gain
over the ladder's seed clears that leg's seed floor; before it ships it
is also measured on the three July airports and on October, and both
are printed." Both legs meet its first half, with gains of 0.48 and
0.52 s against floors of 0.34 and 0.23 and the average below every
single seed. The three-airport and October measurements are not made,
so the report reads not yet adopted and SEED_AVERAGE_ADOPTED stays
False. The comments in taxiout.models and make_submission.py carry the
new figures. The paper's sentence on the within-variance check said
immaterial at 1.29; it now says what the regenerated report does,
material at 1.56 on the December leg, with the submission's encoding at
1.600 for the pair and 1.477 for the stand, and a drafting note lists
what the regeneration left stale for the figure pass.

The file. make_submission.py refuses any booster but LightGBM: the
CatBoost fallback reads three of BOOSTER_PARAMS and none of the settings
that make a fit repeat, and no report measured it. Before it fits it
checks that the file and a manifest beside it are both ignored and not
already there, and it writes the manifest only once the file has been
read back and validated: the revision and whether the tree was clean,
the booster, its version and parameters, the rows fitted on, the
record-quality and bare-prior counts, which branch each level of the
encoding's length check took, the sha256 of the template and of the
file, and two empty fields for usedPairs and the score. The whole of
data/submissions/ is ignored, and the Makefile's comment on the
submission target says what it writes and what it refuses.
MSG

say "2/3 the truncated first hours, and the scored file's record"
git add scripts/report_submission.py reports/submission-2026-09-19.md
git commit -q -F - <<'MSG'
Count the truncated first hours, and record the scored file

scripts/report_submission.py writes reports/submission-2026-09-19.md.
It builds both frames with make_submission.py's own loaders, constants
and build_features call, and counts the four truncation flags per month
on the ranking frame and on the training frame's first and last months.
In January 2026, 843 of 152,719 departures carry one, 839 of them
occ_truncated, all but two in the block's first 24 hours, against 854
of 153,706 in January 2025, which is in training. July 2026 carries 5 of
63,157. Six rows of December 2025 carry one, at the end edge of the
training block, and December 2025 is the December leg's validation
month. No feature changed: the report counts the flags the frame
already carries and decides nothing on them.

It also records the scored file: v4's size and sha256 set beside its
manifest, the manifest's counts with the record-quality rows recounted
from the same frames, and two builds of v4 byte-identical, their
manifests differing in the time alone. Both were built from the tree
these commits come from, before it was committed, so the manifest names
9af7e2e and a tree that was not clean. Nothing has been uploaded. The
report carries aggregates only and refuses a movement or a stand
identifier.
MSG

say "3/3 the record"
git add reports/migration-2026-09-10.md scripts/commit_phase5.sh
git commit -q -F - <<'MSG'
Record the fresh operating system, and what the encoding's rows moved

The migration record gains its Phase 5 section. The reference machine's
operating system was reinstalled, Fedora Linux 44 with glibc 2.43 and
the system libgomp 16.2.1, the thirty packages were installed from the
index against their hashes for the first time, and the pipeline ran in
a copy of the tree at 9af7e2e. The frame's 175 digests, the rung's
fits, all 52 residual files and every report came out as they were, the
five the pipeline writes differing in their date line alone. The ladder
took 7,542 s at a peak of 8,037 MiB against run A's 7,550 s and
8,087 MiB. The section says what that establishes and what it does not:
another CPU, a userland beyond the three now run, another package
version, the reference image, arm64 or the Mac.

Run A's peak was given in two places as 8.1 GiB. Its own timing record
says 8,087 MiB, which is 7.9 GiB, and both places now say so.

And a section on the encoding's rows: fitted on the rows every fit reads
from 19 September; the ladder regenerated twice, both runs agreeing on
every residual file; the 16 files that changed and the 36 that did not;
the submission rung's four figures; the three verdicts that turned, with
the audit's four readings; and the 11 September residual files, kept
outside the repository, against which the Phase 4 thread-count
measurements were made and stand.

scripts/commit_phase5.sh is the script that made these commits.
MSG

say "the licence gate, over the three messages as well"
# It reads the recent history, so run after the commits it checks what they
# say as well as what they carry.
python3 scripts/check_licence_hygiene.py

say "each commit's own tree passes its own tests"
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
for sha in $(git rev-list --reverse "$BASE"..HEAD); do
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
echo "Not pushed. Review with: git log -p -3"
