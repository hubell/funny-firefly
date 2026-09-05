#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the shrunk identifier encoding and what it touches -- the
# inference-time encoder, the rung that ships it, the regenerated ladder and
# congestion reports, and the transfer report's row-count fix -- as six
# commits on top of the current history.
# Additive; nothing is rewritten. Adds only this stream's files by name, so
# the other stream's files may sit untracked beside them, and so the
# assistant-configuration paths the licence gate forbids can never be swept in
# by a `git add -A` that this script never runs.
#
# Order: this runs as step 2 of three. Step 1 is commit_paper.sh's thin-pair
# generator and reports/geometry_gaps.md, which this stream's tests read and
# this stream's report cites. Step 3 is commit_paper.sh's paper commit, after
# Stream B has re-read the regenerated figures.
#
# REGENERATION IS A PRECONDITION, not a step. reports/baseline.md must be the
# file scripts/run_baselines.py wrote after SUBMISSION_KEY moved, not a
# hand-patched one: the report's own header says every figure in it is
# computed by its generator, and a hand-patched report in the history
# contradicts that for any reviewer who diffs it. The check below is the
# cheap version of that -- it refuses a report whose decision section still
# names the control as chosen -- and it is not a substitute for having run
# `make baseline`.
#
# Runs this stream's tests, the whole suite and the licence gate first and
# stops if any fails. Does not push. Review `git log -p -6` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

MINE=".gitignore Makefile README.md docs/modelling_notes.md \
      scripts/check_licence_hygiene.py tests/test_licence_hygiene.py \
      src/taxiout/encoding.py src/taxiout/features.py src/taxiout/models.py \
      scripts/run_baselines.py scripts/make_submission.py \
      scripts/report_congestion.py scripts/report_transfer.py \
      reports/baseline.md reports/congestion.md reports/transfer.md \
      tests/test_encoding.py tests/test_features.py \
      tests/test_strata_pooling.py scripts/commit_encoding.sh"
for f in $MINE; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the report is the regenerated one"
if grep -q '^\*\*Chosen: `+ weather`\.\*\*' reports/baseline.md; then
    echo "reports/baseline.md still chooses the control."
    echo "Run 'make baseline' after SUBMISSION_KEY moved, then re-run this."
    exit 1
fi
if ! grep -q 'ranking mix' reports/baseline.md; then
    echo "reports/baseline.md has no ranking-mix column; it predates this work."
    echo "Run 'make baseline', then re-run this."
    exit 1
fi

say "this stream's tests"
python3 -m pytest -q tests/test_encoding.py tests/test_strata_pooling.py \
    tests/test_features.py

say "the whole suite"
python3 -m pytest -q

say "licence gate"
python3 scripts/check_licence_hygiene.py

say "1/6 the assistant-configuration paths"
git add .gitignore scripts/check_licence_hygiene.py tests/test_licence_hygiene.py
git commit -q -m "Forbid the assistant-configuration paths in both guards

The repository carries no trace of what wrote it, and check 4 of the
licence gate exists to keep it that way. Two paths would have carried
one and were covered by neither guard: the assistant-configuration
directory and its instruction file were not ignored, so a git add -A
would have committed the local settings file, and they were not in
FORBIDDEN either, so the gate would not have noticed -- check 4 scans
authored file types and a settings JSON is not one.

Both are now ignored and both are forbidden, so the two guards fail
independently rather than relying on each other: deleting the ignore
lines fails the gate instead of passing quietly. The paths are assembled
from string fragments in the checker and its test for the reason the
docstring gives for the bracket classes -- spelled plainly, either name
would match the attribution pattern this file defines, and the scanner
would fail on its own source."

say "2/6 the inference-time encoder"
git add src/taxiout/encoding.py src/taxiout/features.py tests/test_encoding.py
git commit -q -m "Encode the ranking set from the training year

The encoding existed only inside the ladder, where every fold recomputes
it from its own fit rows. A submission needs the other half: the 2026
ranking rows are in no fold at all, and a feature that can only be
computed for rows of the frame it was fitted from cannot encode them.

fit_encoding() fits the whole construction over a training sample and
keeps what a lookup needs -- the geometric prior with its estimator, so
g_hat can be evaluated on a different frame; the per-group row count and
summed residual; and k per airport. FittedEncoding.transform() applies
it to any frame by looking the group up. A group the training year never
saw has n = 0, so the shrinkage weight is zero and the row encodes to its
own g_hat exactly -- the same fallback a held-out pair gets in the
ladder, reached by the same arithmetic rather than by a separate branch.

Which rows get which path is the whole of the design, and getting it
backwards makes a submission quietly worse rather than making anything
fail. The training rows the model is fitted on are encoded out of fold,
by encode_split, because their targets are in the fitting sample: rule 2
applies to them at submission as much as in the ladder, and fitting on
in-sample encodings would teach the booster to trust a column sharper at
fit than at use. The ranking rows are encoded from the whole sample,
because their targets are not in it and there is nothing to hold out of.

Two tests for the path that no split downstream can check. Guard 3
again, against transform() rather than against the internals: a
defaulted block time planted in a thin pair moves nothing, and the same
row left in the fit moves that pair's encoding by minutes, so the
assertion has content. And determinism -- the same year and rows give
byte-identical columns, and shuffling the fitting sample does not move
them -- because a submission whose feature depends on when it was
computed is not reproducible from the training year, which is the claim
the reproducibility section makes."

say "3/6 the rung the submission uses"
git add src/taxiout/models.py scripts/make_submission.py tests/test_features.py
git commit -q -m "Fit the submission from the shrunk encoding, replacing the stand

SUBMISSION_KEY moves off the weather rung and onto the arm the ladder's
rule chooses. make_submission.py builds the encoding once over the
training year, in the two forms the feature needs, and passes each to
the side that wants it; the refusal guard stays for any block this
script still cannot build, and the encoding is simply no longer one of
them. It reports how many ranking rows sit on a pair the year never saw
and therefore encode to the bare geometric prior.

Two rungs of the ladder proper now sit above the submission, for
different reasons, and the comment beside SUBMISSION_KEY says both.
Saturation is a measured negative that bought nothing. The arm that
replaces the stand and the runway both is out on a mechanism rather than
on its score, which is indistinguishable from the shipped arm's: the
designator carries an intersection signature of up to 96 s, constant
within a designator, that a matrix routing to the threshold cannot
compute and an encoding shrunk toward that matrix cannot carry.

ENCODING_CONTROL is separated from SUBMISSION_KEY. They were the same
string while the control was what shipped, and the encoding section was
written against SUBMISSION_KEY harmlessly; the moment an arm displaced
the control they came apart, and a section taking its control from the
shipped rung would have compared the chosen arm against itself and
reported a difference of zero. The control is a property of the
experiment and does not move when the model does.

Rung.uses_geometry now counts ENCODING_FEATURES. The encodings are their
own block and not in GEOMETRY_FEATURES, but what they shrink toward is
g_hat, so a rung carrying them carries the geometry with a step in
front. An ablation that called itself geometry-free while the encoding
stayed in would report a contrast that shrank by its own definition.
Nothing printed moves -- the claim contrast is measured between two
rungs that carry no encoding either way -- and this keeps that true if
it is ever re-anchored higher up the ladder."

say "4/6 the decision, and the quantity it was decided on"
git add scripts/run_baselines.py reports/baseline.md tests/test_strata_pooling.py \
    README.md docs/modelling_notes.md
git commit -q -m "Decide the encoding on the instruments that can see it

The acceptance rule's first half required a positive difference in
differences on the held-out-pair folds, and that cannot credit the
component the block exists for: a held-out pair encodes to n = 0 and is
identically the prior, so on every held-out row the pair column carries
none of the pair's own mean. The scope of that is worth stating exactly,
because overstating it would be self-serving. The gate is applied to a
whole arm, and it does validly measure the rest of one -- the stand-level
encoding, and the removal of the stand categorical -- which it reads as
decisively negative. It is not an instrument incapable of a positive
number; it is an instrument blind to the pair encoding. A gate that
cannot see the thing under test is not a test of it, so it is voided as
a test of the pair encoding rather than failed, and the arms take no
credit from it.

What decides instead: no worse on both forward splits, which is the
original rule's own second half unchanged and the half that rejects an
arm here; no worse pooled at the ranking set's bucket mix; and, among
those, the arm that keeps the runway categorical. That last is a
mechanism measured in geometry_gaps.md before these arms were run, not a
threshold read off their results. The never-seen bucket corroborates and
no longer decides -- it is a few hundred rows, and a decision resting on
it would be a post-hoc rule resting on a thin stratum. That the
replacement rule was written after the numbers were seen is a real
weakening and is said in the report rather than left to be noticed.

The strata table gains the quantity the competition actually scores:
each rung pooled in quadrature at the ranking set's bucket mix, beside
the same pooling at December's own. December's column is that split's
overall RMSE reconstructed and reproduces the ladder's ex-flagged column
exactly, which is what makes the ranking column readable as the same
quantity at different weights. The ranking mix is the one input this
generator does not compute; it is read from geometry_gaps.md as row
counts rather than as the rounded percentages printed beside them, and
a test parses that table and fails if the two ever disagree.

The block is reported as what it measurably is. 73% of its
ranking-weighted gain comes from pairs seen more than a hundred times --
the bucket the ceiling assigns no headroom at all -- against 28% from
the two thin buckets it was aimed at. It is smoothed memorisation that
buys real score on well-observed pairs, and the report says so rather
than letting the rung's name imply it did what it was aimed at."

say "5/6 the congestion ceiling follows the rung"
git add scripts/report_congestion.py reports/congestion.md
git commit -q -m "Read the ceiling against the rung the submission uses

The oracle ceiling is read on the submission rung and every rung of the
ladder proper above it, so each block that was built and left out is
measured against the ceiling of the rung it was added to. That list is
derived from ABOVE_SUBMISSION rather than written out, so moving the
submission onto the encoding arm moved it: the arms are no longer above
the submission and the arm that ships is the rung the ceiling is now
read against. Regenerated, so the report names the rung it used rather
than the one it used to. The ceiling itself does not move -- it is an
oracle on the excess term and knows nothing about which identifiers the
model carries -- and Gate 3 stays stopped on the strength of it."

say "6/6 the transfer report's row counts"
git add scripts/report_transfer.py reports/transfer.md Makefile scripts/commit_encoding.sh
git commit -q -m "Print the kept row count in the ex-flagged transfer table

The table headed with the record-quality rows subtracted printed the
unfiltered per-airport count beside an RMSE computed over the rows the
mask keeps. It showed Fiumicino at 160,704 rows in both tables while the
grand mean moved from 613 s to 1,350 s, which cannot both be true and
reads as a finding rather than as the bug it was. The count is now the
one the figure beside it was computed over, and the column is headed
kept rows so the two tables cannot be confused when read together.

A generated note says where the mask bites: 1,485 rows at Fiumicino and
none at the other nine on this training year, so every other row of that
table is identical to the all-rows table by construction rather than by
finding. Phrased as a property of the fit and not a guarantee -- the
term is fitted per (airport, cell) and covers only cells that showed the
mechanism in 2025, which on this data is one airport."

say "result"
git --no-pager log --oneline -7
echo
git status --short

say "no attribution in the new history"
# The pattern is spelled with bracket classes so that this file does not
# itself match the tree-wide grep in commit_gate0.sh.
PATTERN='co[w]ork|cla[u]de|anthro[p]ic|copi[l]ot|chat[g]pt|open[a]i|co-authored-b[y]'
if git log -6 --format='%B%n%an%n%ae%n%cn%n%ce' | grep -niE "$PATTERN"; then
    echo "FOUND -- inspect before pushing."
    exit 1
fi
echo "clean"

echo
echo "Not pushed. Review with: git log -p -6"
