#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Phase 4: what the booster's thread count changes, and the two container pins.
# Four commits, additive, nothing rewritten, nothing pushed.
#
#   1. the feature frame built in one construction, changing no digest
#   2. the thread-count instrument, its generated report, and the comment on
#      n_jobs cut to what was measured
#   3. the migration record's Phase 4 section and the narrowing of its
#      earlier sections, the determinism report regenerated with each fit
#      named by its recorded parameters, and
#      scripts/check_licence_hygiene.py reading check-ignore NUL-delimited
#   4. the container's two pins
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# checks that each generated report is newer than the code that generates it,
# and verifies every commit's own tree in a worktree before finishing.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in src/taxiout/features.py src/taxiout/models.py scripts/fast_lane.py \
         scripts/check_determinism.py scripts/commit_phase4.sh .gitignore \
         reports/threads-2026-09-12.md reports/determinism.md \
         reports/migration-2026-09-10.md scripts/check_licence_hygiene.py \
         Containerfile requirements-freeze.txt; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the container's two pins are filled"
# The inverse of the guard commit_container.sh carried: that one refused to
# commit unless the placeholders were still there, because it was committing a
# draft. This one refuses unless they are gone, because it is committing the
# pins -- and still refuses if the base image is pinned by tag alone.
if grep -q "FILL_ME" Containerfile requirements-freeze.txt; then
    echo "a pin is still a placeholder -- fill it or use the other script"
    exit 1
fi
if ! grep -qE '^FROM [^ ]+@sha256:[0-9a-f]{64}$' Containerfile; then
    echo "the base image is not pinned by digest on the FROM line"
    exit 1
fi
if ! grep -q -- "--hash=sha256:" requirements-freeze.txt; then
    echo "requirements-freeze.txt carries no hashes"
    exit 1
fi
# Every version must carry at least one hash: a single unhashed pin is enough
# for --require-hashes to refuse the whole install, and enough for this commit
# to be claiming more than it has.
if ! python3 - <<'PY'
import re, sys
from pathlib import Path
text = Path("requirements-freeze.txt").read_text(encoding="utf-8")
blocks = re.split(r"(?m)^(?=[A-Za-z0-9_.\-]+==)", text)[1:]
bad = [b.split("==")[0] for b in blocks if "--hash=" not in b]
print(f"  {len(blocks)} versions, {text.count('--hash=')} hashes")
if bad:
    print("  no hash for: " + ", ".join(bad))
sys.exit(1 if bad or not blocks else 0)
PY
then
    exit 1
fi
echo "ok"

say "the image still has not been built, and the record says so"
# The pins are a specification, not evidence. If this ever stops being true the
# sentence has to change rather than quietly stay.
if ! grep -q "HAS STILL NEVER BEEN BUILT" Containerfile; then
    echo "Containerfile no longer says the image is unbuilt -- build it and say"
    echo "so, or restore the sentence."
    exit 1
fi
echo "ok"

say "each generated report is newer than the code that generates it"
for pair in "scripts/fast_lane.py reports/threads-2026-09-12.md" \
            "scripts/check_determinism.py reports/determinism.md"; do
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

BEFORE="$(git rev-parse HEAD)"

say "1/4 the feature frame, built once"
git add src/taxiout/features.py
git commit -q -F - <<'MSG'
Build the feature frame in one construction, changing no digest

build_features assigned each block's columns one at a time, which
reallocates a two-million-row frame on every assignment. pandas warns
about it, and at this size it was the most expensive thing the function
did. The blocks are collected into one mapping and the frame is built
once from the columns instead.

It changes no result, which is the point and was checked rather than
assumed: all 175 digests the determinism harness records -- 122 feature
columns, the encoding and its prior, both geometric fits, the masks and
the designs -- are identical to the build before it. It also bought no
time: 53 s either way. What it buys is one construction instead of
thirty and the warning gone.

A first attempt concatenated two frames and peaked at 10.13 GiB
against 6.61. It was replaced by this one, which is back at 6.66,
because this machine has no swap and a build that peaks half again as
high would end a run rather than slow it.
MSG

say "2/4 the thread-count instrument and its report"
git add scripts/fast_lane.py src/taxiout/models.py \
        reports/threads-2026-09-12.md .gitignore scripts/commit_phase4.sh
git commit -q -F - <<'MSG'
Measure what the booster's thread count changes, and what it does not explain

scripts/fast_lane.py was written as a fast lane, and its own first run
refuted both halves of that premise. The parallel mode is worth 1.08x
rather than the ~3x predicted: a one-thread fit costs about four times
an eight-thread one rather than eight, so the booster already takes
most of what the threads are worth, and running eight fits beside each
other costs another 1.2 to 1.4x on top. And the fits at one thread are
not the ladder's on every rung. It is kept as the instrument for the
question it answered, redescribed as one, and given --report.

What it measured, against the ladder's stored residuals. Fitting
December's 25 rungs afresh at eight threads reproduces every one of
them bit for bit. At one thread three grow different trees --
weather-nolag by 158.3 s, throughput by 40.3 s and corridor-nolag by
4.97 s on a flight -- nine are bit-identical and thirteen differ only
in the last bits of a sum taken in another order. At four threads only
corridor-nolag grows different trees, with seven in the last bits and
seventeen bit-identical, and July's 25 rungs at one thread grow none,
with four in the last bits and twenty-one bit-identical. No printed
RMSE moves at the one decimal the reports carry, so where it happens
it is invisible in a report and real in the model. The reproduction
claim is therefore a claim at n_jobs 8, which BOOSTER_PARAMS pins.

Two things this settles. It is the parameter and not the cores: two of
the three rungs that move were refitted with n_jobs left at eight and
the process confined first to four CPUs and then to two, and gave the
ladder's own digests every time. That was measured by narrowing one
machine rather than by running on a second, so it settles that the core
count is not part of the result and leaves what another host would give
unmeasured. And the mechanism the n_jobs comment proposed is wrong.
With LightGBM's dataset construction logged, all six rungs measured
build a multi-value bin group -- including the three that do not move
-- and the sparse rate does not separate them either: throughput and
corridor both report 0.874267 and only the first moves. Why those three
move is now recorded as unexplained instead of being given a mechanism
it does not have.

reports/threads-2026-09-12.md is generated by --report from the run
records, so every figure in it is computed rather than typed. The
fingerprint check that says this script writes no report and no stored
residual stays; that one file is exempted by name, and only in --report
mode. The six logged fits carry the ladder's own digests, which is what
says raising the verbosity left the model alone.

catboost_info/ is ignored rather than redirected: CatBoost writes its
training log into the working directory wherever lightgbm is not
importable, and no test fits one, so there is no train_dir to point
somewhere else.
MSG

say "3/4 the record"
git add reports/migration-2026-09-10.md reports/determinism.md \
        scripts/check_determinism.py scripts/check_licence_hygiene.py
git commit -q -F - <<'MSG'
Narrow the reproduction claim to the thread count it was measured at

The migration record gains its Phase 4 section: what was run, what it
establishes, and what it does not. The claim that the reference
configuration reproduces is narrowed from the configuration to the
configuration at n_jobs 8, which is what was measured and what every
reported figure was fitted under. The Phase 3 list of what the
container does not establish keeps a core-count line but no longer the
same one: the row blocks follow n_jobs and not the CPUs on offer, so
what stays open is another host, not another core count. The record's
account of 11 September keeps every figure as it was measured then,
and its two sentences on what determinism.md contains are dated,
because the regenerated report counts eight builds and 59 fits in 20
processes where the first said five builds and 51 fits in 17, and five
builds at the default BLAS thread count where the second said three.

LightGBM's own documentation, which threads-2026-09-12.md quotes, is
set beside the measurement that contradicts it, so a reader meets
both: deterministic "should ensure the stable results when using the
same data and the same parameters (and different num_threads)". On the
LightGBM these runs used, 4.7.0 in the two that record a version, three
of December's 25 rungs grow different trees at one thread, fitted eight
at once. At four threads one does, corridor-nolag, and on July at one
thread none of the 25 does, though in each of the three runs others
differ in the last bits: thirteen, seven and four rungs.

determinism.md is regenerated so that it carries the two scaling runs
and the container run, eight fits in three processes, and three further
builds of the frame, all of which were in its records but not in the
committed report. Fits are now named by the parameters each one
recorded rather than by the configuration it was launched under, so
the three processes launched as current after deterministic and
force_col_wise entered BOOSTER_PARAMS are shown under
deterministic-col, and the report names them. The container's fits
are given their own row rather than pooled with the host's: they ran
with the thread counts, the BLAS kernel and the hash seed pinned, which
is another condition even where, as here, it gives the same bits. Every
table entry the committed report carried is unchanged; the component
table gains a column for each new build, and the totals, the fits to a
process and the low end of the load average move with the new records.

Its setup line now names the architecture and core count instead of the
CPU model, because what a reader needs in order to know whether these
numbers could be theirs is the instruction set and the wheels, and the
model name only names one machine. The instruction set is recorded from
now on; no record written before this change carries it, so the
regenerated report names none.

scripts/check_licence_hygiene.py now passes paths to git check-ignore
and reads them back NUL-delimited. It used to split the answer on any
whitespace, so a path containing a space came back in fragments that
matched nothing, and the line format C-quotes a path holding a
non-ASCII byte; either way an ignored file with such a name was left
looking un-ignored, and was scanned and counted. On this tree that was
one ignored file whose name holds a space: the gate counted 137 files
before the change and 136 after, and every check gave the same verdict
both ways, so here it changes no verdict.
MSG

say "4/4 the container's pins"
git add Containerfile requirements-freeze.txt
git commit -q -F - <<'MSG'
Fill the container's two pins, and say what each one is worth

The base image is pinned to the amd64 image behind debian:trixie-slim
rather than to the multi-arch index: everything else in the file
already commits to x86-64 with AVX-512, and an index digest would
pretend otherwise. The thirty packages are pinned by hash as well as by
version, 592 hashes, generated by the command the file already
prescribed and run where there is index access. The versions are
unchanged; what is new is that they now pin bits.

The image has still never been built, and the file says so in its own
first sentence. This machine has neither registry nor index access, so
what is committed is a complete specification that nothing here has
executed.

The two pins do not have the same standing, and the file separates
them. The hashes were generated from the list this repository already
carried, and every version in it is unchanged. The base digest is the
published amd64 image as of 2026-08-25, read from a machine with
registry access, and it is NOT verified to be the image the two
container runs of 2026-09-12 used. The Phase 3 record says that base
was pinned by digest, but no such digest survives anywhere in this
repository, the image itself is no longer in this machine's store, and
there is no network here to go and look; whether the pin was never
recorded or the record of it is lost cannot be told from here, and the
file says so rather than choosing. What those runs establish is what
the container changed about a run -- nothing measurable -- and not that
this base is the one they ran on. Anyone who can reach a registry
should check the pair before trusting it as one.
MSG

say "each commit's own tree passes its own tests"
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

say "result"
git --no-pager log --oneline -5
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -4"
