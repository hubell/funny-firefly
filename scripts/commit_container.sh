#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# The reference container: the draft Containerfile, the frozen package list it
# installs from, and what two runs inside it measured against the host. One
# commit. Additive; nothing is rewritten. Not pushed.
#
# The image does NOT build yet, and the commit message says so: its base digest
# and its package hashes have to be filled on a machine with registry and index
# access. What is committed is the draft and the measurement, which is worth
# having on the record either way -- the runs used the repository's own
# interpreter and wheels, so they measure what the container changes about a
# run rather than what the image would install.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# and refuses to commit unless the container record says the two runs agreed.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in Containerfile requirements-freeze.txt reports/container-2026-09-12.md \
         reports/migration-2026-09-10.md scripts/commit_container.sh; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the two container runs agreed, and the record says so"
# The first summary line of the first comparison in the record: container run 1
# against container run 2, nothing differing and nothing missing.
if ! sed -n '1,8p' reports/container-2026-09-12.md | grep -qE \
        "^[0-9]+ files byte-identical, 0 differing only in their generation date, 0 differing otherwise, 0 missing from a run\.$"; then
    echo "reports/container-2026-09-12.md does not record two identical container runs"
    exit 1
fi
# The draft is a draft: both placeholders must still be visible, so that this
# commit cannot claim an image that was never built.
if ! grep -q "sha256:FILL_ME" Containerfile; then
    echo "Containerfile's base digest is filled -- then build the image and say so"
    echo "in the commit message instead of using this script."
    exit 1
fi
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

say "the container, and what it measured"
git add Containerfile requirements-freeze.txt reports/container-2026-09-12.md \
        reports/migration-2026-09-10.md scripts/commit_container.sh
git commit -q -F - <<'MSG'
Draft the reference container, and measure what it changes

Two full pipeline runs inside a container on the reference machine agree
with each other on all 69 files byte for byte, and with the host's run at
067c4af on all 52 residual files, with each of the five reports differing
in one line: the date it was generated on. Debian trixie, pinned by
digest, brings its own glibc (2.41 against the host's 2.44) and its own
libgomp, which is the runtime deterministic acts on, so this is the first
evidence that the configuration adopted in 067c4af survives a change of
userland. reports/container-2026-09-12.md is the comparison and the Phase
3 section of reports/migration-2026-09-10.md is what it does and does not
establish.

The Containerfile is a draft and does not build. Its base image digest
and the package hashes in requirements-freeze.txt have to be filled on a
machine with registry and index access, which the reference machine is
not, and until they are the --require-hashes install refuses to install
anything. Both failures are loud on purpose: a base image by tag is not a
version, and a version without a hash pins no bits, since a wheel can be
rebuilt and re-uploaded under the same version. The runs above therefore
used the repository's own interpreter and wheels, mounted read-only, and
measure what the container changes about a run rather than what the image
would install.

What is pinned and why is in the file's own header. The one that costs
something is OPENBLAS_CORETYPE: it names an AVX-512 kernel, so the image
requires such a host and fails on one without it rather than quietly
computing with another kernel. Every figure here was measured on one CPU,
one kernel, one libgomp and one container runtime, and the migration
record lists what that leaves untested -- other microarchitectures,
arm64 and the Mac the committed reports came from among them.
MSG

say "the commit's tree passes its own tests"
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
git --no-pager log --oneline -3
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -1"
