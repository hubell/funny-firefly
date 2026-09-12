# SPDX-License-Identifier: GPL-3.0-only
#
# A draft reference environment for reproducing this repository's model reports.
#
# IT DOES NOT BUILD YET, and deliberately so: two pins are unfilled, and both
# have to be filled on a machine with registry and index access, which the
# reference machine is not.
#
#   1. the base image digest on the FROM line, from
#        podman manifest inspect docker.io/library/debian:trixie-slim
#   2. the package hashes in requirements-freeze.txt, from the command that
#      file carries; until they exist, the --require-hashes install below
#      refuses to install anything.
#
# Both failures are loud and neither can be papered over by dropping a flag:
# a base image by tag is not a version, and `==` pins without hashes pin no
# bits at all, because a wheel can be rebuilt and re-uploaded under the same
# version. A build that refuses to run is better than an image that quietly
# is not the one the reports were made with.
#
# The ladder repeats to the bit on one machine (reports/determinism.md, and the
# criterion in reports/migration-2026-09-10.md). That is a property of the
# booster's configuration, not of the machine, and it says nothing about
# another machine reproducing the same bits. This image is the attempt to make
# the second thing checkable: one interpreter build, one set of wheels, one
# BLAS kernel and one thread count, all named rather than inherited from
# whatever the host happens to have.
#
# What is pinned, and why each one could move a figure:
#
#   * the base image, by digest, because the OpenMP runtime LightGBM links
#     against comes from it -- the wheel vendors none -- and `deterministic`
#     is a statement about how that runtime combines its threads' sums;
#   * CPython 3.13.5 from python-build-standalone, as uv installs it, rather
#     than a distribution interpreter;
#   * every package by version and hash (requirements-freeze.txt), from
#     reports/linux-freeze.txt, which records what the reports were made with;
#   * OPENBLAS_CORETYPE, because OpenBLAS ships DYNAMIC_ARCH and picks a kernel
#     from the CPU at run time -- SkylakeX on the reference machine, a Zen 4
#     part with AVX-512 -- and the least squares the geometric term fits is the
#     one place a different kernel showed up in the frame (2.9e-08 s on one
#     rung, between one BLAS thread and eight);
#   * OPENBLAS_NUM_THREADS and OMP_NUM_THREADS at the reference machine's
#     eight, which is what those were there by default;
#   * the booster's own thread count, which is NOT here: it is n_jobs in
#     taxiout.models.BOOSTER_PARAMS, where the reports can see it.
#
# THIS IMAGE REQUIRES AN AVX-512 HOST. The coretype pin names a kernel that a
# CPU without AVX-512 cannot run, so the image will fail there rather than
# quietly compute with another kernel. That is the intended behaviour: the
# figures are reproducible against this kernel and nothing here establishes
# what another one would give.
#
# CPU only. No CUDA, no GPU runtime: the reports are made on the CPU, and a GPU
# fit is a different model.
#
# Build (needs a network, unlike anything else in this repository):
#
#   podman build -f Containerfile -t prc-taxiout:reference .
#
# Run. The pipeline writes into the repository -- reports/ and
# data/processed/residuals/ -- so the repository is mounted writable and only
# the two inputs that must never be written are read-only:
#
#   podman run --rm \
#     -v "$PWD:/repo:rw" \
#     -v "$PWD/data/challenge:/repo/data/challenge:ro" \
#     -v "$PWD/data/raw:/repo/data/raw:ro" \
#     prc-taxiout:reference make baseline
#
# What has been measured, and the list of what it does not establish -- other
# CPUs, other glibc and libgomp versions, other container runtimes, arm64, a
# fresh install from the index rather than the host's wheels -- is the Phase 3
# section of reports/migration-2026-09-10.md.

# FILL THE DIGEST (see 1 above). A tag is not a version.
FROM docker.io/library/debian:trixie-slim@sha256:FILL_ME

# uv installs the interpreter and the packages; it is pinned for the same
# reason everything else is. The installer is fetched at build time, so this
# line is the one place the image trusts the network.
#
# libgomp1 is the one system library the wheels need and none of them vendors:
# lightgbm's lib_lightgbm.so links libgomp.so.1 from the system, and a slim
# base does not carry it. Everything else it needs -- libstdc++, libgcc_s --
# arrives with apt's own dependencies.
ENV UV_VERSION=0.12.3
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl make git libgomp1; \
    rm -rf /var/lib/apt/lists/*; \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/uv.sh; \
    sh /tmp/uv.sh; \
    rm /tmp/uv.sh; \
    mv /root/.local/bin/uv /usr/local/bin/uv

# The interpreter build the reports were generated with.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13.5

# The packages, by version and by hash. No resolver runs here: --no-deps means
# the file is the whole list, and --require-hashes means a wheel that is not
# the one recorded is refused rather than installed.
COPY requirements-freeze.txt /tmp/requirements-freeze.txt
RUN uv venv --python 3.13.5 /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-deps --require-hashes \
      -r /tmp/requirements-freeze.txt \
 && /opt/venv/bin/python -c "import numpy, pandas, pyarrow, scipy, sklearn, lightgbm, catboost, networkx, holidays; print('imports ok')"

# What this image fixes. Four of these were simply unset on the reference
# machine -- OPENBLAS_CORETYPE, PYTHONHASHSEED, TZ and LC_ALL -- and the values
# here are the ones it resolved to or the ones the pipeline's own behaviour
# makes immaterial; naming them is the point of the image, and the container
# check in the migration record is what says they cost nothing.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/repo/src \
    OPENBLAS_CORETYPE=SkylakeX \
    OPENBLAS_NUM_THREADS=8 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    PYTHONHASHSEED=0 \
    TZ=UTC \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /repo
ENTRYPOINT []
CMD ["make", "baseline"]
