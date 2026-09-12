#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Find where the ladder stops repeating: the feature frame, or the booster.

Two consecutive ``make baseline`` runs on one machine agreed on none of the
fifty-two residual files byte for byte, and on twenty-seven of them they
disagreed by whole trees. This separates the two places that can come from
without refitting the ladder: one split, one rung, fitted repeatedly.

``build`` loads the data and builds the feature frame exactly as
``scripts/run_baselines.py`` does -- it imports that script's own loaders --
and records a SHA-256 digest of every column, of the record-quality mask, of
the rows the rung is fitted on, of the shrunk encoding and of the two
geometric fits. The last two are where ``np.linalg.lstsq`` runs. Two builds in
separate processes -- and so under separate string-hash seeds, unless
``PYTHONHASHSEED`` is set -- that agree column for column put the
nondeterminism downstream of the frame. A third under
``OPENBLAS_NUM_THREADS=1`` says whether the linear algebra's thread count
reaches any column, and ``--keep-arrays`` keeps what the linear algebra
produced so that the report can say by how much.

``fit`` reads the cached inputs of the rung and fits it ``--repeats`` times in
one process under one named booster configuration, keeping every prediction.
``report`` compares them bitwise and writes ``reports/determinism.md``.

The rung is December's ``identifiers``: forty seconds a fit, and it moved by
whole trees between the two runs. Its fit here is the ladder's fit -- the rows
``taxiout.models.fit_rows`` picks, the design ``Booster`` builds, the
parameters in ``BOOSTER_PARAMS`` -- except that ``n_jobs`` is set per run,
because the thread count is one of the things under test. ``report`` checks
that claim against the ladder's own stored residuals, at the thread count they
were fitted with where it is given, rather than assuming it.

Everything under ``data/processed/determinism/`` is gitignored: the cached
inputs and the predictions are built from the challenge data. The report
carries aggregates only.

Usage
-----
    python scripts/check_determinism.py build --label a --cache
    python scripts/check_determinism.py build --label b
    OPENBLAS_NUM_THREADS=1 python scripts/check_determinism.py build --label blas1
    python scripts/check_determinism.py fit --config current --n-jobs 8
    python scripts/check_determinism.py fit --config deterministic-col --n-jobs 14
    python scripts/check_determinism.py report --reference "run 1=<dir>/residuals@14"
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_baselines as rb  # noqa: E402
from taxiout.encoding import encode_split  # noqa: E402
from taxiout.features import TARGET, airport_fallback, build_features  # noqa: E402
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER_BY_KEY,
    Booster,
    fit_rows as fitting_rows,
)
from taxiout.unimpeded import fit_geometric, fit_pru_reference, lower_tail  # noqa: E402
from taxiout.validation import december_split, month_of, rmse  # noqa: E402

WORK = REPO / "data" / "processed" / "determinism"
CACHE = WORK / "december_identifiers.pkl"
OUT = REPO / "reports" / "determinism.md"
RUNG = "identifiers"
SPLIT = "december"

# The configurations compared. The first three are the question the gate asks.
# The next three separate its two halves -- whether pinning the histogram
# strategy is enough on its own, and whether ``deterministic`` is -- and
# ``current`` at one thread is the control with no threads to disagree.
CONFIGS: dict[str, dict[str, object]] = {
    "current": {},
    "deterministic-col": {"deterministic": True, "force_col_wise": True},
    "deterministic-row": {"deterministic": True, "force_row_wise": True},
    "col": {"force_col_wise": True},
    "row": {"force_row_wise": True},
    "deterministic": {"deterministic": True},
}

# At or below this, two predictions come from the same trees and differ in the
# last few bits; above it, a split went the other way somewhere. The two
# comparisons already on file (reports/linux-residuals-delta.txt and
# reports/linux-run2-vs-run1.txt) put the largest near-identical file at 9e-9 s
# and the smallest whole-tree one at 0.6 s, so the line is not delicate.
SAME_TREES_S = 1e-6

# The report must carry this phrase, and ``make licence-check`` looks for it.
ATTRIBUTION = "OpenStreetMap contributors"


# --------------------------------------------------------------------------
# recording


def digest(values) -> str:
    """SHA-256 of a column's values, in order.

    Numeric, boolean and naive datetime columns hash their raw bytes, so a
    float that moved in its last bit, or a NaN with a different payload, counts
    as a difference. A categorical hashes its codes and its categories in order,
    because a reordered category list re-codes every row for the booster even
    where no value changed. Anything else goes through pandas' fixed-key row
    hash, which is enough to tell two builds of the same code apart but is not
    a bitwise comparison.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))
    h = hashlib.sha256(str(s.dtype).encode())
    if isinstance(s.dtype, pd.CategoricalDtype):
        h.update(np.ascontiguousarray(s.cat.codes.to_numpy()).tobytes())
        h.update(digest(pd.Series(s.cat.categories)).encode())
    elif isinstance(s.dtype, np.dtype) and s.dtype.kind in "biufcmM":
        h.update(np.ascontiguousarray(s.to_numpy()).tobytes())
    else:
        hashed = pd.util.hash_pandas_object(s, index=False).to_numpy()
        h.update(np.ascontiguousarray(hashed).tobytes())
    return h.hexdigest()


def frame_digests(frame: pd.DataFrame) -> dict[str, str]:
    return {str(c): digest(frame[c]) for c in frame.columns}


def environment() -> dict:
    """What the process ran under: versions, threads, the machine's load."""
    import lightgbm
    import pyarrow
    import scipy
    from threadpoolctl import threadpool_info

    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    keys = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_CORETYPE",
        "PYTHONHASHSEED",
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lightgbm.__version__,
        "machine": platform.machine(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "loadavg_at_start": list(os.getloadavg()),
        "env": {k: os.environ.get(k) for k in keys},
        "threadpools": [
            {
                k: p.get(k)
                for k in ("internal_api", "num_threads", "version", "architecture")
            }
            for p in threadpool_info()
        ],
    }


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_record(name: str, record: dict) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{name}.json"
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return path


def _array_name(key: str) -> str:
    """A digest key as a member name ``np.savez`` will take."""
    return key.replace(" ", "_").replace(":", "__")


# --------------------------------------------------------------------------
# build


def build(args) -> int:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    env = environment()
    seconds: dict[str, float] = {}

    # The order and the arguments of ``run_baselines.main``, lines 4484-4514.
    t0 = time.time()
    tails = rb.Tails()
    matrix = rb.load_matrix()
    dep = rb.load_departures(None)
    y = pd.to_numeric(dep[TARGET], errors="coerce")
    movements = rb.load_all_movements()
    weather = rb.load_weather()
    airports = rb.load_airport_table()
    edges = rb.load_edge_sets()
    seconds["load"] = time.time() - t0

    t0 = time.time()
    features = build_features(
        dep,
        matrix,
        fallback=airport_fallback(matrix),
        movements=movements,
        edges=edges,
        weather=weather,
        airports=airports,
    )
    del movements
    seconds["features"] = time.time() - t0

    # One split, and its record-quality mask and fit rows, as ``run`` and
    # ``fit_and_predict`` compute them (lines 403-416 and 298).
    split = december_split(month_of(dep["MVT_TIME_UTC_mvt"]))
    tail_model = tails.fit(dep.loc[split.train])
    if tail_model is not None:
        tail = tail_model.predict(dep)
        flagged = tail.notna()
    else:
        tail = pd.Series(np.nan, index=dep.index)
        flagged = pd.Series(False, index=dep.index)
    unflagged = ~flagged
    rows = fitting_rows(split.train, ~unflagged, y)

    # The two places ``np.linalg.lstsq`` runs: the shrunk encoding (whose prior
    # is the geometric fit) and the geometric rungs themselves, each with the
    # rung's own options, as ``fit_and_predict`` passes them.
    t0 = time.time()
    encoding = encode_split(features, y, split.train & unflagged)
    seconds["encoding"] = time.time() - t0
    t0 = time.time()
    geometric: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {}
    reference = fit_pru_reference(features, y, mask=rows)
    tail_rows = lower_tail(features, y, reference, mask=rows)
    for key in ("geometric", "geometric-class"):
        geo = fit_geometric(
            features, reference, tail_rows, **dict(LADDER_BY_KEY[key].options)
        )
        values = geo.predict(features)
        geometric[key] = digest(values)
        arrays[_array_name(key)] = np.asarray(values)
    seconds["geometric"] = time.time() - t0
    arrays[_array_name("encoding prior")] = encoding.prior.values.to_numpy()
    for c in encoding.columns:
        arrays[_array_name(f"encoding:{c}")] = encoding.columns[c].to_numpy()

    columns = list(LADDER_BY_KEY[RUNG].features)
    record = {
        "kind": "build",
        "label": args.label,
        "started": started,
        "argv": sys.argv[1:],
        "environment": env,
        "seconds": seconds,
        "peak_rss_mib": peak_rss_mib(),
        "rows": {
            "departures": int(len(dep)),
            "train": int(split.train.sum()),
            "valid": int(split.valid.sum()),
            "fit": int(rows.sum()),
            "flagged_valid": int(flagged[split.valid].sum()),
        },
        "digests": {
            "departures": frame_digests(dep),
            "features": frame_digests(features),
            "target": digest(y),
            "train mask": digest(split.train),
            "valid mask": digest(split.valid),
            "record-quality term": digest(tail),
            "fit rows": digest(rows),
            "encoding": frame_digests(encoding.columns),
            "encoding prior": digest(encoding.prior.values),
            **geometric,
            "design, fit rows": frame_digests(features.loc[rows, columns]),
            "design, valid rows": frame_digests(features.loc[split.valid, columns]),
            "valid movement ids": digest(dep.loc[split.valid, "MVT_ID_mvt"].astype(str)),
        },
        "arrays": None,
    }
    if args.keep_arrays:
        name = f"arrays_{args.label}.npz"
        np.savez(WORK / name, **arrays)
        record["arrays"] = name
    if args.cache:
        t0 = time.time()
        pd.to_pickle(
            {
                "label": args.label,
                "X": features[columns],
                "y": y,
                "train": split.train,
                "valid": split.valid,
                "unflagged": unflagged,
                "design": record["digests"]["design, fit rows"],
            },
            CACHE,
        )
        seconds["cache"] = time.time() - t0
    path = write_record(f"build_{args.label}", record)
    print(
        f"build {args.label}: features {seconds['features']:.0f} s, "
        f"peak {record['peak_rss_mib']:,.0f} MiB -> {path.relative_to(REPO)}"
    )
    return 0


# --------------------------------------------------------------------------
# fit


class _Log:
    """Collects LightGBM's native log lines, which ``verbose: -1`` suppresses."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(str(msg))

    def warning(self, msg: str) -> None:
        self.lines.append(str(msg))


def strategy(lines: list[str], params: dict) -> str:
    """Which histogram layout a fit used: forced by a parameter, or chosen.

    LightGBM logs its choice only when it makes one, and only at ``verbose``
    of one or more. So a forced layout is read off the parameters, and a
    chosen one off the log where the log was kept.
    """
    if params.get("force_col_wise"):
        return "col-wise (forced)"
    if params.get("force_row_wise"):
        return "row-wise (forced)"
    found = re.search(r"Auto-choosing (row|col)-wise", " ".join(lines))
    return f"{found.group(1)}-wise (auto)" if found else "auto (not logged)"


def fit(args) -> int:
    import lightgbm

    if not CACHE.exists():
        print(f"no cached inputs at {CACHE}; run the build step with --cache first")
        return 1
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    env = environment()
    cache = pd.read_pickle(CACHE)
    X, y = cache["X"], cache["y"]
    train, valid, unflagged = cache["train"], cache["valid"], cache["unflagged"]
    rung = LADDER_BY_KEY[RUNG]
    rows = fitting_rows(train, ~unflagged, y)
    if frame_digests(X.loc[rows, list(rung.features)]) != cache["design"]:
        print("the cached design does not match the build that wrote it")
        return 1

    params = dict(BOOSTER_PARAMS)
    params.update(CONFIGS[args.config])
    params["n_jobs"] = args.n_jobs
    if args.verbose:
        params["verbose"] = 1
    log = _Log()
    lightgbm.register_logger(log)

    tag = f"{args.config}_j{args.n_jobs}{'_verbose' if args.verbose else ''}_{args.label}"
    # Claimed before fitting, so that two processes given the same label cannot
    # overwrite one another's predictions under the other's record.
    try:
        (WORK / f"fit_{tag}.json").open("x").close()
    except FileExistsError:
        print(f"a fit record for {tag} exists already; choose another --label")
        return 1
    fits = []
    for i in range(args.repeats):
        log.lines.clear()
        load = os.getloadavg()[0]
        cpu0, t0 = os.times(), time.perf_counter()
        # The booster branch of ``fit_and_predict``, lines 336-339.
        model = Booster(rung.features, params).fit(X.loc[rows], y[rows])
        seconds = time.perf_counter() - t0
        cpu1 = os.times()
        pred = model.predict(X.loc[valid]).to_numpy(dtype=np.float64)
        path = WORK / f"pred_{tag}_{i}.npy"
        np.save(path, pred)
        fits.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
                "seconds": seconds,
                "cpu_user": cpu1.user - cpu0.user,
                "cpu_system": cpu1.system - cpu0.system,
                "loadavg_before": load,
                "strategy": strategy(log.lines, params),
                "log": [s.strip() for s in log.lines if "wise" in s or "eterminis" in s],
            }
        )
        print(f"  {tag} fit {i}: {seconds:.1f} s, {fits[-1]['strategy']}")
    record = {
        "kind": "fit",
        "config": args.config,
        "n_jobs": args.n_jobs,
        "verbose": bool(args.verbose),
        "label": args.label,
        "tag": tag,
        "started": started,
        "argv": sys.argv[1:],
        "params": {k: params[k] for k in sorted(params)},
        "environment": env,
        "build": cache["label"],
        "fit_rows": int(rows.sum()),
        "peak_rss_mib": peak_rss_mib(),
        "fits": fits,
    }
    write_record(f"fit_{tag}", record)
    return 0


# --------------------------------------------------------------------------
# report


def _records(kind: str) -> list[dict]:
    out = []
    for path in sorted(WORK.glob(f"{kind}_*.json")):
        text = path.read_text(encoding="utf-8")
        # A fit still running has claimed its name and written nothing yet.
        if text.strip():
            out.append(json.loads(text))
    return sorted(out, key=lambda r: r["started"])


def _max_delta(a, b) -> float:
    """Largest absolute difference; infinite if the missing values differ."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a)
    if (finite != np.isfinite(b)).any():
        return float("inf")
    return float(np.max(np.abs(a[finite] - b[finite]), initial=0.0))


def _fmt_delta(x: float) -> str:
    if x == 0.0:
        return "0"
    if not np.isfinite(x):
        return "missing values differ"
    return f"{x:.2e}" if x < 0.01 else f"{x:.2f}"


def _label_delta(x: float) -> str:
    """A prediction difference, and which side of ``SAME_TREES_S`` it is on."""
    if x == 0.0 or not np.isfinite(x):
        return _fmt_delta(x)
    kind = "last bits" if x <= SAME_TREES_S else "trees differ"
    return f"{_fmt_delta(x)} ({kind})"


def _references(extra: list[str]) -> list[tuple[str, Path, int | None]]:
    """``NAME=DIR`` or ``NAME=DIR@THREADS``; the ladder's own directory if none."""
    if not extra:
        extra = [f"{rb.RESIDUALS.relative_to(REPO)}={rb.RESIDUALS}"]
    out = []
    for item in extra:
        name, _, where = item.partition("=")
        where, _, threads = where.partition("@")
        out.append(
            (name.strip(), Path(where.strip()), int(threads) if threads.strip() else None)
        )
    return out


def _arrays(b: dict) -> dict[str, np.ndarray] | None:
    name = b.get("arrays")
    if not name or not (WORK / name).exists():
        return None
    with np.load(WORK / name) as z:
        return {k: z[k] for k in z.files}


def _how_far(builds, base, other, key, y_valid, valid) -> str:
    """For a component that differs: its size, from builds that kept arrays.

    ``other`` is measured against any build that kept its arrays and agrees
    with ``base`` on ``key`` -- the same outcome as ``base``, reproduced.
    """
    theirs = _arrays(other)
    if theirs is None:
        return ""
    stem = _array_name(key)
    names = [k for k in theirs if k == stem or k.startswith(stem + "__")]
    if not names:
        return ""
    for b in builds:
        if b is other or b["digests"].get(key) != base["digests"].get(key):
            continue
        mine = _arrays(b)
        if mine is None:
            continue
        d = max(_max_delta(theirs[k], mine[k]) for k in names)
        text = f", largest difference {_fmt_delta(d)} s against `{b['label']}`"
        if key.startswith("geometric") and y_valid is not None:
            moved = abs(
                rmse(y_valid, theirs[stem][valid]) - rmse(y_valid, mine[stem][valid])
            )
            text += f"; December RMSE moves {_fmt_delta(moved)} s"
        return text
    return ""


def _compare_builds(builds: list[dict], y_valid, valid) -> list[str]:
    base = builds[0]
    others = builds[1:]
    lines = [
        "| component | columns | "
        + " | ".join(f"`{b['label']}` vs `{base['label']}`" for b in others)
        + " |",
        "| --- | ---: | " + " | ".join("---" for _ in others) + " |",
    ]
    for key, ref in base["digests"].items():
        cells = []
        if isinstance(ref, dict):
            n = len(ref)
            for b in others:
                got = b["digests"].get(key, {})
                differ = sum(1 for c in ref if got.get(c) != ref[c])
                cells.append(
                    "identical"
                    if differ == 0
                    else f"**{differ} differ**"
                    + _how_far(builds, base, b, key, y_valid, valid)
                )
        else:
            n = 1
            for b in others:
                same = b["digests"].get(key) == ref
                cells.append(
                    "identical"
                    if same
                    else "**differs**" + _how_far(builds, base, b, key, y_valid, valid)
                )
        lines.append(f"| {key} | {n} | " + " | ".join(cells) + " |")
    return lines


def report(args) -> int:
    builds = _records("build")
    runs = _records("fit")
    if not runs:
        print("no fits recorded yet")
        return 1
    cache = pd.read_pickle(CACHE) if CACHE.exists() else None
    y_valid = valid = None
    if cache is not None:
        valid = cache["valid"].to_numpy(dtype=bool)
        y_valid = np.asarray(cache["y"], dtype=np.float64)[valid]

    # The ladder's stored residuals for this rung, each checked to hold the
    # build's validation rows in the build's order before it is compared by
    # position. A reference that fails the check is named, with the reason.
    refs: dict[str, tuple[np.ndarray, int | None]] = {}
    skipped: list[tuple[str, str]] = []
    ids = next((b["digests"].get("valid movement ids") for b in builds), None)
    for name, where, threads in _references(args.reference):
        path = where / f"{SPLIT}_{RUNG}.parquet"
        if not path.exists():
            skipped.append((name, "no residual file for this rung"))
            continue
        if not {"mvt_id", "residual"} <= set(pq.read_schema(path).names):
            skipped.append((name, "the file carries no movement ids to align on"))
            continue
        if ids is None or y_valid is None:
            skipped.append((name, "no build or cache to check its rows against"))
            continue
        frame = pd.read_parquet(path, columns=["mvt_id", "residual"])
        if digest(frame["mvt_id"].astype(str)) != ids:
            skipped.append((name, "its rows are not the build's validation rows"))
            continue
        refs[name] = (frame["residual"].to_numpy(dtype=np.float64), threads)
    for name, why in skipped:
        print(f"reference {name!r} skipped: {why}")

    # Every fit, grouped by configuration and thread count across processes.
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["config"], r["n_jobs"], r["verbose"]), []).append(r)
    preds = {f["file"]: np.load(WORK / f["file"]) for r in runs for f in r["fits"]}
    order = list(CONFIGS)
    keys = sorted(
        groups,
        key=lambda k: (order.index(k[0]) if k[0] in order else len(order), k[2], k[1]),
    )

    def label(key) -> str:
        return f"`{key[0]}`" + (" (logged)" if key[2] else "")

    summary: dict[tuple, dict] = {}
    for key in keys:
        fits = [(r, f) for r in groups[key] for f in r["fits"]]
        distinct: dict[str, np.ndarray] = {}
        for _, f in fits:
            distinct.setdefault(f["sha256"], preds[f["file"]])
        summary[key] = {
            "n": len(fits),
            "processes": len(groups[key]),
            "distinct": distinct,
            "largest": max(
                (
                    _max_delta(a, b)
                    for a, b in itertools.combinations(distinct.values(), 2)
                ),
                default=0.0,
            ),
            "rmse": [rmse(y_valid, p) for p in distinct.values()]
            if y_valid is not None
            else [],
            "seconds": [f["seconds"] for _, f in fits],
            "user": float(np.mean([f["cpu_user"] for _, f in fits])),
            "system": float(np.mean([f["cpu_system"] for _, f in fits])),
            "strategies": sorted(
                {strategy(f.get("log", []), r["params"]) for r, f in fits}
            ),
            "load": [
                f.get("loadavg_before", r["environment"]["loadavg_at_start"][0])
                for r, f in fits
            ],
        }

    def cost(key) -> str:
        """Median fit time against ``current`` at the same thread count.

        The median, because the first fit after a change of process can run
        well over the rest -- one did here by half -- and a mean over six fits
        would charge that to the configuration.
        """
        ref = summary.get(("current", key[1], False))
        if ref is None or key[0] == "current":
            return "--"
        ratio = np.median(summary[key]["seconds"]) / np.median(ref["seconds"]) - 1.0
        return f"{ratio:+.0%}"

    rows = []
    for key in keys:
        s = summary[key]
        one = s["n"] < 2
        r = s["rmse"]
        rows.append(
            [
                label(key),
                str(key[1]),
                f"{s['n']} in {s['processes']}",
                "n/a (1 fit)"
                if one
                else ("**yes**" if len(s["distinct"]) == 1 else "no"),
                str(len(s["distinct"])),
                "--" if one else _label_delta(s["largest"]),
                (f"{min(r):.2f}" if len(r) == 1 else f"{min(r):.2f}-{max(r):.2f}")
                if r
                else "--",
                f"{np.median(s['seconds']):.1f} [{min(s['seconds']):.1f}-"
                f"{max(s['seconds']):.1f}]",
                cost(key),
                f"{s['user']:.0f} / {s['system']:.0f}",
                ", ".join(s["strategies"]),
            ]
        )

    repeatable = [
        k for k in keys if summary[k]["n"] >= 2 and len(summary[k]["distinct"]) == 1
    ]
    varying = [
        k for k in keys if summary[k]["n"] >= 2 and len(summary[k]["distinct"]) > 1
    ]
    cross = []
    for a, b in itertools.combinations(repeatable, 2):
        d = _max_delta(
            next(iter(summary[a]["distinct"].values())),
            next(iter(summary[b]["distinct"].values())),
        )
        cross.append(
            [
                f"{label(a)} at {a[1]}",
                f"{label(b)} at {b[1]}",
                "identical" if d == 0.0 else _label_delta(d),
            ]
        )

    # The ladder's residuals against the harness's ``current``: a harness whose
    # fit is the ladder's cannot be expected to match a configuration that does
    # not repeat, but it has to land among the same outcomes.
    fidelity = []
    for name, (resid, threads) in refs.items():
        ref_rmse = rmse(y_valid, y_valid - resid)
        for key in keys:
            if key[0] != "current" or (threads is not None and key[1] != threads):
                continue
            s = summary[key]
            gaps = [_max_delta(y_valid - p, resid) for p in s["distinct"].values()]
            fidelity.append(
                [
                    name + (f" ({threads} threads)" if threads else ""),
                    f"{ref_rmse:.2f}",
                    f"{label(key)} at {key[1]}",
                    str(len(s["distinct"])),
                    f"{min(s['rmse']):.2f}-{max(s['rmse']):.2f}",
                    _label_delta(min(gaps)),
                    str(sum(1 for g in gaps if g <= SAME_TREES_S)),
                ]
            )

    # The parameters every fit ran under, read from its own record rather than
    # from the tree as it stands when the report is written.
    base_params: dict | None = None
    mixed: list[str] = []
    for r in runs:
        extras = CONFIGS.get(r["config"], {})
        base = {
            k: v
            for k, v in r["params"].items()
            if k not in ("n_jobs", "verbose") and k not in extras
        }
        if base_params is None:
            base_params = base
        if base != base_params or any(r["params"].get(k) != v for k, v in extras.items()):
            mixed.append(r["tag"])
    used = sorted({r["build"] for r in runs})
    per = sorted({len(r["fits"]) for r in runs})
    per_text = f"{per[0]}" if len(per) == 1 else f"{per[0]} to {per[-1]}"
    loads = [x for k in keys for x in summary[k]["load"]]

    env = runs[0]["environment"]
    first = builds[0] if builds else None
    text: list[str] = [
        "# Determinism of one rung",
        "",
        "Generated by `scripts/check_determinism.py` on "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d")
        + ". Two consecutive `make baseline` runs on one machine agreed on none of the",
        "residual files byte for byte. This finds where that comes from on one split",
        f"and one rung -- {SPLIT.title()}, `{RUNG}` -- fitted repeatedly from a cached",
        "feature frame, rather than by refitting the ladder. Every figure below is",
        "computed by the script from its own records under",
        "`data/processed/determinism/`, which are gitignored, except the comparison",
        "with the ladder's stored residuals, which reads the directories that section",
        "names as they stood when the report was written.",
        "",
        "## Setup",
        "",
        f"- Machine: {env['cpu']}, {env['cores']} logical CPUs, {env['machine']}.",
        f"- Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, "
        f"pyarrow {env['pyarrow']}, scipy {env['scipy']}, LightGBM {env['lightgbm']}.",
    ]
    if first:
        rr = first["rows"]
        text += [
            f"- {rr['departures']:,} departures; December's split trains on "
            f"{rr['train']:,} and validates on {rr['valid']:,}; the rung is fitted on "
            f"{rr['fit']:,} of the training rows.",
        ]
    text += [
        f"- {len(runs)} fit processes, {sum(len(r['fits']) for r in runs)} fits, run "
        "one after another. The one-minute load average as each fit began ranged "
        f"from {min(loads):.2f} to {max(loads):.2f}; it carries the tail of the fit "
        "before.",
        "",
        "## The feature frame",
        "",
    ]
    if len(builds) >= 2:
        fixed = {
            b["label"]: v
            for b in builds
            if (v := b["environment"]["env"].get("PYTHONHASHSEED"))
            not in (None, "", "random")
        }
        seeds = (
            "so each also draws its own string-hash seed"
            if not fixed
            else "and each draws its own string-hash seed except "
            + ", ".join(f"`{k}` (fixed at {v})" for k, v in fixed.items())
        )
        text += [
            f"Each build is a separate process reading the data from disk, {seeds}.",
            "A digest is SHA-256 over a column's values: numeric, boolean and naive",
            "datetime columns bit for bit, categoricals by their codes and their",
            "category order, and every other column -- strings, the movement ids,",
            "time-zoned times -- through pandas' fixed-key 64-bit row hash, which tells",
            "two builds apart without being a bitwise comparison. Where a build kept",
            "its arrays, a component that differs is measured against a build that",
            "reproduced the other side.",
            "",
            "| build | `OPENBLAS_NUM_THREADS` | `PYTHONHASHSEED` | load (s) | "
            "features (s) | encoding (s) | geometric (s) | peak memory (GiB) | "
            "arrays kept |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for b in builds:
            e = b["environment"]["env"]
            s = b["seconds"]
            text.append(
                f"| `{b['label']}` | {e.get('OPENBLAS_NUM_THREADS') or 'unset'} | "
                f"{e.get('PYTHONHASHSEED') or 'unset (random)'} | {s['load']:.0f} | "
                f"{s['features']:.0f} | {s['encoding']:.0f} | {s['geometric']:.0f} | "
                f"{b['peak_rss_mib'] / 1024:.1f} | {'yes' if b.get('arrays') else 'no'} |"
            )
        text += ["", *_compare_builds(builds, y_valid, valid), ""]
        # The same builds grouped by thread setting and read against each other
        # rather than against the first: whether each setting repeats on its own.
        by_setting: dict[str, list[dict]] = {}
        for b in builds:
            setting = b["environment"]["env"].get("OPENBLAS_NUM_THREADS") or "unset"
            by_setting.setdefault(setting, []).append(b)
        agree = []
        for setting, group in by_setting.items():
            if len(group) < 2:
                continue
            same = all(g["digests"] == group[0]["digests"] for g in group[1:])
            names = ", ".join(f"`{g['label']}`" for g in group)
            agree.append(
                f"{names} ({setting}): {'identical' if same else '**not identical**'}"
            )
        if agree:
            text += [
                "Builds under the same `OPENBLAS_NUM_THREADS`, against each other on"
                " every digest: " + "; ".join(agree) + ".",
                "",
            ]
    else:
        text += ["Fewer than two builds are recorded; nothing to compare.", ""]

    text += [
        "## The booster",
        "",
        "Each row is one configuration at one thread count; its fits ran in one or",
        f"more processes, {per_text} to a process, from the same cached inputs.",
        "*Byte-identical* means every prediction array hashed the same, and needs",
        "at least two fits to mean anything. *Largest delta* is the largest absolute",
        "difference in any prediction between any two of its distinct outcomes, in",
        f"seconds, marked *last bits* at or below {SAME_TREES_S:g} s and *trees differ*",
        "above. *RMSE* is over December's validation rows, the ladder's own",
        "unflagged-and-flagged figure, across the distinct outcomes. Fit time is the",
        "wall time of `Booster.fit`, its median and range, and *vs current* compares",
        "that median with `current` at the same thread count; CPU is user / system",
        "seconds per fit, all threads. *Strategy* is the histogram layout: read off",
        "the parameters where one is forced, off LightGBM's log where it chose one --",
        "the ladder's `verbose: -1` suppresses that line, so rows marked *(logged)*",
        "are the same configuration fitted with `verbose: 1`, which changes only what",
        "is printed.",
        "",
        "| configuration | threads | fits in processes | byte-identical | outcomes | "
        "largest delta (s) | RMSE (s) | fit (s), median [range] | vs current | "
        "CPU user / system (s) | strategy |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *["| " + " | ".join(r) + " |" for r in rows],
        "",
        "Byte-identical over every one of at least two fits: "
        + (", ".join(f"{label(k)} at {k[1]}" for k in repeatable) or "none")
        + ". More than one outcome: "
        + (
            ", ".join(
                f"{label(k)} at {k[1]} ({len(summary[k]['distinct'])} in "
                f"{summary[k]['n']})"
                for k in varying
            )
            or "none"
        )
        + ".",
        "",
    ]
    if cross:
        text += [
            "The configurations that repeated, compared with each other (largest",
            "absolute prediction difference, seconds):",
            "",
            "| one | other | difference |",
            "| --- | --- | --- |",
            *["| " + " | ".join(c) + " |" for c in cross],
            "",
        ]

    text += ["## The harness against the ladder", ""]
    if fidelity:
        text += [
            "The ladder's stored residuals for this rung against the harness's",
            "`current`, at the reference's thread count where it is given. A fit that",
            "does not repeat cannot be matched bit for bit; a faithful harness has to",
            "land among the same outcomes, so its RMSE range and its closest outcome",
            "are what is read.",
            "",
            "| reference | its RMSE (s) | harness | outcomes | harness RMSE (s) | "
            "closest outcome, largest delta (s) | outcomes within the last bits |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            *["| " + " | ".join(c) + " |" for c in fidelity],
            "",
        ]
    else:
        text += ["No reference residuals could be compared.", ""]
    if skipped:
        text += [
            "Not compared: "
            + "; ".join(f"{name}, because {why}" for name, why in skipped)
            + ".",
            "",
        ]

    text += [
        "## Parameters",
        "",
        "Read from each fit's own record. Every fit read the cached inputs of build "
        + ", ".join(f"`{u}`" for u in used)
        + (
            ""
            if cache is None or used == [cache["label"]]
            else f"; the target and validation rows above are from `{cache['label']}`"
        )
        + ".",
        "",
    ]
    if mixed:
        text += [
            "**These fit processes ran under different base parameters from the",
            "rest and are pooled wrongly:** " + ", ".join(f"`{t}`" for t in mixed) + ".",
            "",
        ]
    text += [
        "Every configuration shares these, with `n_jobs` the row's thread count and",
        "`verbose` -1 except in the logged rows:",
        "",
        "- " + ", ".join(f"`{k}={v}`" for k, v in (base_params or {}).items()),
        "",
        "and adds:",
        "",
        *[
            f"- `{name}`: "
            + (", ".join(f"`{k}={v}`" for k, v in extra.items()) or "nothing")
            for name, extra in CONFIGS.items()
        ],
        "",
        "Distance matrices derived from OpenStreetMap: data (c)",
        "OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied by",
        "EUROCONTROL PRC and the OpenSky Network under the terms of the PRC Data",
        "Challenge 2026.",
        "",
    ]
    body = "\n".join(text)
    if ATTRIBUTION not in body:
        raise RuntimeError("the report lost its OpenStreetMap attribution")
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the frame once and record its digests")
    b.add_argument("--label", required=True)
    b.add_argument("--cache", action="store_true", help="keep the rung's inputs")
    b.add_argument(
        "--keep-arrays",
        action="store_true",
        help="keep the encoding and geometric outputs, to size a difference",
    )
    f = sub.add_parser("fit", help="fit the rung repeatedly under one configuration")
    f.add_argument("--config", choices=list(CONFIGS), required=True)
    f.add_argument("--n-jobs", type=int, required=True)
    f.add_argument("--repeats", type=int, default=3)
    f.add_argument("--verbose", action="store_true", help="log the histogram strategy")
    f.add_argument(
        "--label",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        + f"-p{os.getpid()}",
    )
    r = sub.add_parser("report", help="compare the recorded fits")
    r.add_argument(
        "--reference",
        action="append",
        default=[],
        help="NAME=DIR[@THREADS] of a run's residuals, compared by position",
    )
    args = ap.parse_args()
    return {"build": build, "fit": fit, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
