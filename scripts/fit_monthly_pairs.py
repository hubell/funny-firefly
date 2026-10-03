#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The three held-out-pair folds of ``reports/baseline.md``, refitted for three
rungs, with every held-out and seen-sample prediction stored.

``run_baselines.run`` scores the pair folds and keeps their predictions in
memory; it writes no file for them (``not split.name.startswith("pairs-")``).
The month-by-month reading of the geometry claim needs the per-row predictions,
so this script refits the three folds of
``taxiout.validation.held_out_pair_splits``, on the full training year exactly
as the ladder builds them, for three rungs only: the two of the claim,
``run_baselines.CLAIM``, and the shipped rung, ``SUBMISSION_KEY``, read for its
gap alone. It does so by ``run_baselines.run`` itself, one fold per call, with
``keep_residuals`` off and ``run_baselines.LADDER`` pointed at the three rungs
for the length of the call and then restored, the restore asserted.

**It changes nothing that ships, chooses nothing, and decides nothing.**

Modes::

    --fit      the folds whose three files are not on disk
    --check    from the stored files alone: the fold-mean seen and held-out
               figures, the gaps, the claim and the per-airport claim table of
               ``reports/baseline.md``, at the printed precision

Files: ``data/processed/monthly_pairs/<fold>_<key>.parquet``, gitignored (they
carry the target): movement identifier, pair, month, airport, side, whether the
term flags the row, the recorded taxi-out and the prediction; held-out rows
first, then the seen sample, each in the frame's order.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import report_monthly as rm  # noqa: E402
import run_baselines as rb  # noqa: E402
from taxiout.models import LADDER, LADDER_BY_KEY, SUBMISSION_KEY  # noqa: E402
from taxiout.validation import (  # noqa: E402
    PAIR_FOLDS,
    Split,
    cluster_bootstrap,
    difference_in_differences,
    fold_rmses,
    gap,
    held_out_pair_splits,
)

WORK = REPO / "data" / "processed" / "monthly_pairs"
LADDER_RESIDUALS = REPO / "data" / "processed" / "residuals"
if rb.RESIDUALS != LADDER_RESIDUALS:
    raise SystemExit("run_baselines.RESIDUALS is not the ladder's directory")
FORBIDDEN = (
    REPO / "reports",
    LADDER_RESIDUALS,
    REPO / "data" / "processed" / "monthly",
    REPO / "data" / "processed" / "monthly_ladder",
    REPO / "data" / "raw",
)
# ``reports/baseline.md``, named through its own generator: check 9 reads a
# report path assigned at module level here as a report this script writes.
BASELINE = rb.OUT

# The three rungs, in ladder order: the claim's two and the shipped rung.
KEYS: tuple[str, ...] = tuple(
    r.key for r in LADDER if r.key in set(rb.CLAIM) | {SUBMISSION_KEY}
)
if KEYS != (rb.CLAIM[0], rb.CLAIM[1], SUBMISSION_KEY):
    raise SystemExit(f"the three rungs are not the claim's and the shipped one: {KEYS}")
FULL_LADDER = rb.LADDER
COLUMNS = ("mvt_id", "pair", "month", "icao", "side", "flagged", "y", "pred")
SIDES = (("held", "valid"), ("seen", "seen"))


def fold_names() -> list[str]:
    return [f"pairs-{k + 1}" for k in range(PAIR_FOLDS)]


def fold_path(fold: str, key: str) -> Path:
    return WORK / f"{fold}_{key}.parquet"


def record_path(fold: str) -> Path:
    return WORK / f"{fold}_run.json"


def fold_done(fold: str) -> bool:
    return all(fold_path(fold, k).exists() for k in KEYS)


def fingerprint() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


@contextlib.contextmanager
def ladder_of_three():
    """Point ``run_baselines.LADDER`` at the three rungs for one call."""
    saved = rb.LADDER
    rb.LADDER = tuple(r for r in saved if r.key in KEYS)
    try:
        yield
    finally:
        rb.LADDER = saved
    if rb.LADDER is not FULL_LADDER or tuple(rb.LADDER) != tuple(LADDER):
        raise RuntimeError("run_baselines.LADDER was not restored")


# --------------------------------------------------------------------------
# --fit


def stored_frames(split, results, features, y, dep, month) -> dict[str, pd.DataFrame]:
    """One fold's rows as stored: held-out rows, then the seen sample, each in
    the frame's order, one frame per rung."""
    flagged = results[("meta", split.name, "tail")]["mask"].astype(bool)
    frames = {}
    for key in KEYS:
        parts = []
        for side, which in SIDES:
            mask = split.valid if which == "valid" else split.extra["seen"]
            pred = results[(split.name, key, which)]["pred"]
            idx = y.index[mask.to_numpy()]
            if not isinstance(pred, pd.Series) or not pred.index.equals(idx):
                raise RuntimeError(
                    f"{split.name} {key} {which}: predictions not in frame order"
                )
            rows = dep.loc[idx]
            parts.append(
                pd.DataFrame(
                    {
                        "mvt_id": rows["MVT_ID_mvt"].to_numpy(),
                        "pair": features.loc[idx, "pair"].astype(str).to_numpy(),
                        "month": np.asarray(month)[mask.to_numpy()],
                        "icao": features.loc[idx, "icao"].astype(str).to_numpy(),
                        "side": side,
                        "flagged": flagged.loc[idx].to_numpy(dtype=bool),
                        "y": np.asarray(y.loc[idx], dtype=float),
                        "pred": pred.to_numpy(dtype=float),
                    }
                )
            )
        frames[key] = pd.concat(parts, ignore_index=True)[list(COLUMNS)]
    return frames


def fit() -> int:
    before = fingerprint()
    todo = [f for f in fold_names() if not fold_done(f)]
    skipped = [f for f in fold_names() if fold_done(f)]
    print(
        f"writing only to {WORK.relative_to(REPO)}; {len(before)} files checked untouched",
        flush=True,
    )
    if skipped:
        print(f"on disk, skipped: {', '.join(skipped)}", flush=True)
    if not todo:
        print("nothing to fit\nDONE", flush=True)
        return 0
    t0 = time.monotonic()
    frame = rm.Frame()
    features, y, dep = frame.features, frame.y, frame.dep
    splits = held_out_pair_splits(features["icao"], features["pair"], n_folds=PAIR_FOLDS)
    if [s.name for s in splits] != fold_names():
        raise RuntimeError("the folds are not pairs-1 to pairs-3")
    WORK.mkdir(parents=True, exist_ok=True)
    month = frame.month.to_numpy()
    for split in splits:
        if split.name not in todo:
            continue
        print(
            f"\n== {split.name} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z)",
            flush=True,
        )
        tf = time.monotonic()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tails = rb.Tails()
        with ladder_of_three():
            results = rb.run(features, y, dep, [split], tails, keep_residuals=False)
        fold_s = time.monotonic() - tf
        if rb.RESIDUALS != LADDER_RESIDUALS:
            raise RuntimeError("run_baselines.RESIDUALS moved")
        record = {
            "fold": split.name,
            "started_utc": started,
            "fold_s": fold_s,
            "sizes": split.sizes,
            "flagged": results[("meta", split.name, "tail")]["flagged"],
            "encoding_fit_rows": int(results[("meta", split.name, "encoding")].fit_rows),
            "frame_build_s": frame.build_seconds,
            "peak_rss_gib": rm.peak_gib(),
            "rungs": {},
        }
        frames = stored_frames(split, results, features, y, dep, month)
        for key in KEYS:
            valid = results[(split.name, key, "valid")]
            record["rungs"][key] = {
                "seconds": float(valid["seconds"]),
                "rmse_every_row": float(valid["rmse"]),
                "rows_held": int(split.valid.sum()),
                "rows_seen": int(split.extra["seen"].sum()),
            }
            print(
                f"  {split.name} {key}: every-row RMSE {valid['rmse']:.1f} s, "
                f"{valid['seconds']:.0f} s",
                flush=True,
            )
        record["peak_rss_gib"] = rm.peak_gib()
        record_path(split.name).write_text(json.dumps(record, indent=1), encoding="utf-8")
        for key, out in frames.items():
            tmp = fold_path(split.name, key).with_suffix(".tmp")
            out.to_parquet(tmp, index=False)
            tmp.rename(fold_path(split.name, key))
        del results, frames
        print(f"  {split.name}: {fold_s:.0f} s", flush=True)
        moved = rm.touched(before, fingerprint())
        if moved:
            print("TOUCHED: " + ", ".join(moved), file=sys.stderr)
            return 1
    print(
        f"\n{len(todo)} fold(s) in {(time.monotonic() - t0) / 60:.1f} min; nothing touched"
    )
    print("DONE", flush=True)
    return 0


# --------------------------------------------------------------------------
# reading the stored folds back into run_baselines' own shapes


def load_fold(fold: str) -> dict:
    """One fold's stored rows, checked to agree across the three rungs."""
    first = pd.read_parquet(fold_path(fold, KEYS[0]))
    preds = {}
    for key in KEYS:
        f = pd.read_parquet(fold_path(fold, key))
        for col in COLUMNS:
            if col == "pred":
                continue
            if not np.array_equal(f[col].to_numpy(), first[col].to_numpy()):
                raise SystemExit(
                    f"{fold}_{key}: column {col} differs from {fold}_{KEYS[0]}"
                )
        preds[key] = f["pred"].to_numpy(dtype=float)
    return {"rows": first.drop(columns=["pred"]), "pred": preds}


def as_results(fold: str, data: dict, keep: np.ndarray | None = None):
    """``(results, features, y, split)`` as ``run_baselines._fold_sums`` reads them.

    The index is the stored rows' order: held-out rows, then the seen sample,
    each in the frame's order, which is the order ``_fold_sums`` sees them in,
    so the pairs factorize in the same order and the bootstrap draws the same
    pairs. ``keep`` drops rows from both sides (a month, a 4 h reading).
    """
    rows = data["rows"]
    n = len(rows)
    index = pd.RangeIndex(n)
    held = rows["side"].to_numpy() == "held"
    seen = rows["side"].to_numpy() == "seen"
    if not (held | seen).all():
        raise SystemExit(f"{fold}: a row is on neither side")
    if keep is not None:
        held = held & keep
        seen = seen & keep
    features = pd.DataFrame(
        {"icao": rows["icao"].to_numpy(), "pair": rows["pair"].to_numpy()}, index=index
    )
    y = pd.Series(rows["y"].to_numpy(dtype=float), index=index)
    split = Split(
        name=fold,
        description="stored",
        train=pd.Series(False, index=index),
        valid=pd.Series(held, index=index),
        extra={"seen": pd.Series(seen, index=index)},
    )
    results = {
        ("meta", fold, "tail"): {
            "mask": pd.Series(rows["flagged"].to_numpy(dtype=bool), index=index)
        }
    }
    for key in KEYS:
        pred = pd.Series(data["pred"][key], index=index)
        results[(fold, key, "valid")] = {"pred": pred.where(held)}
        results[(fold, key, "seen")] = {"pred": pred.where(seen)}
    return results, features, y, split


def fold_sums(folds: dict[str, dict], *, airport: str | None = None, keep=None) -> list:
    """``run_baselines._fold_sums`` over the stored folds, ex-flagged, one fold
    per call so each reads its own index; ``keep`` maps a fold to a row mask."""
    out = []
    for fold, data in folds.items():
        results, features, y, split = as_results(
            fold, data, None if keep is None else keep[fold]
        )
        out += rb._fold_sums(
            results, features, y, [split], ex_flagged=True, airport=airport
        )
    return out


# --------------------------------------------------------------------------
# --check


def baseline_tables() -> dict:
    """The three tables of ``reports/baseline.md``'s *Held-out pairs*, as printed."""
    text = BASELINE.read_text(encoding="utf-8").splitlines()
    start = text.index("## Held-out pairs")

    def table_after(heading: str) -> list[list[str]]:
        at = text.index(heading, start)
        out, seen_rule = [], False
        for line in text[at + 1 :]:
            if line.startswith("#"):
                break
            if not line.startswith("|"):
                if out:
                    break
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set(cells[0]) <= {"-", ":", " "}:
                seen_rule = True
                continue
            if seen_rule:
                out.append(cells)
        return out

    return {
        "rungs": {r[0]: r for r in table_after("### Every rung, seen against held out")},
        "steps": table_after("### What each block buys where the pair is new"),
        "airports": table_after("### The claim, per airport"),
    }


def check() -> int:
    before = fingerprint()
    missing = [f for f in fold_names() if not fold_done(f)]
    if missing:
        raise SystemExit(f"not on disk: {', '.join(missing)}; run --fit")
    folds = {f: load_fold(f) for f in fold_names()}
    tables = baseline_tables()
    lines: list[str] = []
    ok_all = True

    def verdict(ok: bool, text: str) -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        line = f"[{'ok ' if ok else 'STOP'}] {text}"
        lines.append(line)
        print("    " + line, flush=True)

    sums = fold_sums(folds)
    if len(sums) != PAIR_FOLDS:
        verdict(False, f"{len(sums)} folds summed, not {PAIR_FOLDS}")
    per_fold = [fold_rmses(f) for f in sums]
    for key in KEYS:
        rung = LADDER_BY_KEY[key]
        printed = tables["rungs"].get(rung.label)
        if printed is None:
            verdict(False, f"no row for {rung.label!r} in baseline.md")
            continue
        seen_m = f"{float(np.mean([f['seen'][key] for f in per_fold])):,.1f}"
        held_m = f"{float(np.mean([f['held'][key] for f in per_fold])):,.1f}"
        g = str(cluster_bootstrap(sums, gap(key)))
        per = " / ".join(f"{f['held'][key] - f['seen'][key]:+,.0f}" for f in per_fold)
        verdict(
            seen_m == printed[1], f"{key}: seen {seen_m} s, baseline.md {printed[1]} s"
        )
        verdict(
            held_m == printed[2],
            f"{key}: held out {held_m} s, baseline.md {printed[2]} s",
        )
        verdict(g == printed[3], f"{key}: gap {g}, baseline.md {printed[3]}")
        verdict(per == printed[4], f"{key}: gap per fold {per}, baseline.md {printed[4]}")

    a, b = rb.CLAIM
    claim_rows = [r for r in tables["steps"] if "(the claim)" in r[0]]
    if len(claim_rows) != 1:
        verdict(False, f"{len(claim_rows)} claim rows in baseline.md")
    else:
        row = claim_rows[0]
        did = str(cluster_bootstrap(sums, difference_in_differences(b, a)))
        seen_d = (
            f"{float(np.mean([f['seen'][a] - f['seen'][b] for f in per_fold])):+,.1f}"
        )
        held_d = (
            f"{float(np.mean([f['held'][a] - f['held'][b] for f in per_fold])):+,.1f}"
        )
        verdict(
            seen_d == row[1],
            f"the claim on seen pairs {seen_d} s, baseline.md {row[1]} s",
        )
        verdict(
            held_d == row[2],
            f"the claim on held-out pairs {held_d} s, baseline.md {row[2]} s",
        )
        verdict(did == row[3], f"the claim {did}, baseline.md {row[3]}")

    printed_ap = {r[0]: r for r in tables["airports"]}
    icaos = sorted(
        set().union(*(set(d["rows"]["icao"].unique()) for d in folds.values()))
    )
    mine_ap = {}
    for ap in icaos:
        ap_sums = fold_sums(folds, airport=ap)
        if len(ap_sums) < PAIR_FOLDS:
            continue
        n_held = int(sum(next(iter(f["held"].values())).n.sum() for f in ap_sums))
        r = cluster_bootstrap(ap_sums, difference_in_differences(b, a))
        clear = "yes" if (r.low > 0 or r.high < 0) else "no"
        mine_ap[ap] = [ap, f"{n_held:,}", str(r), clear]
    verdict(
        set(mine_ap) == set(printed_ap),
        f"airports {sorted(mine_ap)} against baseline.md {sorted(printed_ap)}",
    )
    for ap in sorted(set(mine_ap) | set(printed_ap)):
        got, want = mine_ap.get(ap), printed_ap.get(ap)
        verdict(
            got == want,
            f"per-airport claim {' | '.join(got or ['missing'])}; baseline.md {' | '.join(want or ['missing'])}",
        )

    moved = rm.touched(before, fingerprint())
    if moved:
        verdict(False, "TOUCHED: " + ", ".join(moved))
    (WORK / "check.json").write_text(
        json.dumps(
            {
                "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ok": bool(ok_all),
                "lines": lines,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    if not ok_all:
        print("the pair folds are not the ladder's; stopping", file=sys.stderr)
        return 1
    print(f"{len(lines)} checks held\nDONE", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    return fit() if args.fit else check()


if __name__ == "__main__":
    raise SystemExit(main())
