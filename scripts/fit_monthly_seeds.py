#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The shipped rung under four more seeds in every held-out month.

``scripts/report_monthly.py`` fitted ``SUBMISSION_KEY`` once per held-out
month of 2025, at the
ladder's seed (offset 0 on ``BOOSTER_PARAMS["random_state"]``). The submission
is the mean of ``BAGGING_SEEDS`` fits whose seeds run consecutively from that
one, as ``run_baselines``' seed-averaging section and
``taxiout.models.SeedAverage`` fit it. This script fits the other four, offsets
1 to 4, on each of the twelve held-out-month splits, by that script's own
``fit_leg`` with only the seed offset moved; offset 0 is its file and is
not refitted.

**It changes nothing that ships, chooses nothing, and decides nothing.**

Modes::

    --check                   offset 0 on January, against report_monthly.py's file, to 0 s
    --fit [--months M ...]    the four offsets of every month not yet on disk

``--check`` must have passed before ``--fit`` fits anything: refitting offset 0
for January and finding ``report_monthly.py``'s file to the bit on every row
proves that the path the other offsets take is that script's. The files land
under
``data/processed/monthly_seeds/`` (gitignored: they carry the target), one per
month and offset in ``report_monthly.py``'s seven columns, and ``<month>_run.json`` is
written last, so a stopped run resumes at the next month.
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
from taxiout.models import BAGGING_SEEDS, BOOSTER_PARAMS, SUBMISSION_KEY  # noqa: E402

WORK = REPO / "data" / "processed" / "monthly_seeds"
MONTHLY = REPO / "data" / "processed" / "monthly"
FORBIDDEN = (
    REPO / "reports",
    REPO / "data" / "processed" / "residuals",
    MONTHLY,
    REPO / "data" / "processed" / "monthly_ladder",
    REPO / "data" / "raw",
)
MONTHS: tuple[str, ...] = rm.MONTHS
# The offsets fitted here; 0 is ``report_monthly.py``'s. Together they are the submission's
# seeds, which ``BAGGING_SEEDS`` counts.
OFFSETS: tuple[int, ...] = (1, 2, 3, 4)
if (0,) + OFFSETS != tuple(range(BAGGING_SEEDS)):
    raise SystemExit("the offsets are not the submission's seeds")
CHECK_MONTH = "2025-01"
CHECK_RECORD = WORK / "check.json"


def seed_path(month: str, offset: int) -> Path:
    return WORK / f"{month}_seed{offset}.parquet"


def check_path() -> Path:
    return WORK / f"check_{CHECK_MONTH}_seed0.parquet"


def run_path(month: str) -> Path:
    return WORK / f"{month}_run.json"


def month_done(month: str) -> bool:
    return run_path(month).exists() and all(seed_path(month, o).exists() for o in OFFSETS)


def fingerprint() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


@contextlib.contextmanager
def seed_offset(offset: int):
    """Point ``report_monthly.SEED`` at ``offset`` for one call, then restore it."""
    saved = rm.SEED
    rm.SEED = int(offset)
    try:
        yield
    finally:
        rm.SEED = saved
    if rm.SEED != 0:
        raise RuntimeError("report_monthly.SEED was not restored")


def leg_for(frame: rm.Frame, month: str) -> rm.Leg:
    legs = [leg for leg in frame.legs if leg.name == f"heldout-{month}"]
    if len(legs) != 1:
        raise RuntimeError(f"no held-out leg for {month}")
    return legs[0]


def fit_one(frame: rm.Frame, month: str, offset: int) -> tuple[pd.DataFrame, dict]:
    with seed_offset(offset):
        out, record = rm.fit_leg(frame, leg_for(frame, month))
    record["seed_offset"] = int(offset)
    record["random_state"] = int(BOOSTER_PARAMS["random_state"]) + int(offset)
    return out, record


def same_as_monthly(out: pd.DataFrame, month: str) -> tuple[list[str], bool]:
    """Every column of ``report_monthly.py``'s file, row for row, to the bit."""
    stored = pd.read_parquet(MONTHLY / f"heldout-{month}_{SUBMISSION_KEY}.parquet")
    lines = []
    got, ok = rm.reproduces(out, MONTHLY / f"heldout-{month}_{SUBMISSION_KEY}.parquet")
    lines.append(f"residual, joined on the movement identifier: {got}")
    same_order = len(stored) == len(out) and np.array_equal(
        stored["mvt_id"].to_numpy(), out["mvt_id"].to_numpy()
    )
    lines.append(f"rows in the same order: {same_order}")
    ok &= same_order
    if same_order:
        for col in stored.columns:
            a, b = stored[col].to_numpy(), out[col].to_numpy()
            if col in ("residual", rm.TERM_COLUMN):
                worst = float(np.max(np.abs(a.astype(float) - b.astype(float))))
                eq = worst == 0.0
                lines.append(f"{col}: largest absolute difference {worst:.6g} s")
            else:
                eq = bool(np.array_equal(a, b))
                lines.append(f"{col}: identical {eq}")
            ok &= eq
    return lines, ok


def check() -> int:
    before = fingerprint()
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    frame = rm.Frame()
    out, record = fit_one(frame, CHECK_MONTH, 0)
    out.to_parquet(check_path(), index=False)
    lines, ok = same_as_monthly(out, CHECK_MONTH)
    for line in lines:
        print(f"    [{'ok ' if ok else 'STOP'}] {line}", flush=True)
    record.update(
        {
            "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "against": f"data/processed/monthly/heldout-{CHECK_MONTH}_{SUBMISSION_KEY}.parquet",
            "lines": lines,
            "ok": bool(ok),
            "frame_build_s": frame.build_seconds,
            "total_s": time.monotonic() - t0,
        }
    )
    CHECK_RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
    moved = rm.touched(before, fingerprint())
    if moved:
        print("TOUCHED: " + ", ".join(moved), file=sys.stderr)
        return 1
    if not ok:
        print("the seed path is not report_monthly.py's; stopping", file=sys.stderr)
        return 1
    print(f"seed path proven in {record['total_s']:.0f} s\nDONE", flush=True)
    return 0


def fit(months: list[str] | None) -> int:
    before = fingerprint()
    if not CHECK_RECORD.exists() or not json.loads(CHECK_RECORD.read_text())["ok"]:
        raise SystemExit("--check has not passed; nothing is fitted")
    wanted = list(months or MONTHS)
    unknown = sorted(set(wanted) - set(MONTHS))
    if unknown:
        raise SystemExit(f"not a month of 2025: {', '.join(unknown)}")
    todo = [m for m in MONTHS if m in wanted and not month_done(m)]
    skipped = [m for m in MONTHS if m in wanted and month_done(m)]
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
    for month in todo:
        print(
            f"\n== {month} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z)", flush=True
        )
        run_path(month).unlink(missing_ok=True)
        tm = time.monotonic()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        records = []
        for offset in OFFSETS:
            out, record = fit_one(frame, month, offset)
            out.to_parquet(seed_path(month, offset), index=False)
            keep = ~out["flagged"].to_numpy(dtype=bool)
            print(
                f"  {month} offset {offset} (random_state {record['random_state']}): "
                f"prepare {record['prepare_s']:.0f} s, fit {record['fit_s']:.0f} s, "
                f"predict {record['predict_s']:.0f} s, ex-flagged "
                f"{rm.rmse(out['residual'].to_numpy()[keep]):.2f} s, "
                f"peak {record['peak_rss_gib']:.1f} GiB",
                flush=True,
            )
            records.append(record)
        run = {
            "month": month,
            "started_utc": started,
            "month_s": time.monotonic() - tm,
            "frame_build_s": frame.build_seconds,
            "fits": records,
        }
        run_path(month).write_text(json.dumps(run, indent=1), encoding="utf-8")
        moved = rm.touched(before, fingerprint())
        if moved:
            print("TOUCHED: " + ", ".join(moved), file=sys.stderr)
            return 1
    print(
        f"\n{len(todo)} month(s) in {(time.monotonic() - t0) / 60:.1f} min; nothing touched"
    )
    print("DONE", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--fit", action="store_true")
    ap.add_argument("--months", nargs="*")
    args = ap.parse_args(argv)
    return check() if args.check else fit(args.months)


if __name__ == "__main__":
    raise SystemExit(main())
