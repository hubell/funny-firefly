#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The whole ladder, every month of 2025 held out in turn: ``reports/monthly_ladder.md``.

``scripts/report_monthly.py`` held each month out and fitted the submitted
rung. This script fits **every rung of** ``taxiout.models.LADDER``, in ladder
order, the reference rungs and the one diagnostic rung included, on the same
twelve splits: each month of 2025 held out, the other eleven training. The
design is **not forward in time**. January cannot be validated forward, and
every held-out month trains on months after it as well as before; the
December and July forward legs of ``reports/baseline.md`` stay the proxies for
the two test months.

**It changes nothing that ships, chooses nothing, and decides nothing.** It
calls ``run_baselines.run`` itself, one split at a time, with
``keep_residuals`` on, and passes it nothing but a training mask and a
validation mask. Inside ``run`` and unchanged: the record-quality term
refitted on the split's training months, the shrunk encoding recomputed
inside the split, every rung fitted on ``taxiout.models.fit_rows`` at the
ladder's seed under ``BOOSTER_PARAMS``, and ``write_residuals`` writing each
boosted rung's validation residuals. ``run`` writes into the module's
``RESIDUALS`` directory, which is the ladder's own; for the length of one call
that name is pointed at ``data/processed/monthly_ladder/`` and restored, so
the files land as ``<month>_<key>.parquet`` and the ladder's are never
touched. The four reference rungs, which ``run`` scores and does not store,
are written beside them in the same shape.

Per month, beside the rung files, ``<month>_split.parquet`` carries the two
things only the fit knows: the record-quality term's prediction on each
validation row (NaN where it does not apply) and the row's pair count in that
split's training rows, as ``section_frequency`` counts it. ``<month>_run.json``
is written last and marks the month complete, so a stopped run resumes at the
next month.

Two reproduction checks stop the fit when they fail:

1. **December is the ladder.** Held-out December trains on January to November,
   which is the ladder's December split. Every boosted rung's residuals must
   equal ``data/processed/residuals/december_<key>.parquet`` to a largest
   absolute difference of zero, joined on the movement identifier; every rung
   of ``reports/baseline.md``'s December ladder table, the four reference
   rungs among them, must print its *excluding flagged rows* figure.
2. **Every month is ``scripts/report_monthly.py``'s.** Each held-out
   month's ``plan-diffs``
   residuals must equal ``data/processed/monthly/heldout-<month>_plan-diffs.parquet``
   to zero.

Modes::

    --dry-run                 the twelve splits, their rows, the rungs; fits nothing
    --fit [--months M ...]    fit the months not yet on disk
    --report --out PATH       read the stored residuals and the frame; fits nothing

The report carries aggregates only, and the generator refuses to write a
movement identifier or a literal stand identifier.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER,
    LADDER_BY_KEY,
    SUBMISSION_KEY,
    TAIL_EXCLUSION_ADOPTED_H,
    base_of,
)
from taxiout.validation import Split  # noqa: E402

WORK = REPO / "data" / "processed" / "monthly_ladder"
# The ladder's own residuals, named before anything can repoint the module's.
LADDER_RESIDUALS = REPO / "data" / "processed" / "residuals"
if rb.RESIDUALS != LADDER_RESIDUALS:
    raise SystemExit("run_baselines.RESIDUALS is not the ladder's directory")
# The held-out-month fits' files of the submitted rung, written by
# ``scripts/report_monthly.py`` and read by check 2.
MONTHLY = REPO / "data" / "processed" / "monthly"
# ``reports/baseline.md``, named through its own generator: check 9 reads a
# report path assigned at module level here as a report this script writes.
BASELINE = rb.OUT
DEFAULT_OUT = REPO / "reports" / "monthly_ladder.md"

# Never written: checked rather than promised, before and after every mode.
FORBIDDEN = (
    REPO / "reports",
    LADDER_RESIDUALS,
    REPO / "data" / "raw" / "metar",
    MONTHLY,
)

MONTHS: tuple[str, ...] = rm.MONTHS
# The month whose split is the ladder's December split.
LADDER_MONTH = "2025-12"
LADDER_SPLIT = "december"
SEED = rm.SEED
KEYS: tuple[str, ...] = tuple(r.key for r in LADDER)
BOOSTED: tuple[str, ...] = tuple(r.key for r in LADDER if r.kind == "booster")
REFERENCE: tuple[str, ...] = tuple(r.key for r in LADDER if r.kind != "booster")

# A recorded taxi-out over this many hours is a day-shift: out of every fit,
# scored by the scorer's rule. The second reading drops these rows.
DAY_SHIFT_S = float(TAIL_EXCLUSION_ADOPTED_H) * 3600.0
# A cell under this many rows prints its count only.
MIN_CELL_ROWS = 30

SPLIT_SUFFIX = "split"
RUN_SUFFIX = "run.json"
RESIDUAL_COLUMNS = rm.RESIDUAL_COLUMNS
OFFSET_COLUMN = "residual_offset"


def shown(path: Path) -> str:
    """A path as the repository names it, or as it is when outside it."""
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


def rung_path(month: str, key: str) -> Path:
    return WORK / f"{month}_{key}.parquet"


def split_path(month: str) -> Path:
    return WORK / f"{month}_{SPLIT_SUFFIX}.parquet"


def run_path(month: str) -> Path:
    return WORK / f"{month}_{RUN_SUFFIX}"


def month_files(month: str) -> list[Path]:
    return [rung_path(month, k) for k in KEYS] + [split_path(month), run_path(month)]


def month_done(month: str) -> bool:
    return all(p.exists() for p in month_files(month))


# --------------------------------------------------------------------------
# the splits


def split_plan() -> list[tuple[str, tuple[str, ...]]]:
    """(held-out month, training months), twelve of them; months only."""
    plan = [(m, tuple(o for o in MONTHS if o != m)) for m in MONTHS]
    for month, train in plan:
        if month in train or len(train) != len(MONTHS) - 1:
            raise RuntimeError(f"{month}: the split is not one month against eleven")
    return plan


def make_split(month_of_row: pd.Series, month: str) -> Split:
    train_months = dict(split_plan())[month]
    return Split(
        name=month,
        description=f"train the other eleven months of 2025, validate {month}",
        train=month_of_row.isin(train_months),
        valid=month_of_row == month,
    )


# --------------------------------------------------------------------------
# what this script may not write


def fingerprint() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


touched = rm.touched


@contextlib.contextmanager
def residuals_into(directory: Path):
    """Point ``run_baselines.RESIDUALS`` at ``directory`` for one call."""
    saved = rb.RESIDUALS
    rb.RESIDUALS = directory
    try:
        yield
    finally:
        rb.RESIDUALS = saved
    if rb.RESIDUALS != LADDER_RESIDUALS:
        raise RuntimeError("run_baselines.RESIDUALS was not restored")


class RecordingTails(rb.Tails):
    """``run_baselines.Tails``, keeping the term ``run`` fits on each split."""

    def __init__(self) -> None:
        super().__init__()
        self.fitted: list = []

    def fit(self, training):
        model = super().fit(training)
        self.fitted.append(model)
        return model


# --------------------------------------------------------------------------
# check 1: the December ladder table of reports/baseline.md


def baseline_december() -> dict[str, dict[str, str]]:
    """``reports/baseline.md``'s December ladder table, by rung label, as printed."""
    text = BASELINE.read_text(encoding="utf-8").splitlines()
    start = text.index("## The ablation ladder")
    at = text.index("### December split", start)
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in text[at + 1 :]:
        if line.startswith("#"):
            break
        if not line.startswith("|"):
            if header is not None and rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif not set(cells[0]) <= {"-", ":", " "}:
            rows[cells[0]] = dict(zip(header, cells))
    return rows


EX_FLAGGED_HEADER = "excluding flagged rows (s)"
RMSE_HEADER = "RMSE (s)"
RQ_HEADER = "+ record quality (s)"
OFFSET_HEADER = "+ airport offset (s)"


def check_december(log=print) -> bool:
    """Check 1, read from disk: every boosted file and every printed figure."""
    ok_all = True
    for key in BOOSTED:
        mine = pd.read_parquet(
            rung_path(LADDER_MONTH, key), columns=["mvt_id", "residual"]
        )
        got, ok = rm.reproduces(mine, LADDER_RESIDUALS / f"{LADDER_SPLIT}_{key}.parquet")
        log(
            f"    [{'ok ' if ok else 'STOP'}] {key} against {LADDER_SPLIT}_{key}.parquet: {got}"
        )
        ok_all &= ok
    table = baseline_december()
    for rung in LADDER:
        if rung.diagnostic:
            continue
        printed = table.get(rung.label)
        if printed is None:
            log(f"    [STOP] {rung.key}: no row labelled {rung.label!r} in baseline.md")
            ok_all = False
            continue
        frame = pd.read_parquet(rung_path(LADDER_MONTH, rung.key))
        keep = ~frame["flagged"].to_numpy(dtype=bool)
        mine = f"{rm.rmse(frame['residual'].to_numpy()[keep]):,.1f}"
        ok = mine == printed[EX_FLAGGED_HEADER]
        line = f"    [{'ok ' if ok else 'STOP'}] {rung.key}: excluding flagged {mine} s, baseline.md {printed[EX_FLAGGED_HEADER]} s"
        if OFFSET_COLUMN in frame.columns and frame[OFFSET_COLUMN].notna().any():
            off = f"{rm.rmse(frame[OFFSET_COLUMN]):,.1f}"
            ok_off = off == printed[OFFSET_HEADER]
            line += f"; airport offset {off} s, baseline.md {printed[OFFSET_HEADER]} s"
            ok &= ok_off
        log(line if ok else line.replace("[ok ]", "[STOP]"))
        ok_all &= ok
    return ok_all


def check_monthly(month: str, log=print) -> tuple[str, bool]:
    """Check 2: this month's submitted rung against ``report_monthly.py``'s file."""
    mine = pd.read_parquet(
        rung_path(month, SUBMISSION_KEY), columns=["mvt_id", "residual"]
    )
    got, ok = rm.reproduces(mine, MONTHLY / f"heldout-{month}_{SUBMISSION_KEY}.parquet")
    log(
        f"    [{'ok ' if ok else 'STOP'}] {SUBMISSION_KEY} against report_monthly.py's heldout-{month}: {got}"
    )
    return got, ok


# --------------------------------------------------------------------------
# one month


def fit_month(frame: rm.Frame, month: str) -> dict:
    """Every rung on one held-out month, by ``run_baselines.run`` itself."""
    t0 = time.monotonic()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    split = make_split(frame.month, month)
    tails = RecordingTails()
    with residuals_into(WORK):
        results = rb.run(
            frame.features, frame.y, frame.dep, [split], tails, keep_residuals=True
        )
    if len(tails.fitted) != 1 or tails.fitted[0] is None:
        raise RuntimeError(f"{month}: the record-quality term was not fitted once")
    tail_model = tails.fitted[0]
    tail = tail_model.predict(frame.dep)
    flagged = tail.notna()
    mask = split.valid
    raw = frame.dep.loc[mask]
    y = np.asarray(frame.y[mask], dtype=float)
    if not np.isfinite(y).all():
        raise RuntimeError(f"{month}: a validation row has no recorded taxi time")

    # The reference rungs, which ``run`` scores and does not store, in
    # ``write_residuals``' shape plus the airport-offset residual.
    for key in REFERENCE:
        entry = results[(month, key, "valid")]
        shifted = entry["info"].get("offset_preds")
        out = pd.DataFrame(
            {
                "mvt_id": raw["MVT_ID_mvt"].to_numpy(),
                "icao": raw["ADEP_mvt"].astype(str).to_numpy(),
                "runway": raw["RUNWAY_mvt"].astype(str).to_numpy(),
                "takeoff": pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).to_numpy(),
                "residual": y - np.asarray(entry["pred"], dtype=float),
                "flagged": np.asarray(flagged[mask], dtype=bool),
                OFFSET_COLUMN: (
                    y - np.asarray(shifted[0], dtype=float)
                    if shifted
                    else np.full(len(y), np.nan)
                ),
            }
        )
        out.to_parquet(rung_path(month, key), index=False)

    reference = results[(month, "pru", "valid")]["info"]["reference"]
    pd.DataFrame(
        {
            "mvt_id": raw["MVT_ID_mvt"].to_numpy(),
            "term": np.asarray(tail[mask], dtype=float),
            "pair_count": reference.pair_counts(frame.features[mask]).to_numpy(),
        }
    ).to_parquet(split_path(month), index=False)

    missing = [k for k in KEYS if not rung_path(month, k).exists()]
    if missing:
        raise RuntimeError(f"{month}: no residual file for {', '.join(missing)}")
    # Every file must hold this month's validation rows, in the frame's order.
    ids = raw["MVT_ID_mvt"].to_numpy()
    for key in KEYS:
        got = pd.read_parquet(rung_path(month, key), columns=["mvt_id"])[
            "mvt_id"
        ].to_numpy()
        if len(got) != len(ids) or not np.array_equal(got, ids):
            raise RuntimeError(f"{month}_{key}: rows differ from the validation rows")

    record = {
        "month": month,
        "started_utc": started,
        "train_months": list(dict(split_plan())[month]),
        "train_rows": int(split.train.sum()),
        "valid_rows": int(mask.sum()),
        "fit_rows": int(rb.fitting_rows(split.train, flagged, frame.y).sum()),
        "valid_term_rows": int(flagged[mask].sum()),
        "covered_cells": int(tail_model.cells["covered"].sum()),
        "rung_seconds": {
            key: float(results[(month, key, "valid")]["seconds"]) for key in KEYS
        },
        "month_s": time.monotonic() - t0,
        "peak_rss_gib": rm.peak_gib(),
        "seed_offset": SEED,
        "random_state": int(BOOSTER_PARAMS["random_state"]) + SEED,
    }
    del results
    return record


# --------------------------------------------------------------------------
# the modes


def dry_run() -> int:
    before = fingerprint()
    dep = rb.load_departures(None)
    month = rm.month_of(dep["MVT_TIME_UTC_mvt"])
    outside = int((~month.isin(MONTHS)).sum())
    print(f"{len(dep):,} departures; {outside:,} outside the twelve months")
    print("\nthe twelve splits, and nothing fitted:")
    total = 0
    for held, train in split_plan():
        split = make_split(month, held)
        n_valid = int(split.valid.sum())
        total += n_valid
        print(
            f"  {held}  train {len(train):>2d} months ({int(split.train.sum()):>9,} rows)"
            f"  validate {n_valid:>7,} rows"
            f"  {'on disk' if month_done(held) else 'to fit'}"
        )
    print(f"\nheld-out validation rows summed: {total:,} of {len(dep):,}")
    print(f"\nthe {len(LADDER)} rungs, in ladder order ({len(BOOSTED)} boosted):")
    for i, rung in enumerate(LADDER, 1):
        base = base_of(rung)
        print(
            f"  {i:>2d}  {rung.key:<30s} {rung.kind:<12s}"
            f" read against {base.key if base else '-':<30s}"
            f"{' diagnostic' if rung.diagnostic else ''}"
        )
    print(
        f"\nseed offset {SEED}, random_state {int(BOOSTER_PARAMS['random_state']) + SEED}; "
        f"writes only to {shown(WORK)}"
    )
    moved = touched(before, fingerprint())
    print("\n" + ("nothing was touched" if not moved else f"TOUCHED: {moved}"))
    return 0 if not moved and total + outside == len(dep) else 1


def fit(months: list[str] | None) -> int:
    before = fingerprint()
    wanted = list(months or MONTHS)
    unknown = sorted(set(wanted) - set(MONTHS))
    if unknown:
        raise SystemExit(f"not a month of 2025: {', '.join(unknown)}")
    todo = [m for m in MONTHS if m in wanted and not month_done(m)]
    skipped = [m for m in MONTHS if m in wanted and month_done(m)]
    print(
        f"writing only to {shown(WORK)}; "
        f"{len(before)} files under "
        + ", ".join(shown(p) for p in FORBIDDEN)
        + " checked untouched",
        flush=True,
    )
    if skipped:
        print(f"on disk, skipped: {', '.join(skipped)}", flush=True)
    if not todo:
        print("nothing to fit\nDONE", flush=True)
        return 0
    # The ladder check runs first whenever December is to be fitted.
    if LADDER_MONTH in todo:
        todo.remove(LADDER_MONTH)
        todo.insert(0, LADDER_MONTH)
    elif not month_done(LADDER_MONTH):
        raise SystemExit(f"{LADDER_MONTH} is the ladder check and must be fitted first")
    t0 = time.monotonic()
    frame = rm.Frame()
    WORK.mkdir(parents=True, exist_ok=True)
    stands = WORK / "stands.json"
    if not stands.exists():
        stands.write_text(
            json.dumps(
                sorted(
                    {
                        str(v).strip()
                        for v in frame.dep["STAND_mvt"].dropna().astype(str).unique()
                    }
                )
            ),
            encoding="utf-8",
        )
    print(f"frame built in {frame.build_seconds:.0f} s", flush=True)
    for month in todo:
        print(
            f"\n== {month} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z)", flush=True
        )
        for p in month_files(month):
            p.unlink(missing_ok=True)
        record = fit_month(frame, month)
        record["frame_build_s"] = frame.build_seconds
        print(
            f"  {month}: {len(KEYS)} rungs in {record['month_s'] / 60:.1f} min "
            f"(rungs sum {sum(record['rung_seconds'].values()) / 60:.1f} min), "
            f"peak {record['peak_rss_gib']:.1f} GiB",
            flush=True,
        )
        if month == LADDER_MONTH:
            print("  check 1, the ladder's December split:", flush=True)
            ok1 = check_december(lambda s: print(s, flush=True))
            record["check_ladder"] = ok1
            if not ok1:
                run_path(month).write_text(json.dumps(record, indent=1), encoding="utf-8")
                run_path(month).rename(WORK / f"{month}_FAILED_{RUN_SUFFIX}")
                print(
                    "check 1 failed: the harness is not the ladder's; stopping",
                    file=sys.stderr,
                )
                return 1
        print("  check 2, the held-out-month fit:", flush=True)
        got, ok2 = check_monthly(month, lambda s: print(s, flush=True))
        record["check_monthly"] = {"ok": ok2, "read": got}
        if not ok2:
            run_path(month).write_text(json.dumps(record, indent=1), encoding="utf-8")
            run_path(month).rename(WORK / f"{month}_FAILED_{RUN_SUFFIX}")
            print("check 2 failed: a harness defect; stopping", file=sys.stderr)
            return 1
        # Written last: the month is complete.
        run_path(month).write_text(json.dumps(record, indent=1), encoding="utf-8")
        moved = touched(before, fingerprint())
        if moved:
            print("TOUCHED: " + ", ".join(moved), file=sys.stderr)
            return 1
    print(
        f"\n{len(todo)} month(s) in {(time.monotonic() - t0) / 60:.1f} min; nothing touched"
    )
    print("DONE", flush=True)
    return 0


# --------------------------------------------------------------------------
# reading what was stored


HISTORY = REPO / "data" / "raw" / "metar_history"
MODEL_METAR = REPO / "data" / "raw" / "metar"
# The calendar year the model's own cache holds, and the ten years before it.
YEAR = 2025
HISTORY_YEARS: tuple[int, ...] = tuple(range(2015, 2025))
OUT_OF_RANGE_FLAGS = ("freezing", "low visibility", "precipitation")
# ``taxiout.metar``'s visibility band at which low-visibility procedures apply.
LOW_VIS_BAND = 2.0


class Store:
    """Every stored residual of the twelve months, and the row attributes the
    annexes read, joined from the frame on the movement identifier."""

    def __init__(self, months: tuple[str, ...]) -> None:
        missing = [m for m in months if not month_done(m)]
        if missing:
            raise SystemExit(f"not on disk under {WORK}: {', '.join(missing)}; run --fit")
        self.months = months
        self.records = {
            m: json.loads(run_path(m).read_text(encoding="utf-8")) for m in months
        }
        parts, resid, offsets = [], [], {k: [] for k in REFERENCE}
        for i, m in enumerate(months):
            first = pd.read_parquet(rung_path(m, KEYS[0]))
            ids = first["mvt_id"].to_numpy()
            cols = []
            for key in KEYS:
                f = pd.read_parquet(rung_path(m, key))
                if not np.array_equal(f["mvt_id"].to_numpy(), ids):
                    raise SystemExit(f"{m}_{key}: rows differ from {m}_{KEYS[0]}")
                if not np.array_equal(
                    f["flagged"].to_numpy(), first["flagged"].to_numpy()
                ):
                    raise SystemExit(f"{m}_{key}: flags differ from {m}_{KEYS[0]}")
                cols.append(f["residual"].to_numpy(dtype=float))
                if key in offsets:
                    offsets[key].append(f[OFFSET_COLUMN].to_numpy(dtype=float))
            split = pd.read_parquet(split_path(m))
            if not np.array_equal(split["mvt_id"].to_numpy(), ids):
                raise SystemExit(f"{m}_{SPLIT_SUFFIX}: rows differ")
            part = first[["mvt_id", "icao", "runway", "takeoff", "flagged"]].copy()
            part["month"] = i
            part["term"] = split["term"].to_numpy(dtype=float)
            part["pair_count"] = split["pair_count"].to_numpy(dtype=float)
            parts.append(part)
            resid.append(np.column_stack(cols))
        rows = pd.concat(parts, ignore_index=True)
        self.R = np.vstack(resid)
        self.offset = {k: np.concatenate(v) for k, v in offsets.items()}
        del resid, parts
        if not np.isfinite(self.R).all():
            raise SystemExit("a stored residual is not finite")
        if rows["mvt_id"].duplicated().any():
            raise SystemExit("a movement is held out in two months")
        self.n = len(rows)
        self.month = rows["month"].to_numpy()
        self.icao = rows["icao"].astype(str).to_numpy()
        self.runway = rows["runway"].astype(str).to_numpy()
        self.hour = pd.to_datetime(rows["takeoff"], utc=True).dt.hour.to_numpy()
        self.flagged = rows["flagged"].to_numpy(dtype=bool)
        self.term = rows["term"].to_numpy()
        self.pair_count = rows["pair_count"].to_numpy()
        self.mvt_id = rows["mvt_id"].to_numpy()
        if not np.array_equal(np.isfinite(self.term), self.flagged):
            raise SystemExit("the term's rows are not the flagged rows")
        self._join_frame()

    def _join_frame(self) -> None:
        """y, the aircraft, the plan record, the geometry and the weather, from
        the frame the fit read, built here exactly as ``--fit`` builds it."""
        from taxiout.tails import flag

        t0 = time.monotonic()
        frame = rm.Frame()
        dep_ids = frame.dep["MVT_ID_mvt"].to_numpy()
        if pd.Series(dep_ids).duplicated().any():
            raise SystemExit("a movement identifier repeats in the frame")
        pos = pd.Series(np.arange(len(dep_ids)), index=dep_ids).reindex(self.mvt_id)
        if pos.isna().any():
            raise SystemExit(
                f"{int(pos.isna().sum()):,} stored rows are not in the frame"
            )
        at = pos.to_numpy(dtype=int)
        self.y = np.asarray(frame.y, dtype=float)[at]
        feats = frame.features
        self.wake = (
            feats["wake_class"]
            .astype(object)
            .where(feats["wake_class"].notna(), "none")
            .astype(str)
            .to_numpy()[at]
        )
        self.route_m_year = pd.to_numeric(feats["route_m"], errors="coerce").to_numpy(
            dtype=float
        )
        self.route_m = self.route_m_year[at]
        self.freezing = pd.to_numeric(feats["wx_freezing"], errors="coerce").to_numpy(
            dtype=float
        )[at]
        self.vis_band = pd.to_numeric(feats["wx_vis_band"], errors="coerce").to_numpy(
            dtype=float
        )[at]
        self.precip = pd.to_numeric(feats["wx_precip"], errors="coerce").to_numpy(
            dtype=float
        )[at]
        self.no_plan = flag(frame.dep)["flt_missing"].to_numpy(dtype=bool)[at]
        self.stands = {
            str(v).strip() for v in frame.dep["STAND_mvt"].dropna().astype(str).unique()
        }
        self.frame_s = time.monotonic() - t0
        del frame, feats
        if not np.isfinite(self.y).all():
            raise SystemExit("a stored row has no recorded taxi time")


class Agg:
    """Squared-error sums per (cell, month, rung), pooled by adding them.

    Every figure of the report is a root of one of these sums over its count:
    a margin over months or over airports adds the sums of the rows it covers
    and never averages a figure.
    """

    def __init__(self, st: Store) -> None:
        self.st = st
        self.K = len(KEYS)
        self.M = len(st.months)
        self.ex = ~st.flagged
        self.d4 = self.ex & (st.y <= DAY_SHIFT_S)
        self.over4 = self.ex & ~self.d4
        # The every-row residual with the term substituted on the rows it covers:
        # the term's prediction replaces the rung's, so it is one column for all.
        self.term_resid = st.y - np.where(st.flagged, st.term, 0.0)
        self.R2 = st.R * st.R
        withterm = np.where(st.flagged[:, None], self.term_resid[:, None], st.R)
        self.WT2 = withterm * withterm
        del withterm

    def grid(self, code: np.ndarray, G: int) -> dict:
        """Sums over a partition ``code`` (0..G-1) by month; index M is the year."""
        code = np.asarray(code)
        if code.shape != (self.st.n,) or code.min() < 0 or code.max() >= G:
            raise ValueError("a row is outside the partition")
        key = code * self.M + self.st.month
        size = G * self.M

        def count(mask):
            return np.bincount(key[mask], minlength=size).reshape(G, self.M)

        def sums(mask, values):
            k = key[mask]
            return np.stack(
                [
                    np.bincount(k, weights=values[mask, j], minlength=size).reshape(
                        G, self.M
                    )
                    for j in range(self.K)
                ],
                axis=-1,
            )

        R, R2 = self.st.R, self.R2
        every = np.ones(self.st.n, dtype=bool)
        out = {
            "n_all": count(every),
            "n_ex": count(self.ex),
            "n_d4": count(self.d4),
            "n_term": count(self.st.flagged),
            "s2_ex": sums(self.ex, R2),
            "s1_ex": sums(self.ex, R),
            "s2_d4": sums(self.d4, R2),
            "s1_d4": sums(self.d4, R),
            "s2_all": sums(every, R2),
            "s2_term": sums(every, self.WT2),
        }
        # The year: the sums of the twelve months, never a mean of their roots.
        for name, a in out.items():
            out[name] = np.concatenate([a, a.sum(axis=1, keepdims=True)], axis=1)
        return out


def _root(s2, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.sqrt(s2 / np.where(n > 0, n, 1)), np.nan)


def readings(g: dict, k: int) -> dict:
    """Every reading of rung ``k`` over a grid, shape (G, M + 1)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "ex": _root(g["s2_ex"][..., k], g["n_ex"]),
            "d4": _root(g["s2_d4"][..., k], g["n_d4"]),
            "all": _root(g["s2_all"][..., k], g["n_all"]),
            "term": _root(g["s2_term"][..., k], g["n_all"]),
            "bias": np.where(
                g["n_ex"] > 0, -g["s1_ex"][..., k] / np.maximum(g["n_ex"], 1), np.nan
            ),
            "bias_d4": np.where(
                g["n_d4"] > 0, -g["s1_d4"][..., k] / np.maximum(g["n_d4"], 1), np.nan
            ),
        }


# --------------------------------------------------------------------------
# formatting


fmt = rm.fmt
month_label = rm.month_label


def signed1(x: float) -> str:
    return rm.signed(x, 1)


def cell_text(primary: float, second: float, n_ex: int, left: int, render=fmt) -> str:
    """One cell: the primary reading, and beside it the 4 h diagnostic.

    A cell under ``MIN_CELL_ROWS`` rows prints its count only. Where the cell
    holds rows over 4 h, the second reading follows in brackets with the count
    it leaves out; where it holds none, the two readings are the same number
    and it is printed once.
    """
    if n_ex < MIN_CELL_ROWS:
        return f"({n_ex:,} rows)"
    if left > 0:
        return f"{render(primary)} [{render(second)}; {left:,} out]"
    return render(primary)


def comparisons() -> list[tuple[int, int]]:
    """(rung, the rung it is read against), as ``base_of`` pairs them."""
    out = []
    for rung in LADDER:
        base = base_of(rung)
        if base is not None:
            out.append((KEYS.index(rung.key), KEYS.index(base.key)))
    return out


def label(k: int) -> str:
    rung = LADDER_BY_KEY[KEYS[k]]
    return f"`{rung.label}`" + (" (diagnostic)" if rung.diagnostic else "")


def short_pair(k: int, b: int) -> str:
    return f"{label(k)} against `{LADDER_BY_KEY[KEYS[b]].label}`"


GRID_LEGEND = (
    "Each cell is the RMSE, in seconds, of the rows it names, excluding the rows "
    "the record-quality term covers, as the scorer's rule implies. **Where a cell "
    "holds rows whose recorded taxi-out is over 4 h, a second reading follows in "
    "brackets: the same RMSE over the rows of 4 h or less, a diagnostic, with the "
    "count of rows it leaves out.** A cell under "
    f"{MIN_CELL_ROWS} rows prints its count only. The *year* column pools the "
    "squared errors of the twelve months' rows; no figure is a mean of figures."
)


def level_grid(g: dict, k: int, labels: list[str], what: str = "ex") -> list[list[str]]:
    r = readings(g, k)
    out = []
    for i, name in enumerate(labels):
        row = [name]
        for m in range(g["n_ex"].shape[1]):
            n_ex = int(g["n_ex"][i, m])
            left = int(n_ex - g["n_d4"][i, m])
            if what == "bias":
                row.append(
                    cell_text(
                        r["bias"][i, m],
                        r["bias_d4"][i, m],
                        n_ex,
                        left,
                        lambda x: rm.signed(x, 0),
                    )
                )
            else:
                row.append(cell_text(r["ex"][i, m], r["d4"][i, m], n_ex, left))
        out.append(row)
    return out


def change_grid(g: dict, k: int, b: int, labels: list[str]) -> list[list[str]]:
    rk, rb_ = readings(g, k), readings(g, b)
    out = []
    for i, name in enumerate(labels):
        row = [name]
        for m in range(g["n_ex"].shape[1]):
            n_ex = int(g["n_ex"][i, m])
            left = int(n_ex - g["n_d4"][i, m])
            row.append(
                cell_text(
                    rk["ex"][i, m] - rb_["ex"][i, m],
                    rk["d4"][i, m] - rb_["d4"][i, m],
                    n_ex,
                    left,
                    signed1,
                )
            )
        out.append(row)
    return out


def count_grid(g: dict, labels: list[str]) -> list[list[str]]:
    out = []
    for i, name in enumerate(labels):
        row = [name]
        for m in range(g["n_ex"].shape[1]):
            n_ex, n_d4, n_t = (
                int(g["n_ex"][i, m]),
                int(g["n_d4"][i, m]),
                int(g["n_term"][i, m]),
            )
            row.append(f"{n_ex:,} / {n_ex - n_d4:,} / {n_t:,}")
        out.append(row)
    return out


def stack(*grids: dict) -> dict:
    """Grids over the same months, one above the other (a total row below)."""
    return {k: np.concatenate([g[k] for g in grids], axis=0) for k in grids[0]}


# --------------------------------------------------------------------------
# the partitions; each assigns every row to exactly one cell


def partition(values, order: list) -> np.ndarray:
    """Codes 0..len(order)-1 for ``values``; a value outside ``order`` is an error."""
    index = {v: i for i, v in enumerate(order)}
    codes = np.array([index.get(v, -1) for v in values], dtype=np.int64)
    if (codes < 0).any():
        raise ValueError(f"{int((codes < 0).sum()):,} rows fall in no cell")
    return codes


def coded(values: np.ndarray) -> tuple[np.ndarray, list]:
    """A partition by the distinct values of ``values``, sorted."""
    order, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int64), list(order)


def flag_partition(
    values: np.ndarray, levels: list[tuple[float, str]]
) -> tuple[np.ndarray, list[str]]:
    """A weather flag as the model reads it: each level, then missing."""
    codes = np.full(len(values), len(levels), dtype=np.int64)
    for i, (v, _) in enumerate(levels):
        codes[values == v] = i
    names = [name for _, name in levels] + ["missing"]
    return codes, names


def distance_bands(
    route_year: np.ndarray, route: np.ndarray
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Quartiles of the training year's routed distance, fixed once."""
    q = np.quantile(route_year[np.isfinite(route_year)], [0.25, 0.5, 0.75])
    names = [
        f"Q1, under {q[0]:,.0f} m",
        f"Q2, {q[0]:,.0f} to {q[1]:,.0f} m",
        f"Q3, {q[1]:,.0f} to {q[2]:,.0f} m",
        f"Q4, {q[2]:,.0f} m or more",
        "no routed distance",
    ]
    codes = np.where(np.isfinite(route), np.searchsorted(q, route, side="right"), 4)
    return codes.astype(np.int64), names, q


# --------------------------------------------------------------------------
# the historic weather context


def monthly_shares(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per station, year and calendar month: the share of hours in each regime."""
    h = hourly.copy()
    hour = pd.to_datetime(h["hour"], utc=True)
    h["year"] = hour.dt.year
    h["cal"] = hour.dt.month
    h["low_vis"] = np.where(
        h["vis_band"].notna(), (h["vis_band"] >= LOW_VIS_BAND).astype(float), np.nan
    )
    g = h.groupby(["station", "year", "cal"])
    return pd.DataFrame(
        {
            "hours": g.size(),
            "freezing": g["freezing"].mean(),
            "low visibility": g["low_vis"].mean(),
            "precipitation": g["precip"].mean(),
        }
    ).reset_index()


def weather_context(stations: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    """2025 against 2015 to 2024, with ``taxiout.metar``'s own functions."""
    from taxiout.metar import load_metar, regime_by_hour

    files = sorted(HISTORY.glob("*.csv"))
    hist = monthly_shares(regime_by_hour(load_metar(HISTORY)))
    now = monthly_shares(regime_by_hour(load_metar(MODEL_METAR)))
    hist = hist[hist["year"].isin(HISTORY_YEARS) & hist["station"].isin(stations)]
    now = now[(now["year"] == YEAR) & now["station"].isin(stations)]
    rows = []
    for icao in stations:
        for cal in range(1, 13):
            h = hist[(hist["station"] == icao) & (hist["cal"] == cal)]
            n = now[(now["station"] == icao) & (now["cal"] == cal)]
            row = {
                "icao": icao,
                "cal": cal,
                "years": int(len(h)),
                "hours_2025": int(n["hours"].sum()),
            }
            for f in OUT_OF_RANGE_FLAGS:
                row[f"{f}:2025"] = float(n[f].iloc[0]) if len(n) else np.nan
                row[f"{f}:mean"] = float(h[f].mean()) if len(h) else np.nan
                row[f"{f}:min"] = float(h[f].min()) if len(h) else np.nan
                row[f"{f}:max"] = float(h[f].max()) if len(h) else np.nan
            rows.append(row)
    return pd.DataFrame(rows), files + sorted(MODEL_METAR.glob("*.csv"))


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f}%"


# --------------------------------------------------------------------------
# the report


def build_report(st: Store) -> tuple[str, list[Path]]:
    agg = Agg(st)
    M = len(st.months)
    mcols = [month_label(m) for m in st.months] + ["year"]
    lines: list[str] = []
    add = lines.append
    ship = KEYS.index(SUBMISSION_KEY)
    pairs = comparisons()
    airports = sorted({a.icao for a in AIRPORTS} & set(st.icao))

    # the partitions
    whole = agg.grid(np.zeros(st.n, dtype=np.int64), 1)
    ap_code = partition(st.icao, airports)
    by_ap = agg.grid(ap_code, len(airports))
    ap_all = stack(by_ap, whole)
    ap_labels = airports + [f"all {len(airports)}"]

    def grid_table(rows, first="cell"):
        return rm.table([first] + mcols, rows, "l" + "r" * (M + 1))

    add("# The ladder, every month held out in turn")
    add("")
    add(
        "Generated by `scripts/report_monthly_ladder.py` with `--report` from the residuals "
        "`--fit` stored under `data/processed/monthly_ladder/`; every figure below is "
        "computed there, and the generator refuses to write a movement identifier "
        "or a literal stand identifier."
    )
    add("")

    # 1 ----------------------------------------------------------------------
    add("## What this is")
    add("")
    add(
        f"Every rung of the ladder, all {len(KEYS)} of `taxiout.models.LADDER` in "
        f"ladder order ({len(REFERENCE)} reference rungs and {len(BOOSTED)} boosted, "
        "one of them the diagnostic `+ stand occupancy, schedule-anchored`, which "
        f"`reports/baseline.md`'s ladder table leaves out), on {M} splits: each month "
        "of 2025 held out in turn, the other eleven training. **The design is not "
        "forward in time.** January cannot be validated forward, the training year's "
        "first month being January, and every held-out month here is fitted on "
        "months after it as well as before. The forward December and July splits of "
        "`reports/baseline.md` remain the proxies for the two test months, and "
        "nothing below replaces them."
    )
    add("")
    add(
        "The fitting is `run_baselines.run` itself, called once per held-out month "
        "with that month as the only split: the record-quality term of "
        "`taxiout.tails` refitted on the eleven training months, the shrunk encoding "
        "recomputed inside the split, every rung fitted on `taxiout.models.fit_rows` "
        f"(training, unflagged, and recorded taxi-outs over {TAIL_EXCLUSION_ADOPTED_H:g} h "
        f"out of the fit) at the ladder's seed (offset {SEED} on "
        '`BOOSTER_PARAMS["random_state"]`) under `taxiout.models.BOOSTER_PARAMS`, '
        "unchanged. No held-out-pair folds were fitted per month; the claim about "
        "unseen pairs stays on the folds `reports/baseline.md` reports."
    )
    add("")
    add(
        "**This report decides nothing.** It changes no parameter, no feature and no "
        "rule, and nothing that ships is fitted from it. It is read for how each "
        "block's contribution is distributed over the months, the airports and the "
        "strata of the year."
    )
    add("")
    add(
        "How to read every table. Every RMSE is the root of the pooled squared "
        "errors of the rows the cell names; a margin over months or over airports "
        "pools the rows it covers and is never a mean of figures. The primary "
        "reading excludes the rows the record-quality term covers, as the scorer's "
        "rule implies. The fit leaves out recorded taxi-outs over 4 h that the term "
        "does not cover, and the scorer counts them, so a single such row can decide "
        "a cell; **beside every such figure a second reading is printed, over the "
        "rows of 4 h or less, with the count of rows it leaves out. It is a "
        "diagnostic and is never quoted without the first.** A cell under "
        f"{MIN_CELL_ROWS} rows prints its count only. A change is a rung's RMSE "
        "minus that of the rung it is read against, over the same rows, so a "
        "negative change is an improvement."
    )
    add("")
    add(
        "The main body is three summaries, each pooled directly from the per-month "
        "residuals. The annexes break the same rows down: A by airport and month, B "
        "by take-off hour and runway, C by pair frequency and route length, D by "
        "aircraft, flight-plan record, the term's rows and the weather regime, E sets "
        "the year's weather against its ten-year norm, F lists the rows over 4 h, and "
        "the last section, *What this report was generated from*, is Annex G."
    )
    add("")
    add("The rungs, and the rung each is read against (`taxiout.models.base_of`):")
    add("")
    rows = []
    for i, rung in enumerate(LADDER, 1):
        base = base_of(rung)
        rows.append(
            [
                str(i),
                f"`{rung.key}`",
                rung.label,
                rung.kind,
                f"`{base.key}`" if base else "",
                "diagnostic"
                if rung.diagnostic
                else ("shipped" if rung.key == SUBMISSION_KEY else ""),
            ]
        )
    lines += rm.table(["", "key", "rung", "kind", "read against", ""], rows, "rlllll")
    add("")

    # 2 ----------------------------------------------------------------------
    add("## The harness is the ladder's")
    add("")
    add(
        "Two checks, each a stop on failure when `--fit` ran, recomputed here from "
        "the files on disk."
    )
    add("")
    add(
        "**Check 1, the ladder's December.** Held-out December trains on January to "
        "November, which is the ladder's December split, so its fit is the ladder's. "
        "Every boosted rung's residuals are joined to "
        "`data/processed/residuals/december_<key>.parquet` on the movement "
        "identifier, every row; every rung of `reports/baseline.md`'s December "
        "table must print its *RMSE*, *+ record quality* and *excluding flagged "
        "rows* figures, and the PRU and geometric rungs their *+ airport offset* "
        "figure."
    )
    add("")
    table_dec = baseline_december()
    rows = []
    ok1 = LADDER_MONTH in st.months
    for k, key in enumerate(KEYS if ok1 else ()):
        rung = LADDER_BY_KEY[key]
        sel = st.month == st.months.index(LADDER_MONTH)
        if key in BOOSTED:
            mine = pd.DataFrame({"mvt_id": st.mvt_id[sel], "residual": st.R[sel, k]})
            got, ok = rm.reproduces(
                mine, LADDER_RESIDUALS / f"{LADDER_SPLIT}_{key}.parquet"
            )
        else:
            got, ok = "not stored by the ladder", True
        ex = rm.rmse(st.R[sel & agg.ex, k])
        row_dec = table_dec.get(rung.label, {})
        printed = row_dec.get(EX_FLAGGED_HEADER, "not in the table")
        every = f"{rm.rmse(st.R[sel, k]):,.1f}"
        with_term = f"{np.sqrt(np.mean(agg.WT2[sel, k])):,.1f}"
        agree = rung.diagnostic or (
            f"{ex:,.1f}" == printed
            and every == row_dec.get(RMSE_HEADER)
            and with_term == row_dec.get(RQ_HEADER)
        )
        off = ""
        if key in st.offset and np.isfinite(st.offset[key][sel]).any():
            o = f"{rm.rmse(st.offset[key][sel]):,.1f}"
            off = f"{o} / {table_dec[rung.label][OFFSET_HEADER]}"
            agree &= o == table_dec[rung.label][OFFSET_HEADER]
        ok1 &= ok and agree
        rows.append(
            [
                label(k),
                got,
                f"{every} / {row_dec.get(RMSE_HEADER, 'not in the table')}",
                f"{with_term} / {row_dec.get(RQ_HEADER, 'not in the table')}",
                f"{ex:,.1f} / {printed}",
                off,
                "yes" if ok and agree else "**no**",
            ]
        )
    lines += rm.table(
        [
            "rung",
            "largest absolute difference",
            "RMSE (s), here / baseline.md",
            "+ record quality (s)",
            "excluding flagged (s)",
            "+ airport offset (s)",
            "reproduces",
        ],
        rows,
        "lllllll",
    )
    add("")
    add(
        "Every rung reproduces the ladder's December, so every month below is fitted "
        "by the ladder's own path."
        if ok1
        else "**A rung did not reproduce the ladder's December; no figure below should be read.**"
    )
    add("")
    add(
        "**Check 2, the held-out-month fits.** Each month's `+ plan differences` residuals "
        "are joined to `data/processed/monthly/heldout-<month>_plan-diffs.parquet`, "
        "the same rung on the same split under the same seed, fitted by "
        "`scripts/report_monthly.py`; and on the rows the term covers, the stored "
        "with-the-term residual there must equal the recorded taxi-out here minus "
        "the term stored here, which checks the join of the frame to the files."
    )
    add("")
    rows = []
    ok2 = True
    for i, m in enumerate(st.months):
        sel = st.month == i
        mine = pd.DataFrame({"mvt_id": st.mvt_id[sel], "residual": st.R[sel, ship]})
        path_monthly = MONTHLY / f"heldout-{m}_{SUBMISSION_KEY}.parquet"
        got, ok = rm.reproduces(mine, path_monthly)
        theirs = pd.read_parquet(
            path_monthly, columns=["mvt_id", rm.TERM_COLUMN, "flagged"]
        )
        joined = pd.DataFrame(
            {
                "mvt_id": st.mvt_id[sel],
                "t": st.y[sel] - st.term[sel],
                "f": st.flagged[sel],
            }
        ).merge(theirs, on="mvt_id", validate="one_to_one")
        flags_ok = bool((joined["f"] == joined["flagged"]).all())
        term_rows = joined[joined["f"]]
        worst = (
            float(np.max(np.abs(term_rows["t"] - term_rows[rm.TERM_COLUMN])))
            if len(term_rows)
            else 0.0
        )
        ok &= flags_ok and worst == 0.0 and len(joined) == int(sel.sum())
        ok2 &= ok
        rows.append(
            [
                m,
                f"{int(sel.sum()):,}",
                got,
                f"{worst:.6g} s over {len(term_rows):,} term rows",
                "yes" if ok else "**no**",
            ]
        )
    lines += rm.table(
        [
            "held-out month",
            "rows",
            "plan differences, largest absolute difference",
            "with the term, recomputed",
            "agrees",
        ],
        rows,
        "lrlll",
    )
    add("")
    add(
        "Every month agrees with `reports/monthly.md`'s fits to the bit."
        if ok2
        else "**A month disagrees with `reports/monthly.md`'s fits: a harness defect, which would move every rung.**"
    )
    add("")

    # 3 ----------------------------------------------------------------------
    add("## Summary: the year, pooled from the twelve months")
    add("")
    add(
        f"Per rung, the {M} held-out months' rows pooled: "
        f"{int(whole['n_all'][0, M]):,} rows, of which {int(whole['n_term'][0, M]):,} "
        f"are covered by the term and {int(whole['n_ex'][0, M]):,} are scored "
        f"excluding them; {int(whole['n_ex'][0, M] - whole['n_d4'][0, M]):,} of those "
        "have a recorded taxi-out over 4 h, and the diagnostic column leaves them out. "
        "*With the term* scores every row with the term substituted on the rows it "
        "covers, which is what the submission does; *booster only* leaves those rows "
        "to the rung."
    )
    add("")
    rows = []
    for k, key in enumerate(KEYS):
        r = readings(whole, k)
        base = base_of(LADDER_BY_KEY[key])
        if base is not None:
            b = KEYS.index(base.key)
            rbase = readings(whole, b)
            change = signed1(r["ex"][0, M] - rbase["ex"][0, M])
            change4 = signed1(r["d4"][0, M] - rbase["d4"][0, M])
        else:
            change = change4 = ""
        rows.append(
            [
                label(k),
                f"{int(whole['n_ex'][0, M]):,}",
                fmt(r["ex"][0, M]),
                fmt(r["d4"][0, M]),
                f"`{base.key}`" if base else "",
                change,
                change4,
                fmt(r["term"][0, M]),
                fmt(r["all"][0, M]),
                f"{int(whole['n_all'][0, M]):,}",
            ]
        )
    lines += rm.table(
        [
            "rung",
            "rows scored",
            "excluding flagged (s)",
            "4 h or less, diagnostic (s)",
            "read against",
            "change (s)",
            "change, 4 h or less (s)",
            "every row, with the term (s)",
            "every row, booster only (s)",
            "rows, every row",
        ],
        rows,
        "lrrrlrrrrr",
    )
    add("")
    shipped_year = readings(whole, ship)
    add(
        f"The shipped rung, `{LADDER_BY_KEY[SUBMISSION_KEY].label}`, pools to "
        f"{fmt(shipped_year['ex'][0, M])} s excluding flagged rows over "
        f"{int(whole['n_ex'][0, M]):,} rows ({fmt(shipped_year['d4'][0, M])} s over the "
        f"{int(whole['n_d4'][0, M]):,} of 4 h or less, a diagnostic), and "
        f"{fmt(shipped_year['term'][0, M])} s over every row with the term."
    )
    add("")

    # 4 ----------------------------------------------------------------------
    add("## Summary: every month")
    add("")
    add(GRID_LEGEND)
    add("")
    add("Rows per month, excluding flagged / of which over 4 h / covered by the term:")
    add("")
    lines += grid_table(count_grid(whole, ["rows"]), "")
    add("")
    add("### Excluding flagged rows, per rung (s)")
    add("")
    lines += grid_table(
        [level_grid(whole, k, [label(k)])[0] for k in range(len(KEYS))], "rung"
    )
    add("")
    best = int(np.nanargmin(shipped_year["ex"][0, :M]))
    worst = int(np.nanargmax(shipped_year["ex"][0, :M]))
    add(
        f"The shipped rung's best month is {month_label(st.months[best])}, "
        f"{fmt(shipped_year['ex'][0, best])} s, and its worst "
        f"{month_label(st.months[worst])}, {fmt(shipped_year['ex'][0, worst])} s "
        f"({fmt(shipped_year['d4'][0, worst])} s over the rows of 4 h or less, a diagnostic)."
    )
    add("")
    add("### Changes against the rung each is read against (s)")
    add("")
    lines += grid_table(
        [change_grid(whole, k, b, [short_pair(k, b)])[0] for k, b in pairs],
        "rung against",
    )
    add("")
    add("### In how many months each block helps")
    add("")
    add(
        "A block helps in a month when its change, excluding flagged rows, prints "
        "below zero at the precision printed (0.1 s); a change that prints 0.0 is a "
        "tie. The last column counts the same over the rows of 4 h or less, a "
        "diagnostic."
    )
    add("")
    rows = []
    helps_by_block: dict[str, list[str]] = {}
    for k, b in pairs:
        rk, rb_ = readings(whole, k), readings(whole, b)
        ch = [round(float(rk["ex"][0, m] - rb_["ex"][0, m]), 1) for m in range(M)]
        ch4 = [round(float(rk["d4"][0, m] - rb_["d4"][0, m]), 1) for m in range(M)]
        helps = [month_label(st.months[m]) for m in range(M) if ch[m] < 0]
        ties = [month_label(st.months[m]) for m in range(M) if ch[m] == 0]
        hurts = [month_label(st.months[m]) for m in range(M) if ch[m] > 0]
        helps_by_block[KEYS[k]] = helps
        rows.append(
            [
                short_pair(k, b),
                f"{len(helps)} of {M}",
                ", ".join(hurts) or "none",
                ", ".join(ties) or "none",
                signed1(rk["ex"][0, M] - rb_["ex"][0, M]),
                f"{sum(c < 0 for c in ch4)} of {M}",
            ]
        )
    lines += rm.table(
        [
            "block",
            "helps in",
            "worse in",
            "tie in",
            "pooled change (s)",
            "helps in, 4 h or less",
        ],
        rows,
        "lrllrr",
    )
    add("")

    # 5 ----------------------------------------------------------------------
    add("## Summary: every airport")
    add("")
    add(
        "The shipped rung per airport, the twelve months pooled. Bias is prediction "
        "minus recorded taxi-out over the same rows as the RMSE beside it."
    )
    add("")
    r = readings(ap_all, ship)
    rows = []
    for i, name in enumerate(ap_labels):
        n_ex, n_d4 = int(ap_all["n_ex"][i, M]), int(ap_all["n_d4"][i, M])
        rows.append(
            [
                name,
                f"{n_ex:,}",
                fmt(r["ex"][i, M]),
                rm.signed(r["bias"][i, M], 0),
                fmt(r["d4"][i, M]),
                rm.signed(r["bias_d4"][i, M], 0),
                f"{n_ex - n_d4:,}",
                f"{int(ap_all['n_term'][i, M]):,}",
                fmt(r["term"][i, M]),
            ]
        )
    lines += rm.table(
        [
            "ICAO",
            "rows scored",
            "excluding flagged (s)",
            "bias (s)",
            "4 h or less, diagnostic (s)",
            "bias, 4 h or less (s)",
            "rows over 4 h",
            "term rows",
            "every row, with the term (s)",
        ],
        rows,
        "lrrrrrrrr",
    )
    add("")
    add(
        "Each block's pooled change per airport, excluding flagged rows (s); the "
        "bracket is the diagnostic over the rows of 4 h or less where an airport has "
        "rows over 4 h:"
    )
    add("")
    rows = []
    for k, b in pairs:
        rk, rb_ = readings(ap_all, k), readings(ap_all, b)
        row = [short_pair(k, b)]
        for i in range(len(ap_labels)):
            n_ex = int(ap_all["n_ex"][i, M])
            row.append(
                cell_text(
                    rk["ex"][i, M] - rb_["ex"][i, M],
                    rk["d4"][i, M] - rb_["d4"][i, M],
                    n_ex,
                    int(n_ex - ap_all["n_d4"][i, M]),
                    signed1,
                )
            )
        rows.append(row)
    lines += rm.table(["block"] + ap_labels, rows, "l" + "r" * len(ap_labels))
    add("")

    # Annex A ----------------------------------------------------------------
    add("## Annex A: airport by month, every rung")
    add("")
    add(GRID_LEGEND)
    add("")
    add(
        "Rows per airport and month: excluding flagged / of which over 4 h / covered "
        "by the term. These are the same for every rung."
    )
    add("")
    lines += grid_table(count_grid(ap_all, ap_labels), "ICAO")
    for k in range(len(KEYS)):
        add("")
        add(f"### A.{k + 1} {LADDER_BY_KEY[KEYS[k]].label}")
        add("")
        add("Excluding flagged rows (s):")
        add("")
        lines += grid_table(level_grid(ap_all, k, ap_labels), "ICAO")
        add("")
        add("Bias, prediction minus recorded, excluding flagged rows (s):")
        add("")
        lines += grid_table(level_grid(ap_all, k, ap_labels, "bias"), "ICAO")
    add("")
    add("### Changes, airport by month")
    for k, b in pairs:
        add("")
        add(f"{short_pair(k, b)}, change in seconds:")
        add("")
        lines += grid_table(change_grid(ap_all, k, b, ap_labels), "ICAO")
    add("")

    # Annex B ----------------------------------------------------------------
    add("## Annex B: hour of day and runway")
    add("")
    add(
        "Per airport, by take-off hour (UTC, 24 bands) and by runway designator, per "
        f"month: the shipped rung, `{LADDER_BY_KEY[SUBMISSION_KEY].label}`, then each "
        "block's change against the rung it is read against. " + GRID_LEGEND
    )
    for a, icao in enumerate(airports):
        at = ap_code == a
        hour_code = np.where(at, st.hour, 24)
        hours = [f"{h:02d}" for h in range(24)]
        g_hour = agg.grid(hour_code, 25)
        g_hour = {k: v[:24] for k, v in g_hour.items()}
        rwys = sorted(set(st.runway[at]))
        rw_code = np.where(
            at, partition(np.where(at, st.runway, rwys[0]), rwys), len(rwys)
        )
        g_rw = agg.grid(rw_code, len(rwys) + 1)
        g_rw = {k: v[: len(rwys)] for k, v in g_rw.items()}
        add("")
        add(f"### B.{a + 1} {icao}")
        add("")
        add("Rows: excluding flagged / of which over 4 h / covered by the term.")
        add("")
        lines += grid_table(count_grid(g_hour, hours), "hour")
        add("")
        lines += grid_table(count_grid(g_rw, rwys), "runway")
        add("")
        add("The shipped rung by hour, excluding flagged rows (s):")
        add("")
        lines += grid_table(level_grid(g_hour, ship, hours), "hour")
        add("")
        add("The shipped rung by runway, excluding flagged rows (s):")
        add("")
        lines += grid_table(level_grid(g_rw, ship, rwys), "runway")
        for k, b in pairs:
            add("")
            add(f"{icao}, {short_pair(k, b)}, change by hour (s):")
            add("")
            lines += grid_table(change_grid(g_hour, k, b, hours), "hour")
            add("")
            add(f"{icao}, {short_pair(k, b)}, change by runway (s):")
            add("")
            lines += grid_table(change_grid(g_rw, k, b, rwys), "runway")
    add("")

    # Annex C ----------------------------------------------------------------
    from taxiout.validation import FREQUENCY_BUCKETS, frequency_bucket

    bucket = frequency_bucket(pd.Series(st.pair_count)).to_numpy()
    b_code = partition(bucket, list(FREQUENCY_BUCKETS))
    b_labels = [f"seen {b} times" for b in FREQUENCY_BUCKETS]
    g_b = agg.grid(b_code, len(FREQUENCY_BUCKETS))
    d_code, d_labels, q = distance_bands(st.route_m_year, st.route_m)
    g_d = agg.grid(d_code, len(d_labels))
    add("## Annex C: pair frequency and route length")
    add("")
    add(
        "Two partitions of each month's rows. **Pair frequency**: the four buckets "
        "`reports/baseline.md` uses, by how many times the row's (airport, stand, "
        "runway) pair appears among the rows that month's split fits on (its own "
        "eleven training months, unflagged, 4 h or less), as the PRU reference of "
        "that split counts them. **Route length**: the OpenStreetMap routed distance "
        "(`route_m`) of the row's stand and runway, in quartiles of the training "
        f"year's routed distance over all {len(st.route_m_year):,} departures of 2025, "
        f"fixed once at {q[0]:,.0f}, {q[1]:,.0f} and {q[2]:,.0f} m. The geometry's change "
        "(`identifiers + time + geometry` against `identifiers + time`) and the three "
        "encoding arms' changes against `+ weather` are the ones read here; every "
        "block is printed. " + GRID_LEGEND
    )
    add("")
    add(reconcile_frequency(st, g_b))
    for title, g, labels_ in (
        ("pair frequency", g_b, b_labels),
        ("route length", g_d, d_labels),
    ):
        add("")
        add(f"### C, {title}")
        add("")
        add("Rows: excluding flagged / of which over 4 h / covered by the term.")
        add("")
        lines += grid_table(count_grid(g, labels_), title)
        add("")
        add("The shipped rung, excluding flagged rows (s):")
        add("")
        lines += grid_table(level_grid(g, ship, labels_), title)
        for k, b in pairs:
            add("")
            add(f"{short_pair(k, b)}, change by {title} (s):")
            add("")
            lines += grid_table(change_grid(g, k, b, labels_), title)
    add("")

    # Annex D ----------------------------------------------------------------
    wk_code, wk_labels = coded(st.wake)
    plan_code = st.no_plan.astype(np.int64)
    plan_labels = ["flight-plan record", "no flight-plan record"]
    fz_code, fz_labels = flag_partition(
        st.freezing, [(0.0, "not freezing"), (1.0, "freezing")]
    )
    vb_code, vb_labels = flag_partition(
        st.vis_band, [(0.0, "8 km or more"), (1.0, "1.5 to 8 km"), (2.0, "below 1.5 km")]
    )
    pr_code, pr_labels = flag_partition(
        st.precip, [(0.0, "no precipitation"), (1.0, "precipitation")]
    )
    add("## Annex D: aircraft, record and regime")
    add("")
    add(
        "Per month, by the wake class the model reads (`wake_class`, `none` where the "
        "flight-side join found no class); by whether the flight-side join found a "
        "flight-plan record (`LOBT_flt` present, as `taxiout.tails.flag` reads it); "
        "by the record-quality term's rows; and by the three weather regime flags as "
        "the model reads them (`wx_freezing`, `wx_vis_band`, `wx_precip`, the "
        "hour's worst reading carried up to three hours; *missing* where the hour has "
        "no reading). " + GRID_LEGEND
    )
    for title, code, labels_ in (
        ("wake class", wk_code, wk_labels),
        ("flight-plan record", plan_code, plan_labels),
        ("freezing", fz_code, fz_labels),
        ("visibility", vb_code, vb_labels),
        ("precipitation", pr_code, pr_labels),
    ):
        g = agg.grid(code, len(labels_))
        add("")
        add(f"### D, {title}")
        add("")
        add("Rows: excluding flagged / of which over 4 h / covered by the term.")
        add("")
        lines += grid_table(count_grid(g, labels_), title)
        add("")
        add("The shipped rung, excluding flagged rows (s):")
        add("")
        lines += grid_table(level_grid(g, ship, labels_), title)
        for k, b in pairs:
            add("")
            add(f"{short_pair(k, b)}, change by {title} (s):")
            add("")
            lines += grid_table(change_grid(g, k, b, labels_), title)
    # the term's rows: no excluding-flagged reading exists there by definition
    g_t = agg.grid(st.flagged.astype(np.int64), 2)
    t_labels = ["rows the term does not cover", "rows the term covers"]
    add("")
    add("### D, the record-quality term's rows")
    add("")
    add(
        "On the rows the term covers there is no reading excluding them, so this "
        "table reads every row: *with the term*, where the term's prediction replaces "
        "the rung's on the rows it covers and is the same for every rung, and "
        "*booster only*, where the rung predicts them. Rows per month:"
    )
    add("")
    rows = [
        [t_labels[i]] + [f"{int(g_t['n_all'][i, m]):,}" for m in range(M + 1)]
        for i in range(2)
    ]
    lines += grid_table(rows, "")
    for reading, title in (("term", "with the term"), ("all", "booster only")):
        add("")
        add(f"The shipped rung, every row, {title} (s):")
        add("")
        rr = readings(g_t, ship)[reading]
        rows = [
            [t_labels[i]]
            + [
                fmt(rr[i, m])
                if g_t["n_all"][i, m] >= MIN_CELL_ROWS
                else f"({int(g_t['n_all'][i, m]):,} rows)"
                for m in range(M + 1)
            ]
            for i in range(2)
        ]
        lines += grid_table(rows, "")
    add("")
    add("Each block's change on the rows the term covers, booster only (s):")
    add("")
    rows = []
    for k, b in pairs:
        rk, rb_ = readings(g_t, k)["all"], readings(g_t, b)["all"]
        rows.append(
            [short_pair(k, b)]
            + [
                signed1(rk[1, m] - rb_[1, m])
                if g_t["n_all"][1, m] >= MIN_CELL_ROWS
                else f"({int(g_t['n_all'][1, m]):,} rows)"
                for m in range(M + 1)
            ]
        )
    lines += grid_table(rows, "block")
    add("")

    # Annex E ----------------------------------------------------------------
    wx, wx_files = weather_context(airports)
    add("## Annex E: the historic weather context")
    add("")
    add(
        "Context only. For each airport and calendar month, the share of hours in "
        f"each regime in {YEAR}, against the mean and the range of the same share "
        f"over {HISTORY_YEARS[0]} to {HISTORY_YEARS[-1]}, from the Iowa Environmental "
        "Mesonet's METAR archive (`data/raw/metar_history/`, fetched by "
        "`scripts/fetch_metar_history.py` with `scripts/fetch_metar.py`'s request, "
        f"for {HISTORY_YEARS[0]} to {HISTORY_YEARS[-1]}; `data/raw/metar/` for {YEAR}). "
        "Hours and flags are `taxiout.metar.regime_by_hour`'s, with its thresholds: "
        "freezing is an air temperature at or below 0 C or a freezing or frozen "
        "present-weather code; low visibility is the band below 1.5 km; "
        "precipitation is any falling present-weather code or a measured "
        "accumulation; an hour without a report inherits the previous one for up "
        "to three hours. A share is over the hours with a reading. **This table is "
        "not a weight, not a feature, and enters no figure of this report.**"
    )
    for icao in airports:
        w = wx[wx["icao"] == icao].set_index("cal")
        add("")
        add(f"### E, {icao}")
        add("")
        rows = []
        outside: list[str] = []
        for cal in range(1, 13):
            row = [datetime(YEAR, cal, 1).strftime("%b"), f"{int(w.loc[cal, 'years'])}"]
            flagged_out = []
            for f in OUT_OF_RANGE_FLAGS:
                now, lo, hi = (
                    w.loc[cal, f"{f}:2025"],
                    w.loc[cal, f"{f}:min"],
                    w.loc[cal, f"{f}:max"],
                )
                row += [pct(now), pct(w.loc[cal, f"{f}:mean"]), f"{pct(lo)} to {pct(hi)}"]
                if np.isfinite(now) and np.isfinite(lo) and (now < lo or now > hi):
                    flagged_out.append(f"{f} {'above' if now > hi else 'below'}")
            if flagged_out:
                outside.append(
                    f"{datetime(YEAR, cal, 1).strftime('%B')} ({', '.join(flagged_out)})"
                )
            rows.append(row)
        header = ["month", "years with data"]
        for f in OUT_OF_RANGE_FLAGS:
            header += [f"{f}, {YEAR}", f"{f}, mean", f"{f}, range"]
        lines += rm.table(header, rows, "l" + "r" * (len(header) - 1))
        add("")
        add(
            f"At {icao}, {YEAR} falls outside its {HISTORY_YEARS[0]} to {HISTORY_YEARS[-1]} "
            "range in " + ("; ".join(outside) if outside else "no month") + "."
        )
    add("")

    # Annex F ----------------------------------------------------------------
    add("## Annex F: the rows over four hours")
    add("")
    add(
        "Every validation row whose recorded taxi-out exceeds 4 h and which the "
        "record-quality term does not cover: out of every fit, counted by the scorer. "
        "For each, the recorded value, the shipped rung's prediction, the residual, "
        "its share of the squared error of its airport-month cell and of its whole "
        "month (both excluding flagged rows), and whether it has a flight-plan "
        "record. No movement or stand identifier is printed."
    )
    add("")
    over = np.flatnonzero(agg.over4)
    r_ship = st.R[:, ship]
    cell_s2 = {}
    for a, icao in enumerate(airports):
        for i in range(M):
            cell_s2[(icao, i)] = float(by_ap["s2_ex"][a, i, ship])
    month_s2 = {i: float(whole["s2_ex"][0, i, ship]) for i in range(M)}
    order = sorted(over, key=lambda j: (airports.index(st.icao[j]), st.month[j], st.y[j]))
    rows = []
    for j in order:
        sq = r_ship[j] ** 2
        rows.append(
            [
                st.icao[j],
                month_label(st.months[st.month[j]]),
                f"{st.y[j]:,.0f}",
                f"{st.y[j] - r_ship[j]:,.0f}",
                f"{r_ship[j]:+,.0f}",
                f"{100 * sq / cell_s2[(st.icao[j], st.month[j])]:.1f}%",
                f"{100 * sq / month_s2[st.month[j]]:.1f}%",
                "no" if st.no_plan[j] else "yes",
            ]
        )
    lines += rm.table(
        [
            "ICAO",
            "month",
            "recorded (s)",
            "predicted (s)",
            "residual (s)",
            "share of the cell's squared error",
            "share of the month's",
            "flight-plan record",
        ],
        rows,
        "llrrrrrl",
    )
    add("")
    counts = {icao: int(np.sum(st.icao[over] == icao)) for icao in airports}
    rows = [[icao, f"{counts[icao]:,}"] for icao in airports] + [
        ["all", f"{len(over):,}"]
    ]
    lines += rm.table(["ICAO", "rows over 4 h, the year"], rows, "lr")
    add("")
    at_ = [icao for icao in airports if counts[icao]]
    none_ = [icao for icao in airports if not counts[icao]]
    add(
        f"Over the year, {len(over):,} such rows occur, at "
        + (
            ", ".join(f"{icao} ({counts[icao]:,})" for icao in at_)
            if at_
            else "no airport"
        )
        + (f"; none at {', '.join(none_)}." if none_ else ".")
    )
    add("")

    # Annex G ----------------------------------------------------------------
    stored = sorted(p for m in st.months for p in month_files(m))
    read_monthly = sorted(
        MONTHLY / f"heldout-{m}_{SUBMISSION_KEY}.parquet" for m in st.months
    )
    read_ladder = sorted(
        LADDER_RESIDUALS / f"{LADDER_SPLIT}_{k}.parquet" for k in BOOSTED
    )
    residual_files = stored + read_monthly + read_ladder
    listed = (
        rb._inputs()
        + [BASELINE]
        + sorted(set(wx_files) - set(rb._inputs()))
        + [HISTORY / "MANIFEST.json"]
    )
    return "\n".join(lines) + "\n", (listed, residual_files)


def baseline_frequency() -> tuple[list[str], dict[str, list[str]]]:
    """``reports/baseline.md``'s December pair-frequency table, as printed."""
    text = BASELINE.read_text(encoding="utf-8").splitlines()
    at = text.index("## Error by pair frequency")
    header, rows = None, {}
    for line in text[at + 1 :]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif not set(cells[0]) <= {"-", ":", " "}:
            rows[cells[0]] = cells[1:]
    return header or [], rows


def reconcile_frequency(st: Store, g_b: dict) -> str:
    """December's buckets here against ``reports/baseline.md``'s, every rung."""
    from taxiout.validation import FREQUENCY_BUCKETS

    if LADDER_MONTH not in st.months:
        return "December is not in this run, so the buckets are not reconciled."
    m = st.months.index(LADDER_MONTH)
    header, printed = baseline_frequency()
    want = [
        f"{b} ({int(g_b['n_ex'][i, m]):,} rows)" for i, b in enumerate(FREQUENCY_BUCKETS)
    ]
    counts_ok = header[1 : 1 + len(want)] == want
    checked, bad = 0, []
    for k, key in enumerate(KEYS):
        rung = LADDER_BY_KEY[key]
        if rung.diagnostic or rung.label not in printed:
            continue
        mine = [fmt(v) for v in readings(g_b, k)["ex"][:, m]]
        checked += 1
        if mine != printed[rung.label][: len(mine)]:
            bad.append(rung.label)
    ok = counts_ok and not bad and checked
    return (
        f"Reconciliation: December's four bucket row counts "
        f"({'equal' if counts_ok else '**differ from**'} `reports/baseline.md`'s "
        f"*Error by pair frequency*) and the {checked} rungs' bucket figures there "
        + (
            "are reproduced here cell for cell, so the buckets are the ladder's."
            if ok
            else f"**are not all reproduced: {', '.join(bad) or 'the counts'}.**"
        )
    )


def inputs_section(listed: list[Path], residual_files: list[Path]) -> list[str]:
    """The declared-inputs footer, with the residual files summarized."""
    from taxiout.provenance import digest, report_inputs_footer

    rows = []
    for p in sorted(set(listed) | set(residual_files)):
        rows.append((Path(p).resolve().relative_to(REPO).as_posix(), digest(p)))
    tsv = WORK / "inputs.tsv"
    tsv.write_text(
        "path\tsha256\n" + "".join(f"{a}\t{b}\n" for a, b in rows), encoding="utf-8"
    )
    rel = {Path(p).resolve().relative_to(REPO).as_posix() for p in residual_files}
    digests = sorted(d for a, d in rows if a in rel)
    summary = hashlib.sha256("\n".join(digests).encode()).hexdigest()
    # Annex G is the declared-inputs footer under its own heading, which
    # ``scripts/check_provenance.py`` reads.
    out = report_inputs_footer(REPO, listed)
    out += [
        "",
        f"The residual files read are summarized rather than listed: {len(residual_files):,} "
        f"files ({sum(1 for p in residual_files if MONTHLY in Path(p).parents):,} of "
        f"`reports/monthly.md`'s under `data/processed/monthly/`, "
        f"{sum(1 for p in residual_files if LADDER_RESIDUALS in Path(p).parents):,} of the "
        f"ladder's under `data/processed/residuals/`, and "
        f"{sum(1 for p in residual_files if WORK in Path(p).parents):,} under "
        "`data/processed/monthly_ladder/`, the twelve months' rung, split and run files), "
        f"whose sorted sha256 digests, one per line, digest to `{summary}`. "
        f"The full list, every input with its digest, is `{shown(tsv)}`, beside the "
        "files it lists.",
        "",
        "Distance matrices derived from OpenStreetMap: data (c) "
        "OpenStreetMap contributors, ODbL 1.0. Movement and flight data",
        "supplied by EUROCONTROL PRC and the OpenSky Network under the terms",
        "of the PRC Data Challenge 2026. Weather from the Iowa Environmental",
        "Mesonet of Iowa State University.",
    ]
    return out


def report(out: Path, months: list[str] | None = None) -> int:
    before = fingerprint()
    t0 = time.monotonic()
    use = tuple(m for m in MONTHS if m in set(months or MONTHS))
    st = Store(use)
    text, (listed, residual_files) = build_report(st)
    text += "\n".join(inputs_section(listed, residual_files)) + "\n"
    if "\u2014" in text:
        raise SystemExit("refusing to write the report: it carries an em dash")
    movements = {f"{int(v)}" for v in st.mvt_id}
    stands = st.stands | set(
        json.loads((WORK / "stands.json").read_text(encoding="utf-8"))
    )
    rm.check_no_identifiers(text, stands, movements)
    out.parent.mkdir(parents=True, exist_ok=True)
    moved = touched(before, fingerprint())
    if moved:
        print("TOUCHED: " + ", ".join(moved), file=sys.stderr)
        return 1
    out.write_text(text, encoding="utf-8")
    print(
        f"wrote {out} ({len(text.splitlines()):,} lines) from {len(use)} months in "
        f"{time.monotonic() - t0:.0f} s; frame {st.frame_s:.0f} s; nothing touched"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="the splits and rungs; fit nothing"
    )
    mode.add_argument("--fit", action="store_true", help="fit the months not on disk")
    mode.add_argument(
        "--report", action="store_true", help="write the report; fit nothing"
    )
    ap.add_argument(
        "--months", nargs="*", default=None, help="restrict --fit to these months"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the report's path")
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.fit:
        return fit(args.months)
    return report(args.out, args.months)


if __name__ == "__main__":
    raise SystemExit(main())
