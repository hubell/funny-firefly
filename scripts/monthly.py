# SPDX-License-Identifier: GPL-3.0-only
"""What the five reports that read the held-out months after the ladder share.

``report_monthly_seeds.py``, ``report_monthly_pairs.py``,
``report_planshift.py``, ``report_istanbul.py`` and ``report_fiumicino.py``:
paths, the guard, the fingerprint, the formatting, the declared-inputs footer
and the training and ranking rows they read.

Nothing here fits a model. The departures come from ``run_baselines``' own
loaders, so the rows and their order are the frame's.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import report_monthly as rm  # noqa: E402
import run_baselines as rb  # noqa: E402
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.features import TARGET  # noqa: E402
from taxiout.models import TAIL_EXCLUSION_ADOPTED_H  # noqa: E402
from taxiout.provenance import report_inputs_footer  # noqa: E402

DATA = REPO / "data" / "challenge"
MONTHLY = REPO / "data" / "processed" / "monthly"
MONTHLY_LADDER = REPO / "data" / "processed" / "monthly_ladder"
SEEDS = REPO / "data" / "processed" / "monthly_seeds"
PAIRS = REPO / "data" / "processed" / "monthly_pairs"
METAR = REPO / "data" / "raw" / "metar"
HISTORY = REPO / "data" / "raw" / "metar_history"
V6 = REPO / "data" / "submissions" / "funny-firefly_v6.parquet"
TEMPLATE = DATA / "submitting.parquet"

FORBIDDEN = (
    REPO / "reports",
    REPO / "data" / "processed" / "residuals",
    MONTHLY,
    MONTHLY_LADDER,
    REPO / "data" / "raw",
)
MONTHS: tuple[str, ...] = rm.MONTHS
DAY_SHIFT_S = float(TAIL_EXCLUSION_ADOPTED_H) * 3600.0
MIN_CELL_ROWS = 30
ICAOS: tuple[str, ...] = tuple(sorted(a.icao for a in AIRPORTS))

fmt = rm.fmt
signed = rm.signed
month_label = rm.month_label
table = rm.table
rmse = rm.rmse


def signed1(x: float) -> str:
    return rm.signed(x, 1)


def fingerprint() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


touched = rm.touched


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs_footer(paths: list[Path]) -> list[str]:
    """Every input by sha256, in the declared-inputs footer ``check_provenance.py``
    reads, sorted, and the credit for the data the reports rest on."""
    return report_inputs_footer(REPO, sorted(set(paths), key=str)) + [
        "",
        "Distance matrices derived from OpenStreetMap: data (c) "
        "OpenStreetMap contributors, ODbL 1.0. Movement and flight data",
        "supplied by EUROCONTROL PRC and the OpenSky Network under the terms",
        "of the PRC Data Challenge 2026. Weather from the Iowa Environmental",
        "Mesonet of Iowa State University.",
    ]


def guard(text: str, stands: set[str], movements: set[str]) -> None:
    """``report_monthly.check_no_identifiers``, and no em dash."""
    rm.check_no_identifiers(text, stands, movements)
    if "\u2014" in text:
        raise SystemExit("refusing to write the report: it carries an em dash")


def write(
    out: Path, text: str, stands: set[str], movements: set[str], before: dict
) -> None:
    guard(text, stands, movements)
    moved = touched(before, fingerprint())
    if moved:
        raise SystemExit("TOUCHED: " + ", ".join(moved))
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# rows


def departures() -> pd.DataFrame:
    """The training year's departures, in the frame's order, as the fit reads them."""
    dep = rb.load_departures(None)
    if dep["MVT_ID_mvt"].duplicated().any():
        raise SystemExit("a movement identifier repeats in the training departures")
    return dep


def ranking_departures() -> pd.DataFrame:
    from taxiout.challenge import load_ranking, to_pandas

    columns = sorted(set(rb.REQUIRED_COLUMNS) | set(rb.TAILS_COLUMNS))
    return to_pandas(load_ranking(DATA, columns, phase="DEP")).reset_index(drop=True)


def stands_of(*frames: pd.DataFrame) -> set[str]:
    out: set[str] = set()
    for f in frames:
        out |= {str(v).strip() for v in f["STAND_mvt"].dropna().astype(str).unique()}
    return out


def movement_ids(*arrays) -> set[str]:
    out: set[str] = set()
    for a in arrays:
        v = pd.to_numeric(pd.Series(np.asarray(a)), errors="coerce").dropna()
        out |= {f"{int(x)}" for x in v.unique()}
    return out


def utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def target(dep: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(dep[TARGET], errors="coerce").astype(float)


def position(dep_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Row positions in the frame of the stored identifiers; every one must be there."""
    pos = pd.Series(np.arange(len(dep_ids)), index=dep_ids).reindex(ids)
    if pos.isna().any():
        raise SystemExit(f"{int(pos.isna().sum()):,} stored rows are not in the frame")
    return pos.to_numpy(dtype=int)


def cell_text(primary: float, second: float, n: int, left: int, render=fmt) -> str:
    """``report_monthly_ladder.py``'s cell: under ``MIN_CELL_ROWS`` rows it prints
    its count; the 4 h reading in brackets."""
    if n < MIN_CELL_ROWS:
        return f"({n:,} rows)"
    if left > 0:
        return f"{render(primary)} [{render(second)}; {left:,} out]"
    return render(primary)
