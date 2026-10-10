#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The checks the final file must pass before its one upload, and its audit.

Three documents fix them, each before any v12 prediction existed:

- ``reports/preregistered-2026-10-08-final-phase.md`` §2, the checks before
  the upload, "because there is no second one": the file has exactly the
  template's identifiers, once each, in its order; every prediction is finite
  and positive; any prediction outside the range of v11's is listed with the
  rule that set it, one set by no rule being a stop;
- ``reports/preregistered-2026-10-09-jj-check.md`` §1, which replaces that
  section's comparison of January and July with v11 by three checks against a
  control, the same fits predicting the final file's January and July rows
  alone: **(i)** every January and July row whose features differ between the
  control and the four months is explained by a February or June movement;
  **(ii)** the file's January and July predictions equal the control's on
  every row (i) does not list; **(iii)** per row, the prediction less v11's:
  the distribution, the 50 largest changes, and every change over 600 s with
  the columns that changed, a change no changed column explains being a stop;
- ``reports/preregistered-2026-10-09-records.md`` §3, the audit: **(a)** every
  prediction over 4 h and every row a day-scale term set, with its rule, its
  weight, its lag and its cell's 2025 count; **(b)** such a row whose cell
  held fewer than 20 rows in 2025, apart; **(c)** a weight of exactly 0 or 1
  on such a row, apart; **(d)** the rows whose flight plan lies on the last day
  of a month and whose take-off lies on the first; **(e)** per airport and
  month, predictions over 4 h per thousand rows in February and June 2026
  against the same configuration's held-out February and June 2025, a ratio
  above 2 listed for review row by row before the freeze. A row the rules do
  not explain stops the upload; the audit changes no prediction.

Reads the file ``scripts/make_submission.py`` wrote, its manifest and its parts
file (per row what each part gave), the template and the ranking file, the
control's file, v11's file, and two tables the landing builds from the feature
tables: the rows whose features differ between the control and the four
months (``--neighbours``: identifier, the columns, whether a February or June
movement explains it) and the January and July rows' columns that differ from
v11's (``--v11-columns``: identifier, the columns). The 2025 reference for (e)
is a table of counts per airport and month (``--reference-2025``). Writes a
report of aggregates and listed rows to ``--out`` (never into ``reports/``: it
names ranking rows) and exits 1 on a stop.

Usage::

    python scripts/check_final.py --file data/submissions/<team>_final.parquet \\
        --control <control file> --v11 <v11 file> --neighbours <parquet> \\
        --v11-columns <parquet> --reference-2025 <csv> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# The fusion reads, for each departure, its peers' predictions less their base
# within this many seconds of its take-off, at the same airport on the same day
# (``taxiout.fusion.peer_features``); check (ii) follows the same peers.
from taxiout.fusion import PEER_WINDOW_S  # noqa: E402

DATA = REPO / "data" / "challenge"
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
RULED = ("c6b", "term", "plan_shift")


def rule_numbers() -> dict:
    """The documents' numbers in their words, held in a function so that the
    provenance register has no module-level literal to categorise:

    - "the 50 largest changes, and every change over 600 s" (jj-check §1, iii);
    - "every prediction over 4 h" and "p_D >= 0.01" (records §3, a);
    - "a hedge row whose cell has fewer than 20 rows in 2025" (records §3, b);
    - "any day-scale weight of exactly 0 or 1" (records §3, c);
    - "per thousand rows", "a ratio above 2" (records §3, e).
    """
    return {
        "largest": 50,
        "listed_over_s": 600.0,
        "long_s": 4 * 3600.0,
        "thin_cell": 20,
        "weights_apart": (0.0, 1.0),
        "per": 1000.0,
        "ratio": 2.0,
        "review_months": ("2026-02", "2026-06"),
        "control_months": ("2026-01", "2026-07"),
    }


# --------------------------------------------------------------------------
# reading


def read_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def setter(parts: pd.DataFrame) -> np.ndarray:
    """What set each row's written value: the window, the fusion, a rule or the base."""
    rule = parts["rule"].astype(str).to_numpy()
    out = np.where(rule != "", rule, "base").astype(object)
    out = np.where(parts["fusion_taken"].to_numpy(dtype=bool), "fusion", out)
    out = np.where(parts["window_moved"].to_numpy(dtype=bool), "window", out)
    return out.astype(str)


# --------------------------------------------------------------------------
# the checks before the upload


def check_template(file: pd.DataFrame, template: pd.DataFrame) -> dict:
    """The template's identifiers, once each, in its order, and its two columns."""
    ids = file[ID].to_numpy(dtype=float)
    want = template[ID].to_numpy(dtype=float)
    out = {
        "rows": int(len(file)),
        "template_rows": int(len(template)),
        "columns": list(file.columns) == list(template.columns),
        "unique": bool(pd.Series(ids).is_unique),
        "in_order": bool(len(ids) == len(want) and np.array_equal(ids, want)),
        "types": [str(file[c].dtype) for c in file.columns],
    }
    out["ok"] = out["columns"] and out["unique"] and out["in_order"]
    return out


def check_values(file: pd.DataFrame) -> dict:
    """Every value finite and none negative. A zero is the build's own clip at 0
    (``rint(clip(pred, 0))`` since v7; v11 wrote ten): counted, not a stop."""
    v = file[TARGET].to_numpy(dtype=float)
    finite = np.isfinite(v)
    out = {
        "not_finite": int((~finite).sum()),
        "negative": int((finite & (v < 0)).sum()),
        "zero": int((finite & (v == 0)).sum()),
        "not_positive": int((finite & (v <= 0)).sum()),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
    }
    out["ok"] = out["not_finite"] == 0 and out["negative"] == 0
    return out


def check_v11_range(file: pd.DataFrame, parts: pd.DataFrame, v11: pd.DataFrame) -> dict:
    """Predictions outside v11's range, each with what set it; one set by no rule stops."""
    lo, hi = float(v11[TARGET].min()), float(v11[TARGET].max())
    v = file[TARGET].to_numpy(dtype=float)
    p = parts.set_index(ID).reindex(file[ID].to_numpy(dtype=float))
    who = setter(p.reset_index())
    out_rows = np.flatnonzero((v < lo) | (v > hi))
    listed = [
        {
            "row": int(i),
            "airport": str(p["ADEP_mvt"].iloc[i]),
            "month": str(p["month"].iloc[i]),
            "value": float(v[i]),
            "set_by": who[i],
        }
        for i in out_rows
    ]
    by_no_rule = [r for r in listed if r["set_by"] in ("base", "fusion")]
    return {
        "v11_min": lo,
        "v11_max": hi,
        "outside": len(listed),
        "listed": listed,
        "set_by_no_rule": len(by_no_rule),
        "ok": not by_no_rule,
    }


# --------------------------------------------------------------------------
# (i) to (iii)


def check_neighbours(neighbours: pd.DataFrame) -> dict:
    """(i): every row whose features differ between the control and the four months
    is explained by a February or June movement."""
    explained = neighbours["explained"].to_numpy(dtype=bool)
    month = neighbours["month"].astype(str) if "month" in neighbours else None
    out = {
        "rows": int(len(neighbours)),
        "unexplained": int((~explained).sum()),
        "by_month": {} if month is None else month.value_counts().sort_index().to_dict(),
    }
    out["ok"] = out["unexplained"] == 0
    return out


def peer_explained(
    ids: np.ndarray,
    parts: pd.DataFrame,
    control_parts: pd.DataFrame,
    ranking: pd.DataFrame,
) -> np.ndarray:
    """For each of ``ids``, whether a peer of its (same airport, same UTC day,
    within ``PEER_WINDOW_S`` of its take-off) has a base that differs between the
    file and the control. Such a row's fusion inputs moved with its neighbour's
    base, one step beyond the rows (i) lists."""
    common = control_parts[ID].to_numpy(dtype=float)
    b_file = parts.set_index(ID)["base"].reindex(common).to_numpy(dtype=float)
    b_ctl = control_parts.set_index(ID)["base"].reindex(common).to_numpy(dtype=float)
    adep = parts.set_index(ID)["ADEP_mvt"].reindex(common).astype(str).to_numpy()
    t = pd.to_datetime(
        ranking.set_index(ID)["MVT_TIME_UTC_mvt"].reindex(common), utc=True
    )
    sec = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    day = t.dt.strftime("%Y-%m-%d").to_numpy()
    changed = b_file != b_ctl
    groups: dict[tuple[str, str], np.ndarray] = {}
    for a, d, s in zip(adep[changed], day[changed], sec[changed]):
        groups.setdefault((a, d), []).append(s)
    groups = {k: np.sort(np.asarray(v)) for k, v in groups.items()}
    pos = {x: i for i, x in enumerate(common)}
    out = np.zeros(len(ids), dtype=bool)
    for j, x in enumerate(ids):
        i = pos.get(float(x))
        if i is None:
            continue
        arr = groups.get((adep[i], day[i]))
        if arr is None:
            continue
        k = np.searchsorted(arr, sec[i] - PEER_WINDOW_S, side="left")
        out[j] = k < len(arr) and arr[k] <= sec[i] + PEER_WINDOW_S
    return out


def check_control(
    file: pd.DataFrame,
    control: pd.DataFrame,
    neighbours: pd.DataFrame,
    parts: pd.DataFrame | None = None,
    control_parts: pd.DataFrame | None = None,
    ranking: pd.DataFrame | None = None,
) -> dict:
    """(ii): the January and July predictions equal the control's on every row (i)
    does not list, or on a row whose fusion peers include one whose base moved
    (``peer_explained``), when the control's parts file is there to say so."""
    c = control.set_index(ID)[TARGET]
    f = file.set_index(ID)[TARGET].reindex(c.index)
    if f.isna().any():
        return {"ok": False, "missing": int(f.isna().sum())}
    listed = set(neighbours[ID].to_numpy(dtype=float))
    keep = ~c.index.isin(listed)
    differ = (f.to_numpy(dtype=float) != c.to_numpy(dtype=float)) & keep
    out = {
        "control_rows": int(len(c)),
        "listed_by_i": int((~keep).sum()),
        "listed_rows_that_differ": int(
            ((f.to_numpy(dtype=float) != c.to_numpy(dtype=float)) & ~keep).sum()
        ),
        "differ_elsewhere": int(differ.sum()),
    }
    unexplained = int(differ.sum())
    if control_parts is not None and parts is not None and ranking is not None:
        peer = peer_explained(
            c.index.to_numpy(dtype=float)[differ], parts, control_parts, ranking
        )
        out["explained_by_peers"] = int(peer.sum())
        unexplained = int((~peer).sum())
    out["unexplained"] = unexplained
    out["ok"] = unexplained == 0
    return out


def check_v11(
    file: pd.DataFrame, parts: pd.DataFrame, v11: pd.DataFrame, columns: pd.DataFrame
) -> dict:
    """(iii): per row, the prediction less v11's, on the rows v11 predicted."""
    n = rule_numbers()
    old = v11.set_index(ID)[TARGET].astype(float)
    new = file.set_index(ID)[TARGET].reindex(old.index).astype(float)
    if new.isna().any():
        return {"ok": False, "missing": int(new.isna().sum())}
    d = new - old
    q = d.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    p = parts.set_index(ID).reindex(old.index)
    who = pd.Series(setter(p.reset_index()), index=old.index)
    cols = columns.set_index(ID)["columns"].reindex(old.index)
    changed_cols = cols.fillna("").astype(str)
    order = d.abs().sort_values(ascending=False, kind="stable").index
    largest = [
        {
            "change_s": float(d[i]),
            "v11": float(old[i]),
            "v12": float(new[i]),
            "set_by": who[i],
            "columns": changed_cols[i],
        }
        for i in order[: n["largest"]]
    ]
    over = d.abs() > n["listed_over_s"]
    # a change is explained by a changed column, or by a part whose model or rule
    # changed by definition between v11 and v12: C17's fusion and the window
    explained = (changed_cols != "") | who.isin(["fusion", "window"])
    unexplained = over & ~explained
    return {
        "rows": int(len(d)),
        "changed": int((d != 0).sum()),
        "mean_s": float(d.mean()),
        "quantiles_s": {f"{k:g}": float(v) for k, v in q.items()},
        "largest": largest,
        "over": int(over.sum()),
        "over_by_setter": who[over].value_counts().sort_index().to_dict(),
        "over_rows": [
            {
                "change_s": float(d[i]),
                "set_by": who[i],
                "columns": changed_cols[i],
            }
            for i in d.index[over]
        ],
        "unexplained": int(unexplained.sum()),
        "ok": int(unexplained.sum()) == 0,
    }


# --------------------------------------------------------------------------
# the audit


def audit(parts: pd.DataFrame, ranking: pd.DataFrame, reference: pd.DataFrame) -> dict:
    n = rule_numbers()
    p = parts.reset_index(drop=True)
    who = setter(p)
    v = p["written"].to_numpy(dtype=float)
    hedge = p["rule"].astype(str).isin(RULED).to_numpy()
    long_ = v > n["long_s"]
    rows = []
    for i in np.flatnonzero(long_ | hedge):
        rows.append(
            {
                "row": int(i),
                "airport": str(p["ADEP_mvt"].iloc[i]),
                "month": str(p["month"].iloc[i]),
                "written": float(v[i]),
                "set_by": who[i],
                "rule": str(p["rule"].iloc[i]),
                "weight": float(p["weight"].iloc[i]),
                "lag_s": float(p["lag_s"].iloc[i]),
                "rest_mean_s": float(p["rest_mean_s"].iloc[i]),
                "cell_rows_2025": float(p["cell_rows"].iloc[i]),
            }
        )
    # a long prediction the rules do not explain: set by the base or the fusion
    unexplained = [
        r
        for r in rows
        if r["written"] > n["long_s"] and r["set_by"] in ("base", "fusion")
    ]
    thin = [
        r for r in rows if r["rule"] in RULED and r["cell_rows_2025"] < n["thin_cell"]
    ]
    extreme = [
        r for r in rows if r["rule"] in RULED and r["weight"] in n["weights_apart"]
    ]
    # (d) the flight plan on a month's last day, the take-off on the next month's first
    rk = ranking.set_index(ID).reindex(p[ID].to_numpy(dtype=float))
    lobt = pd.to_datetime(rk["LOBT_flt"], utc=True)
    mvt = pd.to_datetime(rk["MVT_TIME_UTC_mvt"], utc=True)
    last = (lobt + pd.Timedelta(days=1)).dt.day == 1
    first = mvt.dt.day == 1
    seam = (last & first & (mvt.dt.month != lobt.dt.month)).fillna(False).to_numpy()
    d_rows = [
        {
            "row": int(i),
            "airport": str(p["ADEP_mvt"].iloc[i]),
            "month": str(p["month"].iloc[i]),
            "written": float(v[i]),
            "set_by": who[i],
        }
        for i in np.flatnonzero(seam)
    ]
    # (e) per airport and month, predictions over 4 h per thousand rows
    frame = pd.DataFrame(
        {
            "airport": p["ADEP_mvt"].astype(str),
            "month": p["month"].astype(str),
            "long": long_,
        }
    )
    frame = frame[frame["month"].isin(n["review_months"])]
    got = frame.groupby(["airport", "month"]).agg(
        rows=("long", "size"), long=("long", "sum")
    )
    ref = reference.copy()
    # "06", 6 or "2025-06": the month as two digits
    ref["month"] = ref["month"].astype(str).str.split("-").str[-1].str.zfill(2)
    ref = ref.set_index(["airport", "month"])
    e = []
    for (a, m), r in got.iterrows():
        key = (a, m[-2:])
        rr = ref.loc[key] if key in ref.index else None
        per_new = n["per"] * r["long"] / r["rows"]
        per_old = None if rr is None else n["per"] * rr["long"] / rr["rows"]
        ratio = None
        if per_old is not None:
            ratio = (
                (float("inf") if per_new > 0 else 0.0)
                if per_old == 0
                else per_new / per_old
            )
        e.append(
            {
                "airport": a,
                "month": m,
                "rows": int(r["rows"]),
                "long": int(r["long"]),
                "per_thousand": per_new,
                "reference_rows": None if rr is None else int(rr["rows"]),
                "reference_long": None if rr is None else int(rr["long"]),
                "reference_per_thousand": per_old,
                "ratio": ratio,
                "review": ratio is not None and ratio > n["ratio"],
            }
        )
    return {
        "a_rows": rows,
        "a_long": int(long_.sum()),
        "a_hedge": int(hedge.sum()),
        "a_unexplained": unexplained,
        "b_thin_cells": thin,
        "c_weights_0_or_1": extreme,
        "d_month_seam": d_rows,
        "e_per_airport_month": e,
        "e_for_review": [x for x in e if x["review"]],
        "ok": not unexplained,
    }


# --------------------------------------------------------------------------
# the report


def manifest_window(manifest: dict) -> dict:
    w = manifest.get("window") or {}
    return {
        "applied": manifest.get("window_applied"),
        "w_prime_s": w.get("w_prime_s"),
        "rows_with_lobt": w.get("rows_with_lobt"),
        "rows_moved": w.get("rows_moved"),
        "rows_moved_by_airport_month": w.get("rows_moved_by_airport_month"),
        "sum_d2_by_month": w.get("sum_d2_by_month"),
        "rows_by_month": w.get("rows_by_month"),
    }


def run(args) -> dict:
    file = read_file(args.file)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    parts = pd.read_parquet(args.parts)
    template = pd.read_parquet(args.template)
    ranking = pd.read_parquet(
        args.ranking, columns=[ID, "PHASE_mvt", "LOBT_flt", "MVT_TIME_UTC_mvt"]
    )
    ranking = ranking[ranking["PHASE_mvt"].astype(str) == "DEP"]
    control = read_file(args.control)
    cstem = args.control.name.removesuffix(".parquet")
    cparts = args.control.with_name(f"{cstem}.parts.parquet")
    control_parts = pd.read_parquet(cparts) if cparts.exists() else None
    v11 = read_file(args.v11)
    neighbours = pd.read_parquet(args.neighbours)
    columns = pd.read_parquet(args.v11_columns)
    reference = pd.read_csv(args.reference_2025)
    if not np.array_equal(
        np.sort(parts[ID].to_numpy(dtype=float)),
        np.sort(file[ID].to_numpy(dtype=float)),
    ):
        raise SystemExit("the parts file is not of the file's rows")
    p = parts.set_index(ID).reindex(file[ID].to_numpy(dtype=float))
    if not np.array_equal(
        p["written"].to_numpy(dtype=float), file[TARGET].to_numpy(dtype=float)
    ):
        raise SystemExit("the parts file's written values are not the file's")
    out = {
        "template": check_template(file, template),
        "values": check_values(file),
        "v11_range": check_v11_range(file, parts, v11),
        "i_neighbours": check_neighbours(neighbours),
        "ii_control": check_control(
            file, control, neighbours, parts, control_parts, ranking
        ),
        "iii_v11": check_v11(file, parts, v11, columns),
        "audit": audit(parts, ranking, reference),
        "window": manifest_window(manifest),
        "file_sha256": manifest.get("file_sha256"),
    }
    out["ok"] = all(
        out[k]["ok"]
        for k in (
            "template",
            "values",
            "v11_range",
            "i_neighbours",
            "ii_control",
            "iii_v11",
            "audit",
        )
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--parts", type=Path)
    ap.add_argument("--template", type=Path, default=DATA / "submitting.parquet")
    ap.add_argument("--ranking", type=Path, default=DATA / "ranking.parquet")
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--v11", type=Path, required=True)
    ap.add_argument("--neighbours", type=Path, required=True)
    ap.add_argument("--v11-columns", type=Path, required=True)
    ap.add_argument("--reference-2025", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    stem = args.file.name.removesuffix(".parquet")
    args.manifest = args.manifest or args.file.with_name(f"{stem}.manifest.json")
    args.parts = args.parts or args.file.with_name(f"{stem}.parts.parquet")
    out = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check_final.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8"
    )
    for k in (
        "template",
        "values",
        "v11_range",
        "i_neighbours",
        "ii_control",
        "iii_v11",
        "audit",
    ):
        print(f"{k}: {'ok' if out[k]['ok'] else 'STOP'}")
    review = out["audit"]["e_for_review"]
    if review:
        print(
            f"(e): {len(review)} airport-month(s) above the ratio, for review before the freeze"
        )
    print("FINAL CHECKS: " + ("pass" if out["ok"] else "STOP"))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
