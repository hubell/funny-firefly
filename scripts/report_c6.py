#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""C6, the record-quality term's rows predicted from the schedule, under its rule.

Section 2 of ``reports/preregistered-2026-10-04-tails.md`` defines two
candidates for the rows the term covers, C6a and C6b, and the rule that adopts
one of them. Both are in ``taxiout.tails`` (``C6a``, ``C6b``); this script reads
the rule from that code and writes ``reports/c6.md``:

- leave one month out over the twelve months of 2025: each month's rows are
  predicted by the candidate and by the term, both fitted on the other eleven
  months' rows (the term's cells, coverage, shares and rest means, and C6b's
  classifier), and only the rows that fold's term covers are measured;
- the measure, the mean squared error in s² against the recorded taxi-out as it
  is, long values included, after the floor at zero and the rounding the file
  applies (``np.clip`` at zero, then ``np.rint``);
- the three conditions: lower pooled than the term; the 95% interval of the
  pooled difference, from the resamples of days ``rule_numbers`` holds,
  wholly below zero; and not higher than the term in January or in July. If
  both clear, the one with the lower pooled figure.

A fold's month is the month of its training file, and a resampled day is the
UTC day of the take-off. The resamples are drawn once and applied to the term
and both candidates alike, and the interval is the percentile interval.

Then, if a candidate is adopted, it is fitted on every fitting row of the
term's cells over the twelve months and the report prints how many ranking
rows the term covers and what the candidate predicts there, in aggregate. No
truth column of the ranking set is read.

Usage::

    python scripts/report_c6.py            # writes reports/c6.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import monthly as c  # noqa: E402
from taxiout import tails  # noqa: E402
from taxiout.challenge import (  # noqa: E402
    load_ranking,
    movement_files,
    read_table,
    to_pandas,
)
from taxiout.provenance import FROZEN_DOCUMENTS, digest  # noqa: E402

DATA = REPO / "data" / "challenge"
RULE = "reports/preregistered-2026-10-04-tails.md"
DEFAULT_OUT = REPO / "reports" / "c6.md"


def rule_numbers() -> dict:
    """The rule's numbers in the section's words, held in a function so that
    the provenance register has no module-level literal to categorise:

    - "2,000 resamples of days with seed 20260902";
    - "not higher than the term's in January or in July";
    - "the 95% interval", read as the percentile interval's two tails;
    - "the twelve months of 2025".
    """
    return {
        "seed": 20260902,
        "resamples": 2000,
        "months_read": (1, 7),
        "interval": (2.5, 97.5),
        "months": tuple(range(1, 13)),
    }


TRAIN_COLUMNS = sorted(set(tails.C6_PREDICT_INPUTS) | set(tails.C6_FIT_ONLY_INPUTS))
RANK_COLUMNS = sorted(set(tails.C6_PREDICT_INPUTS) | {"MVT_ID_mvt"})
CANDIDATES = ("c6a", "c6b")
LABEL = {"term": "term", "c6a": "C6a", "c6b": "C6b"}


def written(pred) -> np.ndarray:
    """As ``scripts/make_submission.py`` writes a prediction: the floor at zero, then ``np.rint``."""
    return np.rint(np.clip(np.asarray(pred, dtype=float), 0.0, None))


def as_objects(frame: pd.DataFrame) -> pd.DataFrame:
    for col in ("FLIGHT_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADEP_mvt"):
        frame[col] = frame[col].astype(object)
    return frame


def training() -> pd.DataFrame:
    """The 2025 departures, each with the month of the file it came from."""
    parts = []
    for f in movement_files(DATA):
        t = to_pandas(read_table(f, TRAIN_COLUMNS, phase="DEP"))
        t["month"] = int(f.stem.split("_")[1][5:7])
        parts.append(t)
    return as_objects(pd.concat(parts, ignore_index=True))


def rule_copy_checked() -> Path:
    path = REPO / RULE
    held = {f.path: f.sha256 for f in FROZEN_DOCUMENTS}
    if RULE not in held or digest(path) != held[RULE]:
        raise SystemExit(f"{RULE} is not at the digest the register holds")
    return path


def folds(train: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Every fold's measured rows and predictions, and what each fold fitted."""
    rows, info = [], []
    for m in rule_numbers()["months"]:
        tr = train[train["month"] != m]
        te = train[train["month"] == m]
        term = tails.TailModel().fit(tr)
        fit_rows = tr[tails.c6_fitting_rows(term, tr)]
        a = tails.C6a().fit(fit_rows)
        b = tails.C6b(a).fit(fit_rows)
        t_pred = term.predict(te)
        meas = te[t_pred.notna().to_numpy()]
        y = pd.to_numeric(meas["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise SystemExit(f"fold {m}: a measured row has no recorded taxi-out")
        sub = tails.c6_sub_cell(meas)
        cells = tails.c6_cell(meas)
        info.append(
            {
                "month": m,
                "covered": [f"{i} {k}" for i, k in term.covered.index],
                "fitting_rows": int(len(fit_rows)),
                "measured": cells.value_counts().sort_index().to_dict(),
                "thin": sum(1 for v in a.subs.values() if v[3]),
                "unseen": int((~sub.isin(b.sub_levels)).sum()),
                "kept": {k: len(v) for k, v in b.kept.items()},
            }
        )
        rows.append(
            pd.DataFrame(
                {
                    "month": m,
                    "day": tails._utc(meas["MVT_TIME_UTC_mvt"])
                    .dt.strftime("%Y-%m-%d")
                    .to_numpy(),
                    "cell": cells.to_numpy(),
                    "y": y,
                    "term": written(t_pred[t_pred.notna()].to_numpy()),
                    "c6a": written(a.predict(meas)),
                    "c6b": written(b.predict(meas)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), info


def mse(d: pd.DataFrame, col: str) -> float:
    return float(((d[col] - d["y"]) ** 2).mean())


def the_rule(R: pd.DataFrame) -> dict:
    """The table, the intervals, the conditions and the verdict."""
    n = rule_numbers()
    table = {m: {k: mse(R[R["month"] == m], k) for k in LABEL} for m in n["months"]}
    pooled = {k: mse(R, k) for k in LABEL}
    days = np.array(sorted(R["day"].unique()))
    day_idx = pd.Index(days).get_indexer(R["day"])
    se = {k: ((R[k] - R["y"]) ** 2).to_numpy() for k in LABEL}
    n_day = np.bincount(day_idx, minlength=len(days)).astype(float)
    s_day = {
        k: np.bincount(day_idx, weights=v, minlength=len(days)) for k, v in se.items()
    }
    rng = np.random.default_rng(n["seed"])
    draws = rng.integers(0, len(days), size=(n["resamples"], len(days)))
    counts = np.stack([np.bincount(r, minlength=len(days)) for r in draws]).astype(float)
    nb = counts @ n_day
    boot = {k: counts @ s_day[k] / nb for k in se}
    verdict = {}
    for k in CANDIDATES:
        lo, hi = np.percentile(boot[k] - boot["term"], list(n["interval"]))
        verdict[k] = {
            "diff": pooled[k] - pooled["term"],
            "lo": float(lo),
            "hi": float(hi),
            "c1": bool(pooled[k] - pooled["term"] < 0),
            "c2": bool(hi < 0),
            "c3": all(table[m][k] <= table[m]["term"] for m in n["months_read"]),
        }
    clear = [k for k in CANDIDATES if all(verdict[k][x] for x in ("c1", "c2", "c3"))]
    adopted = min(clear, key=lambda k: pooled[k]) if clear else None
    line = f"C6: ADOPTED {LABEL[adopted] if adopted else 'none'}"
    return {
        "table": table,
        "pooled": pooled,
        "days": len(days),
        "rows": len(R),
        "verdict": verdict,
        "adopted": adopted,
        "line": line,
    }


def ranking(train: pd.DataFrame, adopted: str) -> dict:
    """The adopted candidate fitted on the twelve months, on the ranking rows the term covers."""
    term = tails.TailModel().fit(train)
    model = tails.C6Model(term).fit(train)
    rank = as_objects(
        to_pandas(load_ranking(DATA, RANK_COLUMNS, phase="DEP")).reset_index(drop=True)
    )
    tp = term.predict(rank)
    if adopted == "c6b":
        pc = model.predict(rank)
    else:
        pc = pd.Series(np.nan, index=rank.index)
        cov = tp.notna().to_numpy()
        pc[cov] = model.c6a.predict(rank[cov])
    if not pc.notna().equals(tp.notna()):
        raise SystemExit("the candidate does not predict exactly the term's ranking rows")
    return {
        "rank": rank,
        "fitting_rows": model.fitting_rows,
        "covered": term.covered,
        "rows": int(tp.notna().sum()),
        "candidate": written(pc[pc.notna()].to_numpy()),
        "term": written(tp[tp.notna()].to_numpy()),
    }


def build_report(
    rule: dict, info: list[dict], rk: dict | None, inputs: list[Path]
) -> str:
    L: list[str] = []
    add = L.append
    T, P, V = rule["table"], rule["pooled"], rule["verdict"]
    add("# C6: the term's rows predicted from the schedule")
    add("")
    add(
        "Generated by `scripts/report_c6.py` from the twelve 2025 training files and the "
        f"ranking file, under section 2 of `{RULE}`, whose digest is checked against the "
        "register before anything is read. The two candidates are `taxiout.tails.C6a` and "
        "`taxiout.tails.C6b`; the term is `taxiout.tails.TailModel`, unchanged."
    )
    add("")
    add("## The verdict")
    add("")
    add("```")
    add(rule["line"])
    add("```")
    add("")
    add("## The term fitted on the twelve months")
    add("")
    if rk is not None:
        add(
            f"{rk['fitting_rows']:,} fitting rows in the covered cells; on the ranking set "
            f"the term covers {rk['rows']:,} departures."
        )
        add("")
        L.extend(
            c.table(
                ["cell", "n", "share", "rest mean (s)"],
                [
                    [
                        f"{i} `{b}`",
                        f"{int(r['n']):,}",
                        f"{r['share']:.3f}",
                        f"{r['rest_mean']:,.0f}",
                    ]
                    for (i, b), r in rk["covered"].iterrows()
                ],
                "lrrr",
            )
        )
        add("")
    add("## Each fold")
    add("")
    add(
        "Fitted on the other eleven months; measured on the rows that fold's term covers. "
        "A sub-cell under the five-row floor takes its whole cell's values; a level under "
        "the ten-row floor reads as `other`."
    )
    add("")
    cells = sorted({k for f in info for k in f["measured"]})
    L.extend(
        c.table(
            [
                "month",
                "fitting rows",
                *[f"`{k.split('|', 1)[1]}`" for k in cells],
                "measured",
                "thin sub-cells",
                "unseen sub-cells",
                "levels kept (prefix / stand / type)",
            ],
            [
                [
                    str(f["month"]),
                    f"{f['fitting_rows']:,}",
                    *[f"{f['measured'].get(k, 0):,}" for k in cells],
                    f"{sum(f['measured'].values()):,}",
                    str(f["thin"]),
                    str(f["unseen"]),
                    " / ".join(str(f["kept"][x]) for x in tails.C6_CATEGORICAL),
                ]
                for f in info
            ],
            "r" * (len(cells) + 5) + "l",
        )
    )
    add("")
    add("## Mean squared error, s²")
    add("")
    add(
        "Over the measured rows, against the recorded taxi-out as it is, long values included."
    )
    add("")
    L.extend(
        c.table(
            ["month", "rows", "term", "C6a", "C6b"],
            [
                [
                    str(m),
                    f"{sum(info[m - 1]['measured'].values()):,}",
                    *[f"{T[m][k]:,.0f}" for k in LABEL],
                ]
                for m in range(1, 13)
            ]
            + [
                [
                    "**pooled**",
                    f"**{rule['rows']:,}**",
                    *[f"**{P[k]:,.0f}**" for k in LABEL],
                ]
            ],
            "rrrrr",
        )
    )
    add("")
    add("## The intervals and the conditions")
    add("")
    n = rule_numbers()
    add(
        f"{n['resamples']:,} resamples of the {rule['days']:,} days with a measured row, "
        f"seed {n['seed']}, drawn once for the term and both candidates."
    )
    add("")
    jan, jul = n["months_read"]
    L.extend(
        c.table(
            [
                "",
                "pooled difference from the term",
                "95% interval",
                "(1) lower pooled",
                "(2) interval below 0",
                f"(3) {c.month_label(f'2025-{jan:02d}')}, "
                f"{c.month_label(f'2025-{jul:02d}')} not higher",
            ],
            [
                [
                    LABEL[k],
                    f"{V[k]['diff']:,.0f}",
                    f"[{V[k]['lo']:,.0f}, {V[k]['hi']:,.0f}]",
                    "yes" if V[k]["c1"] else "no",
                    "yes" if V[k]["c2"] else "no",
                    ("yes" if V[k]["c3"] else "no")
                    + f": {T[jan][k]:,.0f} against {T[jan]['term']:,.0f}; "
                    f"{T[jul][k]:,.0f} against {T[jul]['term']:,.0f}",
                ]
                for k in CANDIDATES
            ],
            "lrllll",
        )
    )
    add("")
    worse = [
        f"{c.month_label(f'2025-{m:02d}')} ({T[m]['c6b']:,.0f} against {T[m]['term']:,.0f})"
        for m in range(1, 13)
        if T[m]["c6b"] > T[m]["term"]
    ]
    if worse:
        add(
            "C6b is higher than the term in " + ", ".join(worse) + ". The rule reads "
            "January and July only; this is printed so it is on record."
        )
        add("")
    if rk is not None:
        q = [0.1, 0.25, 0.5, 0.75, 0.9]
        desc = pd.DataFrame(
            {
                LABEL[rule["adopted"]]: pd.Series(rk["candidate"]).describe(
                    percentiles=q
                ),
                "term": pd.Series(rk["term"]).describe(percentiles=q),
            }
        )
        add("## The ranking rows")
        add("")
        add(
            f"{LABEL[rule['adopted']]} fitted on every fitting row of the term's cells over the "
            f"twelve months, on the {rk['rows']:,} ranking departures the term covers, floored "
            "and rounded as the file writes them. No truth column of the ranking set was read."
        )
        add("")
        L.extend(
            c.table(
                ["s", LABEL[rule["adopted"]], "term"],
                [
                    [i, f"{r.iloc[0]:,.0f}", f"{r.iloc[1]:,.0f}"]
                    for i, r in desc.iterrows()
                ],
                "lrr",
            )
        )
        add("")
    L += c.inputs_footer(inputs)
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    before = c.fingerprint()
    copy = rule_copy_checked()
    train = training()
    R, info = folds(train)
    rule = the_rule(R)
    print(rule["line"])
    rk = ranking(train, rule["adopted"]) if rule["adopted"] else None
    inputs = list(movement_files(DATA)) + [DATA / "ranking.parquet", copy]
    text = build_report(rule, info, rk, inputs)
    stands = c.stands_of(train, *([rk["rank"]] if rk else []))
    ids = c.movement_ids(*([rk["rank"]["MVT_ID_mvt"]] if rk else []))
    c.write(args.out, text, stands, ids, before)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
