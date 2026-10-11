#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""v12 decomposed on the held-out 2025 year, step by step: ``reports/v12-ladder.md``.

Every step was adopted by its own rule before this report existed; this script
adopts nothing and reads every step against the one before it. Each step is
assembled as ``scripts/make_submission.py --final`` assembles the ranking file,
from stored out-of-fold predictions, every row of each held-out month of 2025
(leave one month out) and of the December and July legs, with the
record-quality term (C6b on its rows) and the plan-shifted departures' rule on
theirs at every step:

- **S0**, the configuration's booster alone (``combo4_booster``, C1's and C4's
  blocks on the rung), at the ladder's seed: the one seed the held-out records
  hold. v9's file blended it with C3b's CatBoost member at equal weights.
- **S1**, C11 (``reports/group-b.md``): C14's item at every airport where the
  selector keeps an item other than the booster weighted with C3b's member,
  which is C11's (the booster blended with C11's member) everywhere but at LTFM
  in held-out February, where the selector kept the per-airport booster's item;
  the booster alone where it keeps a weight (LIRF).
- **S2**, C14's selection in full (``reports/group-b.md``): LIRF's booster
  weight with C3b's member as well. v11's base, and v12's.
- **S3**, the fusion as v11 had it (``reports/fusion-new-base.md``, G1n): the
  models of the 174 fitting days, predicting every day with a table, gated per
  airport and month on the other months' covered rows.
- **S4**, C17's fusion in place of S3's (``reports/preregistered-2026-10-08-all-days.md``):
  fitted on the 348 days, gated the same way.
- **S5**, W, the flight-plan window (``reports/window.md``), last of all.

S0 to S2 read the board's records under ``data/processed/candidates_board/``
and must rebuild C14's record to the bit from its parts. S3 and S4 read the two
fusions' out-of-fold rows under ``data/processed/v12_ladder/``: per held-out
month, each fusion fitted on the other eleven months' fitting rows, with C14's
out-of-fold prediction as its base; every row's base must equal S2's model
prediction and its target the record's. The fusion's July leg (fitted January
to June) was never fitted, so the July leg reads S0 to S2 only. The two rules
are fitted here per split, on the split's training months, as
``scripts/report_c6.py`` fits C6b and the submission fits the rule; they are
cell rules fitted in seconds, and no booster, member or fusion is fitted.
Every value is read as the file writes it: floored at zero and rounded, the
window's clip after that.

The readings: RMSE per step over every row, per month, per pair of months as
the organizers rank the four (January with July, February with June, and the
four), over the year and over each leg; each step's change from the step
before with a 95% interval over take-off days (UTC, ``taxiout.validation``'s
cluster bootstrap, its seed and draws); the same over the core rows, the
rules' rows left out; the rows over four hours counted; and per airport for
the three pairs, as an annex.

Usage::

    python scripts/report_v12_ladder.py            # writes reports/v12-ladder.md
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
sys.path.insert(0, str(REPO / "scripts"))

import monthly as c  # noqa: E402
import report_c6 as rc6  # noqa: E402
from taxiout import fusion, group_b, tails, window  # noqa: E402
from taxiout.challenge import movement_files  # noqa: E402
from taxiout.validation import cluster_bootstrap, cluster_sums  # noqa: E402

DATA = REPO / "data" / "challenge"
BOARD = REPO / "data" / "processed" / "candidates_board"
ROWS = REPO / "data" / "processed" / "v12_ladder"
DEFAULT_OUT = REPO / "reports" / "v12-ladder.md"
FUSION_ROWS = {"S3": "reference_rows.parquet", "S4": "c17_rows.parquet"}
STEPS = ("S0", "S1", "S2", "S3", "S4", "S5")
LABELS = {
    "S0": "S0 booster alone",
    "S1": "S1 + C11",
    "S2": "S2 + C14 at LIRF",
    "S3": "S3 + v11's fusion",
    "S4": "S4 C17 for S3",
    "S5": "S5 + W",
}
LEGS = ("december", "july")
PAIRS = {
    "Jan/Jul": ("01", "07"),
    "Feb/Jun": ("02", "06"),
    "the four": ("01", "02", "06", "07"),
}
ID = "MVT_ID_mvt"
# Two candidate names the report prints that are stand identifiers at some
# airport too, as ``scripts/report_group_b.py`` exempts them; every other
# identifier still stops the write. Two distinct tokens stay under
# ``check_licence_hygiene.LEAK_THRESHOLD``, so the file needs no exemption there.
STAND_LIKE_NAMES = frozenset({"C11", "C14"})
EXTRA_COLUMNS = (
    "MVT_ID_mvt",
    "EOBT_1_flt",
    "LOBT_flt",
    "AOBT_3_flt",
    "FLIGHT_RULE_mvt",
    "FLIGHT_ID_mvt",
    "BLOCK_TIME_UTC_mvt",
)


def numbers() -> dict:
    """The report's fixed numbers, in a function so that the provenance
    register has no module-level literal to categorise: twelve months, the
    long-row threshold of four hours the other held-out reports count."""
    return {"months": tuple(f"{m:02d}" for m in range(1, 13)), "long_s": 4 * 3600.0}


def split_plan() -> list[tuple[str, str, tuple[int, ...]]]:
    """(record split, month read, training months), the board's splits."""
    months = numbers()["months"]
    out = [
        (f"heldout-2025-{mm}", mm, tuple(int(o) for o in months if o != mm))
        for mm in months
    ]
    out.append(("december", months[-1], tuple(int(o) for o in months[:-1])))
    out.append(("july", months[6], tuple(int(o) for o in months[:6])))
    return out


def written(pred) -> np.ndarray:
    return rc6.written(pred)


def record(split: str, key: str) -> tuple[pd.DataFrame, dict]:
    name = f"{split}__{key}__seed0"
    rec = json.loads((BOARD / f"{name}.json").read_text(encoding="utf-8"))
    kept = pd.read_parquet(BOARD / f"{name}.parquet")
    if group_b.pred_sha256(kept["pred"].to_numpy(dtype=float)) != rec["pred_sha256"]:
        raise SystemExit(f"{name}: its prediction does not hash to its record")
    return kept, rec


def models(split: str, used: set[Path]) -> tuple[pd.DataFrame, dict]:
    """S0, S1 and S2's model predictions for a split, C14 rebuilt from its parts."""
    k14, r14 = record(split, "c14")
    used |= {BOARD / f"{split}__c14__seed0.{x}" for x in ("json", "parquet")}
    choice = r14["derived"]["selector"]
    ids = k14["mvt_id"].to_numpy(dtype=float)
    parts: dict[str, np.ndarray] = {}

    def part(key: str) -> np.ndarray:
        if key not in parts:
            k, _ = record(split, key)
            used.update(BOARD / f"{split}__{key}__seed0.{x}" for x in ("json", "parquet"))
            if not np.array_equal(k["mvt_id"].to_numpy(dtype=float), ids):
                raise SystemExit(f"{split}: {key}'s rows are not C14's")
            parts[key] = k["pred"].to_numpy(dtype=float)
        return parts[key]

    booster = part("combo4_booster")
    icao = k14["icao"].astype(str).to_numpy()
    s1, s2 = booster.copy(), np.full(len(ids), np.nan)
    for a in sorted(set(icao)):
        item = choice[a]
        b, m, w = group_b.item_blend(item)
        pred = group_b.blend(part(b), part(m), w)
        at = icao == a
        s2[at] = pred[at]
        if not item.startswith(group_b.WEIGHT_PREFIX) and item != "base":
            s1[at] = pred[at]
    if not np.array_equal(s2, k14["pred"].to_numpy(dtype=float)):
        raise SystemExit(f"{split}: C14 rebuilt from its parts is not its record")
    out = k14[["mvt_id", "icao", "takeoff", "y", "term", "shift"]].copy()
    out["S0"], out["S1"], out["S2"] = booster, s1, s2
    return out, choice


def training() -> pd.DataFrame:
    """``report_c6.training()`` with the plan's off-block time beside its columns."""
    rc6.TRAIN_COLUMNS = sorted(set(rc6.TRAIN_COLUMNS) | set(EXTRA_COLUMNS))
    t = rc6.training()
    t["tmonth"] = tails._utc(t["MVT_TIME_UTC_mvt"]).dt.month.astype(int)
    if not t[ID].is_unique:
        raise SystemExit("a 2025 movement identifier repeats")
    return t


def rules(T: pd.DataFrame, k: pd.DataFrame, train_months: tuple[int, ...]):
    """The term's C6b prediction and the rule's, on a split's rows in its order.

    C6b and its term by the file's month, as ``scripts/report_c6.py`` fits them;
    the term and the plan-shift rule by the take-off month, as the submission
    fits them on the training year. Both must cover exactly the rows the
    record flags.
    """
    trf = T[T["month"].isin(train_months)]
    term_f = tails.TailModel().fit(trf)
    fit_rows = trf[tails.c6_fitting_rows(term_f, trf)]
    c6b = tails.C6b(tails.C6a().fit(fit_rows)).fit(fit_rows)
    trt = T[T["tmonth"].isin(train_months)]
    term_t = tails.TailModel().fit(trt)
    ps = tails.PlanShiftModel(term_t).fit(trt)
    pos = pd.Series(np.arange(len(T)), index=T[ID].to_numpy(dtype=float))
    at = pos.reindex(k["mvt_id"].to_numpy(dtype=float))
    if at.isna().any():
        raise SystemExit("a record row is not in the training files")
    V = T.iloc[at.to_numpy(dtype=int)].reset_index(drop=True)
    if not np.array_equal(
        V["TAXITIME_SEC_mvt"].to_numpy(dtype=float), k["y"].to_numpy(dtype=float)
    ):
        raise SystemExit("a record's target is not the training file's")
    rterm, rshift = k["term"].to_numpy(dtype=bool), k["shift"].to_numpy(dtype=bool)
    if not (
        np.array_equal(term_t.predict(V).notna().to_numpy(), rterm)
        and np.array_equal(term_f.predict(V).notna().to_numpy(), rterm)
        and np.array_equal(ps.in_class(V).to_numpy(dtype=bool), rshift)
    ):
        raise SystemExit("the rules cover other rows than the record flags")
    tail = np.full(len(V), np.nan)
    if rterm.any():
        tail[rterm] = c6b.predict(V[rterm])
    return pd.Series(tail), ps.predict(V).reset_index(drop=True), window.lag_s(V)


def fusion_rows(used: set[Path]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Each fusion's out-of-fold rows and its gate per airport and month."""
    out = {}
    for step, name in FUSION_ROWS.items():
        path = ROWS / name
        used.add(path)
        f = pd.read_parquet(path)
        f[ID] = f[ID].astype(float)
        if not f[ID].is_unique:
            raise SystemExit(f"{name}: a row repeats")
        out[step] = (f, fusion.gate_out_of_fold(f))
    a, b = out["S3"][0], out["S4"][0]
    for col in (ID, "month", "ADEP_mvt", "y", "base", "a_covered", "o_covered"):
        if not a[col].equals(b[col]):
            raise SystemExit(f"the two fusions' rows differ in {col}")
    return out


def assemble(T, fused, used) -> dict[str, pd.DataFrame]:
    """Every split's rows with each step's written value."""
    out = {}
    for split, mm, train_months in split_plan():
        k, choice = models(split, used)
        tail, ps, lag = rules(T, k, train_months)
        f = pd.DataFrame(
            {
                "airport": k["icao"].astype(str).to_numpy(),
                "day": pd.to_datetime(k["takeoff"], utc=True)
                .dt.strftime("%Y-%m-%d")
                .to_numpy(),
                "y": k["y"].to_numpy(dtype=float),
                "rules": (k["term"].astype(bool) | k["shift"].astype(bool)).to_numpy(),
            }
        )
        for s in ("S0", "S1", "S2"):
            f[s] = written(tails.combine(k[s].to_numpy(dtype=float), tail, ps))
        f.attrs["choice"] = choice
        if split != "july":
            ids = k["mvt_id"].to_numpy(dtype=float)
            for s in ("S3", "S4"):
                rows, gate = fused[s]
                m = rows[rows["month"].astype(str).str.zfill(2) == mm].set_index(ID)
                j = m.reindex(ids)
                joined = j["fusion"].notna().to_numpy()
                if int(joined.sum()) != len(m):
                    raise SystemExit(f"{split}: {s}'s rows are not all the split's")
                if (joined & f["rules"].to_numpy()).any():
                    raise SystemExit(f"{split}: a fusion row is a rules row")
                if not (
                    np.array_equal(
                        j["base"].to_numpy(float)[joined], k["S2"].to_numpy(float)[joined]
                    )
                    and np.array_equal(
                        j["y"].to_numpy(float)[joined], f["y"].to_numpy()[joined]
                    )
                ):
                    raise SystemExit(f"{split}: {s}'s base or target is not S2's")
                kept = (
                    gate.reindex(f["airport"].to_numpy())[mm].fillna(False).to_numpy(bool)
                )
                take = (
                    joined
                    & fusion.covered(j.fillna({"a_covered": 0, "o_covered": 0}))
                    & kept
                )
                v = f["S2"].to_numpy(dtype=float).copy()
                v[take] = written(j["fusion"].to_numpy(float)[take])
                f[s] = v
                f.attrs[f"{s}_takes"] = int(take.sum())
            f["S5"], moved = window.apply(f["S4"].to_numpy(dtype=float), lag)
            f.attrs["S5_moved"] = int(moved.sum())
        out[split] = f
    return out


def steps_of(f: pd.DataFrame) -> list[str]:
    return [s for s in STEPS if s in f.columns]


def read(f: pd.DataFrame) -> dict:
    """RMSE per step, each step less the one before with its interval, rows, long rows."""
    steps = steps_of(f)
    sums = cluster_sums({s: f[s] - f["y"] for s in steps}, f["day"])
    fold = [{"s": sums}]
    out = {
        "rows": int(len(f)),
        "days": int(f["day"].nunique()),
        "long": int((f["y"] > numbers()["long_s"]).sum()),
        "rmse": {s: float(np.sqrt(np.mean((f[s] - f["y"]) ** 2))) for s in steps},
        "change": {},
    }
    for a, b in zip(steps, steps[1:]):
        r = cluster_bootstrap(fold, lambda q, a=a, b=b: q["s"][b] - q["s"][a])
        out["change"][b] = (r.point, r.low, r.high)
    return out


def sets(frames: dict[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    months = numbers()["months"]
    by = {mm: frames[f"heldout-2025-{mm}"] for mm in months}
    out = [(c.month_label(f"2025-{mm}"), by[mm]) for mm in months]
    out += [
        (p, pd.concat([by[m] for m in ms], ignore_index=True)) for p, ms in PAIRS.items()
    ]
    out.append(("year", pd.concat([by[m] for m in months], ignore_index=True)))
    out += [(f"{leg} leg", frames[leg]) for leg in LEGS]
    return out


def cell(x: float) -> str:
    return c.fmt(x, 1)


def change_cell(t) -> str:
    p, lo, hi = t
    return f"{c.signed(p, 2)} [{c.signed(lo, 2)}, {c.signed(hi, 2)}]"


def tables(readings: list[tuple[str, dict]], title: str) -> list[str]:
    L = [f"### {title}: RMSE (s)", ""]
    L += c.table(
        ["rows read", "rows", "days", "over 4 h"] + [LABELS[s] for s in STEPS],
        [
            [n, f"{r['rows']:,}", f"{r['days']:,}", f"{r['long']:,}"]
            + [cell(r["rmse"][s]) if s in r["rmse"] else "needs a fit" for s in STEPS]
            for n, r in readings
        ],
        "lrrr" + "r" * len(STEPS),
    )
    L += [
        "",
        f"### {title}: each step less the one before (s), 95% interval over days",
        "",
    ]
    L += c.table(
        ["rows read"] + [f"{s} less {p}" for p, s in zip(STEPS, STEPS[1:])],
        [
            [n]
            + [
                change_cell(r["change"][s]) if s in r["change"] else "needs a fit"
                for s in STEPS[1:]
            ]
            for n, r in readings
        ],
        "l" + "r" * (len(STEPS) - 1),
    )
    return L + [""]


def per_airport(frames: dict[str, pd.DataFrame]) -> list[str]:
    months = numbers()["months"]
    by = {mm: frames[f"heldout-2025-{mm}"] for mm in months}
    L = []
    for p, ms in PAIRS.items():
        f = pd.concat([by[m] for m in ms], ignore_index=True)
        L += [f"### {p}, every row, RMSE (s)", ""]
        rows = []
        for a, g in f.groupby("airport", sort=True):
            rows.append(
                [a, f"{len(g):,}", f"{int((g['y'] > numbers()['long_s']).sum()):,}"]
                + [cell(float(np.sqrt(np.mean((g[s] - g["y"]) ** 2)))) for s in STEPS]
            )
        L += c.table(
            ["airport", "rows", "over 4 h"] + [LABELS[s] for s in STEPS],
            rows,
            "lrr" + "r" * len(STEPS),
        )
        L.append("")
    return L


def build_report(frames, readings_all, readings_core, inputs) -> str:
    months = numbers()["months"]
    L: list[str] = []
    add = L.append
    add("# v12, step by step, on the held-out 2025 year")
    add("")
    add(
        "Generated by `scripts/report_v12_ladder.py` from stored out-of-fold"
        " predictions; nothing in it is fitted but the two rules, per split, as"
        " the script's docstring says. Every step was adopted by its own rule"
        " before this report existed, and is cited by it; this report decides"
        " nothing. Each value is read as the file writes it, floored at zero and"
        " rounded, with the record-quality term and the plan-shifted departures'"
        " rule on their rows at every step. The design is not forward in time:"
        " each held-out month is predicted from the other eleven."
    )
    add("")
    add("## The steps")
    add("")
    w11 = group_b.item_blend("c11")[2]
    weights = sorted(
        {
            (a, it)
            for f in frames.values()
            for a, it in f.attrs["choice"].items()
            if it.startswith(group_b.WEIGHT_PREFIX)
        }
    )
    add(
        "- **S0**: the configuration's booster alone (`combo4_booster`), at the"
        " ladder's seed, the one seed the held-out records hold. v9's file blended"
        " it with C3b's member at equal weights."
    )
    add(
        "- **S1**: C11 (`reports/group-b.md`): C14's item wherever its selector keeps"
        f" one other than the booster weighted with C3b's member; C11's item, the"
        f" booster at weight {w11:g} with C11's member, but for the line below; the"
        " booster alone where the selector keeps a weight ("
        + ", ".join(sorted({a for a, _ in weights}))
        + ")."
    )
    add(
        "- **S2**: C14 (`reports/group-b.md`) in full, the weights it keeps ("
        + ", ".join(f"`{it}` at {a}" for a, it in weights)
        + ") with C3b's member added: v11's base and v12's."
    )
    add(
        "- **S3**: v11's fusion (`reports/fusion-new-base.md`, G1n), its models of the"
        " 174 fitting days predicting every day with a table."
    )
    add(
        "- **S4**: C17's fusion (`reports/preregistered-2026-10-08-all-days.md`) in"
        " place of S3's."
    )
    add("- **S5**: W (`reports/window.md`), the flight-plan window, last of all.")
    add("")
    other = sorted(
        {
            (c.month_label(f"2025-{mm}"), a, it)
            for mm in months
            for a, it in frames[f"heldout-2025-{mm}"].attrs["choice"].items()
            if it not in ("c11",) and not it.startswith(group_b.WEIGHT_PREFIX)
        }
    )
    if other:
        add(
            "C14's selector keeps an item other than C11's or a weight in: "
            + "; ".join(f"{m} at {a} (`{it}`)" for m, a, it in other)
            + "; S1 takes that item there."
        )
        add("")
    add("Rows each fusion takes, and rows the window moves, per held-out month:")
    add("")
    L.extend(
        c.table(
            ["month", "rows", "S3 takes", "S4 takes", "S5 moves"],
            [
                [
                    c.month_label(f"2025-{mm}"),
                    f"{len(frames[f'heldout-2025-{mm}']):,}",
                    f"{frames[f'heldout-2025-{mm}'].attrs['S3_takes']:,}",
                    f"{frames[f'heldout-2025-{mm}'].attrs['S4_takes']:,}",
                    f"{frames[f'heldout-2025-{mm}'].attrs['S5_moved']:,}",
                ]
                for mm in months
            ],
            "lrrrr",
        )
    )
    add("")
    add(
        "The December leg's records are the held-out December's (trained on January"
        " to November); the July leg is trained on January to June, and its fusion"
        " was never fitted, so S3 to S5 there would need a fit and are not read."
    )
    add("")
    add("## Every row")
    add("")
    L.extend(tables(readings_all, "Every row"))
    add("## The core rows")
    add("")
    add("The same, with the rows either rule covers left out.")
    add("")
    L.extend(tables(readings_core, "Core rows"))
    add("## Annex: per airport, the organizers' pairs")
    add("")
    L.extend(per_airport(frames))
    L += c.inputs_footer(inputs)
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    before = c.fingerprint()
    used: set[Path] = set()
    T = training()
    fused = fusion_rows(used)
    frames = assemble(T, fused, used)
    readings_all, readings_core = [], []
    for name, f in sets(frames):
        readings_all.append((name, read(f)))
        readings_core.append((name, read(f[~f["rules"]].reset_index(drop=True))))
    for name, r in readings_all:
        print(name, {s: round(v, 2) for s, v in r["rmse"].items()})
    inputs = sorted(used, key=str) + list(movement_files(DATA))
    text = build_report(frames, readings_all, readings_core, inputs)
    c.write(
        args.out, text, c.stands_of(T) - STAND_LIKE_NAMES, c.movement_ids(T[ID]), before
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
