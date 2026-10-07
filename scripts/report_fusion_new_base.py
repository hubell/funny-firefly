#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The fusion refitted on group B's new base, and G1 and G1n read again: ``reports/fusion-new-base.md``.

§4 of ``reports/preregistered-2026-10-06-airports.md``: if group B's
combination holds, group A's adopted pieces are refitted on the new base's
out-of-fold predictions and re-read by their own rules, and one that no longer
holds is left out of v11. The new base is the combination ``reports/group-b.md``
reads, the selector's choice per airport, whose twelve held-out months at the
ladder's seed sit under ``data/processed/candidates_board/`` and are checked
against ``taxiout.group_b.NEW_BASE_RECORDS``.

The only change from ``scripts/report_fusion.py`` is the base. The fusion
model of §5 of ``reports/preregistered-2026-10-04.md`` is fitted by
``taxiout.fusion`` on the same fitting table (``data/processed/fusion/``,
``make fusion``), its peer features recomputed from the new base, twelve times
leaving one month out; then, with the same functions and floors as
``reports/fusion.md``:

1. **G1** against the new base, under §2's rule of
   ``reports/preregistered-2026-10-06-last-week.md``;
2. **G1n**, its gate printed before any contrast, against G1 under the same
   three conditions (``reports/preregistered-2026-10-06-g1n.md`` §1), and
   beside it against the new base;
3. the twelve-month gate the ranking rows take.

**The fusion part of v11** is read from those verdicts as the working program
read the amendment's branches before any fit, a reading the owner confirmed on
7 October 2026: if G1 holds against the new base, G1n where G1n holds against G1,
otherwise G1 (§3 of the G1n amendment, "G1n if adopted, otherwise G1"); if G1
fails, G1n against the new base by G1's three conditions; otherwise none.

Beside, deciding nothing: covered rows, short rows, per airport, the ungated
fusion, and the same readings on the old base from ``scripts/report_fusion.py``'s
out-of-fold rows. Writes only aggregates. Takes about three minutes on eight
cores.

Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import candidates as cb  # noqa: E402
import monthly as c  # noqa: E402
import report_fusion as rf  # noqa: E402
from taxiout import fusion, group_b  # noqa: E402
from taxiout.tails import plan_shift_threshold_s  # noqa: E402

DATA = REPO / "data" / "challenge"
BOARD = REPO / "data" / "processed" / "candidates_board"
FUSION_DIR = REPO / "data" / "processed" / "fusion"
DEFAULT_OUT = REPO / "reports" / "fusion-new-base.md"
READINGS = ("rule", *cb.OTHER_READINGS)


def read_reports() -> tuple[Path, list[Path]]:
    """The reports this one reads and never writes: the floors, and the rules."""
    seeds, rules = rf.read_reports()
    reports = REPO / "reports"
    return seeds, rules + [reports / "preregistered-2026-10-06-airports.md"]


def fusion_part(g1: bool, g1n_vs_g1: bool, g1n_vs_base: bool) -> str:
    """The fusion part of v11, by the reading of the amendment's branches stated above."""
    if g1:
        return "G1n" if g1n_vs_g1 else "G1"
    return "G1n" if g1n_vs_base else "none"


def why(r: dict) -> list[str]:
    out = []
    if not r["pooled"]["read"]["rule"]:
        out.append("pooled")
    out += r["failing"]
    if not r["december"]["read"]["rule"]:
        out.append("December")
    return out


def build() -> dict:
    run: dict = {}
    fit = pd.read_parquet(FUSION_DIR / "fit_2025.parquet")
    base = group_b.read_new_base(BOARD)
    old = fusion.read_base(BOARD)
    cols = ["mvt_id", "icao", "runway", "takeoff", "term", "shift", "y", "month"]
    if not (
        np.array_equal(base["mvt_id"].to_numpy(), old["mvt_id"].to_numpy())
        and all(base[k].equals(old[k]) for k in cols)
    ):
        raise SystemExit("the new base's rows are not the configuration's, in order")
    rows = cb.booster_rows(base)
    y = base["y"].to_numpy(dtype=float)
    dec = rows & (base["month"] == "12").to_numpy()
    run["base_check"] = {
        name: {
            "rows": int(mask.sum()),
            "configuration": cb.rmse((y - old["pred"].to_numpy(dtype=float))[mask]),
            "new_base": cb.rmse((y - base["pred"].to_numpy(dtype=float))[mask]),
        }
        for name, mask in (("year", rows), ("december", dec))
    }
    frame = fusion.fit_frame(fit, base, plan_shift_threshold_s())
    run["frame_rows"] = len(frame)
    run["fitting_rows"] = int(frame["fit_row"].sum())
    run["scored_rows"] = int(frame["scored_row"].sum())
    t0 = time.monotonic()
    fits: dict = {}
    run["rows"] = fusion.leave_one_month_out(frame, log=fits)
    run["fits_s"] = round(time.monotonic() - t0, 1)
    run["fits"] = fits
    run["frame"] = frame
    run["stands"] = c.stands_of(fit, pd.read_parquet(FUSION_DIR / "rank_2026.parquet"))
    return run


def readings(run: dict) -> dict:
    fl = rf.floors()
    rows = run["rows"].copy()
    cov = fusion.covered(rows)
    rows["g1"] = np.where(cov, rows["fusion"], rows["base"])
    gate = fusion.gate_out_of_fold(rows)
    rows["g1n"] = fusion.apply_gate(rows, gate)
    twelve = fusion.gate_all(rows)
    out = {"gate": gate, "twelve": twelve, "rows": rows, "covered": cov}
    out["g1"] = rf.the_rule(rows, "g1", "base", fl)
    out["g1n_vs_g1"] = rf.the_rule(rows, "g1n", "g1", fl)
    out["g1n_vs_base"] = rf.the_rule(rows, "g1n", "base", fl)
    out["ungated"] = rf.the_rule(rows, "fusion", "base", fl)
    part = fusion_part(
        out["g1"]["adopted"], out["g1n_vs_g1"]["adopted"], out["g1n_vs_base"]["adopted"]
    )
    g1, gn, gb = out["g1"], out["g1n_vs_g1"], out["g1n_vs_base"]
    out["lines"] = [
        "G1 (new base): "
        + ("holds" if g1["adopted"] else f"fails ({', '.join(why(g1))})"),
        "G1n (new base): against G1 "
        + ("holds" if gn["adopted"] else f"fails ({', '.join(why(gn))})")
        + "; against the new base "
        + ("holds" if gb["adopted"] else f"fails ({', '.join(why(gb))})"),
        f"V11 FUSION PART: {part}",
        "twelve-month gate for the ranking rows: keeps "
        + ", ".join(a for a in fusion.AIRPORTS if twelve[a])
        + "; drops "
        + ", ".join(a for a in fusion.AIRPORTS if not twelve[a]),
    ]
    out["part"] = part
    short = rows[rows["y"] <= plan_shift_threshold_s()]
    beside = {}
    for arm, ref in (("g1", "base"), ("g1n", "base"), ("g1n", "g1")):
        beside[f"{arm}_vs_{ref}"] = {
            "covered_rows": rf.contrast(rows[cov], arm, ref),
            "short_rows": rf.contrast(short, arm, ref),
            "per_airport": {
                a: {
                    "all": rf.contrast(
                        rows[(rows["ADEP_mvt"] == a).to_numpy()], arm, ref
                    ),
                    "covered": rf.contrast(
                        rows[(rows["ADEP_mvt"] == a).to_numpy() & cov], arm, ref
                    ),
                }
                for a in fusion.AIRPORTS
            },
        }
    out["beside"] = beside
    # the old base: ``scripts/report_fusion.py``'s out-of-fold rows, the same rows
    lomo = pd.read_parquet(FUSION_DIR / "lomo.parquet")
    o = rows[["MVT_ID_mvt"]].merge(
        lomo, on="MVT_ID_mvt", how="left", validate="one_to_one"
    )
    for k in ("month", "y", "a_covered", "o_covered"):
        if not np.array_equal(o[k].to_numpy(), rows[k].to_numpy()):
            raise SystemExit(f"the old base's out-of-fold rows differ in {k}")
    old = o.copy()
    old["g1"] = np.where(cov, old["fusion"], old["base"])
    old["g1n"] = fusion.apply_gate(old, fusion.gate_out_of_fold(old))
    flo = rf.floors()
    out["old"] = {
        "g1": rf.the_rule(old, "g1", "base", flo),
        "g1n_vs_g1": rf.the_rule(old, "g1n", "g1", flo),
        "g1n_vs_base": rf.the_rule(old, "g1n", "base", flo),
    }
    both = rows.copy()
    both["old_base"] = old["base"].to_numpy(dtype=float)
    out["new_vs_old"] = rf.contrast(both, "base", "old_base")
    out["old_g1n_rmse"] = rf.rmse(old["y"] - old["g1n"])
    out["new_g1n_rmse"] = rf.rmse(rows["y"] - rows["g1n"])
    return out


def build_report(run: dict, got: dict, inputs: list[Path]) -> str:
    L: list[str] = []
    add = L.append
    rows, cov, gate, twelve = got["rows"], got["covered"], got["gate"], got["twelve"]
    add("# The fusion on group B's new base")
    add("")
    add(
        "Generated by `scripts/report_fusion_new_base.py` (`make fusion-new-base`). §4 of "
        "`reports/preregistered-2026-10-06-airports.md` refits group A's adopted pieces on "
        "the new base's out-of-fold predictions and re-reads them by their own rules: the "
        "fusion model of §5 of `reports/preregistered-2026-10-04.md`, gated by coverage "
        "(G1, `reports/preregistered-2026-10-06-last-week.md` §2) and by airport out of fold "
        "(G1n, `reports/preregistered-2026-10-06-g1n.md` §1). The new base is group B's "
        "combination (`reports/group-b.md`). Everything but the base is as "
        "`reports/fusion.md`: the fitting table, the features, the parameters and the "
        "floors. **G1n was chosen after 22d's figures were seen**, as its amendment says."
    )
    add("")
    add("```")
    for line in got["lines"]:
        add(line)
    add("```")
    add("")
    add(
        "**The fusion part is read from these verdicts by the working program's reading of "
        "the amendment's branches, fixed before any fit and confirmed by the owner on "
        "7 October 2026**: "
        "if G1 holds against the new base, the fusion part is G1n where G1n holds against "
        'G1, otherwise G1 (§3 of the G1n amendment, "G1n if adopted, otherwise G1"); if '
        "G1 fails, G1n is read against the new base by G1's three conditions and is the "
        "fusion part only if it holds them; otherwise none. Here G1 holds and G1n holds "
        "against G1."
        if got["part"] == "G1n" and got["g1"]["adopted"] and got["g1n_vs_g1"]["adopted"]
        else "**The fusion part is read by the working program's reading of the "
        "amendment's branches, fixed before any fit and confirmed by the owner on "
        "7 October 2026.**"
    )
    add("")
    add("## The new base")
    add("")
    bc = run["base_check"]
    L.extend(
        c.table(
            [
                "core rows",
                "rows",
                "v9's configuration (s)",
                "new base (s)",
                "contrast (s)",
            ],
            [
                [
                    label,
                    f"{bc[k]['rows']:,}",
                    f"{bc[k]['configuration']:.2f}",
                    f"{bc[k]['new_base']:.2f}",
                    f"{bc[k]['new_base'] - bc[k]['configuration']:+.2f}",
                ]
                for k, label in (
                    ("year", "twelve held-out months"),
                    ("december", "December"),
                )
            ],
            "lrrrr",
        )
    )
    add("")
    add(
        f"The fitting table with the new base and its peers recomputed: {run['frame_rows']:,} "
        f"rows, {run['fitting_rows']:,} fitting rows, {run['scored_rows']:,} scored rows."
    )
    add("")
    add("## The twelve fits")
    add("")
    L.extend(
        c.table(
            [
                "month left out",
                "training rows",
                "scored rows",
                "trees",
                "prediction sha256",
            ],
            [
                [
                    m,
                    f"{f['train_rows']:,}",
                    f"{f['scored_rows']:,}",
                    f"{f['trees']:,}",
                    f"`{f['pred_sha256']}`",
                ]
                for m, f in run["fits"].items()
            ],
            "lrrrl",
        )
    )
    add("")
    add("## The G1n gate on the new base, printed before any contrast")
    add("")
    add(
        "Out of fold, airport by held-out month (1 = the airport keeps the fusion on its "
        "covered rows that month):"
    )
    add("")
    L.extend(
        c.table(
            ["airport"] + fusion.MONTHS,
            [
                [a] + [str(int(gate.loc[a, m])) for m in fusion.MONTHS]
                for a in fusion.AIRPORTS
            ],
            "l" + "r" * 12,
        )
    )
    add("")
    add(
        f"G1 takes the fusion on {int((rows['g1'] != rows['base']).sum()):,} of the "
        f"{len(rows):,} scored rows ({int(cov.sum()):,} covered); G1n on "
        f"{int((rows['g1n'] != rows['base']).sum()):,}."
    )
    add("")
    add("## G1 against the new base")
    add("")
    L.extend(rf.rule_table("G1", "new base", got["g1"]))
    add("")
    add(f"**{got['lines'][0]}.**")
    add("")
    add("## G1n against G1 (its rule)")
    add("")
    L.extend(rf.rule_table("G1n", "G1", got["g1n_vs_g1"]))
    add("")
    add(f"**{got['lines'][1]}.**")
    add("")
    add("## Beside, deciding nothing")
    add("")
    add("G1n against the new base, which decides only if G1 fails:")
    add("")
    L.extend(rf.rule_table("G1n", "new base", got["g1n_vs_base"]))
    add("")
    b = got["beside"]
    add(
        "Covered rows only and short rows (a taxi-out of "
        f"{plan_shift_threshold_s() / 3600:.0f} h or less):"
    )
    add("")
    L.extend(
        c.table(
            ["arm against reference", "covered rows", "short rows"],
            [
                [
                    label,
                    f"{b[k]['covered_rows']['contrast']:+.2f} "
                    f"[{b[k]['covered_rows']['low']:+.2f}, {b[k]['covered_rows']['high']:+.2f}]",
                    f"{b[k]['short_rows']['contrast']:+.2f} "
                    f"[{b[k]['short_rows']['low']:+.2f}, {b[k]['short_rows']['high']:+.2f}]",
                ]
                for k, label in (
                    ("g1_vs_base", "G1 against the new base"),
                    ("g1n_vs_base", "G1n against the new base"),
                    ("g1n_vs_g1", "G1n against G1"),
                )
            ],
            "lrr",
        )
    )
    add("")
    add("Per airport, G1 against the new base:")
    add("")
    pa = b["g1_vs_base"]["per_airport"]
    L.extend(
        c.table(
            ["airport", "all rows", "contrast, all", "covered rows", "contrast, covered"],
            [
                [
                    a,
                    f"{pa[a]['all']['rows']:,}",
                    f"{pa[a]['all']['contrast']:+.2f} "
                    f"[{pa[a]['all']['low']:+.2f}, {pa[a]['all']['high']:+.2f}]",
                    f"{pa[a]['covered']['rows']:,}",
                    f"{pa[a]['covered']['contrast']:+.2f} "
                    f"[{pa[a]['covered']['low']:+.2f}, {pa[a]['covered']['high']:+.2f}]",
                ]
                for a in fusion.AIRPORTS
            ],
            "lrrrr",
        )
    )
    add("")
    u = got["ungated"]
    add(
        f"The ungated fusion against the new base: pooled {u['pooled']['contrast']:+.2f}, "
        f"December {u['december']['contrast']:+.2f}; "
        + (
            "fails "
            + ", ".join(f"{m} ({u['months'][m]['contrast']:+.2f})" for m in u["failing"])
            if u["failing"]
            else "no month fails"
        )
        + "."
    )
    add("")
    o = got["old"]
    add("The same readings on the old base (`reports/fusion.md`'s out-of-fold rows):")
    add("")
    L.extend(
        c.table(
            ["reading", "old base: pooled / Dec", "new base: pooled / Dec"],
            [
                [
                    label,
                    f"{o[k]['pooled']['contrast']:+.2f} / {o[k]['december']['contrast']:+.2f}",
                    f"{got[k]['pooled']['contrast']:+.2f} / {got[k]['december']['contrast']:+.2f}",
                ]
                for k, label in (
                    ("g1", "G1 against its base"),
                    ("g1n_vs_g1", "G1n against G1"),
                    ("g1n_vs_base", "G1n against its base"),
                )
            ],
            "lrr",
        )
    )
    add("")
    nv = got["new_vs_old"]
    add(
        f"On these scored rows the new base reads {nv['arm']:.2f} s against the old base's "
        f"{nv['reference']:.2f} s ({nv['contrast']:+.2f} [{nv['low']:+.2f}, {nv['high']:+.2f}]); "
        f"G1n reads {got['new_g1n_rmse']:.2f} s on the new base and {got['old_g1n_rmse']:.2f} s "
        "on the old."
    )
    add("")
    add("## The gate the ranking rows take")
    add("")
    add(
        "The same comparison over all twelve months' covered rows. A ranking row takes the "
        "fusion where it is covered, neither rule covers it and its airport keeps the fusion."
    )
    add("")
    covrows = rows[cov]
    tab = []
    for a in fusion.AIRPORTS:
        h = covrows[covrows["ADEP_mvt"] == a]
        tab.append(
            [
                a,
                str(int(twelve[a])),
                f"{len(h):,}",
                f"{rf.rmse(h['y'] - h['fusion']):.2f}" if len(h) else "",
                f"{rf.rmse(h['y'] - h['base']):.2f}" if len(h) else "",
            ]
        )
    L.extend(
        c.table(
            ["airport", "keeps", "covered rows", "fusion RMSE", "base RMSE"], tab, "lrrrr"
        )
    )
    add("")
    add(f"**{got['lines'][3]}.**")
    add("")
    L += c.inputs_footer(inputs)
    L += [
        "",
        "Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0.",
        "OPDI v0.0.2 from EUROCONTROL's Open Performance Data Initiative, used on the",
        "organizers' confirmation.",
    ]
    return "\n".join(L) + "\n"


def jsonable(r: dict) -> dict:
    """A reading with its frames dropped, for the record beside the report."""
    keep = (
        "rows",
        "days",
        "reference",
        "arm",
        "contrast",
        "low",
        "high",
        "floor",
        "read",
    )
    return {
        "pooled": {k: r["pooled"][k] for k in keep if k in r["pooled"]},
        "december": {k: r["december"][k] for k in keep if k in r["december"]},
        "months": {m: {k: v[k] for k in keep if k in v} for m, v in r["months"].items()},
        "adopted": r["adopted"],
        "by_reading": r["by_reading"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json", type=Path, default=FUSION_DIR / "readings-new-base.json")
    args = ap.parse_args(argv)
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}", file=sys.stderr)
        return 1
    for p in (FUSION_DIR / "fit_2025.parquet", FUSION_DIR / "lomo.parquet"):
        if not p.exists():
            print(f"no {p.relative_to(REPO)}; make fusion", file=sys.stderr)
            return 1
    run = build()
    got = readings(run)
    seeds, rules = read_reports()
    inputs = (
        [seeds, *rules]
        + [
            FUSION_DIR / f
            for f in ("fit_2025.parquet", "rank_2026.parquet", "lomo.parquet")
        ]
        + [
            BOARD / f"{n}.{x}"
            for n in group_b.NEW_BASE_RECORDS
            for x in ("parquet", "json")
        ]
        + [BOARD / f"{n}.{x}" for n in fusion.BASE_RECORDS for x in ("parquet", "json")]
    )
    text = build_report(run, got, inputs)
    frame, rows = run["frame"], got["rows"]
    c.guard(text, run["stands"], c.movement_ids(frame["MVT_ID_mvt"], rows["MVT_ID_mvt"]))
    args.out.write_text(text, encoding="utf-8")
    record = {
        "lines": got["lines"],
        "fits": {
            m: {k: v for k, v in f.items() if k != "gain"} for m, f in run["fits"].items()
        },
        "gate": {
            a: {m: bool(got["gate"].loc[a, m]) for m in fusion.MONTHS}
            for a in fusion.AIRPORTS
        },
        "gate_ranking": {a: bool(v) for a, v in got["twelve"].items()},
        "g1": jsonable(got["g1"]),
        "g1n_vs_g1": jsonable(got["g1n_vs_g1"]),
        "g1n_vs_base": jsonable(got["g1n_vs_base"]),
        "ungated": jsonable(got["ungated"]),
        "beside": got["beside"],
        "new_vs_old": got["new_vs_old"],
        "g1_pred_sha256": group_b.pred_sha256(rows["g1"]),
        "g1n_pred_sha256": group_b.pred_sha256(rows["g1n"]),
        "base_check": run["base_check"],
    }
    args.json.write_text(
        json.dumps(
            record, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)
        )
        + "\n"
    )
    print("\n".join(got["lines"]))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
