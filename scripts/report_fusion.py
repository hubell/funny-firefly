#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The fusion model and its gates, read under their rules: ``reports/fusion.md``.

Builds the fusion model's two feature tables from the adsb.lol cache
(``data/raw/adsb/``, ``make adsb``), OPDI's matched tables
(``data/interim/opdi/matched/``, ``make opdi``) and the challenge data, and
writes them to ``data/processed/fusion/`` for ``scripts/make_submission.py``.
Then, with the adopted configuration's twelve held-out months at the ladder's
seed as the base (``data/processed/candidates_board/``), it fits §5's model
leaving one month out twelve times and reads, in the order they were
pre-registered:

1. **The fusion** under §5's rule (``reports/preregistered-2026-10-04.md``),
   against the base on every scored row: better pooled by more than the year's
   floor, not worse in any month by more than its floor, and better on
   December, fitted on January to November, by more than December's floor.
2. **G1**, the fusion where a row is covered and the base elsewhere, under the
   same rule against the base (``reports/preregistered-2026-10-06-last-week.md``
   §2).
3. **G1n**, G1 gated by airport out of fold, under the same three conditions
   with G1 as the reference (``reports/preregistered-2026-10-06-g1n.md`` §1),
   and the twelve-month gate the ranking rows take.

Each condition is read with ``candidates.holds`` at the rule's precision, the
other readings beside; the floors are ``reports/monthly_seeds.md``'s; each
contrast carries a day-clustered 95% interval (2,000 draws, seed 20260902)
that decides nothing. G1n was chosen after 22d's figures were seen, and the
report says so where it reads it.

Writes only aggregates. Takes about six minutes on eight cores: three for the
feature tables, two and a half for the twelve fits.

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
from taxiout import fusion  # noqa: E402
from taxiout.adsb import NOTICE  # noqa: E402
from taxiout.challenge import movement_files, ranking_months  # noqa: E402
from taxiout.opdi import match_months  # noqa: E402
from taxiout.tails import plan_shift_threshold_s  # noqa: E402
from taxiout.validation import cluster_bootstrap, cluster_sums  # noqa: E402

DATA = REPO / "data" / "challenge"
ADSB = REPO / "data" / "raw" / "adsb"
MATCHED = REPO / "data" / "interim" / "opdi" / "matched"
BOARD = REPO / "data" / "processed" / "candidates_board"
OUT_DIR = REPO / "data" / "processed" / "fusion"
DEFAULT_OUT = REPO / "reports" / "fusion.md"


def read_reports() -> tuple[Path, list[Path]]:
    """The reports this one reads and never writes: the floors, and the rules."""
    reports = REPO / "reports"
    rules = [
        reports / "preregistered-2026-10-04.md",
        reports / "preregistered-2026-10-06-last-week.md",
        reports / "preregistered-2026-10-06-g1n.md",
    ]
    return reports / "monthly_seeds.md", rules


DRAWS = 2000
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
READINGS = ("rule", *cb.OTHER_READINGS)
OPDI_NOTE = (
    "o_ and i_ columns: OPDI v0.0.2 (EUROCONTROL Open Performance Data Initiative), "
    "flight events and flight lists, matched by taxiout.opdi"
)
CHALLENGE_NOTE = (
    "Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network under "
    "the terms of the PRC Data Challenge 2026; not redistributable, never deposited"
)


# --------------------------------------------------------------------------
# the rule


def floors() -> dict[str, float]:
    """The floors row of ``reports/monthly_seeds.md``: each month's and the year's."""
    seeds, _ = read_reports()
    text = seeds.read_text(encoding="utf-8")
    sec = text.split("The floors (s)", 1)[1]
    for line in sec.splitlines():
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if cells and cells[0] == "floor":
            vals = [x.split()[0] for x in cells[1:]]
            return {k: float(v) for k, v in zip(MONTH_NAMES + ["year"], vals)}
    raise SystemExit(f"no floors row in {seeds}")


def rmse(r) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.sqrt(np.mean(r**2)))


def contrast(sub: pd.DataFrame, arm: str, ref: str, with_interval: bool = True) -> dict:
    """Arm RMSE minus reference RMSE on ``sub``'s rows, with the day-clustered interval."""
    ar = (sub["y"] - sub[arm]).to_numpy(dtype=float)
    br = (sub["y"] - sub[ref]).to_numpy(dtype=float)
    out = {
        "rows": int(len(sub)),
        "days": int(sub["day"].nunique()),
        "reference": rmse(br),
        "arm": rmse(ar),
    }
    out["contrast"] = out["arm"] - out["reference"]
    if with_interval and len(sub):
        sums = cluster_sums({"arm": ar, "base": br}, sub["day"].to_numpy())
        got = cluster_bootstrap(
            [{"s": sums}],
            lambda r: r["s"]["arm"] - r["s"]["base"],
            draws=DRAWS,
            seed=fusion.SEED,
        )
        out["low"], out["high"] = got.low, got.high
    return out


def the_rule(rows: pd.DataFrame, arm: str, ref: str, fl: dict[str, float]) -> dict:
    """§5's three conditions, each read with ``candidates.holds`` at the rule's precision."""
    dec = cb.the_rule().contrast_decimals
    out = {"months": {}}
    for mm, m in zip(fusion.MONTHS, MONTH_NAMES):
        r = contrast(rows[rows["month"] == mm], arm, ref)
        r["floor"] = fl[m]
        r["read"] = {k: cb.holds_read(k, "not worse", r, fl[m], dec) for k in READINGS}
        out["months"][m] = r
    p = contrast(rows, arm, ref)
    p["floor"] = fl["year"]
    p["read"] = {k: cb.holds_read(k, "better", p, fl["year"], dec) for k in READINGS}
    d = dict(out["months"]["Dec"])
    d["floor"] = fl["Dec"]
    d["read"] = {k: cb.holds_read(k, "better", d, fl["Dec"], dec) for k in READINGS}
    out["pooled"], out["december"] = p, d
    out["by_reading"] = {
        k: p["read"][k]
        and all(out["months"][m]["read"][k] for m in MONTH_NAMES)
        and d["read"][k]
        for k in READINGS
    }
    out["adopted"] = out["by_reading"]["rule"]
    out["failing"] = [m for m in MONTH_NAMES if not out["months"][m]["read"]["rule"]]
    return out


def rule_table(name: str, ref_name: str, r: dict) -> list[str]:
    rows = []

    def row(label, x, kind):
        iv = f"[{x['low']:+.2f}, {x['high']:+.2f}]" if "low" in x else ""
        others = " / ".join(
            "holds" if x["read"][k] else "fails" for k in cb.OTHER_READINGS
        )
        rows.append(
            [
                label,
                f"{x['rows']:,}",
                str(x["days"]),
                f"{x['reference']:.2f}",
                f"{x['arm']:.2f}",
                f"{x['contrast']:+.2f} ({x['contrast']:+.6f})",
                iv,
                f"{kind}, {x['floor']:.2f}",
                "holds" if x["read"]["rule"] else "**fails**",
                others,
            ]
        )

    row("(1) pooled, twelve months", r["pooled"], "better")
    row("(3) December, fitted Jan to Nov", r["december"], "better")
    for m in MONTH_NAMES:
        row(f"(2) {m}", r["months"][m], "not worse")
    header = [
        "read on",
        "rows",
        "days",
        ref_name,
        name,
        "contrast",
        "interval",
        "condition",
        "rule",
        " / ".join(cb.OTHER_READINGS),
    ]
    return c.table(header, rows, "lrrrrrl" + "l" * (len(header) - 7))


def per_airport(rows: pd.DataFrame, arm: str, ref: str) -> list[list[str]]:
    out = []
    for a in fusion.AIRPORTS:
        x = contrast(rows[rows["ADEP_mvt"] == a], arm, ref)
        out.append(
            [
                a,
                f"{x['rows']:,}",
                f"{x['reference']:.2f}",
                f"{x['arm']:.2f}",
                f"{x['contrast']:+.2f}",
                f"[{x['low']:+.2f}, {x['high']:+.2f}]",
            ]
        )
    return out


# --------------------------------------------------------------------------
# the run


def tables() -> tuple[dict, dict]:
    """The two feature tables, written to ``data/processed/fusion/``: what
    ``scripts/make_submission.py`` reads, and what ``--tables-only`` writes."""
    run: dict = {}
    t0 = time.monotonic()
    got = fusion.feature_tables(DATA, ADSB, MATCHED)
    run["tables_s"] = round(time.monotonic() - t0, 1)
    meta = {
        "licence": NOTICE,
        "opdi_source": OPDI_NOTE,
        "challenge": CHALLENGE_NOTE,
        "adsb_days_without_table": got["days_without_table"],
    }
    run["fit_sha256"] = fusion.write_table(
        got["fit"][fusion.TABLE_COLUMNS],
        OUT_DIR / "fit_2025.parquet",
        {**meta, "days": got["fit_days"]},
    )
    run["rank_sha256"] = fusion.write_table(
        got["rank"][fusion.TABLE_COLUMNS],
        OUT_DIR / "rank_2026.parquet",
        {**meta, "days": got["rank_days"]},
    )
    return run, got


def build() -> dict:
    run, got = tables()
    run["rules"] = got["rules"]
    run["fit_days"] = got["fit_days"]
    run["days_without_table"] = got["days_without_table"]
    run["absent_from_matched"] = got["absent_from_matched"]
    run["fit_rows"] = len(got["fit"])
    run["rank_rows"] = len(got["rank"])
    shares = {}
    for name, t in (("fit", got["fit"]), ("rank", got["rank"])):
        g = t[(t["before_0100"] == 0) & (t["rules_row"] == 0)]
        shares[name] = {
            a: float(fusion.covered(h).mean())
            for a, h in g.groupby("ADEP_mvt", sort=True)
        }
    run["covered_share"] = shares
    run["stands"] = c.stands_of(got["fit"], got["rank"])

    base = fusion.read_base(BOARD)
    frame = fusion.fit_frame(got["fit"], base, plan_shift_threshold_s())
    run["frame_rows"] = len(frame)
    run["fitting_rows"] = int(frame["fit_row"].sum())
    run["scored_rows"] = int(frame["scored_row"].sum())
    t0 = time.monotonic()
    fits: dict = {}
    rows = fusion.leave_one_month_out(frame, log=fits)
    run["fits_s"] = round(time.monotonic() - t0, 1)
    run["fits"] = fits
    rows.to_parquet(OUT_DIR / "lomo.parquet", index=False)
    run["frame"] = frame
    run["rows"] = rows
    return run


def build_report(run: dict, inputs: list[Path]) -> str:
    fl = floors()
    rows = run["rows"].copy()
    cov = fusion.covered(rows)
    rows["g1"] = np.where(cov, rows["fusion"], rows["base"])
    gate = fusion.gate_out_of_fold(rows)
    rows["g1n"] = fusion.apply_gate(rows, gate)
    twelve = fusion.gate_all(rows)
    r_fusion = the_rule(rows, "fusion", "base", fl)
    r_g1 = the_rule(rows, "g1", "base", fl)
    r_g1n = the_rule(rows, "g1n", "g1", fl)
    r_g1n_base = the_rule(rows, "g1n", "base", fl)
    lines = {
        "fusion": "FUSION: ADOPTED" if r_fusion["adopted"] else "FUSION: not adopted",
        "g1": "G1: ADOPTED" if r_g1["adopted"] else "G1: not adopted",
        "g1n": "G1n: ADOPTED" if r_g1n["adopted"] else "G1n: not adopted",
    }
    run["verdicts"] = lines
    run["gate"] = gate
    run["twelve"] = twelve

    L: list[str] = []
    add = L.append
    add("# The fusion model and its gates")
    add("")
    add(
        "Generated by `scripts/report_fusion.py` (`make fusion`). The fusion model of §5 of "
        "`reports/preregistered-2026-10-04.md`, with the four amendments that define its "
        "events and features (`reports/preregistered-2026-10-04-adsb-leg.md`, `-opdi.md`, "
        "`-adsb-after-gate.md`, `-opdi-tie.md`), read under §5's rule, then gated by coverage "
        "(G1, `reports/preregistered-2026-10-06-last-week.md` §2) and by airport out of fold "
        "(G1n, `reports/preregistered-2026-10-06-g1n.md` §1). **G1n was chosen after 22d's "
        "figures were seen**, as its amendment says."
    )
    add("")
    add("```")
    for k in ("fusion", "g1", "g1n"):
        add(lines[k])
    add("```")
    add("")
    add("## The rows")
    add("")
    rl = run["rules"]
    add(
        f"The rules' rows, fitted on the training year by `taxiout.tails`: {rl['cells']} "
        f"covered cells, {rl['train_term']:,} training rows under the term, "
        f"{rl['plan_shift_n']} plan-shifted ({rl['plan_shift_long']} of them long); "
        f"{rl['rank_term']} and {rl['rank_plan_shift']} ranking rows."
    )
    add("")
    add(
        f"Fitting days: {len(run['fit_days'])} of 2025 with an adsb.lol table, less the two "
        f"training pilot days. Days without a table: {len(run['days_without_table'])} "
        "(`data/raw/adsb/MANIFEST.json` gives each reason). The fitting days hold "
        f"{run['fit_rows']:,} departures, the ranking months {run['rank_rows']:,}; "
        f"{run['absent_from_matched']} training departures stored in the next month's file "
        "have no OPDI row and enter with those features missing. Fitting rows (neither "
        f"rule, a taxi-out of 4 h or less): {run['fitting_rows']:,}; scored rows (neither "
        f"rule): {run['scored_rows']:,}."
    )
    add("")
    add(
        "Share of departures covered (adsb.lol or OPDI), at 01:00Z or later and neither "
        "rule covering them:"
    )
    add("")
    L.extend(
        c.table(
            ["airport", "fitting days", "ranking months"],
            [
                [
                    a,
                    f"{run['covered_share']['fit'].get(a, 0.0):.3f}",
                    f"{run['covered_share']['rank'].get(a, 0.0):.3f}",
                ]
                for a in fusion.AIRPORTS
            ],
            "lrr",
        )
    )
    add("")
    add("## The twelve fits")
    add("")
    add(
        "§5's model, each month's scored rows predicted by the fit on the other eleven "
        "months' fitting rows; December's fit is the January-to-November fit."
    )
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
    add("## 1. The fusion under §5's rule")
    add("")
    add(
        "Contrast = fusion RMSE minus base RMSE on the same scored rows, in seconds. The "
        "rule's reading decides; the other readings are printed beside it."
    )
    add("")
    L.extend(rule_table("fusion", "base", r_fusion))
    add("")
    add(
        f"**{lines['fusion']}.**"
        + (
            f" Condition (2) fails in {', '.join(r_fusion['failing'])}."
            if r_fusion["failing"]
            else ""
        )
    )
    add("")
    add("## 2. G1, the fusion gated by coverage")
    add("")
    add(
        f"A row takes the fusion where `a_covered` or `o_covered` is 1: {int(cov.sum()):,} "
        f"of the {len(rows):,} scored rows. Nothing is refitted."
    )
    add("")
    L.extend(rule_table("G1", "base", r_g1))
    add("")
    add(f"**{lines['g1']}.**")
    add("")
    add("Per airport, pooled over the twelve months, beside the rule:")
    add("")
    L.extend(
        c.table(
            ["airport", "rows", "base", "G1", "contrast", "interval"],
            per_airport(rows, "g1", "base"),
            "lrrrrl",
        )
    )
    add("")
    add("## 3. G1n, the fusion gated by coverage and by airport, out of fold")
    add("")
    add(
        "Chosen after 22d's figures were seen. Per held-out month and airport, over the "
        "airport's covered rows of the other eleven months, the fusion is kept only if its "
        "RMSE is strictly lower than the base's (1 = kept):"
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
        f"Of the {int(cov.sum()):,} covered rows, "
        f"{int((rows['g1n'] != rows['base']).sum()):,} take the fusion under G1n; "
        f"{int((rows['g1n'] != rows['g1']).sum()):,} differ from G1. The rule, G1 as the "
        "reference:"
    )
    add("")
    L.extend(rule_table("G1n", "G1", r_g1n))
    add("")
    add(f"**{lines['g1n']}.**")
    add("")
    add("Beside, deciding nothing: G1n against the base.")
    add("")
    L.extend(rule_table("G1n", "base", r_g1n_base))
    add("")
    add("Per airport, pooled over the twelve months, G1n against G1:")
    add("")
    L.extend(
        c.table(
            ["airport", "rows", "G1", "G1n", "contrast", "interval"],
            per_airport(rows, "g1n", "g1"),
            "lrrrrl",
        )
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
                f"{rmse(h['y'] - h['fusion']):.2f}" if len(h) else "",
                f"{rmse(h['y'] - h['base']):.2f}" if len(h) else "",
            ]
        )
    L.extend(
        c.table(
            ["airport", "keeps", "covered rows", "fusion RMSE", "base RMSE"], tab, "lrrrr"
        )
    )
    add("")
    add(
        "Kept at "
        + ", ".join(a for a in fusion.AIRPORTS if twelve[a])
        + "; dropped at "
        + ", ".join(a for a in fusion.AIRPORTS if not twelve[a])
        + "."
    )
    add("")
    L += c.inputs_footer(inputs)
    L += [
        "",
        "Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0.",
        "OPDI v0.0.2 from EUROCONTROL's Open Performance Data Initiative, used on the",
        "organizers' confirmation.",
    ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json", type=Path, default=OUT_DIR / "readings.json")
    ap.add_argument(
        "--tables-only",
        action="store_true",
        help="write the two feature tables the submission reads, checked against"
        " C17's fitting days, and nothing else: no fit and no report",
    )
    args = ap.parse_args(argv)
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}", file=sys.stderr)
        return 1
    if args.tables_only:
        run, got = tables()
        print(
            f"fitting days: {len(got['fit_days'])}, sha256 "
            f"{fusion.check_fit_days(got['fit_days'])}"
        )
        print(f"fit_2025.parquet {len(got['fit']):,} rows, sha256 {run['fit_sha256']}")
        print(f"rank_2026.parquet {len(got['rank']):,} rows, sha256 {run['rank_sha256']}")
        return 0
    run = build()
    inputs = (
        list(movement_files(DATA))
        + [DATA / "ranking.parquet", read_reports()[0], *read_reports()[1]]
        + sorted(ADSB.glob("*.parquet"))
        + [MATCHED / f"{ym}.parquet" for ym in match_months(ranking_months(DATA))]
        + [BOARD / f"{n}.{x}" for n in fusion.BASE_RECORDS for x in ("parquet", "json")]
    )
    text = build_report(run, inputs)
    frame, rows = run["frame"], run["rows"]
    ids = c.movement_ids(frame["MVT_ID_mvt"], rows["MVT_ID_mvt"])
    c.guard(text, run["stands"], ids)
    args.out.write_text(text, encoding="utf-8")
    record = {
        k: run[k]
        for k in (
            "rules",
            "fit_sha256",
            "rank_sha256",
            "fit_rows",
            "rank_rows",
            "verdicts",
        )
    }
    record["fits"] = {
        m: {k: v for k, v in f.items() if k != "gain"} for m, f in run["fits"].items()
    }
    record["gate"] = {
        a: {m: bool(run["gate"].loc[a, m]) for m in fusion.MONTHS}
        for a in fusion.AIRPORTS
    }
    record["twelve_month_gate"] = {a: bool(v) for a, v in run["twelve"].items()}
    args.json.write_text(json.dumps(record, indent=1) + "\n")
    print("\n".join(run["verdicts"].values()))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
