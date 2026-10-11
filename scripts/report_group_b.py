#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Group B of the last week, read under its rules: ``reports/group-b.md``.

§3 of ``reports/preregistered-2026-10-06-last-week.md`` defines W, C10, C11
and C12 and their combination, and ``reports/preregistered-2026-10-06-airports.md``
amends it: every held-out month fitted for C11 and C12, C13 (one booster per
airport), C14 (the airport-by-airport selector) and the combination rewritten
around C14. The overnight run of 6 October 2026 fitted them through
``scripts/candidates_board.py`` and recorded every fit's per-row predictions;
those records sit under ``data/processed/candidates_board/`` beside the
configuration's, and this script reads them and nothing else that was fitted.

Every derived prediction (the blends, the winter layer, C14's selection) is
formed again by ``candidates_board.GroupB`` and must hash to the record the
overnight run wrote of it, or nothing is written. Every reading is then made
by ``candidates_board.group_b_readings`` as that run made it: the five
conditions of §4 of ``reports/preregistered-2026-10-04.md`` with their floors,
the second-seed check, the other readings beside the rule's, and a
day-clustered 95% interval (2,000 draws) that decides nothing. Two
disclosures, both the copies' own: C10's weight is a selection among four
weights, and C14's forward legs let months after a leg's month inform its
choice.

``--refit NAME --refit-out DIR`` refits one recorded fit with the committed
code into ``DIR`` and compares its prediction with the record's, to the bit; it
reads the record it compares with, so it needs one.

``--fit-c14`` writes what ``scripts/make_submission.py`` reads of group B on a
board that lacks it: C14 at the ladder's seed for the twelve held-out months.
Each month's five parts are fitted where their records are absent, and the
configuration, the menu items and C14 are formed from them by
``candidates_board.GroupB``, each checked against its record where one exists.
On a board holding none, sixty fits, about eleven hours on eight cores.

Writes only aggregates. Reads the records in about four minutes on eight
cores; a refit of one CatBoost member takes about sixteen, as the records
time the overnight run's.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import candidates as cb  # noqa: E402
import candidates_board as board  # noqa: E402
import monthly as c  # noqa: E402
from taxiout.challenge import load_movements, to_pandas  # noqa: E402
from taxiout.validation import SEED as BOOTSTRAP_SEED  # noqa: E402

DATA = REPO / "data" / "challenge"
DEFAULT_OUT = REPO / "reports" / "group-b.md"
# The five candidate names of group B that coincide with stand identifiers at
# some airports; the report names them in its verdict lines, and they are names,
# not extracts. Exempt in this report alone, every other identifier still
# refused, as ``check_licence_hygiene.IDENTIFIER_EXEMPTIONS`` exempts them here
# and in the two published copies. The owner's decision of 7 October 2026.
STAND_LIKE_NAMES = frozenset({"C10", "C11", "C12", "C13", "C14"})
REFIT_RECORDS = "refit"
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
CONDS = (
    ("december", "December leg, all ten", "december:all", "better"),
    ("july", "July leg, all ten", "july:all", "better"),
    ("july_three", "July leg, the three", "july:three", "not worse"),
    ("october", "October leg, all ten", "october:all", "not worse"),
    ("year", "twelve held-out months", "year:all", "not worse"),
)


def read_reports() -> tuple[Path, list[Path]]:
    """The reports this one reads and never writes: the floors, and the rules."""
    reports = REPO / "reports"
    rules = [
        reports / "preregistered-2026-10-04.md",
        reports / "preregistered-2026-10-06-last-week.md",
        reports / "preregistered-2026-10-06-airports.md",
    ]
    return reports / "monthly_seeds.md", rules


def month_floors() -> dict[str, dict[str, str]]:
    """The floors row of ``reports/monthly_seeds.md``: over core rows and, bracketed, over short rows."""
    seeds, _ = read_reports()
    text = seeds.read_text(encoding="utf-8")
    sec = text.split("The floors (s)", 1)[1]
    for line in sec.splitlines():
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if cells and cells[0] == "floor":
            out = {"rows": {}, "short": {}}
            for name, cell in zip(MONTH_NAMES + ["year"], cells[1:]):
                m = re.fullmatch(r"([0-9.]+) \[([0-9.]+)\]", cell)
                if m is None:
                    raise SystemExit(f"the floors row of {seeds} reads {cell!r}")
                out["rows"][name], out["short"][name] = m.group(1), m.group(2)
            return out
    raise SystemExit(f"no floors row in {seeds}")


def month_label(m: str) -> str:
    return MONTH_NAMES[int(m[-2:]) - 1]


def f2(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def s2(x) -> str:
    return "n/a" if x is None else f"{x:+.2f}"


def mark(v) -> str:
    return {True: "holds", False: "FAILS", None: "not reached"}[v]


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out + [""]


def interval(r: dict) -> str:
    return f"[{r['low']:+.2f}, {r['high']:+.2f}]"


def condition_floors() -> dict[str, float]:
    rule = cb.the_rule()
    return {k.key: k.floor for k in cb.conditions(rule)}


def conditions_block(m: dict, v: dict) -> list[str]:
    floors = condition_floors()
    rows = []
    for key, label, slot, kind in CONDS:
        r = m.get("seed0", {}).get(slot)
        floor = floors[key]
        if r is None:
            rows.append([label, kind, f"{floor:.2f}", "", "", "", "", mark(None), ""])
            continue
        x = r["rows"]
        other = v.get("other_readings", {}).get(key, {})
        rows.append(
            [
                label,
                kind,
                f"{floor:.2f}",
                f"{x['rows']:,}",
                f2(x["reference"]),
                f2(x["arm"]),
                s2(x["contrast"]),
                mark(v["conditions"].get(key)),
                ", ".join(
                    f"{k} {'holds' if other.get(k) else 'fails'}"
                    for k in cb.OTHER_READINGS
                ),
            ]
        )
    out = table(
        [
            "condition",
            "kind",
            "floor (s)",
            "core rows",
            "reference (s)",
            "arm (s)",
            "contrast (s)",
            "rule",
            "other readings (decide nothing)",
        ],
        rows,
    )
    s1 = m.get(f"seed{board.board_rule().second_seed}", {})
    srows = []
    for leg in ("december", "july"):
        r0, r1 = m.get("seed0", {}).get(f"{leg}:all"), s1.get(f"{leg}:all")
        srows.append(
            [
                leg,
                s2(r0 and r0["rows"]["contrast"]),
                s2(r1 and r1["rows"]["contrast"]),
                {True: "same sign", False: "SIGN DIFFERS", None: "not measured"}[
                    v["signs"].get(leg)
                ],
            ]
        )
    out += ["Second seed (seed 1 against the base at seed 1):", ""]
    out += table(["leg", "seed 0 contrast (s)", "seed 1 contrast (s)", "check"], srows)
    out.append(f"Status: **{v['status']}**.")
    out.append("")
    return out


def intervals_block(m: dict, floors: dict) -> list[str]:
    """Beside, deciding nothing: day-clustered intervals, the months and the airports."""
    out = [
        "Beside, deciding nothing: the day-clustered 95% intervals"
        f" ({cb.BOOTSTRAP_DRAWS:,} draws) and the short rows.",
        "",
    ]
    rows = []
    for _key, label, slot, _kind in CONDS:
        r = m.get("seed0", {}).get(slot)
        if r is None:
            continue
        rows.append(
            [
                label,
                s2(r["rows"]["contrast"]),
                interval(r["rows"]),
                s2(r["short"]["contrast"]),
                interval(r["short"]),
            ]
        )
    out += table(
        ["reading", "core contrast", "interval", "short contrast", "interval"], rows
    )
    year = m.get("seed0", {}).get("year:all")
    if year:
        rows = []
        for mm, r in sorted(year["months"].items()):
            rows.append(
                [
                    month_label(mm),
                    f"{r['rows']['rows']:,}",
                    f2(r["rows"]["reference"]),
                    f2(r["rows"]["arm"]),
                    s2(r["rows"]["contrast"]),
                    interval(r["rows"]),
                    floors["rows"][month_label(mm)],
                ]
            )
        out += [
            "By held-out month (core rows; the month's floor beside, deciding nothing here):",
            "",
        ]
        out += table(
            ["month", "rows", "reference", "arm", "contrast", "interval", "floor"], rows
        )
    beside = m.get("beside_airports", {})
    if beside:
        cols = [x for x in ("year", "december", "july") if x in beside]
        rows = []
        for a in board.fusion_airports():
            row = [a]
            for x in cols:
                r = beside[x][a]
                row.append(f"{r['contrast']:+.2f} {interval(r)}")
            rows.append(row)
        out += ["Per airport, core rows, contrast and interval:", ""]
        out += table(["airport"] + cols, rows)
    return out


# --------------------------------------------------------------------------
# the fits


def group_b_fit_records() -> list[dict]:
    """The overnight run's fit records: its refit of the December booster and group B's fits."""
    keys = {m.key for m in board.group_b_members().values()} - {"combo4_booster"}
    recs = []
    refit = cb.WORK / REFIT_RECORDS
    for p in sorted(refit.glob("*__*__seed*.json")):
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    for p in sorted(cb.WORK.glob("*__*__seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "fit_s" not in r:
            continue
        key = r["arm"]["key"]
        second = r["seed"] == board.board_rule().second_seed and key == "combo4_booster"
        if key in keys or second:
            recs.append(r)
    return sorted(recs, key=lambda r: r["written_utc"])


def fits_block(recs: list[dict]) -> list[str]:
    rows, total = [], 0.0
    for r in recs:
        t = r["prepare_s"] + r["fit_s"] + r["predict_s"]
        total += t
        rows.append(
            [
                r["name"],
                r["written_utc"],
                f"{r['prepare_s']:.0f}",
                f"{r['fit_s']:.0f}",
                f"{r['predict_s']:.0f}",
                f"{r['peak_rss_gib']:.1f}",
                f"{r.get('suspended_during_s', 0.0):.0f}",
                str(r["n_estimators"]),
                f"`{r['pred_sha256']}`",
            ]
        )
    out = [
        f"{len(recs)} fits. Seconds are monotonic; peak is the process's resident peak so far"
        " (each stage of the run was one process); suspended is BOOTTIME minus MONOTONIC"
        " across the fit.",
        "",
    ]
    out += table(
        [
            "fit",
            "written (UTC)",
            "prepare s",
            "fit s",
            "predict s",
            "peak GiB",
            "suspended s",
            "trees",
            "pred_sha256",
        ],
        rows,
    )
    out.append(
        f"Sum of prepare, fit and predict over the {len(recs)} fits: {total:,.0f} s"
        f" ({total / 3600:.2f} h)."
    )
    out.append("")
    c13 = [r for r in recs if r["arm"]["key"] == "c13_booster"]
    if c13:
        out += ["The per-airport fits of the booster per airport (fit seconds):", ""]
        rows = [
            [r["name"]]
            + [f"{r['airports'][a]['fit_s']:.0f}" for a in board.fusion_airports()]
            for r in sorted(c13, key=lambda r: r["name"])
        ]
        out += table(["fit"] + list(board.fusion_airports()), rows)
    return out


# --------------------------------------------------------------------------
# the report


def checks_rows(got: dict, floors: dict) -> list[list[str]]:
    """The rule's numbers against the reports that print them, and the base."""
    rule = cb.the_rule()
    d = rule.contrast_decimals
    checks = cb.Checks()
    cb.floor_checks(checks, rule, read_reports()[0])
    rows = [
        [r["what"], r["expected"], r["got"], "ok" if r["ok"] else "STOP"]
        for r in checks.rows
    ]
    gb = board.group_b_rule()
    for kind, label, want in (
        ("rows", "core", gb.month_floors),
        ("short", "short", gb.short_floors),
    ):
        for name, value in want.items():
            e, printed = f"{value:.{d}f}", floors[kind][name]
            rows.append(
                [
                    f"floor {name} over {label} rows in `reports/monthly_seeds.md`",
                    e,
                    printed,
                    "ok" if e == printed else "STOP",
                ]
            )
    h = f"{board.board_rule().blend_weight:.1f}"
    for split, x in got["base"]["checks"].items():
        rows.append(
            [
                f"{split}: combo4 as the mean of its two members",
                x["stored"],
                x["mean"],
                "ok" if x["mean"] == x["stored"] else "STOP",
            ]
        )
        rows.append(
            [
                f"{split}: combo4 as {h} x booster + {h} x member",
                x["stored"],
                x["half"],
                "ok" if x["half"] == x["stored"] else "STOP",
            ]
        )
    return rows


def build_report(
    got: dict, recs: list[dict], refit: dict | None, inputs: list[Path]
) -> str:
    floors = month_floors()
    gb = board.group_b_rule()
    L: list[str] = []
    A = L.append
    A("# Group B of the last week")
    A("")
    A(
        "Generated by `scripts/report_group_b.py` (`make group-b`) from the records under"
        " `data/processed/candidates_board/`. Governed by §3 of"
        " `reports/preregistered-2026-10-06-last-week.md` as"
        " `reports/preregistered-2026-10-06-airports.md` amends it, read under §4 of"
        " `reports/preregistered-2026-10-04.md`. The candidates were fitted overnight on"
        " 6 October 2026; every derived prediction is formed again here and hashes to the"
        " record that run wrote. **C10's weight is a selection among four weights**, and"
        " **C14's forward legs let months after a leg's month inform its choice**, as the"
        " two copies say."
    )
    A("")
    A(
        "The intervals beside the readings are the harness's day-clustered percentile"
        f" intervals, {cb.BOOTSTRAP_DRAWS:,} draws (`taxiout.validation.cluster_bootstrap`,"
        f" seed {BOOTSTRAP_SEED}); the shared definitions of the last-week copy's §1 name"
        " 2,000 draws. The overnight run read them through the same harness and printed"
        " the same intervals under the note's count. They decide nothing."
    )
    A("")
    A("## Verdicts")
    A("")
    lines = [got["w"]["line"], got["c10"]["line"]]
    lines += [got["alone"][k]["line"] for k in ("c11", "c12", "c13")]
    lines += [got["c14"]["line"], got["combination"]["line"]]
    L.extend(lines)
    A("")
    latest = max(r["written_utc"] for r in recs)
    A(
        f"Deadline {gb.deadline_utc}: every fit read was written before it (the last at"
        f" {latest})."
    )
    A("")
    A("## The rule's numbers and the base")
    A("")
    L.extend(table(["check", "expected", "got", ""], checks_rows(got, floors)))
    A(
        "The base pooled over the twelve held-out months' core rows: "
        f"{got['base']['rmse']:.2f} s over {got['base']['core_rows']:,} rows."
    )
    A("")
    if refit is not None:
        A(
            f"Refit `{refit['name']}` with the committed code: `{refit['pred_sha256']}`,"
            f" its record's `{refit['record_sha256']}`, "
            + ("equal to the bit." if refit["equal"] else "**DIFFERENT**.")
        )
        A("")
    w = got["w"]
    A("## W, C2a's winter layer on the base")
    A("")
    rows = []
    for k, r in sorted(w["rule"].items()):
        rows.append(
            [
                k,
                r["rows"],
                r["kind"],
                f"{r['floor']:.2f}",
                s2(r["contrast"]),
                "holds" if r["holds"] else "FAILS",
                # in the order of their names, as the overnight run printed them
                ", ".join(
                    f"{x} {'holds' if ok else 'fails'}"
                    for x, ok in sorted(r["other"].items())
                ),
            ]
        )
    L.extend(
        table(
            [
                "condition",
                "rows",
                "kind",
                "floor (s)",
                "contrast (s)",
                "rule",
                "other readings",
            ],
            rows,
        )
    )
    A(
        "The winter leg is read leave one month out (each month's layer fitted on the other"
        " eleven held-out months' short rows of the base); the forward legs with the layer"
        " fitted on the held-out months strictly before each, read over core rows as"
        " section 4 reads the legs."
    )
    A("")
    L.extend(intervals_block(w["measured"], floors))
    A(w["line"])
    A("")
    c10 = got["c10"]
    half = board.board_rule().blend_weight
    A("## C10, the blend weight")
    A("")
    rows = [[k, f"{v:.4f}"] for k, v in sorted(c10["grid_rmse"].items())]
    rows.insert(2, [f"{half:.1f} (the base)", f"{c10['base_rmse']:.4f}"])
    L.extend(
        table(["booster weight w", "RMSE, twelve held-out months, core rows (s)"], rows)
    )
    A(
        f"Chosen: w = {c10['chosen']:.1f}, the lowest of the four; a selection among four"
        " weights, disclosed as such."
    )
    A("")
    L.extend(conditions_block(c10["measured"], c10["verdict"]))
    L.extend(intervals_block(c10["measured"], floors))
    A(c10["line"])
    A("")
    A("## The fits")
    A("")
    L.extend(fits_block(recs))
    for k, label in (
        ("c11", "C11, the member with C1 and C4"),
        ("c12", f"C12, {gb.c12_trees:,} trees"),
        ("c13", "C13, one booster per airport"),
    ):
        x = got["alone"][k]
        A(f"## {label}")
        A("")
        L.extend(conditions_block(x["measured"], x["verdict"]))
        L.extend(intervals_block(x["measured"], floors))
        A(x["line"])
        A("")
    ch = got["c14_choices"]
    A("## C14, the airport-by-airport selector")
    A("")
    A(
        "The choice table, printed before any contrast: per airport and held-out month, the"
        " menu item with the lowest RMSE over the airport's core rows of the other eleven"
        " held-out months (the base unless another is strictly lower). A forward leg takes"
        " its month's choice."
    )
    A("")
    A(
        "So months after a forward leg's month inform its choice, as the amendment says;"
        " the forward legs' readings below carry that disclosure."
    )
    A("")
    airports = list(board.fusion_airports())
    rows = [[month_label(m)] + [ch["table"][m][a] for a in airports] for m in cb.MONTHS]
    L.extend(table(["month"] + airports, rows))
    A("For the ranking rows (all twelve held-out months):")
    A("")
    L.extend(
        table(
            ["airport", "choice"] + ch["items"],
            [
                [a, ch["ranking"][a]]
                + [f"{ch['ranking_scores'][a][it]:.2f}" for it in ch["items"]]
                for a in airports
            ],
        )
    )
    L.extend(conditions_block(got["c14"]["measured"], got["c14"]["verdict"]))
    L.extend(intervals_block(got["c14"]["measured"], floors))
    A(got["c14"]["line"])
    A("")
    comb = got["combination"]
    A("## The combination")
    A("")
    A(
        "What cleared alone: "
        + ", ".join(
            f"{k.upper()} {'yes' if v else 'no'}" for k, v in comb["clears"].items()
        )
        + "."
    )
    A("")
    A(
        "Configuration: "
        + ("; ".join(comb["configuration"]) or "v9's, nothing cleared")
        + "."
    )
    A("")
    if "measured" in comb:
        L.extend(conditions_block(comb["measured"], comb["verdict"]))
        L.extend(intervals_block(comb["measured"], floors))
    if comb.get("why"):
        A(comb["why"] + ".")
        A("")
    A(comb["line"])
    A("")
    A(
        "The ranking rows take each airport's choice over all twelve held-out months, the"
        " table above: " + ", ".join(f"{a} {ch['ranking'][a]}" for a in airports) + "."
    )
    A("")
    L += c.inputs_footer(inputs)
    return "\n".join(L) + "\n"


def report_inputs(g: board.GroupB, recs: list[dict]) -> list[Path]:
    seeds, rules = read_reports()
    paths = [seeds, *rules, REPO / cb.BASELINE]
    for name in sorted(set(g.checked) | set(g.derived)):
        paths += [cb.record_path(name), cb.kept_path(name)]
    for r in recs:
        if (cb.WORK / REFIT_RECORDS / f"{r['name']}.json").exists():
            paths += [
                cb.WORK / REFIT_RECORDS / f"{r['name']}.{x}" for x in ("json", "parquet")
            ]
        else:
            paths += [cb.record_path(r["name"]), cb.kept_path(r["name"])]
    paths += sorted((REPO / "data" / "raw" / "metar").glob("*.csv"))
    return paths


def refit(name: str, out: Path) -> dict:
    """One recorded fit, refitted by the committed code into ``out``, against its record."""
    split, key, seed = name.split("__")
    seed = int(seed.removeprefix("seed"))
    member = board.group_b_members()[key]
    want = json.loads(cb.record_path(name).read_text(encoding="utf-8"))["pred_sha256"]
    frame = board.BoardFrame()
    s = next(x for x in cb.leg_plan() + cb.month_plan() if x.name == split)
    prep = frame.prepare(s)
    record, kept = board.fit_group_b(frame, prep, member, seed)
    out.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out / f"{name}.parquet", index=False)
    (out / f"{name}.json").write_text(
        json.dumps(record, indent=1, sort_keys=True), encoding="utf-8"
    )
    got = {
        "name": name,
        "pred_sha256": record["pred_sha256"],
        "record_sha256": want,
        "equal": record["pred_sha256"] == want,
        "fit_s": record["fit_s"],
    }
    (out / "refit.json").write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
    return got


def fit_c14() -> int:
    """C14 at seed 0 for the twelve held-out months, the records the build reads.

    Each month's five parts (the configuration's booster, C3b's member, C11's
    member, C12's and C13's boosters) are fitted by the committed code where
    their record is absent and read where it is present; the configuration,
    the menu items and C14 are then formed by ``candidates_board.GroupB``,
    each checked against its record where one exists and written where none
    does.
    """
    members = board.group_b_members()
    members["c3b_member"] = board.Member("c3b_member", backend="catboost")
    keys = ("combo4_booster", "c3b_member", "c11_member", "c12_booster", "c13_booster")
    frame = None
    for s in cb.month_plan():
        todo = [
            k for k in keys if not cb.record_path(cb.record_name(s.name, k, 0)).exists()
        ]
        if not todo:
            continue
        if frame is None:
            frame = board.BoardFrame()
        prep = frame.prepare(s)
        for k in todo:
            print(f"fitting {cb.record_name(s.name, k, 0)}", flush=True)
            cb.write_record(*board.fit_group_b(frame, prep, members[k], 0))
        del prep
    g = board.GroupB()
    of = g.c14_of(g.c14_choices())
    missing = [s.name for s in cb.month_plan() if of(s.name, 0) is None]
    if missing:
        print(f"C14 could not be formed for {missing}", file=sys.stderr)
        return 1
    print(
        f"C14 formed for {len(cb.month_plan())} months; {len(g.derived)} derived records"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--fit-c14",
        action="store_true",
        help="fit what C14's twelve held-out months lack, then form C14",
    )
    ap.add_argument("--refit", help="a recorded fit's name, refitted and compared")
    ap.add_argument("--refit-out", type=Path, help="where the refit is written")
    ap.add_argument(
        "--refit-record", type=Path, help="a refit's refit.json, printed in the report"
    )
    args = ap.parse_args(argv)
    board.point_harness()
    if args.fit_c14:
        return fit_c14()
    if args.refit:
        if args.refit_out is None:
            ap.error("--refit needs --refit-out")
        got = refit(args.refit, args.refit_out)
        print(json.dumps(got, indent=1))
        return 0 if got["equal"] else 1
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}", file=sys.stderr)
        return 1
    g = board.GroupB()
    got = board.group_b_readings(g)
    recs = group_b_fit_records()
    refit_got = None
    if args.refit_record is not None:
        refit_got = json.loads(args.refit_record.read_text(encoding="utf-8"))
    text = build_report(got, recs, refit_got, report_inputs(g, recs))
    dep = to_pandas(load_movements(DATA, ["MVT_ID_mvt", "STAND_mvt"], phase="DEP"))
    stands = c.stands_of(dep) - STAND_LIKE_NAMES
    ids = c.movement_ids(*[k["mvt_id"] for k in g.cache.values()])
    c.guard(text, stands, ids)
    args.out.write_text(text, encoding="utf-8")
    record = {
        "lines": [got["w"]["line"], got["c10"]["line"]]
        + [got["alone"][k]["line"] for k in ("c11", "c12", "c13")]
        + [got["c14"]["line"], got["combination"]["line"]],
        "c10_chosen": got["c10"]["chosen"],
        "c14_ranking": got["c14_choices"]["ranking"],
        "c14_table": got["c14_choices"]["table"],
        "derived": got["derived"],
        "checked": got["checked"],
    }
    if args.json is not None:
        args.json.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print("\n".join(record["lines"]))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
