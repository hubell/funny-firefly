#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Where the last three blocks' gains sit at Fiumicino: ``reports/fiumicino.md``.

From the held-out-month ladder's stored residuals of the twelve held-out months
(``scripts/report_monthly_ladder.py``), nothing refitted. For ``+ callsign prefix`` against ``+ stand occupancy``,
``+ destination and holidays`` against ``+ callsign prefix`` and
``+ plan differences`` against ``+ destination and holidays``, at LIRF and at
the other nine airports pooled, the change in squared error and in RMSE by
month, by flight-plan record, by the plan-lag bands of ``reports/residual.md``
and by airline designator. Every margin is pooled from the rows it names.

Usage::

    python scripts/report_fiumicino.py [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import monthly as c  # noqa: E402
from monthly import rb  # noqa: E402
from taxiout.callsign import callsign_features  # noqa: E402
from taxiout.models import LADDER_BY_KEY  # noqa: E402
from taxiout.tails import flag  # noqa: E402

DEFAULT_OUT = c.REPO / "reports" / "fiumicino.md"
AIRPORT = "LIRF"
KEYS = ("occupancy", "callsign", "destination", "plan-diffs")
BLOCKS = (
    ("callsign", "occupancy"),
    ("destination", "callsign"),
    ("plan-diffs", "destination"),
)
# ``reports/monthly_ladder.md``'s printed year-level changes, excluding flagged
# rows, and over the rows of 4 h or less, all ten airports and LIRF (its
# *Summary: every airport*), checked against the residuals here.
PRINTED_LADDER = {
    "callsign": {"all": ("-3.3", "-3.4"), "LIRF": ("-22.1", "-23.5")},
    "destination": {"all": ("-1.4", "-1.7"), "LIRF": ("-6.5", "-8.8")},
    "plan-diffs": {"all": ("-2.2", "-2.0"), "LIRF": ("-6.2", "-4.4")},
}
# The plan-lag bands of reports/residual.md, in minutes of take-off minus EOBT_1_flt.
LAG_EDGES_MIN = (0, 15, 30, 45, 60, 120, np.inf)
LAG_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-120", "over 120"]
NO_ESTIMATE = "no estimate"
BEFORE_ESTIMATE = "before the estimate"
NAMED = 10
POOLED = "the rest"


def lag_bands(lag_min: np.ndarray) -> np.ndarray:
    band = pd.cut(
        pd.Series(lag_min),
        bins=list(LAG_EDGES_MIN),
        labels=LAG_LABELS,
        right=False,
        ordered=True,
    )
    band = band.astype(object).to_numpy()
    return np.where(
        np.isnan(lag_min), NO_ESTIMATE, np.where(pd.isna(band), BEFORE_ESTIMATE, band)
    )


class Rows:
    def __init__(self) -> None:
        dep = c.departures()
        self.dep_ids = dep["MVT_ID_mvt"].to_numpy()
        y_all = c.target(dep).to_numpy()
        t = c.utc(dep["MVT_TIME_UTC_mvt"])
        lag_all = ((t - c.utc(dep["EOBT_1_flt"])).dt.total_seconds() / 60.0).to_numpy(
            dtype=float
        )
        rec_all = ~flag(dep)["flt_missing"].to_numpy(dtype=bool)
        pre_all = callsign_features(dep)["cs_prefix"].astype(str).to_numpy()
        self.stands = c.stands_of(dep)
        R, flagged, month, ids, icao = [], [], [], [], []
        self.files = []
        for i, m in enumerate(c.MONTHS):
            first = None
            cols = []
            for key in KEYS:
                p = c.MONTHLY_LADDER / f"{m}_{key}.parquet"
                self.files.append(p)
                f = pd.read_parquet(p, columns=["mvt_id", "icao", "residual", "flagged"])
                if first is None:
                    first = f
                elif not np.array_equal(
                    f["mvt_id"].to_numpy(), first["mvt_id"].to_numpy()
                ):
                    raise SystemExit(f"{m}_{key}: rows differ")
                cols.append(f["residual"].to_numpy(dtype=float))
            R.append(np.column_stack(cols))
            flagged.append(first["flagged"].to_numpy(dtype=bool))
            month.append(np.full(len(first), i))
            ids.append(first["mvt_id"].to_numpy())
            icao.append(first["icao"].astype(str).to_numpy())
        self.R = np.vstack(R)
        self.flagged = np.concatenate(flagged)
        self.month = np.concatenate(month)
        self.ids = np.concatenate(ids)
        self.icao = np.concatenate(icao)
        at = c.position(self.dep_ids, self.ids)
        self.y = y_all[at]
        self.lag = lag_all[at]
        self.record = rec_all[at]
        self.prefix = pre_all[at]
        self.band = lag_bands(self.lag)
        self.ex = ~self.flagged
        self.d4 = self.ex & (self.y <= c.DAY_SHIFT_S)


def stats(rows: Rows, mask: np.ndarray, k: int, b: int) -> dict:
    """The cell's rows, squared errors and RMSEs for rung ``k`` against ``b``."""
    out = {}
    for name, base in (("ex", rows.ex), ("d4", rows.d4)):
        m = mask & base
        n = int(m.sum())
        sk = float(np.sum(rows.R[m, k] ** 2))
        sb = float(np.sum(rows.R[m, b] ** 2))
        out[f"n_{name}"] = n
        out[f"dsse_{name}"] = sk - sb
        out[f"drmse_{name}"] = (np.sqrt(sk / n) - np.sqrt(sb / n)) if n else float("nan")
    return out


def ms(x: float) -> str:
    """Squared seconds in millions."""
    return f"{x / 1e6:+,.1f}"


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f}%"


def sq_text(s: dict, tot: float, tot4: float) -> str:
    """The change in squared error and its share; the 4 h reading in brackets
    where the cell holds rows over 4 h."""
    left = s["n_ex"] - s["n_d4"]
    out = f"{ms(s['dsse_ex'])} ({pct(s['dsse_ex'] / tot)})"
    if left > 0:
        out += f" [{ms(s['dsse_d4'])} ({pct(s['dsse_d4'] / tot4)}); {left:,} out]"
    return out


def build(rows: Rows) -> tuple[str, dict]:
    L: list[str] = []
    add = L.append
    at = rows.icao == AIRPORT
    rest = ~at
    groups = ((AIRPORT, at), ("the other nine, pooled", rest))
    summary: dict = {}
    add("# The last three blocks at Fiumicino")
    add("")
    add(
        "Generated by `scripts/report_fiumicino.py` from the held-out-month ladder's stored residuals of the "
        "twelve held-out "
        "months; nothing is refitted, every figure is computed there, and the generator refuses to write a "
        "movement identifier or a literal stand identifier. **It decides nothing.** The design is `reports/monthly_ladder.md`'s: "
        "each month of 2025 held out in turn, the model trained on the other eleven, so not forward in time."
    )
    add("")
    add(
        "Every figure is over the rows the record-quality term does not cover, as the scorer's rule implies. "
        "*Change in squared error* is the block's sum of squared residuals minus the rung's it is read against, "
        "over the cell's rows, in millions of s²; negative means the block helps. *Share* is that change as a "
        "fraction of the block's year-level change over the group the table is for, so a group's shares add to "
        "100% over a partition. *Change in RMSE* is the block's RMSE minus the other rung's over the same rows. "
        "Where a cell holds recorded taxi-outs over 4 h, the second reading follows in brackets: the change in "
        "RMSE over the rows of 4 h or less, a diagnostic, with the count it leaves out. A cell under "
        f"{c.MIN_CELL_ROWS} rows prints its count only."
    )
    add("")
    add(
        "The partitions: the held-out month; the flight-plan record as `taxiout.tails` reads it (`LOBT_flt` "
        "present), which is the reading of `reports/monthly_ladder.md`'s Annex D; the plan-lag bands of `reports/residual.md`, "
        "take-off minus `EOBT_1_flt` in minutes, with `no estimate` and `before the estimate` as their own rows; "
        "and the ICAO airline designator as `taxiout.callsign` reads it off `FLIGHT_mvt`, an aggregate over an "
        "operator's departures and not an identifier of any of them, as `reports/residual.md` prints it."
    )
    add("")

    # the reproduction of the ladder report's printed changes
    add("## The held-out-month ladder's year-level changes, recomputed")
    add("")
    chk = []
    for k_key, b_key in BLOCKS:
        k, b = KEYS.index(k_key), KEYS.index(b_key)
        for gname, gmask in (("all", np.ones(len(rows.y), bool)), (AIRPORT, at)):
            s = stats(rows, gmask, k, b)
            got = (c.signed1(s["drmse_ex"]), c.signed1(s["drmse_d4"]))
            want = PRINTED_LADDER[k_key][gname]
            chk.append(
                [
                    f"`{LADDER_BY_KEY[k_key].label}` against `{LADDER_BY_KEY[b_key].label}`",
                    "all 10" if gname == "all" else gname,
                    f"{got[0]} [{got[1]}]",
                    f"{want[0]} [{want[1]}]",
                    "yes" if got == want else "**no**",
                ]
            )
            summary.setdefault("reproduced", []).append(got == want)
    L += c.table(
        [
            "block",
            "airports",
            "here (s) [4 h or less]",
            "`monthly_ladder.md` (s)",
            "same",
        ],
        chk,
        "llrrl",
    )
    add("")

    for k_key, b_key in BLOCKS:
        k, b = KEYS.index(k_key), KEYS.index(b_key)
        label = f"`{LADDER_BY_KEY[k_key].label}` against `{LADDER_BY_KEY[b_key].label}`"
        whole = stats(rows, np.ones(len(rows.y), bool), k, b)
        year = {g: stats(rows, m, k, b) for g, m in groups}
        add(f"## {label}")
        add("")
        # per airport share
        ap_rows = []
        shares, shares4 = {}, {}
        for ap in c.ICAOS:
            m = rows.icao == ap
            if not m.any():
                continue
            s = stats(rows, m, k, b)
            shares[ap] = s["dsse_ex"] / whole["dsse_ex"]
            shares4[ap] = s["dsse_d4"] / whole["dsse_d4"]
            ap_rows.append(
                [
                    ap,
                    f"{s['n_ex']:,}",
                    sq_text(s, whole["dsse_ex"], whole["dsse_d4"]),
                    c.cell_text(
                        s["drmse_ex"],
                        s["drmse_d4"],
                        s["n_ex"],
                        s["n_ex"] - s["n_d4"],
                        c.signed1,
                    ),
                ]
            )
        ap_rows.append(
            [
                "all 10",
                f"{whole['n_ex']:,}",
                sq_text(whole, whole["dsse_ex"], whole["dsse_d4"]),
                c.cell_text(
                    whole["drmse_ex"],
                    whole["drmse_d4"],
                    whole["n_ex"],
                    whole["n_ex"] - whole["n_d4"],
                    c.signed1,
                ),
            ]
        )
        add(
            f"The year, per airport: the change in squared error, its share of the all-airport change of "
            f"{ms(whole['dsse_ex'])} million s² ({ms(whole['dsse_d4'])} million s² over the rows of 4 h or less), "
            "and the change in RMSE; the brackets are the 4 h readings, a diagnostic, where an airport has rows "
            "over 4 h."
        )
        add("")
        L += c.table(
            [
                "ICAO",
                "rows",
                "change in squared error, million s² (share of the year's change)",
                "change in RMSE (s)",
            ],
            ap_rows,
            "lrrr",
        )
        add("")
        summary[k_key] = {
            "share_lirf": shares.get(AIRPORT),
            "shares": shares,
            "dsse_all": whole["dsse_ex"],
        }

        def part_table(
            codes: np.ndarray, order: list, first: str, names=None
        ) -> list[str]:
            out_rows = []
            for j, v in enumerate(order):
                m = codes == v
                cells = [str(v) if names is None else names[j]]
                for g, gm in groups:
                    s = stats(rows, m & gm, k, b)
                    tot, tot4 = year[g]["dsse_ex"], year[g]["dsse_d4"]
                    if s["n_ex"] < c.MIN_CELL_ROWS:
                        cells += [f"({s['n_ex']:,} rows)", "", ""]
                        continue
                    cells += [
                        f"{s['n_ex']:,}",
                        sq_text(s, tot, tot4),
                        c.cell_text(
                            s["drmse_ex"],
                            s["drmse_d4"],
                            s["n_ex"],
                            s["n_ex"] - s["n_d4"],
                            c.signed1,
                        ),
                    ]
                out_rows.append(cells)
            cells = ["the year"]
            for g, gm in groups:
                s = year[g]
                cells += [
                    f"{s['n_ex']:,}",
                    sq_text(s, s["dsse_ex"], s["dsse_d4"]),
                    c.cell_text(
                        s["drmse_ex"],
                        s["drmse_d4"],
                        s["n_ex"],
                        s["n_ex"] - s["n_d4"],
                        c.signed1,
                    ),
                ]
            out_rows.append(cells)
            head = [first]
            for g, _ in groups:
                head += [
                    f"{g}: rows",
                    f"{g}: change in squared error, million s² (share)",
                    f"{g}: change in RMSE (s)",
                ]
            return c.table(head, out_rows, "l" + "rrr" * len(groups))

        add("### By month")
        add("")
        L += part_table(
            rows.month,
            list(range(12)),
            "held-out month",
            [c.month_label(m) for m in c.MONTHS],
        )
        add("")
        add("### By flight-plan record")
        add("")
        rec = np.where(rows.record, "a record", "no record")
        L += part_table(rec, ["a record", "no record"], "record")
        add("")
        add("### By plan-lag band (take-off minus `EOBT_1_flt`, minutes)")
        add("")
        L += part_table(
            rows.band, LAG_LABELS + [NO_ESTIMATE, BEFORE_ESTIMATE], "lag band"
        )
        add("")
        add("### By airline designator")
        add("")
        lirf_pre = rows.prefix[at & rows.ex]
        d_sse = (rows.R[at & rows.ex, k] ** 2) - (rows.R[at & rows.ex, b] ** 2)
        per = pd.Series(d_sse).groupby(lirf_pre).sum().sort_values()
        big = pd.Series(lirf_pre).value_counts()
        per = per[per.index.isin(big[big >= c.MIN_CELL_ROWS].index)]
        named = list(per.index[:NAMED])
        pre = np.where(np.isin(rows.prefix, named), rows.prefix, POOLED)
        add(
            f"The {NAMED} designators with the largest share of the change at {AIRPORT}, ranked there; the same "
            "designators are named for the other nine so the two sides are comparable, and every other "
            f"designator is pooled as `{POOLED}`. Only designators with {c.MIN_CELL_ROWS} rows or more at "
            f"{AIRPORT} are ranked, so that every named one prints its figures."
        )
        add("")
        L += part_table(pre, named + [POOLED], "designator")
        add("")
        # where the reduction sits, one sentence
        lirf, lirf4 = year[AIRPORT]["dsse_ex"], year[AIRPORT]["dsse_d4"]

        def two(mask) -> tuple[float, float]:
            s_ = stats(rows, mask, k, b)
            return s_["dsse_ex"] / lirf, s_["dsse_d4"] / lirf4

        def both(x: tuple[float, float]) -> str:
            return f"{pct(x[0])} [{pct(x[1])}]"

        rec_share = two(at & rows.record)
        bands = [
            v
            for v in LAG_LABELS + [NO_ESTIMATE, BEFORE_ESTIMATE]
            if stats(rows, at & (rows.band == v), k, b)["n_ex"] >= c.MIN_CELL_ROWS
        ]
        band_share = {v: two(at & (rows.band == v)) for v in bands}
        top_band = max(band_share, key=lambda v: band_share[v][0])
        top_pre = named[0]
        top_pre_share = two(at & (rows.prefix == top_pre))
        t3 = [two(at & (rows.prefix == p)) for p in named[:3]]
        top3 = (sum(x[0] for x in t3), sum(x[1] for x in t3))
        others = sorted(
            ((s, ap) for ap, s in shares.items() if ap != AIRPORT), reverse=True
        )
        sentence = (
            f"**{label}: {pct(shares[AIRPORT])} of the year-level reduction sits at {AIRPORT}** "
            f"[{pct(shares4[AIRPORT])} over the rows of 4 h or less, a diagnostic, as is every bracket in this "
            f"sentence]; the next airport is {others[0][1]} at "
            f"{both((others[0][0], shares4[others[0][1]]))}. Within {AIRPORT}, {both(rec_share)} of its change is on "
            f"rows with a flight-plan record, {both(band_share[top_band])} in the {top_band} band, the largest of the "
            f"bands with {c.MIN_CELL_ROWS} rows or more, and {both(top_pre_share)} on one designator, {top_pre} "
            f"({both(top3)} on the top three)."
        )
        add(sentence)
        add("")
        summary[k_key].update(
            {
                "sentence": sentence,
                "rec_share": rec_share,
                "top_band": top_band,
                "shares4": shares4,
                "band_share": band_share[top_band],
                "top_prefix": top_pre,
                "top_prefix_share": top_pre_share,
                "top3": top3,
            }
        )
    L += c.inputs_footer(rb._inputs() + rows.files)
    return "\n".join(L) + "\n", summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    before = c.fingerprint()
    rows = Rows()
    text, summary = build(rows)
    c.write(args.out, text, rows.stands, c.movement_ids(rows.ids), before)
    if not all(summary["reproduced"]):
        print("the ladder report's printed changes were not reproduced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
