#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Istanbul's February 2025, day by day, and its January 2026: ``reports/istanbul.md``.

From the stored residuals of the five-seed average (``reports/monthly_seeds.md``), the
training and ranking files, the record-quality term, and the METAR cache with
``taxiout.metar``'s own regime functions; the plan lag and the arrivals'
control are ``taxiout.exposure``'s, with the record-quality term fitted on the
whole training year and predicted on both sides, as ``report_congestion.py``
fits it for its *Regime exposure*. Nothing is fitted here and nothing is
predicted about the ranking rows.

Usage::

    python scripts/report_istanbul.py [--out PATH]
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
import report_monthly_seeds as rs  # noqa: E402
from taxiout.challenge import load_ranking, to_pandas  # noqa: E402
from taxiout.exposure import (  # noqa: E402
    ARRIVAL_LATE_S,
    LAG_HI_S,
    LAG_LO_S,
    TAXI_IN_HI_S,
    TAXI_IN_LO_S,
    arrival_rows,
    departure_rows,
)
from taxiout.features import MOVEMENT_COLUMNS  # noqa: E402
from taxiout.metar import load_metar, regime_by_hour  # noqa: E402
from taxiout.tails import TailModel  # noqa: E402

DEFAULT_OUT = c.REPO / "reports" / "istanbul.md"
AIRPORT = "LTFM"
FEB = "2025-02"
JAN26 = "2026-01"
LOW_VIS_BAND = 2.0  # ``taxiout.metar``'s band at which visibility is under 1,500 m
FLAGS = (
    ("freezing", "freezing"),
    ("low visibility", "low_vis"),
    ("precipitation", "precip"),
)


def day_of(ts) -> np.ndarray:
    return c.utc(pd.Series(np.asarray(ts))).dt.day.to_numpy()


def month_of(ts) -> np.ndarray:
    return c.utc(pd.Series(np.asarray(ts))).dt.strftime("%Y-%m").to_numpy()


def weather_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day at the airport: hours with a regime reading, and hours in each flag."""
    h = hourly[hourly["station"] == AIRPORT].copy()
    t = c.utc(h["hour"])
    h["month"] = t.dt.strftime("%Y-%m").to_numpy()
    h["day"] = t.dt.day.to_numpy()
    h["low_vis"] = np.where(
        h["vis_band"].notna(), (h["vis_band"] >= LOW_VIS_BAND).astype(float), np.nan
    )
    g = h.groupby(["month", "day"])
    return pd.DataFrame(
        {
            "hours": g.size(),
            "freezing": g["freezing"].sum(),
            "low_vis": g["low_vis"].sum(),
            "precip": g["precip"].sum(),
        }
    )


def monthly_shares(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly[hourly["station"] == AIRPORT].copy()
    t = c.utc(h["hour"])
    h["month"] = t.dt.strftime("%Y-%m").to_numpy()
    h["low_vis"] = np.where(
        h["vis_band"].notna(), (h["vis_band"] >= LOW_VIS_BAND).astype(float), np.nan
    )
    g = h.groupby("month")
    return pd.DataFrame(
        {
            "hours": g.size(),
            "freezing": g["freezing"].mean(),
            "low_vis": g["low_vis"].mean(),
            "precip": g["precip"].mean(),
        }
    )


class Data:
    def __init__(self) -> None:
        dep = c.departures()
        self.dep = dep
        y = c.target(dep).to_numpy()
        term = TailModel().fit(dep)
        flagged = term.flagged(dep).to_numpy(dtype=bool)
        rows = departure_rows(dep, flagged)
        self.dep_rows = pd.DataFrame(
            {
                "icao": rows["icao"].to_numpy(),
                "month": rows["month"].to_numpy(),
                "day": day_of(dep["MVT_TIME_UTC_mvt"]),
                "lag": rows["lag_eobt_s"].to_numpy(),
            }
        )
        moves = rb.load_all_movements()
        arr = arrival_rows(moves)
        self.arr = pd.DataFrame(
            {
                "icao": arr["icao"].to_numpy(),
                "month": arr["month"].to_numpy(),
                "day": day_of(moves.loc[arr.index, "MVT_TIME_UTC_mvt"]),
                "taxi_in": arr["arr_taxi_in_s"].to_numpy(),
                "late": arr["arr_late_s"].to_numpy(),
            }
        )
        del moves
        rank = c.ranking_departures()
        self.rank = rank
        rflag = term.flagged(rank).to_numpy(dtype=bool)
        rrows = departure_rows(rank, rflag)
        self.rank_rows = pd.DataFrame(
            {
                "icao": rrows["icao"].to_numpy(),
                "month": rrows["month"].to_numpy(),
                "day": day_of(rank["MVT_TIME_UTC_mvt"]),
                "lag": rrows["lag_eobt_s"].to_numpy(),
                "flagged": rflag,
            }
        )
        rmoves = to_pandas(load_ranking(c.DATA, list(MOVEMENT_COLUMNS))).reset_index(
            drop=True
        )
        rarr = arrival_rows(rmoves)
        self.rank_arr = pd.DataFrame(
            {
                "icao": rarr["icao"].to_numpy(),
                "month": rarr["month"].to_numpy(),
                "day": day_of(rmoves.loc[rarr.index, "MVT_TIME_UTC_mvt"]),
                "taxi_in": rarr["arr_taxi_in_s"].to_numpy(),
                "late": rarr["arr_late_s"].to_numpy(),
            }
        )
        del rmoves
        self.hourly = regime_by_hour(load_metar(c.METAR))
        self.history = regime_by_hour(load_metar(c.HISTORY))
        self.wx = weather_hours(self.hourly)
        # the five-seed average's residuals at the airport, every held-out month
        st = rs.Seeds(dep)
        self.seed_files = st.files
        parts = []
        for m, d in st.months.items():
            at = d["icao"] == AIRPORT
            pos = c.position(dep["MVT_ID_mvt"].to_numpy(), d["ids"][at])
            parts.append(
                pd.DataFrame(
                    {
                        "month": m,
                        "day": day_of(dep["MVT_TIME_UTC_mvt"].to_numpy()[pos]),
                        "avg": d["avg"][at],
                        "flagged": d["flagged"][at],
                        "y": d["y"][at],
                    }
                )
            )
        self.err = pd.concat(parts, ignore_index=True)
        self.term_rows = int(self.err["flagged"].sum())
        self.over4 = int((self.err["y"] > c.DAY_SHIFT_S).sum())
        self.y = y


def err_cell(e: pd.DataFrame) -> dict:
    ex = ~e["flagged"].to_numpy(dtype=bool)
    r = e["avg"].to_numpy(dtype=float)[ex]
    yy = e["y"].to_numpy(dtype=float)[ex]
    d4 = yy <= c.DAY_SHIFT_S
    return {
        "n": int(ex.sum()),
        "sse": float(np.sum(r**2)),
        "s1": float(np.sum(r)),
        "n4": int(d4.sum()),
        "sse4": float(np.sum(r[d4] ** 2)),
    }


def mean_of(values: np.ndarray) -> tuple[float, int]:
    v = values[np.isfinite(values)]
    return (float(v.mean()) if len(v) else float("nan"), int(len(v)))


def build(d: Data) -> tuple[str, dict]:
    L: list[str] = []
    add = L.append
    log: dict = {}
    days = list(range(1, 29))
    dr = d.dep_rows[d.dep_rows["icao"] == AIRPORT]
    ar = d.arr[d.arr["icao"] == AIRPORT]
    feb_dep, oth_dep = (
        dr[dr["month"] == FEB],
        dr[(dr["month"] != FEB) & dr["month"].str.startswith("2025-")],
    )
    feb_arr, oth_arr = (
        ar[ar["month"] == FEB],
        ar[(ar["month"] != FEB) & ar["month"].str.startswith("2025-")],
    )
    feb_err, oth_err = d.err[d.err["month"] == FEB], d.err[d.err["month"] != FEB]
    wx = d.wx.reset_index()
    feb_wx = wx[wx["month"] == FEB]
    oth_wx = wx[(wx["month"] != FEB) & wx["month"].str.startswith("2025-")]
    feb_month = err_cell(feb_err)
    feb_rmse = np.sqrt(feb_month["sse"] / feb_month["n"])

    add("# Istanbul's February, and January 2026")
    add("")
    add(
        "Generated by `scripts/report_istanbul.py` from the five-seed average's stored residuals "
        "(`reports/monthly_seeds.md`), the training and ranking files and the METAR cache; every figure is computed there, and the "
        "generator refuses to write a movement identifier or a literal stand identifier. **It decides nothing.**"
    )
    add("")
    add("## What is read")
    add("")
    add(
        "The error is the five-seed average's, the model that ships, on the held-out-month splits: each month of "
        "2025 scored by a fit on the other eleven, so not forward in time. RMSE and bias (prediction minus recorded "
        f"taxi-out) are over the rows the record-quality term does not cover; at {AIRPORT} it covers "
        f"{d.term_rows} rows in the year, and {d.over4} rows record more than 4 h, so the second reading over the "
        "rows of 4 h or less equals the first everywhere and is printed once."
    )
    add("")
    add(
        "The weather hours are `taxiout.metar.regime_by_hour` on the cached reports under `data/raw/metar/`, per UTC "
        "day: an hour is freezing, in precipitation or in low visibility (the band of under 1,500 m, `vis_band` 2) "
        "as the model's own weather block reads it, and *hours* counts the hours with a reading. The plan lag is "
        "take-off minus `EOBT_1_flt`, the mean over `taxiout.exposure.departure_rows`: departures unflagged by the "
        "record-quality term and carrying a flight-plan record, the lag windowed on its own to "
        f"{LAG_LO_S / 60:.0f} to {LAG_HI_S / 60:.0f} min. The arrivals' control is `taxiout.exposure.arrival_rows` at "
        f"{AIRPORT}: the mean taxi-in between {TAXI_IN_LO_S / 60:.0f} and {TAXI_IN_HI_S / 60:.0f} min, and the mean "
        f"lateness, landing minus schedule within {ARRIVAL_LATE_S / 3600:.0f} h either way. The term is fitted on the "
        "whole training year and predicted on both sides, as `scripts/report_congestion.py` does for its *Regime "
        "exposure*."
    )
    add("")
    add(
        "Beside each February day, after the slash, the same day of the month in the other eleven months of 2025, "
        "pooled: the error over all their rows, the means over all their rows, the departures and the weather "
        "hours as a mean per day."
    )
    add("")

    # 1 ---------------------------------------------------------------
    add(f"## {AIRPORT}, February 2025, day by day")
    add("")
    rows = []
    feb_days = {}
    for day in days:
        fe = err_cell(feb_err[feb_err["day"] == day])
        oe = err_cell(oth_err[oth_err["day"] == day])
        n_oth_months = int(oth_err.loc[oth_err["day"] == day, "month"].nunique())
        fdep = int((feb_err["day"] == day).sum())
        odep = int((oth_err["day"] == day).sum())
        lag_f, nl_f = mean_of(
            feb_dep.loc[feb_dep["day"] == day, "lag"].to_numpy(dtype=float)
        )
        lag_o, _ = mean_of(
            oth_dep.loc[oth_dep["day"] == day, "lag"].to_numpy(dtype=float)
        )
        ti_f, _ = mean_of(
            feb_arr.loc[feb_arr["day"] == day, "taxi_in"].to_numpy(dtype=float)
        )
        ti_o, _ = mean_of(
            oth_arr.loc[oth_arr["day"] == day, "taxi_in"].to_numpy(dtype=float)
        )
        la_f, _ = mean_of(
            feb_arr.loc[feb_arr["day"] == day, "late"].to_numpy(dtype=float)
        )
        la_o, _ = mean_of(
            oth_arr.loc[oth_arr["day"] == day, "late"].to_numpy(dtype=float)
        )
        fw = feb_wx[feb_wx["day"] == day]
        ow = oth_wx[oth_wx["day"] == day]
        n_ow = max(len(ow), 1)
        rm_f = np.sqrt(fe["sse"] / fe["n"]) if fe["n"] else float("nan")
        rm_o = np.sqrt(oe["sse"] / oe["n"]) if oe["n"] else float("nan")
        feb_days[day] = {
            "rmse": rm_f,
            "bias": -fe["s1"] / fe["n"] if fe["n"] else float("nan"),
            "n": fe["n"],
            "freezing": float(fw["freezing"].sum()),
            "low_vis": float(fw["low_vis"].sum()),
            "precip": float(fw["precip"].sum()),
            "hours": float(fw["hours"].sum()),
            "lag": lag_f,
        }
        rows.append(
            [
                f"{day}",
                f"{fdep:,} / {odep / max(n_oth_months, 1):,.0f}",
                f"{c.fmt(rm_f)} / {c.fmt(rm_o)}",
                f"{c.signed(-fe['s1'] / fe['n'], 0)} / {c.signed(-oe['s1'] / oe['n'], 0)}",
                f"{feb_days[day]['hours']:.0f} / {ow['hours'].sum() / n_ow:.1f}",
                f"{feb_days[day]['freezing']:.0f} / {ow['freezing'].sum() / n_ow:.1f}",
                f"{feb_days[day]['low_vis']:.0f} / {ow['low_vis'].sum() / n_ow:.1f}",
                f"{feb_days[day]['precip']:.0f} / {ow['precip'].sum() / n_ow:.1f}",
                f"{c.signed(lag_f, 0)} / {c.signed(lag_o, 0)}",
                f"{c.fmt(ti_f, 0)} / {c.fmt(ti_o, 0)}",
                f"{c.signed(la_f, 0)} / {c.signed(la_o, 0)}",
            ]
        )
    oe = err_cell(oth_err[oth_err["day"].isin(days)])
    ow = oth_wx[oth_wx["day"].isin(days)]
    lag_f, _ = mean_of(feb_dep["lag"].to_numpy(dtype=float))
    lag_o, _ = mean_of(
        oth_dep.loc[oth_dep["day"].isin(days), "lag"].to_numpy(dtype=float)
    )
    ti_f, _ = mean_of(feb_arr["taxi_in"].to_numpy(dtype=float))
    ti_o, _ = mean_of(
        oth_arr.loc[oth_arr["day"].isin(days), "taxi_in"].to_numpy(dtype=float)
    )
    la_f, _ = mean_of(feb_arr["late"].to_numpy(dtype=float))
    la_o, _ = mean_of(
        oth_arr.loc[oth_arr["day"].isin(days), "late"].to_numpy(dtype=float)
    )
    ow = oth_wx[oth_wx["day"].isin(days)]
    n_ow = max(len(ow), 1)
    n_fw = max(len(feb_wx), 1)
    # departure days: the distinct (month, day) pairs that hold a departure
    n_fd = max(int(feb_err["day"].nunique()), 1)
    od = oth_err[oth_err["day"].isin(days)]
    n_od = max(int(od[["month", "day"]].drop_duplicates().shape[0]), 1)
    rows.append(
        [
            "**the month, a day**",
            f"{len(feb_err) / n_fd:,.0f} / {int(oth_err['day'].isin(days).sum()) / n_od:,.0f}",
            f"{c.fmt(feb_rmse)} / {c.fmt(np.sqrt(oe['sse'] / oe['n']))}",
            f"{c.signed(-feb_month['s1'] / feb_month['n'], 0)} / {c.signed(-oe['s1'] / oe['n'], 0)}",
            f"{feb_wx['hours'].sum() / n_fw:.1f} / {ow['hours'].sum() / n_ow:.1f}",
            f"{feb_wx['freezing'].sum() / n_fw:.1f} / {ow['freezing'].sum() / n_ow:.1f}",
            f"{feb_wx['low_vis'].sum() / n_fw:.1f} / {ow['low_vis'].sum() / n_ow:.1f}",
            f"{feb_wx['precip'].sum() / n_fw:.1f} / {ow['precip'].sum() / n_ow:.1f}",
            f"{c.signed(lag_f, 0)} / {c.signed(lag_o, 0)}",
            f"{c.fmt(ti_f, 0)} / {c.fmt(ti_o, 0)}",
            f"{c.signed(la_f, 0)} / {c.signed(la_o, 0)}",
        ]
    )
    L += c.table(
        [
            "day",
            "departures",
            "RMSE, five-seed average (s)",
            "bias (s)",
            "hours with a reading",
            "freezing hours",
            "low-visibility hours",
            "precipitation hours",
            "take-off - EOBT_1, mean (s)",
            "arrivals' taxi-in, mean (s)",
            "arrivals' lateness, mean (s)",
        ],
        rows,
        "lrrrrrrrrrr",
    )
    add("")
    hi = [dd for dd in days if feb_days[dd]["rmse"] > feb_rmse]
    lo = [dd for dd in days if feb_days[dd]["rmse"] <= feb_rmse]
    add(
        f"February at {AIRPORT} pools to {c.fmt(feb_rmse)} s over {feb_month['n']:,} departures. "
        f"{len(hi)} of its 28 days read above that figure ({', '.join(map(str, hi))}) and {len(lo)} at or below it. "
        + (
            f"Rows over 4 h among them: {feb_month['n'] - feb_month['n4']:,}, so the second reading over the rows of 4 h "
            "or less is the same figure."
            if feb_month["n"] == feb_month["n4"]
            else f"Rows over 4 h among them: {feb_month['n'] - feb_month['n4']:,}; over the rows of 4 h or less the month reads "
            f"{c.fmt(np.sqrt(feb_month['sse4'] / feb_month['n4']))} s, a diagnostic."
        )
    )
    add("")
    log["feb"] = {"rmse": feb_rmse, "n": feb_month["n"], "high_days": hi, "low_days": lo}

    # history context
    hist = monthly_shares(d.history)
    now = monthly_shares(d.hourly)
    add(
        "For context, the share of hours in each regime at the airport in the two calendar months, against the "
        "history `scripts/fetch_metar_history.py` fetched (`data/raw/metar_history/`, no reports before 2018 at this airport):"
    )
    add("")
    rows = []
    for cal in ("01", "02"):
        hm = hist[[m.endswith("-" + cal) for m in hist.index]]
        cur = [
            m for m in ({"01": ("2025-01", JAN26), "02": (FEB,)}[cal]) if m in now.index
        ]
        row = [
            {"01": "January", "02": "February"}[cal],
            f"{len(hm)} ({', '.join(m[:4] for m in hm.index)})",
        ]
        for _, col in FLAGS:
            v = hm[col].dropna()
            row.append(
                f"{100 * v.min():.1f}% to {100 * v.max():.1f}%" if len(v) else "n/a"
            )
            row.append("; ".join(f"{m[:4]} {100 * now.loc[m, col]:.1f}%" for m in cur))
        rows.append(row)
    L += c.table(
        [
            "month",
            "history years",
            "freezing, history",
            "freezing, cache",
            "low visibility, history",
            "low visibility, cache",
            "precipitation, history",
            "precipitation, cache",
        ],
        rows,
        "llrrrrrr",
    )
    add("")

    # 2 ---------------------------------------------------------------
    add(f"## {AIRPORT}, January 2026 (ranking month), day by day")
    add("")
    add(
        "The ranking month: departures, the same weather hours, the plan lag and the arrivals' control. No error "
        "exists for these rows and nothing is predicted about them."
    )
    add("")
    rr = d.rank_rows[(d.rank_rows["icao"] == AIRPORT) & (d.rank_rows["month"] == JAN26)]
    ra = d.rank_arr[(d.rank_arr["icao"] == AIRPORT) & (d.rank_arr["month"] == JAN26)]
    jw = wx[wx["month"] == JAN26]
    rows = []
    jan_days = {}
    for day in range(1, 32):
        rd = rr[rr["day"] == day]
        if not len(rd) and not len(jw[jw["day"] == day]):
            continue
        w = jw[jw["day"] == day]
        lag, _ = mean_of(rd["lag"].to_numpy(dtype=float))
        ti, _ = mean_of(ra.loc[ra["day"] == day, "taxi_in"].to_numpy(dtype=float))
        la, _ = mean_of(ra.loc[ra["day"] == day, "late"].to_numpy(dtype=float))
        jan_days[day] = {
            "freezing": float(w["freezing"].sum()),
            "low_vis": float(w["low_vis"].sum()),
            "precip": float(w["precip"].sum()),
            "hours": float(w["hours"].sum()),
            "lag": lag,
            "n": int(len(rd)),
        }
        rows.append(
            [
                f"{day}",
                f"{len(rd):,}",
                f"{jan_days[day]['hours']:.0f}",
                f"{jan_days[day]['freezing']:.0f}",
                f"{jan_days[day]['low_vis']:.0f}",
                f"{jan_days[day]['precip']:.0f}",
                c.signed(lag, 0),
                c.fmt(ti, 0),
                c.signed(la, 0),
            ]
        )
    lag, _ = mean_of(rr["lag"].to_numpy(dtype=float))
    ti, _ = mean_of(ra["taxi_in"].to_numpy(dtype=float))
    la, _ = mean_of(ra["late"].to_numpy(dtype=float))
    rows.append(
        [
            "**the month**",
            f"{len(rr):,}",
            f"{jw['hours'].sum():.0f}",
            f"{jw['freezing'].sum():.0f}",
            f"{jw['low_vis'].sum():.0f}",
            f"{jw['precip'].sum():.0f}",
            c.signed(lag, 0),
            c.fmt(ti, 0),
            c.signed(la, 0),
        ]
    )
    L += c.table(
        [
            "day",
            "departures",
            "hours with a reading",
            "freezing hours",
            "low-visibility hours",
            "precipitation hours",
            "take-off - EOBT_1, mean (s)",
            "arrivals' taxi-in, mean (s)",
            "arrivals' lateness, mean (s)",
        ],
        rows,
        "lrrrrrrrr",
    )
    add("")
    log["jan26"] = {
        "days": len(jan_days),
        "freezing_hours": float(jw["freezing"].sum()),
        "hours": float(jw["hours"].sum()),
        "lag": lag,
    }

    # 3 ---------------------------------------------------------------
    add("## January 2026 against February 2025's days")
    add("")

    def rng(dd: list[int], key: str) -> tuple[float, float]:
        v = [feb_days[x][key] for x in dd if np.isfinite(feb_days[x][key])]
        return (min(v), max(v)) if v else (float("nan"), float("nan"))

    def within(key: str, lo_hi: tuple[float, float]) -> list[int]:
        return [
            x
            for x, v in jan_days.items()
            if np.isfinite(v[key]) and lo_hi[0] <= v[key] <= lo_hi[1]
        ]

    rows = []
    out = {}
    for name, dd in (("above", hi), ("at or below", lo)):
        fr, lg = rng(dd, "freezing"), rng(dd, "lag")
        a, b = within("freezing", fr), within("lag", lg)
        both = sorted(set(a) & set(b))
        out[name] = {
            "freezing": fr,
            "lag": lg,
            "n_freezing": len(a),
            "n_lag": len(b),
            "n_both": len(both),
        }
        rows.append(
            [
                f"February days whose RMSE is {name} the month's {c.fmt(feb_rmse)} s ({len(dd)})",
                f"{fr[0]:.0f} to {fr[1]:.0f}",
                f"{len(a)} of {len(jan_days)}",
                f"{c.signed(lg[0], 0)} to {c.signed(lg[1], 0)}",
                f"{len(b)} of {len(jan_days)}",
                f"{len(both)} of {len(jan_days)}",
            ]
        )

    def overlaps(a, b):
        return max(a[0], b[0]) <= min(a[1], b[1])

    ov = {
        k: overlaps(out["above"][k], out["at or below"][k]) for k in ("freezing", "lag")
    }
    overlap_text = (
        (
            "The two sets of February ranges "
            + (
                "overlap on both axes, so a January day can sit in both."
                if all(ov.values())
                else "overlap on "
                + (", ".join(k for k, v in ov.items() if v) or "neither axis")
                + " and not on "
                + (", ".join(k for k, v in ov.items() if not v) or "either")
                + "."
            )
        )
        .replace("freezing", "freezing hours")
        .replace("lag", "the plan-lag mean")
    )
    log["overlap"] = ov
    L += c.table(
        [
            "February 2025 days",
            "their freezing hours, range",
            "January 2026 days within it",
            "their plan-lag mean, range (s)",
            "January 2026 days within it",
            "within both",
        ],
        rows,
        "lrrrrr",
    )
    add("")
    add(
        f"Of January 2026's {len(jan_days)} days at {AIRPORT}, {out['above']['n_freezing']} sit within the freezing-hour "
        f"range of the February days that erred above the month's figure and {out['above']['n_lag']} within their "
        f"plan-lag range ({out['above']['n_both']} within both); {out['at or below']['n_freezing']} sit within the "
        f"freezing-hour range of the days at or below it and {out['at or below']['n_lag']} within their plan-lag range "
        f"({out['at or below']['n_both']} within both). " + overlap_text + " "
        "**This is descriptive and decides nothing:** the ranges come from 28 days of one airport-month, the "
        "error on a day is not its weather or its plan lag, and no figure here says what January 2026 will score."
    )
    add("")
    log["comparison"] = out
    files = sorted(c.METAR.glob("*.csv")) + sorted(c.HISTORY.glob("*.csv"))
    L += c.inputs_footer(
        rb._inputs() + d.seed_files + files + [c.DATA / "ranking.parquet"]
    )
    return "\n".join(L) + "\n", log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    before = c.fingerprint()
    d = Data()
    text, _log = build(d)
    c.write(
        args.out,
        text,
        c.stands_of(d.dep, d.rank),
        c.movement_ids(d.dep["MVT_ID_mvt"], d.rank["MVT_ID_mvt"]),
        before,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
