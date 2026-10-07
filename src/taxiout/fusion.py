# SPDX-License-Identifier: GPL-3.0-only
"""The fusion model: the booster's prediction refined by measured ground events.

Defined by §5 of ``reports/preregistered-2026-10-04.md`` and its four
amendments (``-adsb-leg``, ``-opdi``, ``-adsb-after-gate``, ``-opdi-tie``),
gated by ``reports/preregistered-2026-10-06-last-week.md`` §2 (G1, by coverage)
and ``reports/preregistered-2026-10-06-g1n.md`` §1 (G1n, by coverage and by
airport, out of fold). Every step is the one the measurements of 5 and
6 October ran, unchanged; ``reports/fusion.md`` prints their readings again
from this module.

**Rows.** One per departure, keyed on ``MVT_ID_mvt``: every departure of each
fitting day (the UTC date of ``MVT_TIME_UTC_mvt``) and every ranking
departure. Fitting days: the 2025 days of §5 (every second day from
1 January) that have an adsb.lol table, less 2025-01-15 and 2025-07-16, the
two training pilot days the after-gate amendment leaves out of every fit.
``rules_row`` is 1 where the record-quality term or the plan-shifted class
covers the row, both fitted on the whole training year (``taxiout.tails``).

**adsb.lol features** (§5; after-gate §1). Per day, every departure at 01:00Z
or later, the rules' rows included, is matched to the leg at its airport whose
take-off lies within 120 s of ``MVT_TIME_UTC_mvt``, the nearest, ties by the
callsign, a tie left after that to the first leg in the table's order. A
departure before 01:00Z is not matched (the day's file starts at midnight).
``a_matched``; ``a_covered`` (matched, and the leg's take-off minus E1 at least
120 s); ``a_confirmed`` (the leg's flight field equals ``CALLSIGN_flt`` or
``FLIGHT_mvt``, spaces removed, empty never equal, and no other departure of the
day matched the leg); take-off minus E1, E2, E3; the leg's ground-position
count, longest gap and stationary time.

**OPDI features** (OPDI amendment §4), from ``taxiout.opdi``'s matched tables:
``o_matched``, ``o_covered``; take-off minus O0 to O3; take-off minus R; O1's
time in use; the inbound's ``i_linked``, take-off minus its block time, its
block time minus its schedule, and whether its stand is the departure's (both
present and non-empty, else missing). A departure whose take-off falls outside
its training file's month has no matched row (the match reads one month's
file) and enters with its OPDI and inbound features missing and their flags 0.

**Peers** (§5; OPDI §4), per row, over the other rows of its airport and UTC
date with take-off within 1,800 s either side, the rules' rows included, each
with its own base: the median of take-off minus E1 minus base over the
adsb-covered, the same with E3 where present, their count; the median of
take-off minus the first present of O1, O0, O2 minus base over the
OPDI-covered, their count, and the same two over those on the subject's runway.

**The model** (§5): LightGBM regression, 1,000 rounds, learning rate 0.05, 63
leaves, ``min_data_in_leaf`` 50, ``feature_fraction`` 0.9, ``bagging_fraction``
0.8 with ``bagging_freq`` 1, deterministic, column-wise, 8 threads, seed
20260902; the 29 ``FEATURES``; target the recorded taxi-out; fitted on the
rows neither rule covers with a taxi-out of 4 h or less. The base is the
adopted configuration's out-of-fold prediction for 2025 (its twelve held-out
months at the ladder's seed) and the submission's booster prediction for the
ranking rows.

**G1n** (the G1n amendment §1). Per held-out month m and airport a: over a's
covered rows (``a_covered`` or ``o_covered``) of the other eleven months, the
fusion keeps month m only if its RMSE is strictly lower than the base's. For
the ranking rows, the same over all twelve months' covered rows. A row takes
the fusion where it is covered, neither rule covers it and its airport keeps
the fusion; the base everywhere else.

Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

AIRPORTS: tuple[str, ...] = (
    "EDDF",
    "EDDM",
    "EGLL",
    "EHAM",
    "LEBL",
    "LEMD",
    "LFPG",
    "LIRF",
    "LSZH",
    "LTFM",
)

MATCH_S = 120.0  # §5: a leg's take-off within 120 s of MVT_TIME_UTC_mvt
COVER_S = 120.0  # §5: E1 at least 120 s before take-off
EARLY_H = 1  # §5: departures before 01:00Z are read only with the previous day
PEER_WINDOW_S = 1800.0  # §5: peers within 30 minutes of the subject's take-off
SEED = 20260902
ROUNDS = 1000
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "seed": SEED,
}

# §5: every second day of 2025 from 1 January, and every day of January and
# July 2026.
ADSB_FIRST_2025 = dt.date(2025, 1, 1)
ADSB_STEP_DAYS = 2
RANKING_MONTHS: tuple[tuple[int, int], ...] = ((2026, 1), (2026, 7))
# The after-gate amendment §1: the two training pilot days informed it and
# are left out of every fit and every reading.
EXCLUDED_FIT_DAYS: tuple[str, ...] = ("2025-01-15", "2025-07-16")

FEATURE_COLS = [
    "MVT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "EOBT_1_flt",
    "CALLSIGN_flt",
    "FLIGHT_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
]
KEY = [
    "MVT_ID_mvt",
    "day",
    "ADEP_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "MVT_TIME_UTC_mvt",
    "before_0100",
    "rules_row",
]
A_COLS = [
    "a_matched",
    "a_covered",
    "a_confirmed",
    "a_to_e1_s",
    "a_to_e2_s",
    "a_to_e3_s",
    "a_n_ground",
    "a_max_gap_s",
    "a_stationary_s",
]
O_COLS = [
    "o_matched",
    "o_covered",
    "o_to_o0_s",
    "o_to_o1_s",
    "o_to_o2_s",
    "o_to_o3_s",
    "o_mvt_minus_r_s",
    "o_o1_time_in_use_s",
]
I_COLS = ["i_linked", "i_to_inbound_block_s", "i_inbound_late_s", "i_same_stand"]
P_COLS = ["p_a_e1_s", "p_a_e3_s", "p_o_first_s"]
D_COLS = ["d_a_takeoff_minus_mvt_s", "d_a_start_rule", "d_o_tied"]
TABLE_COLUMNS = KEY + A_COLS + O_COLS + I_COLS + P_COLS + D_COLS
OPDI_READ = [
    "MVT_ID_mvt",
    "matched",
    "covered",
    "opdi_tied",
    "O0_time",
    "O1_time",
    "O2_time",
    "O3_time",
    "R_time",
    "O1_time_in_use_s",
    "inbound_linked",
    "inbound_block_time",
    "inbound_sched_time",
    "inbound_stand",
]
PEER_COLS = [
    "peer_a_e1_med",
    "peer_a_e3_med",
    "peer_a_n",
    "peer_o_med",
    "peer_o_n",
    "peer_o_rwy_med",
    "peer_o_rwy_n",
]
FEATURES = [
    "base",
    "airport",
    "a_to_e1_s",
    "a_to_e2_s",
    "a_to_e3_s",
    "a_matched",
    "a_covered",
    "a_confirmed",
    "a_n_ground",
    "a_max_gap_s",
    "a_stationary_s",
    "peer_a_e1_med",
    "peer_a_e3_med",
    "peer_a_n",
    "o_to_o0_s",
    "o_to_o1_s",
    "o_to_o2_s",
    "o_to_o3_s",
    "o_mvt_minus_r_s",
    "o_matched",
    "o_covered",
    "o_o1_time_in_use_s",
    "peer_o_med",
    "peer_o_n",
    "peer_o_rwy_med",
    "peer_o_rwy_n",
    "i_to_inbound_block_s",
    "i_inbound_late_s",
    "i_same_stand",
]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

# The adopted configuration's per-row predictions at the ladder's seed, as
# ``reports/candidates-2026-10-07.md`` tabulates them: rows and the sha256 of
# each held-out month's ``pred``. The fusion's 2025 base, checked on read.
BASE_RECORDS: dict[str, tuple[int, str]] = {
    "heldout-2025-01__combo4__seed0": (
        153706,
        "21260c7d210ca0efdf4b4f51b009aa36f76ac4e0c3dc6b197240e5a8a0d9ddc4",
    ),
    "heldout-2025-02__combo4__seed0": (
        143732,
        "ff5fdae9b7cfd9fc2aa7651c8b595e0ff2a46f437daf8c11cbec604dbef6e0c5",
    ),
    "heldout-2025-03__combo4__seed0": (
        164449,
        "726bfa6a655047e7029825e56a9cffb041af46ae85b90f1da6714bc240e0c8ca",
    ),
    "heldout-2025-04__combo4__seed0": (
        175288,
        "b7513a9d2a0086ab0a1bdc1600ffb55e8c49b2d63ceeaef0565bbdd6d8933bd0",
    ),
    "heldout-2025-05__combo4__seed0": (
        185202,
        "f04d17f7cfa0a8c050cf1a2d3b37b53a69d18e077d1a3b0b996d45e155cc1430",
    ),
    "heldout-2025-06__combo4__seed0": (
        183142,
        "b5124f6b04e8c202426b454695ac799da943ea78b7a5ac40731ae289dd73c07e",
    ),
    "heldout-2025-07__combo4__seed0": (
        190713,
        "85d638ac44ec494920792314a475d6f6effa3974413e3839ae4f44421c500666",
    ),
    "heldout-2025-08__combo4__seed0": (
        191182,
        "df87f1aca518922f56932033eb76d69d6f6ac3d2b41d19f34cbf95b5229153d9",
    ),
    "heldout-2025-09__combo4__seed0": (
        183950,
        "60edd78443640b180395edb595b7bb4c241757645decb7b91fbb39ac1061461c",
    ),
    "heldout-2025-10__combo4__seed0": (
        185674,
        "c47d3d03805cbd1729f742da97b6fe6c2f91847b49207fd0f005faa147ef5a7b",
    ),
    "heldout-2025-11__combo4__seed0": (
        162332,
        "a7ee06a5605a8f4f1d77cb54fec9779dda340b43d6344b402869342eb2d0c3a0",
    ),
    "heldout-2025-12__combo4__seed0": (
        165677,
        "0eb2412677ec677e355f2e954094eb3e83b98fe707cce1e4c06e10f4f1f9652a",
    ),
}


def adsb_days() -> list[str]:
    """§5's days, in calendar order: 183 of 2025 and 62 of 2026."""
    out = []
    d = ADSB_FIRST_2025
    while d.year == ADSB_FIRST_2025.year:
        out.append(d.isoformat())
        d += dt.timedelta(days=ADSB_STEP_DAYS)
    for y, m in RANKING_MONTHS:
        d = dt.date(y, m, 1)
        while d.month == m:
            out.append(d.isoformat())
            d += dt.timedelta(days=1)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pred_sha256(x) -> str:
    """A prediction's digest as the harness takes it: its float64 bytes."""
    return hashlib.sha256(np.asarray(x, dtype=float).tobytes()).hexdigest()


def secs(a: pd.Series, b: pd.Series) -> pd.Series:
    """a minus b in seconds, float64; NaN where either is missing."""
    return (a - b).dt.total_seconds().astype("float64")


def norm(s):
    return s.astype("string").str.replace(" ", "", regex=False)


def read_departures(path: Path, columns: list[str]) -> pd.DataFrame:
    """A challenge file's departures, times as UTC timestamps."""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=columns).to_pandas()
    t = t[t["PHASE_mvt"].astype(str) == "DEP"].reset_index(drop=True)
    for c in columns:
        if c.endswith("_UTC_mvt") or c in ("LOBT_flt", "EOBT_1_flt"):
            t[c] = pd.to_datetime(t[c], utc=True)
    t["ADEP_mvt"] = t["ADEP_mvt"].astype(str)
    return t


# --------------------------------------------------------------------------
# the departure-to-leg matching


def match_day(dep: pd.DataFrame, legs: pd.DataFrame):
    """Per departure: the matched leg's index in ``legs`` (or -1) and tie counts."""
    out = np.full(len(dep), -1)
    ties = callsign_ties = unresolved = 0
    by_ap = {a: g for a, g in legs.groupby("airport", sort=False)}
    cs = norm(dep["CALLSIGN_flt"]).fillna("")
    fl = norm(dep["FLIGHT_mvt"]).fillna("")
    for pos, (ap, mvt) in enumerate(zip(dep["ADEP_mvt"], dep["MVT_TIME_UTC_mvt"])):
        g = by_ap.get(ap)
        if g is None:
            continue
        d = (g["takeoff"] - mvt).dt.total_seconds().abs()
        near = d[d <= MATCH_S]
        if near.empty:
            continue
        best = near[near == near.min()]
        if len(best) > 1:
            ties += 1
            names = {cs.iloc[pos], fl.iloc[pos]} - {""}
            hit = [i for i in best.index if legs.at[i, "flight"] in names]
            if len(hit) >= 1:
                callsign_ties += 1
                best = best.loc[hit]
            if len(best) > 1:
                unresolved += 1
        out[pos] = best.index[0]
    return out, ties, callsign_ties, unresolved


# --------------------------------------------------------------------------
# the features


def adsb_day(dep: pd.DataFrame, legs: pd.DataFrame | None, day: str):
    """adsb.lol features for the departures of one UTC day, aligned to ``dep``."""
    start = pd.Timestamp(day, tz="UTC")
    mvt = dep["MVT_TIME_UTC_mvt"]
    early = (mvt < start + pd.Timedelta(hours=EARLY_H)).to_numpy()
    out = pd.DataFrame(index=dep.index)
    out["before_0100"] = early.astype("int8")
    leg = np.full(len(dep), -1, dtype=np.int64)
    stats = {"ties": 0, "ties_by_callsign": 0, "ties_unresolved": 0}
    sub = dep[~early]
    if legs is not None and len(sub):
        idx, t, c, u = match_day(sub.reset_index(drop=True), legs)
        leg[~early] = idx
        stats = {"ties": t, "ties_by_callsign": c, "ties_unresolved": u}
    m = leg >= 0
    if legs is not None and len(legs):
        lg = legs.iloc[np.where(m, leg, 0)].reset_index(drop=True)
    else:
        lg = pd.DataFrame(
            {
                c: pd.Series([pd.NaT] * len(dep), dtype="datetime64[us, UTC]")
                for c in ("takeoff", "e1", "e2", "e3")
            }
        )
        for c in ("n_ground", "max_gap_s", "stationary_s"):
            lg[c] = np.nan
        lg["flight"] = pd.Series([None] * len(dep), dtype="string")
        lg["start_rule"] = pd.Series([None] * len(dep), dtype="string")
    mv = mvt.reset_index(drop=True)
    covered = m & (secs(lg["takeoff"], lg["e1"]) >= COVER_S).fillna(False).to_numpy()
    n_on_leg = pd.Series(leg[m]).value_counts()
    alone = m & (pd.Series(leg).map(n_on_leg).fillna(0).to_numpy() == 1)
    flight = norm(lg["flight"]).fillna("")
    cs = norm(dep["CALLSIGN_flt"].reset_index(drop=True)).fillna("")
    fl = norm(dep["FLIGHT_mvt"].reset_index(drop=True)).fillna("")
    same = ((flight != "") & ((flight == cs) | (flight == fl))).to_numpy(dtype=bool)
    out["a_matched"] = m.astype("int8")
    out["a_covered"] = covered.astype("int8")
    out["a_confirmed"] = (alone & same).astype("int8")
    for e in ("e1", "e2", "e3"):
        out[f"a_to_{e}_s"] = secs(mv, lg[e]).where(m).to_numpy()
    out["a_n_ground"] = lg["n_ground"].astype("float64").where(m).to_numpy()
    out["a_max_gap_s"] = lg["max_gap_s"].astype("float64").where(m).to_numpy()
    out["a_stationary_s"] = lg["stationary_s"].astype("float64").where(m).to_numpy()
    out["d_a_takeoff_minus_mvt_s"] = secs(lg["takeoff"], mv).where(m).to_numpy()
    out["d_a_start_rule"] = lg["start_rule"].astype("string").where(m).to_numpy()
    out["_leg"] = leg
    stats["matched"] = int(m.sum())
    stats["legs"] = 0 if legs is None else len(legs)
    return out, stats


def opdi_rows(dep: pd.DataFrame, matched: pd.DataFrame, allowed_absent) -> pd.DataFrame:
    """OPDI and inbound features; ``matched`` holds one row per ``MVT_ID_mvt``.

    ``allowed_absent`` names the departures that may lack a matched row; any
    other missing one is an error.
    """
    j = dep[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt"]].merge(
        matched[OPDI_READ],
        on="MVT_ID_mvt",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    absent = (j["_merge"] != "both").to_numpy()
    if (absent & ~j["MVT_ID_mvt"].isin(list(allowed_absent)).to_numpy()).any():
        raise RuntimeError(
            f"{int(absent.sum())} departures have no row in the matched tables"
        )
    for c in ("matched", "covered", "opdi_tied", "inbound_linked"):
        j[c] = j[c].astype("boolean").fillna(False).astype(bool)
    out = pd.DataFrame(index=dep.index)
    mv = j["MVT_TIME_UTC_mvt"]
    out["o_matched"] = j["matched"].astype(bool).astype("int8").to_numpy()
    out["o_covered"] = j["covered"].astype(bool).astype("int8").to_numpy()
    for k in range(4):
        out[f"o_to_o{k}_s"] = secs(mv, j[f"O{k}_time"]).to_numpy()
    out["o_mvt_minus_r_s"] = secs(mv, j["R_time"]).to_numpy()
    out["o_o1_time_in_use_s"] = j["O1_time_in_use_s"].astype("float64").to_numpy()
    linked = j["inbound_linked"].astype(bool).to_numpy()
    out["i_linked"] = linked.astype("int8")
    out["i_to_inbound_block_s"] = (
        secs(mv, j["inbound_block_time"]).where(linked).to_numpy()
    )
    out["i_inbound_late_s"] = (
        secs(j["inbound_block_time"], j["inbound_sched_time"]).where(linked).to_numpy()
    )
    a = j["inbound_stand"].astype("string").str.strip().replace("", pd.NA)
    b = j["STAND_mvt"].astype("string").str.strip().replace("", pd.NA)
    both = (a.notna() & b.notna()).to_numpy() & linked
    eq = (a == b).fillna(False).to_numpy(dtype=bool)
    out["i_same_stand"] = np.where(both, eq.astype("float64"), np.nan)
    out["d_o_tied"] = j["opdi_tied"].astype(bool).astype("int8").to_numpy()
    first = j["O1_time"].fillna(j["O0_time"]).fillna(j["O2_time"])
    out["p_o_first_s"] = secs(mv, first).where(j["covered"].astype(bool)).to_numpy()
    return out


def build_rows(dep, rules_row, days, legs_for, matched, allowed_absent=()):
    """Every departure of ``dep`` whose UTC day is in ``days``, with all features.

    ``rules_row`` is a boolean Series indexed like ``dep``; ``legs_for(day)``
    returns the day's leg table or ``None``. Returns the table, sorted by
    take-off and identifier, and per-day counts.
    """
    day = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    parts, stats = [], {}
    for d in days:
        sel = dep[day == d]
        a, st = adsb_day(sel, legs_for(d), d)
        a["_day_leg"] = np.where(a["_leg"] >= 0, d + ":" + a["_leg"].astype(str), "")
        frame = pd.DataFrame(
            {
                "MVT_ID_mvt": sel["MVT_ID_mvt"].astype("float64"),
                "day": d,
                "ADEP_mvt": sel["ADEP_mvt"].astype("string"),
                "RUNWAY_mvt": sel["RUNWAY_mvt"].astype("string"),
                "STAND_mvt": sel["STAND_mvt"].astype("string"),
                "MVT_TIME_UTC_mvt": sel["MVT_TIME_UTC_mvt"],
            },
            index=sel.index,
        )
        frame["before_0100"] = a["before_0100"]
        frame["rules_row"] = rules_row.loc[sel.index].astype("int8")
        parts.append(pd.concat([frame, a.drop(columns=["before_0100"])], axis=1))
        stats[d] = {"departures": len(sel), **st}
    t = pd.concat(parts)
    o = opdi_rows(dep.loc[t.index], matched, allowed_absent)
    t = pd.concat([t, o], axis=1)
    t["p_a_e1_s"] = t["a_to_e1_s"].where(t["a_covered"] == 1)
    t["p_a_e3_s"] = t["a_to_e3_s"].where(t["a_covered"] == 1)
    t = t.sort_values(["MVT_TIME_UTC_mvt", "MVT_ID_mvt"], kind="stable").reset_index(
        drop=True
    )
    return t[TABLE_COLUMNS + ["_leg", "_day_leg"]], stats


def rules_rows(train: pd.DataFrame, rank: pd.DataFrame):
    """The rules' rows of both sides, the term and the class fitted on ``train``."""
    from .tails import PlanShiftModel, TailModel

    term = TailModel().fit(train)
    cls = PlanShiftModel(term).fit(train)

    def rows(frame):
        return (term.flagged(frame) | cls.in_class(frame)).to_numpy(dtype=bool)

    counts = {
        "cells": int(term.cells["covered"].sum()),
        "train_term": int(term.flagged(train).sum()),
        "plan_shift_n": int(cls.n),
        "plan_shift_long": int(cls.long),
        "rank_term": int(term.flagged(rank).sum()),
        "rank_plan_shift": int(cls.in_class(rank).sum()),
    }
    return rows(train), rows(rank), counts


def feature_tables(challenge: Path, adsb_dir: Path, matched_dir: Path) -> dict:
    """The two feature tables, ``fit`` (the fitting days) and ``rank``.

    Reads the training year and the ranking set, the adsb.lol tables under
    ``adsb_dir`` and the matched tables under ``matched_dir``. Truth is read
    only to fit the rules (``rules_rows``); no feature function receives it.
    """
    import pyarrow.parquet as pq

    from .challenge import movement_files
    from .opdi import MATCH_MONTHS
    from .tails import FIT_ONLY_INPUTS, PLAN_SHIFT_PREDICT_INPUTS, PREDICT_INPUTS

    files = movement_files(challenge)
    rule_cols = sorted(
        set(PREDICT_INPUTS)
        | set(FIT_ONLY_INPUTS)
        | set(PLAN_SHIFT_PREDICT_INPUTS)
        | {"MVT_ID_mvt", "PHASE_mvt"}
    )
    train_rules = pd.concat(
        [read_departures(f, rule_cols) for f in files], ignore_index=True
    )
    rank_rules = read_departures(challenge / "ranking.parquet", rule_cols)
    t_rows, r_rows, counts = rules_rows(train_rules, rank_rules)
    t_rules = pd.Series(t_rows, index=train_rules["MVT_ID_mvt"].to_numpy())
    r_rules = pd.Series(r_rows, index=rank_rules["MVT_ID_mvt"].to_numpy())
    del train_rules, rank_rules

    parts = []
    for f in files:
        d = read_departures(f, FEATURE_COLS)
        d["_file_month"] = f.name[9:13] + f.name[14:16]
        parts.append(d)
    train = pd.concat(parts, ignore_index=True)
    rank = read_departures(challenge / "ranking.parquet", FEATURE_COLS)
    for frame in (train, rank):
        v = frame["MVT_ID_mvt"]
        if not ((v % 1 == 0).all() and v.abs().max() < 2**53 and v.is_unique):
            raise RuntimeError("MVT_ID_mvt is not an exact unique integer")
    # a departure stored in the next month's file is not in its own month's match
    outside = train["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m") != train["_file_month"]
    allowed_absent = set(train.loc[outside, "MVT_ID_mvt"].astype(float))

    matched = pd.concat(
        [
            pq.read_table(matched_dir / f"{ym}.parquet", columns=OPDI_READ).to_pandas()
            for ym in MATCH_MONTHS
        ],
        ignore_index=True,
    )
    if not matched["MVT_ID_mvt"].is_unique:
        raise RuntimeError("MVT_ID_mvt repeats across the matched months")

    days = adsb_days()
    have = {d for d in days if (adsb_dir / f"{d}.parquet").exists()}
    fit_days = [
        d
        for d in days
        if d.startswith("2025") and d in have and d not in EXCLUDED_FIT_DAYS
    ]
    rank_days = [d for d in days if not d.startswith("2025")]
    cache: dict = {}

    def legs_for(d):
        if d not in have:
            return None
        if d not in cache:
            cache.clear()
            cache[d] = pq.read_table(adsb_dir / f"{d}.parquet").to_pandas()
        return cache[d]

    tr = pd.Series(
        t_rules.reindex(train["MVT_ID_mvt"].to_numpy()).to_numpy(), index=train.index
    )
    rr = pd.Series(
        r_rules.reindex(rank["MVT_ID_mvt"].to_numpy()).to_numpy(), index=rank.index
    )
    if tr.isna().any() or rr.isna().any():
        raise RuntimeError("a departure lacks its rules flag")
    rk, rk_stats = build_rows(rank, rr.astype(bool), rank_days, legs_for, matched)
    ft, ft_stats = build_rows(
        train, tr.astype(bool), fit_days, legs_for, matched, allowed_absent
    )
    absent = sorted(set(ft["MVT_ID_mvt"]) - set(matched["MVT_ID_mvt"]))
    return {
        "fit": ft,
        "rank": rk,
        "rules": counts,
        "fit_days": fit_days,
        "rank_days": rank_days,
        "days_without_table": [d for d in days if d not in have],
        "absent_from_matched": len(absent),
        "stats": {**ft_stats, **rk_stats},
    }


def write_table(t: pd.DataFrame, path: Path, meta: dict) -> str:
    """A feature table, its metadata carrying the notices, by a temporary name."""
    import json
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pa.Table.from_pandas(t, preserve_index=False)
    md = dict(tbl.schema.metadata or {})
    md.update(
        {
            k.encode(): (
                v if isinstance(v, str) else json.dumps(v, sort_keys=True)
            ).encode()
            for k, v in meta.items()
        }
    )
    md.pop(b"pandas", None)
    tbl = tbl.replace_schema_metadata(md)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(tbl, tmp, compression="zstd")
    os.replace(tmp, path)
    return sha256_file(path)


# --------------------------------------------------------------------------
# the base, the peers and the frame


def read_base(board: Path, records: dict | None = None) -> pd.DataFrame:
    """The configuration's twelve held-out months at the ladder's seed, checked.

    ``records`` maps each file's name to (rows, pred sha256); the file's
    ``pred`` must hash to it, as must its record's. Returns one row per
    departure of the training year with ``base`` and the held-out ``month``.
    """
    import json

    records = BASE_RECORDS if records is None else records
    frames = []
    for mm in MONTHS:
        name = f"heldout-2025-{mm}__combo4__seed0"
        rows, want = records[name]
        rec = json.loads((board / f"{name}.json").read_text())
        kept = pd.read_parquet(board / f"{name}.parquet")
        got = pred_sha256(kept["pred"].to_numpy(dtype=float))
        if not (
            rec["pred_sha256"] == want == got and rec["valid_rows"] == rows == len(kept)
        ):
            raise RuntimeError(f"{name}: its predictions are not the record's")
        k = kept.copy()
        k["month"] = mm
        frames.append(k)
    base = pd.concat(frames, ignore_index=True)
    if base["mvt_id"].duplicated().any():
        raise RuntimeError("a departure has two base values")
    return base[
        ["mvt_id", "icao", "runway", "takeoff", "term", "shift", "y", "pred", "month"]
    ]


def peer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The seven peer columns for every row of ``frame``, in its order."""
    n = len(frame)
    out = {c: np.full(n, np.nan) for c in PEER_COLS}
    for c in ("peer_a_n", "peer_o_n", "peer_o_rwy_n"):
        out[c] = np.zeros(n, dtype=np.int64)
    ts = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
    t_all = (
        (ts - pd.Timestamp("1970-01-01", tz="UTC"))
        .dt.total_seconds()
        .to_numpy(dtype=float)
    )
    base = frame["base"].to_numpy(dtype=float)
    a_cov = frame["a_covered"].to_numpy() == 1
    o_cov = frame["o_covered"].to_numpy() == 1
    va1 = frame["p_a_e1_s"].to_numpy(dtype=float) - base
    va3 = frame["p_a_e3_s"].to_numpy(dtype=float) - base
    vo = frame["p_o_first_s"].to_numpy(dtype=float) - base
    has3 = np.isfinite(frame["p_a_e3_s"].to_numpy(dtype=float))
    rwy = (
        frame["RUNWAY_mvt"]
        .astype(object)
        .where(frame["RUNWAY_mvt"].notna(), None)
        .to_numpy()
    )
    groups = frame.groupby(["ADEP_mvt", "day"], sort=False).indices
    for idx in groups.values():
        idx = np.asarray(idx)
        order = idx[np.argsort(t_all[idx], kind="stable")]
        t = t_all[order]
        lo = np.searchsorted(t, t - PEER_WINDOW_S, side="left")
        hi = np.searchsorted(t, t + PEER_WINDOW_S, side="right")
        for k, i in enumerate(order):
            peers = order[lo[k] : hi[k]]
            peers = peers[peers != i]
            if not len(peers):
                continue
            pa = peers[a_cov[peers]]
            out["peer_a_n"][i] = len(pa)
            if len(pa):
                out["peer_a_e1_med"][i] = np.median(va1[pa])
                p3 = pa[has3[pa]]
                if len(p3):
                    out["peer_a_e3_med"][i] = np.median(va3[p3])
            po = peers[o_cov[peers]]
            out["peer_o_n"][i] = len(po)
            if len(po):
                out["peer_o_med"][i] = np.median(vo[po])
                if rwy[i] is not None:
                    pr_ = po[rwy[po] == rwy[i]]
                    out["peer_o_rwy_n"][i] = len(pr_)
                    if len(pr_):
                        out["peer_o_rwy_med"][i] = np.median(vo[pr_])
    return pd.DataFrame(out, index=frame.index)


def check_peer_inputs(frame: pd.DataFrame) -> dict:
    """The peers' inputs present exactly where their coverage says."""
    a = frame["a_covered"].to_numpy() == 1
    o = frame["o_covered"].to_numpy() == 1
    p1 = np.isfinite(frame["p_a_e1_s"].to_numpy(dtype=float))
    p3 = np.isfinite(frame["p_a_e3_s"].to_numpy(dtype=float))
    po = np.isfinite(frame["p_o_first_s"].to_numpy(dtype=float))
    got = {
        "a_covered_vs_p_a_e1_differ": int((a != p1).sum()),
        "p_a_e3_without_a_covered": int((p3 & ~a).sum()),
        "o_covered_vs_p_o_first_differ": int((o != po).sum()),
    }
    if any(got.values()):
        raise RuntimeError(f"peer inputs: {got}")
    return got


def fit_frame(fit: pd.DataFrame, base: pd.DataFrame, threshold_s: float) -> pd.DataFrame:
    """The 2025 fusion frame: features, base, truth, peers, the fitting and scored rows."""
    j = fit.merge(
        base, left_on="MVT_ID_mvt", right_on="mvt_id", how="left", validate="one_to_one"
    )
    if j["pred"].isna().any() or len(j) != len(fit):
        raise RuntimeError(
            f"{int(j['pred'].isna().sum())} fitting-day rows without a base"
        )
    j = j.rename(columns={"pred": "base"})
    if (j["takeoff"] != j["MVT_TIME_UTC_mvt"]).any() or (
        j["icao"] != j["ADEP_mvt"]
    ).any():
        raise RuntimeError(
            "a base row's take-off or airport differs from its departure's"
        )
    if (j["day"].str[5:7] != j["month"]).any():
        raise RuntimeError("a fitting day's month differs from its base month")
    if ((j["rules_row"] == 1).to_numpy() != (j["term"] | j["shift"]).to_numpy()).any():
        raise RuntimeError("the rules' rows differ from the held-out files' flags")
    check_peer_inputs(j)
    j = pd.concat([j, peer_features(j)], axis=1)
    rr = j["rules_row"].to_numpy() == 1
    j["fit_row"] = (~rr) & (j["y"].to_numpy() <= threshold_s)
    j["scored_row"] = ~rr
    keep = [
        "MVT_ID_mvt",
        "day",
        "month",
        "ADEP_mvt",
        "RUNWAY_mvt",
        "MVT_TIME_UTC_mvt",
        "rules_row",
        "y",
        "fit_row",
        "scored_row",
    ] + [c for c in FEATURES if c != "airport"]
    return j[keep]


def rank_frame(rank: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    """The ranking rows' features with ``base`` (indexed by identifier) and peers."""
    j = rank.copy()
    j["base"] = base.reindex(j["MVT_ID_mvt"].to_numpy()).to_numpy(dtype=float)
    if np.isnan(j["base"].to_numpy()).any():
        raise RuntimeError("a ranking row has no base prediction")
    day = pd.to_datetime(j["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d")
    if (day != j["day"]).any():
        raise RuntimeError("a ranking row's day is not its take-off's UTC date")
    check_peer_inputs(j)
    return pd.concat([j, peer_features(j)], axis=1)


# --------------------------------------------------------------------------
# the model


def design(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.copy()
    X["airport"] = pd.Categorical(X["ADEP_mvt"], categories=list(AIRPORTS))
    return X[FEATURES]


def train(frame: pd.DataFrame):
    """§5's model on ``frame``'s rows, every one of them fitted on."""
    import lightgbm as lgb

    X = design(frame)
    ds = lgb.Dataset(
        X,
        label=frame["y"].to_numpy(dtype=float),
        categorical_feature=["airport"],
        free_raw_data=True,
    )
    return lgb.train(dict(PARAMS), ds, num_boost_round=ROUNDS)


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(design(frame), num_threads=PARAMS["num_threads"])


def leave_one_month_out(frame: pd.DataFrame, months=MONTHS, log=None) -> pd.DataFrame:
    """Each month's scored rows predicted by the fit on the other months' fitting rows."""
    import time

    out = []
    for mm in months:
        tr = frame[(frame["month"] != mm) & frame["fit_row"]]
        sc = frame[(frame["month"] == mm) & frame["scored_row"]]
        t0 = time.monotonic()
        model = train(tr)
        pred = predict(model, sc)
        if log is not None:
            log[mm] = {
                "train_rows": int(len(tr)),
                "scored_rows": int(len(sc)),
                "trees": int(model.num_trees()),
                "pred_sha256": pred_sha256(pred),
                "seconds": round(time.monotonic() - t0, 1),
                "gain": dict(
                    zip(model.feature_name(), model.feature_importance("gain").tolist())
                ),
            }
        out.append(
            pd.DataFrame(
                {
                    "MVT_ID_mvt": sc["MVT_ID_mvt"].to_numpy(),
                    "day": sc["day"].to_numpy(),
                    "month": mm,
                    "ADEP_mvt": sc["ADEP_mvt"].to_numpy(),
                    "y": sc["y"].to_numpy(),
                    "base": sc["base"].to_numpy(),
                    "fusion": pred,
                    "a_covered": sc["a_covered"].to_numpy(),
                    "o_covered": sc["o_covered"].to_numpy(),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# the gates


def covered(frame: pd.DataFrame) -> np.ndarray:
    """G1: a row is covered where ``a_covered`` or ``o_covered`` is 1."""
    return (frame["a_covered"].to_numpy() == 1) | (frame["o_covered"].to_numpy() == 1)


def _rmse(r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(r**2)))


def keeps(rows: pd.DataFrame) -> bool:
    """The fusion is kept only if its RMSE over these covered rows is strictly lower."""
    if not len(rows):
        return False
    return _rmse((rows["y"] - rows["fusion"]).to_numpy(dtype=float)) < _rmse(
        (rows["y"] - rows["base"]).to_numpy(dtype=float)
    )


def gate_out_of_fold(
    frame: pd.DataFrame, airports=AIRPORTS, months=MONTHS
) -> pd.DataFrame:
    """Airport by month: keeps the fusion in month m, on the other months' covered rows."""
    cov = frame[covered(frame)]
    out = pd.DataFrame(False, index=list(airports), columns=list(months))
    for a in airports:
        ca = cov[cov["ADEP_mvt"] == a]
        for m in months:
            out.loc[a, m] = keeps(ca[ca["month"] != m])
    return out


def gate_all(frame: pd.DataFrame, airports=AIRPORTS) -> pd.Series:
    """Airport: keeps the fusion, on every month's covered rows (the ranking rows' gate)."""
    cov = frame[covered(frame)]
    return pd.Series({a: keeps(cov[cov["ADEP_mvt"] == a]) for a in airports})


def apply_gate(frame: pd.DataFrame, gate: pd.DataFrame) -> np.ndarray:
    """``fusion`` where covered and the airport keeps it in the row's month; else ``base``."""
    kept = np.array(
        [
            bool(gate.loc[a, m]) if a in gate.index and m in gate.columns else False
            for a, m in zip(frame["ADEP_mvt"], frame["month"])
        ],
        dtype=bool,
    )
    take = covered(frame) & kept
    return np.where(
        take, frame["fusion"].to_numpy(dtype=float), frame["base"].to_numpy(dtype=float)
    )


def applied_rows(frame: pd.DataFrame, gate: pd.Series) -> np.ndarray:
    """The ranking rows G1n takes the fusion on: covered, no rule, the airport kept."""
    kept = np.array(
        [bool(gate.get(a, False)) for a in frame["ADEP_mvt"].astype(str)], bool
    )
    return covered(frame) & (frame["rules_row"].to_numpy() == 0) & kept


__all__ = [
    "AIRPORTS",
    "FEATURES",
    "PARAMS",
    "ROUNDS",
    "adsb_days",
    "applied_rows",
    "apply_gate",
    "build_rows",
    "covered",
    "feature_tables",
    "fit_frame",
    "gate_all",
    "gate_out_of_fold",
    "leave_one_month_out",
    "match_day",
    "peer_features",
    "predict",
    "rank_frame",
    "read_base",
    "rules_rows",
    "train",
]
