# SPDX-License-Identifier: GPL-3.0-only
"""``taxiout.fusion`` on synthetic frames: the matching, the features, the peers
and the two gates. No data and no fit; every row is written here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxiout import fusion


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def deps(rows):
    """Departures: (id, airport, take-off, callsign, flight)."""
    return pd.DataFrame(
        {
            "MVT_ID_mvt": [float(r[0]) for r in rows],
            "ADEP_mvt": [r[1] for r in rows],
            "MVT_TIME_UTC_mvt": pd.to_datetime([r[2] for r in rows], utc=True),
            "CALLSIGN_flt": [r[3] for r in rows],
            "FLIGHT_mvt": [r[4] for r in rows],
        }
    )


def legs(rows):
    """Legs: (airport, take-off, e1, flight)."""
    t = pd.to_datetime([r[1] for r in rows], utc=True)
    e1 = pd.to_datetime([r[2] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "airport": [r[0] for r in rows],
            "takeoff": t,
            "e1": e1,
            "e2": e1 + pd.Timedelta(seconds=60),
            "e3": pd.Series([pd.NaT] * len(rows), dtype="datetime64[ns, UTC]"),
            "n_ground": [10] * len(rows),
            "max_gap_s": [5.0] * len(rows),
            "stationary_s": [0.0] * len(rows),
            "flight": pd.array([r[3] for r in rows], dtype="string"),
            "start_rule": pd.array(["i"] * len(rows), dtype="string"),
        }
    )


def test_the_days_of_section_5():
    days = fusion.adsb_days()
    assert len(days) == 245
    assert days[0] == "2025-01-01" and days[182] == "2025-12-31"
    assert "2025-01-02" not in days
    assert days[183] == "2026-01-01" and days[-1] == "2026-07-31"
    assert set(fusion.EXCLUDED_FIT_DAYS) <= set(days)


def test_the_match_window_is_120_s_and_a_tie_goes_to_the_callsign():
    d = deps(
        [
            (1, "EDDF", "2025-03-01 10:00:00", "DLH1", "LH1"),
            (2, "EDDF", "2025-03-01 11:00:00", "DLH2", "LH2"),
            (3, "EDDF", "2025-03-01 12:00:00", "DLH3", "LH3"),
        ]
    )
    lg = legs(
        [
            ("EDDF", "2025-03-01 10:02:00", "2025-03-01 09:50:00", "X"),  # 120 s: in
            ("EDDF", "2025-03-01 11:02:01", "2025-03-01 10:50:00", "X"),  # 121 s: out
            ("EDDF", "2025-03-01 12:01:00", "2025-03-01 11:50:00", "OTHER"),
            ("EDDF", "2025-03-01 11:59:00", "2025-03-01 11:50:00", "DLH3"),
        ]
    )
    idx, ties, by_callsign, unresolved = fusion.match_day(d, lg)
    assert list(idx) == [0, -1, 3]
    assert (ties, by_callsign, unresolved) == (1, 1, 0)


def test_before_0100_is_not_matched_and_confirmation_needs_a_lone_leg():
    d = deps(
        [
            (1, "EDDF", "2025-03-01 00:30:00", "DLH1", "LH1"),
            (2, "EDDF", "2025-03-01 10:00:00", "DLH2", "LH2"),
            (3, "EDDF", "2025-03-01 12:00:00", "DLH3", "LH3"),
            (4, "EDDF", "2025-03-01 12:00:30", "DLH4", "LH4"),
        ]
    )
    lg = legs(
        [
            ("EDDF", "2025-03-01 00:30:00", "2025-03-01 00:10:00", "DLH1"),
            ("EDDF", "2025-03-01 10:00:10", "2025-03-01 09:50:00", "DLH2"),
            ("EDDF", "2025-03-01 12:00:10", "2025-03-01 11:59:00", "DLH3"),
        ]
    )
    out, stats = fusion.adsb_day(d, lg, "2025-03-01")
    assert list(out["before_0100"]) == [1, 0, 0, 0]
    assert list(out["a_matched"]) == [0, 1, 1, 1]
    # take-off minus E1 of 70 s is matched but not covered
    assert list(out["a_covered"]) == [0, 1, 0, 0]
    # two departures share the third leg, so neither is confirmed
    assert list(out["a_confirmed"]) == [0, 1, 0, 0]
    assert out["a_to_e1_s"].iloc[1] == 600.0


def test_opdi_rows_stop_on_an_unexplained_absent_departure():
    d = pd.DataFrame(
        {
            "MVT_ID_mvt": [1.0, 2.0],
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-03-01 10:00"] * 2, utc=True),
            "STAND_mvt": ["A1", "B2"],
        }
    )
    t = pd.to_datetime(["2025-03-01 09:50"], utc=True)
    matched = pd.DataFrame(
        {
            "MVT_ID_mvt": [1.0],
            "matched": [True],
            "covered": [True],
            "opdi_tied": [False],
            "O0_time": t,
            "O1_time": pd.Series([pd.NaT], dtype="datetime64[ns, UTC]"),
            "O2_time": pd.Series([pd.NaT], dtype="datetime64[ns, UTC]"),
            "O3_time": pd.Series([pd.NaT], dtype="datetime64[ns, UTC]"),
            "R_time": t,
            "O1_time_in_use_s": [np.nan],
            "inbound_linked": [True],
            "inbound_block_time": pd.to_datetime(["2025-03-01 08:00"], utc=True),
            "inbound_sched_time": pd.to_datetime(["2025-03-01 07:50"], utc=True),
            "inbound_stand": [" A1 "],
        }
    )
    with pytest.raises(RuntimeError):
        fusion.opdi_rows(d, matched, allowed_absent=())
    out = fusion.opdi_rows(d, matched, allowed_absent={2.0})
    assert list(out["o_covered"]) == [1, 0] and list(out["i_linked"]) == [1, 0]
    assert out["i_same_stand"].iloc[0] == 1.0 and np.isnan(out["i_same_stand"].iloc[1])
    assert out["o_to_o0_s"].iloc[0] == 600.0 and out["i_inbound_late_s"].iloc[0] == 600.0
    assert out["p_o_first_s"].iloc[0] == 600.0 and np.isnan(out["p_o_first_s"].iloc[1])


def peer_frame(times, base, a_cov, p_e1, runway=None):
    n = len(times)
    return pd.DataFrame(
        {
            "ADEP_mvt": ["EDDF"] * n,
            "day": ["2025-03-01"] * n,
            "RUNWAY_mvt": runway or ["25C"] * n,
            "MVT_TIME_UTC_mvt": pd.to_datetime(times, utc=True),
            "base": base,
            "a_covered": a_cov,
            "o_covered": a_cov,
            "p_a_e1_s": p_e1,
            "p_a_e3_s": [np.nan] * n,
            "p_o_first_s": p_e1,
        }
    )


def test_the_peers_exclude_the_subject_and_hold_the_window():
    f = peer_frame(
        ["2025-03-01 10:00:00", "2025-03-01 10:30:00", "2025-03-01 10:30:01"],
        [500.0, 400.0, 300.0],
        [1, 1, 1],
        [800.0, 600.0, 900.0],
    )
    p = fusion.peer_features(f)
    # the first row sees the second (1,800 s) and not the third (1,801 s)
    assert p["peer_a_n"].iloc[0] == 1 and p["peer_a_e1_med"].iloc[0] == 200.0
    assert p["peer_a_n"].iloc[1] == 2
    assert np.isnan(p["peer_a_e3_med"]).all()


def test_an_uncovered_peer_counts_for_nothing_and_the_runway_subset():
    f = peer_frame(
        ["2025-03-01 10:00:00", "2025-03-01 10:10:00", "2025-03-01 10:20:00"],
        [500.0, 400.0, 300.0],
        [1, 0, 1],
        [800.0, np.nan, 900.0],
        runway=["25C", "25C", "18"],
    )
    p = fusion.peer_features(f)
    assert p["peer_a_n"].iloc[0] == 1 and p["peer_o_rwy_n"].iloc[0] == 0
    assert np.isnan(p["peer_o_rwy_med"].iloc[0])


def oof(rows):
    """Out-of-fold rows: (airport, month, y, base, fusion, covered)."""
    return pd.DataFrame(
        {
            "ADEP_mvt": [r[0] for r in rows],
            "month": [r[1] for r in rows],
            "y": [float(r[2]) for r in rows],
            "base": [float(r[3]) for r in rows],
            "fusion": [float(r[4]) for r in rows],
            "a_covered": [r[5] for r in rows],
            "o_covered": [0] * len(rows),
        }
    )


def test_the_own_month_never_informs_its_gate():
    rows = oof(
        [
            ("EDDF", "01", 100, 150, 101, 1),  # the fusion far better in January
            ("EDDF", "02", 100, 110, 130, 1),  # and worse in February
        ]
    )
    gate = fusion.gate_out_of_fold(rows, airports=["EDDF"], months=["01", "02"])
    assert not gate.loc["EDDF", "01"]  # decided on February alone
    assert gate.loc["EDDF", "02"]  # decided on January alone


def test_equal_rmse_takes_the_base_and_no_covered_row_takes_the_base():
    rows = oof([("EDDF", "01", 100, 110, 90, 1), ("LTFM", "01", 100, 110, 100, 0)])
    gate = fusion.gate_all(rows, airports=["EDDF", "LTFM"])
    assert not gate["EDDF"] and not gate["LTFM"]


def test_g1n_takes_the_fusion_only_where_covered_and_kept():
    rows = oof(
        [
            ("EDDF", "01", 100, 120, 101, 1),
            ("EDDF", "02", 100, 120, 101, 1),
            ("EDDF", "02", 100, 120, 101, 0),
        ]
    )
    gate = fusion.gate_out_of_fold(rows, airports=["EDDF"], months=["01", "02"])
    got = fusion.apply_gate(rows, gate)
    assert list(got) == [101.0, 101.0, 120.0]


def rank_rows(rows):
    """Ranking rows: (airport, a_covered, o_covered, rules_row)."""
    return pd.DataFrame(
        {
            "ADEP_mvt": [r[0] for r in rows],
            "a_covered": [r[1] for r in rows],
            "o_covered": [r[2] for r in rows],
            "rules_row": [r[3] for r in rows],
        }
    )


def test_the_ranking_rows_the_fusion_takes():
    gate = pd.Series({"EDDF": True, "EGLL": False})
    f = rank_rows(
        [
            ("EDDF", 1, 0, 0),  # taken
            ("EDDF", 0, 1, 0),  # taken: OPDI coverage counts
            ("EDDF", 0, 0, 0),  # uncovered: keeps the base
            ("EGLL", 1, 1, 0),  # a dropped airport: keeps the base
            ("EDDF", 1, 1, 1),  # a rules' row: never touched
            ("LTFM", 1, 0, 0),  # not in the gate: keeps the base
        ]
    )
    assert list(fusion.applied_rows(f, gate)) == [True, True, False, False, False, False]


def test_the_model_constants_are_section_5s():
    assert len(fusion.FEATURES) == 29 and fusion.FEATURES[:2] == ["base", "airport"]
    p = fusion.PARAMS
    assert (p["learning_rate"], p["num_leaves"], p["min_data_in_leaf"]) == (0.05, 63, 50)
    assert (p["feature_fraction"], p["bagging_fraction"], p["bagging_freq"]) == (
        0.9,
        0.8,
        1,
    )
    assert p["deterministic"] and p["force_col_wise"] and p["num_threads"] == 8
    assert p["seed"] == 20260902 and fusion.ROUNDS == 1000
