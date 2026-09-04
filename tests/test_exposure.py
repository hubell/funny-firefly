# SPDX-License-Identifier: GPL-3.0-only
"""Regime exposure: the row set, the shift, the slope, the bound.

Every number here is checkable by eye. The point of the section this
supports is a bound, and a bound computed one row wide of its definition is
not a bound at all.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.exposure import (  # noqa: E402
    ARRIVAL_COLUMNS,
    ARRIVAL_LATE_S,
    LAG_COLUMNS,
    LAG_HI_S,
    LAG_LO_S,
    TAXI_IN_HI_S,
    TAXI_IN_LO_S,
    arrival_rows,
    cell_means,
    departure_rows,
    lag_slopes,
    pooled_cost,
    joint_window_rows,
    shift_table,
    window_losses,
    worst_case,
)

MINUTE = 60.0


def dep(icao, when, *, eobt_lag, lobt_lag=None, sched_lag=None, plan=True):
    """A departure taking off at ``when`` with the given lags, in minutes."""
    t = pd.Timestamp(when, tz="UTC")
    lobt_lag = eobt_lag if lobt_lag is None else lobt_lag
    sched_lag = eobt_lag if sched_lag is None else sched_lag
    return {
        "ADEP_mvt": icao,
        "MVT_TIME_UTC_mvt": t,
        "EOBT_1_flt": (t - pd.Timedelta(minutes=eobt_lag)) if plan else pd.NaT,
        "LOBT_flt": (t - pd.Timedelta(minutes=lobt_lag)) if plan else pd.NaT,
        "SCHED_TIME_UTC_mvt": t - pd.Timedelta(minutes=sched_lag),
    }


def arr(icao, when, *, taxi_in, late):
    t = pd.Timestamp(when, tz="UTC")
    return {
        "PHASE_mvt": "ARR",
        "ADES_mvt": icao,
        "MVT_TIME_UTC_mvt": t,
        "SCHED_TIME_UTC_mvt": t - pd.Timedelta(minutes=late),
        "TAXITIME_SEC_mvt": taxi_in,
    }


# --------------------------------------------------------------------------
# the row set


def test_the_record_set_is_unflagged_and_planned():
    rows = pd.DataFrame(
        [
            dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=20),  # ordinary
            dep("AAAA", "2025-01-05T11:00:00Z", eobt_lag=20),  # flagged below
            dep("AAAA", "2025-01-05T12:00:00Z", eobt_lag=20, plan=False),  # no record
        ]
    )
    out = departure_rows(rows, np.array([False, True, False]))
    assert out["in_record"].tolist() == [True, False, False]
    assert out["lag_eobt_s"].notna().tolist() == [True, False, False]
    assert out["icao"].tolist() == ["AAAA"] * 3
    assert out["month"].tolist() == ["2025-01"] * 3
    assert list(LAG_COLUMNS) == ["lag_eobt_s", "lag_lobt_s", "lag_sched_s"]


def test_each_lag_is_windowed_on_its_own_and_not_across_the_three():
    """The filter that removes artefacts must not also remove the shift.

    A row whose schedule lag runs long is usually a row that took off late.
    Windowing the three together drops it from the take-off-minus-EOBT mean
    as well, and drops more such rows on the side that ran later -- which is
    the side whose shift is being measured.
    """
    rows = pd.DataFrame(
        [
            dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=20, sched_lag=200),
            dep("AAAA", "2025-01-05T11:00:00Z", eobt_lag=-31, sched_lag=20),
            dep("AAAA", "2025-01-05T12:00:00Z", eobt_lag=121, sched_lag=20),
        ]
    )
    out = departure_rows(rows)
    # the first row's EOBT lag is in range and counts, though its schedule lag is not
    assert out["lag_eobt_s"].notna().tolist() == [True, False, False]
    assert out["lag_eobt_s"].iloc[0] == pytest.approx(20 * MINUTE)
    assert out["lag_sched_s"].notna().tolist() == [False, True, True]
    assert out["in_record"].all()


def test_the_window_edges_are_inclusive():
    rows = pd.DataFrame(
        [
            dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=LAG_LO_S / MINUTE),
            dep("AAAA", "2025-01-05T11:00:00Z", eobt_lag=LAG_HI_S / MINUTE),
        ]
    )
    assert departure_rows(rows)["lag_eobt_s"].notna().tolist() == [True, True]


def test_without_a_flag_mask_every_planned_row_in_the_window_counts():
    rows = pd.DataFrame([dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=20)])
    assert departure_rows(rows, None)["lag_eobt_s"].notna().tolist() == [True]


def test_the_window_losses_are_countable_per_airport_and_month():
    rows = pd.DataFrame(
        [
            dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=20),
            dep("AAAA", "2025-01-05T11:00:00Z", eobt_lag=200),  # in record, out of window
            dep("AAAA", "2025-01-05T12:00:00Z", eobt_lag=20, plan=False),  # not in record
        ]
    )
    losses = window_losses(departure_rows(rows))
    assert losses.loc[("AAAA", "2025-01"), "in_record"] == 2.0
    assert losses.loc[("AAAA", "2025-01"), "in_window"] == 1.0


def test_a_joint_window_keeps_fewer_rows_than_each_lags_own():
    rows = pd.DataFrame(
        [
            dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=20, sched_lag=20),
            dep("AAAA", "2025-01-05T11:00:00Z", eobt_lag=20, sched_lag=200),
        ]
    )
    losses = window_losses(departure_rows(rows))
    assert losses.loc[("AAAA", "2025-01"), "in_record"] == 2.0
    assert (
        losses.loc[("AAAA", "2025-01"), "in_window"] == 2.0
    )  # both EOBT lags are in range
    assert losses.loc[("AAAA", "2025-01"), "in_joint"] == 1.0  # one schedule lag is not
    # the worst lag of the three is the schedule's, which keeps one of the two
    assert losses.loc[("AAAA", "2025-01"), "in_window_min"] == 1.0


def test_the_joint_window_is_kept_as_a_row_set_for_comparison():
    rows = departure_rows(
        pd.DataFrame(
            [
                dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=20, sched_lag=20),
                dep("AAAA", "2025-01-05T11:00:00Z", eobt_lag=90, sched_lag=200),
            ]
        )
    )
    # on its own window the EOBT mean is over both rows; jointly, over the first
    assert rows["lag_eobt_s"].mean() == pytest.approx(55 * MINUTE)
    joint = joint_window_rows(rows)
    assert joint["lag_eobt_s"].notna().tolist() == [True, False]
    assert joint["lag_eobt_s"].mean() == pytest.approx(20 * MINUTE)
    # the original is not modified
    assert rows["lag_eobt_s"].notna().all()


def test_a_frame_without_a_planning_column_is_refused():
    rows = pd.DataFrame([dep("AAAA", "2025-01-05T10:00:00Z", eobt_lag=20)]).drop(
        columns=["EOBT_1_flt"]
    )
    with pytest.raises(KeyError):
        departure_rows(rows)


# --------------------------------------------------------------------------
# the arrivals control


def test_arrivals_are_keyed_on_the_destination_and_windowed_per_column():
    movements = pd.DataFrame(
        [
            arr("AAAA", "2025-01-05T10:00:00Z", taxi_in=300.0, late=10),
            arr("AAAA", "2025-01-05T11:00:00Z", taxi_in=TAXI_IN_LO_S - 1, late=10),
            arr("AAAA", "2025-01-05T12:00:00Z", taxi_in=TAXI_IN_HI_S + 1, late=10),
            arr("AAAA", "2025-01-05T13:00:00Z", taxi_in=600.0, late=-200),
            {
                "PHASE_mvt": "DEP",  # a departure is never read here
                "ADES_mvt": "BBBB",
                "MVT_TIME_UTC_mvt": pd.Timestamp("2025-01-05T10:00:00Z"),
                "SCHED_TIME_UTC_mvt": pd.Timestamp("2025-01-05T09:00:00Z"),
                "TAXITIME_SEC_mvt": 900.0,
            },
        ]
    )
    out = arrival_rows(movements)
    assert len(out) == 4 and set(out["icao"]) == {"AAAA"}
    assert out["arr_taxi_in_s"].notna().tolist() == [True, False, False, True]
    # a lateness beyond the symmetric window is dropped, an early arrival is not
    assert out["arr_late_s"].notna().tolist() == [True, True, True, False]
    assert abs(-200 * MINUTE) > ARRIVAL_LATE_S
    assert list(ARRIVAL_COLUMNS) == ["arr_taxi_in_s", "arr_late_s"]


# --------------------------------------------------------------------------
# the shift


def _cell(icao, month, lags):
    """Rows for one airport-month with the given eobt lags, in minutes."""
    return pd.DataFrame(
        {
            "icao": icao,
            "month": month,
            "lag_eobt_s": [m * MINUTE for m in lags],
            "in_record": True,
        },
        index=range(len(lags)),
    )


def test_the_shift_is_the_difference_of_the_two_means():
    train = pd.concat(
        [
            _cell("AAAA", "2025-01", [10, 20, 30] * 50),
            _cell("BBBB", "2025-01", [5] * 200),
        ],
        ignore_index=True,
    )
    rank = pd.concat(
        [
            _cell("AAAA", "2026-01", [30, 40, 50] * 50),
            _cell("BBBB", "2026-01", [5] * 200),
        ],
        ignore_index=True,
    )
    t = shift_table(train, rank, ["lag_eobt_s"]).set_index("icao")
    assert t.loc["AAAA", "lag_eobt_s_shift"] == pytest.approx(20 * MINUTE)
    assert t.loc["AAAA", "lag_eobt_s_train"] == pytest.approx(20 * MINUTE)
    assert t.loc["BBBB", "lag_eobt_s_shift"] == pytest.approx(0.0)
    assert t.loc["AAAA", "ranking_month"] == "2026-01"
    assert t.loc["AAAA", "training_month"] == "2025-01"


def test_a_thin_cell_on_either_side_is_not_compared():
    train = _cell("AAAA", "2025-01", [10] * 200)
    rank = _cell("AAAA", "2026-01", [30] * 3)
    t = shift_table(train, rank, ["lag_eobt_s"])
    assert len(t) == 1
    assert np.isnan(t["lag_eobt_s_shift"].iloc[0])
    assert t["lag_eobt_s_n_rank"].iloc[0] == 3.0  # the count is still reported


def test_an_airport_missing_from_one_side_is_absent_and_a_month_pair_is_matched():
    train = pd.concat(
        [_cell("AAAA", "2025-01", [10] * 200), _cell("CCCC", "2025-01", [10] * 200)],
        ignore_index=True,
    )
    rank = pd.concat(
        [_cell("AAAA", "2026-01", [10] * 200), _cell("AAAA", "2026-07", [10] * 200)],
        ignore_index=True,
    )
    t = shift_table(train, rank, ["lag_eobt_s"])
    # CCCC is not in the ranking set; July 2026 has no July 2025 twin in `train`
    assert t["icao"].tolist() == ["AAAA"]
    assert t["ranking_month"].tolist() == ["2026-01"]


def test_cell_means_ignore_missing_values_and_count_what_carried_one():
    rows = pd.DataFrame(
        {
            "icao": ["AAAA"] * 4,
            "month": ["2025-01"] * 4,
            "arr_taxi_in_s": [100.0, 200.0, np.nan, 300.0],
        }
    )
    m = cell_means(rows, ["arr_taxi_in_s"])
    assert m.loc[("AAAA", "2025-01"), "arr_taxi_in_s_mean"] == pytest.approx(200.0)
    assert m.loc[("AAAA", "2025-01"), "arr_taxi_in_s_n"] == 3.0


# --------------------------------------------------------------------------
# the slope and the bound


def test_the_slope_recovers_a_planted_relation_within_airport():
    rng = np.random.default_rng(0)
    n = 500
    lag = rng.uniform(0, 60 * MINUTE, size=n)
    rows = pd.DataFrame(
        {
            "icao": ["AAAA"] * n + ["BBBB"] * n,
            "lag_eobt_s": np.concatenate([lag, lag]),
        }
    )
    # AAAA taxis a third of a second per second of lag; BBBB not at all
    y = np.concatenate([600 + 0.33 * lag, 600 + 0.0 * lag])
    s = lag_slopes(rows, y).set_index("icao")
    assert s.loc["AAAA", "slope"] == pytest.approx(0.33, abs=1e-9)
    assert s.loc["BBBB", "slope"] == pytest.approx(0.0, abs=1e-9)
    assert s.loc["AAAA", "n"] == float(n)


def test_the_slope_reads_only_windowed_rows_and_needs_enough_of_them():
    lag = np.arange(300, dtype=float) * 10.0
    rows = pd.DataFrame({"icao": "AAAA", "lag_eobt_s": lag})
    y = 600 + 0.5 * lag
    thin = rows.copy()
    thin.loc[thin.index[:250], "lag_eobt_s"] = np.nan  # only 50 rows survive: too few
    assert np.isnan(lag_slopes(thin, y)["slope"].iloc[0])
    assert lag_slopes(rows, y)["slope"].iloc[0] == pytest.approx(0.5)
    # a row the window excluded cannot move the slope, however wild its target
    poisoned = rows.copy()
    poisoned.loc[poisoned.index[0], "lag_eobt_s"] = np.nan
    y2 = y.copy()
    y2[0] = 1e9
    assert lag_slopes(poisoned, y2)["slope"].iloc[0] == pytest.approx(0.5)


def test_the_worst_case_is_the_bias_in_quadrature():
    w = worst_case(shift_s=300.0, slope=0.3, rmse_s=200.0)
    assert w["bias_s"] == pytest.approx(90.0)
    assert w["rmse_with_bias_s"] == pytest.approx(np.hypot(200.0, 90.0))
    assert w["cost_s"] == pytest.approx(np.hypot(200.0, 90.0) - 200.0)
    # the sign of the shift does not change the cost, only the bias
    down = worst_case(shift_s=-300.0, slope=0.3, rmse_s=200.0)
    assert down["bias_s"] == pytest.approx(-90.0)
    assert down["cost_s"] == pytest.approx(w["cost_s"])
    assert np.isnan(worst_case(np.nan, 0.3, 200.0)["cost_s"])


def test_the_pooled_cost_weighs_airports_by_rows_and_biases_do_not_cancel():
    rows = pd.DataFrame(
        {
            "icao": ["AAAA", "BBBB"],
            "n": [100.0, 100.0],
            "rmse_s": [200.0, 200.0],
            "bias_s": [90.0, -90.0],
        }
    )
    p = pooled_cost(rows)
    assert p["rmse_s"] == pytest.approx(200.0)
    assert p["rmse_with_bias_s"] == pytest.approx(np.hypot(200.0, 90.0))
    assert p["cost_s"] > 0
    # an airport with no bias dilutes it; one with no rows does not count
    rows2 = pd.DataFrame(
        {
            "icao": ["AAAA", "BBBB"],
            "n": [100.0, 300.0],
            "rmse_s": [200.0, 200.0],
            "bias_s": [90.0, 0.0],
        }
    )
    assert pooled_cost(rows2)["cost_s"] < p["cost_s"]
    assert pooled_cost(pd.DataFrame(columns=["n", "rmse_s", "bias_s"]))["n"] == 0.0


def test_the_module_is_a_diagnostic_and_not_a_block_of_the_frame():
    """It re-derives the plan lag on purpose; it must never be wired to the model.

    The column names here are the plan-lag block's own, because they are the
    same quantity -- take-off minus a planning value. What separates this
    module from that block is that it registers nothing and the frame builder
    does not know it exists.
    """
    import inspect

    import taxiout.exposure as exposure
    import taxiout.features as features

    assert not hasattr(exposure, "SOURCES")
    assert not hasattr(exposure, "MOVEMENT_COLUMNS")
    assert "exposure" not in inspect.getsource(features)
    # and it reads no hidden departure field of its own accord
    from taxiout.challenge import FORBIDDEN_FEATURE_INPUTS

    source = inspect.getsource(exposure)
    for column in FORBIDDEN_FEATURE_INPUTS:
        if column == "TAXITIME_SEC_mvt":
            continue  # arrivals' taxi-in is the control, and is exposed
        assert column not in source, column
