# SPDX-License-Identifier: GPL-3.0-only
"""The unimpeded term: the PRU reference, the fitted speed, and the calibration.

The frames here are built by hand rather than through ``build_features``, so
that the quantity a fit is supposed to recover is known exactly and the test
fails on arithmetic rather than on a plausible-looking number.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.unimpeded import (  # noqa: E402
    MIN_PAIR_COUNT,
    GeometricUnimpeded,
    calibration,
    fit_geometric,
    fit_pru_reference,
    lower_tail,
    pair_key,
    runway_key,
)


def _frame(
    pairs: dict[tuple[str, str, str], list[float]],
    *,
    distances: dict[tuple[str, str, str], float] | None = None,
    modulator: dict[tuple[str, str, str], float] | None = None,
    wake: dict[tuple[str, str, str], str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rows for each (icao, stand, designator) with the listed taxi-out times."""
    recs, target = [], []
    for (icao, stand, designator), times in pairs.items():
        for t in times:
            recs.append(
                {
                    "icao": icao,
                    "stand": stand,
                    "designator": designator,
                    "pair": f"{icao}|{stand}|{designator}",
                    "wake_class": (wake or {}).get((icao, stand, designator), "M"),
                    "taxi_m": (distances or {}).get((icao, stand, designator), 1000.0),
                    "geom_imputed": 0.0,
                    "turn_deg_per_km": (modulator or {}).get(
                        (icao, stand, designator), 0.0
                    ),
                    "junctions_per_km": 0.0,
                    "straight_frac": 0.0,
                    "backtrack_frac": 0.0,
                    "detour_ratio": 0.0,
                }
            )
            target.append(t)
    frame = pd.DataFrame(recs)
    return frame, pd.Series(target, index=frame.index, dtype=float)


# --------------------------------------------------------------------------
# keys


def test_keys():
    frame = pd.DataFrame({"icao": ["ZZZZ"], "stand": ["B5"], "designator": ["27"]})
    assert pair_key(frame).tolist() == ["ZZZZ|B5|27"]
    assert runway_key(frame).tolist() == ["ZZZZ|27"]
    # a frame that already carries the pair uses it rather than rebuilding it
    assert pair_key(frame.assign(pair=["cached"])).tolist() == ["cached"]


# --------------------------------------------------------------------------
# the PRU reference


def test_reference_is_the_tenth_percentile_of_the_pair():
    times = list(np.arange(100, 100 + MIN_PAIR_COUNT * 10, 10, dtype=float))
    frame, y = _frame({("ZZZZ", "B5", "27"): times})
    ref = fit_pru_reference(frame, y)
    expected = float(np.quantile(times, 0.10))
    assert ref.pair["ZZZZ|B5|27"] == pytest.approx(expected)
    assert ref.predict(frame).iloc[0] == pytest.approx(expected)
    assert (ref.level(frame) == "pair").all()


def test_a_pair_below_the_minimum_count_falls_through_the_cascade():
    many = [600.0 + 10 * i for i in range(MIN_PAIR_COUNT)]
    frame, y = _frame({("ZZZZ", "B5", "27"): many, ("ZZZZ", "B6", "27"): [900.0, 950.0]})
    ref = fit_pru_reference(frame, y)
    assert "ZZZZ|B6|27" not in ref.pair.index
    level = ref.level(frame)
    assert set(level[frame["stand"] == "B6"]) == {"runway"}
    # and the value it gets is the runway's percentile, not its own two rows
    got = ref.predict(frame)[frame["stand"] == "B6"].iloc[0]
    assert got == pytest.approx(ref.runway["ZZZZ|27"])


def test_cascade_reaches_the_airport_for_an_unseen_runway():
    frame, y = _frame({("ZZZZ", "B5", "27"): [600.0] * MIN_PAIR_COUNT})
    ref = fit_pru_reference(frame, y)
    other = frame.iloc[[0]].assign(designator="09", pair="ZZZZ|B5|09")
    assert ref.level(other).iloc[0] == "airport"
    unknown = frame.iloc[[0]].assign(icao="QQQQ", pair="QQQQ|B5|27")
    assert ref.level(unknown).iloc[0] == "none"
    assert ref.predict(unknown).iloc[0] == pytest.approx(ref.overall)


def test_reference_only_reads_the_rows_the_mask_allows():
    frame, y = _frame({("ZZZZ", "B5", "27"): [600.0] * (2 * MIN_PAIR_COUNT)})
    mask = pd.Series(np.arange(len(frame)) < MIN_PAIR_COUNT, index=frame.index)
    poisoned = y.copy()
    poisoned[~mask] = 1.0
    ref = fit_pru_reference(frame, poisoned, mask=mask)
    assert ref.pair["ZZZZ|B5|27"] == pytest.approx(600.0)
    assert ref.counts["ZZZZ|B5|27"] == MIN_PAIR_COUNT


def test_pair_counts_are_zero_for_a_pair_never_seen():
    frame, y = _frame({("ZZZZ", "B5", "27"): [600.0] * MIN_PAIR_COUNT})
    ref = fit_pru_reference(frame, y)
    other = frame.iloc[[0]].assign(stand="B9", pair="ZZZZ|B9|27")
    assert ref.pair_counts(other).iloc[0] == 0.0
    assert ref.pair_counts(frame).iloc[0] == MIN_PAIR_COUNT


# --------------------------------------------------------------------------
# the lower tail


def test_lower_tail_is_the_rows_at_or_below_the_reference():
    times = [float(t) for t in range(500, 500 + MIN_PAIR_COUNT * 10, 10)]
    frame, y = _frame({("ZZZZ", "B5", "27"): times})
    ref = fit_pru_reference(frame, y)
    tail = lower_tail(frame, y, ref)
    assert tail.sum() >= 1
    assert (y[tail] <= ref.predict(frame)[tail]).all()
    assert (y[~tail] > ref.predict(frame)[~tail]).all()


def test_lower_tail_drops_flagged_rows_and_impossible_times():
    times = [0.0] + [float(t) for t in range(500, 500 + MIN_PAIR_COUNT * 10, 10)]
    frame, y = _frame({("ZZZZ", "B5", "27"): times})
    ref = fit_pru_reference(frame, y)
    unflagged = pd.Series(True, index=frame.index)
    unflagged.iloc[1] = False
    tail = lower_tail(frame, y, ref, mask=unflagged)
    assert not tail.iloc[0]  # a zero-second taxi is a record, not a taxi
    assert not tail.iloc[1]  # flagged out by the caller


# --------------------------------------------------------------------------
# the fitted speed


def test_speed_recovers_an_exact_distance_over_speed_reference():
    """Where the reference *is* d/v, the fit must return v to floating point."""
    v = 8.0
    pairs, dist = {}, {}
    for i, d in enumerate([500.0, 1500.0, 2500.0, 3500.0, 4500.0]):
        key = ("ZZZZ", f"S{i}", "27")
        dist[key] = d
        # every row of the pair sits at exactly d/v, so the 10th percentile is
        # d/v and every row is in the lower tail
        pairs[key] = [d / v] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref))
    assert model.speed["ZZZZ"] == pytest.approx(v, rel=1e-9)
    assert model.class_intercept[("ZZZZ", "M")] == pytest.approx(0.0, abs=1e-6)
    assert model.predict(frame).iloc[0] == pytest.approx(500.0 / v, rel=1e-6)


def test_speed_is_fitted_per_airport():
    pairs, dist = {}, {}
    for icao, v in (("AAAA", 6.0), ("BBBB", 12.0)):
        for i, d in enumerate([800.0, 1600.0, 2400.0, 3200.0, 4000.0]):
            key = (icao, f"S{i}", "27")
            dist[key] = d
            pairs[key] = [d / v] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref))
    assert model.speed["AAAA"] == pytest.approx(6.0, rel=1e-9)
    assert model.speed["BBBB"] == pytest.approx(12.0, rel=1e-9)


def test_modulator_recovers_a_planted_coefficient():
    """Plant ``r = a + d * (1/v + g * z)`` and read every parameter back."""
    a, v, g = 120.0, 8.0, 0.004
    pairs, dist, mod = {}, {}, {}
    rng = np.random.default_rng(0)
    for i in range(24):
        key = ("ZZZZ", f"S{i}", "27")
        d = float(500 + 200 * i)
        z = float(rng.normal())
        dist[key], mod[key] = d, z
        pairs[key] = [a + d * (1.0 / v + g * z)] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist, modulator=mod)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(
        frame, ref, lower_tail(frame, y, ref), modulators=("turn_deg_per_km",)
    )
    # the descriptor is standardised in the fit: the coefficient is g times its
    # standard deviation, and its mean has moved into the inverse speed
    sd, mu = model.scale["turn_deg_per_km"], model.centre["turn_deg_per_km"]
    assert model.coef["turn_deg_per_km"] / sd == pytest.approx(g, rel=1e-6)
    assert model.inverse_speed["ZZZZ"] == pytest.approx(1.0 / v + g * mu, rel=1e-6)
    assert model.class_intercept[("ZZZZ", "M")] == pytest.approx(a, rel=1e-6)
    assert np.allclose(model.predict(frame), ref.predict(frame), rtol=1e-6)


def test_fit_refuses_rows_whose_pair_has_no_reference_of_its_own():
    """A pair below the minimum count would fit the speed to a fallback."""
    pairs, dist = {}, {}
    for i, d in enumerate([1000.0, 1500.0, 2000.0, 2500.0, 3000.0]):
        key = ("ZZZZ", f"S{i}", "27")
        dist[key] = d
        pairs[key] = [d / 8.0] * MIN_PAIR_COUNT
    thin = ("ZZZZ", "S9", "27")
    dist[thin] = 9000.0
    pairs[thin] = [125.0]  # absurd, and below the minimum count
    frame, y = _frame(pairs, distances=dist)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref))
    assert model.speed["ZZZZ"] == pytest.approx(8.0, rel=1e-9)
    assert model.n_fit == 5 * MIN_PAIR_COUNT


def test_fit_refuses_imputed_geometry():
    pairs = {("ZZZZ", "S0", "27"): [125.0] * MIN_PAIR_COUNT}
    frame, y = _frame(pairs, distances={("ZZZZ", "S0", "27"): 1000.0})
    frame["geom_imputed"] = 1.0
    ref = fit_pru_reference(frame, y)
    with pytest.raises(ValueError):
        fit_geometric(frame, ref, lower_tail(frame, y, ref))


def test_prediction_is_floored_and_clipped():
    model = GeometricUnimpeded(
        inverse_speed={"ZZZZ": 1.0 / 8.0},
        default_inverse_speed=1.0 / 8.0,
        coef={"turn_deg_per_km": 50.0},
        centre={"turn_deg_per_km": 0.0},
        scale={"turn_deg_per_km": 1.0},
        class_column=None,
    )
    frame = pd.DataFrame(
        {
            "icao": ["ZZZZ", "ZZZZ", "QQQQ"],
            "taxi_m": [1.0, 4000.0, 4000.0],
            "turn_deg_per_km": [0.0, 10.0, 0.0],
        }
    )
    out = model.predict(frame)
    assert out.iloc[0] >= 30.0  # the physical floor
    assert out.iloc[1] <= 2.0 * 4000.0 / 8.0  # the clip on the modulated speed
    assert out.iloc[2] == pytest.approx(500.0)  # unknown airport, default speed


# --------------------------------------------------------------------------
# the calibration check


def test_calibration_is_exact_where_the_geometry_reproduces_the_reference():
    v = 8.0
    pairs, dist = {}, {}
    for i, d in enumerate([600.0, 1500.0, 2400.0, 3300.0, 4200.0]):
        key = ("ZZZZ", f"S{i}", "27")
        dist[key] = d
        pairs[key] = [d / v] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref))
    cal = calibration(frame, ref, model)
    assert list(cal["icao"]) == ["ZZZZ", "all"]
    row = cal.iloc[-1]
    assert row["pairs"] == 5
    assert row["rmse_s"] == pytest.approx(0.0, abs=1e-6)
    assert row["ratio_p50"] == pytest.approx(1.0, abs=1e-9)


def test_calibration_counts_pairs_once_however_busy_they_are():
    v = 8.0
    pairs = {
        ("ZZZZ", "S0", "27"): [1000.0 / v] * (5 * MIN_PAIR_COUNT),
        ("ZZZZ", "S1", "27"): [3000.0 / v] * MIN_PAIR_COUNT,
    }
    dist = {("ZZZZ", "S0", "27"): 1000.0, ("ZZZZ", "S1", "27"): 3000.0}
    frame, y = _frame(pairs, distances=dist)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref))
    cal = calibration(frame, ref, model)
    assert int(cal.loc[cal["icao"] == "all", "pairs"].iloc[0]) == 2


# --------------------------------------------------------------------------
# the per-class intercept


def test_the_intercept_is_fitted_per_airport_and_class():
    """Two classes at one speed, two intercepts; each recovered exactly."""
    v, a_heavy, a_medium = 8.0, 300.0, 120.0
    pairs, dist, wake = {}, {}, {}
    for i, d in enumerate([800.0, 1600.0, 2400.0, 3200.0, 4000.0]):
        for cls, a in (("H", a_heavy), ("M", a_medium)):
            key = ("ZZZZ", f"S{i}{cls}", "27")
            dist[key], wake[key] = d, cls
            pairs[key] = [a + d / v] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist, wake=wake)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref))
    assert model.speed["ZZZZ"] == pytest.approx(v, rel=1e-9)
    assert model.class_intercept[("ZZZZ", "H")] == pytest.approx(a_heavy, rel=1e-9)
    assert model.class_intercept[("ZZZZ", "M")] == pytest.approx(a_medium, rel=1e-9)
    assert np.allclose(model.predict(frame), ref.predict(frame), rtol=1e-9)


def test_without_a_class_column_there_is_one_intercept_per_airport():
    v, a = 8.0, 200.0
    pairs, dist, wake = {}, {}, {}
    for i, d in enumerate([800.0, 1600.0, 2400.0, 3200.0, 4000.0]):
        for cls in ("H", "M"):
            key = ("ZZZZ", f"S{i}{cls}", "27")
            dist[key], wake[key] = d, cls
            pairs[key] = [a + d / v] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist, wake=wake)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref), class_column=None)
    assert list(model.class_intercept) == [("ZZZZ", "all")]
    assert model.class_intercept[("ZZZZ", "all")] == pytest.approx(a, rel=1e-9)
    # and a frame carrying classes is predicted the same, because none is read
    assert np.allclose(model.predict(frame), ref.predict(frame), rtol=1e-9)


def test_a_thin_class_is_pooled_and_an_unseen_class_cascades():
    from taxiout.unimpeded import OTHER_CLASS

    v = 8.0
    pairs, dist, wake = {}, {}, {}
    for i, d in enumerate([800.0, 1600.0, 2400.0]):
        key = ("ZZZZ", f"S{i}", "27")
        dist[key], wake[key] = d, "M"
        pairs[key] = [100.0 + d / v] * MIN_PAIR_COUNT
    # a light with one row at every distance: too thin for its own intercept
    for i, d in enumerate([800.0, 1600.0, 2400.0]):
        key = ("ZZZZ", f"L{i}", "27")
        dist[key], wake[key] = d, "L"
        pairs[key] = [400.0 + d / v] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist, wake=wake)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(
        frame, ref, lower_tail(frame, y, ref), min_class_rows=4 * MIN_PAIR_COUNT
    )
    assert ("ZZZZ", "L") not in model.class_intercept
    assert ("ZZZZ", OTHER_CLASS) in model.class_intercept
    # a class the fit never saw at all cascades to the pool, then the airport
    unseen = frame.iloc[[0]].assign(wake_class="J")
    assert model.intercepts(unseen)[0] == pytest.approx(
        model.class_intercept[("ZZZZ", OTHER_CLASS)]
    )
    other_airport = frame.iloc[[0]].assign(icao="QQQQ", wake_class="M")
    assert model.intercepts(other_airport)[0] == pytest.approx(model.default_intercept)


def test_a_non_positive_slope_falls_back_to_the_pooled_one():
    """An airport whose own fit runs backwards is not allowed to."""
    v = 8.0
    pairs, dist = {}, {}
    for i, d in enumerate([800.0, 1200.0, 1600.0, 2000.0, 2400.0]):
        key = ("AAAA", f"S{i}", "27")
        dist[key] = d
        pairs[key] = [100.0 + d / v] * MIN_PAIR_COUNT
    # BBBB: the longer the route, the *shorter* the reference
    for i, (d, r) in enumerate(
        [
            (800.0, 500.0),
            (1200.0, 450.0),
            (1600.0, 400.0),
            (2000.0, 350.0),
            (2400.0, 300.0),
        ]
    ):
        key = ("BBBB", f"S{i}", "27")
        dist[key] = d
        pairs[key] = [r] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref), class_column=None)
    assert model.inverse_speed["AAAA"] == pytest.approx(1.0 / v, rel=1e-9)
    # BBBB is pooled outright: a slope that fails must not leave its intercept
    # behind, and the shared modulators are refitted without its rows
    assert model.pooled_airports == ("BBBB",)
    assert "BBBB" not in model.inverse_speed and "BBBB" not in model.airport_intercept
    assert model.default_inverse_speed > 0
    got = model.predict(frame[frame["icao"] == "BBBB"])
    assert (got > 0).all() and (got < 10_000).all()


def test_an_airport_with_too_few_pairs_takes_the_pooled_parameters():
    """A single pair cannot identify a line, so the airport is pooled and said so."""
    v = 8.0
    pairs, dist = {}, {}
    for i, d in enumerate([800.0, 1200.0, 1600.0, 2000.0, 2400.0]):
        key = ("AAAA", f"S{i}", "27")
        dist[key] = d
        pairs[key] = [100.0 + d / v] * MIN_PAIR_COUNT
    thin = ("BBBB", "S0", "27")
    dist[thin] = 3000.0
    pairs[thin] = [100.0 + 3000.0 / v] * MIN_PAIR_COUNT
    frame, y = _frame(pairs, distances=dist)
    ref = fit_pru_reference(frame, y)
    model = fit_geometric(frame, ref, lower_tail(frame, y, ref), class_column=None)
    assert model.pooled_airports == ("BBBB",)
    assert "BBBB" not in model.inverse_speed
    assert model.inverse_speed["AAAA"] == pytest.approx(1.0 / v, rel=1e-9)
    # the pooled line was fitted on every row, thin airport included, and on
    # this exact data it is the same line
    assert model.default_inverse_speed == pytest.approx(1.0 / v, rel=1e-9)
    assert model.default_intercept == pytest.approx(100.0, rel=1e-9)
    got = model.predict(frame[frame["icao"] == "BBBB"]).iloc[0]
    assert got == pytest.approx(100.0 + 3000.0 / v, rel=1e-9)
