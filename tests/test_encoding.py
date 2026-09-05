# SPDX-License-Identifier: GPL-3.0-only
"""The leakage guards on the shrunk identifier encoding, and the estimator for ``k``.

Every test in the first half exists because the corresponding mistake makes the
number *better*. A target encoding that quietly sees the row it is encoding, or
a pair that quietly sees itself across a fold boundary, does not fail loudly --
it reports an improvement that is not there, and the held-out-pair contrast the
whole entry rests on becomes a measurement of the bug. So each guard is
asserted rather than argued: that a pair held out of a fold encodes to the
prior exactly, that a row's own target is absent from its own feature, that a
flagged row cannot reach a group mean, and that computing the encoding once and
reusing it is a visibly different number from computing it per fold.

The second half checks the estimator. ``k`` is not tuned, so what can be
checked is that it is *estimated correctly*: that the between-group variance
has the sampling variance of the group means taken out of it, that it is
per-airport rather than pooled, and that the length-dependence rule fires when
the variance really does grow with route length and stays quiet when it does
not.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.encoding import (  # noqa: E402
    ENCODED_COLUMN,
    ENCODING_FEATURES,
    INNER_FOLDS,
    LEVELS,
    MIN_BETWEEN_FRACTION,
    encode_level,
    encode_split,
    estimate_shrinkage,
    fit_encoding,
    geometric_prior,
    group_key,
    with_encoding,
    within_variance,
)

MODULATORS = (
    "turn_deg_per_km",
    "junctions_per_km",
    "straight_frac",
    "backtrack_frac",
    "detour_ratio",
)


def _features(
    spec: list[tuple[str, str, str, int, float]],
    *,
    seed: int = 0,
    within_sd: float = 60.0,
    slope_s_per_m: float = 0.10,
    intercept_s: float = 300.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """A features frame and a target from ``(icao, stand, runway, rows, taxi_m)``.

    The target is ``intercept + slope * taxi_m`` plus a per-pair offset and
    per-row noise, so the geometry explains most of it and the pair means have
    a real spread around the geometry -- which is the situation the encoding is
    built for.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for icao, stand, runway, n, taxi_m in spec:
        pair_offset = rng.normal(0.0, 120.0)
        for _ in range(n):
            rows.append(
                {
                    "icao": icao,
                    "stand": stand,
                    "designator": runway,
                    "taxi_m": taxi_m,
                    "geom_imputed": 0.0,
                    "_mu": intercept_s + slope_s_per_m * taxi_m + pair_offset,
                }
            )
    frame = pd.DataFrame(rows)
    frame["pair"] = frame["icao"] + "|" + frame["stand"] + "|" + frame["designator"]
    frame["route_m"] = frame["taxi_m"]
    frame["straight_m"] = 0.5 * frame["taxi_m"]
    frame["turn_deg_per_km"] = 90.0
    frame["junctions_per_km"] = 4.0
    frame["straight_frac"] = 0.5
    frame["backtrack_frac"] = 0.0
    frame["detour_ratio"] = 1.25
    y = pd.Series(
        frame["_mu"].to_numpy() + rng.normal(0.0, within_sd, len(frame)),
        index=frame.index,
    )
    return frame.drop(columns=["_mu"]), y


def _grid(icao: str, n_pairs: int, n_rows: int, base_m: float = 800.0):
    return [(icao, f"S{i}", "27", n_rows, base_m + 150.0 * i) for i in range(n_pairs)]


def _flat_prior(frame: pd.DataFrame, value: float = 500.0) -> pd.Series:
    """A prior that is constant within a pair, which is what the pair level assumes."""
    per_pair = {p: value + 5.0 * i for i, p in enumerate(sorted(frame["pair"].unique()))}
    return frame["pair"].map(per_pair).astype(float)


# --------------------------------------------------------------------------
# the leakage guards


def test_a_pair_held_out_of_the_fold_encodes_to_the_prior_exactly():
    """The guard the held-out-pair contrast depends on.

    If the encoding were computed once over all training rows and reused inside
    the folds, a held-out pair would arrive carrying its own mean, the contrast
    would inflate, and nothing would look wrong. Held out, it must see ``n = 0``
    and fall back to the prior with no empirical component at all.
    """
    frame, y = _features(_grid("ZZZZ", 9, 40))
    held = frame["pair"] == frame["pair"].unique()[3]
    fit_rows = ~held

    prior = _flat_prior(frame)
    encoded, shrink = encode_level(frame, y, prior, fit_rows, "pair")

    assert np.allclose(encoded[held], prior[held])
    # and the pairs that stayed in are not at the prior: the feature is doing
    # something for them, so the assertion above is not vacuous.
    assert not np.allclose(encoded[~held], prior[~held])
    assert shrink.level == "pair"


def test_a_rows_own_target_is_absent_from_its_own_encoding():
    """Out of fold within training, not merely out of sample.

    Moving one row's target by an hour must leave that row's own encoding
    untouched, because the row sits in an inner fold its own encoding is built
    without. Its group-mates in other folds do move, which is what makes this
    an out-of-fold encoding rather than no encoding at all.
    """
    frame, y = _features(_grid("ZZZZ", 8, 50))
    fit_rows = pd.Series(True, index=frame.index)
    prior = _flat_prior(frame)

    before, _ = encode_level(frame, y, prior, fit_rows, "pair")
    poisoned = y.copy()
    victim = 17
    poisoned.iloc[victim] += 3600.0
    after, _ = encode_level(frame, poisoned, prior, fit_rows, "pair")

    # Unchanged to floating point. Not asserted as exact equality: the group
    # totals are formed and then differenced, so a changed summand reassociates
    # the arithmetic and moves the last bit. A nanosecond against the hour that
    # was added is the difference between "not in the arithmetic" and "in it".
    assert abs(after.iloc[victim] - before.iloc[victim]) < 1e-9

    same_pair = frame.index[frame["pair"] == frame["pair"].iloc[victim]]
    delta = pd.Series(
        np.abs(after[same_pair].to_numpy() - before[same_pair].to_numpy()),
        index=same_pair,
    )
    # the test is not vacuous: group-mates in other folds move by seconds
    assert delta.max() > 1.0
    assert delta.loc[frame.index[victim]] < 1e-9

    # Rows of other pairs do move a little, and that is correct rather than a
    # leak: they move through ``k``, which is a per-airport variance estimated
    # from the fit rows and is therefore a function of every other row's target
    # the way any fitted quantity is. What must not happen is a row's *own*
    # target reaching it, and ``k`` is estimated out of fold for exactly that
    # reason -- with a single ``k`` over all the fit rows this assertion failed
    # by twelve seconds, which is how the out-of-fold ``k`` came to exist.
    other = frame.index.difference(same_pair)
    assert np.abs(after[other].to_numpy() - before[other].to_numpy()).max() > 0.0


def test_the_inner_folds_leave_a_row_out_and_not_its_whole_group():
    """The reason the inner folds are over rows rather than over groups.

    Grouped inner folds would remove every one of a row's group-mates with it,
    leave ``n = 0`` for every training row, and hand the model a feature that is
    the prior in training and a real mean in validation -- which is worse than
    no feature.
    """
    frame, y = _features(_grid("ZZZZ", 6, 60))
    fit_rows = pd.Series(True, index=frame.index)
    prior = _flat_prior(frame)
    encoded, _ = encode_level(frame, y, prior, fit_rows, "pair")
    assert not np.allclose(encoded, prior)


def test_a_flagged_row_cannot_reach_a_group_mean():
    """One defaulted block time at 50,000 s in a thin pair would destroy it.

    The caller passes fit rows with the record-quality mask already removed, so
    the assertion is that a flagged row present in the frame changes nothing
    at all against the same frame without it.
    """
    frame, y = _features(_grid("ZZZZ", 7, 40))
    thin = frame["pair"].unique()[2]
    extra = frame[frame["pair"] == thin].iloc[:1].copy()
    with_extra = pd.concat([frame, extra], ignore_index=True)
    y_extra = pd.concat([y, pd.Series([50_000.0])], ignore_index=True)

    fit_rows = pd.Series(True, index=with_extra.index)
    fit_rows.iloc[-1] = False  # the flagged row
    prior_extra = _flat_prior(with_extra)

    encoded_extra, shrink_extra = encode_level(
        with_extra, y_extra, prior_extra, fit_rows, "pair"
    )
    encoded_plain, shrink_plain = encode_level(
        frame, y, _flat_prior(frame), pd.Series(True, index=frame.index), "pair"
    )

    assert np.allclose(encoded_extra.iloc[:-1], encoded_plain)
    assert shrink_extra.k == pytest.approx(shrink_plain.k)
    # the flagged row itself still gets a value, from the unflagged rows only
    assert np.isfinite(encoded_extra.iloc[-1])
    assert encoded_extra.iloc[-1] != pytest.approx(50_000.0, abs=1_000.0)


def test_computing_the_encoding_once_and_reusing_it_is_a_different_number():
    """The defect stated as a measurement, so it cannot be argued away.

    Computed on all the rows, the held-out pair receives its own mean; computed
    on the fold's own fit rows, it receives the prior. The two differ, and the
    once-computed version is the one that looks better.
    """
    frame, y = _features(_grid("ZZZZ", 9, 40))
    held = frame["pair"] == frame["pair"].unique()[5]
    prior = _flat_prior(frame)

    correct, _ = encode_level(frame, y, prior, ~held, "pair")
    leaky, _ = encode_level(frame, y, prior, pd.Series(True, index=frame.index), "pair")

    assert not np.allclose(correct[held], leaky[held])
    # the leaky version has pulled the held-out pair toward its own mean
    own_mean = float(y[held].mean())
    assert abs(float(leaky[held].mean()) - own_mean) < abs(
        float(correct[held].mean()) - own_mean
    )


def test_the_encoding_is_the_stated_formula_at_pair_level():
    """``(n * ybar + k * g_hat) / (n + k)``, recovered from the residual form.

    The module writes the encoding as a shrinkage of the group residual so that
    it also works at stand level, where the prior varies within the group. At
    pair level, where it does not, the two forms must be the same number.
    """
    frame, y = _features(_grid("ZZZZ", 8, 45))
    held = frame["pair"] == frame["pair"].unique()[1]
    fit_rows = ~held
    prior = _flat_prior(frame)
    encoded, shrink = encode_level(frame, y, prior, fit_rows, "pair")

    assert np.isfinite(shrink.default_k)
    seen = frame.index[fit_rows][0]
    g = frame.loc[seen, "pair"]
    mates = fit_rows & (frame["pair"] == g)
    assert mates.sum() > 1, "no group-mates to rebuild from"
    ghat = float(prior.loc[seen])
    value = float(encoded.loc[seen])
    # whatever the row's inner fold was, the encoding lies between the prior
    # and the group mean: it is a convex combination of the two and nothing else
    ybar = float(y[mates].mean())
    lo, hi = sorted((ghat, ybar))
    assert lo - 1e-6 <= value <= hi + 1e-6

    # the identity itself, checked exactly on a held-out pair (n = 0) and on a
    # synthetic group whose sums are known
    assert float(encoded[held].iloc[0]) == pytest.approx(float(prior[held].iloc[0]))
    n, ybar_, ghat_, k_ = 7.0, 900.0, 500.0, 3.0
    residual_form = ghat_ + n / (n + k_) * (ybar_ - ghat_)
    stated_form = (n * ybar_ + k_ * ghat_) / (n + k_)
    assert residual_form == pytest.approx(stated_form)


def test_the_stand_level_is_the_same_construction_on_the_coarser_key():
    frame, y = _features(
        [("ZZZZ", "S0", "27", 40, 800.0), ("ZZZZ", "S0", "09", 40, 1400.0)]
        + _grid("ZZZZ", 6, 40)
    )
    keys = group_key(frame, "stand")
    assert keys.nunique() < group_key(frame, "pair").nunique()
    held = keys == "ZZZZ|S0"
    prior = pd.Series(400.0 + 0.1 * frame["taxi_m"], index=frame.index)
    encoded, _ = encode_level(frame, y, prior, ~held, "stand")
    assert np.allclose(encoded[held], prior[held])


def test_group_key_rejects_an_unknown_level():
    frame, _ = _features(_grid("ZZZZ", 3, 5))
    with pytest.raises(KeyError):
        group_key(frame, "runway")


# --------------------------------------------------------------------------
# the estimator


def _moments(n, resid, s2_within, length):
    """Group moments as ``_group_moments`` would produce them, from planted values.

    ``resid`` is the group's mean residual against the prior, ``ybar - g_hat_bar``:
    the moments the estimator works from are of ``y - g_hat`` and not of ``y``.
    """
    n = np.asarray(n, dtype=float)
    dbar = np.asarray(resid, dtype=float)
    s2 = np.asarray(s2_within, dtype=float)
    length = np.asarray(length, dtype=float)
    return {
        "n": n,
        "sum_d": n * dbar,
        "sum_d2": s2 * np.maximum(n - 1, 0) + n * dbar**2,
        "sum_len": n * length,
    }


def test_the_between_variance_has_the_sampling_variance_subtracted():
    """The correction, planted so its size is known rather than asserted.

    Groups whose true means all sit exactly on the prior still show a spread in
    their *observed* means, purely because each mean is an average of finitely
    many noisy rows. Left in, that spread is read as between-group variance,
    the between-variance is too large and ``k`` too small. Subtracting
    ``E[sigma2_within / n]`` is what removes it.
    """
    rng = np.random.default_rng(11)
    n_groups, n_rows, s2 = 400, 40, 3600.0
    n = np.full(n_groups, float(n_rows))
    # true group means equal the prior; the observed means differ by noise only
    resid = rng.normal(0.0, np.sqrt(s2 / n_rows), n_groups)
    moments = _moments(n, resid, np.full(n_groups, s2), np.full(n_groups, 900.0))
    airports = np.array(["ZZZZ"] * n_groups, dtype=object)

    shrink = estimate_shrinkage("pair", airports, moments, min_group_rows=30)
    raw = shrink.raw_between["ZZZZ"]
    corrected = shrink.between["ZZZZ"]
    expected_correction = s2 / n_rows

    assert shrink.correction["ZZZZ"] == pytest.approx(expected_correction, rel=1e-6)
    assert raw == pytest.approx(expected_correction, rel=0.20)
    # the correction removes essentially all of it: what is left is far smaller
    assert corrected < 0.35 * raw
    # and there is a real difference in k, in the stated direction
    assert shrink.within.constant["ZZZZ"] == pytest.approx(s2, rel=1e-6)
    assert s2 / corrected > s2 / raw


def test_leaving_the_correction_out_understates_k():
    """The 13% the prompt for this work predicted, as a direction rather than a size."""
    rng = np.random.default_rng(3)
    n_groups, n_rows, s2, between = 600, 45, 3600.0, 20_000.0
    n = np.full(n_groups, float(n_rows))
    truth = rng.normal(0.0, np.sqrt(between), n_groups)
    resid = truth + rng.normal(0.0, np.sqrt(s2 / n_rows), n_groups)
    moments = _moments(n, resid, np.full(n_groups, s2), np.full(n_groups, 900.0))
    airports = np.array(["ZZZZ"] * n_groups, dtype=object)
    shrink = estimate_shrinkage("pair", airports, moments, min_group_rows=30)

    k_corrected = shrink.k["ZZZZ"]
    k_uncorrected = s2 / shrink.raw_between["ZZZZ"]
    assert k_corrected > k_uncorrected
    # the planted between-variance is recovered to within sampling error
    assert shrink.between["ZZZZ"] == pytest.approx(between, rel=0.15)


def test_k_is_estimated_per_airport_and_not_pooled():
    """An airport whose prior fits well must shrink toward it more, not equally."""
    rng = np.random.default_rng(5)
    per = 300
    n = np.full(2 * per, 40.0)
    good = rng.normal(0.0, np.sqrt(2_000.0), per)
    poor = rng.normal(0.0, np.sqrt(80_000.0), per)
    resid = np.concatenate([good, poor])
    moments = _moments(n, resid, np.full(2 * per, 3600.0), np.full(2 * per, 900.0))
    airports = np.array(["GOOD"] * per + ["POOR"] * per, dtype=object)
    shrink = estimate_shrinkage("pair", airports, moments, min_group_rows=30)

    assert shrink.k["GOOD"] > shrink.k["POOR"]
    assert set(shrink.groups_used) == {"GOOD", "POOR"}
    # a row at each airport gets its own k
    k = shrink.k_for(np.array(["GOOD", "POOR"], dtype=object), np.array([900.0, 900.0]))
    assert k[0] > k[1]


def test_an_airport_the_geometry_explains_to_within_noise_takes_the_floor():
    """Rather than dividing by a negative variance and reporting an absurdity."""
    rng = np.random.default_rng(7)
    per = 200
    n = np.full(2 * per, 40.0)
    exact = rng.normal(0.0, np.sqrt(3600.0 / 40.0) * 0.05, per)
    spread = rng.normal(0.0, np.sqrt(50_000.0), per)
    moments = _moments(
        n,
        np.concatenate([exact, spread]),
        np.full(2 * per, 3600.0),
        np.full(2 * per, 900.0),
    )
    airports = np.array(["TIGHT"] * per + ["LOOSE"] * per, dtype=object)
    shrink = estimate_shrinkage("pair", airports, moments, min_group_rows=30)
    assert "TIGHT" in shrink.floored
    assert "LOOSE" not in shrink.floored
    assert np.isfinite(shrink.k["TIGHT"])
    assert shrink.between["TIGHT"] == pytest.approx(
        MIN_BETWEEN_FRACTION * shrink.raw_between["TIGHT"]
    )
    # the floor is what keeps k finite; without it this airport divides by zero
    assert shrink.k["TIGHT"] > shrink.k["LOOSE"]


def test_the_length_rule_fires_when_the_variance_grows_with_the_route():
    """Planted both ways, against the threshold fixed in the module."""
    n_groups = 400
    length = np.linspace(400.0, 4000.0, n_groups)
    n = np.full(n_groups, 40.0)
    airports = np.array(["ZZZZ"] * n_groups, dtype=object)

    growing = _moments(n, np.zeros(n_groups), 4.0 * length, length)
    flat = _moments(n, np.zeros(n_groups), np.full(n_groups, 3600.0), length)

    assert within_variance(airports, growing).material
    assert not within_variance(airports, flat).material
    grew = within_variance(airports, growing)
    assert grew.ratio > 1.5
    assert grew.per_airport_ratio["ZZZZ"] == pytest.approx(grew.ratio, rel=1e-9)


def test_a_material_length_dependence_makes_k_move_with_the_route():
    n_groups = 400
    length = np.linspace(400.0, 4000.0, n_groups)
    n = np.full(n_groups, 40.0)
    airports = np.array(["ZZZZ"] * n_groups, dtype=object)
    rng = np.random.default_rng(13)
    resid = rng.normal(0.0, 150.0, n_groups)
    moments = _moments(n, resid, 4.0 * length, length)
    shrink = estimate_shrinkage("pair", airports, moments, min_group_rows=30)
    assert shrink.within.material
    k = shrink.k_for(np.array(["ZZZZ", "ZZZZ"], dtype=object), np.array([500.0, 3800.0]))
    assert k[1] > k[0]


def test_the_stand_levels_within_variance_is_of_the_residual_and_not_of_the_target():
    """The estimator works on ``y - g_hat``, and at stand level that is the difference.

    A stand's pairs route to different runways, so the prior is not constant
    within the group. Taking the moments of ``y`` would read every one of those
    geometric differences as within-group noise, inflate ``sigma2_within``,
    inflate ``k`` with it, and shrink the stand encoding toward the prior for a
    reason that is arithmetic rather than data. Here the planted within-group
    noise is the same at both levels, so the two estimates must agree.
    """
    # One offset per *stand*, shared by its three pairs, so a stand is
    # homogeneous once the geometry is taken out and the residual scatter is
    # the planted 60 s at both levels. Anything above that at stand level is
    # the prior's own between-runway spread being counted as noise.
    rng = np.random.default_rng(4)
    rows, targets = [], []
    for i in range(40):
        offset = rng.normal(0.0, 120.0)
        for runway, taxi_m in (("27", 600.0), ("09", 2000.0), ("18", 3400.0)):
            for _ in range(40):
                rows.append(
                    {
                        "icao": "ZZZZ",
                        "stand": f"S{i}",
                        "designator": runway,
                        "taxi_m": taxi_m,
                        "geom_imputed": 0.0,
                    }
                )
                targets.append(300.0 + 0.10 * taxi_m + offset + rng.normal(0.0, 60.0))
    frame = pd.DataFrame(rows)
    frame["pair"] = frame["icao"] + "|" + frame["stand"] + "|" + frame["designator"]
    y = pd.Series(targets, index=frame.index)
    fit_rows = pd.Series(True, index=frame.index)
    prior = pd.Series(300.0 + 0.10 * frame["taxi_m"], index=frame.index)

    _, pair = encode_level(frame, y, prior, fit_rows, "pair")
    _, stand = encode_level(frame, y, prior, fit_rows, "stand")

    # the planted noise is 60 s, and both levels recover it
    assert np.sqrt(pair.within.constant["ZZZZ"]) == pytest.approx(60.0, rel=0.05)
    assert np.sqrt(stand.within.constant["ZZZZ"]) == pytest.approx(60.0, rel=0.05)

    # and the failure this guards against, exhibited: moments of the raw target
    # at stand level see the between-runway spread of the prior as well, which
    # is 280 s of route difference on top of 60 s of noise.
    raw = float(y.groupby(frame["icao"] + "|" + frame["stand"]).var(ddof=1).mean())
    assert raw > 4.0 * stand.within.constant["ZZZZ"]


def test_the_length_ratio_is_read_off_an_unfloored_fit():
    """A guard must not be able to decide the check it guards.

    ``WithinVariance.at`` floors the fitted variance so an extrapolated line
    cannot return a negative one. If the *ratio* were read after that floor, a
    variance growing faster than linearly -- whose straight-line fit has a
    negative intercept -- would have its short end pinned at the floor, and the
    measured ratio would be an artefact of the guard: able to overstate the
    length dependence and never to understate it. An airport whose fit is not
    positive across the range abstains instead.
    """
    n_groups = 400
    length = np.linspace(400.0, 4000.0, n_groups)
    n = np.full(n_groups, 40.0)
    airports = np.array(["ZZZZ"] * n_groups, dtype=object)

    # quadratic growth: the linear fit's intercept goes negative
    quadratic = _moments(n, np.zeros(n_groups), 4e-4 * length**2, length)
    w = within_variance(airports, quadratic)
    assert w.intercept["ZZZZ"] < 0
    assert "ZZZZ" not in w.per_airport_ratio
    assert not w.decisive
    assert not w.material

    # a genuinely linear growth still fires, so the abstention is not blanket
    linear = within_variance(
        airports, _moments(n, np.zeros(n_groups), 4.0 * length, length)
    )
    assert linear.decisive and linear.material


def test_the_reported_k_span_is_what_the_feature_uses():
    """When the length rule fires, one number per airport is not the answer."""
    n_groups = 400
    length = np.linspace(400.0, 4000.0, n_groups)
    n = np.full(n_groups, 40.0)
    airports = np.array(["ZZZZ"] * n_groups, dtype=object)
    rng = np.random.default_rng(13)
    moments = _moments(n, rng.normal(0.0, 150.0, n_groups), 4.0 * length, length)
    shrink = estimate_shrinkage("pair", airports, moments, min_group_rows=30)
    assert shrink.within.material
    low, high = shrink.k_span("ZZZZ")
    assert low < high
    lo_len, hi_len = shrink.within.span["ZZZZ"]
    direct = shrink.k_for(
        np.array(["ZZZZ", "ZZZZ"], dtype=object), np.array([lo_len, hi_len])
    )
    assert (low, high) == pytest.approx(tuple(direct))

    # and where it is not material the span collapses to the single number
    flat = estimate_shrinkage(
        "pair",
        airports,
        _moments(n, rng.normal(0.0, 150.0, n_groups), np.full(n_groups, 3600.0), length),
        min_group_rows=30,
    )
    assert not flat.within.material
    assert flat.k_span("ZZZZ") == (flat.k["ZZZZ"], flat.k["ZZZZ"])


def test_the_within_variance_recovers_a_planted_one():
    n_groups = 50
    n = np.full(n_groups, 30.0)
    moments = _moments(
        n,
        np.full(n_groups, 200.0),
        np.full(n_groups, 2500.0),
        np.full(n_groups, 900.0),
    )
    airports = np.array(["ZZZZ"] * n_groups, dtype=object)
    within = within_variance(airports, moments)
    assert within.constant["ZZZZ"] == pytest.approx(2500.0)
    assert within.pooled == pytest.approx(2500.0)


# --------------------------------------------------------------------------
# the scale, and the frame


def test_k_scale_moves_the_weight_toward_the_prior_and_away_from_it():
    """The sensitivity knob, which is a diagnostic and never a fitted quantity."""
    frame, y = _features(_grid("ZZZZ", 8, 45))
    fit_rows = pd.Series(True, index=frame.index)
    prior = _flat_prior(frame)
    at_k, _ = encode_level(frame, y, prior, fit_rows, "pair")
    half, _ = encode_level(frame, y, prior, fit_rows, "pair", k_scale=0.5)
    double, _ = encode_level(frame, y, prior, fit_rows, "pair", k_scale=2.0)

    # a bigger k shrinks harder: closer to the prior
    to_prior = lambda v: float(np.abs(v - prior).mean())  # noqa: E731
    assert to_prior(double) < to_prior(at_k) < to_prior(half)


def test_encode_split_produces_both_columns_and_their_shrinkages():
    frame, y = _features(_grid("ZZZZ", 9, 45))
    held = frame["pair"] == frame["pair"].unique()[4]
    out = encode_split(frame, y, ~held)
    assert list(out.columns.columns) == list(ENCODING_FEATURES)
    assert set(out.shrinkage) == set(LEVELS)
    assert out.folds == INNER_FOLDS
    assert out.fit_rows == int((~held).sum())
    assert out.columns.index.equals(frame.index)
    for level in LEVELS:
        assert ENCODED_COLUMN[level] in out.columns.columns
    # the prior is fitted, not passed: it has a speed and a fit size
    assert out.prior.n_fit > 0
    assert np.isfinite(out.prior.default_speed)


def test_with_encoding_overlays_without_copying_the_rest():
    frame, y = _features(_grid("ZZZZ", 8, 40))
    out = encode_split(frame, y, pd.Series(True, index=frame.index))
    merged = with_encoding(frame, out)
    for c in ENCODING_FEATURES:
        assert c in merged.columns
    assert merged["taxi_m"].to_numpy().base is not None or np.shares_memory(
        merged["taxi_m"].to_numpy(), frame["taxi_m"].to_numpy()
    )
    assert with_encoding(frame, None) is frame


def test_the_prior_is_offset_onto_the_scale_of_the_target():
    """Otherwise it is an unimpeded time and every thin pair is dragged minutes low."""
    frame, y = _features(_grid("ZZZZ", 10, 60))
    fit_rows = pd.Series(True, index=frame.index)
    prior = geometric_prior(frame, y, fit_rows)
    assert abs(float((y - prior.values).mean())) < abs(
        float((y - prior.unshifted).mean())
    )
    assert float((y - prior.values).mean()) == pytest.approx(0.0, abs=1.0)
    assert set(prior.offset.index) == {"ZZZZ"}


def test_the_builder_leaves_the_block_missing_without_an_encoder():
    """Missing, not zero: "not computed" is not "no history"."""
    from taxiout.features import build_features
    from taxiout.routing import AGGREGATED_FIELDS

    matrix = pd.DataFrame(
        [
            {
                "icao": "ZZZZ",
                "stand_id": "B5",
                "designator": "27",
                "runway_ref": "27/xx",
                "n_features": 1,
                "spread_m": 0.0,
                "snap_max_m": 0.0,
                **{f: 0.0 for f in AGGREGATED_FIELDS},
                "taxi_m": 1000.0,
                "route_m": 1000.0,
                "straight_m": 500.0,
                "turn_deg": 90.0,
                "junctions": 4.0,
                "gc_m": 800.0,
                "detour_ratio": 1.25,
            }
        ]
    )
    movements = pd.DataFrame(
        {
            "MVT_ID_mvt": [1000.0, 1001.0],
            "ADEP_mvt": ["ZZZZ"] * 2,
            "STAND_mvt": ["B5"] * 2,
            "RUNWAY_mvt": ["27"] * 2,
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-03-04T08:30:00Z"] * 2, utc=True),
            "SCHED_TIME_UTC_mvt": pd.to_datetime(["2025-03-04T08:00:00Z"] * 2, utc=True),
            "LOBT_flt": pd.to_datetime(["2025-03-04T08:10:00Z"] * 2, utc=True),
            "IOBT_flt": pd.to_datetime(["2025-03-04T08:00:00Z"] * 2, utc=True),
            "EOBT_1_flt": pd.to_datetime(["2025-03-04T08:00:00Z"] * 2, utc=True),
            "WK_TBL_CAT_flt": ["M"] * 2,
            "AIRCRAFT_TYPE_mvt": ["A320"] * 2,
        }
    )
    built = build_features(movements, matrix)
    for c in ENCODING_FEATURES:
        assert built[c].isna().all()

    overlay = pd.DataFrame(
        {c: [1.0, 2.0] for c in ENCODING_FEATURES}, index=movements.index
    )
    with_block = build_features(movements, matrix, encoded=overlay)
    for c in ENCODING_FEATURES:
        assert with_block[c].tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------
# the inference-time encoder
#
# The ladder's guards are about its own folds. These are about the submission,
# which is a different arrangement: the fitting sample is the whole 2025 year
# and the rows being encoded are 2026 rows that are in no fold at all. Two
# things have to hold there, and neither is implied by the tests above --
# ``fit_encoding`` and ``FittedEncoding.transform`` are a separate code path
# from ``encode_level``, and a defect in them would reach the submitted model
# without any split downstream to catch it.


def test_the_fitted_encoder_reads_unflagged_rows_only():
    """Guard 3, at submission, where it matters more and not less.

    In the ladder a defaulted block time inside a thin pair corrupts one fold's
    feature and the other folds disagree with it. In the submission there is no
    other fold: the corrupted encoding is what ships. Same plant as
    ``test_a_flagged_row_cannot_reach_a_group_mean``, against the inference
    path this time, and asserted on the *transformed* columns rather than on
    the internals, because the columns are what the model reads.
    """
    frame, y = _features(_grid("ZZZZ", 7, 40))
    thin = frame["pair"].unique()[2]
    extra = frame[frame["pair"] == thin].iloc[:1].copy()
    with_extra = pd.concat([frame, extra], ignore_index=True)
    y_extra = pd.concat([y, pd.Series([50_000.0])], ignore_index=True)

    fit_rows = pd.Series(True, index=with_extra.index)
    fit_rows.iloc[-1] = False  # the flagged row, excluded by the caller
    poisoned = fit_encoding(with_extra, y_extra, fit_rows)
    clean = fit_encoding(frame, y, pd.Series(True, index=frame.index))

    # The rows the flagged one shares a pair with must not have moved.
    got = poisoned.transform(with_extra).iloc[:-1]
    want = clean.transform(frame)
    for column in ENCODING_FEATURES:
        assert np.allclose(got[column], want[column], atol=1e-4)

    # And the assertion is not vacuous: the same row left *in* the fit moves
    # the pair's encoding by minutes.
    leaked = fit_encoding(
        with_extra, y_extra, pd.Series(True, index=with_extra.index)
    ).transform(with_extra)
    same_pair = (with_extra["pair"] == thin).to_numpy()
    moved = abs(
        float(leaked.loc[same_pair, ENCODED_COLUMN["pair"]].mean())
        - float(
            want.loc[(frame["pair"] == thin).to_numpy(), ENCODED_COLUMN["pair"]].mean()
        )
    )
    assert moved > 60.0


def test_a_group_the_fitting_year_never_saw_encodes_to_the_prior_exactly():
    """The ranking set's own case: a 2026 pair with no 2025 rows behind it.

    Reached by the arithmetic rather than by a branch -- ``n = 0`` makes the
    shrinkage weight zero -- which is why it is worth asserting: a lookup that
    silently filled a missing group with a mean, or with zero, would be a
    plausible-looking bug that only shows on the thinnest rows of the test set.
    """
    frame, y = _features(_grid("ZZZZ", 8, 40))
    known = frame["pair"].unique()[:6]
    fitting = frame[frame["pair"].isin(known)].reset_index(drop=True)
    y_fit = y[frame["pair"].isin(known)].reset_index(drop=True)
    unseen = frame[~frame["pair"].isin(known)].reset_index(drop=True)

    encoder = fit_encoding(fitting, y_fit, pd.Series(True, index=fitting.index))
    columns = encoder.transform(unseen)
    prior = encoder.prior.prior_for(unseen)

    assert np.allclose(columns[ENCODED_COLUMN["pair"]], prior, atol=1e-4)
    # a pair the year did see is not at its prior, so the check has content
    seen_cols = encoder.transform(fitting)
    assert not np.allclose(
        seen_cols[ENCODED_COLUMN["pair"]], encoder.prior.prior_for(fitting), atol=1e-4
    )


def test_the_encoding_written_at_submission_is_reproducible_from_the_year():
    """Determinism: the same year and the same rows give the same columns.

    ``make_submission.py`` computes the encoding once and writes a submission
    from it. If that computation depended on row order, on a seed, or on any
    state carried between calls, the submission would not be reproducible from
    the training year -- which is the claim §2.3 of the paper makes and the one
    a reproduction would check. There are no folds on this path, so unlike
    ``encode_split`` there is no seed to hold fixed: equality is exact.
    """
    frame, y = _features(_grid("ZZZZ", 9, 40))
    fit_rows = pd.Series(True, index=frame.index)
    rank, _ = _features(_grid("ZZZZ", 9, 5), seed=11)

    first = fit_encoding(frame, y, fit_rows).transform(rank)
    second = fit_encoding(frame, y, fit_rows).transform(rank)
    for column in ENCODING_FEATURES:
        assert first[column].equals(second[column])

    # Row order of the fitting sample must not move the answer either: the
    # moments are sums over groups, and a sum is order-free only if nothing
    # downstream of it depends on which row arrived first.
    shuffled = frame.sample(frac=1.0, random_state=3)
    reordered = fit_encoding(
        shuffled, y.loc[shuffled.index], fit_rows.loc[shuffled.index]
    ).transform(rank)
    for column in ENCODING_FEATURES:
        assert np.allclose(first[column], reordered[column], atol=1e-4)


def test_the_inference_path_and_the_fold_path_agree_on_a_row_outside_the_fit():
    """The two paths are one construction, so they must meet where both apply.

    ``encode_level`` encodes a non-fit row from all the fit rows -- which is
    exactly what ``transform`` does for every row it sees. Where the two are
    asked the same question they must give the same answer, or one of them is
    not the encoding this module documents.
    """
    frame, y = _features(_grid("ZZZZ", 9, 40))
    held = (frame["pair"] == frame["pair"].unique()[4]) | (frame.index % 7 == 0)
    fit_rows = ~held

    prior = geometric_prior(frame, y, fit_rows)
    from_fold, _ = encode_level(frame, y, prior.values, fit_rows, "pair")
    from_lookup = fit_encoding(frame, y, fit_rows, prior=prior).transform(frame)

    assert np.allclose(
        from_fold[held], from_lookup.loc[held, ENCODED_COLUMN["pair"]], atol=1e-4
    )
