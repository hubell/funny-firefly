# SPDX-License-Identifier: GPL-3.0-only
"""Splits and metrics.

The properties worth asserting are the ones a split can lose silently: that
the two sides are disjoint, that the forward splits do not train on a month
they validate on, that a held-out pair is held out *entirely* rather than
mostly, and that the seen-pair comparison set is excluded from the fit as
well. A split that leaks is not a wrong number, it is a right-looking one.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.validation import (  # noqa: E402
    DECEMBER_TRAIN,
    DECEMBER_VALID,
    FREQUENCY_BUCKETS,
    JULY_TRAIN,
    JULY_VALID,
    PAIR_FOLDS,
    Split,
    assign_pair_folds,
    december_split,
    evaluate,
    evaluate_by_bucket,
    frequency_bucket,
    held_out_pair_splits,
    july_split,
    month_of,
    rmse,
)


def _year(per_month: int = 4) -> pd.Series:
    stamps = []
    for m in range(1, 13):
        for d in range(per_month):
            stamps.append(f"2025-{m:02d}-{d + 1:02d}T09:00:00Z")
    return pd.Series(pd.to_datetime(stamps, utc=True))


def _pairs(n_airports: int = 3, n_pairs: int = 9, rows: int = 5):
    icao, pair = [], []
    for a in range(n_airports):
        for p in range(n_pairs):
            for _ in range(rows):
                icao.append(f"AP{a:02d}")
                pair.append(f"AP{a:02d}|S{p}|27")
    return pd.Series(icao), pd.Series(pair)


# --------------------------------------------------------------------------
# forward splits


def test_month_of_is_utc_year_month():
    t = pd.Series(pd.to_datetime(["2025-12-31T23:30:00Z"], utc=True))
    assert month_of(t).tolist() == ["2025-12"]


def test_december_split_trains_before_it_validates():
    month = month_of(_year())
    s = december_split(month)
    assert set(month[s.train]) == set(DECEMBER_TRAIN)
    assert set(month[s.valid]) == set(DECEMBER_VALID)
    assert not (s.train & s.valid).any()
    assert s.sizes["train"] + s.sizes["valid"] == len(month)


def test_july_split_stops_at_june():
    month = month_of(_year())
    s = july_split(month)
    assert set(month[s.train]) == set(JULY_TRAIN)
    assert set(month[s.valid]) == set(JULY_VALID)
    # August onwards is in neither side: it is after the month being validated
    assert not month[~(s.train | s.valid)].isin(JULY_TRAIN + JULY_VALID).any()


def test_a_split_that_overlaps_is_refused():
    mask = pd.Series([True, True, False])
    with pytest.raises(ValueError):
        Split(name="bad", description="", train=mask, valid=mask)


def test_extra_sets_may_not_overlap_the_fit():
    train = pd.Series([True, True, False])
    valid = pd.Series([False, False, True])
    with pytest.raises(ValueError):
        Split(name="bad", description="", train=train, valid=valid, extra={"seen": train})


# --------------------------------------------------------------------------
# held-out pairs


def test_a_pair_lands_in_exactly_one_fold():
    icao, pair = _pairs()
    fold = assign_pair_folds(icao, pair)
    per_pair = pd.DataFrame({"pair": pair, "fold": fold}).groupby("pair")["fold"]
    assert (per_pair.nunique() == 1).all()


def test_folds_are_dealt_within_each_airport():
    icao, pair = _pairs(n_airports=3, n_pairs=9)
    fold = assign_pair_folds(icao, pair)
    counts = (
        pd.DataFrame({"icao": icao, "pair": pair, "fold": fold})
        .drop_duplicates(["icao", "pair"])
        .groupby(["icao", "fold"])
        .size()
    )
    # nine pairs dealt round-robin into three folds, at every airport
    assert set(counts) == {3}


def test_pair_folds_are_deterministic():
    icao, pair = _pairs()
    assert assign_pair_folds(icao, pair).equals(assign_pair_folds(icao, pair))
    a = assign_pair_folds(icao, pair, seed=1)
    b = assign_pair_folds(icao, pair, seed=2)
    assert not a.equals(b)


def test_held_out_pairs_are_absent_from_training():
    icao, pair = _pairs(n_airports=2, n_pairs=12, rows=20)
    splits = held_out_pair_splits(icao, pair)
    assert len(splits) == PAIR_FOLDS
    covered = pd.Series(False, index=icao.index)
    for s in splits:
        assert not (s.train & s.valid).any()
        assert not (s.train & s.extra["seen"]).any()
        assert not (s.valid & s.extra["seen"]).any()
        held = set(pair[s.valid])
        assert held and not (set(pair[s.train]) & held)
        # the seen set is exactly what its name claims
        assert set(pair[s.extra["seen"]]) <= set(pair[s.train])
        covered |= s.valid
    assert covered.all()


def test_eligible_keeps_rows_out_of_every_side():
    icao, pair = _pairs(n_airports=2, n_pairs=6, rows=10)
    eligible = pd.Series(np.arange(len(icao)) % 2 == 0, index=icao.index)
    for s in held_out_pair_splits(icao, pair, eligible=eligible):
        assert not (s.train & ~eligible).any()
        assert not (s.valid & ~eligible).any()
        assert not (s.extra["seen"] & ~eligible).any()


# --------------------------------------------------------------------------
# metrics


def test_rmse_ignores_rows_it_cannot_score():
    assert rmse([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert rmse([0.0, 10.0], [0.0, 0.0]) == pytest.approx(np.sqrt(50.0))
    assert rmse([1.0, np.nan], [1.0, 5.0]) == pytest.approx(0.0)
    assert np.isnan(rmse([np.nan], [1.0]))


def test_frequency_buckets_sit_on_the_stated_boundaries():
    counts = pd.Series([0, 1, 10, 11, 100, 101, np.nan])
    assert frequency_bucket(counts).tolist() == [
        "0",
        "1-10",
        "1-10",
        "11-100",
        "11-100",
        ">100",
        "0",
    ]
    assert set(FREQUENCY_BUCKETS) == set(frequency_bucket(counts)) | {">100"}


def test_evaluate_reports_pooled_and_per_airport_with_and_without_flags():
    y = pd.Series([100.0, 200.0, 300.0, 90000.0])
    pred = pd.Series([100.0, 200.0, 300.0, 300.0])
    icao = pd.Series(["AAAA", "AAAA", "BBBB", "BBBB"])
    flagged = pd.Series([False, False, False, True])
    out = evaluate(y, pred, icao=icao, flagged=flagged).set_index("icao")
    assert list(out.index) == ["all", "AAAA", "BBBB"]
    assert out.loc["all", "flagged"] == 1
    assert out.loc["all", "rmse_ex_flagged_s"] == pytest.approx(0.0)
    assert out.loc["all", "rmse_s"] > 40000
    assert out.loc["AAAA", "rmse_s"] == pytest.approx(0.0)


def test_evaluate_without_flags_repeats_the_pooled_figure():
    y = pd.Series([100.0, 300.0])
    pred = pd.Series([150.0, 300.0])
    out = evaluate(y, pred, icao=pd.Series(["AAAA", "AAAA"])).set_index("icao")
    assert out.loc["all", "flagged"] == 0
    assert out.loc["all", "rmse_s"] == pytest.approx(out.loc["all", "rmse_ex_flagged_s"])


def test_evaluate_by_bucket_covers_every_bucket():
    y = pd.Series([100.0, 200.0, 300.0, 400.0])
    pred = pd.Series([100.0, 200.0, 300.0, 500.0])
    counts = pd.Series([0, 5, 50, 500])
    out = evaluate_by_bucket(y, pred, counts).set_index("bucket")
    assert list(out.index) == list(FREQUENCY_BUCKETS)
    assert out.loc[">100", "rmse_s"] == pytest.approx(100.0)
    assert out.loc["0", "rmse_s"] == pytest.approx(0.0)
    assert out["rows"].sum() == 4


# --------------------------------------------------------------------------
# the cluster bootstrap


def _sums(resid_by_rung, clusters):
    from taxiout.validation import cluster_sums

    return cluster_sums(resid_by_rung, np.asarray(clusters))


def test_cluster_sums_reduce_rows_to_pairs():
    s = _sums({"a": [1.0, 2.0, 3.0, 4.0]}, ["p", "p", "q", "q"])
    assert s["a"].n.tolist() == [2.0, 2.0]
    assert s["a"].sse.tolist() == [5.0, 25.0]
    assert s["a"].rmse == pytest.approx(np.sqrt(30.0 / 4.0))


def test_cluster_sums_drop_a_row_any_rung_cannot_score():
    s = _sums({"a": [1.0, np.nan, 3.0], "b": [1.0, 1.0, 1.0]}, ["p", "p", "q"])
    # the middle row is gone for both rungs, so they stay on the same rows
    assert s["a"].n.tolist() == [1.0, 1.0]
    assert s["b"].n.tolist() == [1.0, 1.0]
    assert s["b"].sse.tolist() == [1.0, 1.0]


def test_point_estimate_is_the_unresampled_statistic():
    from taxiout.validation import cluster_bootstrap, gap

    held = _sums({"a": [3.0, 3.0, 1.0, 1.0]}, ["p", "p", "q", "q"])
    seen = _sums({"a": [1.0, 1.0]}, ["r", "s"])
    out = cluster_bootstrap([{"held": held, "seen": seen}], gap("a"), draws=50)
    assert out.point == pytest.approx(np.sqrt(5.0) - 1.0)
    assert out.low <= out.point <= out.high
    assert out.draws == 50


def test_a_constant_statistic_has_a_degenerate_interval():
    from taxiout.validation import cluster_bootstrap

    held = _sums({"a": [2.0, 2.0, 2.0, 2.0]}, ["p", "p", "q", "q"])
    seen = _sums({"a": [2.0, 2.0]}, ["r", "s"])
    out = cluster_bootstrap([{"held": held, "seen": seen}], lambda r: 7.0, draws=20)
    assert (out.point, out.low, out.high) == (7.0, 7.0, 7.0)


def test_resampling_is_by_pair_not_by_row():
    """Two pairs of a thousand rows are two observations, not two thousand."""
    from taxiout.validation import cluster_bootstrap, gap

    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(1.0, 0.1, 1000), rng.normal(3.0, 0.1, 1000)])
    two = _sums({"a": values}, ["p"] * 1000 + ["q"] * 1000)
    many = _sums({"a": values}, [f"c{i}" for i in range(2000)])
    seen = _sums({"a": [0.0, 0.0]}, ["r", "s"])
    wide = cluster_bootstrap([{"held": two, "seen": seen}], gap("a"), draws=300)
    narrow = cluster_bootstrap([{"held": many, "seen": seen}], gap("a"), draws=300)
    assert (wide.high - wide.low) > 5 * (narrow.high - narrow.low)


def test_bootstrap_is_deterministic_under_a_seed():
    from taxiout.validation import cluster_bootstrap, gap

    rng = np.random.default_rng(1)
    held = _sums({"a": rng.normal(size=40)}, [f"p{i % 8}" for i in range(40)])
    seen = _sums({"a": rng.normal(size=40)}, [f"s{i % 8}" for i in range(40)])
    fold = {"held": held, "seen": seen}
    a = cluster_bootstrap([fold], gap("a"), draws=100, seed=3)
    b = cluster_bootstrap([fold], gap("a"), draws=100, seed=3)
    c = cluster_bootstrap([fold], gap("a"), draws=100, seed=4)
    assert (a.low, a.high) == (b.low, b.high)
    assert (a.low, a.high) != (c.low, c.high)


def test_difference_in_differences_reads_as_stated():
    """(gap without) - (gap with): positive when the block buys more on new pairs."""
    from taxiout.validation import difference_in_differences

    stat = difference_in_differences("with", "without")
    r = {"held": {"without": 10.0, "with": 6.0}, "seen": {"without": 5.0, "with": 4.0}}
    # held-out delta 4, seen delta 1: the block buys 3 s more where the pair is new
    assert stat(r) == pytest.approx(3.0)
    # and it is invariant to a constant composition shift on both rungs
    r2 = {
        "held": {"without": 110.0, "with": 106.0},
        "seen": {"without": 5.0, "with": 4.0},
    }
    assert stat(r2) == pytest.approx(3.0)


def test_fold_mean_is_the_mean_over_folds():
    from taxiout.validation import cluster_bootstrap, gap

    f1 = {"held": _sums({"a": [2.0]}, ["p"]), "seen": _sums({"a": [1.0]}, ["q"])}
    f2 = {"held": _sums({"a": [4.0]}, ["p"]), "seen": _sums({"a": [1.0]}, ["q"])}
    out = cluster_bootstrap([f1, f2], gap("a"), draws=10)
    assert out.point == pytest.approx((1.0 + 3.0) / 2.0)


def test_empty_folds_give_nan_not_an_error():
    from taxiout.validation import cluster_bootstrap, gap

    out = cluster_bootstrap([], gap("a"), draws=10)
    assert np.isnan(out.point) and out.draws == 0
