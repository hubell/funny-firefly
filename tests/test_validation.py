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
