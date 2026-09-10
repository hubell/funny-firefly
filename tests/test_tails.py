# SPDX-License-Identifier: GPL-3.0-only
"""The record-quality term: flags, cells, the mixture, and what it may read."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import (  # noqa: E402
    FORBIDDEN_FEATURE_INPUTS,
    HIDDEN_FOR_TEST_DEPARTURES,
)
from taxiout.tails import (  # noqa: E402
    BUCKETS,
    FIT_ONLY_INPUTS,
    FLAG_INPUTS,
    PREDICT_INPUTS,
    TailModel,
    combine,
    defaulted_block,
    flag,
    lag_band,
)

HOUR = 3600
T0 = pd.Timestamp("2025-03-01T10:00:00Z")


def _frame(rows):
    """Synthetic departures at an invented airport.

    Each row is ``(icao, lag_s, flt_present, block_offset_s, taxi_s)``: the
    take-off is ``lag_s`` after the schedule, the block time is
    ``block_offset_s`` after the schedule, and ``taxi_s`` is the recorded
    target. ``None`` for ``block_offset_s`` derives it from the taxi so that
    ``taxi = take-off - block`` holds as it does in the data.
    """
    recs = []
    for i, (icao, lag_s, flt, block_off, taxi) in enumerate(rows):
        sched = T0 + pd.Timedelta(minutes=7 * i)
        mvt = sched + pd.Timedelta(seconds=lag_s)
        if block_off is None:
            block = mvt - pd.Timedelta(seconds=taxi)
        else:
            block = sched + pd.Timedelta(seconds=block_off)
        recs.append(
            {
                "ADEP_mvt": icao,
                "MVT_TIME_UTC_mvt": mvt,
                "SCHED_TIME_UTC_mvt": sched,
                "BLOCK_TIME_UTC_mvt": block,
                "LOBT_flt": sched if flt else pd.NaT,
                "TAXITIME_SEC_mvt": np.int32(taxi),
            }
        )
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# what may be read


def test_predict_inputs_are_exposed_fields():
    assert not set(PREDICT_INPUTS) & set(FORBIDDEN_FEATURE_INPUTS)
    assert set(FLAG_INPUTS) <= set(PREDICT_INPUTS)


def test_fit_only_inputs_are_exactly_the_hidden_fields():
    assert set(FIT_ONLY_INPUTS) == set(HIDDEN_FOR_TEST_DEPARTURES)


def test_flag_reads_only_its_inputs():
    f = _frame([("ZZZZ", 600, True, None, 900)])
    out = flag(f[list(FLAG_INPUTS)])
    assert list(out.columns) == ["flt_missing", "lag_s", "tail_bucket"]
    assert out.index.equals(f.index)


def test_predict_runs_without_hidden_columns():
    train = _frame([("ZZZZ", 5 * HOUR, False, 0, 5 * HOUR)] * 5)
    model = TailModel().fit(train)
    ranking = train[list(PREDICT_INPUTS)]
    assert model.predict(ranking).notna().all()
    with pytest.raises(KeyError):
        model.predict(ranking.drop(columns=["ADEP_mvt"]))
    with pytest.raises(KeyError):
        TailModel().fit(ranking)


# --------------------------------------------------------------------------
# flags and buckets


@pytest.mark.parametrize(
    ("lag_s", "band"),
    [
        (-HOUR - 1, "<-1h"),
        (-HOUR, "normal"),  # -1 h is inside the normal band
        (0, "normal"),
        (4 * HOUR, "normal"),  # 4 h is inside the normal band
        (4 * HOUR + 1, "4-12h"),
        (12 * HOUR, "4-12h"),  # 12 h is inside 4-12 h
        (12 * HOUR + 1, ">12h"),
        (np.nan, "normal"),
    ],
)
def test_lag_band_edges_match_the_data_review(lag_s, band):
    assert lag_band(pd.Series([lag_s])).iloc[0] == band


def test_flag_combines_record_and_band():
    f = _frame(
        [
            ("ZZZZ", 600, True, None, 900),
            ("ZZZZ", 600, False, None, 900),
            ("ZZZZ", 5 * HOUR, False, None, 900),
            ("ZZZZ", 13 * HOUR, True, None, 900),
            ("ZZZZ", -2 * HOUR, False, None, 900),
        ]
    )
    out = flag(f)
    assert list(out["tail_bucket"].astype(str)) == [
        "flt:normal",
        "noflt:normal",
        "noflt:4-12h",
        "flt:>12h",
        "noflt:<-1h",
    ]
    assert list(out["tail_bucket"].cat.categories) == list(BUCKETS)
    assert out["flt_missing"].tolist() == [False, True, True, False, True]
    assert out["lag_s"].tolist() == [600, 600, 5 * HOUR, 13 * HOUR, -2 * HOUR]


def test_flag_accepts_naive_and_aware_timestamps():
    f = _frame([("ZZZZ", 600, True, None, 900)])
    naive = f.copy()
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt"):
        naive[c] = naive[c].dt.tz_localize(None)
    assert flag(naive)["lag_s"].iloc[0] == 600


def test_defaulted_block_needs_near_and_long():
    f = _frame(
        [
            ("ZZZZ", 5 * HOUR, False, 3, 5 * HOUR - 3),  # block 3 s after schedule
            ("ZZZZ", 5 * HOUR, False, 61, 5 * HOUR - 61),  # 61 s: not near
            ("ZZZZ", 600, False, 0, 600),  # near, but not a long lag
            ("ZZZZ", 5 * HOUR, False, None, 900),  # an ordinary taxi, late
        ]
    )
    assert defaulted_block(f).tolist() == [True, False, False, False]


# --------------------------------------------------------------------------
# the model


def _training():
    rows = []
    # Airport A: a cell where 3 of 4 rows are defaulted block times, and a
    # clean cell of ordinary rows.
    rows += [("AAAA", 5 * HOUR, False, 0, 5 * HOUR)] * 3
    rows += [("AAAA", 5 * HOUR, False, None, 1000)]
    rows += [("AAAA", 600, True, None, 900)] * 10
    # Airport B: the same bucket, every row ordinary.
    rows += [("BBBB", 5 * HOUR, False, None, 800)] * 6
    # Airport C: a thin cell, entirely defaulted.
    rows += [("CCCC", 13 * HOUR, False, 0, 13 * HOUR)] * 2
    return _frame(rows)


def test_fit_measures_every_cell():
    m = TailModel(min_count=4, min_share=0.05).fit(_training())
    cells = m.cells
    a = cells.loc[("AAAA", "noflt:4-12h")]
    assert a["n"] == 4
    assert a["share"] == pytest.approx(0.75)
    assert a["mean"] == pytest.approx((3 * 5 * HOUR + 1000) / 4)
    assert a["rest_mean"] == pytest.approx(1000)
    assert bool(a["covered"])
    b = cells.loc[("BBBB", "noflt:4-12h")]
    assert b["share"] == 0 and not bool(b["covered"])
    c = cells.loc[("CCCC", "noflt:>12h")]
    assert c["share"] == 1 and np.isnan(c["rest_mean"]) and not bool(c["covered"])
    assert not bool(cells.loc[("AAAA", "flt:normal"), "covered"])
    assert list(m.covered.index) == [("AAAA", "noflt:4-12h")]


def test_mixture_prediction_uses_the_row_lag():
    m = TailModel(min_count=4).fit(_training())
    test = _frame(
        [
            ("AAAA", 6 * HOUR, False, None, 0),  # covered cell, lag 6 h
            ("AAAA", 600, True, None, 0),  # clean cell
            ("BBBB", 6 * HOUR, False, None, 0),  # same bucket, clean airport
            ("DDDD", 6 * HOUR, False, None, 0),  # airport never seen
        ]
    )
    p = m.predict(test)
    assert p.iloc[0] == pytest.approx(0.75 * 6 * HOUR + 0.25 * 1000)
    assert p.iloc[1:].isna().all()
    assert m.flagged(test).tolist() == [True, False, False, False]
    assert p.index.equals(test.index)


def test_mean_kind_predicts_the_cell_mean():
    m = TailModel(kind="mean", min_count=4).fit(_training())
    test = _frame([("AAAA", 6 * HOUR, False, None, 0)])
    assert m.predict(test).iloc[0] == pytest.approx((3 * 5 * HOUR + 1000) / 4)


def test_fully_defaulted_cell_predicts_the_lag():
    m = TailModel(min_count=2).fit(_training())
    test = _frame([("CCCC", 14 * HOUR, False, None, 0)])
    assert m.predict(test).iloc[0] == pytest.approx(14 * HOUR)


def test_min_count_and_min_share_gate_coverage():
    train = _training()
    assert TailModel(min_count=5).fit(train).covered.empty
    assert TailModel(min_count=4, min_share=0.8).fit(train).covered.empty
    assert len(TailModel(min_count=4, min_share=0.75).fit(train).covered) == 1


def test_no_pooled_fallback_across_airports():
    """A clean airport's thin cell stays unpredicted, whatever another airport shows."""
    m = TailModel(min_count=1).fit(_training())
    test = _frame(
        [("BBBB", 6 * HOUR, False, None, 0), ("EEEE", 6 * HOUR, False, None, 0)]
    )
    assert m.predict(test).isna().all()


def test_unfitted_and_bad_kind():
    with pytest.raises(RuntimeError):
        TailModel().predict(_frame([("AAAA", 600, True, None, 900)]))
    with pytest.raises(ValueError):
        TailModel(kind="median")


def test_nan_target_rows_are_ignored_in_fit():
    train = _training()
    train.loc[train.index[:2], "TAXITIME_SEC_mvt"] = np.nan
    m = TailModel(min_count=1).fit(train)
    assert m.cells.loc[("AAAA", "noflt:4-12h"), "n"] == 2


# --------------------------------------------------------------------------
# combine


def test_combine_prefers_the_term_where_present():
    idx = pd.Index([10, 11, 12])
    phys = pd.Series([900.0, 950.0, 1000.0], index=idx)
    tail = pd.Series([np.nan, 20000.0, np.nan], index=idx)
    out = combine(phys, tail)
    assert out.tolist() == [900.0, 20000.0, 1000.0]
    assert out.index.equals(idx)
    assert combine(np.array([1.0, 2.0, 3.0]), tail).tolist() == [1.0, 20000.0, 3.0]


def test_combine_rejects_misaligned_inputs():
    tail = pd.Series([np.nan, 1.0], index=[0, 1])
    with pytest.raises(ValueError):
        combine(pd.Series([1.0, 2.0], index=[5, 6]), tail)
    with pytest.raises(ValueError):
        combine(np.array([1.0]), tail)


# --------------------------------------------------------------------------
# the recorded-taxi tail, and the one place the rule lives


def test_the_fit_row_rule_drops_the_recorded_taxi_tail():
    from taxiout.models import TAIL_EXCLUSION_ADOPTED_H, fit_rows

    y = pd.Series([600.0, 3600.0, 14400.0, 87341.0, float("nan")])
    train = pd.Series([True] * 5)
    flagged = pd.Series([False] * 5)
    kept = fit_rows(train, flagged, y)
    if TAIL_EXCLUSION_ADOPTED_H is None:
        assert kept.tolist() == [True] * 5
        return
    # the day-shift goes; exactly four hours stays, the comparison being strict
    assert kept.tolist() == [True, True, True, False, True]
    # a missing target is NOT this rule's business: ``NaN > x`` is False, so it
    # passes through, and the caller's own training mask is what excludes it
    # (``scripts/make_submission.py`` passes ``y.notna()``). Asserted so that a
    # future change to make this rule drop nulls is a deliberate one.
    assert bool(kept.iloc[4])


def test_the_fit_row_rule_still_drops_the_flagged_rows():
    from taxiout.models import fit_rows

    y = pd.Series([600.0, 700.0])
    train = pd.Series([True, True])
    assert fit_rows(train, pd.Series([False, True]), y).tolist() == [True, False]
    assert fit_rows(pd.Series([False, True]), pd.Series([False, False]), y).tolist() == [
        False,
        True,
    ]


def test_every_script_that_fits_reads_the_same_fit_row_rule():
    """The ladder, the submission and the transfer report, from one constant.

    They came apart once over the seed count, and the fix was to name the
    constant in one place and import it. A second fit-side rule invented in one
    script would put a different model in the submission than the one
    ``reports/baseline.md`` measures, and nothing would say so.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("run_baselines.py", "make_submission.py", "report_transfer.py"):
        text = (root / "scripts" / name).read_text()
        assert "fit_rows as fitting_rows" in text, name
        # and no script may re-derive the exclusion for itself
        assert "TAIL_EXCLUSION_ADOPTED_H * 3600" not in text, name


def test_the_submission_cannot_seed_average_against_the_rule():
    """The estimator choice is a constant too, for the same reason.

    ``reports/baseline.md`` applies the pre-registered rule and prints whether
    seed averaging was adopted. If ``make_submission`` decided that for itself,
    the shipped model could average five seeds while the report said it had
    been measured and not adopted -- which is what happened to the fit rows.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "make_submission.py").read_text()
    assert "SEED_AVERAGE_ADOPTED" in text
    # the averaging must be reached through the constant, never unconditionally
    for line in text.splitlines():
        if "SeedAverage(" in line and "import" not in line:
            assert "if SEED_AVERAGE_ADOPTED" in text, line
