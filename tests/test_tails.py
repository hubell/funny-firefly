# SPDX-License-Identifier: GPL-3.0-only
"""The record-quality term: flags, cells, the mixture, and what it may read."""

from __future__ import annotations

import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.challenge import (  # noqa: E402
    EXCLUDED_BY_DECISION,
    FORBIDDEN_FEATURE_INPUTS,
    HIDDEN_FOR_TEST_DEPARTURES,
    MOVEMENTS_GLOB,
    RANKING,
)
from taxiout.tails import (  # noqa: E402
    BUCKETS,
    FIT_ONLY_INPUTS,
    FLAG_INPUTS,
    PLAN_SHIFT_ADD_ONE,
    PLAN_SHIFT_FIT_ONLY_INPUTS,
    PLAN_SHIFT_PREDICT_INPUTS,
    PREDICT_INPUTS,
    PlanShiftModel,
    TailModel,
    combine,
    defaulted_block,
    flag,
    lag_band,
    plan_shift_threshold_s,
    plan_shifted,
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


def _submission_script():
    """``scripts/make_submission.py`` as a module; its ``main`` is not run."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_submission.py"
    spec = importlib.util.spec_from_file_location("make_submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# With ``substitute`` the file is what built v7, ``combine``; without it, which
# is ``taxiout.models.TAIL_RULES_SUBSTITUTED`` since
# reports/preregistered-2026-10-04.md, the physical prediction on every row.
SUBSTITUTE = pytest.mark.parametrize("substitute", [True, False], ids=["v7", "rules-off"])


@SUBSTITUTE
def test_combine_prefers_the_term_where_present(substitute):
    submitted = _submission_script().submitted
    idx = pd.Index([10, 11, 12])
    phys = pd.Series([900.0, 950.0, 1000.0], index=idx)
    tail = pd.Series([np.nan, 20000.0, np.nan], index=idx)
    out = combine(phys, tail)
    assert out.tolist() == [900.0, 20000.0, 1000.0]
    assert out.index.equals(idx)
    assert combine(np.array([1.0, 2.0, 3.0]), tail).tolist() == [1.0, 20000.0, 3.0]
    # what the file is written from, under the flag
    written = submitted(phys, tail, None, substitute)
    assert written.index.equals(idx)
    if substitute:
        pd.testing.assert_series_equal(written, out)
    else:
        assert written.tolist() == phys.tolist()
    arr = submitted(np.array([1.0, 2.0, 3.0]), tail, None, substitute)
    assert arr.tolist() == ([1.0, 20000.0, 3.0] if substitute else [1.0, 2.0, 3.0])


@SUBSTITUTE
def test_combine_rejects_misaligned_inputs(substitute):
    submitted = _submission_script().submitted
    tail = pd.Series([np.nan, 1.0], index=[0, 1])
    with pytest.raises(ValueError):
        combine(pd.Series([1.0, 2.0], index=[5, 6]), tail)
    with pytest.raises(ValueError):
        combine(np.array([1.0]), tail)
    # and the file refuses them under either setting
    with pytest.raises(ValueError):
        submitted(pd.Series([1.0, 2.0], index=[5, 6]), tail, None, substitute)
    with pytest.raises(ValueError):
        submitted(np.array([1.0]), tail, None, substitute)
    with pytest.raises(ValueError):
        submitted(
            np.array([1.0, 2.0]), tail, pd.Series([np.nan] * 2, index=[7, 8]), substitute
        )


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


def test_the_submission_averages_the_five_seeds_the_rule_adopted():
    """The decision the freeze records: the constant is True, and the script
    fits the five-seed average when it is.

    ``reports/baseline.md``'s *Seed averaging* computes *Adopted* under the
    parameters that ship, and the paper describes the submitted model as the
    five-seed average. ``make_submission`` reads the constant, so the constant
    has to say the same. The script is read structurally, because its ``main``
    needs the challenge data and this runs without it.
    """
    import ast
    import pathlib

    from taxiout.models import BAGGING_SEEDS, SEED_AVERAGE_ADOPTED, SeedAverage

    assert SEED_AVERAGE_ADOPTED is True
    assert BAGGING_SEEDS == 5
    assert SeedAverage(("x",)).seeds == BAGGING_SEEDS

    root = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts" / "make_submission.py").read_text())
    decides = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "SEED_AVERAGE_ADOPTED"
    ]
    assert len(decides) == 1, "one place decides how the submission is fitted"
    taken = ast.unparse(ast.Module(body=decides[0].body, type_ignores=[]))
    assert "SeedAverage(RUNG.features)" in taken
    assert "BAGGING_SEEDS" in taken
    # and the manifest records the seeds the file was fitted under
    seeds = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "seeds"
    ]
    assert [ast.unparse(v) for v in seeds] == [
        "BAGGING_SEEDS if SEED_AVERAGE_ADOPTED else 1"
    ]


# --------------------------------------------------------------------------
# the plan-shifted departures, as reports/preregistered-2026-10-02.md fixes them


def _plan_frame(rows):
    """Synthetic departures for the plan-shifted class.

    Each row is ``(icao, lag_s, eobt_s, lobt_s, taxi_s)``: the take-off is
    ``lag_s`` after the schedule, ``eobt_s`` after ``EOBT_1_flt`` and
    ``lobt_s`` after ``LOBT_flt``, ``None`` for an estimate that is absent, and
    the block time is the take-off minus ``taxi_s``, as in the data.
    ``AOBT_3_flt`` sits a minute after the block time, where the Network
    Manager's record would put it.
    """
    recs = []
    for i, (icao, lag_s, eobt_s, lobt_s, taxi) in enumerate(rows):
        sched = T0 + pd.Timedelta(minutes=7 * i)
        mvt = sched + pd.Timedelta(seconds=lag_s)
        block = mvt - pd.Timedelta(seconds=taxi)
        recs.append(
            {
                "ADEP_mvt": icao,
                "MVT_TIME_UTC_mvt": mvt,
                "SCHED_TIME_UTC_mvt": sched,
                "BLOCK_TIME_UTC_mvt": block,
                "EOBT_1_flt": pd.NaT
                if eobt_s is None
                else mvt - pd.Timedelta(seconds=eobt_s),
                "LOBT_flt": pd.NaT
                if lobt_s is None
                else mvt - pd.Timedelta(seconds=lobt_s),
                "AOBT_3_flt": block + pd.Timedelta(minutes=1),
                "TAXITIME_SEC_mvt": np.int32(taxi),
            }
        )
    return pd.DataFrame(recs)


def _plan_training():
    """A year with two class departures, one long, and a covered cell.

    AAAA carries ordinary departures with a record, one without, and the two
    class departures: plan lags of five and twenty-four hours, the first
    recording an ordinary taxi-out and the second a long one. CCCC carries a
    cell the term covers whose rows are plan-shifted too, five defaulted block
    times and one ordinary taxi-out in the same cell with a record, and two
    ordinary departures with a record outside it.
    """
    h = plan_shift_threshold_s()
    rows = [("AAAA", 600, 900, 800, 600)] * 4
    rows += [("AAAA", 700, None, None, 2000)]  # no record: not in the rest mean
    rows += [("AAAA", 600, h + 3600, h + 3600, 1000)]  # class, ordinary
    rows += [("AAAA", 24 * HOUR, 24 * HOUR, 24 * HOUR, 24 * HOUR - 600)]  # class, long
    rows += [("CCCC", 5 * HOUR, 5 * HOUR, 5 * HOUR, 5 * HOUR)] * 5  # covered, shifted
    rows += [("CCCC", 5 * HOUR, 5 * HOUR, 5 * HOUR, 900)]  # same cell, ordinary
    rows += [("CCCC", 600, 900, 800, 700)] * 2
    return _plan_frame(rows)


def _plan_model(train=None):
    train = _plan_training() if train is None else train
    term = TailModel().fit(train)
    return term, PlanShiftModel(term).fit(train)


def test_plan_shift_inputs_are_the_pre_registration_s():
    assert PlanShiftModel.PREDICT_INPUTS == PLAN_SHIFT_PREDICT_INPUTS
    assert PlanShiftModel.FIT_ONLY_INPUTS == PLAN_SHIFT_FIT_ONLY_INPUTS
    assert set(PLAN_SHIFT_PREDICT_INPUTS) == {
        "MVT_TIME_UTC_mvt",
        "EOBT_1_flt",
        "LOBT_flt",
        "ADEP_mvt",
    }
    assert PLAN_SHIFT_FIT_ONLY_INPUTS == ("TAXITIME_SEC_mvt",)
    assert not set(PLAN_SHIFT_PREDICT_INPUTS) & set(FORBIDDEN_FEATURE_INPUTS)
    assert set(PLAN_SHIFT_FIT_ONLY_INPUTS) <= set(HIDDEN_FOR_TEST_DEPARTURES)
    assert not set(PLAN_SHIFT_FIT_ONLY_INPUTS) & set(EXCLUDED_BY_DECISION)


def test_the_threshold_is_read_from_models_and_not_copied(monkeypatch):
    import taxiout.models as models
    import taxiout.tails as tails

    assert tails.TAIL_EXCLUSION_ADOPTED_H is models.TAIL_EXCLUSION_ADOPTED_H
    assert plan_shift_threshold_s() == models.TAIL_EXCLUSION_ADOPTED_H * 3600
    # read at the call and not fixed in the function: another value moves it
    monkeypatch.setattr(tails, "TAIL_EXCLUSION_ADOPTED_H", 5.0)
    assert plan_shift_threshold_s() == 5 * 3600
    # and with no adopted exclusion there is no class, rather than a guessed one
    monkeypatch.setattr(tails, "TAIL_EXCLUSION_ADOPTED_H", None)
    with pytest.raises(RuntimeError):
        plan_shift_threshold_s()


@pytest.mark.parametrize(
    ("eobt_past", "lobt_past", "picked"),
    [
        (1, 1, True),
        (0, 1, False),  # exactly the threshold is not past it
        (1, 0, False),
        (-600, 3600, False),
        (3600, -600, False),
        (None, 3600, False),  # an estimate absent
        (3600, None, False),
        (None, None, False),  # no record at all
    ],
)
def test_the_class_needs_both_lags_past_the_threshold_and_both_estimates(
    eobt_past, lobt_past, picked
):
    h = plan_shift_threshold_s()
    eobt = None if eobt_past is None else h + eobt_past
    lobt = None if lobt_past is None else h + lobt_past
    _, model = _plan_model()
    row = _plan_frame([("AAAA", 600, eobt, lobt, 900)])
    assert plan_shifted(row).tolist() == [picked]
    assert model.in_class(row).tolist() == [picked]
    assert model.predict(row).notna().tolist() == [picked]


def test_a_row_the_term_covers_is_never_in_the_class():
    train = _plan_training()
    term, model = _plan_model(train)
    both = plan_shifted(train) & term.flagged(train)
    # the six rows of CCCC's covered cell, every one of them plan-shifted too
    assert train.loc[both, "ADEP_mvt"].tolist() == ["CCCC"] * 6
    assert not model.in_class(train)[both].any()
    # the fit counts the two class departures and none of the term's six
    assert model.n == 2 and model.long == 1
    # and where the rule has a value on a covered row, the term's is submitted
    phys = pd.Series(1.0, index=train.index)
    tail, plan = term.predict(train), model.predict(train)
    assert plan[both].notna().all()
    out = combine(phys, tail, plan)
    assert out[both].tolist() == tail[both].tolist()
    assert out[model.in_class(train)].tolist() == plan[model.in_class(train)].tolist()


@pytest.mark.parametrize(("n", "long"), [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)])
def test_the_share_is_the_add_one_estimate(n, long):
    h = plan_shift_threshold_s()
    rows = [("AAAA", 600, 900, 800, 600)] * 3
    rows += [("AAAA", 600, h + 1, h + 1, h + 1)] * long  # long: over the threshold
    rows += [("AAAA", 600, h + 1, h + 1, h)] * (n - long)  # exactly it: not long
    _, model = _plan_model(_plan_frame(rows))
    assert PLAN_SHIFT_ADD_ONE == (1, 2)
    assert (model.n, model.long) == (n, long)
    assert model.share == pytest.approx((long + 1) / (n + 2))


def test_the_prediction_is_the_stated_mixture():
    h = plan_shift_threshold_s()
    # A third class departure, long, so the share is the training year's 3 / 5
    # and not 1 / 2, where the mixture and its mirror image coincide; and an
    # ordinary departure with a record whose taxi-out is exactly the threshold,
    # which the rest mean holds, "four hours or less".
    train = pd.concat(
        [
            _plan_training(),
            _plan_frame(
                [
                    ("AAAA", 600, 30 * HOUR, 30 * HOUR, 30 * HOUR - 600),
                    ("AAAA", 600, 900, 800, h),
                ]
            ),
        ],
        ignore_index=True,
    )
    _, model = _plan_model(train)
    assert (model.n, model.long, model.share) == (3, 2, pytest.approx(0.6))
    # The rest mean: every departure with a record and a taxi-out of four hours
    # or less, the class's ordinary one and the term's ordinary one included.
    assert model.rest_mean["AAAA"] == pytest.approx((4 * 600 + 1000 + h) / 6)
    assert model.rest_mean["CCCC"] == pytest.approx((900 + 2 * 700) / 3)
    share = 0.6
    lag = 24 * HOUR + 3
    test = _plan_frame(
        [
            ("AAAA", 26 * HOUR, 23 * HOUR, lag, 0),
            ("CCCC", 600, h + 60, h + 60, 0),
            ("AAAA", 600, 900, 800, 0),
        ]
    )
    p = model.predict(test)
    assert p.iloc[0] == pytest.approx(0.6 * lag + 0.4 * (4 * 600 + 1000 + h) / 6)
    assert p.iloc[0] == pytest.approx(share * lag + (1 - share) * model.rest_mean["AAAA"])
    assert p.iloc[1] == pytest.approx(
        share * (h + 60) + (1 - share) * model.rest_mean["CCCC"]
    )
    assert np.isnan(p.iloc[2])
    assert p.index.equals(test.index)


def test_hidden_fields_set_to_nonsense_move_no_prediction():
    train = _plan_training()
    term, model = _plan_model(train)
    test = _plan_frame(
        [
            ("AAAA", 26 * HOUR, 23 * HOUR, 24 * HOUR, 900),
            ("CCCC", 5 * HOUR, 5 * HOUR, 5 * HOUR, 900),
            ("AAAA", 600, 900, 800, 900),
        ]
    )
    nonsense = test.copy()
    for c in HIDDEN_FOR_TEST_DEPARTURES + EXCLUDED_BY_DECISION:
        if c == "TAXITIME_SEC_mvt":
            nonsense[c] = np.int32(-987654)
        else:
            nonsense[c] = pd.Timestamp("1970-01-01T00:00:00Z")
    assert set(HIDDEN_FOR_TEST_DEPARTURES + EXCLUDED_BY_DECISION) <= set(test.columns)
    want = model.predict(test)
    pd.testing.assert_series_equal(model.predict(nonsense), want)
    pd.testing.assert_series_equal(
        model.predict(nonsense[list(PLAN_SHIFT_PREDICT_INPUTS)]), want
    )
    phys = pd.Series(1.0, index=test.index)
    pd.testing.assert_series_equal(
        combine(phys, term.predict(nonsense), model.predict(nonsense)),
        combine(phys, term.predict(test), want),
    )
    # and the fit never reads AOBT_3_flt either
    blanked = train.copy()
    blanked["AOBT_3_flt"] = pd.Timestamp("1970-01-01T00:00:00Z")
    again = PlanShiftModel(term).fit(blanked)
    assert (again.n, again.long, again.share) == (model.n, model.long, model.share)
    pd.testing.assert_series_equal(again.rest_mean, model.rest_mean)


def test_an_empty_class_predicts_nothing():
    """Departures none of which is in the class: no prediction, and the
    submission is the term's and the model's as it was without the rule."""
    _, model = _plan_model()
    test = _plan_frame([("AAAA", 600, 900, 800, 0), ("BBBB", 5 * HOUR, None, None, 0)])
    term = model.term
    plan = model.predict(test)
    assert not model.in_class(test).any() and plan.isna().all()
    phys = pd.Series([900.0, 1000.0], index=test.index)
    tail = term.predict(test)
    pd.testing.assert_series_equal(combine(phys, tail, plan), combine(phys, tail))


def test_a_fit_with_no_class_still_has_the_add_one_share():
    """The other reading of an empty class, stated rather than assumed: a fit
    given no class departure has n = 0 and the add-one share (0 + 1) / (0 + 2),
    and the rule as written still predicts for a class row it is shown."""
    h = plan_shift_threshold_s()
    rows = [("AAAA", 600, 900, 800, 600)] * 3 + [("BBBB", 700, None, None, 800)]
    _, model = _plan_model(_plan_frame(rows))
    assert (model.n, model.long, model.share) == (0, 0, 0.5)
    test = _plan_frame([("AAAA", 600, h + 60, h + 60, 0)])
    assert model.predict(test).iloc[0] == pytest.approx(0.5 * (h + 60) + 0.5 * 600)


def test_a_class_row_with_no_rest_mean_is_refused():
    """The pre-registration does not say what the rule predicts at an airport
    its fit holds no rest mean for, so the rule refuses rather than choose.
    An ordinary row at such an airport is simply not the rule's."""
    h = plan_shift_threshold_s()
    _, model = _plan_model()
    assert "ZZZZ" not in model.rest_mean.index
    assert model.predict(_plan_frame([("ZZZZ", 600, 900, 800, 0)])).isna().all()
    with pytest.raises(ValueError):
        model.predict(_plan_frame([("ZZZZ", 600, h + 60, h + 60, 0)]))


def test_plan_shift_predict_reads_only_its_inputs_and_needs_a_fit():
    term, model = _plan_model()
    test = _plan_frame([("AAAA", 26 * HOUR, 23 * HOUR, 24 * HOUR, 0)])
    assert model.predict(test[list(PLAN_SHIFT_PREDICT_INPUTS)]).notna().all()
    with pytest.raises(KeyError):
        model.predict(test.drop(columns=["EOBT_1_flt"]))
    with pytest.raises(KeyError):
        PlanShiftModel(term).fit(test.drop(columns=["TAXITIME_SEC_mvt"]))
    with pytest.raises(RuntimeError):
        PlanShiftModel(term).predict(test)


@SUBSTITUTE
def test_combine_puts_the_term_first_then_the_class_then_the_model(substitute):
    submitted = _submission_script().submitted
    idx = pd.Index([10, 11, 12, 13])
    phys = pd.Series([900.0, 950.0, 1000.0, 1100.0], index=idx)
    tail = pd.Series([np.nan, 20000.0, np.nan, 30000.0], index=idx)
    plan = pd.Series([np.nan, 5000.0, 52000.0, np.nan], index=idx)
    out = combine(phys, tail, plan)
    assert out.tolist() == [900.0, 20000.0, 52000.0, 30000.0]
    assert out.index.equals(idx)
    assert combine(phys.to_numpy(), tail, plan).tolist() == out.tolist()
    with pytest.raises(ValueError):
        combine(phys, tail, pd.Series([np.nan] * 4, index=[0, 1, 2, 3]))
    # what the file is written from: v7's order, or the physical model alone,
    # equal on every row neither rule covers either way
    written = submitted(phys, tail, plan, substitute)
    assert written.index.equals(idx)
    if substitute:
        pd.testing.assert_series_equal(written, out)
    else:
        assert written.tolist() == phys.tolist()
    untouched = (tail.isna() & plan.isna()).to_numpy()
    assert written[untouched].tolist() == phys[untouched].tolist()
    assert submitted(phys.to_numpy(), tail, plan, substitute).tolist() == written.tolist()


def test_the_class_on_the_training_year_and_the_ranking_set():
    """The counts the pre-registration was written knowing: three training
    departures, two of them long, and one in the ranking set, at LIRF in July
    2026. Its first two stops, held here so a change to the data or to the rule
    that moves them fails rather than ships."""
    import pathlib

    from taxiout.challenge import load_movements, load_ranking, to_pandas

    data = pathlib.Path(__file__).resolve().parents[1] / "data" / "challenge"
    if not (data / RANKING).exists() or not any(data.glob(MOVEMENTS_GLOB)):
        pytest.skip(
            "no challenge data in this checkout: data/ is gitignored, so a clean"
            " checkout has none"
        )
    plan = sorted(set(PREDICT_INPUTS) | set(PLAN_SHIFT_PREDICT_INPUTS))
    train = to_pandas(
        load_movements(data, sorted(set(plan) | set(FIT_ONLY_INPUTS)), phase="DEP")
    ).reset_index(drop=True)
    rank = to_pandas(load_ranking(data, plan, phase="DEP")).reset_index(drop=True)
    term = TailModel().fit(train)
    model = PlanShiftModel(term).fit(train)
    assert (model.n, model.long) == (3, 2)
    assert model.share == pytest.approx(3 / 5)
    members = model.in_class(rank)
    assert int(members.sum()) == 1
    assert rank.loc[members, "ADEP_mvt"].astype(str).tolist() == ["LIRF"]
    month = pd.to_datetime(rank.loc[members, "MVT_TIME_UTC_mvt"], utc=True)
    assert month.dt.strftime("%Y-%m").tolist() == ["2026-07"]
    # what the submission takes the rule on is exactly the class
    taken = model.predict(rank).notna() & term.predict(rank).isna()
    assert (taken.to_numpy() == members.to_numpy()).all()


# --------------------------------------------------------------------------
# the published copy of the note the rule was fixed in

# The working note is not published; reports/preregistered-2026-10-02.md is
# the note with the places its closing section lists changed and nothing else.
# Each substitution is held by a pattern for the note's side, so this file
# names no work block either, and the copy's words for the other side.
PUBLISHED_COPY = "reports/preregistered-2026-10-02.md"
PUBLISHED_TAIL = "\n## What differs from the note\n"
# The working notes are on hand where the environment variable names their
# directory; there is no default, and without it the copy is checked alone.
WORKING_NOTE = os.path.join(
    os.environ.get("PREREGISTRATION_NOTES", ""), "preregistration_2026-10-02_dayshift.md"
)
WORKING_NOTE_SHA256 = "d62b48f32d3ba9da07aa346cb04a2560feaff9664a9734228d7523e669dda9a1"
# A numbered work block, with or without a letter after its number; the
# pattern of 20 September stops at the number and misses the lettered ones.
WORK_BLOCK_TOKEN = r"\bBlock\s+\d+[A-Za-z]*\b"
COPY_SUBSTITUTIONS = (
    (
        WORK_BLOCK_TOKEN + r" copies this file unchanged into\n"
        r"`reports/preregistered-2026-10-02\.md` and freezes it under its digest; the\n"
        r"digest recorded in the cha[t] when this file was written is the proof of the\n"
        r"order\.",
        "The working note this is copied from was digested when it\nwas written, sha256\n"
        f"`{WORKING_NOTE_SHA256}`, before the\n"
        "rule was evaluated on any fold; that digest is the proof of the order. This\n"
        "copy differs from the note only in the places listed at its end.",
    ),
    (r"after reading " + WORK_BLOCK_TOKEN, "after reading the held-out-month ladder"),
    (
        r"the twelve held-out months of " + WORK_BLOCK_TOKEN,
        "the twelve held-out months of the held-out-month ladder",
    ),
)


def _published_copy() -> str:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    return (root / PUBLISHED_COPY).read_text(encoding="utf-8")


def test_the_note_with_the_listed_substitutions_is_the_published_copy():
    """Where the note is on hand: its digest first, then each listed
    substitution applied once, and the result is the published copy up to its
    closing section byte for byte, which is all the copy adds after it."""
    import hashlib
    import pathlib
    import re

    note = pathlib.Path(WORKING_NOTE)
    if not os.environ.get("PREREGISTRATION_NOTES") or not note.exists():
        pytest.skip(
            "the working note is not on hand; the copy is checked alone"
            " (PREREGISTRATION_NOTES names the notes' directory)"
        )
    raw = note.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == WORKING_NOTE_SHA256
    text = raw.decode("utf-8")
    for pattern, words in COPY_SUBSTITUTIONS:
        text, n = re.subn(pattern, lambda _m, w=words: w, text)
        assert n == 1, pattern
    copy = _published_copy()
    assert copy.count(PUBLISHED_TAIL) == 1
    assert copy[: copy.index(PUBLISHED_TAIL)] == text


def test_the_published_copy_names_no_work_block_and_credits_openstreetmap():
    import re
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_licence_hygiene as hygiene
    import preregistered as pr

    copy = _published_copy()
    assert not re.search(WORK_BLOCK_TOKEN, copy)
    assert not re.search(r"\bcha[t]\b|\bsessio[n]\b", copy, re.IGNORECASE)
    assert not pr.NOTE_REFERENCE_TOKEN.search(copy)
    assert not hygiene.attribution_in(copy)
    assert copy.count(hygiene.ATTRIBUTION) == 1
    # and the note's digest, which fixed the order, is the one it names
    assert set(re.findall(r"`([0-9a-f]{64})`", copy)) == {WORKING_NOTE_SHA256}
