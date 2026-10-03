# SPDX-License-Identifier: GPL-3.0-only
"""The reading rules of ``scripts/candidates.py``, on constructed figures.

Nothing here fits a model or reads challenge data. The figures below are
constructed to make each rule's boundary visible, and none of them is a
measurement of anything.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import candidates as cand  # noqa: E402
from taxiout.tails import plan_shift_threshold_s  # noqa: E402

RULE = cand.the_rule()
D = RULE.contrast_decimals
HOUR = 3600
T0 = pd.Timestamp("2025-03-01T10:00:00Z")


# --------------------------------------------------------------------------
# the rule's constants are the published rule's words


def _copy(path: str) -> str:
    return " ".join((REPO / path).read_text(encoding="utf-8").split())


def test_the_floors_are_the_words_of_the_rule():
    text = _copy(cand.RULE_TEXT)
    assert f"December, all ten: better by more than {RULE.floor_december} s" in text
    assert f"July, all ten: better by more than {RULE.floor_july} s" in text
    assert f"July, the three: not worse by more than {RULE.floor_july_three} s" in text
    assert f"October: not worse by more than {RULE.floor_october} s" in text
    assert f"the year's floor, {RULE.floor_year} s" in text
    assert "at the precision `reports/baseline.md` prints" in text


def test_the_candidates_are_the_words_of_the_pre_registration():
    text = _copy(cand.CANDIDATES_TEXT)
    assert f"`n_estimators` {RULE.more_trees:,} in place of 6,000" in text
    assert f"halving every {RULE.half_life_months} months" in text
    assert RULE.half_life_months in cand.pr.HALF_LIVES


def test_the_plan_shifted_rows_are_the_words_of_the_rule():
    text = _copy(cand.RULE_TEXT)
    assert (
        "the rule covers one validation row, recorded"
        f" {RULE.plan_shift_july_recorded_s:,.0f} s" in text
    )
    assert "December and October hold no such row" in text
    assert RULE.plan_shift_rows == {"december": 0, "july": 1, "october": 0}


def test_the_floors_on_disk_are_the_rule_s():
    """`reports/baseline.md` prints the four floors the rule fixes."""
    text = (REPO / cand.BASELINE).read_text(encoding="utf-8")
    got = cand.read_floors(text)
    assert got == {
        "december": f"{RULE.floor_december:.{D}f}",
        "july": f"{RULE.floor_july:.{D}f}",
        "july_three": f"{RULE.floor_july_three:.{D}f}",
        "october": f"{RULE.floor_october:.{D}f}",
    }
    assert re.fullmatch(r"\d+\.\d\d", cand.read_october(text))


def test_the_year_floor_is_read_from_the_floor_column_of_the_year_row():
    text = (
        "# x\n\n## Per month and over the year\n\nprose\n\n"
        "| month | rows | seed 0 | standard deviation | floor | other |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        "| Jan | 1 | 2.00 | 0.50 | 0.71 | 9.99 |\n"
        "| year | 2 | 3.00 | 0.11 | 0.16 | 9.98 |\n\n"
        "| month | floor |\n| --- | --- |\n| year | 0.99 |\n\n## Next\n"
    )
    assert cand.read_year_floor(text) == "0.16"


# --------------------------------------------------------------------------
# the five conditions, each side of each threshold


def _cond(key):
    return {c.key: c for c in cand.conditions(RULE)}[key]


@pytest.mark.parametrize(
    "key,contrast,expected",
    [
        # better by more than 0.23: the printed gain must exceed the printed floor
        ("december", -0.24, True),
        ("december", -0.23, False),
        ("december", -0.2349, False),  # prints -0.23
        ("december", -0.2351, True),  # prints -0.24
        ("december", +0.10, False),
        # better by more than 0.54
        ("july", -0.55, True),
        ("july", -0.54, False),
        ("july", -0.5449, False),
        ("july", -0.5451, True),
        # not worse by more than 1.02: any gain holds, a printed loss of 1.02 holds
        ("july_three", +1.02, True),
        ("july_three", +1.03, False),
        ("july_three", +1.0249, True),
        ("july_three", +1.0251, False),
        ("july_three", -3.00, True),
        # not worse by more than 0.21
        ("october", +0.21, True),
        ("october", +0.22, False),
        ("october", -0.50, True),
        # the twelve months, not worse by more than 0.16
        ("year", +0.16, True),
        ("year", +0.17, False),
        ("year", +0.1649, True),
        ("year", +0.1651, False),
        ("year", -1.00, True),
    ],
)
def test_each_condition_on_each_side_of_its_floor(key, contrast, expected):
    c = _cond(key)
    assert cand.holds(c.kind, contrast, c.floor, D) is expected


def test_the_conditions_are_the_five_in_the_rule_s_order():
    got = [(c.key, c.kind, c.floor) for c in cand.conditions(RULE)]
    assert got == [
        ("december", "better", 0.23),
        ("july", "better", 0.54),
        ("july_three", "not worse", 1.02),
        ("october", "not worse", 0.21),
        ("year", "not worse", 0.16),
    ]


def test_the_contrast_is_read_as_printed():
    assert cand.hundredths(-0.2351, D) == -24
    assert cand.hundredths(-0.001, D) == 0
    assert cand.hundredths(1.0249, D) == 102


def test_the_other_readings_can_disagree_and_are_only_reported():
    # 214.36 against 214.59 is a printed gain of 0.23, which does not clear
    # 0.23; rounded to one decimal first it reads 214.4 against 214.6, a gain
    # of 0.2, which does not either. A gain of 0.2349 prints 0.23 and does not
    # clear at the rule's precision, and does clear unrounded: the two readings
    # disagree there, and the report says whether any verdict turns on it.
    c = _cond("december")
    assert cand.holds(c.kind, 214.36 - 214.59, c.floor, D) is False
    assert cand.holds_at_figure_precision(c.kind, 214.36, 214.59, c.floor) is False
    assert cand.holds(c.kind, -0.2349, c.floor, D) is False
    assert cand.holds_unrounded(c.kind, -0.2349, c.floor) is True
    assert cand.holds_unrounded(c.kind, -0.2299, c.floor) is False


# --------------------------------------------------------------------------
# the sign check


@pytest.mark.parametrize(
    "seed0,seed1,expected",
    [
        (-0.40, -0.01, True),
        (-0.40, -0.004, False),  # prints -0.00, which has no sign
        (-0.40, +0.01, False),
        (-0.40, 0.0, False),
        (+0.30, +0.20, True),
    ],
)
def test_the_sign_check(seed0, seed1, expected):
    assert cand.same_sign(seed0, seed1, D) is expected


# --------------------------------------------------------------------------
# the rows every figure is over


def _departures(rows):
    """``(icao, month, lag_s, eobt_s, lobt_s, taxi_s)`` per row, as test_tails builds them.

    The take-off is ``lag_s`` after the schedule, ``eobt_s`` after
    ``EOBT_1_flt`` and ``lobt_s`` after ``LOBT_flt`` (``None`` for an estimate
    that is absent), and the block time is the take-off minus ``taxi_s``.
    """
    recs = []
    for i, (icao, month, lag_s, eobt_s, lobt_s, taxi) in enumerate(rows):
        sched = pd.Timestamp(f"{month}-05T10:00:00Z") + pd.Timedelta(minutes=7 * i)
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
                "TAXITIME_SEC_mvt": np.int32(taxi),
            }
        )
    return pd.DataFrame(recs)


def test_the_rows_read_exclude_the_term_s_and_the_rule_s_and_keep_the_rest():
    h = plan_shift_threshold_s()
    train_rows = [("AAAA", "2025-01", 600, 900, 800, 600)] * 4
    train_rows += [("AAAA", "2025-01", 600, h + 3600, h + 3600, 1000)]  # class
    train_rows += [("AAAA", "2025-01", 24 * HOUR, 24 * HOUR, 24 * HOUR, 24 * HOUR - 600)]
    train_rows += [
        ("CCCC", "2025-01", 5 * HOUR, 5 * HOUR, None, 5 * HOUR)
    ] * 5  # term cell
    train_rows += [("CCCC", "2025-01", 600, 900, 800, 700)] * 2
    valid_rows = [
        ("AAAA", "2025-02", 600, 900, 800, 650),  # ordinary: read
        ("AAAA", "2025-02", 600, h + 60, h + 60, 90000),  # plan-shifted: the rule's
        ("CCCC", "2025-02", 5 * HOUR, 5 * HOUR, None, 5 * HOUR),  # the term's
        ("CCCC", "2025-02", 600, 900, 800, 20000),  # ordinary, over 4 h: read
    ]
    dep = _departures(train_rows + valid_rows)
    month = pd.Series(["2025-01"] * len(train_rows) + ["2025-02"] * len(valid_rows))
    train, valid = month == "2025-01", month == "2025-02"
    got = cand.coverage(dep, train, valid)
    v = valid.to_numpy()
    assert got["flagged"].to_numpy()[v].tolist() == [False, False, True, False]
    assert got["shift"].to_numpy()[v].tolist() == [False, True, False, False]
    kept = pd.DataFrame(
        {
            "icao": dep.loc[valid, "ADEP_mvt"].to_numpy(),
            "term": got["flagged"].to_numpy()[v],
            "shift": got["shift"].to_numpy()[v],
            "y": dep.loc[valid, "TAXITIME_SEC_mvt"].to_numpy(dtype=float),
        }
    )
    rows = cand.booster_rows(kept)
    assert rows.tolist() == [True, False, False, True]
    assert cand.short_rows(kept, rows).tolist() == [True, False, False, False]


def test_the_three_are_the_july_airports_among_the_rows_read():
    kept = pd.DataFrame(
        {
            "icao": ["EDDF", "EGLL", "EHAM", "LIRF", "EDDF"],
            "term": [False, False, False, False, True],
            "shift": [False, True, False, False, False],
        }
    )
    assert cand.booster_rows(kept, "three").tolist() == [True, False, True, False, False]


# --------------------------------------------------------------------------
# the candidates as arguments, and the combination


def test_each_candidate_is_one_variation_of_the_rung():
    by = {a.key: a for a in cand.candidate_arms(RULE)}
    ref = cand.reference_arm()
    base = cand.columns_of(ref)
    assert cand.columns_of(by["saturation"]) == base + tuple(cand.SATURATION_FEATURES)
    assert cand.columns_of(by["recency"]) == base == cand.columns_of(by["trees"])
    assert cand.params_of(by["trees"], 0)["n_estimators"] == RULE.more_trees
    assert cand.params_of(ref, 0) == {**cand.BOOSTER_PARAMS}
    assert (
        cand.params_of(ref, 1)["random_state"] == cand.BOOSTER_PARAMS["random_state"] + 1
    )
    assert by["recency"].half_life_months == RULE.half_life_months
    month = pd.Series(["2025-01", "2025-11", "2025-12"])
    rows = pd.Series([True, True, False])
    assert cand.weights_of(ref, month, rows, "2025-11") is None
    w = cand.weights_of(by["recency"], month, rows, "2025-11")
    assert np.allclose(w, [0.5 ** (10 / 12), 1.0])


def test_the_combination_carries_every_clearing_variation():
    by = {a.key: a for a in cand.candidate_arms(RULE)}
    got = cand.combined([by["saturation"], by["trees"]])
    assert (got.saturation, got.half_life_months, got.n_estimators) == (
        True,
        None,
        RULE.more_trees,
    )
    with pytest.raises(ValueError):
        cand.combined([by["trees"]])


def test_the_splits_are_the_legs_and_the_twelve_months():
    legs = {s.name: s for s in cand.leg_plan()}
    assert (legs["december"].last, legs["july"].last, legs["october"].last) == (
        "2025-11",
        "2025-06",
        "2025-09",
    )
    months = cand.month_plan()
    assert len(months) == 12 and len(cand.MONTHS) == 12
    for s in months:
        assert s.valid_month not in s.train_months and len(s.train_months) == 11
    assert cand.check_month() == "2025-01"


# --------------------------------------------------------------------------
# the sequence the rule fixes


def _r(contrast):
    return {"rows": {"contrast": contrast}, "short": {"contrast": contrast}}


def _legs(dec, jul, three, octo, seed="seed0"):
    return {
        seed: {
            "december:all": _r(dec),
            "july:all": _r(jul),
            "july:three": _r(three),
            "october:all": _r(octo),
        }
    }


GOOD = (-0.30, -0.60, +0.50, +0.10)
BAD_DEC = (-0.20, -0.60, +0.50, +0.10)


def test_nothing_clears_the_four_and_nothing_is_fitted_after():
    measured = {k: _legs(*BAD_DEC) for k in ("saturation", "recency", "trees")}
    got = cand.decide(measured, RULE)
    assert got["next"] is None and got["adopted"] is None
    assert got["outcome"] == "none clears four"
    assert "adopts nothing" in cand.verdict_line(got)


def test_a_missing_leg_asks_for_the_legs():
    measured = {"saturation": _legs(*GOOD), "recency": _legs(*GOOD)}
    assert cand.decide(measured, RULE)["next"] == "legs"


def test_the_four_cleared_ask_for_the_months_of_those_alone():
    measured = {
        "saturation": _legs(*GOOD),
        "recency": _legs(*BAD_DEC),
        "trees": _legs(*GOOD),
    }
    got = cand.decide(measured, RULE)
    assert got["next"] == "months" and got["fifth_for"] == ["saturation", "trees"]


def _with_year(m, year):
    m["seed0"]["year:all"] = _r(year)
    return m


def test_failing_the_fifth_adopts_nothing():
    measured = {
        "saturation": _with_year(_legs(*GOOD), +0.17),
        "recency": _legs(*BAD_DEC),
        "trees": _legs(*BAD_DEC),
    }
    got = cand.decide(measured, RULE)
    assert got["outcome"] == "none clears five" and got["adopted"] is None


def test_one_clearing_the_five_goes_to_the_seed_check():
    measured = {
        "saturation": _with_year(_legs(*GOOD), +0.16),
        "recency": _legs(*BAD_DEC),
        "trees": _legs(*BAD_DEC),
    }
    got = cand.decide(measured, RULE)
    assert got["next"] == "seed1" and got["winner"] == "saturation"


def test_two_clearing_the_five_ask_for_the_combination():
    measured = {
        "saturation": _with_year(_legs(*GOOD), 0.0),
        "recency": _legs(*BAD_DEC),
        "trees": _with_year(_legs(-0.40, -0.70, 0.0, 0.0), 0.0),
    }
    got = cand.decide(measured, RULE)
    assert got["next"] == "combination" and got["combine"] == ["saturation", "trees"]


def test_a_combination_that_clears_ships_and_one_that_fails_gives_the_best_december():
    base = {
        "saturation": _with_year(_legs(*GOOD), 0.0),
        "recency": _legs(*BAD_DEC),
        "trees": _with_year(_legs(-0.40, -0.70, 0.0, 0.0), 0.0),
    }
    cleared = {**base, "combination": _with_year(_legs(-0.50, -0.80, 0.0, 0.0), 0.0)}
    assert cand.decide(cleared, RULE)["winner"] == "combination"
    failed = {**base, "combination": _with_year(_legs(-0.50, -0.80, 0.0, 0.30), 0.0)}
    got = cand.decide(failed, RULE)
    assert got["winner"] == "trees"  # -0.40 on December against saturation's -0.30


def test_the_seed_check_decides_the_adoption():
    measured = {
        "saturation": _with_year(_legs(*GOOD), 0.0),
        "recency": _legs(*BAD_DEC),
        "trees": _legs(*BAD_DEC),
    }
    measured["saturation"]["seed1"] = _legs(-0.05, -0.10, +2.0, 0.0)["seed0"]
    got = cand.decide(measured, RULE)
    assert got["adopted"] == "saturation" and got["outcome"] == "adopted"
    assert "The rule adopts saturation" in cand.verdict_line(got)
    # The three are read beside the rule at seed 1 and decide nothing; a sign
    # that turns on December or July does.
    measured["saturation"]["seed1"] = _legs(-0.05, +0.01, -2.0, 0.0)["seed0"]
    got = cand.decide(measured, RULE)
    assert got["adopted"] is None and got["outcome"] == "sign check fails"
    assert got["signs"] == {"december": True, "july": False}


# --------------------------------------------------------------------------
# what the script may and may not do


def test_the_dry_run_fits_nothing(monkeypatch, capsys):
    class FakeFrame:
        def __init__(self):
            self.prepared = []

        def prepare(self, split):
            self.prepared.append(split.name)
            return {"split": split}

        def counts(self, prep):
            return {
                "valid": 1,
                "term": 0,
                "plan_shift": 0,
                "booster": 1,
                "four_hours_or_less": 1,
            }

    def refuse(*_a, **_k):
        raise AssertionError("the dry run fitted something")

    monkeypatch.setattr(cand, "Frame", FakeFrame)
    monkeypatch.setattr(cand, "fit_one", refuse)
    monkeypatch.setattr(cand.Booster, "fit", refuse)
    monkeypatch.setattr(cand, "write_record", refuse)
    monkeypatch.setattr(cand, "preflight", lambda *_a: {})
    monkeypatch.setattr(cand, "fingerprint", lambda exempt=None: {})
    assert cand.dry_run(RULE, Path("unused")) == 0
    out = capsys.readouterr().out
    assert "december__trees__seed0" in out and "nothing was touched" in out


def test_the_records_and_the_report_stay_where_they_belong():
    assert cand.WORK.relative_to(cand.REPO).as_posix() == "data/processed/candidates"
    assert (
        cand.REPORT.relative_to(cand.REPO).as_posix()
        == "reports/candidates-2026-10-03.md"
    )
    forbidden = {p.relative_to(cand.REPO).as_posix() for p in cand.FORBIDDEN}
    assert {"reports", "data/processed/residuals", "data/processed/monthly"} <= forbidden
    assert not any(cand.WORK.is_relative_to(p) for p in cand.FORBIDDEN)


def test_record_names_cannot_read_as_a_stand_identifier():
    name = cand.record_name("heldout-2025-03", "trees", 1)
    assert name == "heldout-2025-03__trees__seed1"
    cand.pr.check_no_identifiers(f"| `{name}` |", {"seed1", "S1"}, set())


def test_the_guard_refuses_a_work_block_an_assistant_and_a_dash():
    credit = "OpenStreetMap contributors"
    cand.guard(f"plain text. {credit}", set(), set())
    # Built from fragments, so this file names no work block itself.
    for bad in ("Bl" + "ock 9q ran it", "as said in the ch" + "at", "a \u2014 dash"):
        with pytest.raises(SystemExit):
            cand.guard(f"{bad}. {credit}", set(), set())
    with pytest.raises(SystemExit):
        cand.guard("no credit", set(), set())
    with pytest.raises(SystemExit):
        cand.guard("written by cla" + f"ude. {credit}", set(), set())


def test_the_module_names_no_assistant_and_no_work_block():
    import check_licence_hygiene as hygiene

    source = (REPO / "scripts" / "candidates.py").read_text(encoding="utf-8")
    assert not re.findall(hygiene.ATTRIBUTION_PATTERN, source, re.IGNORECASE)
    assert not re.search(cand.WORK_BLOCK_PATTERN, source)


# --------------------------------------------------------------------------
# the deadline, and a stop in place of a verdict


def test_a_verdict_measured_after_the_deadline_adopts_nothing():
    measured = {
        "saturation": _with_year(_legs(*GOOD), 0.0),
        "recency": _legs(*BAD_DEC),
        "trees": _legs(*BAD_DEC),
    }
    measured["saturation"]["seed1"] = _legs(-0.05, -0.10, 0.0, 0.0)["seed0"]
    before = cand.decide(measured, RULE, "2026-10-07T11:59:59+00:00")
    assert before["adopted"] == "saturation"
    after = cand.decide(measured, RULE, "2026-10-07T12:00:01+00:00")
    assert after["adopted"] is None and after["outcome"] == "past the deadline"
    assert "7 October 2026, 12:00 UTC" in cand.verdict_line(after)
    assert "adopts nothing" in cand.verdict_line(after)


def test_a_failed_check_replaces_the_verdict():
    line = cand.stop_line([{"what": "the December reference against the ladder"}])
    assert "No verdict" in line and "nothing is adopted" in line
    assert "the December reference against the ladder" in line


def test_every_run_of_a_stage_is_kept():
    first = {"stage": "legs", "fits": ["a"], "started_utc": "t1"}
    second = {"stage": "legs", "fits": [], "started_utc": "t2", "previous": [first]}
    assert [r["started_utc"] for r in cand.runs(second)] == ["t1", "t2"]
    assert cand.runs(None) == []


# --------------------------------------------------------------------------
# the report, built from constructed records


def _kept_frame(split, seed, factor, n_days=6, per_day=20, shift_row=False):
    """A kept frame whose residuals are the reference's times ``factor``.

    The reference's residuals are plus and minus 100 s, so its RMSE is 100 s
    and an arm's contrast against it is ``100 * (factor - 1)`` s on every row
    set. One row of the first day is the term's, and one the plan-shift
    rule's if asked.
    """
    month = {"december": "2025-12", "july": "2025-07", "october": "2025-10"}.get(
        split, split.removeprefix("heldout-")
    )
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day)
    takeoff = pd.to_datetime(
        [f"{month}-{d + 1:02d}T10:00:00Z" for d in day], utc=True
    ) + pd.to_timedelta(np.tile(np.arange(per_day), n_days), unit="min")
    icao = np.array(["EDDF", "EGLL", "EHAM", "LIRF", "LFPG"] * (n // 5))
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    y = np.full(n, 1000.0)
    term = np.zeros(n, dtype=bool)
    term[0] = True
    shift = np.zeros(n, dtype=bool)
    if shift_row:
        shift[1] = True
    pred = y - 100.0 * sign * factor
    return pd.DataFrame(
        {
            "mvt_id": np.arange(n, dtype=float) + 1e9,
            "icao": icao,
            "runway": "09L",
            "takeoff": takeoff,
            "term": term,
            "shift": shift,
            "y": y,
            "pred": pred,
            "with_term": pred,
        }
    )


def _record(split, arm, seed, written="2026-10-03T08:00:00+00:00"):
    return {
        "name": cand.record_name(split, arm.key, seed),
        "split": split,
        "arm": cand.asdict(arm),
        "seed": seed,
        "n_estimators": cand.params_of(arm, seed)["n_estimators"],
        "columns": len(cand.columns_of(arm)),
        "weight_sum": None if arm.half_life_months is None else 1.0,
        "fit_rows": 10,
        "fit_s": 1.0,
        "predict_s": 1.0,
        "peak_rss_gib": 1.0,
        "written_utc": written,
    }


def _world(tmp_path, monkeypatch, factors, months=(), seed1=None):
    """Records on disk for the reference and the arms, at the given factors."""
    monkeypatch.setattr(cand, "WORK", tmp_path)
    by = {a.key: a for a in cand.candidate_arms(RULE)}
    ref = cand.reference_arm()
    for leg in ("december", "july", "october"):
        cand.write_record(
            _record(leg, ref, 0), _kept_frame(leg, 0, 1.0, shift_row=leg == "july")
        )
        for key, legs in factors.items():
            cand.write_record(
                _record(leg, by[key], 0),
                _kept_frame(leg, 0, legs[leg], shift_row=leg == "july"),
            )
    for key in months:
        for split in cand.month_plan():
            cand.write_record(
                _record(split.name, ref, 0), _kept_frame(split.name, 0, 1.0)
            )
            cand.write_record(
                _record(split.name, by[key], 0),
                _kept_frame(split.name, 0, factors[key]["year"]),
            )
    for key, legs in (seed1 or {}).items():
        for leg in ("december", "july"):
            cand.write_record(
                _record(leg, ref, 1), _kept_frame(leg, 1, 1.0, shift_row=leg == "july")
            )
            cand.write_record(
                _record(leg, by[key], 1),
                _kept_frame(leg, 1, legs[leg], shift_row=leg == "july"),
            )
    # The base rung's held-out months are its stored files in the repository;
    # here they are the constructed reference records.
    monkeypatch.setattr(
        cand,
        "base_month",
        lambda name, kept: cand.read_kept(cand.record_name(name, "reference", 0)),
    )
    monkeypatch.setattr(cand, "report_inputs", lambda records: [REPO / cand.BASELINE])
    return cand.read_records()


FLOORS = {"seeds_report_sha256": "0" * 64}
NONE_CLEAR = {
    k: {"december": 0.999, "july": 0.999, "october": 1.0}
    for k in ("saturation", "recency", "trees")
}


def _checks(ok=True):
    checks = cand.Checks()
    checks.add("a constructed check", "x", "x" if ok else "y", ok)
    return checks


def test_the_report_when_nothing_clears_the_four(tmp_path, monkeypatch):
    records = _world(tmp_path, monkeypatch, NONE_CLEAR)
    text = cand.build_report(records, RULE, _checks(), FLOORS, "2026-10-03")
    assert text.count("**The rule adopts nothing.** No candidate holds the four") == 2
    assert "**Holding all four:** none of the three." in text
    assert "and none did, so none was fitted." in text
    assert "## Seed 1" not in text and "## The combination" not in text
    cand.guard(text, set(), set())


def test_the_report_of_an_adoption(tmp_path, monkeypatch):
    factors = {
        "saturation": {"december": 0.99, "july": 0.99, "october": 1.0, "year": 1.0},
        "recency": NONE_CLEAR["recency"],
        "trees": NONE_CLEAR["trees"],
    }
    records = _world(
        tmp_path,
        monkeypatch,
        factors,
        months=("saturation",),
        seed1={"saturation": {"december": 0.995, "july": 0.995}},
    )
    text = cand.build_report(records, RULE, _checks(), FLOORS, "2026-10-03")
    assert text.count("**The rule adopts saturation on top of") == 2
    assert "### Saturation on top of" in text and "**the year**" in text
    assert "## Seed 1, the sign check" in text and "**not the same sign**" not in text
    cand.guard(text, set(), set())


def test_the_report_of_a_stopped_run_reads_nothing(tmp_path, monkeypatch):
    records = _world(tmp_path, monkeypatch, NONE_CLEAR)
    text = cand.build_report(records, RULE, _checks(ok=False), FLOORS, "2026-10-03")
    assert text.count("**No verdict: a check disagreed") == 2
    adopts = text.split("## What the rule adopts\n\n")[1].split("\n")[0]
    assert adopts.startswith("**No verdict: a check disagreed")
    assert "## The four conditions" not in text
    cand.guard(text, set(), set())


def test_the_fifth_section_names_what_held_the_four_when_months_are_missing(
    tmp_path, monkeypatch
):
    factors = dict(NONE_CLEAR)
    factors["trees"] = {"december": 0.99, "july": 0.99, "october": 1.0}
    records = _world(tmp_path, monkeypatch, factors)
    text = cand.build_report(records, RULE, _checks(), FLOORS, "2026-10-03")
    assert "Not reached yet: more trees held the four conditions" in text
    assert "none did" not in text


def test_the_contrast_and_floor_at_one_decimal_is_reported_beside_the_rule():
    # +1.04 against 1.02 fails at the rule's precision and holds when both are
    # first rounded to one decimal; the report lists it and decides nothing on it.
    c = _cond("july_three")
    assert cand.holds(c.kind, 1.04, c.floor, D) is False
    assert cand.holds_at_contrast_precision(c.kind, 1.04, c.floor) is True
    assert cand.holds_read("contrast", c.kind, {"contrast": 1.04}, c.floor, D) is True
    with pytest.raises(ValueError):
        cand.holds_read("tenths", c.kind, {"contrast": 1.04}, c.floor, D)


def test_the_readings_section_names_what_turns_and_what_does_not(tmp_path, monkeypatch):
    factors = dict(NONE_CLEAR)
    # recency: a July three loss of 1.04 s on a 100 s RMSE
    factors["recency"] = {"december": 0.999, "july": 1.0104, "october": 1.0}
    records = _world(tmp_path, monkeypatch, factors)
    text = cand.build_report(records, RULE, _checks(), FLOORS, "2026-10-03")
    section = text.split("## Whether a verdict turns on the precision")[1].split("## ")[0]
    assert "the mild recency schedule, July, the three" in section
    assert (
        "Under every reading the same candidates hold the conditions read: no candidate."
        in section
    )
