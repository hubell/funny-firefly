# SPDX-License-Identifier: GPL-3.0-only
"""The pooled record-quality rule: the pooled cell, the premise test, the arm.

What is worth holding to a test here is the arithmetic the decision turned on,
on data small enough to check by hand: that a pooled share is the row-weighted
mean of the airports' shares and nothing cleverer, that a pooled cell under the
term's own minimum share covers nobody, that the two-standard-error test says
"one share" where the rows were drawn from one and says the opposite of the
case that raised the question, that the stake is counted from exposed fields
alone, and that the arm changes no cell the term already covers.

Beside them, one alarm rather than an arithmetic check: ``reports/tails.md`` is
written by ``scripts/report_tails.py`` in full and by ``scripts/pooled_tails.py``
one section at a time, so a rebuild that ran only the first would drop the
section silently. The last test here fails loudly instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pooled_tails as pt  # noqa: E402
from taxiout.tails import MIN_COUNT, MIN_SHARE, PREDICT_INPUTS, TailModel  # noqa: E402

HOUR = 3600
T0 = pd.Timestamp("2025-03-01T10:00:00Z")


def _frame(rows) -> pd.DataFrame:
    """Departures as ``(icao, lag_s, flt_present, block_offset_s, taxi_s)``.

    The same shape ``tests/test_tails.py`` uses: the take-off is ``lag_s``
    after the schedule, the block time is ``block_offset_s`` after it, and
    ``None`` derives the block time from the taxi so that ``taxi = take-off -
    block`` holds as it does in the data.
    """
    recs = []
    for i, (icao, lag_s, flt, block_off, taxi) in enumerate(rows):
        sched = T0 + pd.Timedelta(minutes=7 * i)
        mvt = sched + pd.Timedelta(seconds=lag_s)
        block = (
            mvt - pd.Timedelta(seconds=taxi)
            if block_off is None
            else (sched + pd.Timedelta(seconds=block_off))
        )
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


def _late(icao: str, defaulted: int, ordinary: int, lag_s: int = 5 * HOUR) -> list:
    """One airport's rows in a late no-record bucket, so many of them defaulted."""
    return [(icao, lag_s, False, 0, lag_s)] * defaulted + [
        (icao, lag_s, False, None, 900)
    ] * ordinary


# --------------------------------------------------------------------------
# the pooled cell


def test_the_pooled_share_is_the_row_weighted_mean_of_the_airports_shares():
    train = _frame(_late("AAAA", 9, 1) + _late("BBBB", 0, 30) + _late("CCCC", 3, 7))
    cells = TailModel().fit(train).cells
    pooled = pt.pool(cells)
    bucket = "noflt:4-12h"
    n = cells.xs(bucket, level="tail_bucket")["n"]
    share = cells.xs(bucket, level="tail_bucket")["share"]
    assert pooled.loc[bucket, "n"] == 50
    assert pooled.loc[bucket, "share"] == pytest.approx(
        float((n * share).sum() / n.sum())
    )
    # and that is the same thing as the bucket's own defaulted fraction
    assert pooled.loc[bucket, "share"] == pytest.approx(12 / 50)
    # the pooled rest mean weights the rows that were not defaulted, and only
    # those: 38 ordinary rows of 900 s each
    assert pooled.loc[bucket, "rest_mean"] == pytest.approx(900.0)


def test_the_rounded_pooling_reproduces_a_reading_from_the_printed_shares():
    """The printed table rounds a share to a tenth of a percent; pooling follows."""
    train = _frame(_late("AAAA", 3, 397) + _late("BBBB", 0, 600))
    cells = TailModel().fit(train).cells
    bucket = "noflt:4-12h"
    measured = pt.pool(cells).loc[bucket, "share"]
    rounded = pt.pool(cells, rounded=True).loc[bucket, "share"]
    assert measured == pytest.approx(3 / 1000)
    # the printed share is taken from the report's own formatter and read back,
    # and the pooled figure follows it rather than the measured share
    printed = float(pt.rt.pct(3 / 400)[:-1]) / 100
    assert rounded == pytest.approx(400 * printed / 1000)
    assert rounded != measured


def test_a_pooled_share_under_the_minimum_yields_no_covered_pooled_cell():
    """A cell under ``MIN_SHARE`` is not covered, so it reaches nobody."""
    train = _frame(_late("AAAA", 1, 9) + _late("BBBB", 0, 190))
    cells = TailModel().fit(train).cells
    pooled = pt.pool(cells)
    bucket = "noflt:4-12h"
    assert pooled.loc[bucket, "n"] == 200
    assert pooled.loc[bucket, "share"] == pytest.approx(0.005)
    assert pooled.loc[bucket, "share"] < MIN_SHARE
    assert not bool(pooled.loc[bucket, "covered"])
    assert not pt.serves(cells, pooled)[bucket]
    # and nothing is predicted for the airport that has no covered cell
    test = _frame([("BBBB", 6 * HOUR, False, None, 0)])
    assert pt.arm_prediction(train, test).isna().all()


def test_a_bucket_every_airport_already_covers_is_not_served():
    train = _frame(_late("AAAA", 9, 1) + _late("BBBB", 8, 2))
    cells = TailModel().fit(train).cells
    pooled = pt.pool(cells)
    bucket = "noflt:4-12h"
    assert bool(pooled.loc[bucket, "covered"])
    assert cells.xs(bucket, level="tail_bucket")["covered"].all()
    assert not pt.serves(cells, pooled)[bucket]


# --------------------------------------------------------------------------
# the premise test


def test_two_airports_drawn_from_one_share_pass_the_two_standard_error_test():
    train = _frame(_late("AAAA", 50, 50) + _late("BBBB", 50, 50))
    cells = TailModel().fit(train).cells
    pooled = pt.pool(cells)
    got = pt.premise(cells, pooled)["noflt:4-12h"]
    assert got["tested"] == 2
    assert got["failed"] == []
    assert got["holds"] is True
    assert all(r["passes"] for r in got["rows"])


def test_the_case_that_raised_the_question_fails_it():
    """A 0.0% cell of 77 rows against an 84.7% cell of 98, which is the case."""
    train = _frame(_late("AAAA", 83, 15) + _late("BBBB", 0, 77))
    cells = TailModel().fit(train).cells
    pooled = pt.pool(cells)
    bucket = "noflt:4-12h"
    assert cells.loc[("AAAA", bucket), "n"] == 98
    assert cells.loc[("AAAA", bucket), "share"] == pytest.approx(83 / 98, abs=5e-4)
    assert cells.loc[("BBBB", bucket), "n"] == 77
    assert cells.loc[("BBBB", bucket), "share"] == 0.0
    got = pt.premise(cells, pooled)[bucket]
    assert got["tested"] == 2
    assert sorted(got["failed"]) == ["AAAA", "BBBB"]
    assert got["holds"] is False
    # the interval is two standard errors of the pooled share at each airport's
    # own row count, and it is arithmetic rather than a threshold
    p = float(pooled.loc[bucket, "share"])
    for r in got["rows"]:
        se = np.sqrt(p * (1 - p) / r["n"])
        assert r["low"] == pytest.approx(p - 2 * se)
        assert r["high"] == pytest.approx(p + 2 * se)


def test_a_cell_under_the_minimum_count_is_not_tested():
    train = _frame(_late("AAAA", 50, 50) + _late("BBBB", 0, MIN_COUNT - 1))
    cells = TailModel().fit(train).cells
    got = pt.premise(cells, pt.pool(cells))["noflt:4-12h"]
    assert got["tested"] == 1
    assert [r["tested"] for r in got["rows"]] == [True, False]
    assert got["holds"] is True


# --------------------------------------------------------------------------
# the stake


def _stake_setup():
    train = _frame(_late("AAAA", 9, 1) + _late("BBBB", 0, 10))
    cells = TailModel().fit(train).cells
    return train, cells, pt.pool(cells)


def test_the_stake_reads_no_field_outside_predict_inputs():
    train, cells, pooled = _stake_setup()
    ranking = _frame(
        [
            ("AAAA", 6 * HOUR, False, None, 0),  # covered today
            ("BBBB", 6 * HOUR, False, None, 0),  # uncovered, the pooled cell reaches it
            ("CCCC", 6 * HOUR, False, None, 0),  # an airport with no cell at all
            ("BBBB", 600, False, None, 0),  # a bucket no pooled cell covers
        ]
    )
    exposed = ranking[list(PREDICT_INPUTS)]
    got = pt.stake_rows(exposed, cells, pooled)
    assert got["covered_now"].tolist() == [True, False, False, False]
    assert got["gained"].tolist() == [False, True, True, False]
    # the hidden fields are present in the wider frame and change nothing,
    # because the frame is narrowed before anything is computed
    wide = pt.stake_rows(ranking, cells, pooled)
    assert wide["gained"].tolist() == got["gained"].tolist()
    with pytest.raises(KeyError):
        pt.stake_rows(exposed.drop(columns=["ADEP_mvt"]), cells, pooled)


def test_the_stake_is_grouped_by_airport_month_and_bucket():
    train, cells, pooled = _stake_setup()
    ranking = _frame([("BBBB", 6 * HOUR, False, None, 0)] * 3)
    got = pt.stake_rows(ranking[list(PREDICT_INPUTS)], cells, pooled)
    counted = (
        got[got["gained"]].groupby(["icao", "month", "bucket"], observed=True).size()
    )
    assert counted.to_dict() == {("BBBB", "2025-03", "noflt:4-12h"): 3}


# --------------------------------------------------------------------------
# the arm


def test_the_arm_leaves_every_already_covered_cell_untouched():
    train = _frame(_late("AAAA", 9, 1) + _late("BBBB", 0, 10))
    model = TailModel().fit(train)
    test = _frame(
        [
            ("AAAA", 6 * HOUR, False, None, 0),  # covered by the term itself
            ("BBBB", 6 * HOUR, False, None, 0),  # uncovered, and the pooled cell covers
            ("AAAA", 600, True, None, 0),  # a clean cell, which nothing covers
        ]
    )
    base = model.predict(test)
    arm = pt.arm_prediction(train, test)
    covered = base.notna().to_numpy()
    assert covered.tolist() == [True, False, False]
    assert arm[covered].tolist() == base[covered].tolist()
    pooled = pt.pool(model.cells)
    p = float(pooled.loc["noflt:4-12h", "share"])
    rest = float(pooled.loc["noflt:4-12h", "rest_mean"])
    assert arm.iloc[1] == pytest.approx(p * 6 * HOUR + (1 - p) * rest)
    assert np.isnan(arm.iloc[2])


def test_the_arm_is_the_adopted_term_where_no_pooled_cell_is_covered():
    """With every pooled cell under the minimum share, the arm is the term."""
    train = _frame(_late("AAAA", 9, 1) + _late("BBBB", 0, 400))
    model = TailModel().fit(train)
    assert not pt.pool(model.cells)["covered"].any()
    test = _frame(
        [("AAAA", 6 * HOUR, False, None, 0), ("BBBB", 6 * HOUR, False, None, 0)]
    )
    base, arm = model.predict(test), pt.arm_prediction(train, test)
    assert arm.fillna(-1).tolist() == base.fillna(-1).tolist()


# --------------------------------------------------------------------------
# the report the two generators share


def test_the_committed_report_still_carries_the_section():
    """``make tails`` runs both generators; only one of them writes this section.

    ``scripts/report_tails.py`` rebuilds ``reports/tails.md`` from end to end
    and knows nothing about the pooled measurement, so a rebuild that stopped
    there would drop the section and no figure elsewhere would move to say so.
    """
    report = REPO / "reports" / "tails.md"
    text = report.read_text(encoding="utf-8")
    assert pt.SECTION_TITLE in text, "reports/tails.md has lost the pooled section"
    assert text.index(pt.SECTION_TITLE) < text.index(pt.FOOTER_ANCHOR)
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "scripts/pooled_tails.py --report" in makefile
