# SPDX-License-Identifier: GPL-3.0-only
"""The guard on seed averaging, and the two columns the rule asks for.

The rule of 19 September is read on the two forward legs and has been tested
through the report it prints. What is new, and what is held to a test here, is
the guard of 20 September and the leg the seed section now fits for itself:
the three airports July 2026 contains, and October, which no rung of the
ladder scores. A wrong verdict computed from right figures reads exactly like
a right one, so the verdict is what is tested and not the figures.

Nothing here fits a booster or reads the challenge data. The figures are
constructed to put each boundary in view and none of them is a measurement.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_baselines as rb  # noqa: E402
from taxiout.models import SUBMISSION_KEY  # noqa: E402


@dataclass
class _Split:
    name: str


SPLITS = [_Split("december"), _Split("july"), _Split("pairs-0")]
FLOORS = {
    "December, all ten": 0.34,
    "July, all ten": 0.23,
    "July, the three": 1.02,
    rb.SEED_AVERAGE_EXTRA_LABEL: 0.19,
}


def _bagging(
    *,
    three_single: float = 237.80,
    three_avg: float = 236.80,
    october_single: float = 217.28,
    october_avg: float = 216.88,
    october: bool = True,
) -> dict:
    """The two forward legs meeting the rule, plus the two guard columns."""
    got = {
        "december": {
            "rmse_ex_flagged": 217.11,
            "single": 217.59,
            "reproduced": 0.0,
            "seeds": [217.59, 217.71, 217.83, 217.95, 218.11],
            "seeds_three": [float("nan")] * 5,
            "three": float("nan"),
            "single_three": float("nan"),
        },
        "july": {
            "rmse_ex_flagged": 325.29,
            "single": 325.81,
            "reproduced": 0.0,
            "seeds": [325.81, 326.14, 326.09, 326.23, 326.08],
            "seeds_three": [237.80, 237.14, 238.83, 237.90, 237.02],
            "three": three_avg,
            "single_three": three_single,
        },
    }
    if october:
        got[rb.SEED_AVERAGE_EXTRA_LEG] = {
            "rmse_ex_flagged": october_avg,
            "single": october_single,
            "reproduced": float("nan"),
            "seeds": [217.28, 217.65, 217.43, 217.37, 217.43],
            "seeds_three": [float("nan")] * 5,
            "three": float("nan"),
            "single_three": float("nan"),
            "rows": 185674,
            "fit_rows": 1570127,
        }
    return got


# --------------------------------------------------------------------------
# the two columns the rule asks for


def test_the_pending_measurements_are_made_by_the_section_itself():
    """The tuple is empty because the section makes them, not because it was cleared."""
    assert rb.SEED_AVERAGE_PENDING == ()
    guard = rb.seed_average_guard(_bagging(), SPLITS, FLOORS)
    assert [g["column"] for g in guard] == ["the three", rb.SEED_AVERAGE_EXTRA_LABEL]
    assert all(g["known"] for g in guard)


def test_october_is_not_a_split_of_the_ladder():
    """Adding it to the splits would move what the second forward leg means."""
    assert rb.SEED_AVERAGE_EXTRA_LEG not in {s.name for s in SPLITS}
    assert rb.OCTOBER_VALID == ("2025-10",)
    assert rb.OCTOBER_TRAIN[-1] == "2025-09"
    assert len(rb.OCTOBER_TRAIN) == 9


def test_the_floor_of_the_extra_leg_is_measured_from_its_own_five_singles():
    floors = rb._noise_floors(_bagging(), SPLITS)
    seeds = np.asarray([217.28, 217.65, 217.43, 217.37, 217.43])
    expected = rb.NOISE_FLOOR_FACTOR * float(np.std(seeds, ddof=1))
    assert floors[rb.SEED_AVERAGE_EXTRA_LABEL] == pytest.approx(expected)
    # Root two times the sample standard deviation, not the population one.
    assert floors[rb.SEED_AVERAGE_EXTRA_LABEL] > rb.NOISE_FLOOR_FACTOR * float(
        np.std(seeds)
    )


def test_no_floor_for_the_extra_leg_where_the_section_did_not_fit_it():
    assert rb.SEED_AVERAGE_EXTRA_LABEL not in rb._noise_floors(
        _bagging(october=False), SPLITS
    )


# --------------------------------------------------------------------------
# the guard itself


def test_the_guard_holds_where_both_columns_are_better():
    guard = rb.seed_average_guard(_bagging(), SPLITS, FLOORS)
    assert all(g["holds"] for g in guard)
    # Positive is worse, as everywhere in the report.
    assert all(g["change"] < 0 for g in guard)


def test_the_guard_holds_where_a_column_is_worse_by_less_than_its_floor():
    guard = rb.seed_average_guard(
        _bagging(october_single=217.28, october_avg=217.38), SPLITS, FLOORS
    )
    october = guard[-1]
    assert october["change"] == pytest.approx(0.10)
    assert october["holds"] is True


def test_the_guard_trips_where_a_column_is_worse_by_more_than_its_floor():
    guard = rb.seed_average_guard(
        _bagging(october_single=217.28, october_avg=217.68), SPLITS, FLOORS
    )
    october = guard[-1]
    assert october["change"] == pytest.approx(0.40)
    assert october["holds"] is False


def test_worse_by_exactly_the_floor_does_not_trip_the_guard():
    """The guard trips on worse by *more* than the floor, which is its wording.

    The boundary is written in halves because a decimal like 0.19 is not exact
    in binary, and a test written in decimals would be testing the rounding
    rather than the rule.
    """
    floors = {**FLOORS, rb.SEED_AVERAGE_EXTRA_LABEL: 0.5}
    guard = rb.seed_average_guard(
        _bagging(october_single=217.0, october_avg=217.5), SPLITS, floors
    )
    assert guard[-1]["change"] == pytest.approx(0.5)
    assert guard[-1]["holds"] is True
    guard = rb.seed_average_guard(
        _bagging(october_single=217.0, october_avg=217.75), SPLITS, floors
    )
    assert guard[-1]["holds"] is False


def test_the_guard_reads_the_three_on_the_july_leg():
    guard = rb.seed_average_guard(
        _bagging(three_single=237.80, three_avg=239.00), SPLITS, FLOORS
    )
    three = guard[0]
    assert three["column"] == "the three"
    assert three["change"] == pytest.approx(1.20)
    assert three["floor"] == pytest.approx(1.02)
    assert three["holds"] is False


def test_a_column_that_was_not_measured_is_not_a_column_that_held():
    guard = rb.seed_average_guard(_bagging(october=False), SPLITS, FLOORS)
    assert [g["column"] for g in guard] == ["the three"]
    guard = rb.seed_average_guard(_bagging(), SPLITS, {})
    assert all(g["holds"] is None and not g["known"] for g in guard)


# --------------------------------------------------------------------------
# what the section says


def _section(**kwargs) -> str:
    bagging = _bagging(**kwargs)
    floors = rb._noise_floors(bagging, SPLITS)
    return "\n".join(rb.section_bagging(bagging, SPLITS, SUBMISSION_KEY, floors))


def test_the_section_prints_both_columns_with_their_floors():
    text = _section()
    assert "the three" in text
    assert rb.SEED_AVERAGE_EXTRA_LABEL in text
    assert rb.SEED_AVERAGE_GUARD in text
    assert "| the three | 237.80 | 236.80 | -1.00 |" in text


def test_the_section_adopts_where_the_rule_and_the_guard_both_hold():
    text = _section()
    assert "**Adopted:**" in text
    assert "the guard holds on each" in text


def test_the_section_refuses_where_the_guard_trips():
    text = _section(october_single=217.28, october_avg=218.28)
    assert "the guard of 20 September trips" in text
    assert "**Adopted:**" not in text
    assert "fitted from a single seed" in text


def test_the_section_says_not_yet_where_a_column_was_not_measured():
    text = _section(october=False)
    assert "**Not yet adopted.**" in text
    assert rb.SEED_AVERAGE_EXTRA_LABEL in text


def test_the_section_still_reads_the_rule_on_the_two_forward_legs():
    """The guard is added beside the rule and does not replace it."""
    text = _section()
    assert "What the two forward legs show against it." in text
    assert rb.SEED_AVERAGE_RULE in text
