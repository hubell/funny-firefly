# SPDX-License-Identifier: GPL-3.0-only
"""The recomputed residual-fit headroom, on synthetic residuals.

What is worth holding to a test here is not the size of any published figure --
that depends on a month of real residuals -- but the four properties that make
the figure mean what the report says it means:

- the folds group by take-off day, and **no day appears in two folds**, because
  a fold that split a day would let a fit read its own afternoon off its own
  morning and report the leak as headroom;
- a block of pure noise reads no headroom, beyond a tolerance stated here;
- a block holding a column that reconstructs the residual exactly reads the
  whole residual RMSE, which is the other end of the same scale;
- the estimator refuses a column outside the block's registry entry, because a
  headroom that read one column from outside the block would not be an estimate
  of that block at all and the failure would be invisible in the number.

Beside them, the guard every published report carries: the generator must
refuse to write a stand identifier it was handed.
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

import report_headroom as rh  # noqa: E402

# How much headroom a block of pure noise is allowed to read before this test
# calls it a defect, in seconds. Not zero: the out-of-fold RMSE of a fit on
# noise is a random variable, and on a frame this size a tenth of a second
# either way is the estimator working. Anything approaching the size of a real
# block's figure is not.
NOISE_TOLERANCE_S = 0.5

# How close the exact-reconstruction case has to come to the whole residual
# RMSE, as a share of it. Not zero either: the fit is boosted trees at a finite
# tree count, so it approaches a step function rather than arriving at it.
EXACT_TOLERANCE = 0.02

T0 = pd.Timestamp("2025-12-01T00:00:00Z")


def _frame(days: int = 20, per_day: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic residuals: ``days`` whole days of ``per_day`` take-offs each."""
    rng = np.random.default_rng(seed)
    n = days * per_day
    day = np.repeat(np.arange(days), per_day)
    minute = np.tile(np.arange(per_day) * 7, days)
    frame = pd.DataFrame(
        {
            "mvt_id": np.arange(n, dtype=float),
            "icao": np.where(day % 2 == 0, "AAAA", "BBBB"),
            "takeoff": T0
            + pd.to_timedelta(day, unit="D")
            + pd.to_timedelta(minute, unit="m"),
            "residual": rng.normal(0.0, 300.0, n),
        }
    )
    frame["day"] = frame["takeoff"].dt.strftime("%Y-%m-%d")
    frame["fold"] = rh.day_folds(frame["takeoff"])
    return frame


def test_the_folds_are_whole_days_and_no_day_is_in_two_of_them():
    frame = _frame()
    crossing = frame.groupby("day")["fold"].nunique()
    assert set(crossing.unique()) == {1}
    assert sorted(frame["fold"].unique()) == list(range(rh.HEADROOM_FOLDS))
    # Whole days dealt round robin, so the folds hold within one day of each
    # other however the month divides.
    days = frame.groupby("fold")["day"].nunique()
    assert int(days.max()) - int(days.min()) <= 1


def test_the_fold_assignment_is_a_function_of_the_seed_and_the_calendar():
    frame = _frame()
    again = rh.day_folds(frame["takeoff"])
    assert (again.to_numpy() == frame["fold"].to_numpy()).all()
    other = rh.day_folds(frame["takeoff"], seed=rh.HEADROOM_FOLD_SEED + 1)
    assert not (other.to_numpy() == frame["fold"].to_numpy()).all()
    # Shuffling the rows cannot move a row's fold: the day decides it.
    shuffled = frame.sample(frac=1.0, random_state=3)
    assert (
        rh.day_folds(shuffled["takeoff"]).to_numpy() == shuffled["fold"].to_numpy()
    ).all()


def _headroom(frame: pd.DataFrame, columns: tuple[str, ...], **override: object) -> float:
    """The headroom of ``columns`` over the whole frame, at the named capacity.

    ``override`` is for a test that wants a cheaper fit than the published one.
    The thread count is cut in both cases; it changes which bits a sum comes
    out at and not whether either property below holds.
    """
    params = dict(rh.HEADROOM_BOOSTER_PARAMS) | {"n_jobs": 2} | override
    oof = rh.out_of_fold(frame, columns, params=params)
    return rh.headroom_on(frame, oof, None)["gain"]


def test_a_block_of_pure_noise_reads_no_headroom():
    frame = _frame()
    rng = np.random.default_rng(11)
    for c in ("occ_gap1_s", "occ_gap2_s"):
        frame[c] = rng.normal(0.0, 1.0, len(frame))
    assert (
        _headroom(frame, ("occ_gap1_s", "occ_gap2_s"), n_estimators=120)
        <= NOISE_TOLERANCE_S
    )


def test_a_column_that_reconstructs_the_residual_reads_the_whole_rmse():
    """The other end of the scale, and the check that the reduction is a reduction.

    At the published tree count, not a cut one: a boosted fit approaches a step
    function at a rate the tree count sets, so at 120 trees and this learning
    rate the tolerance would be measuring the tree count rather than the
    estimator. One feature and 2,400 rows, so the full count is still quick.
    """
    frame = _frame()
    rng = np.random.default_rng(13)
    # A column the residual is a step function of, so a tree can represent it
    # exactly and a fold can learn it from the other four.
    level = rng.integers(0, 12, len(frame))
    frame["residual"] = level * 100.0 - 550.0
    frame["occ_gap1_s"] = level.astype(float)
    before = rh.headroom_on(frame, np.zeros(len(frame)), None)["before"]
    gain = _headroom(frame, ("occ_gap1_s",))
    assert before > 0
    assert abs(gain - before) <= EXACT_TOLERANCE * before


def test_the_estimator_refuses_a_column_outside_the_block():
    """The columns are the registry's, and the guard says so in the traceback."""
    for key, _label, columns in rh.HEADROOM_BLOCKS:
        assert rh.check_block_columns(key, columns) == tuple(columns)
    with pytest.raises(KeyError, match="occ_gap1_s"):
        # A real feature column, and one the stand-occupancy block owns -- so
        # the guard has to read the block and not merely the feature list.
        rh.check_block_columns("callsign", ("cs_prefix", "occ_gap1_s"))
    with pytest.raises(KeyError, match="lag_eobt_s"):
        rh.check_block_columns("plan-diffs", rh.PLAN_DIFF_FEATURES + ("lag_eobt_s",))
    with pytest.raises(KeyError):
        rh.block_columns("saturation")


def test_the_blocks_are_the_four_the_gated_constant_carries():
    """Not a list of this file's own: the audit's four rungs and their columns."""
    from taxiout.models import LADDER_BY_KEY

    assert [k for k, _l, _c in rh.HEADROOM_BLOCKS] == list(rh.EXPECTED_HEADROOM_S)
    # Each block's columns are exactly what that rung added to the one below it.
    for key, _label, columns in rh.HEADROOM_BLOCKS:
        rung = LADDER_BY_KEY[key]
        base = LADDER_BY_KEY[rung.base] if rung.base else None
        added = tuple(c for c in rung.features if not base or c not in base.features)
        assert set(added) == set(columns)


def test_the_capacity_is_named_and_is_not_the_ladder_s_own():
    """The substantive addition, held to being an addition."""
    from taxiout.models import BOOSTER_PARAMS

    for axis in rh.CAPACITY_AXES:
        assert axis in rh.HEADROOM_BOOSTER_PARAMS
    assert rh.HEADROOM_BOOSTER_PARAMS != BOOSTER_PARAMS
    # Smaller than the ladder's cell on the two axes that set the capacity, and
    # the report's own capacity table is what says why that matters.
    assert rh.HEADROOM_BOOSTER_PARAMS["n_estimators"] < BOOSTER_PARAMS["n_estimators"]
    assert rh.HEADROOM_BOOSTER_PARAMS["num_leaves"] < BOOSTER_PARAMS["num_leaves"]
    # The reconstruction rule the report states: the smallest value on each of
    # the four axes of the grid that was live when the review was written.
    from sweep_params import FIRST_SWEEP

    assert rh.HEADROOM_BOOSTER_PARAMS["n_estimators"] == min(FIRST_SWEEP["trees"])
    assert rh.HEADROOM_BOOSTER_PARAMS["learning_rate"] == min(
        FIRST_SWEEP["learning_rate"]
    )
    assert rh.HEADROOM_BOOSTER_PARAMS["num_leaves"] == min(FIRST_SWEEP["num_leaves"])
    assert rh.HEADROOM_BOOSTER_PARAMS["min_child_samples"] == min(
        FIRST_SWEEP["min_child_samples"]
    )


def test_this_report_moves_no_threshold_and_reads_the_frozen_arm():
    """The prohibition the block was written under, as a test rather than a promise."""
    from taxiout.models import SUBMISSION_KEY

    assert rh.HEADROOM_TRIGGER_S == 5.0
    assert rh.EXPECTED_HEADROOM_S == {
        "occupancy": (3.9, 10.3),
        "callsign": (2.1, 2.8),
        "destination": (1.9, 1.7),
        "plan-diffs": (1.0, 1.0),
    }
    # The rung the review fitted against, which is no longer the submission rung.
    assert rh.RESIDUAL_RUNG == "encoding-for-stand"
    assert rh.RESIDUAL_RUNG != SUBMISSION_KEY


def test_the_generator_refuses_to_write_a_literal_stand_identifier():
    """The guard every published report carries, on this report's own text."""
    from run_baselines import check_no_identifiers

    matrix = pd.DataFrame(
        {
            "stand_id": ["A12", "B01"],
            "designator": ["09L", "27R"],
            "stand": ["A12", "B01"],
        }
    )
    with pytest.raises(SystemExit):
        check_no_identifiers("the headroom is largest at stand A12", {"A12"}, matrix)
    # Airports, runway designators and the report's own column names are fine.
    check_no_identifiers(
        "stand occupancy, July the three: EDDF on 09L, +10.35 s", {"A12"}, matrix
    )


def test_the_published_report_carries_the_capacity_and_the_audit_s_verdict():
    """A rebuild that dropped either would leave the file misleading on its own."""
    report = REPO / "reports" / "headroom.md"
    if not report.exists():
        pytest.skip("reports/headroom.md is not built here")
    text = report.read_text(encoding="utf-8")
    for wanted in (
        "## The capacity, named",
        "`HEADROOM_BOOSTER_PARAMS`, in full:",
        "capacity below is a reconstruction",
        "## The recomputed headroom, beside the quoted figures",
        "**The audit is unchanged by this report.**",
        "This report draws no conclusion about the trigger.",
        "OpenStreetMap contributors",
    ):
        assert wanted in text, wanted
