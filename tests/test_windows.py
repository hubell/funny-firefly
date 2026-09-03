# SPDX-License-Identifier: GPL-3.0-only
"""The window machinery every congestion feature is built on.

The three rules the module fixes -- closed at both ends, the subject removed
by identity, truncation flagged -- are each pinned by a test that would fail
if the rule moved, because a rule that moves between the training side and
the ranking side is exactly the leak the module exists to prevent.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.windows import (  # noqa: E402
    Sorted,
    centred_count,
    configuration,
    coverage_end,
    coverage_start,
    seconds,
    trailing_count,
    trailing_mean,
    truncated,
    truncated_centred,
)


def ts(*stamps: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(stamps), utc=True))


T0 = "2025-03-04T10:00:00Z"


def at(offset_s: int) -> str:
    return (pd.Timestamp(T0) + pd.Timedelta(seconds=offset_s)).isoformat()


# --------------------------------------------------------------------------
# trailing counts


def test_window_is_closed_at_both_ends():
    events = Sorted.build(["X"] * 4, ts(at(-600), at(-599), at(0), at(1)))
    n = trailing_count(events, ["X"], ts(at(0)), 600)
    # -600 counts (the boundary), -599 counts, 0 counts (same second), +1 does not
    assert n.tolist() == [3.0]


def test_subject_is_removed_by_identity_not_by_time():
    """Two take-offs in the same second: the other one counts, the subject does not."""
    events = Sorted.build(["X", "X", "X"], ts(at(-300), at(0), at(0)))
    n = trailing_count(events, ["X", "X"], ts(at(0), at(0)), 600, subtract_self=True)
    # each subject sees the -300 event and the *other* same-second event
    assert n.tolist() == [2.0, 2.0]


def test_groups_do_not_see_each_other():
    events = Sorted.build(["X", "Y", "Y"], ts(at(-10), at(-10), at(-20)))
    n = trailing_count(events, ["X", "Y", "Z"], ts(at(0), at(0), at(0)), 600)
    assert n.tolist() == [1.0, 2.0, 0.0]


def test_an_unseen_group_counts_nothing_and_never_goes_negative():
    events = Sorted.build(["X"], ts(at(-10)))
    n = trailing_count(events, ["Q"], ts(at(0)), 600, subtract_self=True)
    assert n.tolist() == [0.0]


def test_counts_are_independent_of_event_order():
    a = Sorted.build(["X", "X", "X"], ts(at(-5), at(-500), at(-50)))
    b = Sorted.build(["X", "X", "X"], ts(at(-500), at(-50), at(-5)))
    anchors = ts(at(0), at(-49))
    assert (
        trailing_count(a, ["X", "X"], anchors, 100).tolist()
        == trailing_count(b, ["X", "X"], anchors, 100).tolist()
    )


def test_a_missing_timestamp_is_never_inside_a_window():
    events = Sorted.build(
        ["X", "X"], pd.Series([pd.Timestamp(at(-10), tz="UTC"), pd.NaT])
    )
    assert trailing_count(events, ["X"], ts(at(0)), 600).tolist() == [1.0]


# --------------------------------------------------------------------------
# centred counts and means


def test_centred_window_reaches_both_ways():
    events = Sorted.build(["X"] * 5, ts(at(-901), at(-900), at(0), at(900), at(901)))
    n = centred_count(events, ["X"], ts(at(0)), 900, subtract_self=True)
    assert n.tolist() == [2.0]  # -900 and +900 count; the subject at 0 is removed


def test_trailing_mean_divides_by_the_rows_that_carried_a_value():
    events = Sorted.build(
        ["X"] * 4,
        ts(at(-100), at(-200), at(-300), at(-400)),
        values={"v": [10.0, np.nan, 30.0, 1000.0]},
    )
    mean, n = trailing_mean(events, "v", ["X"], ts(at(0)), 350)
    assert n.tolist() == [2.0]
    assert mean.tolist() == [20.0]


def test_trailing_mean_is_nan_on_an_empty_window():
    events = Sorted.build(["X"], ts(at(-1000)), values={"v": [5.0]})
    mean, n = trailing_mean(events, "v", ["X"], ts(at(0)), 60)
    assert n.tolist() == [0.0] and np.isnan(mean[0])


# --------------------------------------------------------------------------
# coverage


def test_coverage_blocks_are_runs_of_consecutive_months():
    t = ts("2026-01-15T00:00:00Z", "2026-01-01T00:30:00Z", "2026-07-03T00:00:00Z")
    start = coverage_start(t)
    end = coverage_end(t)
    jan = seconds(ts("2026-01-01T00:00:00Z"))[0]
    jul = seconds(ts("2026-07-01T00:00:00Z"))[0]
    assert start.tolist() == [jan, jan, jul]
    feb = seconds(ts("2026-02-01T00:00:00Z"))[0]
    aug = seconds(ts("2026-08-01T00:00:00Z"))[0]
    assert end.tolist() == [feb, feb, aug]


def test_the_training_year_is_one_block():
    stamps = [f"2025-{m:02d}-15T12:00:00Z" for m in range(1, 13)]
    t = ts(*stamps)
    assert len(set(coverage_start(t).tolist())) == 1
    assert len(set(coverage_end(t).tolist())) == 1


def test_truncation_is_flagged_exactly_where_the_window_leaves_the_block():
    t = ts("2026-01-01T00:59:59Z", "2026-01-01T01:00:00Z", "2026-07-01T00:10:00Z")
    start = coverage_start(t)
    flags = truncated(t, 3600, start)
    # 00:59:59 minus an hour is before the block; 01:00:00 minus an hour is the
    # block's first instant, which is inside it
    assert flags.tolist() == [True, False, True]


def test_centred_truncation_looks_both_ways():
    t = ts("2026-01-31T23:50:00Z", "2026-01-15T12:00:00Z")
    start, end = coverage_start(t), coverage_end(t)
    assert truncated_centred(t, 900, start, end).tolist() == [True, False]


# --------------------------------------------------------------------------
# configuration


def test_configuration_is_the_set_in_use_and_how_long_it_has_held():
    stamps = [at(0), at(60), at(120), at(180), at(240), at(300)]
    des = ["09", "09", "27", "27", "27", "27"]
    cfg, since, n = configuration(["X"] * 6, ts(*stamps), des, min_movements=2)
    # one 09 is a stray; two make a runway; at 120 s a 27 appears and at 180 s
    # two of them make it a runway alongside 09, which is the change
    assert cfg.tolist() == ["", "09", "09", "09|27", "09|27", "09|27"]
    assert n.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert since.tolist()[3:] == [0.0, 60.0, 120.0]


def test_configuration_forgets_a_runway_the_window_no_longer_holds():
    stamps = [at(0), at(30), at(4000), at(4030)]
    des = ["09", "09", "27", "27"]
    cfg, since, _ = configuration(
        ["X"] * 4, ts(*stamps), des, window_s=3600, min_movements=2
    )
    assert cfg.tolist()[1] == "09"
    assert cfg.tolist()[3] == "27"
    assert since.tolist()[3] == 0.0


def test_configuration_is_per_group_and_ignores_missing_designators():
    stamps = [at(0), at(10), at(20), at(30)]
    cfg, _, _ = configuration(
        ["X", "X", "Y", "Y"], ts(*stamps), ["09", None, "27", "27"], min_movements=1
    )
    assert cfg.tolist() == ["09", "09", "27", "27"]


def test_since_is_capped():
    """A configuration that has held for ten hours reads as the cap, not ten hours."""
    stamps = [at(0), at(3000), at(6000), at(10 * 3600)]
    _, since, _ = configuration(
        ["X"] * 4,
        ts(*stamps),
        ["09", "09", "09", "09"],
        min_movements=1,
        max_since_s=7200,
    )
    assert since.tolist() == [0.0, 3000.0, 6000.0, 7200.0]


def test_a_runway_forgotten_by_the_window_is_a_change():
    """The set going empty is a change like any other, and is timed from there."""
    stamps = [at(0), at(10), at(10 * 3600)]
    cfg, since, _ = configuration(
        ["X"] * 3, ts(*stamps), ["09", "09", "09"], min_movements=2
    )
    assert cfg.tolist() == ["", "09", ""]
    assert since.tolist()[-1] == 0.0
