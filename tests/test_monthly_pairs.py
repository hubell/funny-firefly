# SPDX-License-Identifier: GPL-3.0-only
"""The held-out pair folds and the plan-shift rule as evaluated, on constructed
data: no frame, no fit.

What is held to a test: that the stored pair folds give the per-pair sums and
the bootstrap intervals of the ladder's in-memory path, that a month is the
frame restricted to it, that the stored rows keep the frame's order, that the
ladder is restored even on an exception, and that the plan-shift rule
``scripts/report_planshift.py`` evaluates follows the pre-registration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import fit_monthly_pairs as fp  # noqa: E402
import report_planshift as ps  # noqa: E402
from monthly import rb  # noqa: E402
from taxiout.validation import (  # noqa: E402
    cluster_bootstrap,
    difference_in_differences,
    gap,
    held_out_pair_splits,
)


def synthetic(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    icao = pd.Series(rng.choice(["AAAA", "BBBB", "CCCC"], n))
    pair = icao + "|" + pd.Series(rng.integers(0, 60, n)).astype(str)
    y = pd.Series(rng.gamma(4.0, 200.0, n))
    y.iloc[:5] = 20000.0  # a few rows over 4 h
    month = np.array([f"2025-{m:02d}" for m in rng.integers(1, 13, n)])
    features = pd.DataFrame({"icao": icao, "pair": pair})
    dep = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float) + 1e6})
    flagged = pd.Series(rng.random(n) < 0.03)
    splits = held_out_pair_splits(features["icao"], features["pair"])
    results = {}
    for s in splits:
        results[("meta", s.name, "tail")] = {"mask": flagged}
        for k in fp.KEYS:
            base = y + rng.normal(0, 150 + 20 * fp.KEYS.index(k), n)
            for which, mask in (("valid", s.valid), ("seen", s.extra["seen"])):
                idx = y.index[mask.to_numpy()]
                results[(s.name, k, which)] = {
                    "pred": pd.Series(base.loc[idx].to_numpy(), index=idx)
                }
    return features, y, dep, month, splits, results


def stored(features, y, dep, month, splits, results):
    folds = {}
    for s in splits:
        frames = fp.stored_frames(s, results, features, y, dep, month)
        first = frames[fp.KEYS[0]]
        folds[s.name] = {
            "rows": first.drop(columns=["pred"]),
            "pred": {k: f["pred"].to_numpy() for k, f in frames.items()},
        }
    return folds


def same_sums(a, b):
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert list(fa) == list(fb)
        for side in fa:
            assert list(fa[side]) == list(fb[side])
            for k in fa[side]:
                np.testing.assert_array_equal(fa[side][k].n, fb[side][k].n)
                np.testing.assert_array_equal(fa[side][k].sse, fb[side][k].sse)


def test_stored_folds_reproduce_fold_sums_and_the_bootstrap():
    features, y, dep, month, splits, results = synthetic()
    folds = stored(features, y, dep, month, splits, results)
    orig = rb._fold_sums(results, features, y, splits, ex_flagged=True)
    mine = fp.fold_sums(folds)
    same_sums(orig, mine)
    a, b = rb.CLAIM
    assert str(cluster_bootstrap(orig, difference_in_differences(b, a))) == str(
        cluster_bootstrap(mine, difference_in_differences(b, a))
    )
    assert str(cluster_bootstrap(orig, gap(fp.KEYS[2]))) == str(
        cluster_bootstrap(mine, gap(fp.KEYS[2]))
    )
    for ap in ("AAAA", "BBBB", "CCCC"):
        same_sums(
            rb._fold_sums(results, features, y, splits, ex_flagged=True, airport=ap),
            fp.fold_sums(folds, airport=ap),
        )


def test_a_month_is_the_same_as_restricting_the_frame():
    features, y, dep, month, splits, results = synthetic()
    folds = stored(features, y, dep, month, splits, results)
    m = "2025-03"
    keep = {f: d["rows"]["month"].to_numpy() == m for f, d in folds.items()}
    mine = fp.fold_sums(folds, keep=keep)
    # the frame restricted to the month, by the ladder's own function
    rows = pd.Series(month == m, index=y.index)
    flagged = results[("meta", splits[0].name, "tail")]["mask"]
    res2 = dict(results)
    for s in splits:
        res2[("meta", s.name, "tail")] = {"mask": flagged | ~rows}
    orig = rb._fold_sums(res2, features, y, splits, ex_flagged=True)
    same_sums(orig, mine)


def test_stored_rows_keep_the_frame_order_and_sides():
    features, y, dep, month, splits, results = synthetic()
    folds = stored(features, y, dep, month, splits, results)
    for s in splits:
        rows = folds[s.name]["rows"]
        held = rows[rows["side"] == "held"]
        np.testing.assert_array_equal(
            held["mvt_id"].to_numpy(),
            dep.loc[s.valid.to_numpy(), "MVT_ID_mvt"].to_numpy(),
        )
        seen = rows[rows["side"] == "seen"]
        np.testing.assert_array_equal(
            seen["mvt_id"].to_numpy(),
            dep.loc[s.extra["seen"].to_numpy(), "MVT_ID_mvt"].to_numpy(),
        )


def test_ladder_of_three_restores_the_ladder():
    before = rb.LADDER
    with fp.ladder_of_three():
        assert tuple(r.key for r in rb.LADDER) == fp.KEYS
    assert rb.LADDER is before
    with pytest.raises(ValueError):
        with fp.ladder_of_three():
            raise ValueError("inside")
    assert rb.LADDER is before


def plan_frame():
    t = pd.Timestamp("2025-07-01 12:00", tz="UTC")
    h = pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "LFPG", "LIRF"],
            "MVT_TIME_UTC_mvt": [t] * 5,
            "EOBT_1_flt": [
                t - 24 * h,
                t - 5 * h,
                t - 30 * pd.Timedelta(minutes=1),
                t - 6 * h,
                pd.NaT,
            ],
            "LOBT_flt": [
                t - 24 * h,
                t - 3 * h,
                t - 30 * pd.Timedelta(minutes=1),
                t - 6 * h,
                t - 9 * h,
            ],
            "SCHED_TIME_UTC_mvt": [t - 24 * h, t - 5 * h, t - h, t - 6 * h, t - 9 * h],
        }
    )


def test_planshift_class_needs_both_lags_over_four_hours_and_no_term():
    f = plan_frame()
    cls = ps.planshift_class(f, [False] * 5)
    assert cls.tolist() == [True, False, False, True, False]
    assert ps.planshift_class(f, [True, False, False, False, False]).tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]


def test_planshift_rule_is_the_add_one_share_and_the_rest_mean():
    f = plan_frame()
    y = [87000.0, 900.0, 1000.0, 1200.0, 800.0]
    rule = ps.PlanShift().fit(f, [False] * 5, y)
    assert (rule.n, rule.long) == (2, 1)
    assert rule.share == pytest.approx(2 / 4)
    # rest mean: rows with LOBT present and y <= 4 h, by airport
    assert rule.rest_mean["LIRF"] == pytest.approx((900 + 1000 + 800) / 3)
    assert rule.rest_mean["LFPG"] == pytest.approx(1200.0)
    p = rule.predict(f, [False] * 5)
    assert p.iloc[0] == pytest.approx(0.5 * 24 * 3600 + 0.5 * rule.rest_mean["LIRF"])
    assert np.isnan(p.iloc[1]) and np.isnan(p.iloc[2]) and np.isnan(p.iloc[4])
    assert ps.apply_rule([1.0, 2.0], [np.nan, 5.0]).tolist() == [1.0, 5.0]
