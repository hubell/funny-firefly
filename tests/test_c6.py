# SPDX-License-Identifier: GPL-3.0-only
"""C6 in ``taxiout.tails``: the sub-cells, C6a's fallback, C6b's level folding,
the rows it predicts, and what it may not read. Synthetic frames only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from taxiout import tails  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2025-03-01 10:00", tz="UTC")


def frame(lags_h, at, y, flight=None, stand=None, actype=None, icao="LIRF", lobt=None):
    n = len(lags_h)
    sched = pd.Series([T0] * n)
    mvt = sched + pd.to_timedelta(np.asarray(lags_h, dtype=float) * 3600, unit="s")
    block = np.where(at, sched, sched - pd.Timedelta(minutes=30))
    return pd.DataFrame(
        {
            "ADEP_mvt": [icao] * n,
            "LOBT_flt": pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
            if lobt is None
            else lobt,
            "SCHED_TIME_UTC_mvt": sched,
            "MVT_TIME_UTC_mvt": mvt,
            "BLOCK_TIME_UTC_mvt": pd.to_datetime(block, utc=True),
            "TAXITIME_SEC_mvt": np.asarray(y, dtype=float),
            "FLIGHT_mvt": flight if flight is not None else ["AZA123"] * n,
            "STAND_mvt": stand if stand is not None else ["A1"] * n,
            "AIRCRAFT_TYPE_mvt": actype if actype is not None else ["A320"] * n,
        }
    )


def test_sub_cell_bands_and_edges():
    f = frame([-1.0, 0.0, 0.01, 1.0, 1.5, 2.0, 3.0, 4.0], [False] * 8, [600] * 8)
    assert tails.c6_sub_cell(f).tolist() == [
        "LIRF|noflt:normal|[-1h,0]",
        "LIRF|noflt:normal|[-1h,0]",
        "LIRF|noflt:normal|(0,1h]",
        "LIRF|noflt:normal|(0,1h]",
        "LIRF|noflt:normal|(1h,2h]",
        "LIRF|noflt:normal|(1h,2h]",
        "LIRF|noflt:normal|(2h,4h]",
        "LIRF|noflt:normal|(2h,4h]",
    ]


def test_other_cells_are_not_split():
    f = frame([5.0, 13.0], [True, True], [18000, 46800])
    assert tails.c6_sub_cell(f).tolist() == ["LIRF|noflt:4-12h", "LIRF|noflt:>12h"]


def test_c6a_thin_subcell_takes_whole_cell():
    # (0,1h]: 6 rows, half at the schedule; (2h,4h]: 4 rows (< 5), all at it.
    lags = [0.5] * 6 + [3.0] * 4
    at = [True, True, True, False, False, False] + [True] * 4
    y = [1800] * 3 + [500, 700, 900] + [10800] * 4
    m = tails.C6a().fit(frame(lags, at, y))
    share, rest, n, fell = m.subs["LIRF|noflt:normal|(0,1h]"]
    assert (n, fell) == (6, False) and share == 0.5 and rest == 700
    whole = m.cells["LIRF|noflt:normal"]
    assert whole[0] == 0.7 and whole[1] == 700 and whole[2] == 10
    share, rest, n, fell = m.subs["LIRF|noflt:normal|(2h,4h]"]
    assert (n, fell) == (4, True) and (share, rest) == whole[:2]
    p = m.predict(frame([3.0], [False], [0]))
    assert np.isclose(p[0], 0.7 * 3 * 3600 + 0.3 * 700)


def test_c6a_five_rows_is_enough():
    lags = [3.0] * 5 + [0.5] * 5
    at = [True] * 5 + [False] * 5
    m = tails.C6a().fit(frame(lags, at, [10800] * 5 + [600] * 5))
    share, rest, n, fell = m.subs["LIRF|noflt:normal|(2h,4h]"]
    assert (n, fell, share) == (5, False, 1.0) and np.isnan(rest)
    p = m.predict(frame([3.0], [False], [0]))
    assert p[0] == 3 * 3600  # NaN rest under weight zero


def test_at_schedule_has_no_lag_condition():
    # at the schedule with a short lag counts, unlike the term's defaulted rows
    f = frame([0.5, 0.5], [True, False], [1800, 600])
    assert tails.at_schedule(f).tolist() == [True, False]
    assert tails.defaulted_block(f).fillna(False).tolist() == [False, False]


def test_c6b_level_folding():
    n = 30
    flight = ["AZA1"] * 10 + ["RYR2"] * 9 + ["EZY3"] * 1 + [None] * 10
    stand = ["S1"] * 12 + [None] * 9 + ["S2"] * 9
    f = frame([0.5] * n, [True, False] * 15, [600] * n, flight, stand, ["A320"] * n)
    raw = tails.c6_levels(f)
    assert raw["prefix"].tolist()[:20] == ["AZA"] * 10 + ["RYR"] * 9 + ["EZY"]
    assert (raw["prefix"].iloc[20:] == tails.C6_MISSING).all()
    folded, kept = tails.c6_fold_levels(raw["prefix"])
    assert kept == sorted(["AZA", tails.C6_MISSING])
    assert (folded.iloc[10:20] == tails.C6_OTHER).all()
    assert (folded.iloc[20:] == tails.C6_MISSING).all()
    folded, kept = tails.c6_fold_levels(raw["STAND_mvt"])
    assert kept == ["S1"]  # missing (9) and S2 (9) fall below ten
    assert (folded.iloc[12:] == tails.C6_OTHER).all()
    again, _ = tails.c6_fold_levels(pd.Series(["ZZZ", "AZA"]), ["AZA", tails.C6_MISSING])
    assert again.tolist() == [tails.C6_OTHER, "AZA"]


def test_c6b_features_use_fitted_levels():
    n = 40
    flight = ["AZA1"] * 20 + ["RYR2"] * 20
    f = frame([0.5] * 20 + [3.0] * 20, [False] * 20 + [True] * 20, [600] * n, flight)
    m = tails.C6b(tails.C6a().fit(f)).fit(f)
    assert m.kept["prefix"] == ["AZA", "RYR"]
    X = m.features(frame([0.5], [False], [0], ["XYZ9"], [None], ["B738"]))
    assert X["prefix"].iloc[0] == tails.C6_OTHER
    assert X["STAND_mvt"].iloc[0] == tails.C6_OTHER
    p = m.predict(f)
    assert p.shape == (n,) and np.isfinite(p).all()


def test_the_section_words_are_the_constants():
    """Section 2's numbers, as ``reports/preregistered-2026-10-04-tails.md`` gives them."""
    text = (REPO / "reports" / "preregistered-2026-10-04-tails.md").read_text(
        encoding="utf-8"
    )
    flat = " ".join(text.split())
    assert (
        "objective binary, 300 trees, learning rate 0.03, 15 leaves, minimum child 20"
        in flat
    )
    assert "no bagging, deterministic, column-wise, eight threads, seed 20260902" in flat
    assert "[−1 h, 0], (0, 1 h], (1 h, 2 h] and (2 h, 4 h]" in flat
    assert (
        "fewer than five fitting rows" in flat and "fewer than ten fitting rows" in flat
    )
    assert "within 60 s of its scheduled time" in flat
    p = tails.C6_CLASSIFIER_PARAMS
    assert (p["objective"], tails.C6_CLASSIFIER_ROUNDS, p["learning_rate"]) == (
        "binary",
        300,
        0.03,
    )
    assert (p["num_leaves"], p["min_child_samples"], p["num_threads"], p["seed"]) == (
        15,
        20,
        8,
        20260902,
    )
    assert (
        p["bagging_fraction"],
        p["bagging_freq"],
        p["deterministic"],
        p["force_col_wise"],
    ) == (
        1.0,
        0,
        True,
        True,
    )
    assert tails.C6_SUB_EDGES_H == ((-1, 0), (0, 1), (1, 2), (2, 4))
    assert (tails.C6_SUB_MIN, tails.C6_LEVEL_MIN, tails.NEAR_S) == (5, 10, 60.0)


# --------------------------------------------------------------------------
# the model on a year's worth of synthetic rows


def year(seed: int = 0) -> pd.DataFrame:
    """A covered LIRF ``noflt:normal`` cell, some ``flt:normal`` rows outside it."""
    rng = np.random.default_rng(seed)
    n = 400
    lags = rng.uniform(-1.0, 4.0, n)
    at = rng.random(n) < 0.4
    y = np.where(at, lags * 3600, rng.uniform(300, 1800, n))
    f = frame(
        lags,
        at,
        y,
        flight=[f"AZA{i % 7}" for i in range(n)],
        stand=[f"S{i % 5}" for i in range(n)],
    )
    lobt = pd.Series([T0] * 60)
    g = frame(rng.uniform(0, 1, 60), [False] * 60, rng.uniform(300, 900, 60), lobt=lobt)
    return pd.concat([f, g], ignore_index=True)


def fitted(train: pd.DataFrame) -> tails.C6Model:
    return tails.C6Model(tails.TailModel().fit(train)).fit(train)


def test_c6_predicts_exactly_the_terms_rows():
    train = year()
    model = fitted(train)
    got = model.predict(train)
    assert got.notna().equals(model.term.predict(train).notna())
    assert int(got.notna().sum()) == 400 and model.fitting_rows == 400


def test_poison_hidden_fields_are_not_read_at_prediction():
    """The ranking set hides the block time and the taxi-out: prediction must not read them."""
    train = year()
    model = fitted(train)
    test = year(seed=1)
    want = model.predict(test)
    poisoned = test.drop(columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
    pd.testing.assert_series_equal(model.predict(poisoned), want)


def test_poison_rows_outside_the_terms_cells_do_not_reach_the_fit():
    train = year()
    want = fitted(train).predict(train)
    poisoned = train.copy()
    outside = poisoned["LOBT_flt"].notna()
    poisoned.loc[outside, "TAXITIME_SEC_mvt"] = 9.9e9
    poisoned.loc[outside, "BLOCK_TIME_UTC_mvt"] = poisoned.loc[
        outside, "SCHED_TIME_UTC_mvt"
    ]
    pd.testing.assert_series_equal(fitted(poisoned).predict(train), want)


def test_poison_a_row_without_a_target_does_not_reach_the_fit():
    train = year()
    want = fitted(train).predict(train)
    extra = train.iloc[:1].copy()
    extra["TAXITIME_SEC_mvt"] = np.nan
    extra["BLOCK_TIME_UTC_mvt"] = extra["SCHED_TIME_UTC_mvt"]
    extra["STAND_mvt"] = "POISON"
    model = fitted(pd.concat([train, *[extra] * 20], ignore_index=True))
    assert model.fitting_rows == 400
    assert "POISON" not in model.c6b.kept["STAND_mvt"]
    pd.testing.assert_series_equal(model.predict(train), want)


def test_the_term_is_unchanged_by_fitting_c6():
    train = year()
    term = tails.TailModel().fit(train)
    before = term.cells.copy()
    tails.C6Model(term).fit(train)
    pd.testing.assert_frame_equal(term.cells, before)


def test_the_report_floors_and_rounds_half_even_as_the_file_does():
    sys.path.insert(0, str(REPO / "scripts"))
    import report_c6

    assert report_c6.written([-5.0, 2.5, 3.5, 10.4]).tolist() == [0.0, 2.0, 4.0, 10.0]


def test_unfitted_c6_refuses():
    with pytest.raises(RuntimeError):
        tails.C6Model(tails.TailModel().fit(year())).predict(year())
