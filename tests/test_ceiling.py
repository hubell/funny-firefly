# SPDX-License-Identifier: GPL-3.0-only
"""The ceiling diagnostic, on hand-built residual sequences."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.ceiling import (  # noqa: E402
    MIN_ROWS,
    lag1_autocorrelation,
    neighbour_mean,
    oracle,
)

T0 = pd.Timestamp("2025-12-01T10:00:00Z")


def _seq(residuals, gaps_min, icao="ZZZZ", runway="27", start=T0):
    """Residuals at consecutive take-offs separated by ``gaps_min``."""
    t, times = start, []
    for g in [0.0, *gaps_min]:
        t = t + pd.Timedelta(minutes=g)
        times.append(t)
    return pd.DataFrame(
        {
            "icao": icao,
            "runway": runway,
            "takeoff": times,
            "residual": list(residuals),
        }
    )


def test_lag1_autocorrelation_is_one_for_a_copied_sequence_and_bucketed_by_gap():
    # each residual is the one before it plus one: an exact linear copy, at
    # ninety-second gaps, so the 0-2 min bucket holds every pair at correlation one
    exact = _seq(np.arange(MIN_ROWS + 1, dtype=float), [1.5] * MIN_ROWS)
    out = lag1_autocorrelation(exact)
    assert list(out["bucket"]) == ["0-2 min", "2-5 min", "5-15 min"]
    assert out.loc[out["bucket"] == "0-2 min", "n"].iloc[0] == MIN_ROWS
    assert out.loc[out["bucket"] == "0-2 min", "corr"].iloc[0] == pytest.approx(1.0)
    assert out.loc[out["bucket"] == "2-5 min", "n"].iloc[0] == 0
    assert np.isnan(out.loc[out["bucket"] == "2-5 min", "corr"].iloc[0])
    # gaps of three and ten minutes land in the other two buckets; twenty in none
    spread = _seq(np.arange(4, dtype=float), [3.0, 10.0, 20.0])
    by = lag1_autocorrelation(spread).set_index("bucket")["n"]
    assert by.to_dict() == {"0-2 min": 0, "2-5 min": 1, "5-15 min": 1}


def test_pairs_do_not_chain_across_runways_or_airports():
    r = np.arange(MIN_ROWS + 1, dtype=float)
    one = _seq(r, [1.5] * MIN_ROWS, runway="27")
    other = _seq(r, [1.5] * MIN_ROWS, runway="09", start=T0 + pd.Timedelta(seconds=45))
    out = lag1_autocorrelation(pd.concat([one, other], ignore_index=True))
    # interleaved in time but on different runways: pairs count per runway only
    assert out.loc[out["bucket"] == "0-2 min", "n"].iloc[0] == 2 * MIN_ROWS
    by = lag1_autocorrelation(
        pd.concat([one, other.assign(icao="YYYY")], ignore_index=True), by_airport=True
    )
    assert set(by["icao"]) == {"YYYY", "ZZZZ"}


def test_neighbour_mean_excludes_the_subject_and_the_far():
    res = _seq([10.0, 20.0, 30.0, 40.0], [5, 5, 20])  # 10:00, 10:05, 10:10, 10:30
    m = neighbour_mean(res)
    assert m[0] == pytest.approx((20 + 30) / 2)  # 10:05 and 10:10 within 15 min of 10:00
    assert m[1] == pytest.approx((10 + 30) / 2)
    assert np.isnan(m[3])  # 10:30 has nothing within fifteen minutes


def test_the_oracle_removes_a_shared_shock_and_leaves_noise():
    # blocks of ten departures a minute apart, an hour between blocks, each
    # block late together by one shock: a subject's neighbours are exactly its
    # block-mates, so the neighbours' mean is the shock plus a little noise
    rng = np.random.default_rng(1)
    blocks, size = 40, 10
    frames = []
    for b in range(blocks):
        shock = rng.normal(scale=100.0)
        noise = rng.normal(scale=10.0, size=size)
        frames.append(
            _seq(shock + noise, [1.0] * (size - 1), start=T0 + pd.Timedelta(hours=b))
        )
    res = pd.concat(frames, ignore_index=True)
    o = oracle(res)
    assert o["n"] == blocks * size and o["with_neighbour"] == 1.0
    assert o["slope"] == pytest.approx(1.0, abs=0.1)
    assert o["rmse_before"] > 60.0
    assert o["rmse_after"] < 0.2 * o["rmse_before"]
    assert list(o["per_airport"]["icao"]) == ["ZZZZ"]
    assert o["per_airport"]["rmse_after"].iloc[0] == pytest.approx(o["rmse_after"])


def test_the_oracle_leaves_independent_residuals_alone():
    rng = np.random.default_rng(2)
    n = 3000
    res = _seq(rng.normal(scale=50.0, size=n), [1.0] * (n - 1))
    o = oracle(res)
    assert abs(o["slope"]) < 0.3
    assert o["rmse_after"] == pytest.approx(o["rmse_before"], rel=0.03)


def test_residuals_need_their_columns():
    with pytest.raises(KeyError):
        oracle(pd.DataFrame({"icao": ["Z"], "residual": [1.0]}))
