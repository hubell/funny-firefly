# SPDX-License-Identifier: GPL-3.0-only
"""C1's block, ``taxiout.aobt3``, on constructed departures.

The registry half: ``AOBT_3_flt`` is read by this module and by no other
module's ``SOURCES``, the main registry still names it under no phase, and the
challenge module still excludes it by decision. The builder half: every count
against a brute-force count, the subject removed by identity, and the hidden
fields poisoned without moving anything.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import taxiout  # noqa: E402
from taxiout.aobt3 import (  # noqa: E402
    AOBT3_FEATURES,
    AOBT3_OWN,
    QUEUE_FEATURES,
    SOURCES,
    aobt3_features,
)
from taxiout.challenge import (  # noqa: E402
    EXCLUDED_BY_DECISION,
    HIDDEN_FOR_TEST_DEPARTURES,
    SOURCE_PHASES,
)
from taxiout.features import FEATURE_SOURCES, source_phase  # noqa: E402

T0 = pd.Timestamp("2025-03-01T10:00:00Z")


def _departures(rows):
    """(airport, runway, off-block minute or None, take-off minute) per row."""
    recs = []
    for i, (ap, rw, a, t) in enumerate(rows):
        recs.append(
            {
                "MVT_ID_mvt": float(i + 1),
                "ADEP_mvt": ap,
                "RUNWAY_mvt": rw,
                "AOBT_3_flt": pd.NaT if a is None else T0 + pd.Timedelta(minutes=a),
                "MVT_TIME_UTC_mvt": T0 + pd.Timedelta(minutes=t),
                "LOBT_flt": T0,
                "EOBT_1_flt": T0 + pd.Timedelta(minutes=5),
                "BLOCK_TIME_UTC_mvt": T0,
                "TAXITIME_SEC_mvt": 600.0,
            }
        )
    out = pd.DataFrame(recs)
    out["AOBT_3_flt"] = pd.to_datetime(out["AOBT_3_flt"], utc=True)
    return out


def _brute(dep: pd.DataFrame) -> pd.DataFrame:
    a = (dep["AOBT_3_flt"] - T0).dt.total_seconds().to_numpy()
    t = (dep["MVT_TIME_UTC_mvt"] - T0).dt.total_seconds().to_numpy()
    ap, rw = dep["ADEP_mvt"].to_numpy(), dep["RUNWAY_mvt"].to_numpy()
    out = {c: [] for c in QUEUE_FEATURES}
    for i in range(len(dep)):
        if not np.isfinite(a[i]):
            for c in out:
                out[c].append(np.nan)
            continue
        others = [
            j for j in range(len(dep)) if j != i and np.isfinite(a[j]) and ap[j] == ap[i]
        ]
        same = [j for j in others if rw[j] == rw[i]]
        out["q_surface_off"].append(sum(a[j] <= a[i] < t[j] for j in others))
        out["q_surface_off_rwy"].append(sum(a[j] <= a[i] < t[j] for j in same))
        out["q_takeoffs_during"].append(sum(a[i] <= t[j] < t[i] for j in others))
        out["q_takeoffs_during_rwy"].append(sum(a[i] <= t[j] < t[i] for j in same))
        out["q_overtakes_rwy"].append(sum(a[j] > a[i] and t[j] < t[i] for j in same))
    return pd.DataFrame(out, dtype=float)


def test_every_count_is_the_brute_force_count():
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(80):
        a = int(rng.integers(0, 60))
        rows.append(
            (
                str(rng.choice(["EDDF", "LSZH"])),
                str(rng.choice(["25C", "07R"])),
                None if rng.random() < 0.15 else a,
                a + int(rng.integers(-3, 40)),
            )
        )
    dep = _departures(rows)
    got = aobt3_features(dep)
    want = _brute(dep)
    for c in want.columns:
        np.testing.assert_array_equal(got[c].to_numpy(), want[c].to_numpy(), err_msg=c)


def test_the_differences_and_the_missing_flag():
    dep = _departures([("EDDF", "25C", 3, 20), ("EDDF", "25C", None, 22)])
    got = aobt3_features(dep)
    assert got.loc[0, "aobt3_lag"] == 17 * 60
    assert got.loc[0, "aobt3_vs_lobt"] == 3 * 60
    assert got.loc[0, "aobt3_vs_eobt"] == -2 * 60
    assert got.loc[0, "aobt3_missing"] == 0.0
    assert got.loc[1, "aobt3_missing"] == 1.0
    assert got.loc[1, list(AOBT3_OWN[:3]) + list(QUEUE_FEATURES)].isna().all()


def test_a_departure_without_aobt_is_not_one_of_the_others():
    with_it = _departures([("EDDF", "25C", 0, 30), ("EDDF", "25C", 5, 10)])
    without = _departures([("EDDF", "25C", 0, 30), ("EDDF", "25C", None, 10)])
    assert aobt3_features(with_it).loc[0, "q_takeoffs_during"] == 1
    assert aobt3_features(without).loc[0, "q_takeoffs_during"] == 0


def test_poisoning_the_hidden_fields_moves_nothing():
    dep = _departures([("EDDF", "25C", i, i + 12) for i in range(10)])
    before = aobt3_features(dep)
    poisoned = dep.copy()
    poisoned["BLOCK_TIME_UTC_mvt"] = T0 + pd.Timedelta(hours=7)
    poisoned["TAXITIME_SEC_mvt"] = 99999.0
    pd.testing.assert_frame_equal(before, aobt3_features(poisoned))


def test_the_sources_name_no_hidden_field_and_known_phases():
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (None, *SOURCE_PHASES), (name, src)
            assert column not in HIDDEN_FOR_TEST_DEPARTURES, (name, src)
    assert set(SOURCES) == set(AOBT3_FEATURES)


def test_only_this_module_reads_the_excluded_field():
    """``AOBT_3_flt`` stays excluded by decision everywhere but C1's module."""
    assert "AOBT_3_flt" in EXCLUDED_BY_DECISION
    for sources in FEATURE_SOURCES.values():
        assert all(source_phase(s)[1] != "AOBT_3_flt" for s in sources)
    readers = []
    for info in pkgutil.iter_modules(taxiout.__path__):
        module = importlib.import_module(f"taxiout.{info.name}")
        for attr in ("SOURCES", "TFREE_SOURCES", "DIFF_SOURCES"):
            got = getattr(module, attr, None)
            if not isinstance(got, dict):
                continue
            if any(source_phase(s)[1] == "AOBT_3_flt" for v in got.values() for s in v):
                readers.append(info.name)
    assert readers == ["aobt3"]
