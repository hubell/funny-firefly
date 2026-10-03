# SPDX-License-Identifier: GPL-3.0-only
"""The pre-registered experiments' arithmetic, on synthetic frames.

Nothing here fits a booster or reads the challenge data. What is held to a
test is the part of ``scripts/preregistered.py`` that decides things: the
weight schedule, the screen, the adoption verdict, the reproduction check and
the report's refusal to publish an identifier. Those are the pieces where a
mistake would not announce itself, because a wrong verdict computed from right
figures reads exactly like a right one.

The figures below are constructed to make each rule's boundary visible, and
none of them is a measurement of anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import preregistered as pr  # noqa: E402


# --------------------------------------------------------------------------
# the weight schedule


def test_the_schedule_halves_at_every_half_life():
    """1, 0.5 and 0.25 at ages 0, H and 2H, whatever H is."""
    month = pd.Series(["2025-06", "2025-03", "2024-12", "2025-05"])
    rows = pd.Series([True, True, True, True])
    weights = pr.recency_weights(month, rows, "2025-06", 3)
    assert np.allclose(weights, [1.0, 0.5, 0.25, 0.5 ** (1 / 3)])

    weights = pr.recency_weights(month, rows, "2025-06", 12)
    assert np.allclose(weights[:3], [1.0, 0.5 ** (3 / 12), 0.5 ** (6 / 12)])


def test_the_age_is_whole_calendar_months_across_the_year_boundary():
    month = pd.Series(["2025-01", "2024-12", "2024-11"])
    rows = pd.Series([True, True, True])
    weights = pr.recency_weights(month, rows, "2025-01", 1)
    assert np.allclose(weights, [1.0, 0.5, 0.25])


def test_the_schedule_only_weights_the_fitting_rows():
    month = pd.Series(["2025-06", "2025-05", "2025-04"])
    rows = pd.Series([True, False, True])
    assert len(pr.recency_weights(month, rows, "2025-06", 3)) == 2


def test_a_row_later_than_the_last_training_month_is_refused():
    """A negative age would weight a row above one, which is not a schedule."""
    month = pd.Series(["2025-07"])
    rows = pd.Series([True])
    with pytest.raises(RuntimeError, match="later than"):
        pr.recency_weights(month, rows, "2025-06", 3)


# --------------------------------------------------------------------------
# the reading rules


def test_no_worse_is_read_at_the_printed_precision():
    assert pr.no_worse(217.64, 217.6)  # both print 217.6: a tie
    assert pr.no_worse(217.1, 217.6)
    assert not pr.no_worse(217.66, 217.6)  # prints 217.7: worse
    assert pr.PRINTED_PRECISION_S == 0.1


def test_clearing_the_floor_is_read_on_the_unrounded_figure():
    assert pr.clears_floor(217.20, 217.60, 0.34)
    assert not pr.clears_floor(217.30, 217.60, 0.34)
    # A gain exactly equal to the floor does not clear it: the rule says the
    # gain must exceed the floor. The boundary is written in halves because a
    # decimal like 0.34 is not exact in binary, which `baseline.md` says of
    # this same test -- "a change printed equal to its floor can sit on either
    # side of it" -- and a test written in decimals would be testing the
    # rounding rather than the rule.
    assert not pr.clears_floor(217.5, 218.0, 0.5)
    assert pr.clears_floor(217.25, 218.0, 0.5)


def test_the_floor_is_root_two_times_the_sample_standard_deviation():
    singles = [217.59, 217.71, 217.83, 217.95, 218.11]
    expected = float(np.sqrt(2.0) * np.std(np.asarray(singles), ddof=1))
    assert pr.floor_of(singles) == pytest.approx(expected)
    # And not the population standard deviation, which would read lower.
    assert pr.floor_of(singles) > np.sqrt(2.0) * float(np.std(np.asarray(singles)))


# --------------------------------------------------------------------------
# the reproduction check


def _kept(residuals, ids=None) -> pd.DataFrame:
    n = len(residuals)
    ids = list(range(1, n + 1)) if ids is None else ids
    return pd.DataFrame(
        {
            "mvt_id": [float(i) for i in ids],
            "icao": ["EDDF"] * n,
            "takeoff": pd.to_datetime(["2025-12-01"] * n, utc=True),
            "flagged": [False] * n,
            "y": [1000.0] * n,
            "pred": [1000.0 - r for r in residuals],
        }
    )


def _stored(tmp_path: Path, residuals, ids=None) -> Path:
    n = len(residuals)
    ids = list(range(1, n + 1)) if ids is None else ids
    path = tmp_path / "stored.parquet"
    pd.DataFrame(
        {"mvt_id": [float(i) for i in ids], "residual": [float(r) for r in residuals]}
    ).to_parquet(path, index=False)
    return path


def test_the_reproduction_check_passes_only_on_an_exact_match(tmp_path):
    residuals = [12.0, -3.5, 0.0, 400.25]
    got, ok, rows = pr.reproduces(_kept(residuals), _stored(tmp_path, residuals))
    assert ok and rows == 4
    assert got.startswith("0 s")


def test_the_reproduction_check_refuses_a_nonzero_difference(tmp_path):
    """One row off by a millionth of a second is still a difference."""
    stored = _stored(tmp_path, [12.0, -3.5, 0.0, 400.25])
    _, ok, _ = pr.reproduces(_kept([12.0, -3.5, 1e-6, 400.25]), stored)
    assert not ok


def test_the_reproduction_check_refuses_a_row_count_that_disagrees(tmp_path):
    stored = _stored(tmp_path, [12.0, -3.5, 0.0])
    got, ok, _ = pr.reproduces(_kept([12.0, -3.5]), stored)
    assert not ok and "rows" in got


def test_the_reproduction_check_refuses_identifiers_that_do_not_join(tmp_path):
    """Right count, right values, wrong rows: the join is the check."""
    stored = _stored(tmp_path, [1.0, 2.0, 3.0], ids=[10, 20, 30])
    got, ok, _ = pr.reproduces(_kept([1.0, 2.0, 3.0], ids=[10, 20, 40]), stored)
    assert not ok and "joined" in got


def test_the_reproduction_check_refuses_a_file_that_is_not_there(tmp_path):
    _, ok, _ = pr.reproduces(_kept([1.0]), tmp_path / "absent.parquet")
    assert not ok


# --------------------------------------------------------------------------
# the screen and the adoption verdict


def _record(leg: str, schedule: str, pooled: float, **extra) -> dict:
    entry = {
        "rows": 1000,
        "flagged": 0,
        "pooled_ex_flagged": pooled,
        "per_airport": {},
        "three": extra.get("three"),
        "ranking_mix": extra.get("ranking_mix"),
        "sha256": extra.get("sha256", "a" * 64),
    }
    sets = {"valid": entry}
    if "gap" in extra:
        sets["gap"] = dict(
            entry, pooled_ex_flagged=extra["gap"], three=None, ranking_mix=None
        )
    got = {
        "leg": leg,
        "seed": 0,
        "schedule": schedule,
        "features": "submission",
        "wall_s": 1.0,
        "peak_rss_kib": 1,
        "sets": sets,
    }
    if schedule != "none":
        got["model_sha256"] = extra.get("model_sha256", "b" * 64)
    return got


def _world(**moves) -> dict:
    """The unweighted references, and whatever the schedules did to them."""
    records = {
        "october_seed0": _record("october", "none", 220.00),
        "july_seed0": _record("july", "none", 325.81, three=237.80, gap=300.00),
        "december_seed0": _record("december", "none", 217.60, ranking_mix=218.60),
        "october_H12_seed0": _record("october", "H=12", moves.get("oct12", 220.00)),
        "october_H12_seed0_repeat": _record(
            "october", "H=12", moves.get("oct12", 220.00)
        ),
        "october_H3_seed0": _record("october", "H=3", moves.get("oct3", 220.00)),
    }
    for h in (3, 12):
        if f"gap{h}" in moves:
            records[f"july_H{h}_seed0"] = _record(
                "july",
                f"H={h}",
                moves.get(f"julyall{h}", 325.81),
                three=moves.get(f"three{h}", 237.80),
                gap=moves[f"gap{h}"],
            )
        if f"dec{h}" in moves:
            records[f"december_H{h}_seed0"] = _record(
                "december",
                f"H={h}",
                moves[f"dec{h}"],
                ranking_mix=moves.get(f"mix{h}", 218.60),
            )
    return records


FLOORS = {"October, all ten": 0.30, "the gap": 0.40}


def test_the_screen_stops_a_schedule_that_is_worse_and_passes_one_that_ties():
    """220.04 prints 220.0 and is a tie; 220.06 prints 220.1 and is worse."""
    got = pr.experiment_2(
        _world(
            oct12=220.04,
            oct3=220.06,
            gap12=300.00,
            dec12=217.60,
            mix12=218.60,
            three12=237.80,
        ),
        pr.Checks(),
        FLOORS,
    )
    assert got["schedules"]["H=12"]["october_passes"]
    assert not got["schedules"]["H=3"]["october_passes"]
    assert got["schedules"]["H=3"]["stopped_at"] == "the October screen"
    # And a schedule that stopped on October is never read on the gap.
    assert "gap" not in got["schedules"]["H=3"]


def test_a_schedule_worse_on_the_gap_stops_there_and_cannot_be_adopted():
    got = pr.experiment_2(
        _world(oct12=219.50, gap12=300.40, julyall12=325.00, three12=236.00),
        pr.Checks(),
        FLOORS,
    )
    entry = got["schedules"]["H=12"]
    assert entry["stopped_at"] == "the gap screen"
    assert entry["adopted"] is False
    # Its July readings come from the same fit and are printed anyway.
    assert entry["july_three"] == pytest.approx(236.00)
    assert got["adopted"] is None


def test_a_schedule_no_worse_everywhere_and_clearing_a_floor_is_adopted():
    got = pr.experiment_2(
        _world(
            oct12=219.00,
            gap12=299.00,
            julyall12=325.00,
            three12=236.00,
            dec12=217.00,
            mix12=218.00,
        ),
        pr.Checks(),
        FLOORS,
    )
    entry = got["schedules"]["H=12"]
    assert entry["worse_on"] == []
    assert entry["clears_floor_on"] == ["December all ten", "July the three"]
    assert entry["adopted"] is True
    assert got["adopted"] == "H=12"


def test_a_schedule_that_moves_nothing_outside_seed_noise_is_not_adopted():
    """No worse on all five, and no gain clearing a floor: not adopted."""
    got = pr.experiment_2(
        _world(
            oct12=220.00,
            gap12=300.00,
            julyall12=325.81,
            three12=237.75,
            dec12=217.55,
            mix12=218.60,
        ),
        pr.Checks(),
        FLOORS,
    )
    entry = got["schedules"]["H=12"]
    assert entry["worse_on"] == []
    assert entry["clears_floor_on"] == []
    assert entry["adopted"] is False
    assert "seed noise" in entry["why"]


def test_one_reading_worse_at_printed_precision_blocks_adoption():
    """Better on four readings and worse on the mix is still not adopted."""
    got = pr.experiment_2(
        _world(
            oct12=219.00,
            gap12=299.00,
            julyall12=325.00,
            three12=236.00,
            dec12=217.00,
            mix12=218.70,
        ),
        pr.Checks(),
        FLOORS,
    )
    entry = got["schedules"]["H=12"]
    assert entry["worse_on"] == ["the ranking mix"]
    assert entry["adopted"] is False


def test_the_mild_schedule_wins_a_tie_unless_the_strong_one_clears_every_floor():
    both = dict(
        oct12=219.00,
        gap12=299.00,
        julyall12=325.00,
        three12=236.00,
        dec12=217.00,
        mix12=218.00,
        oct3=218.90,
        gap3=298.90,
        julyall3=324.90,
        three3=235.90,
        dec3=216.90,
        mix3=217.90,
    )
    got = pr.experiment_2(_world(**both), pr.Checks(), FLOORS)
    assert got["schedules"]["H=3"]["adopted"] and got["schedules"]["H=12"]["adopted"]
    # The strong one is ahead everywhere, but by less than the floors.
    assert got["adopted"] == "H=12"

    clear = dict(
        both,
        oct3=217.00,
        gap3=297.00,
        julyall3=323.00,
        three3=233.00,
        dec3=215.00,
        mix3=216.00,
    )
    got = pr.experiment_2(_world(**clear), pr.Checks(), FLOORS)
    assert got["adopted"] == "H=3"


def test_the_digest_check_stops_the_experiment_before_anything_is_read():
    records = _world(oct12=219.00, oct3=219.00)
    records["october_H12_seed0_repeat"]["sets"]["valid"]["sha256"] = "c" * 64
    got = pr.experiment_2(records, pr.Checks(), FLOORS)
    assert got["stopped"] == "the digest check"
    assert got["schedules"] == {}


def test_the_digest_check_needs_the_model_dumps_too():
    records = _world(oct12=219.00, oct3=219.00)
    records["october_H12_seed0_repeat"]["model_sha256"] = "d" * 64
    got = pr.experiment_2(records, pr.Checks(), FLOORS)
    assert got["stopped"] == "the digest check"


# --------------------------------------------------------------------------
# Experiment 1's guard


def _seed_world(july_three, october, gap):
    records, kept = {}, {}
    for i, (t, o, g) in enumerate(zip(july_three, october, gap)):
        records[f"july_seed{i}"] = _record(
            "july", "none", 325.0 + i * 0.1, three=t, gap=g
        )
        records[f"october_seed{i}"] = _record("october", "none", o)
        kept[f"july_seed{i}"] = {
            "valid": _kept([0.0] * 6),
            "gap": _kept([0.0] * 6),
        }
        kept[f"october_seed{i}"] = {"valid": _kept([0.0] * 6)}
    return records, kept


def test_the_guard_trips_where_the_average_regresses_past_the_floor(monkeypatch):
    """The average is constructed, so only the guard's arithmetic is tested."""
    records, kept = _seed_world(
        july_three=[237.0, 237.2, 237.4, 237.6, 237.8],
        october=[220.0, 220.1, 220.2, 220.3, 220.4],
        gap=[300.0, 300.1, 300.2, 300.3, 300.4],
    )
    monkeypatch.setattr(
        pr,
        "score",
        lambda frame, three=False: {"three": 240.0, "pooled": 221.0}[
            "three" if three else "pooled"
        ],
    )
    got = pr.experiment_1(records, kept, pr.Checks())
    # The three: 240.0 against the ladder's 237.0 is +3.0, past a floor of 1.02.
    assert got["verdict"]["the three"]["guard_trips"]
    assert not got["verdict"]["the three"]["beats_every_single"]
    assert "the three" in got["guard_trips_on"]


def test_the_guard_holds_where_the_average_gains(monkeypatch):
    records, kept = _seed_world(
        july_three=[237.0, 237.2, 237.4, 237.6, 237.8],
        october=[220.0, 220.1, 220.2, 220.3, 220.4],
        gap=[300.0, 300.1, 300.2, 300.3, 300.4],
    )
    monkeypatch.setattr(
        pr,
        "score",
        lambda frame, three=False: 236.0 if three else 219.0,
    )
    got = pr.experiment_1(records, kept, pr.Checks())
    assert got["guard_trips_on"] == []
    assert got["verdict"]["the three"]["beats_every_single"]
    assert got["verdict"]["October"]["beats_every_single"]
    # October's floor is measured from its own five singles, by the same rule.
    assert got["floors"]["October, all ten"] == pytest.approx(
        pr.floor_of([220.0, 220.1, 220.2, 220.3, 220.4])
    )


# --------------------------------------------------------------------------
# what the report may not carry


def test_the_report_refuses_a_stand_identifier():
    text = "The worst cell is stand A12 at EDDF, which is not publishable."
    with pytest.raises(SystemExit, match="stand identifier"):
        pr.check_no_identifiers(text, {"A12", "B7"}, set())


def test_the_report_refuses_a_movement_identifier():
    text = "The largest residual is on movement 900000001, at EDDF."
    with pytest.raises(SystemExit, match="movement identifier"):
        pr.check_no_identifiers(text, set(), {"900000001"})


def test_the_report_allows_an_airport_code_and_a_figure():
    """ICAO codes are public, and a number that is not an identifier is a figure."""
    text = "EDDF, EGLL and EHAM pooled at 237.8 s over 20,790 rows in 1,234 s."
    pr.check_no_identifiers(text, {"EDDF", "A12"}, {"900000001"})


def test_a_movement_identifier_inside_a_digest_is_not_a_hit():
    """The word boundaries are the test, so a hex digest cannot trip it."""
    text = "digest `f3900000001ab...`, over 12 rows"
    pr.check_no_identifiers(text, set(), {"900000001"})


def test_the_pre_registration_carried_here_names_no_session_and_no_note():
    """What this repository publishes must not name a chat, a session or a note.

    The criteria are quoted from a working note that does name both. The
    substitutions made on the way in are what this asserts: no numbered work
    block, no note file, and nothing the licence gate's own attribution
    pattern matches. That pattern is imported rather than restated, so this
    test carries none of the words it is looking for and cannot drift from
    the gate.
    """

    import check_licence_hygiene as hygiene

    body = pr.PREREGISTRATION
    assert "Block 0" not in body and "Block 1" not in body
    assert "2026-09-19.md" not in body and "2026-09-20.md" not in body
    assert not hygiene.attribution_in(body)
    # And the same over the whole report, which is what gets published.
    assert not hygiene.attribution_in(pr.__doc__)


def test_the_criteria_are_constants_and_not_computed():
    """Every reference and floor the report reads against is a fixed number."""
    assert (pr.REF_DECEMBER_ALL, pr.REF_JULY_THREE, pr.REF_RANKING_MIX) == (
        217.6,
        237.8,
        218.6,
    )
    assert (pr.FLOOR_DECEMBER_ALL, pr.FLOOR_JULY_ALL, pr.FLOOR_JULY_THREE) == (
        0.34,
        0.23,
        1.02,
    )
    assert pr.HALF_LIVES == (3, 12)
    assert pr.SEASON_COLUMNS == ("tko_month", "tko_doy_sin", "tko_doy_cos")
    assert pr.JULY_AVERAGE_POOLED == 325.29
    # The rule of 19 September reads two forward legs this run does not fit
    # for it, so its figures are quoted with their lines and are constants on
    # the same footing as the references and the floors above.
    legs = {r["leg"]: r for r in pr.RULE_LEGS}
    assert set(legs) == {"December, all ten", "July, all ten"}
    assert (legs["December, all ten"]["gain"], legs["December, all ten"]["floor"]) == (
        0.48,
        0.34,
    )
    assert legs["December, all ten"]["average"] == 217.11
    assert legs["December, all ten"]["best_single"] == 217.59
    assert (legs["July, all ten"]["gain"], legs["July, all ten"]["floor"]) == (
        0.52,
        0.23,
    )
    assert legs["July, all ten"]["average"] == pr.JULY_AVERAGE_POOLED
    assert legs["July, all ten"]["best_single"] == pr.JULY_SINGLES_POOLED[0]
    for leg in pr.RULE_LEGS:
        assert "baseline.md line" in leg["source"]
        # Each leg's floor is the one this repository already prints for it.
        assert leg["floor"] in (pr.FLOOR_DECEMBER_ALL, pr.FLOOR_JULY_ALL)


# --------------------------------------------------------------------------
# the paired interval


def _two_arm_frames(n_days=12, shift=0.0):
    days = np.repeat(
        pd.to_datetime([f"2025-12-{d + 1:02d}" for d in range(n_days)], utc=True), 30
    )
    rng = np.random.default_rng(3)
    y = rng.normal(1000.0, 200.0, size=len(days))
    base = pd.DataFrame(
        {
            "mvt_id": np.arange(len(days), dtype=float),
            "icao": ["EDDF"] * len(days),
            "takeoff": days,
            "flagged": [False] * len(days),
            "y": y,
            "pred": y - rng.normal(0.0, 100.0, size=len(days)),
        }
    )
    arm = base.copy()
    arm["pred"] = base["pred"] - shift
    return arm, base


def test_the_interval_is_paired_and_brackets_the_point():
    arm, base = _two_arm_frames(shift=5.0)
    got = pr.paired_interval(arm, base)
    assert got is not None
    assert got.low <= got.point <= got.high
    assert got.draws == pr.BOOTSTRAP_DRAWS


def test_an_arm_identical_to_its_base_moves_nothing():
    arm, base = _two_arm_frames(shift=0.0)
    got = pr.paired_interval(arm, base)
    assert got.point == pytest.approx(0.0, abs=1e-9)
    assert got.low == pytest.approx(0.0, abs=1e-9)
    assert got.high == pytest.approx(0.0, abs=1e-9)


def test_only_a_step_inside_a_second_carries_an_interval():
    arm, base = _two_arm_frames(shift=5.0)
    assert pr.maybe_interval(arm, base, 4.2) == "not inside a second"
    assert pr.maybe_interval(arm, base, 0.4).startswith("[")
    assert pr.INTERVAL_BELOW_S == 1.0


def test_the_mix_interval_draws_the_same_days_for_every_bucket():
    arm, base = _two_arm_frames(shift=3.0)
    arm["bucket"] = np.resize(np.asarray(["0", "1-10", "11-100", ">100"]), len(arm))
    base["bucket"] = arm["bucket"]
    got = pr.mix_interval(arm, base)
    assert got is not None and got.low <= got.point <= got.high
    same = pr.mix_interval(arm, base)
    # The draw is seeded, so the interval is the same figure twice.
    assert (same.low, same.high) == (got.low, got.high)


# --------------------------------------------------------------------------
# the report assembles from the records alone


def _summary(tmp_path: Path) -> dict:
    records = _world(
        oct12=219.00,
        gap12=299.00,
        julyall12=325.00,
        three12=236.00,
        dec12=217.00,
        mix12=218.00,
        oct3=221.00,
    )
    for i in range(1, pr.SEEDS):
        records[f"july_seed{i}"] = _record(
            "july", "none", 325.81 + i * 0.1, three=237.8 + i * 0.2, gap=300.0 + i * 0.1
        )
        records[f"october_seed{i}"] = _record("october", "none", 220.0 + i * 0.1)
    records["december_noseason_seed0"] = _record(
        "december", "none", 218.90, ranking_mix=219.90
    )
    records["july_noseason_seed0"] = _record("july", "none", 327.00, three=239.00)
    for name, record in records.items():
        record["name"] = name
        record["features"] = "no-season" if "noseason" in name else "submission"
    return records


def test_the_report_builds_from_the_records_and_says_nothing_shipped(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pr, "WORK", tmp_path)
    records = _summary(tmp_path)
    # Every frame the interval sections read, written where --report finds them.
    arm, base = _two_arm_frames(shift=0.3)
    arm["bucket"] = np.resize(np.asarray(["0", "1-10", "11-100", ">100"]), len(arm))
    base["bucket"] = arm["bucket"]
    for name in records:
        for set_name in records[name]["sets"]:
            frame = arm if name != "december_seed0" else base
            frame.to_parquet(tmp_path / f"{name}__{set_name}.parquet", index=False)

    checks = pr.Checks()
    summary = {
        "checks": checks.rows,
        "stopped": None,
        "records": records,
        "experiment_1": pr.experiment_1(
            records,
            {n: {s: arm for s in records[n]["sets"]} for n in records},
            checks,
        ),
        "experiment_2": pr.experiment_2(records, checks, dict(FLOORS)),
        "experiment_3": pr.experiment_3(records, checks),
        "fits": len(records),
        "fit_seconds": float(sum(r["wall_s"] for r in records.values())),
        "frame": {
            "build_seconds": 56.0,
            "rows": 2085047,
            "legs": {
                "december": {"train": 1, "valid": 2, "fit_rows": 3, "flagged_train": 4}
            },
        },
        "run": {"wall_seconds": 100.0, "peak_rss_kib": 8000000},
        "fingerprint": {
            "fitting": {"files": 78, "touched": []},
            "report": {"files": 78, "exempt": "reports/x.md", "touched": []},
        },
    }
    text = pr.build_report(summary, "2026-09-20", pr.environment())

    assert "Nothing is decided here and nothing ships from it" in text
    assert "no adoption follows from it in either direction" in text
    assert pr.ATTRIBUTION in text
    # The screen's stop is visible in the report, not only in the records.
    assert "worse: stops" in text
    # And the pre-registration is carried whole.
    assert "## Definitions, fixed here" in text
    assert "## Experiment 5" in text
    # The computed sections and the quoted criteria are told apart by heading,
    # so a reader can never mistake a criterion for a result.
    for n in (1, 2, 3):
        assert text.count(f"## What Experiment {n} measured") == 1
        assert text.count(f"## Experiment {n}:") == 1
    pr.check_no_identifiers(text, {"A12"}, {"900000001"})


def test_a_stopped_run_writes_a_report_that_reads_no_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "WORK", tmp_path)
    records = _summary(tmp_path)
    summary = {
        "checks": [
            {
                "what": "December first seed against december_plan-diffs.parquet",
                "expected": "a largest absolute difference of 0 on every row",
                "got": "4.97 s over 165,677 rows",
                "ok": False,
                "source": "the pre-registration, Reproduction",
            }
        ],
        "stopped": "reproduction",
        "records": records,
        "fits": len(records),
        "fit_seconds": float(sum(r["wall_s"] for r in records.values())),
        "frame": {"build_seconds": 56.0, "rows": 1, "legs": {}},
        "run": {"wall_seconds": 1.0, "peak_rss_kib": 1},
        "fingerprint": {
            "fitting": {"files": 78, "touched": []},
            "report": {"files": 78, "exempt": "reports/x.md", "touched": []},
        },
    }
    text = pr.build_report(summary, "2026-09-20", pr.environment())
    assert "stopped at reproduction" in text
    assert "**stop**" in text
    # No experiment section is written, so no figure of theirs can be read.
    assert "## What Experiment 1 measured" not in text
    assert "## What Experiment 2 measured" not in text
    assert "## What Experiment 3 measured" not in text
    # The criteria are still quoted in full, so the stop can be read against them.
    assert "## Experiment 1: seed averaging on the three and on October" in text


def test_the_records_round_trip_as_json(tmp_path, monkeypatch):
    """--report reads what the run wrote, so the records have to be JSON."""
    monkeypatch.setattr(pr, "WORK", tmp_path)
    record = _record("december", "H=12", 217.0, ranking_mix=218.0)
    record["name"] = "december_H12_seed0"
    arm, _ = _two_arm_frames()
    path = pr.write_record(record, {"valid": arm})
    assert json.loads(path.read_text(encoding="utf-8"))["schedule"] == "H=12"
    assert len(pr.read_kept("december_H12_seed0", "valid")) == len(arm)
    assert pr.movements_on_disk() == {str(int(v)) for v in arm["mvt_id"]}


# --------------------------------------------------------------------------
# what was changed when the pre-registration was copied in


def test_every_recorded_place_is_in_the_carried_text_exactly_once():
    """The manifest is tied to the text, so a drifting entry is a failure."""
    for place in pr.SUBSTITUTIONS:
        assert pr.PREREGISTRATION.count(place.marker) == 1, place.where


def test_the_counts_are_summed_from_the_list_and_not_written_down():
    audit = pr.substitution_audit()
    assert audit["places"] == len(pr.SUBSTITUTIONS)
    assert audit["work_block_tokens"] == sum(
        p.tokens for p in pr.SUBSTITUTIONS if p.kind == pr.WORK_BLOCK
    )
    assert audit["note_references"] == sum(
        p.tokens for p in pr.SUBSTITUTIONS if p.kind == pr.NOTE_REFERENCE
    )
    assert audit["tokens"] == audit["work_block_tokens"] + audit["note_references"]
    assert audit["markers_found_once"] and not audit["stray_markers"]
    assert audit["carried_is_clean"]


def test_the_list_holds_only_the_published_side():
    """Recording that a name was removed must not put the name back."""
    for place in pr.SUBSTITUTIONS:
        for field in (place.where, place.marker, place.kind, place.widens or ""):
            assert not pr.WORK_BLOCK_TOKEN.search(field), place.where
            assert not pr.NOTE_REFERENCE_TOKEN.search(field), place.where


def test_every_kind_in_the_list_is_one_of_the_two():
    assert {p.kind for p in pr.SUBSTITUTIONS} == {pr.WORK_BLOCK, pr.NOTE_REFERENCE}
    assert all(p.tokens >= 1 for p in pr.SUBSTITUTIONS)


def test_the_note_comes_from_the_environment_and_has_no_default(monkeypatch, tmp_path):
    """No path on any machine is written into the script: without the variable
    there is no note, and with it the note is the record's file in that
    directory. ``--note`` sets the same name from the command line."""
    import importlib.util

    script = Path(pr.__file__)
    for value, want in ((None, None), (str(tmp_path), tmp_path / pr.NOTE_NAME)):
        if value is None:
            monkeypatch.delenv(pr.NOTES_ENV, raising=False)
        else:
            monkeypatch.setenv(pr.NOTES_ENV, value)
        spec = importlib.util.spec_from_file_location("_preregistered_env", script)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "_preregistered_env", module)
        spec.loader.exec_module(module)
        assert module.NOTE == want


def test_a_clean_checkout_still_audits_against_the_published_text(monkeypatch, tmp_path):
    """No note on disk: the report says the check was not run, not that it passed."""
    monkeypatch.setattr(pr, "NOTE", tmp_path / "absent.md")
    audit = pr.substitution_audit()
    assert audit["note_checked"] is False
    assert "not run" in audit["note_verdict"]
    assert "sha256" not in audit["note_verdict"]
    assert "note_sha256" not in audit
    # And the checks that read the published text alone were still made.
    assert audit["markers_found_once"] and audit["carried_is_clean"]
    assert audit["places"] == len(pr.SUBSTITUTIONS)
    text = "\n".join(pr.section_substitutions())
    assert "**not run**" in text
    assert f"{audit['places']} places" in text


def test_the_audit_notices_a_manifest_that_has_drifted(monkeypatch):
    """A recorded place that is no longer in the text is caught, not ignored."""
    stray = pr.Place("nowhere", "a line this text does not carry", pr.WORK_BLOCK, 1)
    monkeypatch.setattr(pr, "SUBSTITUTIONS", pr.SUBSTITUTIONS + (stray,))
    audit = pr.substitution_audit()
    assert not audit["markers_found_once"]
    assert audit["stray_markers"] == ["nowhere"]
    # With the note on hand the place count no longer agrees either.
    if audit.get("note_checked"):
        assert audit["note_problems"]


def test_the_section_prints_the_counts_it_computed():
    audit = pr.substitution_audit()
    text = "\n".join(pr.section_substitutions())
    assert f"**{audit['places']} places, {audit['tokens']} substitutions.**" in text
    assert f"{audit['work_block_tokens']} work-block names" in text
    assert f"{audit['note_references']} note file names" in text
    for place in pr.SUBSTITUTIONS:
        assert place.where in text
    # The one place that does more than render a name says so.
    assert sum(1 for p in pr.SUBSTITUTIONS if p.widens) >= 1


# --------------------------------------------------------------------------
# the guard, and what actually defeats it


def test_the_underscores_defeat_the_guard_and_the_code_span_does_not():
    """``H12`` is a stand identifier at one of the ten airports.

    The guard searches for it with a word boundary on each side. A backtick is
    not a word character, so a code span is no protection at all; an
    underscore is, which is why a record name is rendered with its own.
    """
    stands = {"H12"}
    pr.check_no_identifiers("| `july_H12_seed0` | valid |", stands, set())
    for rendered in ("`H12`", "july H12 seed0", "july-H12-seed0", "july H12", "(H12)"):
        with pytest.raises(SystemExit, match="stand identifier"):
            pr.check_no_identifiers(f"| {rendered} | valid |", stands, set())


def test_the_fit_table_renders_record_names_with_their_underscores(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "WORK", tmp_path)
    records = _summary(tmp_path)
    text = "\n".join(
        pr.section_fits({"records": records, "fits": len(records), "fit_seconds": 1.0})
    )
    assert "`july_H12_seed0`" in text
    assert "july H12 seed0" not in text
    # And the guard is content with the table as rendered.
    pr.check_no_identifiers(text, {"H12"}, set())


def test_any_label_of_letters_and_digits_can_collide(tmp_path):
    """The next block prints cell labels; the answer is to rename, not to loosen."""
    for label in ("A12", "B7X", "C03"):
        pr.check_no_identifiers(f"cell `sweep_{label}_seed0`", {label}, set())
        with pytest.raises(SystemExit, match="stand identifier"):
            pr.check_no_identifiers(f"cell {label}", {label}, set())


# --------------------------------------------------------------------------
# the verdict of the rule, which Experiment 1 owes


def test_experiment_1_writes_the_verdict_of_the_rule_and_the_guard(monkeypatch):
    records, kept = _seed_world(
        july_three=[237.0, 237.2, 237.4, 237.6, 237.8],
        october=[220.0, 220.1, 220.2, 220.3, 220.4],
        gap=[300.0, 300.1, 300.2, 300.3, 300.4],
    )
    monkeypatch.setattr(pr, "score", lambda frame, three=False: 236.0 if three else 219.0)
    checks = pr.Checks()
    got = pr.experiment_1(records, kept, checks)
    assert got["rule_holds"] is True
    assert [r["leg"] for r in got["rule"]] == ["December, all ten", "July, all ten"]
    assert all(r["beats_every_single"] and r["clears_floor"] for r in got["rule"])
    what = [r["what"] for r in checks.rows]
    assert "Experiment 1, the rule of 19 September" in what
    assert "Experiment 1, the guard of 20 September" in what


def test_the_rule_fails_where_a_leg_does_not_clear_its_floor(monkeypatch):
    thin = (
        {**pr.RULE_LEGS[0], "gain": 0.10},
        pr.RULE_LEGS[1],
    )
    monkeypatch.setattr(pr, "RULE_LEGS", thin)
    records, kept = _seed_world(
        july_three=[237.0, 237.2, 237.4, 237.6, 237.8],
        october=[220.0, 220.1, 220.2, 220.3, 220.4],
        gap=[300.0, 300.1, 300.2, 300.3, 300.4],
    )
    monkeypatch.setattr(pr, "score", lambda frame, three=False: 236.0 if three else 219.0)
    got = pr.experiment_1(records, kept, pr.Checks())
    assert got["rule_holds"] is False
    assert got["rule"][0]["clears_floor"] is False


# --------------------------------------------------------------------------
# what the two wall figures are


def test_the_fit_seconds_are_summed_per_fit_and_not_per_row(tmp_path, monkeypatch):
    """A fit scored on two sets is one wall time, not two."""
    monkeypatch.setattr(pr, "WORK", tmp_path)
    records = _summary(tmp_path)
    total = sum(r["wall_s"] for r in records.values())
    rows = sum(len(r["sets"]) for r in records.values())
    assert rows > len(records), "the fixture needs a fit scored on two sets"
    assert total == pytest.approx(float(len(records)))


def test_inside_the_floor_reads_the_size_and_not_the_sign():
    assert pr.inside_floor(0.10, 0.40) == "inside it"
    assert pr.inside_floor(-0.10, 0.40) == "inside it"
    assert pr.inside_floor(0.37, 0.19) == "outside it"
    assert pr.inside_floor(-0.84, 0.23) == "outside it"
    assert pr.inside_floor(None, 0.40) == "n/a"
    # Equal to the floor is not inside it, the same reading clears_floor uses.
    assert pr.inside_floor(0.5, 0.5) == "outside it"
