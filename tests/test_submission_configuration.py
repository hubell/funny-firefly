# SPDX-License-Identifier: GPL-3.0-only
"""What ``scripts/make_submission.py`` writes from v9: the configuration, C6b on
the term's rows, and the booster file beside the submission.

The fits themselves need the challenge data; the script's ``--december-check``
compares them with the harness's stored records. Here: the substitution under
each setting that built a file, the booster file's rows and order, C1's columns
per side, and the configuration as the harness forms it. Synthetic frames only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taxiout import submission
from taxiout.aobt3 import aobt3_features
from taxiout.models import BOOSTER_PARAMS, SUBMISSION_CONFIGURATION

REPO = Path(__file__).resolve().parents[1]
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def _script():
    path = REPO / "scripts" / "make_submission.py"
    spec = importlib.util.spec_from_file_location("make_submission_v9_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the substitution


IDX = pd.Index([10, 11, 12, 13])
PHYSICAL = pd.Series([900.0, 950.0, 1000.0, 1100.0], index=IDX)
TERM = pd.Series([np.nan, 20000.0, np.nan, np.nan], index=IDX)
C6B = pd.Series([np.nan, 7000.0, np.nan, np.nan], index=IDX)
SHIFT = pd.Series([np.nan, np.nan, 87000.0, np.nan], index=IDX)

# (what the term's rows take, substituted, the file) for each file built so far
SETTINGS = {
    "v7: the term and the rule": (TERM, True, [900.0, 20000.0, 87000.0, 1100.0]),
    "v8: neither": (TERM, False, [900.0, 950.0, 1000.0, 1100.0]),
    "v9: C6b and the rule": (C6B, True, [900.0, 7000.0, 87000.0, 1100.0]),
}


@pytest.mark.parametrize("setting", sorted(SETTINGS))
def test_each_file_substitutes_what_its_flags_say(setting):
    record_quality, substitute, want = SETTINGS[setting]
    got = _script().submitted(PHYSICAL, record_quality, SHIFT, substitute)
    assert got.index.equals(IDX)
    assert got.tolist() == want


def test_c6b_takes_the_terms_rows_and_nothing_else():
    got = _script().submitted(PHYSICAL, C6B, SHIFT, True)
    moved = got.to_numpy() != PHYSICAL.to_numpy()
    assert moved.tolist() == (C6B.notna() | SHIFT.notna()).tolist()
    # a C6b series off the term's rows would be refused by main before this
    assert C6B.notna().equals(TERM.notna())


# --------------------------------------------------------------------------
# the booster file


def _template(ids):
    return pd.DataFrame(
        {
            ID: np.asarray(ids, dtype=np.float64),
            TARGET: np.zeros(len(ids), dtype=np.int32),
        }
    )


def test_the_booster_file_is_the_prediction_before_substitution_in_template_order(
    tmp_path,
):
    rank = pd.DataFrame({ID: [3.0, 1.0, 2.0, 4.0]})
    physical = pd.Series([1200.4, -3.0, 600.5, 601.5], index=rank.index)
    template = _template([4.0, 2.0, 3.0, 1.0])
    path = _script().write_booster_file(
        tmp_path / "x_v9.booster.parquet", rank, physical, template, submission
    )
    back = pd.read_parquet(path)
    assert list(back.columns) == [ID, TARGET, "prediction"]
    assert back[ID].tolist() == [4.0, 2.0, 3.0, 1.0]
    assert str(back[TARGET].dtype) == "int32"
    # floored at zero, rounded half to even, as the file writes a prediction
    assert back[TARGET].tolist() == [602, 600, 1200, 0]
    assert back["prediction"].tolist() == [601.5, 600.5, 1200.4, -3.0]


def test_the_booster_file_refuses_a_row_the_template_lacks(tmp_path):
    rank = pd.DataFrame({ID: [1.0, 2.0]})
    physical = pd.Series([600.0, 700.0])
    with pytest.raises(submission.SubmissionError):
        _script().write_booster_file(
            tmp_path / "x_v9.booster.parquet",
            rank,
            physical,
            _template([1.0, 5.0]),
            submission,
        )


# --------------------------------------------------------------------------
# C1's columns, per side

T0 = pd.Timestamp("2026-01-31T23:00:00Z")


def _departures(minutes):
    """Departures at one airport and runway, off-block a quarter hour before take-off."""
    t = [T0 + pd.Timedelta(minutes=m) for m in minutes]
    out = pd.DataFrame(
        {
            ID: np.arange(1, len(t) + 1, dtype=float),
            "ADEP_mvt": "LIRF",
            "RUNWAY_mvt": "25",
            "MVT_TIME_UTC_mvt": pd.to_datetime(t, utc=True),
            "AOBT_3_flt": pd.to_datetime(t, utc=True) - pd.Timedelta(minutes=15),
            "LOBT_flt": pd.to_datetime(t, utc=True) - pd.Timedelta(minutes=20),
            "EOBT_1_flt": pd.to_datetime(t, utc=True) - pd.Timedelta(minutes=25),
        }
    )
    return out


def test_a_ranking_side_counts_each_month_on_its_own():
    # three in January, the last taking off at 23:55, and two in February whose
    # off-block falls before it: counted together they would see each other
    dep = _departures([0, 5, 55, 61, 66])
    frame = dep[[ID]].copy()
    got = _script().aobt3_side(frame, dep, by_month=True)
    month = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    want = pd.concat(
        [
            aobt3_features(dep[month == m].reset_index(drop=True)).set_index(
                dep.index[month == m]
            )
            for m in sorted(month.unique())
        ]
    )
    pd.testing.assert_frame_equal(got, want.loc[frame.index])
    # and the training year's side counts the whole frame at once
    whole = _script().aobt3_side(frame, dep, by_month=False)
    pd.testing.assert_frame_equal(whole, aobt3_features(dep))
    assert not got.equals(whole)  # the boundary row sees different neighbours


def test_c1_columns_refuse_rows_out_of_order():
    dep = _departures([0, 5, 10])
    frame = dep[[ID]].iloc[::-1].reset_index(drop=True)
    with pytest.raises(SystemExit):
        _script().aobt3_side(frame, dep, by_month=False)


# --------------------------------------------------------------------------
# the configuration, as the harness forms it


def test_the_configuration_is_the_harness_combination_of_the_adopted_three():
    assert SUBMISSION_CONFIGURATION == ("c1", "c4", "c3b")
    combo = _script().configuration()
    assert combo.kind == "combination" and combo.layers == ()
    assert combo.member.key == "combo4_booster"
    assert combo.member.blocks == ("aobt3", "flight") and not combo.member.weighted
    assert [m.key for m in combo.members] == ["c3b_member"]
    assert combo.members[0].backend == "catboost" and not combo.members[0].gpu


def test_the_configuration_fits_at_the_ladders_seeds():
    script = _script()
    import candidates_board as board

    combo = script.configuration()
    five = script.Configuration(combo, 5)
    assert five.params["random_state"] == BOOSTER_PARAMS["random_state"]
    assert five.params == board.member_params(combo.member, 0)
    assert len(five.columns) == len(board.member_columns(combo.member))
    member = combo.members[0]
    for s in range(5):
        p = board.member_params(member, s)
        assert p["random_state"] == BOOSTER_PARAMS["random_state"] + s
        assert (p["n_estimators"], p["learning_rate"]) == (3000, 0.08)
