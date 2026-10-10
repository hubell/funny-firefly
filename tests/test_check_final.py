# SPDX-License-Identifier: GPL-3.0-only
"""``scripts/check_final.py`` on a synthetic final file: each check passes on a
file built to pass it and stops on one built to fail it."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

REPO = Path(__file__).resolve().parents[1]
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def _module():
    spec = importlib.util.spec_from_file_location(
        "_check_final", REPO / "scripts" / "check_final.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cf = _module()

# Eight departures: two in each of the four months, at two airports.
MONTHS = [
    "2026-01",
    "2026-01",
    "2026-02",
    "2026-02",
    "2026-06",
    "2026-06",
    "2026-07",
    "2026-07",
]
TAKEOFF = [
    "2026-01-10 10:00",
    "2026-01-31 23:50",
    "2026-02-01 00:20",
    "2026-02-10 10:00",
    "2026-06-10 10:00",
    "2026-06-11 10:00",
    "2026-07-10 10:00",
    "2026-07-11 10:00",
]
LOBT = [
    "2026-01-10 09:50",
    "2026-01-31 23:30",
    "2026-01-31 23:55",
    "2026-02-10 09:50",
    "2026-06-10 09:50",
    "2026-06-11 09:50",
    "2026-07-10 09:50",
    "2026-07-11 09:50",
]


def _case(tmp_path, **change):
    ids = np.arange(1.0, 9.0)
    values = np.array([600, 610, 620, 630, 640, 650, 660, 670], dtype=np.int32)
    parts = pd.DataFrame(
        {
            ID: ids,
            "ADEP_mvt": ["EDDF", "LIRF"] * 4,
            "month": MONTHS,
            "booster": values.astype(float),
            "base": values.astype(float),
            "rule": [""] * 8,
            "rule_value": np.nan,
            "weight": np.nan,
            "lag_s": np.nan,
            "rest_mean_s": np.nan,
            "cell_rows": np.nan,
            "sub_cell_rows": np.nan,
            "submitted": values.astype(float),
            "fusion_taken": False,
            "fusion_value": np.nan,
            "before_window": values.astype(float),
            "window_lag_s": 600.0,
            "window_moved": False,
            "written": values.astype(float),
        }
    )
    # one hedge row: C6b at LIRF in July, its cell held 30 rows in 2025
    parts.loc[7, ["rule", "weight", "lag_s", "rest_mean_s", "cell_rows"]] = [
        "c6b",
        0.25,
        2400.0,
        700.0,
        30.0,
    ]
    for k, v in change.get("parts", {}).items():
        parts.loc[k[0], k[1]] = v
    values = parts["written"].to_numpy(dtype=float)
    file = pd.DataFrame({ID: ids, TARGET: values.astype(np.int32)})
    template = pd.DataFrame({ID: ids, TARGET: np.zeros(8, dtype=np.int32)})
    if change.get("file"):
        file = change["file"](file)
    ranking = pd.DataFrame(
        {
            ID: ids,
            "PHASE_mvt": "DEP",
            "LOBT_flt": pd.to_datetime(LOBT, utc=True),
            "MVT_TIME_UTC_mvt": pd.to_datetime(TAKEOFF, utc=True),
        }
    )
    jj = [0, 1, 6, 7]
    control = file.iloc[jj].copy()
    if change.get("control"):
        control = change["control"](control)
    v11 = file.iloc[jj].copy()
    v11[TARGET] = np.array([590, 600, 650, 700], dtype=np.int32)
    if change.get("v11"):
        v11 = change["v11"](v11)
    neighbours = pd.DataFrame(
        {
            ID: [8.0],
            "columns": ["occ_gap1_s"],
            "explained": [True],
            "month": ["2026-07"],
        }
    )
    if change.get("neighbours"):
        neighbours = change["neighbours"](neighbours)
    columns = pd.DataFrame({ID: v11[ID].to_numpy(), "columns": ["txi_mean_60m"] * 4})
    if change.get("columns"):
        columns = change["columns"](columns)
    reference = pd.DataFrame(
        {
            "airport": ["EDDF", "LIRF", "EDDF", "LIRF"],
            "month": ["02", "02", "06", "06"],
            "rows": [1000, 1000, 1000, 1000],
            "long": [0, 1, 0, 1],
        }
    )
    paths = {}
    for name, frame in (
        ("file", file),
        ("parts", parts),
        ("template", template),
        ("ranking", ranking),
        ("control", control),
        ("v11", v11),
        ("neighbours", neighbours),
        ("v11_columns", columns),
    ):
        paths[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(paths[name], index=False)
    paths["reference_2025"] = tmp_path / "reference.csv"
    reference.to_csv(paths["reference_2025"], index=False)
    paths["manifest"] = tmp_path / "manifest.json"
    paths["manifest"].write_text(json.dumps({"window_applied": True, "window": {}}))
    return cf.run(Namespace(**paths))


def test_a_file_built_to_pass_passes(tmp_path):
    out = _case(tmp_path)
    assert out["ok"], {
        k: v for k, v in out.items() if isinstance(v, dict) and not v.get("ok", True)
    }
    assert out["iii_v11"]["changed"] == 4
    # the hedge row is listed under (a), its cell is not thin, its weight not 0 or 1
    assert [r["rule"] for r in out["audit"]["a_rows"]] == ["c6b"]
    assert not out["audit"]["b_thin_cells"] and not out["audit"]["c_weights_0_or_1"]
    # (d): the plan on 31 January, the take-off on 1 February
    assert [r["month"] for r in out["audit"]["d_month_seam"]] == ["2026-02"]


def test_a_template_out_of_order_stops(tmp_path):
    out = _case(tmp_path, file=lambda f: f.iloc[[1, 0, 2, 3, 4, 5, 6, 7]])
    assert not out["template"]["ok"] and not out["ok"]


def test_a_zero_prediction_is_counted_and_a_negative_one_stops(tmp_path):
    out = _case(tmp_path, parts={(2, "written"): 0.0})
    assert out["values"]["zero"] == 1 and out["values"]["ok"]
    out = _case(tmp_path, parts={(2, "written"): -1.0})
    assert out["values"]["negative"] == 1 and not out["ok"]


def test_outside_v11s_range_stops_only_when_no_rule_set_it(tmp_path):
    out = _case(tmp_path, parts={(3, "written"): 5000.0})
    assert out["v11_range"]["set_by_no_rule"] == 1 and not out["ok"]
    out = _case(tmp_path, parts={(3, "written"): 5000.0, (3, "window_moved"): True})
    assert out["v11_range"]["outside"] == 1 and out["v11_range"]["ok"]


def test_ii_a_row_not_listed_by_i_that_differs_from_the_control_stops(tmp_path):
    def bump(c, row):
        c = c.copy()
        c.loc[c[ID] == row, TARGET] += 1
        return c

    assert not _case(tmp_path, control=lambda c: bump(c, 1.0))["ii_control"]["ok"]
    # the row (i) lists may differ
    assert _case(tmp_path, control=lambda c: bump(c, 8.0))["ii_control"]["ok"]


def test_i_an_unexplained_neighbour_stops(tmp_path):
    out = _case(tmp_path, neighbours=lambda n: n.assign(explained=False))
    assert not out["i_neighbours"]["ok"] and not out["ok"]


def test_iii_a_large_change_no_column_explains_stops(tmp_path):
    big = lambda v: v.assign(**{TARGET: np.array([-5, 600, 650, 700], dtype=np.int32)})  # noqa: E731
    out = _case(tmp_path, v11=big)
    assert out["iii_v11"]["over"] == 1 and out["iii_v11"]["ok"]
    out = _case(
        tmp_path, v11=big, columns=lambda c: c.assign(columns=["", "x", "x", "x"])
    )
    assert out["iii_v11"]["unexplained"] == 1 and not out["ok"]
    # unless the fusion set it: C17's model changed by definition
    out = _case(
        tmp_path,
        v11=big,
        columns=lambda c: c.assign(columns=["", "x", "x", "x"]),
        parts={(0, "fusion_taken"): True},
    )
    assert out["iii_v11"]["ok"]


def test_the_audit_lists_apart_and_stops_on_a_long_base_row(tmp_path):
    out = _case(tmp_path, parts={(7, "cell_rows"): 12.0, (7, "weight"): 1.0})
    assert len(out["audit"]["b_thin_cells"]) == 1
    assert len(out["audit"]["c_weights_0_or_1"]) == 1
    assert out["audit"]["ok"]
    out = _case(tmp_path, parts={(4, "written"): 20000.0, (4, "window_moved"): True})
    assert out["audit"]["a_long"] == 1 and out["audit"]["ok"]
    out = _case(tmp_path, parts={(4, "written"): 20000.0})
    assert len(out["audit"]["a_unexplained"]) == 1 and not out["ok"]


def test_e_a_ratio_above_two_is_for_review_and_not_a_stop(tmp_path):
    # EDDF in June: one long row of one, against none of a thousand in 2025
    out = _case(tmp_path, parts={(4, "written"): 20000.0, (4, "window_moved"): True})
    review = out["audit"]["e_for_review"]
    assert [(r["airport"], r["month"]) for r in review] == [("EDDF", "2026-06")]
    assert out["ok"]


# --------------------------------------------------------------------------
# the window the written values are clipped to (taxiout.window)


def test_the_window_is_the_records_notes():
    from taxiout import window

    assert (window.W_STAR_S, window.REDRAW_S, window.w_prime_s()) == (3606, 12, 3618)


def test_the_lag_is_take_off_less_plan_and_missing_without_a_plan():
    from taxiout import window

    frame = pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": pd.to_datetime(
                ["2026-02-10 10:00", "2026-02-10 10:00"], utc=True
            ),
            "LOBT_flt": pd.to_datetime(["2026-02-10 09:50", None], utc=True),
        }
    )
    lag = window.lag_s(frame)
    assert lag[0] == 600.0 and np.isnan(lag[1])


def test_the_window_binds_both_ways_and_leaves_rows_without_a_plan():
    from taxiout import window

    lag = np.array([600.0, 600.0, 5000.0, np.nan])
    got = window.clip([100.0, 9000.0, 100.0, 9000.0], lag)
    # capped at the lag plus W', raised to the lag less W', untouched without one
    assert got.tolist() == [100.0, 600.0 + 3618, 5000.0 - 3618, 9000.0]
    with pytest.raises(ValueError):
        window.clip([1.0, 2.0], lag)


def test_the_written_values_are_floored_and_rounded_after_the_clip():
    from taxiout import window

    # a lag below zero puts the window's floor under zero: the clip leaves 0
    # where it is, and the floor at zero follows the clip
    values = np.array([100.0, 9000.0, 0.0, 50.0])
    lag = np.array([5000.0, 600.0, -4000.0, np.nan])
    out, moved = window.apply(values, lag)
    assert out.tolist() == [1382.0, 4218.0, 0.0, 50.0]
    assert moved.tolist() == [True, True, False, False]


def test_ii_a_row_whose_fusion_peer_moved_is_explained():
    """Two July departures at one airport ten minutes apart: the first is a
    neighbour whose base moved, so the second's fusion inputs moved with it."""
    ids = np.array([1.0, 2.0, 3.0])
    ranking = pd.DataFrame(
        {
            ID: ids,
            "MVT_TIME_UTC_mvt": pd.to_datetime(
                ["2026-07-01 10:00", "2026-07-01 10:10", "2026-07-01 12:00"], utc=True
            ),
        }
    )
    parts = pd.DataFrame(
        {ID: ids, "ADEP_mvt": ["EDDF"] * 3, "base": [600.0, 610.0, 620.0]}
    )
    control_parts = parts.assign(base=[590.0, 610.0, 620.0])
    file = pd.DataFrame({ID: ids, TARGET: np.array([600, 611, 625], dtype=np.int32)})
    control = pd.DataFrame({ID: ids, TARGET: np.array([590, 610, 620], dtype=np.int32)})
    neighbours = pd.DataFrame({ID: [1.0], "explained": [True]})
    out = cf.check_control(file, control, neighbours, parts, control_parts, ranking)
    # row 2 is within 30 minutes of row 1; row 3 is not, and stops the check
    assert out["differ_elsewhere"] == 2 and out["explained_by_peers"] == 1
    assert out["unexplained"] == 1 and not out["ok"]
    out = cf.check_control(
        file.assign(**{TARGET: [600, 611, 620]}),
        control,
        neighbours,
        parts,
        control_parts,
        ranking,
    )
    assert out["ok"] and out["explained_by_peers"] == 1


def test_the_peer_window_is_the_fusions():
    fusion = pytest.importorskip("taxiout.fusion")
    assert cf.PEER_WINDOW_S == fusion.PEER_WINDOW_S
