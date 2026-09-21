# SPDX-License-Identifier: GPL-3.0-only
"""The wider sweep's arithmetic, on synthetic frames.

Nothing here fits a booster or reads the challenge data. What is held to a
test is the part of ``scripts/sweep_params.py`` that decides things: the shape
of the grid, the claim that six tree counts come out of one fit, the reading
rule and its boundary, the edge verdict, the floor, and the form of a cell
label, which is what the report guards meet.

The figures below are constructed to make each rule's boundary visible, and
none of them is a measurement of anything.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import preregistered as pr  # noqa: E402
import sweep_params as sw  # noqa: E402


# --------------------------------------------------------------------------
# the grid


def test_the_grid_is_twenty_seven_cells_and_six_tree_counts():
    """Three axes of three, and the trees axis read from one fit per cell."""
    assert [len(v) for v in sw.GRID.values()] == [3, 3, 3]
    assert len(sw.CELLS) == 27
    assert sw.TREES == (300, 600, 1000, 1500, 2000, 3000)
    assert len(sw.CELLS) * len(sw.TREES) == 162


def test_the_grid_holds_the_current_cell_and_is_the_pre_registered_one():
    assert sw.GRID["learning_rate"] == (0.015, 0.03, 0.06)
    assert sw.GRID["num_leaves"] == (63, 127, 255)
    assert sw.GRID["min_child_samples"] == (20, 50, 200)
    assert tuple(sw.CURRENT_CELL[k] for k in sw.GRID) in sw.CELLS
    assert sw.CURRENT_TREES in sw.TREES
    # The rung is the submission rung, read from the key and not pinned to
    # the rung the first sweep ran on.
    assert sw.SWEEP_RUNG.key == "plan-diffs"


def test_the_current_cell_is_a_member_of_the_grid_whatever_it_is():
    """The reading rule compares the best cell against a row of the same grid.

    ``CURRENT_CELL`` is read out of ``BOOSTER_PARAMS`` rather than written
    down, so it moves when the parameters move. What has to stay true is that
    it is one row of this grid, because the rule reads the current cell's
    figure from the grid and from nowhere else.
    """
    assert tuple(sw.CURRENT_CELL[k] for k in sw.GRID) in sw.CELLS
    assert sw.CURRENT_TREES in sw.TREES


# --------------------------------------------------------------------------
# six readings out of one fit


class _Model:
    """A booster whose prediction depends on how many trees are read."""

    def __init__(self, trees: int) -> None:
        self.trees = trees

    def predict(self, design, num_iteration=None):
        k = self.trees if num_iteration is None else int(num_iteration)
        return np.asarray(design.iloc[:, 0], dtype=float) + float(k)


class _Booster:
    fits = 0

    def __init__(self, features, params) -> None:
        self.features = tuple(features)
        self.params = dict(params)
        self.pooled = ()
        self.categorical = ()
        self._kept: dict = {}
        self._levels: dict = {}
        self.model = None

    def fit(self, X, y, sample_weight=None):
        assert sample_weight is None, "no weight may reach the backend in this sweep"
        type(self).fits += 1
        self.model = _Model(int(self.params["n_estimators"]))
        return self


class _Frame:
    """The three things ``fit_cell`` asks a frame for."""

    def __init__(self, n: int = 60) -> None:
        index = pd.RangeIndex(n)
        self.y = pd.Series(np.linspace(400.0, 900.0, n), index=index)
        self.features = pd.DataFrame(
            {"icao": ["EDDF"] * (n // 2) + ["EGLL"] * (n - n // 2)}, index=index
        )
        columns = {c: np.arange(n, dtype=float) for c in sw.SWEEP_RUNG.features}
        self._prep = {
            "rows": pd.Series([True] * n, index=index),
            "flagged": pd.Series([False] * n, index=index),
            "frame": pd.DataFrame(columns, index=index),
        }
        self._valid = pd.Series([True] * n, index=index)

    def prepare(self, name):
        return self._prep

    def sets(self, name):
        return {"valid": self._valid}


def test_six_tree_counts_come_out_of_one_fit(monkeypatch):
    """One fit per cell, six readings, and the readings are not the same one."""
    monkeypatch.setattr(sw, "Booster", _Booster)
    _Booster.fits = 0
    record, predictions = sw.fit_cell(_Frame(), sw.cell_values(sw.CELLS[0]))
    assert _Booster.fits == 1
    assert sorted(int(k) for k in record["rmse_oct_s"]) == sorted(sw.TREES)
    assert sorted(int(k) for k in predictions) == sorted(sw.TREES)
    assert len(set(record["sha256"].values())) == len(sw.TREES)
    # The fit is made at the top of the trees axis and read downward.
    assert record["n_estimators_fitted"] == max(sw.TREES)
    assert record["fit_wall_s"] > 0


def test_the_cell_is_fitted_at_the_swept_parameters_and_nothing_else(monkeypatch):
    seen = {}

    class _Recording(_Booster):
        def fit(self, X, y, sample_weight=None):
            seen.update(self.params)
            return super().fit(X, y, sample_weight=sample_weight)

    monkeypatch.setattr(sw, "Booster", _Recording)
    values = sw.cell_values(sw.CELLS[-1])
    sw.fit_cell(_Frame(), values)
    for axis, value in values.items():
        assert seen[axis] == value
    assert seen["n_estimators"] == max(sw.TREES)
    assert seen["random_state"] == int(sw.BOOSTER_PARAMS["random_state"])
    # Everything the sweep does not touch is the module's own.
    for key in ("objective", "deterministic", "force_col_wise", "subsample"):
        assert seen[key] == sw.BOOSTER_PARAMS[key]


# --------------------------------------------------------------------------
# the reading rule


CURRENT = tuple(sw.CURRENT_CELL[k] for k in sw.GRID)
# Some cell that is not the current one, whichever the current one is. The
# boundary tests need two distinguishable rows and must not assume which cell
# ``BOOSTER_PARAMS`` holds today.
BEST = next(c for c in sw.CELLS if c != CURRENT)
BEST_TREES = next(t for t in sw.TREES if t != sw.CURRENT_TREES)


def _grid(best_rmse: float, current_rmse: float = 218.0) -> pd.DataFrame:
    """A grid where one cell holds ``best_rmse`` and the current cell the other.

    Every other row is put well above both, so the ordering is not in
    question and only the boundary is.
    """
    rows = []
    best = BEST
    for cell in sw.CELLS:
        for trees in sw.TREES:
            value = 400.0
            if cell == CURRENT:
                value = current_rmse if trees == sw.CURRENT_TREES else 401.0
            elif cell == best and trees == BEST_TREES:
                value = best_rmse
            rows.append(
                {**sw.cell_values(cell), "n_estimators": trees, "rmse_oct_s": value}
            )
    return pd.DataFrame(rows)


def test_a_gain_larger_than_the_floor_replaces_the_current_cell():
    got = sw.apply_reading_rule(_grid(217.25), 0.5)
    assert got["booster_params_moves"] is True
    assert got["best"]["rmse_oct_s"] == pytest.approx(217.25)
    assert got["chosen_label"] == sw.cell_label(sw.cell_values(BEST), BEST_TREES)
    # And the axes come back as the types the grid holds, not as the floats a
    # row pulled out of a frame of one dtype would give.
    assert isinstance(got["chosen"]["num_leaves"], int)
    assert isinstance(got["chosen"]["min_child_samples"], int)


def test_a_gain_equal_to_the_floor_keeps_the_current_cell():
    """Equal to the floor is not inside it, and the rule asks for more.

    The boundary is written in halves because a decimal like 0.19 is not
    exact in binary, and a test written in decimals would be testing the
    rounding rather than the rule.
    """
    got = sw.apply_reading_rule(_grid(217.5), 0.5)
    assert got["difference_s"] == pytest.approx(-0.5)
    assert got["booster_params_moves"] is False
    assert got["chosen"]["num_leaves"] == sw.CURRENT_CELL["num_leaves"]
    assert got["chosen"]["rmse_oct_s"] == pytest.approx(218.0)


def test_a_gain_smaller_than_the_floor_keeps_the_current_cell():
    got = sw.apply_reading_rule(_grid(217.75), 0.5)
    assert got["booster_params_moves"] is False
    assert got["chosen_label"] == sw.cell_label(sw.CURRENT_CELL, sw.CURRENT_TREES)


def test_the_current_cell_figure_is_this_run_s_and_comes_from_the_grid():
    """Not the figure in models.py, and not any earlier run's."""
    got = sw.apply_reading_rule(_grid(217.25, current_rmse=219.5), 0.5)
    assert got["current"]["rmse_oct_s"] == pytest.approx(219.5)
    assert got["difference_s"] == pytest.approx(217.25 - 219.5)


def test_the_rule_reads_all_one_hundred_and_sixty_two_rows():
    got = sw.apply_reading_rule(_grid(217.25), 0.5)
    assert got["rows"] == 162
    assert len(got["top_eight"]) == 8


# --------------------------------------------------------------------------
# the edge verdict


# The one cell of this grid that is interior on all four axes, written out so
# the edge test does not rest on which cell ``BOOSTER_PARAMS`` holds today.
INTERIOR = {
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 50,
    "n_estimators": 1000,
}


def test_an_edge_cell_is_detected_on_each_of_the_four_axes():
    assert sw.edges_of(INTERIOR) == []
    for axis, ends in sw.EDGES.items():
        for end in ends:
            assert sw.edges_of({**INTERIOR, axis: end}) == [axis]


def test_a_cell_on_more_than_one_edge_names_every_axis():
    row = {**sw.LARGEST_CELL, "n_estimators": 3000}
    assert sw.edges_of(row) == ["num_leaves", "min_child_samples", "n_estimators"]


def test_the_edges_are_the_ends_of_this_grid_and_not_the_first_one_s():
    assert sw.EDGES["num_leaves"] == (63, 255)
    assert sw.EDGES["min_child_samples"] == (20, 200)
    assert sw.EDGES["n_estimators"] == (300, 3000)
    # 31 and 500 were the first grid's far values and are dropped here.
    assert 31 not in sw.GRID["num_leaves"]
    assert 500 not in sw.GRID["min_child_samples"]


# --------------------------------------------------------------------------
# the floor


def test_the_floor_is_root_two_times_the_sample_standard_deviation():
    singles = [217.28, 217.65, 217.43, 217.37, 217.43]
    expected = float(np.sqrt(2.0) * np.std(np.asarray(singles), ddof=1))
    assert pr.floor_of(singles) == pytest.approx(expected)
    # And not the population standard deviation, which would read lower.
    assert pr.floor_of(singles) > np.sqrt(2.0) * float(np.std(np.asarray(singles)))


def _stored(values, digests=None):
    digests = digests or [f"{i:064x}" for i in range(len(values))]
    return [
        {
            "fit_rows": sw.BLOCK_07_OCTOBER_FIT_ROWS,
            "sets": {
                "valid": {
                    "rows": sw.BLOCK_07_OCTOBER_ROWS,
                    "flagged": sw.BLOCK_07_OCTOBER_FLAGGED,
                    "pooled_ex_flagged": v,
                    "sha256": d,
                }
            },
        }
        for v, d in zip(values, digests)
    ]


def _gate(monkeypatch, found, stored=None, digests=None):
    stored = _stored(stored if stored is not None else found, digests)
    monkeypatch.setattr(sw, "block_07_october", lambda: stored)

    def fake(frame, leg, *, seed, schedule, features):
        assert schedule == "none" and features == "submission"
        record = {
            "fit_rows": sw.BLOCK_07_OCTOBER_FIT_ROWS,
            "wall_s": 1.0,
            "sets": {
                "valid": {
                    "rows": sw.BLOCK_07_OCTOBER_ROWS,
                    "flagged": sw.BLOCK_07_OCTOBER_FLAGGED,
                    "pooled_ex_flagged": found[seed],
                    "sha256": stored[seed]["sets"]["valid"]["sha256"],
                }
            },
        }
        kept = {"valid": pd.DataFrame({"pred": np.zeros(3)})}
        return record, kept

    monkeypatch.setattr(pr, "fit_once", fake)
    return sw.gate(object())


def test_the_gate_measures_the_floor_from_its_own_five_singles(monkeypatch):
    singles = [217.28, 217.65, 217.43, 217.37, 217.43]
    got = _gate(monkeypatch, singles)
    assert got["floor_s"] == pytest.approx(pr.floor_of(singles))
    assert sw.gate_verdict(got, None) == []


def test_the_gate_stops_where_a_single_does_not_reproduce(monkeypatch):
    """Compared against the stored double, not against the printed decimals."""
    stored = [217.28, 217.65, 217.43, 217.37, 217.43]
    found = list(stored)
    found[2] = 217.4300001
    got = _gate(monkeypatch, found, stored=stored)
    bad = sw.gate_verdict(got, None)
    assert bad and "seed +2" in bad[0]
    # It prints the same two decimals, which is exactly why the gate does not
    # read the printed figure.
    assert round(found[2], 2) == round(stored[2], 2)


def test_the_tree_reading_check_is_a_stop_where_the_prefix_differs():
    gate_result = {
        "singles": [
            {
                "seed": 0,
                "found_s": 217.28,
                "found_sha256": "a" * 64,
                "same_row_set": True,
                "equal_double": True,
                "equal_digest": True,
            }
        ],
        "floor_s": 0.19,
        "floor_expected_printed_s": 0.19,
    }
    trees = str(sw.CURRENT_TREES)
    same = {"rmse_oct_s": {trees: 217.28}, "sha256": {trees: "a" * 64}}
    assert sw.tree_reading_check(gate_result, same)["equal_digest"] is True
    assert sw.gate_verdict(gate_result, sw.tree_reading_check(gate_result, same)) == []
    other = {"rmse_oct_s": {trees: 217.28}, "sha256": {trees: "b" * 64}}
    bad = sw.gate_verdict(gate_result, sw.tree_reading_check(gate_result, other))
    assert bad and "tree-reading" in bad[0]


# --------------------------------------------------------------------------
# the labels the report guards meet


WORD_RUN = re.compile(r"\w+")


def _collidable(token: str) -> bool:
    """The shape ``check_no_identifiers`` looks for: three or more characters,
    at least one letter and at least one digit."""
    return (
        len(token) >= 3
        and bool(re.search(r"[A-Za-z]", token))
        and bool(re.search(r"[0-9]", token))
    )


def test_no_cell_label_carries_a_token_shaped_like_a_stand_identifier():
    for cell in sw.CELLS:
        for trees in (None, *sw.TREES):
            label = sw.cell_label(sw.cell_values(cell), trees)
            bad = [t for t in WORD_RUN.findall(label) if _collidable(t)]
            assert bad == [], f"{label} carries {bad}"


def test_a_short_label_is_refused_and_the_written_out_one_is_not():
    """The answer to a refusal is to rename the label, never to loosen the guard."""
    stands = {"A12", "B7X", "C03", "H12"}
    for label in sorted(stands):
        with pytest.raises(SystemExit, match="stand identifier"):
            pr.check_no_identifiers(f"the best cell is {label}", stands, set())
        # A backtick is not a word character, so a code span is no protection.
        with pytest.raises(SystemExit, match="stand identifier"):
            pr.check_no_identifiers(f"the best cell is `{label}`", stands, set())
    for cell in sw.CELLS:
        pr.check_no_identifiers(sw.cell_label(sw.cell_values(cell), 3000), stands, set())


def test_the_label_form_and_not_these_values_is_what_makes_it_safe():
    """Written out in full, no grid value can put a colliding token in a label."""
    for values in (
        {"learning_rate": 0.5, "num_leaves": 12, "min_child_samples": 7},
        {"learning_rate": 1e-05, "num_leaves": 1024, "min_child_samples": 3},
    ):
        label = sw.cell_label(values, 12345)
        assert [t for t in WORD_RUN.findall(label) if _collidable(t)] == []


# --------------------------------------------------------------------------
# the records the grid is read from


def test_the_grid_is_read_from_one_record_per_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "WORK", tmp_path)
    for index, cell in enumerate(sw.CELLS):
        values = sw.cell_values(cell)
        (tmp_path / f"cell_{index:02d}.json").write_text(
            json.dumps(
                {
                    "label": sw.cell_label(values),
                    "params": values,
                    "rmse_oct_s": {str(t): 220.0 + index for t in sw.TREES},
                }
            ),
            encoding="utf-8",
        )
    grid = sw.read_grid()
    assert len(grid) == 162
    assert sorted(grid["n_estimators"].unique()) == sorted(sw.TREES)
    assert len(grid.groupby(list(sw.GRID))) == 27


def test_a_missing_cell_stops_the_reading_rather_than_shortening_the_grid(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sw, "WORK", tmp_path)
    with pytest.raises(SystemExit, match="the grid is not complete"):
        sw.read_grid()
