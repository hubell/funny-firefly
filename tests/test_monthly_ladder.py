# SPDX-License-Identifier: GPL-3.0-only
"""The every-rung, every-month report's splits, resume, pooling and guards.

Nothing here fits a booster or reads the challenge data. What is held to a
test is the part of ``scripts/report_monthly_ladder.py`` that would be wrong
without announcing it: a split that trains on the month it scores, a rung
missing or out of order, a resume that refits a finished month, a margin that
averages figures instead of pooling rows, a stratum that loses or doubles a
row, a dry run that fits, and a history fetch that writes into the model's
own weather cache.
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

import fetch_metar_history as fmh  # noqa: E402
import report_monthly_ladder as rml  # noqa: E402
from taxiout.models import LADDER  # noqa: E402

# --------------------------------------------------------------------------
# the splits and the rungs


def test_there_are_twelve_splits_each_one_month_against_eleven():
    plan = rml.split_plan()
    assert [m for m, _ in plan] == list(rml.MONTHS)
    assert len(plan) == 12
    for month, train in plan:
        assert month not in train
        assert len(train) == 11
        assert set(train) | {month} == set(rml.MONTHS)


def test_the_split_masks_follow_the_months_and_never_overlap():
    month = pd.Series(["2025-01", "2025-02", "2025-07", "2025-12", "2025-12"])
    total = 0
    for m in rml.MONTHS:
        split = rml.make_split(month, m)
        assert not bool((split.train & split.valid).any())
        assert int(split.valid.sum()) == int((month == m).sum())
        assert int(split.train.sum()) == int((month != m).sum())
        total += int(split.valid.sum())
    assert total == len(month)


def test_december_is_the_ladders_december_split():
    from taxiout.validation import DECEMBER_TRAIN, DECEMBER_VALID

    assert rml.LADDER_MONTH == DECEMBER_VALID[0]
    assert dict(rml.split_plan())[rml.LADDER_MONTH] == tuple(DECEMBER_TRAIN)


def test_every_rung_of_the_ladder_in_ladder_order():
    assert rml.KEYS == tuple(r.key for r in LADDER)
    assert len(rml.KEYS) == len(set(rml.KEYS))
    # The reference rungs are included, and every boosted rung is stored by run().
    assert set(rml.REFERENCE) == {r.key for r in LADDER if r.kind != "booster"}
    assert set(rml.BOOSTED) | set(rml.REFERENCE) == set(rml.KEYS)
    # The ladder table of reports/baseline.md prints every rung but the diagnostic.
    assert len([r for r in LADDER if not r.diagnostic]) == len(LADDER) - 1


# --------------------------------------------------------------------------
# resume


def _write_month(work: Path, month: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for path in rml.month_files(month):
        path.write_bytes(b"x")


def test_a_resume_skips_a_month_whose_files_exist(monkeypatch, tmp_path, capsys):
    work = tmp_path / "monthly_ladder"
    monkeypatch.setattr(rml, "WORK", work)
    monkeypatch.setattr(rml, "FORBIDDEN", (tmp_path / "reports",))
    for m in (rml.LADDER_MONTH, "2025-03"):
        _write_month(work, m)

    def refuse(*_a, **_k):
        raise AssertionError("a finished month was refitted")

    monkeypatch.setattr(rml.rm, "Frame", refuse)
    monkeypatch.setattr(rml, "fit_month", refuse)
    assert rml.main(["--fit", "--months", "2025-03", rml.LADDER_MONTH]) == 0
    out = capsys.readouterr().out
    assert "on disk, skipped: 2025-03, 2025-12" in out
    assert "nothing to fit" in out


def test_a_month_missing_one_file_is_not_done(monkeypatch, tmp_path):
    work = tmp_path / "monthly_ladder"
    monkeypatch.setattr(rml, "WORK", work)
    _write_month(work, "2025-04")
    assert rml.month_done("2025-04")
    rml.rung_path("2025-04", rml.KEYS[-1]).unlink()
    assert not rml.month_done("2025-04")
    _write_month(work, "2025-05")
    rml.run_path("2025-05").unlink()  # the marker is written last
    assert not rml.month_done("2025-05")


def test_no_month_fits_before_the_ladder_check(monkeypatch, tmp_path):
    monkeypatch.setattr(rml, "WORK", tmp_path / "monthly_ladder")
    monkeypatch.setattr(rml, "FORBIDDEN", (tmp_path / "reports",))
    with pytest.raises(SystemExit, match="ladder check"):
        rml.main(["--fit", "--months", "2025-01"])


# --------------------------------------------------------------------------
# pooling


class _Fake:
    """A stand-in for ``Store`` with only what ``Agg`` reads."""

    def __init__(self, months, month, R, flagged, y, term):
        self.months = months
        self.month = np.asarray(month)
        self.R = np.asarray(R, dtype=float)
        self.flagged = np.asarray(flagged, dtype=bool)
        self.y = np.asarray(y, dtype=float)
        self.term = np.asarray(term, dtype=float)
        self.n = len(self.month)


def _fake():
    # Two months, a small one with a large error and a large one with a small.
    month = [0] * 2 + [1] * 8
    r = np.array([10.0, -10.0] + [1.0, -1.0] * 4)
    R = np.column_stack([r] + [r * 0.5] * (len(rml.KEYS) - 1))
    return _Fake(
        ("2025-01", "2025-02"), month, R, [False] * 10, [600.0] * 10, [np.nan] * 10
    )


def test_a_pooled_cell_is_the_rmse_of_the_concatenated_residuals():
    st = _fake()
    agg = rml.Agg(st)
    g = agg.grid(np.zeros(st.n, dtype=np.int64), 1)
    year = rml.readings(g, 0)["ex"][0, 2]
    assert year == pytest.approx(np.sqrt(np.mean(st.R[:, 0] ** 2)))
    monthly = rml.readings(g, 0)["ex"][0, :2]
    assert monthly == pytest.approx([10.0, 1.0])
    # The mean of the monthly figures is not the pooled figure.
    assert abs(year - np.mean(monthly)) > 0.5


def test_a_margin_over_cells_pools_their_rows():
    st = _fake()
    agg = rml.Agg(st)
    code = np.array([0, 1] * 5, dtype=np.int64)
    g = agg.grid(code, 2)
    whole = agg.grid(np.zeros(st.n, dtype=np.int64), 1)
    s2 = g["s2_ex"][..., 0].sum(axis=0)
    n = g["n_ex"].sum(axis=0)
    assert np.sqrt(s2 / n) == pytest.approx(rml.readings(whole, 0)["ex"][0])


def test_the_flagged_rows_leave_only_the_ex_flagged_reading_and_the_term_replaces_them():
    st = _fake()
    st.flagged[0] = True
    st.term[0] = 590.0  # residual with the term: 600 - 590 = 10
    st.y[1] = 5 * 3600.0  # a day-shift row, unflagged
    agg = rml.Agg(st)
    g = agg.grid(np.zeros(st.n, dtype=np.int64), 1)
    r = rml.readings(g, 0)
    keep = ~st.flagged
    assert r["ex"][0, 2] == pytest.approx(np.sqrt(np.mean(st.R[keep, 0] ** 2)))
    d4 = keep & (st.y <= rml.DAY_SHIFT_S)
    assert r["d4"][0, 2] == pytest.approx(np.sqrt(np.mean(st.R[d4, 0] ** 2)))
    assert int(g["n_ex"][0, 2] - g["n_d4"][0, 2]) == 1
    with_term = st.R[:, 0].copy()
    with_term[0] = 10.0
    assert r["term"][0, 2] == pytest.approx(np.sqrt(np.mean(with_term**2)))
    assert r["all"][0, 2] == pytest.approx(np.sqrt(np.mean(st.R[:, 0] ** 2)))


def test_a_small_cell_prints_its_count_only():
    assert (
        rml.cell_text(123.4, 120.0, rml.MIN_CELL_ROWS - 1, 0)
        == f"({rml.MIN_CELL_ROWS - 1} rows)"
    )
    assert rml.cell_text(123.4, 120.0, 500, 0) == "123.4"
    assert rml.cell_text(809.9, 159.5, 500, 1) == "809.9 [159.5; 1 out]"


# --------------------------------------------------------------------------
# the strata


def test_every_stratum_assigns_every_row_to_exactly_one_cell():
    values = np.array(["EDDF", "LIRF", "EDDF", "EGLL"])
    codes = rml.partition(values, ["EDDF", "EGLL", "LIRF"])
    assert codes.tolist() == [0, 2, 0, 1]
    with pytest.raises(ValueError, match="no cell"):
        rml.partition(values, ["EDDF", "EGLL"])

    flags, names = rml.flag_partition(
        np.array([0.0, 1.0, np.nan, 2.0, 1.0]), [(0.0, "no"), (1.0, "yes")]
    )
    assert names == ["no", "yes", "missing"]
    assert flags.tolist() == [0, 1, 2, 2, 1]

    year = np.array([100.0, 200.0, 300.0, 400.0, 500.0, np.nan])
    bands, labels, q = rml.distance_bands(year, year)
    assert len(labels) == 5
    assert np.bincount(bands, minlength=5).sum() == len(year)
    assert bands[-1] == 4  # no routed distance

    st = _fake()
    agg = rml.Agg(st)
    g = agg.grid(np.array([0, 1, 2] * 3 + [0], dtype=np.int64), 3)
    assert int(g["n_all"][:, -1].sum()) == st.n
    with pytest.raises(ValueError, match="outside the partition"):
        agg.grid(np.array([0, 3] * 5, dtype=np.int64), 3)


# --------------------------------------------------------------------------
# --dry-run fits nothing


def test_the_dry_run_fits_nothing_and_writes_nothing(monkeypatch, tmp_path, capsys):
    takeoff = pd.Series(pd.date_range("2025-01-01", "2025-12-31", freq="7D", tz="UTC"))
    monkeypatch.setattr(
        rml.rb,
        "load_departures",
        lambda _frac: pd.DataFrame({"MVT_TIME_UTC_mvt": takeoff}),
    )
    monkeypatch.setattr(rml, "WORK", tmp_path / "monthly_ladder")
    monkeypatch.setattr(rml, "FORBIDDEN", (tmp_path / "reports",))

    def refuse(*_a, **_k):
        raise AssertionError("the dry run fitted")

    monkeypatch.setattr(rml, "fit_month", refuse)
    monkeypatch.setattr(rml.rb, "run", refuse)
    monkeypatch.setattr(rml.rm, "Frame", refuse)
    assert rml.main(["--dry-run"]) == 0
    assert not (tmp_path / "monthly_ladder").exists()
    out = capsys.readouterr().out
    assert "nothing was touched" in out
    assert f"the {len(LADDER)} rungs" in out


# --------------------------------------------------------------------------
# the history fetch


def _stat(directory: Path) -> dict:
    return {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_the_history_fetcher_writes_nowhere_under_the_models_cache(monkeypatch, tmp_path):
    fm = fmh.fm
    model = Path(fmh.MODEL_CACHE)
    before = _stat(model) if model.exists() else {}
    # Restore the fetcher's module names after the run repoints them.
    monkeypatch.setattr(fm, "CACHE", fm.CACHE)
    monkeypatch.setattr(fm, "MANIFEST", fm.MANIFEST)
    monkeypatch.setattr(fm, "PAUSE_S", 0.0)
    monkeypatch.setattr(
        fm, "fetch", lambda url: "station,valid,tmpf\nEDDF,2015-01-01 00:20,30\n"
    )
    monkeypatch.setattr(fmh, "CACHE", tmp_path / "metar_history")
    monkeypatch.setattr(fmh, "MANIFEST", tmp_path / "metar_history" / "MANIFEST.json")
    monkeypatch.setattr(fmh, "stations", lambda: ["EDDF"])
    monkeypatch.setattr(fmh, "PERIODS", (("2015-01-01", "2016-01-01"),))
    assert fmh.main([]) == 0
    written = sorted(p.name for p in (tmp_path / "metar_history").iterdir())
    assert written == ["EDDF_2015-01-01_2016-01-01.csv", "MANIFEST.json"]
    assert (_stat(model) if model.exists() else {}) == before


def test_the_history_fetcher_refuses_a_cache_inside_the_models():
    assert fmh.outside_model_cache(fmh.CACHE)
    assert not fmh.outside_model_cache(fmh.MODEL_CACHE)
    assert not fmh.outside_model_cache(fmh.MODEL_CACHE / "history")


def test_the_history_covers_the_ten_years_before_the_training_year():
    years = [int(start[:4]) for start, _ in fmh.PERIODS]
    assert years == list(range(2015, 2025))
    assert tuple(years) == rml.HISTORY_YEARS
