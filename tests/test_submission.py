# SPDX-License-Identifier: GPL-3.0-only
"""Submission validity: identifiers, target, schema, filename, overwrite."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.submission import (  # noqa: E402
    ID,
    INT32_SQUARE_LIMIT_S,
    TARGET,
    SubmissionError,
    above_square_limit,
    check_filename,
    prepare,
    read,
    validate,
    write,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _template(n: int = 5) -> pd.DataFrame:
    """Identifiers as doubles and a null int target, as the organizers' file has them."""
    ids = np.array([900000000 + 3 * i for i in range(n)], dtype=np.float64)
    return pd.DataFrame({ID: ids, TARGET: np.full(n, np.nan)})


def _pred(template: pd.DataFrame, values=None, shuffle: bool = True) -> pd.DataFrame:
    p = template[[ID]].copy()
    p[TARGET] = 900.0 if values is None else values
    if shuffle:
        p = p.iloc[::-1].reset_index(drop=True)
    return p


def test_valid_frame_passes():
    t = _template()
    validate(_pred(t), t)
    validate(_pred(t).assign(extra="ignored"), t)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.iloc[:-1], "rows"),
        (lambda p: pd.concat([p, p.iloc[:1]], ignore_index=True), "duplicated"),
        (lambda p: p.assign(**{ID: p[ID] + 1}), "identifier sets differ"),
        (lambda p: p.assign(**{ID: [np.nan, *p[ID].iloc[1:]]}), "null"),
        (lambda p: p.assign(**{TARGET: [np.nan, *p[TARGET].iloc[1:]]}), "null"),
        (lambda p: p.assign(**{TARGET: [np.inf, *p[TARGET].iloc[1:]]}), "non-finite"),
        (lambda p: p.assign(**{TARGET: 900.5}), "non-integer"),
        (lambda p: p.assign(**{TARGET: -1.0}), "negative"),
        (lambda p: p.assign(**{TARGET: 2.0**31}), "int32"),
        (lambda p: p.drop(columns=[TARGET]), "no"),
        (lambda p: p.drop(columns=[ID]), "no"),
    ],
)
def test_invalid_frames_are_rejected(mutate, message):
    t = _template()
    with pytest.raises(SubmissionError, match=message):
        validate(mutate(_pred(t)), t)


def test_extra_row_is_rejected_with_a_count():
    t = _template()
    p = _pred(t)
    extra = pd.concat([p, pd.DataFrame({ID: [1.0], TARGET: [900.0]})], ignore_index=True)
    with pytest.raises(SubmissionError, match="rows"):
        validate(extra, t)
    # same row count, one swapped identifier: one missing, one extra
    swapped = p.copy()
    swapped.loc[0, ID] = 1.0
    with pytest.raises(SubmissionError, match="1 template rows missing.*1 extra"):
        validate(swapped, t)


def test_broken_template_is_rejected():
    t = _template()
    p = _pred(t)
    with pytest.raises(SubmissionError, match="template"):
        validate(p, pd.concat([t, t.iloc[:1]], ignore_index=True))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("team_v1.parquet", ("team", 1)),
        ("my_team_v12.parquet", ("my_team", 12)),
        ("a-b_v3.parquet", ("a-b", 3)),
    ],
)
def test_filename_pattern_accepts(name, expected):
    assert check_filename(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "team.parquet",
        "team_v0.parquet",
        "team_v01.parquet",
        "team_v1.csv",
        "_v1.parquet",
        "team_v1.parquet.bak",
    ],
)
def test_filename_pattern_rejects(name):
    with pytest.raises(SubmissionError, match="filename"):
        check_filename(name)


def test_prepare_rounds_orders_and_types():
    t = _template()
    p = _pred(t, values=[900.4, 900.6, 0.0, 1.0e6, 12.5])
    out = prepare(p, t)
    assert list(out.columns) == [ID, TARGET]
    assert out[ID].dtype == np.float64 and out[TARGET].dtype == np.int32
    # Template order, not the (reversed) prediction's; values follow their
    # identifiers. rint rounds half to even, so 12.5 -> 12.
    assert out[ID].tolist() == t[ID].tolist()
    assert out[TARGET].tolist() == [900, 901, 0, 1_000_000, 12]


def test_write_reads_back_with_the_template_schema(tmp_path):
    t = _template()
    path = write(_pred(t), t, tmp_path / "team_v1.parquet")
    table = pq.read_table(path)
    assert table.schema.names == [ID, TARGET]
    assert table.schema.field(ID).type == pa.float64()
    assert table.schema.field(TARGET).type == pa.int32()
    assert table.schema.metadata is None
    back = read(path)
    assert back[ID].tolist() == t[ID].tolist()
    assert back[TARGET].tolist() == [900] * len(t)
    validate(back, t)


def test_write_refuses_bad_name_overwrite_and_invalid_frame(tmp_path):
    t = _template()
    with pytest.raises(SubmissionError, match="filename"):
        write(_pred(t), t, tmp_path / "team.parquet")
    path = write(_pred(t), t, tmp_path / "team_v1.parquet")
    with pytest.raises(SubmissionError, match="exists"):
        write(_pred(t), t, path)
    write(_pred(t, values=1.0), t, path, overwrite=True)
    assert read(path)[TARGET].tolist() == [1] * len(t)
    with pytest.raises(SubmissionError):
        write(_pred(t).iloc[:-1], t, tmp_path / "team_v2.parquet")
    assert not (tmp_path / "team_v2.parquet").exists()


def test_square_limit_is_the_largest_residual_whose_square_fits_int32():
    top = int(np.iinfo(np.int32).max)
    assert INT32_SQUARE_LIMIT_S**2 <= top < (INT32_SQUARE_LIMIT_S + 1) ** 2
    # What the limit is about: numpy squares an int32 array in int32 and wraps.
    with np.errstate(over="ignore"):
        d = np.array([INT32_SQUARE_LIMIT_S, INT32_SQUARE_LIMIT_S + 1], dtype=np.int32)
        sq = d * d
    assert sq[0] == INT32_SQUARE_LIMIT_S**2 and sq[1] < 0


def test_above_square_limit_counts_strictly_above():
    limit = INT32_SQUARE_LIMIT_S
    values = np.array([0, 900, limit, limit + 1, 102_506], dtype=np.int32)
    assert above_square_limit(values) == 2
    assert above_square_limit(pa.array(values, type=pa.int32())) == 2
    assert above_square_limit([]) == 0
