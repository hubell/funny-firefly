# SPDX-License-Identifier: GPL-3.0-only
"""Validate and write a submission file.

The challenge's ranking page
(https://ansperformance.eu/study/data-challenge/dc2026/ranking.html), in
its own words: "The ``submitting.parquet`` file is provided as a template.
To make a submission, fill in the ``TAXITIME_SEC_mvt`` column with your
predicted taxi-out times for the corresponding ``MVT_ID_mvt`` values."
"For a submission to be valid, each row must match the corresponding
``MVT_ID_mvt`` before the ranking script can use the prediction to calculate
the RMSE." "The ranking script will issue an error if: an ``MVT_ID_mvt``
does not match; any rows are missing; or extra rows are included." The file
is named "``<team-name>_v<incremental integer>.parquet``". The template
itself fixes the rest: two columns, ``MVT_ID_mvt`` as a double and
``TAXITIME_SEC_mvt`` as a 32-bit integer, one row per ranking departure.

Everything here exists so that a file which would be rejected, or scored on
the wrong rows, cannot be written by accident. ``validate`` raises
``SubmissionError`` with a count-level message -- never an identifier -- on
the first failure it finds. ``write`` rounds, validates, puts the rows in
the template's order so that "each row must match" holds positionally as
well as by set, writes the template's schema, reads the file back and
validates that too, and refuses to overwrite: the version is in the filename
precisely so that a rewritten file is a new one. The team name is whatever
the caller passes, checked for shape only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
COLUMNS: tuple[str, str] = (ID, TARGET)

# ``<team>_v<n>.parquet``. The team part may contain underscores; the
# version is the last ``_v<digits>`` before the extension, unpadded.
FILENAME = re.compile(
    r"^(?P<team>[A-Za-z0-9][A-Za-z0-9_\-]*)_v(?P<version>[1-9]\d*)\.parquet$"
)

INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


class SubmissionError(ValueError):
    """The frame or file would be rejected, or scored on the wrong rows."""


# --------------------------------------------------------------------------
# checks


def _ids(frame: pd.DataFrame, what: str) -> pd.Series:
    if ID not in frame.columns:
        raise SubmissionError(f"{what} has no {ID!r} column")
    ids = frame[ID]
    if ids.isna().any():
        raise SubmissionError(f"{what} has {int(ids.isna().sum())} null {ID!r} values")
    if ids.duplicated().any():
        raise SubmissionError(
            f"{what} has {int(ids.duplicated().sum())} duplicated {ID!r} values"
        )
    return ids


def validate(pred: pd.DataFrame, template: pd.DataFrame) -> None:
    """Raise ``SubmissionError`` unless ``pred`` is a valid submission.

    Checks: both columns present; identifiers non-null and unique; the
    identifier set equal to the template's, so no row is missing or extra;
    the target non-null, finite, integer-valued, non-negative and within
    ``int32``. Extra columns are tolerated here and dropped by ``write``.
    """
    if TARGET not in pred.columns:
        raise SubmissionError(f"prediction has no {TARGET!r} column")
    ids = _ids(pred, "prediction")
    tids = _ids(template, "template")
    if len(pred) != len(template):
        raise SubmissionError(
            f"prediction has {len(pred):,} rows, template {len(template):,}"
        )
    # Compare as float64: the template stores the identifier as a double and
    # the values are integers far below 2**53, so the comparison is exact.
    p_set = set(ids.astype(np.float64).to_numpy())
    t_set = set(tids.astype(np.float64).to_numpy())
    if p_set != t_set:
        raise SubmissionError(
            f"identifier sets differ: {len(t_set - p_set):,} template rows missing "
            f"from the prediction, {len(p_set - t_set):,} extra"
        )
    t = pd.to_numeric(pred[TARGET], errors="coerce")
    if t.isna().any():
        raise SubmissionError(
            f"{int(t.isna().sum()):,} null or non-numeric {TARGET!r} values"
        )
    v = t.to_numpy(dtype=np.float64)
    if not np.isfinite(v).all():
        raise SubmissionError(
            f"{int((~np.isfinite(v)).sum()):,} non-finite {TARGET!r} values"
        )
    if (v != np.rint(v)).any():
        raise SubmissionError(
            f"{int((v != np.rint(v)).sum()):,} non-integer {TARGET!r} values; round first"
        )
    if (v < 0).any():
        raise SubmissionError(f"{int((v < 0).sum()):,} negative {TARGET!r} values")
    if (v > INT32_MAX).any() or (v < INT32_MIN).any():
        raise SubmissionError(f"{TARGET!r} values outside int32")


def check_filename(path: Path | str) -> tuple[str, int]:
    """The (team, version) a filename encodes; raises if it does not."""
    name = Path(path).name
    m = FILENAME.match(name)
    if m is None:
        raise SubmissionError(f"filename {name!r} does not match <team>_v<n>.parquet")
    return m.group("team"), int(m.group("version"))


# --------------------------------------------------------------------------
# writing


def prepare(pred: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """The frame ``write`` writes: template order, template dtypes, two columns.

    Rounds a float target to the nearest integer first, so a model's raw
    output can be passed in; then validates.
    """
    if TARGET not in pred.columns:
        raise SubmissionError(f"prediction has no {TARGET!r} column")
    rounded = pred.loc[:, [c for c in pred.columns if c in COLUMNS]].copy()
    t = pd.to_numeric(rounded[TARGET], errors="coerce")
    finite = np.isfinite(t.to_numpy(dtype=np.float64))
    rounded[TARGET] = np.where(finite, np.rint(t.to_numpy(dtype=np.float64)), np.nan)
    validate(rounded, template)
    out = template.loc[:, [ID]].copy()
    out[ID] = out[ID].astype(np.float64)
    lookup = pd.Series(
        rounded[TARGET].to_numpy(dtype=np.float64),
        index=rounded[ID].astype(np.float64).to_numpy(),
    )
    out[TARGET] = lookup.reindex(out[ID].to_numpy()).to_numpy().astype(np.int32)
    return out.reset_index(drop=True)


def write(
    pred: pd.DataFrame,
    template: pd.DataFrame,
    path: Path | str,
    overwrite: bool = False,
) -> Path:
    """Write ``pred`` as a submission file at ``path`` and return the path.

    Validates before and after writing; refuses an existing file unless
    ``overwrite`` is set; the filename must read ``<team>_v<n>.parquet``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    check_filename(path)
    if path.exists() and not overwrite:
        raise SubmissionError(
            f"{path.name} exists; bump the version or pass overwrite=True"
        )
    out = prepare(pred, template)
    # Built from arrays rather than from the frame so the file carries the
    # template's two typed columns and no pandas metadata.
    table = pa.table(
        {
            ID: pa.array(out[ID].to_numpy(), type=pa.float64()),
            TARGET: pa.array(out[TARGET].to_numpy(), type=pa.int32()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    back = pq.read_table(path)
    if back.schema.names != list(COLUMNS) or back.num_rows != len(template):
        raise SubmissionError(f"{path.name} did not read back with the template's shape")
    if (
        back.schema.field(ID).type != pa.float64()
        or back.schema.field(TARGET).type != pa.int32()
    ):
        raise SubmissionError(f"{path.name} did not read back with the template's types")
    validate(back.to_pandas(), template)
    return path


def read(path: Path | str) -> pd.DataFrame:
    """A written submission, as a frame."""
    import pyarrow.parquet as pq

    return pq.read_table(Path(path)).to_pandas()


__all__ = [
    "COLUMNS",
    "FILENAME",
    "ID",
    "TARGET",
    "SubmissionError",
    "check_filename",
    "prepare",
    "read",
    "validate",
    "write",
]
