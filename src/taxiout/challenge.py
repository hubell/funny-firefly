# SPDX-License-Identifier: GPL-3.0-only
"""Access to the challenge datasets, and the rules for what may be read from them.

The datasets live under ``data/challenge/`` and are never committed; see the
README there. This module knows their layout and, more importantly, records
which fields the ranking set hides for test departures, because every feature
built later has to be checked against that list.

What the ranking set exposes was measured, not assumed, and is written up in
``reports/data_review.md``. The short version: ``BLOCK_TIME_UTC_mvt`` and
``TAXITIME_SEC_mvt`` are blanked for departures, ``MVT_TIME_UTC_mvt`` (the
take-off time) is kept, and ``AOBT_3_flt`` survives but is a minute-resolution
estimate that disagrees with the blanked block time by minutes. The anchor for
a test departure is therefore its take-off time, and no feature may read the
blanked fields of any departure in the same month -- they are hidden for
every departure, not just the subject.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MOVEMENTS_GLOB = "movements/training_*.parquet"
RANKING = "ranking.parquet"
SUBMITTING = "submitting.parquet"

# Blanked for departures in the ranking set. A feature that reads either field
# of any departure -- the subject or its neighbours -- is unavailable at
# inference in January and July 2026.
HIDDEN_FOR_TEST_DEPARTURES: tuple[str, ...] = (
    "BLOCK_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt",
)

# Present in the ranking set but excluded from every feature by decision.
# ``AOBT_3_flt`` is the Network Manager's off-block time for the flown
# trajectory: minute-resolution, a median 51 s and an interquartile range of
# five minutes away from the blanked block time, and no more informative as a
# single predictor than the filed EOBT. It is still a noisy copy of the hidden
# field, and an entry judged on originality has nothing to gain from it.
EXCLUDED_BY_DECISION: tuple[str, ...] = ("AOBT_3_flt",)

FORBIDDEN_FEATURE_INPUTS: tuple[str, ...] = (
    HIDDEN_FOR_TEST_DEPARTURES + EXCLUDED_BY_DECISION
)

# What an *arrival* row exposes in the ranking set. The admission rule is
# measured rather than assumed: a column enters this list when
# ``reports/data_review.md`` puts it at **1% or less null in the ranking
# arrivals**, and the rate is recorded beside it so a reader can check the
# rule rather than the list. Most of these are at 0.00%; the two flight-side
# columns are at 0.76%, which is the share of arrivals whose left join to the
# Network Manager record found nothing, and a null there is a level of its own
# rather than a hole -- the blocks that read them carry NaN for that row and
# say so.
#
# Two of these are the very fields hidden for departures. That is not a
# contradiction but the rule -- a feature may read any field the ranking set
# exposes for the rows it reads from -- and the feature registry says which
# phase a source is read from so the test can hold each side to its own list.
# ``AOBT_3_flt`` is excluded for every phase by decision.
AVAILABLE_FOR_ARRIVALS: tuple[str, ...] = (
    "MVT_TIME_UTC_mvt",  # 0.00%
    "BLOCK_TIME_UTC_mvt",  # 0.00%
    "TAXITIME_SEC_mvt",  # 0.00%
    "SCHED_TIME_UTC_mvt",  # 0.00%
    "RUNWAY_mvt",  # 0.00%
    "STAND_mvt",  # 0.00%
    "ADES_mvt",  # 0.00%
    "ADEP_mvt",  # 0.00%
    "PHASE_mvt",  # 0.00%
    "MVT_ID_mvt",  # 0.00%
    "AIRCRAFT_TYPE_mvt",  # 0.00%
    "AIRCRAFT_OPERATOR_flt",  # 0.76%
)

# Phase prefixes a registered source may carry. A bare column name is the
# subject's own row; ``DEP.`` is another departure's; ``ARR.`` an arrival's.
SOURCE_PHASES: tuple[str, ...] = ("DEP", "ARR")

# ``RUNWAY_mvt`` values that name no runway. ``NA`` is documented by the
# organisers; the other two mark helicopter movements.
NON_RUNWAY_VALUES: frozenset[str] = frozenset({"NA", "HEL", "H"})

STRING_COLUMNS: tuple[str, ...] = (
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "CALLSIGN_flt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
)


def movement_files(root: Path) -> list[Path]:
    """The monthly training files, in calendar order."""
    return sorted(root.glob(MOVEMENTS_GLOB))


def read_table(
    path: Path,
    columns: list[str] | None = None,
    phase: str | None = None,
) -> pa.Table:
    """Read one parquet file, dictionary-encoding strings to keep memory small.

    ``phase`` restricts to ``DEP`` or ``ARR`` rows at read time. The phase
    column is read regardless so the filter can be applied, and dropped again
    unless it was asked for.
    """
    want = list(columns) if columns is not None else None
    if want is not None and phase is not None and "PHASE_mvt" not in want:
        want = want + ["PHASE_mvt"]
    schema_names = pq.read_schema(path).names
    dictionary = [c for c in STRING_COLUMNS if c in schema_names]
    table = pq.read_table(path, columns=want, read_dictionary=dictionary)
    if phase is not None:
        import pyarrow.compute as pc

        mask = pc.equal(pc.cast(table["PHASE_mvt"], pa.string()), phase)
        table = table.filter(mask)
        if columns is not None and "PHASE_mvt" not in columns:
            table = table.drop_columns(["PHASE_mvt"])
    return table


def load_movements(
    root: Path,
    columns: list[str] | None = None,
    phase: str | None = None,
) -> pa.Table:
    """Concatenate the twelve monthly training files."""
    files = movement_files(root)
    if not files:
        raise FileNotFoundError(f"no training files under {root / 'movements'}")
    tables = [read_table(f, columns, phase) for f in files]
    return pa.concat_tables(tables, promote_options="default")


def load_ranking(
    root: Path,
    columns: list[str] | None = None,
    phase: str | None = None,
) -> pa.Table:
    return read_table(root / RANKING, columns, phase)


def load_submitting(root: Path) -> pa.Table:
    return pq.read_table(root / SUBMITTING)


def to_pandas(table: pa.Table):
    """Pandas frame; dictionary-encoded strings arrive as categoricals."""
    return table.to_pandas()
