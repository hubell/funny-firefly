# SPDX-License-Identifier: GPL-3.0-only
"""Join challenge stand identifiers to the OpenStreetMap distance matrix.

``STAND_mvt`` and the OSM ``ref`` tag name the same stands in slightly
different conventions, and the differences are per airport: one side pads
with zeros, the other splits a wide stand into lettered halves, a third
writes a space between pier letter and number. The join therefore runs in
stages, each a little more forgiving than the last, and a movement is matched
at the first stage that finds its stand. Precision is kept where it exists --
a lettered half-stand that OSM maps as such is matched as such -- and only
rows that would otherwise be lost fall through to the coarser key.

The stages, and what each was found to fix (``reports/data_review.md``):

1. **Case and whitespace.** Upper-case, internal whitespace removed. OSM refs
   at Frankfurt carry a space between letter and number.
2. **Leading zeros** in every digit run. Schiphol's movement data zero-pads
   the number; OSM does not.
3. **One trailing letter after a digit.** Munich and Istanbul map split
   stands as lettered halves where the movement data names the whole.

Whatever the stage, the normalisation is applied to *both* sides, and it was
chosen by join rate and inspection of what failed to match -- never by which
choice lowered a validation error.

Two things the staged key does that the matrix builder did not have to. A
handful of OSM refs carry two identifiers in one tag; those are split so each
identifier gets the cell. And a coarser key can collapse several OSM
identifiers onto one -- ``101``, ``101L`` and ``101R`` become one ``101`` at
stage 3 -- so the collapsed cell records how many identifiers it stands for
and how far their taxi distances disagree, exactly as the matrix already does
for duplicate features behind one identifier. Nothing here decides what to do
with an ambiguous cell; it makes the ambiguity visible so the modelling stage
can down-weight or drop it.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from taxiout.coverage import identifier_shape
from taxiout.routing import AGGREGATED_FIELDS

STAGES: tuple[tuple[int, str], ...] = (
    (1, "case and whitespace"),
    (2, "leading zeros"),
    (3, "trailing letter"),
)
UNMATCHED = 0

# A token that reads as a stand identifier on its own: optional pier letters,
# digits, optional half-stand letter. Used only to recognise a ref that holds
# two identifiers, so it is deliberately narrow.
_ID_TOKEN = re.compile(r"^[A-Z]{0,2}\d+[A-Z]?$")

KEY_COLUMNS = ("icao", "key", "designator")
CARRIED_COLUMNS = ("runway_ref", "n_features", "spread_m", "snap_max_m") + tuple(
    AGGREGATED_FIELDS
)


def normalise(value: str, stage: int) -> str:
    """Normalisation key for ``value`` at ``stage`` (cumulative)."""
    key = re.sub(r"\s+", "", str(value).strip().upper())
    if stage >= 2:
        key = re.sub(r"(?<![0-9])0+(?=[0-9])", "", key)
    if stage >= 3:
        key = re.sub(r"(?<=[0-9])[A-Z]$", "", key)
    return key


def split_compound(value: str) -> list[str]:
    """Identifiers held in one tag value.

    ``"1 1A"`` and ``"Y01;Y02"`` are two identifiers each; ``"V 117"`` is one,
    because ``V`` alone is not an identifier. Anything not recognisably a list
    is returned whole.
    """
    tokens = [t for t in re.split(r"[\s;]+", str(value).strip().upper()) if t]
    if len(tokens) >= 2 and all(_ID_TOKEN.match(t) for t in tokens):
        return tokens
    return [str(value).strip()]


def expand_compound(matrix: pd.DataFrame) -> pd.DataFrame:
    """One matrix row per identifier, duplicating cells whose ref held two."""
    out = matrix.copy()
    out["stand_id"] = out["stand_id"].map(split_compound)
    return out.explode("stand_id", ignore_index=True)


def collapse(matrix: pd.DataFrame, stage: int) -> pd.DataFrame:
    """Matrix keyed at ``stage``: one row per (icao, key, designator).

    Where several identifiers share a key, the representative is the cell
    whose ``taxi_m`` is closest to the group mean -- the same rule
    ``routing.aggregate`` uses for duplicate features -- so the row stays a real
    route. ``n_ids`` counts the identifiers collapsed and ``id_spread_m`` the
    range of their taxi distances; ``spread_m`` and ``snap_max_m`` are taken as
    the worst in the group, so a cell's ambiguity never shrinks by collapsing.
    """
    m = expand_compound(matrix)
    m["key"] = m["stand_id"].map(lambda v: normalise(v, stage))
    grp = m.groupby(list(KEY_COLUMNS), sort=True)
    mean = grp["taxi_m"].transform("mean")
    m["_dev"] = (m["taxi_m"] - mean).abs()
    pick = (
        m.sort_values(["_dev", "stand_id"]).groupby(list(KEY_COLUMNS), sort=True).head(1)
    )
    stats = grp.agg(
        n_ids=("stand_id", "nunique"),
        id_spread_m=("taxi_m", lambda s: float(s.max() - s.min())),
        _spread=("spread_m", "max"),
        _snap=("snap_max_m", "max"),
    ).reset_index()
    out = pick.merge(stats, on=list(KEY_COLUMNS), how="left")
    out["spread_m"] = out[["spread_m", "_spread", "id_spread_m"]].max(axis=1)
    out["snap_max_m"] = out[["snap_max_m", "_snap"]].max(axis=1)
    out["stage"] = stage
    cols = list(KEY_COLUMNS) + ["stand_id", "stage", "n_ids", "id_spread_m"]
    cols += [c for c in CARRIED_COLUMNS if c in out.columns]
    return out[cols].reset_index(drop=True)


def _codes(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes (-1 for missing) and the string each code stands for."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes.to_numpy(), s.cat.categories.astype(str).to_numpy()
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    return codes, pd.Index(uniques).astype(str).to_numpy()


def join_triples(triples: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Join distinct ``(icao, stand, designator)`` triples through the stages.

    ``triples`` has string columns ``icao``, ``stand``, ``designator``. The
    result adds the matrix columns and ``join_stage``.
    """
    out = triples.reset_index(drop=True).copy()
    out["join_stage"] = UNMATCHED
    have = out[["icao", "stand", "designator"]].notna().all(axis=1)
    tables = {
        stage: collapse(matrix, stage).drop(columns=["stage"]) for stage, _ in STAGES
    }
    value_cols = [c for c in tables[1].columns if c not in KEY_COLUMNS]
    for c in value_cols:
        out[c] = pd.Series(index=out.index, dtype=tables[1][c].dtype)
    for stage, _ in STAGES:
        pending = have & (out["join_stage"] == UNMATCHED)
        if not pending.any():
            break
        sub = out.loc[pending, ["icao", "stand", "designator"]].astype(str)
        sub["key"] = sub["stand"].map(lambda v: normalise(v, stage))
        table = tables[stage]
        merged = sub.merge(
            table,
            left_on=["icao", "key", "designator"],
            right_on=list(KEY_COLUMNS),
            how="left",
        )
        merged.index = sub.index
        hit = merged["stand_id"].notna()
        if not hit.any():
            continue
        for c in value_cols:
            out.loc[merged.index[hit], c] = merged.loc[hit, c].to_numpy()
        out.loc[merged.index[hit], "join_stage"] = stage
    return out


def join_stands(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    icao: str = "ADEP_mvt",
    stand: str = "STAND_mvt",
    designator: str = "RUNWAY_mvt",
) -> pd.DataFrame:
    """Attach matrix cells to movements, trying each stage in turn.

    Returns ``frame`` with the matrix columns appended and ``join_stage`` set
    to the stage that matched, or ``UNMATCHED`` (0) where none did. Rows with
    a missing stand or designator are unmatched by construction.

    Millions of movements share a few thousand distinct (airport, stand,
    runway) triples, so the triples are joined once and the result is spread
    back over the rows by integer code -- no per-row string work.
    """
    c1, u1 = _codes(frame[icao])
    c2, u2 = _codes(frame[stand])
    c3, u3 = _codes(frame[designator])
    n2, n3 = len(u2), len(u3)
    combined = (c1.astype(np.int64) * n2 + c2) * n3 + c3
    combined[(c1 < 0) | (c2 < 0) | (c3 < 0)] = -1
    gid, uniques = pd.factorize(combined)
    triples = pd.DataFrame(
        {
            "icao": np.where(uniques >= 0, u1[np.maximum(uniques // (n2 * n3), 0)], None),
            "stand": np.where(
                uniques >= 0, u2[np.maximum((uniques // n3) % n2, 0)], None
            ),
            "designator": np.where(uniques >= 0, u3[np.maximum(uniques % n3, 0)], None),
        }
    )
    joined = join_triples(triples, matrix)
    out = frame.copy()
    for c in joined.columns:
        if c in ("icao", "stand", "designator"):
            continue
        out[c] = joined[c].to_numpy()[gid]
    return out


def join_rates(
    joined: pd.DataFrame,
    icao: str = "ADEP_mvt",
    stand: str = "STAND_mvt",
    top_shapes: int = 4,
) -> pd.DataFrame:
    """Movement-weighted share matched at each stage, per airport.

    Also lists the commonest *shapes* of the unmatched identifiers (see
    ``coverage.identifier_shape``) -- never the identifiers themselves.
    """
    rows = []
    for ap, g in joined.groupby(icao, sort=True, observed=True):
        n = len(g)
        if n == 0:
            continue
        rec: dict[str, object] = {"icao": ap, "rows": n}
        for stage, _ in STAGES:
            rec[f"stage{stage}"] = float((g["join_stage"] == stage).mean())
        rec["matched"] = float((g["join_stage"] != UNMATCHED).mean())
        un = g.loc[g["join_stage"] == UNMATCHED, stand]
        shapes = Counter(
            identifier_shape(str(v)) if pd.notna(v) else "<null>" for v in un
        )
        rec["unmatched_shapes"] = ", ".join(
            f"`{s}` {c / n:.1%}" for s, c in shapes.most_common(top_shapes)
        )
        rows.append(rec)
    return pd.DataFrame(rows)
