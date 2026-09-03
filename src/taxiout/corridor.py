# SPDX-License-Identifier: GPL-3.0-only
"""Shared-corridor load: the traffic on the subject's own taxiways.

A count of concurrent movements cannot tell two aircraft leaving opposite
ends of the field from two aircraft on the same taxiway. The graph can: it
knows which taxiway edges every (stand, runway) route uses, so for a subject
it can weigh each other movement in the window by how much of the subject's
route that movement's route shares. That is the one feature here no other
entry has, and *Where the geometry could go next* in ``docs/modelling_notes.md``
named it the strongest idea latent in the geometry.

**Edges are weighted from the field, not from traffic.** An edge's weight is
the inverse of the number of (stand, runway) routes that use it -- a static
property of the airport's layout, the holding-point-convergence idea from the
notes doing double duty. A holding-point edge every route funnels through
weighs almost nothing; an apron spur three stands share weighs a third. No
per-split refit, nothing fitted on the ranking months, no leakage surface.
The overlap between two routes is the summed weight of the edges they share,
divided by the summed weight of the subject's own route, so it reads as the
share of the subject's route the other aircraft was on.

**Two sides, both physical.** Other departures in the window whose route
shares edges with the subject's, anchored on their take-off; and arrivals
whose runway-to-stand route shares edges with it, anchored on their on-block
-- opposite-direction traffic on a taxilane is the more physical interaction,
the ranking set exposes on-block times in full, and the edge sets are the
same undirected routes. Windows of 5 and 15 minutes only: at an hour,
"shared an edge" means nothing.

**The degeneracy test that matters** is not whether the overlap matrix is
dense but whether the weighted load is a new column or the throughput count
renamed. ``reports/congestion.md`` prints the correlation per airport, and the
rung's place on the ladder -- after throughput -- settles it.

The per-airport overlap matrix is derived from OpenStreetMap and is computed
in memory from the ``<ICAO>_edges.csv`` files under ``data/processed/``,
which are gitignored with the distance matrices; nothing OSM-derived is
written by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxiout.join import UNMATCHED, expand_compound, join_stands
from taxiout.routing import decode_edges
from taxiout.windows import seconds

CORRIDOR_WINDOWS_MIN: tuple[int, ...] = (5, 15)

CORRIDOR_FEATURES: tuple[str, ...] = tuple(
    name for w in CORRIDOR_WINDOWS_MIN for name in (f"corr_dep_{w}m", f"corr_arr_{w}m")
) + ("corr_route_weight",)

_SELF = ("MVT_TIME_UTC_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_ID_mvt")
_DEP = (
    "DEP.MVT_TIME_UTC_mvt",
    "DEP.ADEP_mvt",
    "DEP.STAND_mvt",
    "DEP.RUNWAY_mvt",
    "DEP.MVT_ID_mvt",
)
_ARR = ("ARR.BLOCK_TIME_UTC_mvt", "ARR.ADES_mvt", "ARR.STAND_mvt", "ARR.RUNWAY_mvt")

SOURCES: dict[str, tuple[str, ...]] = {
    **{f"corr_dep_{w}m": _SELF + _DEP for w in CORRIDOR_WINDOWS_MIN},
    **{f"corr_arr_{w}m": _SELF + _ARR for w in CORRIDOR_WINDOWS_MIN},
    "corr_route_weight": ("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"),
}


def load_edges(files: list) -> pd.DataFrame:
    """The ``<ICAO>_edges.csv`` files, one row per cell, edges decoded."""
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"stand_id": str, "designator": str, "edges": str})
        df["edges"] = df["edges"].map(decode_edges)
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["icao", "stand_id", "designator", "n_edges", "edges"]
        )
    return pd.concat(frames, ignore_index=True)


@dataclass
class Overlap:
    """One airport's overlap matrix over the cells that the data uses.

    ``cells`` maps ``stand_id|designator`` to a row of ``matrix``; ``matrix``
    is normalised so ``matrix[i, j]`` is the share of route ``i``'s weight
    that route ``j`` shares. ``weight`` is each route's own summed weight,
    the thing the row was divided by, and a feature in its own right: it
    says how much of the route is pavement few others use.
    """

    cells: dict[str, int]
    matrix: np.ndarray
    weight: np.ndarray


def overlap_matrix(edges: pd.DataFrame, used: list[str]) -> Overlap:
    """Overlap over ``used`` cells (``stand_id|designator``) of one airport.

    Edge weights are the inverse of how many of *all* the airport's routed
    cells use the edge -- the field's property, not the traffic's -- and the
    matrix is then restricted to the cells the data uses.
    """
    keys = (
        edges["stand_id"].astype(str) + "|" + edges["designator"].astype(str)
    ).to_numpy()
    sets = edges["edges"].tolist()
    use: dict[tuple[int, int], int] = {}
    for es in sets:
        for e in es:
            use[e] = use.get(e, 0) + 1
    weight_of = {e: 1.0 / n for e, n in use.items()}
    index = {k: i for i, k in enumerate(keys)}
    cells = {k: i for i, k in enumerate(u for u in used if u in index)}
    n = len(cells)
    if n == 0:
        return Overlap(
            cells={}, matrix=np.zeros((0, 0), dtype=np.float32), weight=np.zeros(0)
        )
    # sparse incidence over the used cells, then one product
    edge_index: dict[tuple[int, int], int] = {}
    rows, cols, vals = [], [], []
    own = np.zeros(n)
    for k, i in cells.items():
        for e in sets[index[k]]:
            j = edge_index.setdefault(e, len(edge_index))
            rows.append(i)
            cols.append(j)
            vals.append(weight_of[e])
            own[i] += weight_of[e]
    from scipy.sparse import csr_matrix

    a = csr_matrix((vals, (rows, cols)), shape=(n, len(edge_index)))
    # A[i, e] = w_e where i uses e; (A @ B.T)[i, j] = sum over shared edges of w_e
    b = csr_matrix((np.ones(len(vals)), (rows, cols)), shape=a.shape)
    shared = (a @ b.T).toarray()
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(own[:, None] > 0, shared / own[:, None], 0.0)
    return Overlap(cells=cells, matrix=norm.astype(np.float32), weight=own)


def _cell_key(joined: pd.DataFrame) -> pd.Series:
    """``stand_id|designator`` of the matched cell, or NaN where unmatched."""
    matched = joined["join_stage"] != UNMATCHED
    key = joined["stand_id"].astype(str) + "|" + joined["RUNWAY_mvt"].astype(str)
    return key.where(matched)


def _window_load(
    overlap: Overlap,
    subject_row: np.ndarray,
    subject_t: np.ndarray,
    event_row: np.ndarray,
    event_t: np.ndarray,
    event_is_self: np.ndarray,
    window_s: int,
) -> np.ndarray:
    """Sum of overlap with every event in ``[t - w, t]``, self excluded.

    Events are sorted by time; each subject's window is an index range, and
    the gather over all (subject, event) pairs is one fancy index into the
    matrix followed by a segmented sum.
    """
    order = np.argsort(event_t, kind="stable")
    et = event_t[order]
    er = event_row[order]
    es = event_is_self[order]
    lo = np.searchsorted(et, subject_t - window_s, side="left")
    hi = np.searchsorted(et, subject_t, side="right")
    lengths = hi - lo
    total = int(lengths.sum())
    out = np.zeros(len(subject_t))
    if total == 0:
        return out
    subj = np.repeat(np.arange(len(subject_t)), lengths)
    # the flat event positions: for each subject, lo..hi-1
    starts = np.repeat(lo, lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    pos = starts + offsets
    values = overlap.matrix[subject_row[subj], er[pos]]
    # a subject is its own event; its self-overlap is 1 and is not traffic
    values = np.where(es[pos] & (subject_row[subj] == er[pos]), 0.0, values)
    np.add.at(out, subj, values)
    return out


def corridor_features(
    departures: pd.DataFrame,
    movements: pd.DataFrame,
    matrix: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """The corridor block for ``departures``, one airport at a time.

    Subjects and other movements are joined to their matrix cell through
    ``taxiout.join`` exactly as the geometry block is; an unmatched movement
    has no route and contributes nothing, and an unmatched subject gets NaN,
    which is the honest value for "the geometry could not place it".
    """
    out = pd.DataFrame(index=departures.index)
    for c in CORRIDOR_FEATURES:
        out[c] = np.nan
    if len(edges) == 0:
        return out
    edges = expand_compound(edges)

    sub = join_stands(departures[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]], matrix)
    sub_key = _cell_key(sub)
    dep = movements[movements["PHASE_mvt"].astype(str) == "DEP"]
    arr = movements[movements["PHASE_mvt"].astype(str) == "ARR"]
    dep_j = join_stands(dep[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]], matrix)
    arr_j = join_stands(
        arr[["ADES_mvt", "STAND_mvt", "RUNWAY_mvt"]].rename(
            columns={"ADES_mvt": "ADEP_mvt"}
        ),
        matrix,
    )
    dep_key, arr_key = _cell_key(dep_j), _cell_key(arr_j)
    sub_icao = departures["ADEP_mvt"].astype(str).to_numpy()
    dep_icao = dep["ADEP_mvt"].astype(str).to_numpy()
    arr_icao = arr["ADES_mvt"].astype(str).to_numpy()
    sub_t = seconds(departures["MVT_TIME_UTC_mvt"])
    dep_t = seconds(dep["MVT_TIME_UTC_mvt"])
    arr_t = seconds(arr["BLOCK_TIME_UTC_mvt"])
    sub_id = departures["MVT_ID_mvt"].to_numpy()
    dep_id = dep["MVT_ID_mvt"].to_numpy()

    for icao in np.unique(sub_icao):
        s_mask = sub_icao == icao
        d_mask = (dep_icao == icao) & dep_key.notna().to_numpy()
        a_mask = (arr_icao == icao) & arr_key.notna().to_numpy()
        used = sorted(
            set(sub_key[s_mask].dropna()) | set(dep_key[d_mask]) | set(arr_key[a_mask])
        )
        ov = overlap_matrix(edges[edges["icao"].astype(str) == icao], used)
        if not ov.cells:
            continue
        s_row = sub_key[s_mask].map(ov.cells)
        placed = s_row.notna().to_numpy()
        s_idx = np.flatnonzero(s_mask)[placed]
        s_rows = s_row.to_numpy()[placed].astype(int)
        s_times = sub_t[s_mask][placed]
        out.loc[departures.index[s_idx], "corr_route_weight"] = ov.weight[s_rows]

        d_rows = dep_key[d_mask].map(ov.cells).to_numpy().astype(int)
        d_times = dep_t[d_mask]
        # self by identity: the event row is the subject's own movement
        d_self = np.isin(dep_id[d_mask], sub_id[s_mask][placed])
        a_rows = arr_key[a_mask].map(ov.cells).to_numpy().astype(int)
        a_times = arr_t[a_mask]
        for w in CORRIDOR_WINDOWS_MIN:
            out.loc[departures.index[s_idx], f"corr_dep_{w}m"] = _window_load(
                ov, s_rows, s_times, d_rows, d_times, d_self, 60 * w
            )
            out.loc[departures.index[s_idx], f"corr_arr_{w}m"] = _window_load(
                ov, s_rows, s_times, a_rows, a_times, np.zeros(len(a_rows), bool), 60 * w
            )
    return out[list(CORRIDOR_FEATURES)]


def degeneracy(features: pd.DataFrame) -> pd.DataFrame:
    """Per airport: is the corridor load a new column or throughput renamed?

    The correlation of the weighted load with the plain take-off count in the
    same window, and the share of subjects whose load is zero although the
    count is not -- traffic that was there but nowhere near the subject.
    """
    rows = []
    icao = features["icao"].astype(str)
    for ap, g in list(features.groupby(icao, observed=True)) + [("all", features)]:
        rec = {"icao": ap, "rows": len(g)}
        for w in CORRIDOR_WINDOWS_MIN:
            load = g[f"corr_dep_{w}m"]
            count = g[f"tko_all_{w}m"]
            ok = load.notna()
            rec[f"corr_{w}m"] = (
                float(np.corrcoef(load[ok], count[ok])[0, 1])
                if ok.sum() > 2
                else float("nan")
            )
            rec[f"zero_load_share_{w}m"] = float(((load == 0) & (count > 0))[ok].mean())
            rec[f"median_load_{w}m"] = float(load[ok].median())
        rec["placed"] = float(features.loc[g.index, "corr_route_weight"].notna().mean())
        rows.append(rec)
    return pd.DataFrame(rows)
