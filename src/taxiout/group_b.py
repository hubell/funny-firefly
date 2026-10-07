# SPDX-License-Identifier: GPL-3.0-only
"""Group B of the last week: the blends, the selector and the new base it builds.

Defined by §3 of ``reports/preregistered-2026-10-06-last-week.md`` (W, C10 to
C12 and their combination) as ``reports/preregistered-2026-10-06-airports.md``
amends it (every held-out month fitted, C13, C14 and the combination rewritten
around C14). ``scripts/candidates_board.py`` fits and forms the candidates
with these pieces and ``scripts/report_group_b.py`` reads them in
``reports/group-b.md``; ``scripts/make_submission.py`` builds the new base for
the ranking rows from them when ``taxiout.models.GROUP_B_APPLIED`` is set.

**The blend.** w x booster + (1 - w) x member, in that form, so that at
w = 0.5 it is the mean of the two to the bit and at any other weight it is the
expression the candidates were formed with.

**The menu** (airports amendment §3), each item a blend of two fitted parts:
the base (``combo4_booster`` and ``c3b_member`` at 0.5); the base's members at
the booster weights of C10's grid, named ``w`` and the weight; C11's blend
(``combo4_booster`` and ``c11_member`` at 0.5); C12's (``c12_booster`` and
``c3b_member``); C13's (``c13_booster`` and ``c3b_member``).

**The choice**, per airport and held-out month: the item with the lowest RMSE
over the airport's core rows of the other held-out months, the base unless
another is strictly lower, the first in menu order among equals. For the
ranking rows the choice uses all twelve held-out months. C10's weight is a
selection among four, and C14's forward legs let months after a leg's month
inform its choice; the copies and the report say both.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from taxiout.fusion import BASE_RECORDS as _CONFIGURATION_RECORDS
from taxiout.fusion import MONTHS

# The menu's parts: each item's booster key and member key, in menu order. The
# weight of a ``w`` item is its name's; every other item is at the half.
ITEM_PARTS: dict[str, tuple[str, str]] = {
    "base": ("combo4_booster", "c3b_member"),
    "c11": ("combo4_booster", "c11_member"),
    "c12": ("c12_booster", "c3b_member"),
    "c13": ("c13_booster", "c3b_member"),
}
WEIGHT_PREFIX = "w"
WEIGHT_PARTS = ("combo4_booster", "c3b_member")

# C14's held-out months at the ladder's seed, the new base's out-of-fold
# prediction for 2025, as ``reports/group-b.md`` derives them: the sha256 of
# each month's ``pred``. Their rows are the configuration's, month for month.
NEW_BASE_RECORDS: dict[str, str] = {
    "heldout-2025-01__c14__seed0": "144c67eb12f67be2c30cf9d92b44910ccd7b36f01523fd51c3465243798dfbf5",
    "heldout-2025-02__c14__seed0": "6384c2ab10c4f7ea398d8f90bcf8782e22ac51421cab4d058e4fb44110312325",
    "heldout-2025-03__c14__seed0": "4d928bee0161dbbbdc4693aaccec239620542c3caf61f00100d6de4f2587a56f",
    "heldout-2025-04__c14__seed0": "0696003ec9c41ebf87f244a5f6963e0a176c2140c66f97ba0e849e53bd75f420",
    "heldout-2025-05__c14__seed0": "87d9b93d2c90a2edb84e2759bbcf32fb975810330559c923993b1fb77355c3ef",
    "heldout-2025-06__c14__seed0": "4720a365eb78ac4eed4b695dba8213680e1dd01990085413821756fb502f2f7d",
    "heldout-2025-07__c14__seed0": "b951bc90bbc2cfeff818c90645f67869f2590b7e4e57066e8fc379ef4c0faaeb",
    "heldout-2025-08__c14__seed0": "e0388cc1fc8fddb93a50144486f851ab26eb0bdb406b5c21aabd0c156b288992",
    "heldout-2025-09__c14__seed0": "7ce4445cc77ec9d98a768ca47f7908f2ec190bb9e056990ecd468c85661fd693",
    "heldout-2025-10__c14__seed0": "d2283037dbafd0394812dc973c46a9ac51f679e9372ff7221cb1bf4c80924104",
    "heldout-2025-11__c14__seed0": "2da53eb136d832e424efdba4783767ed214e8fe5a268dc99655e3d116835a27b",
    "heldout-2025-12__c14__seed0": "41111aaca047dcd5a3ffd05850d3c1e62665973d7b5cd28efc5c0741d5c097d9",
}


def blend(booster, member, w: float) -> np.ndarray:
    """w x booster + (1 - w) x member; at 0.5 equal to the bit to the mean of the two."""
    return w * np.asarray(booster, dtype=float) + (1.0 - w) * np.asarray(
        member, dtype=float
    )


def weight_item(w: float) -> str:
    """The menu's name for the base's members at booster weight ``w``."""
    return f"{WEIGHT_PREFIX}{w:.1f}"


def item_blend(item: str) -> tuple[str, str, float]:
    """An item's booster key, member key and booster weight."""
    if item in ITEM_PARTS:
        booster, member = ITEM_PARTS[item]
        return booster, member, 0.5
    if item.startswith(WEIGHT_PREFIX):
        w = float(item.removeprefix(WEIGHT_PREFIX))
        if not 0.0 < w < 1.0:
            raise ValueError(f"not a weight: {item!r}")
        return WEIGHT_PARTS[0], WEIGHT_PARTS[1], w
    raise KeyError(f"not a menu item: {item!r}")


def join_by_airport(icao, parts: dict[str, np.ndarray]) -> np.ndarray:
    """Each airport's predictions at its own rows, in row order; every row exactly once."""
    icao = np.asarray(icao).astype(str)
    pred = np.full(len(icao), np.nan)
    filled = np.zeros(len(icao), dtype=int)
    for a, p in parts.items():
        pos = np.flatnonzero(icao == a)
        p = np.asarray(p, dtype=float)
        if len(pos) != len(p):
            raise ValueError(f"{a}: {len(p)} predictions for {len(pos)} rows")
        pred[pos] = p
        filled[pos] += 1
    if not np.all(filled == 1):
        raise ValueError(f"{int((filled != 1).sum())} rows not predicted exactly once")
    if not np.all(np.isfinite(pred)):
        raise ValueError("a joined prediction is not finite")
    return pred


def choose(
    sse: dict, items: list[str], airport: str, month: str | None, base_key: str = "base"
) -> tuple[str, dict]:
    """The lowest RMSE over the airport's core rows of the held-out months other than ``month``.

    ``sse`` is ``{item: {month: {airport: (squared error, rows)}}}``. ``month``
    None uses all twelve (the ranking rows). The base unless another item is
    strictly lower; among equal non-base items, the first in menu order.
    """
    scores = {}
    for it in items:
        s = n = 0
        for m, cells in sse[it].items():
            if m == month:
                continue
            s += cells[airport][0]
            n += cells[airport][1]
        scores[it] = float(np.sqrt(s / n)) if n else float("inf")
    best = base_key
    for it in items:
        if it != base_key and scores[it] < scores[best]:
            best = it
    return best, scores


def apply_choice(icao, base_pred, item_preds: dict, choice: dict[str, str]) -> np.ndarray:
    """The base's prediction with each airport's rows taken from its chosen item."""
    pred = np.asarray(base_pred, dtype=float).copy()
    icao = np.asarray(icao)
    for a, it in choice.items():
        sel = icao == a
        pred[sel] = np.asarray(item_preds[it], dtype=float)[sel]
    return pred


def base_per_airport(
    icao, parts: dict[str, np.ndarray], choice: dict[str, str]
) -> np.ndarray:
    """The new base for rows at ``icao``: each airport's chosen item, blended from ``parts``.

    ``parts`` maps a booster or member key to its prediction for every row;
    ``choice`` maps every airport the rows hold to a menu item.
    """
    icao = np.asarray(icao).astype(str)
    missing = sorted(set(icao) - set(choice))
    if missing:
        raise ValueError(f"no choice for {missing}")
    pred = np.full(len(icao), np.nan)
    for it in dict.fromkeys(choice[a] for a in sorted(set(icao))):
        booster, member, w = item_blend(it)
        sel = np.isin(icao, [a for a, x in choice.items() if x == it])
        pred[sel] = blend(parts[booster], parts[member], w)[sel]
    if not np.all(np.isfinite(pred)):
        raise ValueError("a row of the new base is not finite")
    return pred


def pred_sha256(x) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(x, dtype=float)).tobytes()
    ).hexdigest()


def read_new_base(board: Path, records: dict | None = None) -> pd.DataFrame:
    """C14's twelve held-out months at the ladder's seed, checked, as ``fusion.read_base`` reads the old base.

    Each file's ``pred`` must hash to ``NEW_BASE_RECORDS`` and to its record's,
    and hold the configuration's rows for that month.
    """
    records = NEW_BASE_RECORDS if records is None else records
    frames = []
    for mm in MONTHS:
        name = f"heldout-2025-{mm}__c14__seed0"
        rows = _CONFIGURATION_RECORDS[f"heldout-2025-{mm}__combo4__seed0"][0]
        rec = json.loads((board / f"{name}.json").read_text())
        kept = pd.read_parquet(board / f"{name}.parquet")
        got = pred_sha256(kept["pred"].to_numpy(dtype=float))
        if not (
            rec["pred_sha256"] == records[name] == got
            and rec["valid_rows"] == rows == len(kept)
        ):
            raise RuntimeError(f"{name}: its predictions are not the record's")
        k = kept.copy()
        k["month"] = mm
        frames.append(k)
    base = pd.concat(frames, ignore_index=True)
    if base["mvt_id"].duplicated().any():
        raise RuntimeError("a departure has two base values")
    return base[
        ["mvt_id", "icao", "runway", "takeoff", "term", "shift", "y", "pred", "month"]
    ]
