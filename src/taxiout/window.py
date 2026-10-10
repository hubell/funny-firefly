# SPDX-License-Identifier: GPL-3.0-only
"""W, the flight-plan window: the last step of the submitted prediction.

Defined by §1 of ``reports/preregistered-2026-10-09-records.md`` and measured by
``scripts/report_window.py`` (``reports/window.md``). Over every 2025 departure
with ``LOBT_flt`` present, the off-block time less the plan,
``BLOCK_TIME_UTC_mvt - LOBT_flt``, never exceeds ``W_STAR_S`` in absolute
value, and the rows near that edge stand as a wall rather than a tail thinning
into it, as a join condition on the two records would leave them. A
departure's recorded taxi-out is the lag ``L = MVT_TIME_UTC_mvt - LOBT_flt``
less that difference, so it lies in ``[L - W*, L + W*]``.

The ranking set's take-off times are redrawn within ``REDRAW_S`` seconds either
way, so the window applied to it is ``W' = W* + 12 s``: every ranking row with
``LOBT_flt`` present has its prediction clipped to ``[L - W', L + W']``, after
every other part of the submission, the two rules and the fusion included.
The clip cannot move a prediction away from a truth inside the interval. It
binds both ways: it raises a prediction far below the lag less ``W'`` and caps
any prediction, a day-scale hedge included, at the lag plus ``W'``. It
overrides the plan-shifted departures' rule where the two disagree, an
amendment of the 2 October rule's output that §1 of the note makes before any
figure and discloses. A row without ``LOBT_flt`` is unchanged. The floor at
zero and the rounding of the file follow the clip.

Public discussion of the challenge prompted the test, and the note says so;
the window is this entry's own measurement on its own 2025 data, with
thresholds the note fixed before any count existed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# §1 of the note: W* is the largest |BLOCK_TIME_UTC_mvt - LOBT_flt| over the
# 2025 departures with LOBT_flt, as ``reports/window.md`` measures it.
W_STAR_S = 3606
# §1 of the note: "the ranking set's take-off times are redrawn within
# ±12 s".
REDRAW_S = 12


def w_prime_s() -> int:
    """W' = W* + the redraw, the half-width of the window on the ranking rows."""
    return W_STAR_S + REDRAW_S


def lag_s(frame: pd.DataFrame) -> np.ndarray:
    """L = ``MVT_TIME_UTC_mvt - LOBT_flt`` in seconds; NaN where ``LOBT_flt`` is missing."""
    mvt = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
    lobt = pd.to_datetime(frame["LOBT_flt"], utc=True)
    return (mvt - lobt).dt.total_seconds().to_numpy(dtype=float)


def clip(pred, lag) -> np.ndarray:
    """``pred`` clipped to ``[L - W', L + W']`` where L exists; unchanged where it does not."""
    out = np.asarray(pred, dtype=float).copy()
    lag = np.asarray(lag, dtype=float)
    if out.shape != lag.shape:
        raise ValueError("the predictions and the lags differ in length")
    has = np.isfinite(lag)
    w = float(w_prime_s())
    out[has] = np.clip(out[has], lag[has] - w, lag[has] + w)
    return out


def apply(values, lag) -> tuple[np.ndarray, np.ndarray]:
    """The written values after the window: clipped, floored at zero, rounded.

    Returns the values and a mask of the rows the window moved. ``values`` are
    the file's values before the window, already floored and rounded, so a row
    the window leaves alone keeps its value exactly.
    """
    before = np.asarray(values, dtype=float)
    out = np.rint(np.clip(clip(before, lag), 0.0, None))
    return out, out != before


__all__ = ["REDRAW_S", "W_STAR_S", "apply", "clip", "lag_s", "w_prime_s"]
