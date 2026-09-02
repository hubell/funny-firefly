# SPDX-License-Identifier: GPL-3.0-only
"""Splits and metrics, fixed before any model was fitted.

The splits are the argument, not a formality, and *Validation* in
``docs/modelling_notes.md`` says why. The geometry block is an injective
encoding of (stand, runway): every column of the matrix is a deterministic
function of that pair, so a saturated one-hot interaction already contains
everything the geometry knows about a pair it has seen. With a million
departures a year against a few thousand used cells there are hundreds of
training rows per cell, so a random K-fold would put the same pair on both
sides of every split, show the geometry "working", and test nothing. The value
of the physical layer is entirely that it behaves on pairs it has *not* seen.

Three splits, therefore, and each answers a different question.

**Forward in time.** Train January-November 2025 and validate December, at all
ten airports: the winter proxy for a January test month that the training year
cannot hold out forward, its only January being its first month. Train
January-June and validate July, reported for Frankfurt, Heathrow and Schiphol
separately, because those three are the whole of the July test set and 53% of
the test rows. Never a random K-fold on rows: congestion is autocorrelated
over hours, so neighbouring departures share queue state.

**Held-out pairs.** Whole (airport, stand, runway) pairs are removed from
training, balanced across airports so every fold holds out pairs everywhere.
The fold then evaluates twice from *one* fitted model: on the held-out pairs,
and on a random sample of rows whose pairs stayed in training and which were
themselves excluded from the fit. Same model, same period, same airports --
the only difference is whether the pair was seen. The gap between those two
numbers is the entry's central claim, and running both from one fit is what
makes it a clean comparison rather than two experiments differing in several
things at once.

**Pair frequency.** Test rows bucketed by how often their pair appeared in
training: 0, 1-10, 11-100, more than 100. Geometry should buy most in the
first two buckets. If it does not, the physical layer is decoration, and the
write-up says so rather than burying it.

Every metric is reported per airport beside the pooled figure, and with and
without the record-quality flags, so the physical model's contribution stays
visible underneath the record-quality component's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The forward splits, fixed on 2 September 2026 once the ranking set's shape
# was known. Written out rather than derived so that changing one is a visible
# edit to a constant.
DECEMBER_TRAIN = tuple(f"2025-{m:02d}" for m in range(1, 12))
DECEMBER_VALID = ("2025-12",)
JULY_TRAIN = tuple(f"2025-{m:02d}" for m in range(1, 7))
JULY_VALID = ("2025-07",)

# The three airports July 2026 contains, and so the only ones the July split
# reports separately.
JULY_AIRPORTS = ("EDDF", "EGLL", "EHAM")

PAIR_FOLDS = 3
SEEN_FRACTION = 0.05
SEED = 20260902

FREQUENCY_BUCKETS = ("0", "1-10", "11-100", ">100")


@dataclass
class Split:
    """One fit and what to evaluate it on.

    ``valid`` is the primary evaluation set. ``extra`` carries any further
    evaluation sets scored from the *same* fitted model -- the held-out-pair
    folds put their seen-pair comparison there, which is what makes the gap a
    within-model contrast.
    """

    name: str
    description: str
    train: pd.Series
    valid: pd.Series
    extra: dict[str, pd.Series] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = int((self.train & self.valid).sum())
        if overlap:
            raise ValueError(f"{self.name}: {overlap} rows are in both sides")
        for key, mask in self.extra.items():
            bad = int((self.train & mask).sum())
            if bad:
                raise ValueError(f"{self.name}/{key}: {bad} rows are also in train")

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.sum()),
            "valid": int(self.valid.sum()),
            **{k: int(v.sum()) for k, v in self.extra.items()},
        }


def month_of(takeoff: pd.Series) -> pd.Series:
    """``YYYY-MM`` of the take-off time, in UTC."""
    return pd.to_datetime(takeoff, utc=True).dt.strftime("%Y-%m")


def forward_split(
    month: pd.Series,
    train_months: tuple[str, ...],
    valid_months: tuple[str, ...],
    *,
    name: str,
    description: str,
) -> Split:
    return Split(
        name=name,
        description=description,
        train=month.isin(train_months),
        valid=month.isin(valid_months),
    )


def december_split(month: pd.Series) -> Split:
    return forward_split(
        month,
        DECEMBER_TRAIN,
        DECEMBER_VALID,
        name="december",
        description=(
            "train January-November 2025, validate December, all ten airports "
            "-- the winter proxy for the January test month"
        ),
    )


def july_split(month: pd.Series) -> Split:
    return forward_split(
        month,
        JULY_TRAIN,
        JULY_VALID,
        name="july",
        description=(
            "train January-June 2025, validate July -- the three airports the "
            "July test set contains are reported separately"
        ),
    )


def assign_pair_folds(
    icao: pd.Series, pair: pd.Series, *, n_folds: int = PAIR_FOLDS, seed: int = SEED
) -> pd.Series:
    """Fold index per row, dealing whole pairs round-robin within each airport.

    Dealing within the airport rather than over the pooled pair list keeps
    every fold's holdout spread across all ten airports; pooled dealing would
    let one fold take a disproportionate share of Istanbul's several hundred
    pairs and turn a per-airport figure into noise.
    """
    keys = pd.DataFrame({"icao": icao.astype(str), "pair": pair.astype(str)})
    pairs = keys.drop_duplicates().sort_values(["icao", "pair"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(pairs), dtype=int)
    for ap in pairs["icao"].unique():
        m = (pairs["icao"] == ap).to_numpy()
        n = int(m.sum())
        deal = np.arange(n) % n_folds
        fold[m] = rng.permutation(deal)
    lookup = pd.Series(fold, index=pairs["pair"].to_numpy())
    return pd.Series(lookup.reindex(keys["pair"]).to_numpy(), index=icao.index, dtype=int)


def held_out_pair_splits(
    icao: pd.Series,
    pair: pd.Series,
    *,
    n_folds: int = PAIR_FOLDS,
    seen_fraction: float = SEEN_FRACTION,
    seed: int = SEED,
    eligible: pd.Series | None = None,
) -> list[Split]:
    """One split per fold, each carrying a matched seen-pair evaluation set.

    ``eligible`` restricts which rows may be fitted or evaluated on -- the
    runner passes the training year so the ranking months cannot leak in.
    """
    ok = pd.Series(True, index=icao.index) if eligible is None else eligible.astype(bool)
    fold = assign_pair_folds(icao, pair, n_folds=n_folds, seed=seed)
    rng = np.random.default_rng(seed + 1)
    draw = pd.Series(rng.random(len(icao)), index=icao.index)
    splits = []
    for k in range(n_folds):
        unseen = ok & (fold == k)
        rest = ok & ~unseen
        seen = rest & (draw < seen_fraction)
        splits.append(
            Split(
                name=f"pairs-{k + 1}",
                description=(
                    f"fold {k + 1} of {n_folds}: whole (airport, stand, runway) "
                    "pairs held out of training, scored against a held-out "
                    "sample of rows whose pairs were seen"
                ),
                train=rest & ~seen,
                valid=unseen,
                extra={"seen": seen},
            )
        )
    return splits


def rmse(y: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


def frequency_bucket(counts: pd.Series) -> pd.Series:
    """Pair-frequency buckets, as ``FREQUENCY_BUCKETS``."""
    c = pd.to_numeric(counts, errors="coerce").fillna(0)
    return pd.Series(
        np.select(
            [c <= 0, c <= 10, c <= 100],
            list(FREQUENCY_BUCKETS[:3]),
            default=FREQUENCY_BUCKETS[3],
        ),
        index=counts.index,
    )


def evaluate(
    y: pd.Series,
    pred: pd.Series,
    *,
    icao: pd.Series,
    flagged: pd.Series | None = None,
) -> pd.DataFrame:
    """Pooled and per-airport RMSE, with and without the flagged rows.

    ``flagged`` is the record-quality mask from ``taxiout.tails``. Where it is
    absent the ex-flag columns repeat the pooled figures and the report says
    the flags were unavailable, rather than quietly reporting one number twice
    under two headings.
    """
    f = (
        pd.Series(False, index=y.index)
        if flagged is None
        else flagged.astype(bool).reindex(y.index).fillna(False)
    )
    frame = pd.DataFrame(
        {
            "y": y.astype(float),
            "p": pred.astype(float),
            "icao": icao.astype(str),
            "flag": f,
        }
    )
    rows = []
    for label, g in [("all", frame)] + list(frame.groupby("icao", sort=True)):
        keep = ~g["flag"]
        rows.append(
            {
                "icao": label,
                "rows": len(g),
                "flagged": int(g["flag"].sum()),
                "rmse_s": rmse(g["y"], g["p"]),
                "rmse_ex_flagged_s": rmse(g.loc[keep, "y"], g.loc[keep, "p"]),
                "bias_s": float((g["p"] - g["y"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_by_bucket(
    y: pd.Series,
    pred: pd.Series,
    counts: pd.Series,
    *,
    keep: pd.Series | None = None,
) -> pd.DataFrame:
    """RMSE stratified by how often the row's pair was seen in training.

    ``keep`` restricts the rows scored. The runner passes the rows the
    record-quality term does not cover: this table exists to compare rungs
    across frequency buckets, and a bucket holding one defaulted block time
    reports that record rather than anything about the rungs.
    """
    bucket = frequency_bucket(counts)
    ok = pd.Series(True, index=y.index) if keep is None else keep.astype(bool)
    rows = []
    for b in FREQUENCY_BUCKETS:
        m = (bucket == b) & ok
        rows.append(
            {
                "bucket": b,
                "rows": int(m.sum()),
                "rmse_s": rmse(y[m], pred[m]) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)
