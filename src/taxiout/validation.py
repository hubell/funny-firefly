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
training, dealt round-robin within each airport so every fold holds out a
random third of every airport's pairs -- a volume-representative third, not
the rare ones. The fold then evaluates twice from *one* fitted model: on the
held-out pairs, and on a random sample of rows whose pairs stayed in training
and which were themselves excluded from the fit. Same model, same period,
same airports -- the only difference is whether the pair was seen.

The claim is a difference in differences. What the geometry buys on seen
pairs is one number; what it buys on held-out pairs is another; the claim is
that the second exceeds the first, because on a seen pair an identifier model
already knows everything the geometry could tell it. Read that way, whatever
composition difference exists between the two row sets cancels, because it
enters both rungs alike. The held-out unit is the *pair*: every stand and
every runway is still seen in other pairs, so this bounds what the geometry
buys on a new pair and does not isolate a new stand.

**Uncertainty.** The difference in differences is a mean over three folds,
and rows within a pair share a stand, a runway and a route, so they are not
independent. ``cluster_bootstrap`` resamples *pairs* with replacement from the
stored residuals -- no refit -- and reports a percentile interval on the
fold-mean statistic. It is the same machinery for every contrast the report
makes, on every rung the ladder will ever have.

**Pair frequency.** Test rows bucketed by how often their pair appeared in
training: 0, 1-10, 11-100, more than 100. Geometry should buy most in the
first two buckets. If it does not, the physical layer is decoration, and the
write-up says so rather than burying it. The rare-pair question is answered
here, by the strata, and not by the held-out folds, which are not rare.

Every metric is reported per airport beside the pooled figure, and with and
without the record-quality flags, so the physical model's contribution stays
visible underneath the record-quality component's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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

# Draws for the cluster bootstrap. A thousand puts the 2.5th and 97.5th
# percentiles on about 25 draws each, which is coarse but honest for an
# interval that is read to the nearest second; it costs nothing, since no
# model is refitted.
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_LEVEL = 0.95

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


# --------------------------------------------------------------------------
# cluster bootstrap over pairs


@dataclass
class ClusterSums:
    """One rung's residuals on one row set, reduced to per-pair sums.

    ``sse[k]`` and ``n[k]`` are the squared-residual sum and row count of the
    ``k``-th pair, in the same pair order for every rung of the same row set.
    A bootstrap draw is then a draw of pair indices, and an RMSE is
    ``sqrt(sse[draw].sum() / n[draw].sum())`` -- the whole resample costs one
    fancy-index per rung rather than a pass over the rows.
    """

    sse: np.ndarray
    n: np.ndarray

    @property
    def rmse(self) -> float:
        total = float(self.n.sum())
        return float(np.sqrt(self.sse.sum() / total)) if total else float("nan")

    def resampled(self, draw: np.ndarray) -> float:
        total = float(self.n[draw].sum())
        return float(np.sqrt(self.sse[draw].sum() / total)) if total else float("nan")


def cluster_sums(
    residuals: Mapping[str, np.ndarray | pd.Series], clusters: np.ndarray | pd.Series
) -> dict[str, ClusterSums]:
    """Per-pair sums of squared residuals for each rung on one row set.

    Rows where any rung's residual is not finite are dropped for every rung,
    so the rungs are compared on the same rows.
    """
    codes, _ = pd.factorize(pd.Series(np.asarray(clusters)), use_na_sentinel=True)
    keep = codes >= 0
    arrays = {k: np.asarray(v, dtype=float) for k, v in residuals.items()}
    for v in arrays.values():
        keep &= np.isfinite(v)
    codes = codes[keep]
    if not codes.size:
        return {k: ClusterSums(np.zeros(0), np.zeros(0)) for k in arrays}
    n_clusters = int(codes.max()) + 1
    n = np.bincount(codes, minlength=n_clusters).astype(float)
    return {
        k: ClusterSums(
            sse=np.bincount(codes, weights=v[keep] ** 2, minlength=n_clusters), n=n
        )
        for k, v in arrays.items()
    }


# One fold of a bootstrap: for each named row set ("held", "seen"), the
# per-pair sums of every rung on that set.
FoldSums = Mapping[str, Mapping[str, ClusterSums]]

# A statistic reads one fold's RMSEs, as side -> rung -> RMSE, to a number.
Statistic = Callable[[Mapping[str, Mapping[str, float]]], float]


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    low: float
    high: float
    draws: int

    def __str__(self) -> str:
        return f"{self.point:+,.1f} s [{self.low:+,.1f}, {self.high:+,.1f}]"


def fold_rmses(fold: FoldSums, draws: Mapping[str, np.ndarray] | None = None) -> dict:
    """Each rung's RMSE per row set, from the stored sums; resampled if ``draws``."""
    out: dict[str, dict[str, float]] = {}
    for side, rungs in fold.items():
        draw = None if draws is None else draws[side]
        out[side] = {
            k: (s.rmse if draw is None else s.resampled(draw)) for k, s in rungs.items()
        }
    return out


def cluster_bootstrap(
    folds: Sequence[FoldSums],
    statistic: Statistic,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    level: float = BOOTSTRAP_LEVEL,
    seed: int = SEED,
) -> BootstrapResult:
    """Percentile interval for the fold-mean of ``statistic`` over resampled pairs.

    Each draw resamples the pairs of every row set of every fold with
    replacement -- the same draw for every rung on a row set, so paired
    contrasts stay paired -- recomputes each rung's RMSE from the stored
    per-pair sums, evaluates ``statistic`` on each fold and averages over
    folds. The point estimate is the statistic on the unresampled data. No
    model is fitted here, which is what makes a thousand draws free.
    """
    if not folds:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0)
    point = float(np.mean([statistic(fold_rmses(f)) for f in folds]))
    rng = np.random.default_rng(seed)
    sizes = [
        {side: len(next(iter(rungs.values())).n) for side, rungs in f.items()}
        for f in folds
    ]
    stats = np.empty(draws)
    for b in range(draws):
        values = []
        for f, size in zip(folds, sizes):
            draw = {
                side: rng.integers(0, k, k) if k else np.zeros(0, int)
                for side, k in size.items()
            }
            values.append(statistic(fold_rmses(f, draw)))
        stats[b] = float(np.mean(values))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(stats, [alpha, 1.0 - alpha])
    return BootstrapResult(point, float(lo), float(hi), draws)


def gap(rung: str, held: str = "held", seen: str = "seen") -> Statistic:
    """Held-out minus seen RMSE for one rung."""

    def stat(r: Mapping[str, Mapping[str, float]]) -> float:
        return r[held][rung] - r[seen][rung]

    return stat


def difference_in_differences(
    with_rung: str, without_rung: str, held: str = "held", seen: str = "seen"
) -> Statistic:
    """How much more ``with_rung`` beats ``without_rung`` on held-out pairs than on seen.

    Positive means the added block buys more where the pair is new. Read
    as (gap without) - (gap with): the composition of the two row sets
    enters both rungs alike and cancels.
    """

    def stat(r: Mapping[str, Mapping[str, float]]) -> float:
        held_delta = r[held][without_rung] - r[held][with_rung]
        seen_delta = r[seen][without_rung] - r[seen][with_rung]
        return held_delta - seen_delta

    return stat
