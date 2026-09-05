# SPDX-License-Identifier: GPL-3.0-only
"""The identifier encoding, shrunk toward the geometry rather than toward a mean.

An empirical-Bayes encoding of the (airport, stand, runway) pair::

    encoded(pair) = (n * ybar_oof + k * g_hat) / (n + k)

``ybar_oof`` is the out-of-fold mean of taxi-out over that pair's *unflagged*
training rows, ``g_hat`` is the geometric unimpeded prediction for that pair
(``taxiout.unimpeded``), and ``k`` is estimated rather than chosen. The same
construction at stand level is carried as a second feature.

**Why the geometry is the prior.** Shrinking a thin cell's mean toward the
airport mean is the standard remedy, and the airport mean knows nothing about
this stand: it is the same number for a stand at the end of the field and a
stand beside the runway entry. The geometry is the only prior in this
repository that knows the first is far from that runway and the second is not.
``reports/baseline.md``'s strata table is what motivates it -- the top rung has
no deficit at all on pairs the training year never saw, where it already falls
back gracefully, and its deficit is on pairs seen once to ten times, where the
identifier exists but its mean is a handful of draws.

**What is shrunk.** Written as a shrinkage of the group's *residual* against
the prior rather than of its mean::

    encoded(row) = g_hat(row) + n / (n + k) * (ybar_oof - g_hat_bar)

with ``g_hat_bar`` the mean prior over the same rows ``ybar_oof`` averages. At
pair level the two forms are the same expression: ``g_hat`` is fitted with one
intercept per airport (``class_column=None``) so that it is a function of the
matrix cell alone and therefore constant within a pair, and the identity
``g_hat + n/(n+k) * (ybar - g_hat) = (n * ybar + k * g_hat) / (n + k)`` is
immediate. Written this way it also generalises to the stand level, where the
prior is not constant within the group because the stand's pairs route to
different runways, and it degrades correctly: a group with ``n = 0`` encodes to
the row's own ``g_hat`` exactly.

The prior carries the per-airport additive offset the ladder's geometric rungs
already carry (``+ airport offset`` in ``reports/baseline.md``): ``g_hat`` on
its own estimates *unimpeded* taxi-out and ``ybar`` is a mean of *actual*
taxi-out, so without it the two quantities differ by the whole excess term and
the prior would drag every thin pair low by several minutes. The offset is one
number per airport, fitted on the same rows as everything else here.

Estimating ``k``
----------------

``k = sigma2_within / sigma2_between`` is the Bayes weight, and both variances
are estimated **per airport**. A single ``k`` across ten airports would be
wrong: ``reports/baseline.md`` fits Schiphol's geometry at a very different
quality from Barcelona's, and an airport whose prior is poor must shrink
toward it less. ``sigma2_within`` is the pooled within-group variance of the
residual; ``sigma2_between`` is the variance of the group-level residual
``ybar - g_hat_bar`` **minus** the mean sampling variance ``E[sigma2_within/n]``
of the group means that enter it. Without that subtraction the between-variance
is the true between-variance plus the noise in the pair means, so it is
inflated and ``k`` comes out too low; the correction is reported beside the raw
number so the size of it is visible rather than asserted.

Both variances are moments of the *residual* ``y - g_hat``, not of ``y``. At
pair level the distinction is invisible, because ``g_hat`` is a function of the
matrix cell and is constant within a pair; at stand level it is the whole
difference, since a stand's pairs route to different runways and the moments of
``y`` would count every one of those geometric differences as within-group
noise. ``_group_moments`` says what that would cost.

``sigma2_within`` is not homoscedastic -- a longer route has more room to vary
-- so a length-dependent within-variance is fitted alongside the constant one
and the two are compared. The rule is fixed here rather than after the fact:
the length dependence is *material* if the fitted within-variance at the 90th
percentile of route length is at least ``LENGTH_RATIO_MATERIAL`` times its
value at the 10th, row-weighted across airports. Material, and ``k`` varies
with route length; not material, and the constant stands and the report says
so.

Two conditions on that ratio, which are part of the rule and not adjustments to
it. The ratio is read off the fit **before** ``MIN_VARIANCE_FRACTION`` is
applied, and an airport whose fitted variance is not positive at both ends of
the range does not vote: flooring first would pin the short end at the guard's
value whenever the true dependence is faster than linear, and the measured
ratio could then overstate the dependence but never understate it, which is a
check that cannot fail. And the answer is only read if the airports that do
vote carry at least half the rows, since a check decided by a minority of the
data has not been decided.

``k`` is the Bayes weight for predicting the target **marginally**. This
encoding is a feature inside a conditional model that also carries plan lags,
congestion and weather, so ``k`` is a principled default and not an optimum,
and it is not tuned: ``k_scale`` exists so the report can print the RMSE at
``k/2`` and ``2k`` as a sensitivity diagnostic, and the model uses the
estimated ``k`` whichever of the three scores best.

Leakage
-------

Four rules, each with a test in ``tests/test_encoding.py``, because any one of
them missed silently produces a *better*-looking wrong answer:

1. **The encoding is recomputed inside every fold.** ``encode_split`` takes the
   fold's own fit rows and knows nothing else; a pair held out of a fold has
   ``n = 0`` there and encodes to pure ``g_hat``. Computed once over all
   training rows and reused, every held-out pair would receive its own mean and
   the held-out-pair contrast would be fiction.
2. **``ybar`` is out of fold within training too**, by ``INNER_FOLDS``-fold
   over the fit rows: a row's encoding is built from the fit rows outside its
   own inner fold, so its own target never sits in its own feature. Inner folds
   are over *rows*, not groups -- grouped inner folds would give every training
   row ``n = 0`` and the feature would be constant.
3. **``ybar`` reads unflagged rows only.** The caller passes fit rows with the
   record-quality mask already removed. One defaulted block time at 50,000 s
   inside a three-row pair would otherwise destroy that pair's encoding.
4. **The prior is fitted on the fold's fit rows**, which are also the rows
   ``ybar`` averages. For a thin pair the effect is negligible -- its handful of
   rows against a fit of hundreds of thousands -- but it is an overlap, it is
   stated here and in ``reports/baseline.md``, and it is not hidden in a
   docstring a reviewer has to find.

At inference the encoding is computed once over the whole training year and
read by the 2026 ranking set. That is 2025 predicting 2026 across a boundary no
ranking row is on either side of, so there is no *split* fold to be out of and
no leakage of the ladder's kind to guard against; ``taxiout.features`` says the
same where the registry is documented.

``fit_encoding`` and ``FittedEncoding.transform`` are that path, and the
division of labour between them and ``encode_split`` is worth stating, because
getting it backwards is the one mistake here that makes a submission worse
without making anything fail:

* the **training** rows the submission model is fitted on are encoded by
  ``encode_split``, out of fold, exactly as in the ladder. Their targets *are*
  in the fitting sample, so rule 2 applies to them at submission as much as
  anywhere. Fitting on in-sample encodings would teach the booster to trust a
  column that is strictly more informative at fit than at use, and the model
  would lean on it and then be disappointed;
* the **ranking** rows are encoded by ``FittedEncoding.transform``, from the
  whole training sample. Their targets are not in the sample at all, so there
  is nothing to hold out of their lookup, and holding a fifth of the year out
  anyway would weaken the feature to prevent a leak that cannot occur.

Rule 3 is unchanged and applies to both: ``fit_rows`` excludes the flagged
rows. It matters more here than in the ladder, not less -- a defaulted block
time inside a thin pair would be that pair's encoding in the *submitted* model,
and no split downstream would catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from taxiout.unimpeded import (
    MIN_PAIR_COUNT,
    GeometricUnimpeded,
    fit_geometric,
    fit_pru_reference,
    lower_tail,
)

# The two grouping levels, and the feature each produces.
LEVELS: tuple[str, ...] = ("pair", "stand")
ENCODED_COLUMN: dict[str, str] = {
    "pair": "pair_encoded_s",
    "stand": "stand_encoded_s",
}
ENCODING_FEATURES: tuple[str, ...] = tuple(ENCODED_COLUMN[level] for level in LEVELS)

# Inner folds for the out-of-fold group means. Five leaves each row's encoding
# built from four fifths of its group, which is enough of the group to be the
# group's mean and never includes the row itself.
INNER_FOLDS = 5
SEED = 20260904

# Groups thin enough that their mean is a handful of draws are kept out of the
# between-variance estimate. ``taxiout.unimpeded.MIN_PAIR_COUNT`` already names
# this number for the same reason -- below it a pair statistic is not a
# statistic -- and it is reused rather than introduced.
MIN_GROUP_ROWS = MIN_PAIR_COUNT

# The pre-registered threshold for the length-dependence check. The within
# variance is length-dependent *materially* if the fitted variance at the 90th
# percentile of route length is at least this many times its value at the 10th.
LENGTH_RATIO_MATERIAL = 1.5

# A fitted within-variance may not fall below this fraction of the airport's
# constant one. A guard, not a fitted quantity: a linear fit extrapolated to a
# very short route can go negative, and a negative variance is an absurdity
# rather than a small number.
MIN_VARIANCE_FRACTION = 0.05

# The corrected between-group variance may not fall below this fraction of the
# uncorrected one. The same kind of guard, for the same kind of reason: where
# the group residuals are no more spread than the sampling noise in their own
# means, the subtraction can take the estimate to zero or past it, and a ``k``
# of infinity is not a number to put in a feature. At the floor the encoding is
# already shrinking twenty times harder than the uncorrected estimate would,
# which is "shrink to the prior" in every practical sense.
MIN_BETWEEN_FRACTION = 0.05


def group_key(features: pd.DataFrame, level: str) -> pd.Series:
    """The grouping key at ``level``, as a string series.

    ``pair`` is ``taxiout.features``' own pair column, built from the raw
    triple so that an unplaceable stand still forms a group. ``stand`` is the
    airport and the stand, which is the pair without the runway.
    """
    if level == "pair":
        return features["pair"].astype(str)
    if level == "stand":
        return features["icao"].astype(str) + "|" + features["stand"].astype(str)
    raise KeyError(f"unknown level {level!r}; expected one of {LEVELS}")


# --------------------------------------------------------------------------
# the prior


@dataclass(frozen=True)
class GeometricPrior:
    """``g_hat`` per row, and what it was built from.

    ``values`` is the geometric unimpeded prediction plus the per-airport
    additive offset that puts it on the same scale as observed taxi-out. The
    unshifted prediction is kept beside it so the report can print how large
    that offset is, since it is the excess term held as a constant and is the
    part of the prior the geometry does not explain.
    """

    values: pd.Series
    unshifted: pd.Series
    offset: pd.Series
    default_offset: float
    speed: dict[str, float]
    default_speed: float
    n_fit: int
    # The fitted estimator itself, so the prior can be evaluated on rows that
    # were not in the frame it was fitted from. The ladder never needs this --
    # every fold scores rows of its own frame -- but the submission does: the
    # ranking set is a different frame entirely, and a prior that can only
    # answer for the training rows cannot encode a 2026 departure at all.
    estimator: GeometricUnimpeded | None = None

    def prior_for(self, features: pd.DataFrame) -> pd.Series:
        """``g_hat`` for any frame, on the scale of observed taxi-out.

        The same two steps ``geometric_prior`` applies to its own frame: the
        geometric unimpeded prediction, plus the per-airport offset, with the
        pooled offset for an airport the fit never saw.
        """
        if self.estimator is None:
            raise ValueError(
                "this prior carries no estimator and can only answer for the "
                "frame it was fitted from"
            )
        raw = self.estimator.predict(features)
        icao = features["icao"].astype(str)
        return raw + icao.map(self.offset).astype(float).fillna(self.default_offset)


def geometric_prior(
    features: pd.DataFrame, y: pd.Series, fit_rows: pd.Series
) -> GeometricPrior:
    """Fit the geometric unimpeded term on ``fit_rows`` and offset it per airport.

    One intercept per airport rather than per wake class, so that the prior is
    a function of the matrix cell alone and is therefore constant within a
    pair; the encoding's stated form depends on that.

    ``fit_rows`` is the fold's training side with the record-quality rows
    already removed. Everything the prior knows comes from those rows.
    """
    mask = fit_rows.astype(bool)
    ref = fit_pru_reference(features, y, mask=mask)
    geo = fit_geometric(
        features, ref, lower_tail(features, y, ref, mask=mask), class_column=None
    )
    raw = geo.predict(features)
    icao = features["icao"].astype(str)
    residual = pd.to_numeric(y[mask], errors="coerce") - raw[mask]
    offset = residual.groupby(icao[mask], observed=True).mean()
    default = float(offset.mean()) if len(offset) else 0.0
    shifted = raw + icao.map(offset).astype(float).fillna(default)
    return GeometricPrior(
        values=shifted,
        unshifted=raw,
        offset=offset,
        default_offset=default,
        speed=dict(geo.speed),
        default_speed=float(geo.default_speed),
        n_fit=int(geo.n_fit),
        estimator=geo,
    )


# --------------------------------------------------------------------------
# the within-group variance


@dataclass(frozen=True)
class WithinVariance:
    """Pooled within-group variance per airport, constant and length-dependent.

    ``constant`` is the pooled within-group variance. ``intercept`` and
    ``slope`` are a weighted least squares of each group's own variance on its
    mean route length, per airport, weighted by that group's degrees of
    freedom, and ``span`` is the 10th and 90th percentile of route length the
    ratio is read between. ``ratio`` is the fitted variance at the 90th
    percentile over its value at the 10th, row-weighted across the airports
    whose fit is positive at both ends; ``decisive`` says whether those
    airports carry at least half the rows, and ``material`` is ``decisive``
    together with ``ratio`` against ``LENGTH_RATIO_MATERIAL``.
    """

    constant: dict[str, float]
    pooled: float
    intercept: dict[str, float] = field(default_factory=dict)
    slope: dict[str, float] = field(default_factory=dict)
    per_airport_ratio: dict[str, float] = field(default_factory=dict)
    span: dict[str, tuple[float, float]] = field(default_factory=dict)
    ratio: float = float("nan")
    decisive: bool = False
    decisive_weight: float = float("nan")
    material: bool = False
    groups: int = 0

    def at(self, icao: np.ndarray, length: np.ndarray) -> np.ndarray:
        """The within-group variance for each row.

        The constant one unless the length dependence was found material, in
        which case the fitted line, floored at ``MIN_VARIANCE_FRACTION`` of the
        airport's constant so an extrapolated fit cannot return a negative
        variance.
        """
        base = np.array([self.constant.get(a, self.pooled) for a in icao], dtype=float)
        if not self.material:
            return base
        b0 = np.array([self.intercept.get(a, np.nan) for a in icao], dtype=float)
        b1 = np.array([self.slope.get(a, np.nan) for a in icao], dtype=float)
        fitted = b0 + b1 * np.nan_to_num(length, nan=0.0)
        fitted = np.where(np.isfinite(fitted), fitted, base)
        return np.maximum(fitted, MIN_VARIANCE_FRACTION * base)


def _group_moments(
    codes: np.ndarray, n_groups: int, residual: np.ndarray, length: np.ndarray
) -> dict[str, np.ndarray]:
    """Count, and the first two moments of the residual, per group.

    The moments are of ``y - g_hat`` and not of ``y``, and the difference is
    not cosmetic. The hierarchical model behind the encoding is

        y_i = g_hat_i + delta_g + e_i,      e_i ~ (0, sigma2_within)

    -- a group *offset* against the prior, plus noise -- so ``sigma2_within``
    is the scatter of the residual within the group. At pair level the two
    readings coincide, because ``g_hat`` is a function of the matrix cell and
    is therefore constant within a pair. At stand level they do not: a stand's
    pairs route to different runways, ``g_hat`` differs between them by however
    much the routes differ, and taking the moments of ``y`` would count every
    one of those geometric differences as within-group noise. That inflates
    ``sigma2_within``, inflates ``k`` with it, and shrinks the stand encoding
    toward the prior for a reason that is an artefact of the arithmetic rather
    than a property of the data.

    Working on the residual also removes a numerical hazard. Taxi times are
    several hundred seconds and their residuals are a few tens, so
    ``sum(y^2) - sum(y)^2/n`` differences two large and nearly equal numbers
    where ``sum(d^2) - sum(d)^2/n`` does not.
    """
    return {
        "n": np.bincount(codes, minlength=n_groups).astype(float),
        "sum_d": np.bincount(codes, weights=residual, minlength=n_groups),
        "sum_d2": np.bincount(codes, weights=residual * residual, minlength=n_groups),
        "sum_len": np.bincount(codes, weights=length, minlength=n_groups),
    }


def within_variance(
    airport_of_group: np.ndarray,
    moments: dict[str, np.ndarray],
    *,
    ratio_material: float = LENGTH_RATIO_MATERIAL,
) -> WithinVariance:
    """Pooled within-group variance per airport, and the length-dependence check.

    Groups with a single row carry no within-group information and are left
    out of both. The length fit is a weighted least squares of the group
    variance on the group's mean route length, weighted by ``n - 1``, which is
    the weight a pooled variance already gives each group.
    """
    n = moments["n"]
    usable = n >= 2
    dof = np.where(usable, n - 1.0, 0.0)
    ssq = np.where(
        usable, moments["sum_d2"] - moments["sum_d"] ** 2 / np.maximum(n, 1), 0.0
    )
    ssq = np.maximum(ssq, 0.0)
    length = np.where(n > 0, moments["sum_len"] / np.maximum(n, 1), np.nan)

    pooled = float(ssq.sum() / dof.sum()) if dof.sum() > 0 else float("nan")
    constant: dict[str, float] = {}
    intercept: dict[str, float] = {}
    slope: dict[str, float] = {}
    ratios: dict[str, float] = {}
    span: dict[str, tuple[float, float]] = {}
    weights: dict[str, float] = {}
    all_weight = 0.0

    for a in np.unique(airport_of_group[usable]):
        sel = usable & (airport_of_group == a)
        d = dof[sel].sum()
        if d <= 0:
            continue
        constant[a] = float(ssq[sel].sum() / d)
        all_weight += float(n[sel].sum())
        # The per-group variance against the group's route length.
        s2 = ssq[sel] / dof[sel]
        x = length[sel]
        w = dof[sel]
        ok = np.isfinite(x) & np.isfinite(s2)
        if ok.sum() < 3 or np.nanstd(x[ok]) <= 0:
            continue
        design = np.column_stack([np.ones(int(ok.sum())), x[ok]])
        rt = np.sqrt(w[ok])
        beta, *_ = np.linalg.lstsq(design * rt[:, None], s2[ok] * rt, rcond=None)
        lo, hi = (float(q) for q in np.nanquantile(x[ok], [0.10, 0.90]))
        v_lo = float(beta[0] + beta[1] * lo)
        v_hi = float(beta[0] + beta[1] * hi)
        intercept[a], slope[a] = float(beta[0]), float(beta[1])
        span[a] = (lo, hi)
        # The ratio is read off the *unfloored* fit, and an airport whose fitted
        # variance is not positive across the range does not vote. Flooring
        # first would have made this test one-sided: a straight line through a
        # variance that grows faster than linearly has a negative intercept, so
        # ``v_lo`` would be pinned at the floor and the ratio would be an
        # artefact of the guard -- able to overstate the length dependence and
        # never to understate it. A fit that predicts a negative variance at the
        # short end is not evidence of a 1.5-fold ratio; it is evidence that the
        # straight line is the wrong shape, and that airport abstains.
        if v_lo > 0 and v_hi > 0:
            ratios[a] = v_hi / v_lo
            weights[a] = float(n[sel].sum())

    total = sum(weights.values())
    ratio = (
        float(sum(ratios[a] * weights[a] for a in ratios) / total)
        if total > 0
        else float("nan")
    )
    # A check decided by a minority of the data is not a check. Half the rows
    # have to sit at an airport whose fit is usable before the answer is read.
    decisive = all_weight > 0 and total >= 0.5 * all_weight
    return WithinVariance(
        constant=constant,
        pooled=pooled,
        intercept=intercept,
        slope=slope,
        per_airport_ratio=ratios,
        span=span,
        ratio=ratio,
        decisive=bool(decisive),
        decisive_weight=float(total / all_weight) if all_weight > 0 else float("nan"),
        material=bool(decisive and np.isfinite(ratio) and ratio >= ratio_material),
        groups=int(usable.sum()),
    )


# --------------------------------------------------------------------------
# the shrinkage weight


@dataclass(frozen=True)
class Shrinkage:
    """``k`` per airport, and the two variances behind it.

    ``raw_between`` is the variance of the group residuals before the sampling
    variance of the group means is subtracted, and ``correction`` is what was
    subtracted. Both are kept so the report can print the size of a correction
    that is otherwise invisible and that moves ``k`` by more than a tenth.
    """

    level: str
    k: dict[str, float]
    default_k: float
    within: WithinVariance
    between: dict[str, float]
    raw_between: dict[str, float]
    correction: dict[str, float]
    groups_used: dict[str, int]
    pooled_between: float
    floored: tuple[str, ...] = ()
    min_group_rows: int = MIN_GROUP_ROWS

    def k_for(self, icao: np.ndarray, length: np.ndarray) -> np.ndarray:
        """The shrinkage weight per row.

        Constant per airport unless the within variance was found materially
        length-dependent, in which case ``k`` moves with the route.
        """
        s2w = self.within.at(icao, length)
        between = np.array(
            [self.between.get(a, self.pooled_between) for a in icao], dtype=float
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = s2w / between
        return np.where(np.isfinite(out) & (out > 0), out, self.default_k)

    def k_span(self, airport: str) -> tuple[float, float]:
        """``k`` at the 10th and 90th percentile of that airport's route length.

        ``k`` is the constant-variance number, which is what the feature uses
        while the length dependence is not material and is *not* what it uses
        once it is. Reporting the constant alone in that case would print a
        quantity the model uses nowhere, so the report prints this span beside
        it. Where the dependence is not material both ends are ``k`` itself.
        """
        one = self.k.get(airport, self.default_k)
        span = self.within.span.get(airport)
        if not self.within.material or span is None:
            return one, one
        icao = np.array([airport, airport], dtype=object)
        low, high = self.k_for(icao, np.array(span, dtype=float))
        return float(low), float(high)


def estimate_shrinkage(
    level: str,
    airport_of_group: np.ndarray,
    moments: dict[str, np.ndarray],
    *,
    min_group_rows: int = MIN_GROUP_ROWS,
    ratio_material: float = LENGTH_RATIO_MATERIAL,
) -> Shrinkage:
    """``k = sigma2_within / sigma2_between`` per airport, sampling variance removed.

    ``sigma2_between`` is ``Var(ybar - g_hat_bar)`` over the groups with at
    least ``min_group_rows`` rows, minus the mean of ``sigma2_within / n`` over
    the same groups. The subtraction is the correction the estimator needs:
    the observed spread of group means is the spread of the true group means
    *plus* the noise in estimating each one, and leaving the second in inflates
    the between-variance and shrinks ``k``.

    An airport whose corrected between-variance falls to or below
    ``MIN_BETWEEN_FRACTION`` of its uncorrected one -- its group residuals are
    barely more spread than sampling noise alone would make them -- takes that
    floor and is listed in ``floored``. The floor is a guard and not a fitted
    quantity: the reading is that the geometry explains that airport's groups
    to within the noise, so the encoding should shrink almost entirely to the
    prior, and "almost entirely" is a number where "entirely" is a division by
    zero.
    """
    n = moments["n"]
    within = within_variance(airport_of_group, moments, ratio_material=ratio_material)

    keep = n >= min_group_rows
    length = np.where(n > 0, moments["sum_len"] / np.maximum(n, 1), np.nan)
    resid = np.where(n > 0, moments["sum_d"] / np.maximum(n, 1), np.nan)
    s2w = within.at(airport_of_group, length)

    raw_between: dict[str, float] = {}
    correction: dict[str, float] = {}
    between: dict[str, float] = {}
    used: dict[str, int] = {}
    floored: list[str] = []

    # Pooled, over every qualifying group with its airport's mean removed, so
    # a difference in level between airports does not enter the variance.
    pooled_resid = []
    pooled_correction = []
    for a in np.unique(airport_of_group[keep]):
        sel = keep & (airport_of_group == a)
        if sel.sum() < 2:
            continue
        used[a] = int(sel.sum())
        raw_between[a] = float(np.var(resid[sel], ddof=1))
        correction[a] = float(np.mean(s2w[sel] / n[sel]))
        pooled_resid.append(resid[sel] - float(np.mean(resid[sel])))
        pooled_correction.append(s2w[sel] / n[sel])

    if pooled_resid:
        stacked = np.concatenate(pooled_resid)
        pooled_raw = float(np.var(stacked, ddof=1))
        pooled_corr = float(np.mean(np.concatenate(pooled_correction)))
        pooled_between = max(pooled_raw - pooled_corr, MIN_BETWEEN_FRACTION * pooled_raw)
    else:
        pooled_between = float("nan")

    for a, raw in raw_between.items():
        value = raw - correction[a]
        floor = MIN_BETWEEN_FRACTION * raw
        if value > floor:
            between[a] = value
        else:
            between[a] = floor if np.isfinite(floor) and floor > 0 else pooled_between
            floored.append(a)

    default_k = (
        float(within.pooled / pooled_between)
        if np.isfinite(pooled_between) and pooled_between > 0
        else float("nan")
    )
    k = {
        a: float(within.constant.get(a, within.pooled) / v)
        for a, v in between.items()
        if v > 0
    }
    return Shrinkage(
        level=level,
        k=k,
        default_k=default_k,
        within=within,
        between=between,
        raw_between=raw_between,
        correction=correction,
        groups_used=used,
        pooled_between=pooled_between,
        floored=tuple(sorted(floored)),
        min_group_rows=min_group_rows,
    )


# --------------------------------------------------------------------------
# the encoding itself


def _inner_folds(n_rows: int, folds: int, seed: int) -> np.ndarray:
    """A fold index per fit row, dealt at random over rows.

    Over rows and not over groups: a grouped inner fold would remove every one
    of a row's group-mates along with the row, leave ``n = 0`` for every
    training row, and make the feature the prior everywhere in training while
    the validation rows carried a real mean. The row is the unit that must be
    excluded, and the fold is what excludes it.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, folds, n_rows)


def encode_level(
    features: pd.DataFrame,
    y: pd.Series,
    prior: pd.Series,
    fit_rows: pd.Series,
    level: str,
    *,
    k_scale: float = 1.0,
    folds: int = INNER_FOLDS,
    seed: int = SEED,
    min_group_rows: int = MIN_GROUP_ROWS,
) -> tuple[pd.Series, Shrinkage]:
    """One level's encoded column for every row of ``features``, and its ``k``.

    Rows in ``fit_rows`` are encoded out of fold: from the fit rows of the
    other ``folds - 1`` inner folds, so a row's own target is never in its own
    encoding. Every other row -- the fold's validation rows, and the flagged
    training rows the model is not fitted on -- is encoded from all the fit
    rows, which is what a row outside the training sample is entitled to.
    """
    keys = group_key(features, level)
    icao = features["icao"].astype(str).to_numpy()
    length = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    ghat = pd.to_numeric(prior, errors="coerce").to_numpy(dtype=float)

    codes, levels = pd.factorize(keys, use_na_sentinel=False)
    n_groups = len(levels)

    # Only fit rows with a usable target and a usable prior contribute: the
    # group's mean residual has to be a mean over the same rows on both sides.
    fit = fit_rows.to_numpy(dtype=bool) & np.isfinite(target) & np.isfinite(ghat)
    fit_codes = codes[fit]
    fit_d = target[fit] - ghat[fit]
    fit_len = np.nan_to_num(length[fit], nan=0.0)
    moments = _group_moments(fit_codes, n_groups, fit_d, fit_len)

    # The airport each group belongs to, for the per-airport variances.
    airport_of_group = np.empty(n_groups, dtype=object)
    airport_of_group[:] = ""
    first = np.full(n_groups, len(codes))
    np.minimum.at(first, codes, np.arange(len(codes)))
    seen = first < len(codes)
    airport_of_group[seen] = icao[first[seen]]

    # ``k`` over all the fit rows. This is the estimate the report prints and
    # the one a submission would use, where there are no folds to be out of.
    shrink = estimate_shrinkage(
        level, airport_of_group, moments, min_group_rows=min_group_rows
    )

    # Per (group, inner fold) sums, so a row's out-of-fold quantities are the
    # totals less its own fold's -- for the group mean *and* for ``k``. ``k`` is
    # a per-airport scalar over hundreds of thousands of rows, so a single row
    # moves it by nothing measurable; it is still computed out of fold, because
    # "moves it by nothing measurable" is an argument and "the row's own target
    # is not in the arithmetic" is a property, and the property is testable.
    fold_of_fit = _inner_folds(int(fit.sum()), folds, seed)
    cell = fit_codes * folds + fold_of_fit
    size = n_groups * folds
    per_fold = {
        "n": np.bincount(cell, minlength=size).astype(float),
        "sum_d": np.bincount(cell, weights=fit_d, minlength=size),
        "sum_d2": np.bincount(cell, weights=fit_d * fit_d, minlength=size),
        "sum_len": np.bincount(cell, weights=fit_len, minlength=size),
    }
    per_fold = {key: v.reshape(n_groups, folds) for key, v in per_fold.items()}

    own = np.zeros(len(codes), dtype=int)
    own[fit] = fold_of_fit
    n_out = moments["n"][codes] - np.where(fit, per_fold["n"][codes, own], 0.0)
    d_out = moments["sum_d"][codes] - np.where(fit, per_fold["sum_d"][codes, own], 0.0)

    k = shrink.k_for(icao, length)
    for f in range(folds):
        rows = fit & (own == f)
        if not rows.any():
            continue
        out_of_fold = {key: v - per_fold[key][:, f] for key, v in moments.items()}
        fold_shrink = estimate_shrinkage(
            level, airport_of_group, out_of_fold, min_group_rows=min_group_rows
        )
        k[rows] = fold_shrink.k_for(icao[rows], length[rows])
    k = k * float(k_scale)

    # ``ybar - g_hat_bar`` is the group's mean residual, which the moments carry
    # directly; the encoding is the row's own prior plus a shrunken share of it.
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = n_out / (n_out + k)
        shift = np.where(
            n_out > 0, weight * (d_out / np.where(n_out > 0, n_out, 1.0)), 0.0
        )
    encoded = ghat + shift
    return pd.Series(encoded, index=features.index, dtype=float), shrink


@dataclass(frozen=True)
class SplitEncoding:
    """Both encoded columns for one split, and everything they were built from."""

    columns: pd.DataFrame
    shrinkage: dict[str, Shrinkage]
    prior: GeometricPrior
    k_scale: float
    folds: int
    fit_rows: int


def encode_split(
    features: pd.DataFrame,
    y: pd.Series,
    fit_rows: pd.Series,
    *,
    prior: GeometricPrior | None = None,
    k_scale: float = 1.0,
    folds: int = INNER_FOLDS,
    seed: int = SEED,
    min_group_rows: int = MIN_GROUP_ROWS,
) -> SplitEncoding:
    """Both encoded columns for one fold, from that fold's fit rows alone.

    ``fit_rows`` is the fold's training side with the record-quality rows
    removed -- the rows the model is fitted on, and the only rows any of this
    may read a target from. Call it once per fold: reusing one call's columns
    across folds is the defect this whole module is arranged to prevent, and
    the one that makes the number look better.

    ``prior`` is fitted here unless passed in; passing it in is for the
    sensitivity diagnostic, which varies ``k_scale`` at a fixed prior so the
    two effects do not move together.
    """
    if prior is None:
        prior = geometric_prior(features, y, fit_rows)
    columns: dict[str, pd.Series] = {}
    shrinkage: dict[str, Shrinkage] = {}
    for level in LEVELS:
        values, shrink = encode_level(
            features,
            y,
            prior.values,
            fit_rows,
            level,
            k_scale=k_scale,
            folds=folds,
            seed=seed,
            min_group_rows=min_group_rows,
        )
        columns[ENCODED_COLUMN[level]] = values.astype("float32")
        shrinkage[level] = shrink
    return SplitEncoding(
        columns=pd.DataFrame(columns, index=features.index),
        shrinkage=shrinkage,
        prior=prior,
        k_scale=float(k_scale),
        folds=folds,
        fit_rows=int(fit_rows.sum()),
    )


@dataclass(frozen=True)
class FittedEncoding:
    """The training year's encoding, ready to apply to rows outside it.

    The ladder never needs this. Every fold there encodes rows of its own
    frame, out of fold, and ``encode_split`` does it. The submission is the
    other case and the one this exists for: the ranking set is 2026, the
    fitting sample is 2025, and a 2026 departure has to be encoded by *looking
    up* what the training year says about its group rather than by being
    inside the computation.

    That distinction is the whole of the design, so it is worth stating which
    rows get which. A training row is encoded out of fold, by ``encode_split``,
    because its own target is in the sample. A ranking row is encoded from the
    whole training sample, by ``transform`` here, because its target is not in
    the sample at all -- there is nothing to be out of fold of, and holding a
    fifth of the training year out of its lookup would weaken the feature for
    no leakage it prevents. Fitting on out-of-fold columns and predicting from
    full-sample ones is the standard arrangement and the only one where the
    column means the same thing at fit and at use.

    ``moments`` holds, per level, the group keys and the two sums the encoding
    needs: the number of contributing fit rows and the sum of their residuals
    against the prior. A group the training year never saw is absent, which
    yields ``n = 0`` and an encoding that is the bare prior -- the same
    fallback a held-out pair gets in the ladder, reached by the same
    arithmetic rather than by a separate branch.
    """

    prior: GeometricPrior
    moments: dict[str, pd.DataFrame]
    shrinkage: dict[str, Shrinkage]
    fit_rows: int

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Both encoded columns for ``features``, by lookup into the fit sample.

        Deterministic and stateless: no folds, no seed, no randomness of any
        kind. The same frame and the same encoder give the same columns, which
        is what ``tests/test_encoding.py`` asserts, because a submission whose
        feature depends on when it was computed is not reproducible.
        """
        ghat = self.prior.prior_for(features)
        icao = features["icao"].astype(str).to_numpy()
        length = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
        out: dict[str, pd.Series] = {}
        for level in LEVELS:
            table = self.moments[level]
            keys = group_key(features, level)
            n = keys.map(table["n"]).to_numpy(dtype=float)
            sum_d = keys.map(table["sum_d"]).to_numpy(dtype=float)
            n = np.nan_to_num(n, nan=0.0)
            sum_d = np.nan_to_num(sum_d, nan=0.0)
            k = self.shrinkage[level].k_for(icao, length)
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = n / (n + k)
                shift = np.where(n > 0, weight * (sum_d / np.where(n > 0, n, 1.0)), 0.0)
            values = ghat.to_numpy(dtype=float) + shift
            out[ENCODED_COLUMN[level]] = pd.Series(
                values, index=features.index, dtype="float32"
            )
        return pd.DataFrame(out, index=features.index)


def fit_encoding(
    features: pd.DataFrame,
    y: pd.Series,
    fit_rows: pd.Series,
    *,
    prior: GeometricPrior | None = None,
    min_group_rows: int = MIN_GROUP_ROWS,
) -> FittedEncoding:
    """Fit the encoding over a whole training sample, for use at inference.

    ``fit_rows`` is the training side with the record-quality rows removed --
    the same rule guard 3 states for the ladder, and it matters at least as
    much here: one defaulted block time inside a thin pair would otherwise be
    that pair's encoding in the submitted model, where nothing downstream
    would catch it.
    """
    if prior is None:
        prior = geometric_prior(features, y, fit_rows)
    ghat = pd.to_numeric(prior.values, errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    length = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
    icao = features["icao"].astype(str).to_numpy()
    fit = fit_rows.to_numpy(dtype=bool) & np.isfinite(target) & np.isfinite(ghat)

    moments: dict[str, pd.DataFrame] = {}
    shrinkage: dict[str, Shrinkage] = {}
    for level in LEVELS:
        keys = group_key(features, level)
        codes, levels = pd.factorize(keys, use_na_sentinel=False)
        n_groups = len(levels)
        group_moments = _group_moments(
            codes[fit],
            n_groups,
            target[fit] - ghat[fit],
            np.nan_to_num(length[fit], nan=0.0),
        )
        airport_of_group = np.empty(n_groups, dtype=object)
        airport_of_group[:] = ""
        first = np.full(n_groups, len(codes))
        np.minimum.at(first, codes, np.arange(len(codes)))
        seen = first < len(codes)
        airport_of_group[seen] = icao[first[seen]]
        shrinkage[level] = estimate_shrinkage(
            level, airport_of_group, group_moments, min_group_rows=min_group_rows
        )
        moments[level] = pd.DataFrame(
            {"n": group_moments["n"], "sum_d": group_moments["sum_d"]},
            index=pd.Index(levels, name=level),
        )
    return FittedEncoding(
        prior=prior,
        moments=moments,
        shrinkage=shrinkage,
        fit_rows=int(fit.sum()),
    )


def with_encoding(
    features: pd.DataFrame, encoding: SplitEncoding | pd.DataFrame | None
) -> pd.DataFrame:
    """``features`` with the encoded columns overlaid, sharing the other columns.

    A shallow copy: the ninety-odd existing columns are not duplicated, only
    referenced, so a fold's overlay costs the two new columns and nothing else.

    Accepts either a ``SplitEncoding`` -- the ladder's per-fold object -- or a
    bare frame of the two columns, which is what ``FittedEncoding.transform``
    returns for rows outside the fitting sample. Both carry the same columns
    under the same names; only how they were computed differs, and that
    difference belongs to the caller rather than to the overlay.
    """
    if encoding is None:
        return features
    columns = encoding if isinstance(encoding, pd.DataFrame) else encoding.columns
    out = features.copy(deep=False)
    for name, values in columns.items():
        out[name] = values
    return out


__all__ = [
    "ENCODED_COLUMN",
    "ENCODING_FEATURES",
    "INNER_FOLDS",
    "LENGTH_RATIO_MATERIAL",
    "LEVELS",
    "MIN_GROUP_ROWS",
    "FittedEncoding",
    "GeometricPrior",
    "Shrinkage",
    "SplitEncoding",
    "WithinVariance",
    "encode_level",
    "encode_split",
    "estimate_shrinkage",
    "fit_encoding",
    "geometric_prior",
    "group_key",
    "with_encoding",
    "within_variance",
]
