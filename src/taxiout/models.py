# SPDX-License-Identifier: GPL-3.0-only
"""The ablation ladder, and the estimators each of its rungs uses.

The ladder is the decomposition of ``docs/modelling_notes.md`` read bottom-up.
Each rung adds one term, and the rung-to-rung difference is what the write-up
reports:

1. **airport mean** -- the null model the data review already measured.
2. **PRU reference** -- the organisers' own empirical unimpeded time, the
   10th percentile per (stand, runway), with its cascade for pairs it has not
   seen often enough.
3. **geometric unimpeded** -- the same quantity from route length and shape,
   alone, with no learned model above it. It is on the ladder as a predictor
   in its own right rather than only as a feature, because a physical term
   that cannot beat an airport mean unaided is not a physical term.
4. **identifiers and time** -- a gradient-boosted model on the airport, stand
   and runway as categoricals plus the take-off clock and the flight plan's
   delay signal. This is the saturated baseline: on a pair it has seen, it
   already contains everything the geometry could tell it.
5. **identifiers, time and geometry** -- the same model with the matrix block
   and its quality flags added. Rung 4 to rung 5 is the claim, and it is only
   readable on the held-out-pair split, where rung 4's advantage of having
   memorised the pair is taken away.
6. **plus the aircraft** -- wake class and type as categoricals. On a seen
   pair the stand absorbs the class, because heavies park at wide-body
   stands; on a new pair nothing does, so this too is read on the held-out
   split. The geometric rung has a twin for the same reason: one intercept
   per airport, then one per (airport, wake class).
7. **plus the plan lags** -- take-off minus a planning value fixed before
   departure (the first and last off-block estimates and the schedule), so
   taxi plus the pushback delay against the plan; the subject's own and the
   mean over the other departures on its runway within fifteen minutes. It
   sits before the congestion blocks so that what they are worth is read net
   of it, and the congestion rungs are kept without it as a second chain at
   the foot of the ladder, so the reader sees both.
8. **plus the excess term, one block at a time** -- runway throughput,
   arrivals, scheduled demand, runway configuration, and the shared corridor
   (Gate 2). Each is its own rung so the report shows what each is worth,
   and each is built from what the ranking set exposes about surrounding
   movements: never a departure's block time or taxi time. The corridor sits
   last, after throughput, so that its rung says whether it is a new column
   or the take-off count renamed.
9. **plus weather** -- the regime flags per airport-hour (Gate 2b), the one
   block from outside the challenge data. Skipped by the runner when the
   archive is not cached, and said so, rather than fitted on a missing column.
10. **plus the shrunk identifier encoding**, in three arms read against the
   weather rung rather than against each other. The pair and the stand as
   empirical-Bayes encodings shrunk toward the geometric unimpeded prediction
   (``taxiout.encoding``): added alongside the identifier block, replacing the
   stand within it, and replacing both the stand and the runway. The control
   is the weather rung itself, unchanged. The second and third arms exist
   because removing the model's capacity to memorise a thin pair may
   generalise better than giving it a smoothed version of the same thing, and
   because "replacing the pair categorical" has two readings in a frame where
   ``pair`` is not itself a feature -- so both are run rather than one being
   guessed at.
11. **the Gate 2c blocks, four rungs in a chain on top of the chosen arm** --
   stand occupancy, the callsign prefix, the destination geometry with the
   holiday calendar, and the two plan differences. They sit on the arm rather
   than on the weather rung because the arm is what the submission is fitted
   from, and each is read against the rung below it in the chain so the report
   says what each is worth net of the others. They are the fields the model
   never read: the exposed arrival columns that bound the hidden one through
   the stand, the operator as the callsign carries it, where the aircraft is
   going, and the calendar.
12. **plus runway saturation** -- the cheap Gate 3: from the other departures'
   take-off times on the subject's runway, the share of consecutive gaps
   under ninety seconds, the smallest gap, and the span of the last three.
   Measured against the ceiling in ``reports/congestion.md``, found to be
   worth nothing, and therefore **not** in the submission. The rung stays on
   the ladder because a cheap queue proxy that was built, measured and bought
   nothing is a result. It is read against the weather rung it was added to,
   not against whatever rung the submission has since moved to.

Each rung is scored twice, once plain and once with the record-quality term on
top. That term lives in ``taxiout.tails`` and is deliberately not reimplemented
here: it is not a model of taxiing but the conditional mean of a cell that
mixes ordinary departures with defaulted block times, which is the
RMSE-optimal prediction under the stated scoring rule and nothing else
(*Tail handling*). It is applied by ``taxiout.tails.combine`` as an overlay on
the rows it covers rather than fitted jointly, so the physical model
underneath stays legible and the two can never be confused for one another.
Every rung here is fitted with those rows excluded and evaluated with them in.

**No submission-driven tuning.** The booster's hyperparameters are written out
below. They were set once, for a compute budget, before any RMSE existed, and
then chosen once more by the one permitted sweep: fitted on January to
September 2025 and scored on October 2025, a month none of the ladder's
splits reads, over one grid, with the October number recorded beside them.
There is no early stopping on the validation set, because stopping on the set
you report is tuning on it; the number of rounds is fixed. Nothing here has
been or will be chosen by a leaderboard score, and nothing was chosen on
December, July or the held-out pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxiout.calendar import HOLIDAY_FEATURES
from taxiout.callsign import (
    CALLSIGN_FEATURES,
    OTHER_LEVEL,
    frequent_levels,
    pool_rare,
)
from taxiout.congestion import (
    ARRIVAL_FEATURES,
    CONFIGURATION_FEATURES,
    COVERAGE_FEATURES,
    DEMAND_FEATURES,
    THROUGHPUT_FEATURES,
)
from taxiout.corridor import CORRIDOR_FEATURES
from taxiout.destination import DESTINATION_FEATURES
from taxiout.encoding import ENCODING_FEATURES
from taxiout.metar import WEATHER_FEATURES
from taxiout.occupancy import OCCUPANCY_FEATURES, OCCUPANCY_TFREE_FEATURES
from taxiout.planlags import PLAN_DIFF_FEATURES, PLAN_LAG_FEATURES
from taxiout.saturation import SATURATION_FEATURES
from taxiout.features import (
    AIRCRAFT_FEATURES,
    CATEGORICAL_FEATURES,
    GEOMETRY_FEATURES,
    PLAN_FEATURES,
    QUALITY_FEATURES,
    TIME_FEATURES,
)

# The identifier block the ladder's saturated rung uses. ``pair`` is
# deliberately not here: as a categorical it has tens of thousands of levels,
# which no boosted tree splits usefully, and the airport-stand-runway triple
# recovers the interaction through successive splits. It stays in the frame as
# the key the PRU reference and the held-out-pair split are built on.
IDENTIFIER_BLOCK: tuple[str, ...] = ("icao", "stand", "designator")

# What a model may keep at an airport it has never seen: everything in the top
# rung that is not an identifier. The airport, the stand and the runway all go
# -- at an unseen airport none of the three has a level the model has met --
# and what remains is fitted across the other airports and applied through the
# held-out airport's own geometry. ``scripts/report_transfer.py`` reads this.
TRANSFERABLE_EXCLUDED: tuple[str, ...] = ("icao", "stand", "designator", "pair")

# Fixed once, for a compute budget, before any RMSE existed. L2 because the
# metric is RMSE. No early stopping: see the module docstring.
# Chosen by the two permitted sweeps (scripts/sweep_params.py), both of which
# fit January to September 2025 and score October 2025 -- a month the ladder
# never reads -- over the rows the record-quality term leaves to the physical
# model. THE TWO SWEEPS' OCTOBER FIGURES ARE NOT COMPARABLE WITH EACH OTHER,
# and the paragraph after next says why.
#
# The first sweep, 3 September 2026: trees (300, 600, 1000), learning rate
# (0.03, 0.06, 0.1), leaves (31, 63, 127) and minimum child samples
# (50, 200, 500), on the saturation rung's features -- the top rung when that
# sweep ran, before that block was dropped from the submission -- fitted on
# the unflagged training rows, with no four-hour exclusion and no shrunk
# encoding, because that is how the ladder fitted then. The cell it chose,
# (1000 trees, 0.03, 127 leaves, 50), scored 235.0 s on October; that grid ran
# from 235.0 to 244.8 s and its top eight cells lie within 0.8 s of each other.
# The parameters that stood before it (500 trees, 0.06, 63 leaves, 200) sit
# between the 238.2 s of their 300-tree and the 236.8 s of their 600-tree
# neighbours in that grid. Its terms are kept in the sweep script under
# FIRST_SWEEP, which no longer runs them.
#
# The second sweep, 21 September 2026, pre-registered on 20 September and
# permitted once: learning rate (0.015, 0.03, 0.06), leaves (63, 127, 255) and
# minimum child samples (20, 50, 200), twenty-seven cells, each fitted once at
# 3,000 trees and read at (300, 600, 1000, 1500, 2000, 3000), on the SUBMISSION
# rung's features under the fit rules that stand: fit_rows with the four-hour
# exclusion, the shrunk encoding on those rows, the record-quality model on the
# training months, and no recency schedule. The cell below is that sweep's, and
# it scored 214.90 s on October; that grid ran from 214.90 to 235.44 s. The
# cell that stood before it, (1000 trees, 0.03, 127 leaves, 50), scores 217.28 s
# under the same rules in the same run, so the move is worth 2.38 s against a
# seed floor of 0.19 s measured from five single-seed fits of that cell in that
# run. Its day-clustered interval against the cell it replaced is
# [-2.89, -1.89] s, printed because the sweep could have chosen a cell whose
# interval spanned zero; the rule that chose it reads the floor and nothing
# else. The chosen cell sits on an edge of this grid on all four axes, which is
# recorded here because the pre-registration fixes that the grid is not
# extended again whatever it shows.
#
# WHY 235.0 AND 214.9 CANNOT BE SUBTRACTED. The first figure is the saturation
# rung with no shrunk encoding and no four-hour exclusion; the second is the
# submission rung with both. The same parameter cell moves from 235.0 s to
# 217.28 s on the rules alone, before any parameter changes. What the second
# sweep bought is 217.28 to 214.90, and nothing else here is a difference
# between two comparable numbers.
#
# Both grids' numbers stay under data/processed/, gitignored, and are not
# reported anywhere else.
BOOSTER_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 3000,
    "learning_rate": 0.015,
    "num_leaves": 255,
    "min_child_samples": 20,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "subsample_freq": 1,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "random_state": 20260902,
    # Repeatable to the bit, which the booster was not: two ladder runs on one
    # machine agreed byte for byte on none of the fifty-two residual files
    # (``reports/linux-run2-vs-run1.txt``), and fifteen fits of one rung gave
    # fifteen different models (``reports/determinism.md``). ``deterministic``
    # makes LightGBM's threaded sums -- the initial score and the per-leaf
    # totals -- run in a fixed order; without it the order in which threads
    # combine them varies from run to run and moves whole trees. Forcing the
    # column-wise histograms removes the second source: left to itself LightGBM
    # times both layouts at the start of every fit and keeps the faster, so the
    # layout could change from one fit to the next. Column-wise rather than
    # row-wise because, on the rung measured, it gave the same bits at 8 and at
    # 14 threads and the same bits as the old parameters at one thread;
    # row-wise repeats only at a fixed thread count. It was chosen on that
    # property before any ladder figure under it existed, and costs between a
    # quarter and a third more per fit than the setting it replaces at the same
    # thread count.
    "deterministic": True,
    "force_col_wise": True,
    # Eight threads, and part of the result rather than only of the speed. On
    # December's ``identifiers`` rung the column-wise setting gives the same
    # bits at 1, 2, 4, 8 and 14 threads, and that does NOT generalise. Fitted at
    # one thread instead, December's 25 rungs split three ways: nine are
    # bit-identical to their eight-thread fits, thirteen differ only in the last
    # bits of a sum taken in another order, and three grow different trees --
    # ``weather-nolag`` by 158.3 s, ``throughput`` by 40.3 s and
    # ``corridor-nolag`` by 4.97 s on a flight. At four threads, on December,
    # only ``corridor-nolag`` grows different trees; seven differ in the last
    # bits and seventeen are bit-identical. July's 25 rungs, fitted at one
    # thread, grow no different trees at all: four differ in the last bits and
    # twenty-one are bit-identical. No printed RMSE moves at the precision the
    # reports carry, so where it happens it is invisible in a report and real
    # in the model. Most rungs therefore survive a change of thread count and
    # some do not, which is why every reported run pins this value rather than
    # trusting the ones that do.
    #
    # Which rungs move is unexplained. All six rungs measured with LightGBM's
    # dataset construction logged build a multi-value bin group, the three that
    # move and the three that do not alike, and the sparse rate does not
    # separate them either, so neither is what distinguishes them.
    #
    # What it is NOT is the machine's core count. Two of the rungs that move at
    # one thread were refitted with ``n_jobs`` left at eight and the process
    # confined first to four CPUs and then to two, and both gave the ladder's
    # own digests every time: the row blocks follow the parameter and not the
    # cores on offer. That was measured by narrowing one machine rather than by
    # running on another, so what it settles is that the core count is not part
    # of the result -- this pins a thread count, not a host.
    # ``reports/threads-2026-09-12.md`` is the measurement and
    # ``scripts/fast_lane.py`` is what makes it; fits at other counts belong
    # there rather than here.
    #
    # The -1 it replaces meant the machine's logical CPU count, which made the
    # fit depend on whatever host it ran on.
    "n_jobs": 8,
    "verbose": -1,
}

# Seed averaging, measured three times and NOT yet adopted. The submission rung
# is fitted five times
# with nothing changed but the seed and the predictions averaged. It is an
# ESTIMATOR CHOICE and not a rung of the ladder: it changes how the model is
# fitted and not what it reads, so the ladder stays single-seed -- an ablation
# is about blocks, and averaging every rung would spend five times the compute
# to move every number by the same amount.
#
# Measured three times, against a rule fixed before the first of them: adopt
# only if the average gains at least one second on BOTH forward splits'
# ex-flagged RMSE. On the rung below the submission it gained 0.78 and 0.98 and
# was not adopted. On the submission rung it gained 1.10 and 1.37 and was. And
# once the recorded-taxi tail came out of every fit
# (``TAIL_EXCLUSION_ADOPTED_H``) it gained 0.81 and 0.56, and by the same rule
# it was **not** adopted. Refitted once the booster was made repeatable it
# gained 0.86 and 0.54, and refitted on 19 September with the encoding on the
# same rows as the booster it gains 0.48 and 0.52.
#
# The one-second rule was retired on 19 September, before any measurement under
# the rule that replaced it existed: its basis was the run-to-run variation the
# ladder showed before the booster was made repeatable, and that no longer
# holds. The rule now: adopt it if, on both forward legs, the five-seed average
# beats every one of the five single seeds and its gain over the ladder's seed
# clears that leg's seed floor, and measure it on the three July airports and
# on October, both printed, before it ships. On 19 September both legs meet the
# first half, 0.48 and 0.52 against floors of 0.34 and 0.23, and the second has
# not been measured, so it is not yet adopted.
#
# What came out of the gain between the second measurement and the latest is
# worth stating at the size it holds: 56% of it in December (1.10 -> 0.48) and
# 62% in July (1.37 -> 0.52), as ``reports/baseline.md`` prints them on 19
# September. Three things changed in between -- the tail came out of the fit,
# the booster was made repeatable, and the encoding was fitted on the booster's
# rows -- and nothing measured separates them.
#
# ``reports/baseline.md``'s *Seed averaging* section is the measurement and
# carries a reproducibility check: the first of the seeds is the one the ladder
# itself fits with, so its predictions reproduce the ladder's to the last bit.
BAGGING_SEEDS = 5

# Whether the submission actually averages those seeds. The rule above decides
# it and this constant records the decision, so that
# ``scripts/make_submission.py`` cannot ship a model the report says was not
# adopted. It came apart once over the fit rows and the fix was the same: name
# it once and import it.
SEED_AVERAGE_ADOPTED: bool = False


# The top rung's feature block, named once so the rungs that build on it say
# so rather than repeating fifteen tuples and drifting from one another.
WEATHER_BLOCK: tuple[str, ...] = (
    IDENTIFIER_BLOCK
    + TIME_FEATURES
    + PLAN_FEATURES
    + GEOMETRY_FEATURES
    + QUALITY_FEATURES
    + AIRCRAFT_FEATURES
    + PLAN_LAG_FEATURES
    + THROUGHPUT_FEATURES
    + COVERAGE_FEATURES
    + ARRIVAL_FEATURES
    + DEMAND_FEATURES
    + CONFIGURATION_FEATURES
    + CORRIDOR_FEATURES
    + WEATHER_FEATURES
)

# The arm the submission is fitted from, named once. The rung that defines it
# and the four Gate 2c rungs that build on it all read this tuple, so an edit to
# the arm cannot leave the rungs above it standing on the old block -- which is
# a failure that would show up as four rungs quietly measuring something else.
ENCODING_STAND_BLOCK: tuple[str, ...] = (
    tuple(c for c in WEATHER_BLOCK if c != "stand") + ENCODING_FEATURES
)

# Categoricals whose rare levels are pooled to ``other`` at fit time, counted on
# the rows the model is fitted on. This is a property of the *fitted estimator*
# and not of the feature frame, which is why it lives here rather than in
# ``taxiout.features``: it is the same arrangement the geometric fit uses for
# the wake class, where a class with too few rows at an airport is pooled inside
# ``fit_geometric`` and the fitted object maps an unseen one to ``other`` at
# prediction. The callsign prefix is the one column that needs it -- some three
# hundred airline designators, most of them a handful of flights a year -- and
# pooling keeps the ranking set from presenting a level the model never saw.
POOLED_CATEGORICALS: tuple[str, ...] = CALLSIGN_FEATURES


@dataclass(frozen=True)
class Rung:
    """One step of the ladder."""

    key: str
    label: str
    kind: str  # "airport-mean" | "pru" | "geometric" | "booster"
    features: tuple[str, ...] = ()
    note: str = ""
    # Keyword options for the rung's fitter, as pairs so the dataclass stays
    # frozen and hashable. The geometric rungs use it to say which class
    # column, if any, the intercept reads.
    options: tuple[tuple[str, object], ...] = ()
    # The rung this one is read against. ``None`` is the rung before it on
    # the ladder; a key names another, for the rungs kept as a second chain.
    base: str | None = None
    # A rung that is fitted and scored but is not a step of anything: it
    # answers one question beside the ladder rather than standing on it. The
    # report's ladder, held-out-pair and frequency tables skip these, because a
    # diagnostic printed in a column headed "vs below" reads as a step, and the
    # section that owns the question prints it instead. Never in
    # ``GATE_2C_RUNGS`` and never ``SUBMISSION_KEY``.
    diagnostic: bool = False

    @property
    def uses_geometry(self) -> bool:
        """Whether this rung reads the geometry at all, however indirectly.

        ``ENCODING_FEATURES`` counts. The shrunk encodings are not in
        ``GEOMETRY_FEATURES`` -- they are their own block -- but the quantity
        they shrink toward is ``g_hat``, so a rung that carries them carries the
        geometry with an extra step in front of it. Classifying them as
        geometry-free would let an ablation call itself "geometry removed" while
        the geometry was still arriving through the prior, and the contrast it
        reported would shrink as an artefact of that definition rather than as a
        finding. The claim contrast is measured between two rungs that carry no
        encoding either way, so nothing currently printed moves; this keeps that
        true if the claim is ever re-anchored above the arms.
        """
        return self.kind == "geometric" or bool(
            set(self.features) & (set(GEOMETRY_FEATURES) | set(ENCODING_FEATURES))
        )


LADDER: tuple[Rung, ...] = (
    Rung(
        key="airport-mean",
        label="airport mean",
        kind="airport-mean",
        note="the null model; one number per airport",
    ),
    Rung(
        key="pru",
        label="PRU reference",
        kind="pru",
        note=(
            "10th percentile of taxi-out per (airport, stand, runway), "
            "cascading to (airport, runway) then airport below the minimum count"
        ),
    ),
    Rung(
        key="geometric",
        label="geometric unimpeded",
        kind="geometric",
        options=(("class_column", None),),
        note=(
            "an intercept per airport plus taxi_m over a per-airport speed, "
            "modulated by route shape; no learned model above it"
        ),
    ),
    Rung(
        key="geometric-class",
        label="geometric unimpeded + wake class",
        kind="geometric",
        options=(("class_column", "wake_class"),),
        note="the same, with the intercept fitted per (airport, wake class)",
    ),
    Rung(
        key="identifiers",
        label="identifiers + time",
        kind="booster",
        features=IDENTIFIER_BLOCK + TIME_FEATURES + PLAN_FEATURES,
        note="the saturated baseline: airport, stand and runway as categoricals",
    ),
    Rung(
        key="identifiers-geometry",
        label="identifiers + time + geometry",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES,
        note="the same model with the matrix block and its join-quality flags",
    ),
    Rung(
        key="identifiers-geometry-aircraft",
        label="identifiers + time + geometry + aircraft",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES,
        note="the same model with the wake class and the aircraft type",
    ),
    Rung(
        key="plan-lags",
        label="+ plan lags",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + PLAN_LAG_FEATURES,
        note=(
            "take-off minus a planning value fixed before departure -- the first "
            "and last off-block estimates and the schedule -- so taxi plus the "
            "pushback delay against the plan; the subject's own and the mean over "
            "the other departures on its runway within fifteen minutes"
        ),
    ),
    # Gate 2: the excess term, one block at a time, each on top of the last,
    # read net of the plan lags. The coverage flags ride with the first block
    # that needs them.
    Rung(
        key="throughput",
        label="+ runway throughput",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + PLAN_LAG_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES,
        note="take-offs from the subject's runway and from all runways, trailing",
    ),
    Rung(
        key="arrivals",
        label="+ arrivals",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + PLAN_LAG_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES,
        note="landings, on-blocks, and arrivals' taxi-in times, trailing",
    ),
    Rung(
        key="demand",
        label="+ demand",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + PLAN_LAG_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES
        + DEMAND_FEATURES,
        note="scheduled departures around the subject's schedule",
    ),
    Rung(
        key="configuration",
        label="+ configuration",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + PLAN_LAG_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES
        + DEMAND_FEATURES
        + CONFIGURATION_FEATURES,
        note="the runways in use in the trailing hour, and how long they have held",
    ),
    Rung(
        key="corridor",
        label="+ shared corridor",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + PLAN_LAG_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES
        + DEMAND_FEATURES
        + CONFIGURATION_FEATURES
        + CORRIDOR_FEATURES,
        note=(
            "other movements in the window weighted by how much of the subject's "
            "own taxiway route theirs shares"
        ),
    ),
    Rung(
        key="weather",
        label="+ weather",
        kind="booster",
        features=WEATHER_BLOCK,
        note="visibility band, precipitation and freezing, per airport-hour (Gate 2b)",
    ),
    # The three encoding arms. Each is read against the weather rung and not
    # against the arm before it: they are alternatives, not steps, and reading
    # one against another would report the difference between two ways of
    # spending the same block as though it were a block of its own.
    Rung(
        key="encoding-added",
        label="+ shrunk encoding",
        kind="booster",
        features=WEATHER_BLOCK + ENCODING_FEATURES,
        base="weather",
        note=(
            "the pair and the stand as empirical-Bayes encodings shrunk toward "
            "the geometric unimpeded prediction, added alongside the identifier "
            "block"
        ),
    ),
    Rung(
        key="encoding-for-stand",
        label="+ shrunk encoding, replacing the stand",
        kind="booster",
        features=ENCODING_STAND_BLOCK,
        base="weather",
        note=(
            "the same, with the stand categorical removed: the encodings replace "
            "the level the model would memorise a thin pair through, while the "
            "runway stays because it is an operational variable and not a way of "
            "identifying one cell"
        ),
    ),
    Rung(
        key="encoding-for-identifiers",
        label="+ shrunk encoding, replacing stand and runway",
        kind="booster",
        features=tuple(c for c in WEATHER_BLOCK if c not in ("stand", "designator"))
        + ENCODING_FEATURES,
        base="weather",
        note=(
            "the wider reading of the same arm: both halves of the pair removed, "
            "leaving the airport, so nothing but the encodings carries the cell"
        ),
    ),
    # Gate 2c: four blocks on top of the chosen arm, as a chain. Each is read
    # against the one before it and the first against the arm, so a rung's
    # number is what that block is worth net of the blocks below it. They sit on
    # the arm and not on the weather rung because the arm is the rung the
    # submission is fitted from, and a block measured against a rung the model
    # does not use would be measuring a model nobody runs.
    Rung(
        key="occupancy",
        label="+ stand occupancy",
        kind="booster",
        features=ENCODING_STAND_BLOCK + OCCUPANCY_FEATURES,
        base="encoding-for-stand",
        note=(
            "the gap from the last arrival's on-block at the subject's stand to "
            "its take-off, that arrival's lateness, taxi-in, type and operator "
            "against the subject's, and the gap to the previous departure from "
            "the same stand -- a stand holds one aircraft, so an arrival that "
            "on-blocked there before the subject took off bounds the subject's "
            "taxi-out from below"
        ),
    ),
    # A diagnostic beside the block above, not a step of the chain. The
    # question it answers is what the stand-occupancy block is worth if it is
    # forbidden the subject's take-off time: the same arrival fields, but the
    # arrival is chosen at or before the subject's SCHEDULED time and every
    # column is read from that choice. Dropping the columns whose arithmetic
    # names the take-off would not have answered it -- the take-off also picks
    # which arrival the arithmetic is about -- so the anchor moves instead.
    # Read against the same rung the full block is read against, so the two
    # gains are comparable; the difference between them is what the take-off
    # anchor buys, and it is the part of the block that is admissible only
    # under the challenge's post-hoc framing.
    Rung(
        key="occupancy-tfree",
        label="+ stand occupancy, schedule-anchored",
        kind="booster",
        features=ENCODING_STAND_BLOCK + OCCUPANCY_TFREE_FEATURES,
        base="encoding-for-stand",
        diagnostic=True,
        note=(
            "the stand's last arrival at or before the subject's scheduled "
            "time, and that arrival's lateness, taxi-in, type and operator -- "
            "the same block with the take-off taken out of both the arithmetic "
            "and the selection, so it is admissible under an operational "
            "framing as well as a post-hoc one"
        ),
    ),
    Rung(
        key="callsign",
        label="+ callsign prefix",
        kind="booster",
        features=ENCODING_STAND_BLOCK + OCCUPANCY_FEATURES + CALLSIGN_FEATURES,
        base="occupancy",
        note=(
            "the operator, as the leading three letters of the callsign rather "
            "than as the flight-side field: the same information, and present "
            "on the one to two percent of departures with no Network Manager "
            "record, which are the worst rows in the data"
        ),
    ),
    Rung(
        key="destination",
        label="+ destination and holidays",
        kind="booster",
        features=ENCODING_STAND_BLOCK
        + OCCUPANCY_FEATURES
        + CALLSIGN_FEATURES
        + DESTINATION_FEATURES
        + HOLIDAY_FEATURES,
        base="callsign",
        note=(
            "the initial great-circle bearing and the distance to the "
            "destination, from OurAirports' coordinates, and whether the "
            "take-off date is a public holiday in the airport's country or the "
            "day beside one"
        ),
    ),
    Rung(
        key="plan-diffs",
        label="+ plan differences",
        kind="booster",
        features=ENCODING_STAND_BLOCK
        + OCCUPANCY_FEATURES
        + CALLSIGN_FEATURES
        + DESTINATION_FEATURES
        + HOLIDAY_FEATURES
        + PLAN_DIFF_FEATURES,
        base="destination",
        note=(
            "the filed off-block against the schedule and the last against the "
            "first, which the booster already sees as two lags and can only "
            "reach by splitting on each in turn"
        ),
    ),
    Rung(
        key="saturation",
        label="+ runway saturation",
        kind="booster",
        features=WEATHER_BLOCK + SATURATION_FEATURES,
        # Explicit, because the encoding arms now sit between this rung and the
        # one it was measured against. Without it this rung would silently be
        # re-read against an arm, and a measured negative would change meaning
        # without anybody editing the number.
        base="weather",
        note=(
            "the share of consecutive take-off gaps under ninety seconds, the "
            "smallest gap, and the span of the last three, from other departures' "
            "take-off times on the subject's runway (the cheap Gate 3)"
        ),
    ),
)

# The arms, by key: alternatives to one another rather than steps of the
# ladder, and excluded from the chain kept without the plan lags below.
ENCODING_ARMS: tuple[str, ...] = (
    "encoding-added",
    "encoding-for-stand",
    "encoding-for-identifiers",
)

# --------------------------------------------------------------------------
# which rows a model is fitted on

# Recorded taxi times above this many hours are out of every fit. ``None``
# keeps the whole tail.
#
# The record-quality term (``taxiout.tails``) removes rows by covered CELL --
# an airport crossed with whether a Network Manager record exists and how long
# the take-off lag is -- and coverage needs enough of the cell's rows to be
# defaulted. That is blind to a row whose cell is uncovered: the training year
# holds a departure recording a taxi of over twenty-four hours at an airport
# whose cells are all uncovered, with a take-off lag of fifteen minutes, so
# nothing about the row except the recorded taxi says it is a day-shift.
#
# This cannot be a wider flag. ``tails.flag`` and ``tails.predict`` are
# restricted by construction to fields the ranking set exposes, because they
# run there unchanged; recorded taxi time is the target and is hidden for
# departures. So it is a fit-side exclusion and nothing else -- the target is
# never clipped, and every row is still scored.
#
# Fixed at four hours on the mechanism, not by a sweep: no aircraft taxis for
# four hours. ``reports/baseline.md``'s *The recorded-taxi tail* reports it
# against an un-excluded fit and against a two-hour arm, both fixed before the
# run, and records that two hours is worse on July because the two-to-four hour
# band holds real long holds.
TAIL_EXCLUSION_ADOPTED_H: float | None = 4.0


def fit_rows(train, flagged, y):
    """The rows a model is fitted on: training, unflagged, not a day-shift.

    Every fit in this repository reads this -- the ladder, the k-sensitivity
    diagnostic, the seed average, the leave-one-airport-out transfer report and
    the shipped submission -- so that the model the reports measure and the
    model that is submitted are fitted on the same rows. They came apart once
    before, over the seed count, and the fix was to name the constant in one
    place and import it.
    """
    rows = train & ~flagged
    if TAIL_EXCLUSION_ADOPTED_H is None:
        return rows
    return rows & ~(pd.to_numeric(y, errors="coerce") > TAIL_EXCLUSION_ADOPTED_H * 3600.0)


# The Gate 2c chain, in the order it is climbed. Named here so the runner's
# acceptance section reads the same order the ladder is built in rather than
# rediscovering it from the ``base`` links.
GATE_2C_RUNGS: tuple[str, ...] = (
    "occupancy",
    "callsign",
    "destination",
    "plan-diffs",
)


def _without_plan_lags(rung: Rung, base: str) -> Rung:
    return Rung(
        key=f"{rung.key}-nolag",
        label=f"{rung.label} (without plan lags)",
        kind=rung.kind,
        features=tuple(c for c in rung.features if c not in PLAN_LAG_FEATURES),
        note=rung.note,
        options=rung.options,
        base=base,
    )


# The ladder proper, before the chain kept without the plan lags is appended.
LADDER_PROPER: tuple[Rung, ...] = LADDER

# The congestion and weather rungs kept without the plan lags, each read
# against the one before it in this chain and the first against the aircraft
# rung, so the report shows what each block is worth both gross and net.
#
# Every rung carrying the shrunk encoding is excluded, which is the arms and the
# four Gate 2c rungs that stand on one. The arms are alternatives rather than
# steps, so a chain through them would read one arm against another; the Gate 2c
# rungs sit *on* an arm, so a twin of one would have to be read against a rung
# that does not exist -- there is no arm without the plan lags, and there is no
# reason to build one, since the question this chain answers is what the
# congestion blocks are worth gross of a signal that arrived after them. Stated
# as a property of the features rather than as a list of keys, so a rung added
# above the arms is excluded by construction instead of by somebody remembering.
_NET_CHAIN = tuple(
    r
    for r in LADDER
    if PLAN_LAG_FEATURES[0] in r.features
    and r.key != "plan-lags"
    and not set(r.features) & set(ENCODING_FEATURES)
)
_WITHOUT: list[Rung] = []
_base = "identifiers-geometry-aircraft"
for _r in _NET_CHAIN:
    _WITHOUT.append(_without_plan_lags(_r, _base))
    _base = _WITHOUT[-1].key
LADDER = LADDER + tuple(_WITHOUT)

LADDER_BY_KEY: dict[str, Rung] = {r.key: r for r in LADDER}

# The rung the submission is fitted from: the arm that replaces the stand with
# the shrunk encoding, and then all four Gate 2c blocks -- stand occupancy, the
# callsign prefix, the destination geometry with the holiday calendar, and the
# plan's own differences. Chosen by two rules, each written down before the
# numbers that applied it: the encoding section's for the arm, and the Gate 2c
# section's for how far up the chain above it the submission goes.
#
# One thing about that second rule is not clean, and ``reports/baseline.md``
# devotes a subsection to it rather than a footnote. The rule says a rung must
# be "no worse" than the rung below it, and ``+ destination and holidays`` came
# out 0.0093 s worse on July while improving December by 1.16 s and the ranking
# mix by 1.13. Read as a comparison of two real numbers, that stops the chain
# and strands two rungs; read at the precision the report prints -- one decimal
# place, which is what every RMSE in every table here is given to -- the two
# rungs print the same number and it is a tie. The second reading is the one
# adopted, it was adopted after the number was seen, and the report says so in
# those words, prints the figures to four decimals so the reader can see how
# far below the last printed digit the difference sits, gives a day-clustered
# bootstrap interval on the step that straddles zero by two orders of
# magnitude, and computes every other comparison in the report to show that
# this reading moves that one step and no other. The precedent is in the same
# report: the encoding section already resolved two arms that sat "inside the
# tenth of a second this table prints".
#
# Two rungs of the ladder proper sit above this one.
#
# The saturation rung is the older case: it takes none of the ceiling, and with
# the booster made repeatable it moves the ex-flagged RMSE by -0.21 s in
# December and +0.29 s in July, at the edge of the seed floor on the first leg
# and inside it on the second (``reports/baseline.md``, *The noise floor*). A
# block worth that little belongs in the ladder and the write-up rather than
# in the model.
#
# ``+ shrunk encoding, replacing stand and runway`` is out too, though it sits
# below this rung on the ladder and so no longer appears in
# ``ABOVE_SUBMISSION``. It goes one step further than the shipped arm and
# replaces the designator as well, and it is out on a mechanism rather than on
# its score, which is indistinguishable: ``reports/geometry_gaps.md`` measures
# an intersection signature of up to 96 s per designator that the routed
# distance cannot compute, so the runway categorical carries something no
# encoding shrunk toward that distance can.
#
# Named here so that ``scripts/make_submission.py`` and the report's
# per-airport and importance sections all read the same rung. Moving it here
# is what puts the encoding into the submitted model: ``make_submission.py``
# computes the encoding once over the training year -- out of fold for the
# rows it fits on, by lookup for the rows it predicts -- as
# ``taxiout.encoding`` describes.
SUBMISSION_KEY = "plan-diffs"

# The rung the encoding arms were read against, and it is *not* whichever rung
# the submission happens to use. Before the arms shipped these were the same
# string, and writing the encoding section against ``SUBMISSION_KEY`` was
# harmless; the moment an arm displaced the control they came apart, and a
# section that took its control from the shipped rung would have compared the
# chosen arm against itself and reported a difference of zero. The control is a
# property of the experiment, fixed when it was designed, so it is named
# separately and does not move when the model does.
ENCODING_CONTROL = "weather"
TOP_RUNG: Rung = LADDER_BY_KEY[SUBMISSION_KEY]

# The rungs of the ladder proper above it: blocks that were built, measured,
# and left out of the model. The report measures each against the ceiling of
# the rung it was added to.
ABOVE_SUBMISSION: tuple[Rung, ...] = LADDER_PROPER[LADDER_PROPER.index(TOP_RUNG) + 1 :]

# The top rung with every identifier removed. Fitted across nine airports and
# applied to the tenth, this is what a model has left at an airport it has
# never seen: the geometry of that airport's own taxiways, the clock, the
# aircraft, the plan lags, the traffic around the movement and the weather --
# and no level of any categorical that names a place. The shrunk encodings are
# absent by the same rule: at an unseen airport every group has ``n = 0`` and
# the encoding is the geometric prior with an extra step in front of it.
TRANSFER_FEATURES: tuple[str, ...] = tuple(
    c
    for c in TOP_RUNG.features
    if c not in TRANSFERABLE_EXCLUDED and c not in ENCODING_FEATURES
)


def base_of(rung: Rung) -> Rung | None:
    """The rung ``rung`` is read against: its ``base``, else the one before it."""
    if rung.base is not None:
        return LADDER_BY_KEY[rung.base]
    i = LADDER.index(rung)
    return LADDER[i - 1] if i > 0 else None


class AirportMean:
    """Mean taxi-out per airport, with the pooled mean for an unseen airport."""

    def __init__(self) -> None:
        self.by_airport: pd.Series = pd.Series(dtype=float)
        self.overall: float = float("nan")

    def fit(self, icao: pd.Series, y: pd.Series) -> "AirportMean":
        t = pd.to_numeric(y, errors="coerce")
        self.by_airport = t.groupby(icao.astype(str), sort=True).mean()
        self.overall = float(t.mean())
        return self

    def predict(self, icao: pd.Series) -> pd.Series:
        return icao.astype(str).map(self.by_airport).fillna(self.overall).astype(float)


def booster_backend() -> str:
    """``"lightgbm"``, ``"catboost"`` or ``"none"``, whichever is installed."""
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass
    try:
        import catboost  # noqa: F401

        return "catboost"
    except ImportError:
        return "none"


class Booster:
    """Gradient-boosted trees over a named feature block.

    LightGBM by preference, CatBoost as the fallback, with the same fixed
    parameters translated across. Categoricals are passed as categoricals
    rather than expanded: for a tree model that is what a one-hot buys, at a
    fraction of the memory.

    A categorical named in ``POOLED_CATEGORICALS`` has its rare levels pooled to
    ``other`` at fit time, counted on the rows this model is fitted on, and any
    level absent from that set is mapped to ``other`` at prediction. The
    threshold belongs to the fit and not to the frame, which is the whole reason
    it is here: the fitting sample is what decides whether a level has enough
    rows to be worth a split, and the ranking set then cannot present a level
    the model never saw.
    """

    def __init__(self, features: tuple[str, ...], params: dict | None = None) -> None:
        self.features = tuple(features)
        self.params = dict(BOOSTER_PARAMS if params is None else params)
        self.backend = booster_backend()
        self.model = None
        self.categorical = tuple(c for c in self.features if c in CATEGORICAL_FEATURES)
        self.pooled = tuple(c for c in self.features if c in POOLED_CATEGORICALS)
        self._levels: dict[str, pd.Index | None] = {}
        self._kept: dict[str, frozenset[str]] = {}

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[list(self.features)].copy()
        for c in self.categorical:
            out[c] = out[c].astype("category")
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "Booster":
        """Fit on ``X``, ``y``, optionally weighting the rows.

        ``sample_weight`` is ``None`` for every fit this repository made before
        it existed, and with ``None`` no weight argument reaches the backend at
        all, so the unweighted fit is bit for bit the fit it was. With an array
        it is passed straight through as the backend's own ``sample_weight``.
        The array is not kept on the model: it belongs to one fitting sample,
        and a model carrying it would invite a later fit to reuse weights
        computed for other rows.
        """
        design = self._design(X)
        weight = None
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=float)
            if weight.shape != (len(design),):
                raise ValueError(
                    f"sample_weight has {weight.shape} for {len(design)} rows"
                )
        for c in self.pooled:
            values = design[c].astype(str)
            self._kept[c] = frequent_levels(values)
            # ``other`` is a level even where nothing was pooled into it, so that
            # a level the fit never saw has somewhere to go at prediction rather
            # than falling through to a null the model was never fitted on.
            design[c] = pd.Categorical(
                pool_rare(values, self._kept[c]),
                categories=sorted(set(self._kept[c]) | {OTHER_LEVEL}),
            )
        if self.backend == "lightgbm":
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(**self.params)
            extra = {} if weight is None else {"sample_weight": weight}
            self.model.fit(
                design,
                np.asarray(y, dtype=float),
                categorical_feature=list(self.categorical),
                **extra,
            )
        elif self.backend == "catboost":
            from catboost import CatBoostRegressor

            cb = CatBoostRegressor(
                iterations=int(self.params["n_estimators"]),
                learning_rate=float(self.params["learning_rate"]),
                depth=8,
                loss_function="RMSE",
                random_seed=int(self.params["random_state"]),
                verbose=False,
            )
            for c in self.categorical:
                design[c] = design[c].astype(str)
            extra = {} if weight is None else {"sample_weight": weight}
            cb.fit(
                design,
                np.asarray(y, dtype=float),
                cat_features=list(self.categorical),
                **extra,
            )
            self.model = cb
        else:
            raise RuntimeError(
                "neither lightgbm nor catboost is installed; "
                'pip install -e ".[modelling]"'
            )
        # The categories seen in training, so prediction cannot silently
        # re-code a level to a different integer on a differently-shaped frame.
        self._levels = {
            c: design[c].cat.categories if hasattr(design[c], "cat") else None
            for c in self.categorical
        }
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("predict before fit")
        design = X[list(self.features)].copy()
        for c in self.pooled:
            design[c] = pool_rare(design[c].astype(str), self._kept.get(c, frozenset()))
        for c in self.categorical:
            levels = self._levels.get(c)
            if self.backend == "catboost":
                design[c] = design[c].astype(str)
            elif levels is not None:
                design[c] = pd.Categorical(design[c].astype(str), categories=levels)
            else:
                design[c] = design[c].astype("category")
        return pd.Series(self.model.predict(design), index=X.index, dtype=float)

    def importances(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        raw = getattr(self.model, "feature_importances_", None)
        if raw is None:
            return pd.Series(dtype=float)
        return pd.Series(raw, index=list(self.features)).sort_values(ascending=False)


class SeedAverage:
    """``BAGGING_SEEDS`` boosters over the same rows, their predictions averaged.

    The estimator the submission is fitted with, and nothing the ladder uses:
    ``BAGGING_SEEDS`` says why. The seeds run consecutively from the one
    ``BOOSTER_PARAMS`` carries, so the first member is exactly the model every
    rung of the ladder is fitted with -- which is what lets the report check
    that its own fits reproduce, and what makes the average an addition to that
    model rather than a different one.

    The same interface as ``Booster``, so a caller swaps one for the other and
    nothing else changes.
    """

    def __init__(
        self,
        features: tuple[str, ...],
        params: dict | None = None,
        seeds: int = BAGGING_SEEDS,
    ) -> None:
        self.features = tuple(features)
        self.params = dict(BOOSTER_PARAMS if params is None else params)
        self.seeds = int(seeds)
        self.models: list[Booster] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "SeedAverage":
        """Every member fitted on the same rows, under the same weights.

        ``sample_weight`` is handed to each member unchanged, so the average is
        an average of the fit the caller asked for rather than of a different
        one; ``None`` reaches the backend as no weight at all, as it does in
        ``Booster.fit``.
        """
        base = int(self.params["random_state"])
        self.models = []
        for i in range(self.seeds):
            params = dict(self.params)
            params["random_state"] = base + i
            self.models.append(
                Booster(self.features, params).fit(X, y, sample_weight=sample_weight)
            )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.models:
            raise RuntimeError("predict before fit")
        total = None
        for model in self.models:
            p = model.predict(X)
            total = p if total is None else total + p
        return total / float(len(self.models))

    def importances(self) -> pd.Series:
        """The members' split shares, averaged.

        Averaged rather than summed so the scale is one model's and the number
        is comparable with a single fit's, which is what every table that reads
        it assumes.
        """
        got = [m.importances() for m in self.models]
        got = [s for s in got if len(s)]
        if not got:
            return pd.Series(dtype=float)
        return pd.concat(got, axis=1).mean(axis=1).sort_values(ascending=False)


__all__ = [
    "BAGGING_SEEDS",
    "BOOSTER_PARAMS",
    "ABOVE_SUBMISSION",
    "ENCODING_ARMS",
    "ENCODING_STAND_BLOCK",
    "GATE_2C_RUNGS",
    "SEED_AVERAGE_ADOPTED",
    "TAIL_EXCLUSION_ADOPTED_H",
    "fit_rows",
    "LADDER_PROPER",
    "ENCODING_CONTROL",
    "POOLED_CATEGORICALS",
    "SUBMISSION_KEY",
    "TOP_RUNG",
    "TRANSFERABLE_EXCLUDED",
    "TRANSFER_FEATURES",
    "WEATHER_BLOCK",
    "base_of",
    "IDENTIFIER_BLOCK",
    "LADDER",
    "LADDER_BY_KEY",
    "AirportMean",
    "Booster",
    "SeedAverage",
    "Rung",
    "booster_backend",
]
