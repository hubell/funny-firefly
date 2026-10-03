#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The pre-registered experiments, and nothing else.

Three questions whose criteria were fixed in writing on 20 September 2026,
before any of the figures below existed, and are carried here as constants:

1. **Seed averaging** on the two columns the rule of 19 September asks for
   before the five-seed average may ship -- the three airports July 2026
   contains, and October, the one forward leg no report has read.
2. **Recency weighting**, under two half-lives fixed in advance, screened on
   October and then on a *gap* leg built to break the confound that every
   other leg carries: train January to June and score December, so that the
   most recent training months are the seasonally furthest from the month
   being scored. That is the July 2026 geometry as nearly as one training
   year permits.
3. **The season columns**, removed and refitted, for the paper and not for
   the model: both forward legs validate on a month the training side never
   held and the ranking months do not, so **no adoption follows from the
   measurement in either direction**.

**Nothing is decided here and nothing ships from it.** No parameter, no
feature and no rule in ``taxiout.models`` is changed by this script, and it
does not write a submission. It fits twenty models, records what they scored,
and prints each criterion's verdict as the criterion computes it.

The reproduction checks run first and a failure stops the run. The December
and July first-seed fits must reproduce the ladder's stored residuals with a
largest absolute difference of zero, joined on the movement identifier, which
is never printed; the printed figures must be the ones ``baseline.md`` carries.
A fit that does not reproduce would make every other figure here a reading of
a model that cannot be repeated, so none of them is read until it does.

Like ``scripts/fast_lane.py``, this writes under one gitignored directory of
its own and to the single report it generates, and it fingerprints
``reports/`` and ``data/processed/residuals/`` before the run and verifies
them after it. An experiment that quietly overwrote a stored residual would
put an unreported fit where a reported one belongs.

Usage
-----
    python scripts/preregistered.py --dry-run     # the legs' row counts, no fit
    python scripts/preregistered.py               # twenty fits, then the report
    python scripts/preregistered.py --report      # the report, from the records
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_baselines as rb  # noqa: E402
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.encoding import encode_split, with_encoding  # noqa: E402
from taxiout.features import TARGET, airport_fallback, build_features  # noqa: E402
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER_BY_KEY,
    SUBMISSION_KEY,
    Booster,
    fit_rows as fitting_rows,
)
from taxiout.unimpeded import fit_pru_reference  # noqa: E402
from taxiout.validation import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    DECEMBER_TRAIN,
    DECEMBER_VALID,
    JULY_AIRPORTS,
    JULY_TRAIN,
    JULY_VALID,
    ClusterSums,
    cluster_bootstrap,
    cluster_sums,
    evaluate,
    evaluate_by_bucket,
    forward_split,
    frequency_bucket,
    month_of,
)

WORK = REPO / "data" / "processed" / "preregistered"
# The two directories this script may never write, checked rather than
# promised, as ``fast_lane.py`` checks them. Its own report is exempted by
# name in the one mode that writes it, so that exempting it cannot be done by
# leaving a whole directory out of the check.
FORBIDDEN = (REPO / "reports", REPO / "data" / "processed" / "residuals")


def report_path(generated: str) -> Path:
    return REPO / "reports" / f"preregistered-{generated}.md"


# --------------------------------------------------------------------------
# the pre-registration, carried verbatim
#
# Everything below this line down to ``PRINTED_PRECISION_S`` is quoted from
# the pre-registration of 20 September, from its *Definitions, fixed here*
# heading onward, and is not edited by this script or by its author after the
# fits. What was changed when it was copied in is recorded place by place in
# ``SUBSTITUTIONS`` below, and ``substitution_audit`` checks that record
# against this text and, where the note is on hand, against the note. No
# criterion, schedule, grid value or reading rule differs.

PREREGISTRATION = """## Definitions, fixed here

**The legs.** Every leg is fitted exactly as the ladder fits the submission
rung `+ plan differences` (`SUBMISSION_KEY = "plan-diffs"`): the
record-quality model on the leg's training months, `taxiout.models.fit_rows`
(train, unflagged, recorded taxi-out at most 4 h), the shrunk encoding fitted
on those rows, `BOOSTER_PARAMS` as `taxiout.models` carries them, the first
seed the ladder's (`random_state` 20260902). Scored ex-flagged, as
`taxiout.validation.evaluate` scores.

| leg | train (2025) | validate (2025) | what is read | floor (s) |
| --- | --- | --- | --- | --- |
| December | January to November | December, all ten | pooled ex-flagged; the ranking mix (December's bucket figures at the ranking set's bucket weights, `RANKING_MIX`) | 0.34 (`baseline.md` line 1061) |
| July | January to June | July | pooled ex-flagged; the three July airports (EDDF, EGLL, EHAM) | 0.23 pooled, 1.02 the three (lines 1062 to 1063) |
| October | January to September | October, all ten | pooled ex-flagged | measured in this run |
| the gap | January to June | December, all ten | pooled ex-flagged | measured in this run |

The gap leg is not a separate fit: its training months, fit rows, encoding,
record-quality model, seeds and weight ages are the July leg's, so it is
the July leg's fitted models scored on December's rows (`Split.extra`, the
mechanism the held-out-pair folds already use for a second evaluation set
from one fit), with December's flags from the July leg's record-quality
model. Its floor comes from the same five July seeds scored there.

A floor measured in this run is root two times the sample standard
deviation (`ddof = 1`, as `_noise_floors` computes it) of five single-seed
ex-flagged RMSEs at `random_state` + 0 to + 4, the rule `baseline.md`'s
*The noise floor* states. It is fixed by that rule, not chosen.

**The readings.** "No worse" means not worse at the precision the reports
print, one decimal of a second (`PRINTED_PRECISION_S = 0.1`), the standing
rule of 8 September. "Clears the floor" means the gain on the unrounded
figure exceeds that leg's floor. Any step inside a second prints the
day-clustered percentile interval `taxiout.validation.cluster_bootstrap`
gives, with days as the clusters and the same draw for both arms, so the
contrast stays paired.

**The references.** December all ten 217.6 s, July the three 237.8 s, the
ranking mix 218.6 s, July all ten 325.8 s (`baseline.md` lines 234, 894,
481, 270). The October and gap references are the unweighted first-seed
fits made in the same run.

**Reproduction, before anything is read.** The December and July first-seed
fits must reproduce the stored residuals `december_plan-diffs.parquet` and
`july_plan-diffs.parquet` (which hold `y - pred`, joined on the movement
identifier, which is never printed) with a largest absolute difference of 0
on residuals formed the same way; the July
five singles must span 325.81 to 326.23 s pooled and 237.02 to 238.83 s on
the three, and their average must print 325.29 s pooled (`baseline.md` lines
1023, 1062, 1063); the December first seed must print 218.6 s at the ranking
mix. A disagreement is a stop, before any experiment's figure is read.

## Experiment 1: seed averaging on the three and on October

**What is run.** The five July seeds, scored pooled and on the three, singly
and averaged. Five October seeds, scored pooled, singly and averaged, and
October's floor from them.

**The rule it serves,** as fixed on 19 September and printed in
`baseline.md` line 1010: "Adopt it for the submission if, on both forward
legs, the five-seed average beats every one of the five single seeds and its
gain over the ladder's seed clears that leg's seed floor; before it ships it
is also measured on the three July airports and on October, and both are
printed."

**One guard, added here on 20 September, before either measurement existed:**
the average is not adopted if, on the three or on October, it is worse than
the ladder's seed by more than that leg's floor. The three is the column the
July ranking month asks about and October is the one leg no report has read;
a regression clear of noise on either is a reason not to ship, and the rule
as fixed would not have read it.

**What it writes.** The ten singles, the two averages, October's floor, and
the verdict of the rule and the guard at the present parameters, in
`reports/preregistered-<date>.md`.

**What is done with each outcome.** Nothing ships from this run. The rule is
read for the submission from `baseline.md` after the ladder run that follows
the wider sweep prints the two measurements under the final parameters (the
generator's `SEED_AVERAGE_PENDING` is emptied then and not before). If that
sweep does not change the parameters, its figures must equal this run's to the
bit. If the rule and the guard hold there, `SEED_AVERAGE_ADOPTED` is set true
and the submission is the five-seed average; otherwise it stays a single seed
and the paper says what the average did on each leg.

## Experiment 2: recency weighting

**The confound, stated first.** On every 2025 leg the most recent training
month is also the seasonally nearest to the validation month (November for
December, which January, the oldest month, ties across the year boundary;
June for July; September for October), so a gain on those legs cannot
separate recency from season. For the January 2026 ranking rows the
two agree (December 2025 is the most recent training month and seasonally
adjacent). For the July 2026 rows they do not: the most recent training
month, December 2025, is seasonally opposite, and the seasonal twin, July
2025, is five months older. **The gap leg** (train January to June, validate
December) is the one forward leg the data allows in which the recent months
are seasonally far from the validation month and the seasonally nearest rows
(January, February) are the oldest: the July 2026 geometry as nearly as one
training year permits, without the twin month. It is built to break the
confound, and a schedule that fails it fails the July 2026 rows.

**The schedules, two, fixed here.** Weight per training row
`w = 0.5 ** (age / H)`, where `age` is the whole number of calendar months
from the row's take-off month (UTC, `taxiout.validation.month_of`) to the
leg's last training month (0 for that month), and `H` is the half-life in
months: **H = 3** (the strong schedule, chosen to make the confound's sign
visible: it discards most of the year) and **H = 12** (the mild schedule,
the plausible candidate: the oldest month of a full year keeps about half
its weight). Two, and no third: a third schedule chosen after these two are
read would be chosen on the legs. The weights enter the booster's fit only
(`sample_weight`); the record-quality model, `fit_rows`, the encoding and
the scoring are unweighted and unchanged.

**The digest check, before any figure is read.** `deterministic` was
measured without sample weights. The mild schedule's October fit is made
twice; the two prediction vectors must be identical bit for bit and the two
models' dumps identical. A disagreement is a stop for the experiment.

**The sequence, fixed.**

1. October screen: each schedule against the unweighted October first-seed
   fit. A schedule worse at printed precision stops here; its figure is
   recorded and it is not fitted on any other leg.
2. The July fit, read on the gap first: each surviving schedule is fitted
   on the July leg's training months and its December (gap) reading is
   read against the unweighted gap first seed. A schedule worse at printed
   precision stops here: its July readings are printed, since they come
   from the same fit, but it is not fitted on December and cannot be
   adopted.
3. The forward legs: each surviving schedule on December, and its July
   readings, read against 217.6 (December all ten), 237.8 (July the three)
   and 218.6 (the ranking mix) at printed precision, the bootstrap printed
   on any step inside a second, and July all ten printed beside.

**Adoption.** A schedule is adopted for the submission only if it is no
worse at printed precision on all five readings (October, the gap, December
all ten, July the three, the ranking mix) and its gain clears the floor on
at least one of December all ten and July the three. A change to how the
model is fitted that moves nothing outside seed noise is not adopted. If
both schedules qualify, the mild one is adopted unless the strong one beats
it on all five readings by more than the respective floors.

**What it writes.** Every fit's figures and the sequence's stops, in
`reports/preregistered-<date>.md`.

**What is done with each outcome.** Adopted: `scripts/make_submission.py`
applies the schedule with December 2025 as the last training month, the
manifest records the half-life, and the wider sweep and the ladder run that
follows it run under it as part of the current fit rules. Not adopted:
recorded, and the paper's §7.1 carries the confound and the gap-leg reading
as the reason no recency term ships. A two-model submission (one fit per
ranking month with its own weighting) is not opened, whatever the readings:
no leg in the data validates the July 2026 geometry with its twin month
present, the one-upload rule leaves it unmeasurable, and it doubles the
reproduction the paper defends.

## Experiment 3: the season columns

**What is run.** The submission rung refitted on December and on July
without `tko_month`, `tko_doy_sin` and `tko_doy_cos` (`taxiout.features`),
read against 217.6, 237.8 and 218.6 at printed precision with the bootstrap
on any step inside a second, and July all ten printed beside.

**What it is for, fixed now: the paper, not the model.** Both forward legs
validate on a month and day-of-year range the training side never held
(December after January to November; July after January to June), and the
ranking months do not: January 2026 and July 2026 have their twins in the
training year. The measurement answers a question the ranking months do not
ask, so **no adoption follows from it in either direction**. Its sign and
size go to the paper's §7.1 as the validation design's limitation.

**What it writes.** The two fits' figures in `reports/preregistered-<date>.md`.

## Experiment 4: operator cells at LIRF. Not run.

Closed by `tails.md`'s section of 20 September: the note's test is the
term's first condition, the two-hour condition is the test, and a prefix
split under the coverage rule opens one cell (ADB, `flt:normal`, 13 training
rows, 7.7% defaulted) touching 1 of LIRF's 11,061 ranking rows. Off the
wider sweep's list.

## Experiment 5: the wider sweep, permitted once on 19 September

**The grid, fixed here.** `learning_rate` (0.015, 0.03, 0.06);
`num_leaves` (63, 127, 255); `min_child_samples` (20, 50, 200); trees read
at (300, 600, 1000, 1500, 2000, 3000) from one 3,000-tree fit per cell,
as the first sweep read its trees from one fit. Twenty-seven cells. The
chosen cell (0.03, 127, 50, 1,000) sat on the edge of the first grid on all
four axes (`scripts/sweep_params.py`: 0.03/0.06/0.1, 31/63/127, 50/200/500,
300/600/1000); here it is interior on three axes and the trees axis extends
past it. Each new value is one step past the old edge at the old grid's own
spacing, and the far values of the old grid (0.1, 31, 500) are dropped.

**The fit rules.** The submission rung's features as they stand after the
adoptions that precede the sweep (the season columns are not changed by
Experiment 3); the 4 h exclusion; the shrunk encoding on `fit_rows`; the
record-quality model on the training months; the recency schedule if
Experiment 2 adopted one. Train January to September 2025, score October
2025 over the rows the record-quality term leaves to the physical model, as
the first sweep scored.

**The floor, measured in the same run.** Five single-seed fits of the
current cell, `random_state` + 0 to + 4; root two times their standard
deviation. If this run's October singles were made at the same parameters
and rules, the two sets must agree to the bit.

**The reading.** The best cell replaces `BOOSTER_PARAMS` only if its October
figure is below the current cell's figure from the same run by more than the
floor; a difference inside the floor keeps the current cell. If the best
cell lies on an edge of this grid, that is recorded and the grid is not
extended again. Nothing is chosen on December, July or the held-out pairs.
One ladder run follows, under the standing acceptance rules; the rung chain
is re-read under the new parameters, and the seed section prints the three
and October.

**What it writes.** `data/processed/sweep_october.csv` (gitignored, as
before); the chosen cell and the October figures beside `BOOSTER_PARAMS`
in `taxiout.models`, with the date; `baseline.md` regenerated.

## Experiment 6: Heathrow. Nothing.

`residual.md`'s two descriptive sections: the afternoon and the no-record
class are structured at intervals clear of zero, in columns the rung already
reads, with no mechanism the frame lacks and the oracle's ceiling at 9.9 s
for the same-runway sequence at EGLL on July (`congestion.md`). Paper
paragraphs in §9.5 and §7.3; no fit.

## Also in this run, not experiments

- A dated entry in `docs/plan.md`: the decisions of 19 September (the seed
  rule re-fixed at the floors, quoting `baseline.md` line 1010; the sweep
  reopened once with the grid above; `AOBT_3_flt` excluded and unmeasured;
  the July column as the 9 September entry records) and this
  pre-registration's criteria verbatim. Written before the fits and the
  first commit of the block, so the record precedes the numbers in the
  history as it does on disk.
- `Booster.fit` and `SeedAverage.fit` in `taxiout.models` take
  `sample_weight=None`; with `None`, no weight reaches the backend and the
  stored residuals reproduce bit for bit.
- `reports/residual.md` lines 211 to 213: "Two row sets that table leaves
  implicit" becomes the one row set that table does not print,
  `before the estimate`; `no estimate` is already a row of *The error by
  plan lag*. Generator wording only, regenerated.

## Not run, and why

- **The `AOBT_3_flt` diagnostic.** Excluded by decision (`taxiout.challenge`),
  unmeasured; the paper says it is excluded and why and does not claim to
  know what it would have been worth (Decision 3, 19 September).
- **The January boundary.** Dropped: 839 of
  152,719 January 2026 rows carry `occ_truncated` against 853 of 153,706 in
  January 2025, the training year's own first day; the flag is a column the
  booster reads; no forward leg validates a month that begins a block, so
  the change could not be measured; it sits on the scored-file path.
- **Operator cells at LIRF.** Closed, above.
- **A held-out-month split** for the season columns (train around a month,
  validate inside it): it would answer the interpolation question the
  ranking months ask, and it breaks the forward discipline every
  validation leg here keeps. Named so the paper can say what would have
  measured it.
- **A per-day table of Heathrow's afternoons.** A search for the days, not
  a description; the day-clustered interval already says the effect rests
  on few of them.
- **The pooled record-quality rule** for airports with no covered cell
  (EHAM's six January 2026 days). The paper's §4 and §7.3 record the
  decision; no instrument in the data measures the alternative; not
  reopened on stakes alone. Flagged for Hunter's own review.
- **SUMO and the OpenSky scientific datasets.** Diagnostics for the paper
  at most (the README's section on the datasets considered); nothing in the
  model.

## The fits, counted

October 5 (the floor and the reference) + 2 schedules + 1 repeat for the
digest check; July 5 (Experiment 1, the first of them the reproduction;
scored on July and on December, which is the gap) + 2 schedules; December
1 (reproduction and reference) + 2; the season columns 2. Twenty fits of
the submission rung if no schedule stops early, on one feature frame,
sequential, at the ladder's eight threads. Every wall and peak RSS is
recorded; none is estimated here.

## Amendment, 20 September, before the run started

Written the same day, after an independent read and before any fit: the
gap leg restated as the July leg's models scored on December (it was
first written as separate fits, which would have refitted the same models),
the sequence's second step reworded to match, the fit count from
twenty-seven to twenty, the floor's standard deviation named as the sample
one, the December confound sentence given January's tie. No criterion,
schedule, grid value or reading rule changed.
"""


# --------------------------------------------------------------------------
# the criteria as constants, each copied from the text above
#
# Every one of these is a number or a rule the pre-registration fixed. None is
# computed from this run, and none is an estimate. The source is given beside
# each so that a reader can check the constant against the file it came from.

# The precision every report prints and every "no worse" is read at, imported
# rather than restated so this file cannot drift from the standing rule.
PRINTED_PRECISION_S = rb.PRINTED_PRECISION_S

# ``baseline.md`` lines 234, 894, 481, 270. The unweighted first-seed fits of
# this run must reproduce them, which is what lets a later arm be read against
# the unrounded figure rather than against the printed one.
REF_DECEMBER_ALL = 217.6
REF_JULY_THREE = 237.8
REF_RANKING_MIX = 218.6
REF_JULY_ALL = 325.8

# ``baseline.md`` lines 1061 to 1063. October's and the gap's are measured in
# this run, by the same rule, and are not constants.
FLOOR_DECEMBER_ALL = 0.34
FLOOR_JULY_ALL = 0.23
FLOOR_JULY_THREE = 1.02

# ``baseline.md`` lines 1062, 1063 and 1023: the five July singles' range on
# each column and the average the five of them make.
JULY_SINGLES_POOLED = (325.81, 326.23)
JULY_SINGLES_THREE = (237.02, 238.83)
JULY_AVERAGE_POOLED = 325.29

# The two half-lives, in months. Two, and no third.
HALF_LIVES: tuple[int, ...] = (3, 12)

# The columns Experiment 3 removes.
SEASON_COLUMNS: tuple[str, ...] = ("tko_month", "tko_doy_sin", "tko_doy_cos")

# The seed count and the floor's factor, from the modules that own them.
SEEDS = 5
NOISE_FLOOR_FACTOR = rb.NOISE_FLOOR_FACTOR

# A step smaller than this in absolute value carries the day-clustered
# interval. "Any step inside a second", the pre-registration's words.
INTERVAL_BELOW_S = 1.0

ATTRIBUTION = "Distance matrices derived from OpenStreetMap"

# The commit that carries the dated entry in ``docs/plan.md``. It is the first
# of the three this run's work was committed as, and the entry is older on disk
# than the earliest record the run wrote, which is the claim the commit
# script's own gate checks. Named here so the report can say which commit
# rather than say "committed ahead of this run", which was not true of anything
# at the moment the generator first wrote it.
PLAN_ENTRY_COMMIT = "f7cb53c"

# The rule of 19 September reads two forward legs this run does not fit, so its
# figures are quoted from ``baseline.md`` with their lines, on the same footing
# as the references and the floors above. Nothing here recomputes them. July's
# are reproduced by this run's own five seeds, which is what the reproduction
# checks already establish, and December's are not measured here at all.
RULE_LEGS: tuple[dict, ...] = (
    {
        "leg": "December, all ten",
        "ladder_seed": 217.6,
        "average": 217.11,
        "best_single": 217.59,
        "gain": 0.48,
        "floor": FLOOR_DECEMBER_ALL,
        "source": "baseline.md lines 1020 and 1023",
        "reproduced_here": False,
    },
    {
        "leg": "July, all ten",
        "ladder_seed": 325.8,
        "average": 325.29,
        "best_single": 325.81,
        "gain": 0.52,
        "floor": FLOOR_JULY_ALL,
        "source": "baseline.md lines 1021 and 1023",
        "reproduced_here": True,
    },
)


# --------------------------------------------------------------------------
# what was changed when the pre-registration was copied in
#
# The note this file quotes is a working note and is not published. Two things
# in it cannot be carried into a public repository: the numbered work blocks of
# the working program, which name sessions, and the file names of other working
# notes. Every place where the carried copy differs from the note is listed
# below, and the list is the only record of it: the counts the report prints
# are summed from these entries, so a sentence and a table cannot drift apart.
#
# Each entry holds the PUBLISHED side only. The note's own wording is not here,
# because writing a work block's name into this repository in order to record
# that one was taken out would defeat the purpose of taking it out.

WORK_BLOCK = "a numbered work block"
NOTE_REFERENCE = "a working note"


@dataclass(frozen=True)
class Place:
    """One place where the carried copy differs from the note it quotes.

    ``marker`` is a line of the published text that occurs exactly once in
    ``PREREGISTRATION``, which is what ties this entry to a place in the text
    without quoting the note. ``tokens`` is how many names were replaced
    there, which is not always one: the table cell holds two and Experiment
    1's outcome paragraph holds five.
    """

    where: str
    marker: str
    kind: str
    tokens: int
    widens: str | None = None


SUBSTITUTIONS: tuple[Place, ...] = (
    Place(
        "The legs table, the October and the gap floor cells",
        "| October | January to September | October, all ten | pooled ex-flagged | measured in this run |",
        WORK_BLOCK,
        2,
    ),
    Place(
        "Definitions, the floor rule",
        "A floor measured in this run is root two times",
        WORK_BLOCK,
        1,
    ),
    Place(
        "Experiment 1, what is done with each outcome",
        "Nothing ships from this run. The rule is",
        WORK_BLOCK,
        5,
    ),
    Place(
        "Experiment 2, what is done with each outcome",
        "and the wider sweep and the ladder run that",
        WORK_BLOCK,
        1,
    ),
    Place(
        "Experiment 4, the closing sentence",
        "ranking rows. Off the",
        WORK_BLOCK,
        1,
    ),
    Place(
        "Experiment 5, the heading",
        "## Experiment 5: the wider sweep, permitted once on 19 September",
        WORK_BLOCK,
        1,
    ),
    Place(
        "Experiment 5, the fit rules",
        "The submission rung's features as they stand after the",
        WORK_BLOCK,
        1,
        widens=(
            "the only one of the thirteen that is not purely a rendering of a"
            " name. The note scopes the clause to the adoptions of one named"
            " block; the published copy scopes it to the adoptions that precede"
            " the sweep, which is a wider class. Nothing is adopted, so nothing"
            " turns on it today, and the record is the note"
        ),
    ),
    Place(
        "Experiment 5, the floor",
        "deviation. If this run's October singles were made",
        WORK_BLOCK,
        1,
    ),
    Place(
        "Experiment 6, the opening citation",
        "`residual.md`'s two descriptive sections: the afternoon and the no-record",
        NOTE_REFERENCE,
        1,
        widens=(
            "points at the repository's own file instead, and the section"
            " number that pointed into the note went with it. It is the only"
            " number anywhere in the quoted text that the copy does not carry,"
            " and it is a pointer into a file this repository does not have"
            " rather than a criterion"
        ),
    ),
    Place(
        "Also in this run, the heading",
        "## Also in this run, not experiments",
        WORK_BLOCK,
        1,
    ),
    Place(
        "Not run, the January boundary",
        "- **The January boundary.** Dropped: 839 of",
        NOTE_REFERENCE,
        1,
    ),
    Place(
        "Not run, the external datasets",
        "at most (the README's section on the datasets considered); nothing in the",
        NOTE_REFERENCE,
        1,
    ),
    Place(
        "The amendment, the heading",
        "## Amendment, 20 September, before the run started",
        WORK_BLOCK,
        1,
    ),
)

# Where the note lives when it is on hand: ``--note PATH``, or the directory the
# environment variable ``PREREGISTRATION_NOTES`` names, and no default. A clean
# checkout has neither, and the report says the byte-for-byte check was not run
# rather than saying it passed.
NOTES_ENV = "PREREGISTRATION_NOTES"
NOTE_NAME = "preregistration_2026-09-20.md"
NOTE: Path | None = (
    Path(os.environ[NOTES_ENV]) / NOTE_NAME if os.environ.get(NOTES_ENV) else None
)
NOTE_HEADING = "## Definitions, fixed here"

# Patterns, not names. Each matches any numbered work block or any dated
# working note, and neither names one, which is the same footing the committed
# test that asserts their absence from the carried text already stands on.
WORK_BLOCK_TOKEN = re.compile(r"\bBlock\s+\d+\b")
NOTE_REFERENCE_TOKEN = re.compile(r"\b[A-Za-z0-9_]+_\d{4}-\d{2}-\d{2}\.md\b")


def substitution_audit() -> dict:
    """What was changed when the pre-registration was copied in, computed.

    Two levels, and the report says which of them ran.

    The first needs only the published text, so it runs in a clean checkout:
    every marker occurs exactly once in ``PREREGISTRATION``, the counts are
    summed from the manifest rather than written down anywhere, and the
    carried text carries no work-block name and no note file name.

    The second needs the note on hand. It aligns the carried text against the
    note's quoted region line by line and requires that the changed places are
    exactly the manifest's, in order, with exactly the manifest's token counts;
    then it rebuilds the carried text from the note by taking the unchanged
    lines from the note and the changed ones from here, and requires the
    rebuild to equal the carried text byte for byte. Outside these places the
    two texts are the same bytes, which is the whole of the claim.
    """
    out: dict = {
        "places": len(SUBSTITUTIONS),
        "work_block_tokens": sum(p.tokens for p in SUBSTITUTIONS if p.kind == WORK_BLOCK),
        "note_references": sum(
            p.tokens for p in SUBSTITUTIONS if p.kind == NOTE_REFERENCE
        ),
        "work_block_places": sum(1 for p in SUBSTITUTIONS if p.kind == WORK_BLOCK),
        "note_reference_places": sum(
            1 for p in SUBSTITUTIONS if p.kind == NOTE_REFERENCE
        ),
    }
    out["tokens"] = out["work_block_tokens"] + out["note_references"]
    stray = [p.where for p in SUBSTITUTIONS if PREREGISTRATION.count(p.marker) != 1]
    out["markers_found_once"] = not stray
    out["stray_markers"] = stray
    out["carried_is_clean"] = not (
        WORK_BLOCK_TOKEN.search(PREREGISTRATION)
        or NOTE_REFERENCE_TOKEN.search(PREREGISTRATION)
    )

    if NOTE is None or not NOTE.exists():
        out["note_checked"] = False
        out["note_verdict"] = (
            "not run: the note is not on this machine, so only the checks that"
            " read the published text alone were made"
        )
        return out

    raw = NOTE.read_bytes()
    out["note_checked"] = True
    out["note_sha256"] = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    if NOTE_HEADING not in text:
        out["note_verdict"] = f"the note carries no {NOTE_HEADING!r} heading"
        return out
    region = text[text.index(NOTE_HEADING) :]
    a, b = region.split("\n"), PREREGISTRATION.split("\n")
    opcodes = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    changed = [op for op in opcodes if op[0] != "equal"]
    problems = []
    if len(changed) != len(SUBSTITUTIONS):
        problems.append(
            f"{len(changed)} changed places against {len(SUBSTITUTIONS)} recorded"
        )
    else:
        for place, (_, i1, i2, j1, j2) in zip(SUBSTITUTIONS, changed):
            was, now = "\n".join(a[i1:i2]), "\n".join(b[j1:j2])
            if place.marker not in now:
                problems.append(
                    f"{place.where}: the published side is not the recorded one"
                )
            pattern = (
                WORK_BLOCK_TOKEN if place.kind == WORK_BLOCK else NOTE_REFERENCE_TOKEN
            )
            found = len(pattern.findall(was))
            if found != place.tokens:
                problems.append(
                    f"{place.where}: {found} token(s) replaced against {place.tokens} recorded"
                )
    rebuilt = []
    for tag, i1, i2, j1, j2 in opcodes:
        rebuilt.extend(a[i1:i2] if tag == "equal" else b[j1:j2])
    if "\n".join(rebuilt) != PREREGISTRATION:
        problems.append("the note does not rebuild the carried text")
    out["note_problems"] = problems
    out["note_verdict"] = (
        "the note rebuilds the carried text byte for byte with exactly these"
        " places changed"
        if not problems
        else "; ".join(problems)
    )
    return out


# --------------------------------------------------------------------------
# what this script may not write


def fingerprint(exempt: Path | None = None) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file this script must not write."""
    out: dict[str, tuple[int, int]] = {}
    for directory in FORBIDDEN:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != exempt:
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


def touched(before: dict, after: dict) -> list[str]:
    return sorted(
        {k for k in before if after.get(k) != before[k]} | (set(after) - set(before))
    )


# --------------------------------------------------------------------------
# the report refuses an identifier


def check_no_identifiers(text: str, stands: set[str], movements: set[str]) -> None:
    """Refuse to write a report carrying a stand or a movement identifier.

    Two rules, both with a zero threshold. The stand rule is the one
    ``check_licence_hygiene.py`` applies -- an identifier with a letter, a
    digit and at least three characters -- with runway designators and ICAO
    codes exempt, because those are public. The movement rule is this
    report's own: the reproduction check joins on the movement identifier and
    must never print one, so every identifier it touched is looked for as a
    whole word.
    """
    exempt = {a.icao for a in AIRPORTS}
    leaked = sorted(
        v
        for v in stands - exempt
        if len(v) >= 3
        and re.search(r"[A-Za-z]", v)
        and re.search(r"[0-9]", v)
        and v in text
        and re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", text)
    )
    if leaked:
        raise SystemExit(
            f"refusing to write the report: {len(leaked)} literal stand "
            "identifier(s) would be published"
        )
    # The movement identifiers are hundreds of thousands of whole numbers, so
    # the text is tokenised once and intersected rather than searched once per
    # identifier. Every movement identifier is a run of digits, so a token set
    # is exactly the same test as the word-boundary search above, and is the
    # difference between a check that runs and one that does not.
    ids = sorted(set(re.findall(r"(?<![\w])\d+(?![\w])", text)) & movements)
    if ids:
        raise SystemExit(
            f"refusing to write the report: {len(ids)} movement identifier(s) "
            "would be published"
        )


# --------------------------------------------------------------------------
# the legs


OCTOBER_TRAIN = tuple(f"2025-{m:02d}" for m in range(1, 10))
OCTOBER_VALID = ("2025-10",)

# The last training month of each leg, which is where a row's age is measured
# from. Written out rather than derived from the split, so that changing one
# is a visible edit to a constant.
LAST_TRAIN_MONTH = {"december": "2025-11", "july": "2025-06", "october": "2025-09"}


def build_legs(month: pd.Series) -> dict:
    """The three fitted legs. The gap is the July leg's second evaluation set."""
    december = forward_split(
        month,
        DECEMBER_TRAIN,
        DECEMBER_VALID,
        name="december",
        description="train January to November 2025, validate December, all ten",
    )
    july = forward_split(
        month,
        JULY_TRAIN,
        JULY_VALID,
        name="july",
        description="train January to June 2025, validate July",
    )
    # The gap: the same fit, scored on December. ``Split.extra`` is the
    # mechanism the held-out-pair folds already use for a second evaluation
    # set from one fitted model, and its ``__post_init__`` checks that no
    # December row is also a July training row.
    july.extra = {"gap": month.isin(DECEMBER_VALID)}
    july.__post_init__()
    october = forward_split(
        month,
        OCTOBER_TRAIN,
        OCTOBER_VALID,
        name="october",
        description="train January to September 2025, validate October, all ten",
    )
    return {"december": december, "july": july, "october": october}


def month_index(month: pd.Series) -> pd.Series:
    """``YYYY-MM`` as a count of months, so a difference is calendar months."""
    text = month.astype(str)
    return text.str.slice(0, 4).astype(int) * 12 + text.str.slice(5, 7).astype(int)


def recency_weights(month: pd.Series, rows: pd.Series, last: str, half_life: int):
    """``0.5 ** (age / H)`` over the fitting rows, age in whole months.

    ``age`` is 0 for the leg's last training month and counts up going back,
    so the weight is 1 there and halves every ``half_life`` months.
    """
    last_index = int(last[:4]) * 12 + int(last[5:7])
    age = last_index - month_index(month[rows])
    if int(age.min()) < 0:
        raise RuntimeError("a fitting row is later than the leg's last training month")
    return np.asarray(0.5 ** (age.to_numpy(dtype=float) / float(half_life)))


# --------------------------------------------------------------------------
# one fit


class Frame:
    """The feature frame, the target and the per-leg preparation, built once."""

    def __init__(self) -> None:
        t0 = time.time()
        self.tails = rb.Tails()
        matrix = rb.load_matrix()
        self.dep = rb.load_departures(None)
        self.y = pd.to_numeric(self.dep[TARGET], errors="coerce")
        movements = rb.load_all_movements()
        self.features = build_features(
            self.dep,
            matrix,
            fallback=airport_fallback(matrix),
            movements=movements,
            edges=rb.load_edge_sets(),
            weather=rb.load_weather(),
            airports=rb.load_airport_table(),
        )
        del movements
        self.build_seconds = time.time() - t0
        self.build_peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.month = month_of(self.dep["MVT_TIME_UTC_mvt"])
        self.legs = build_legs(self.month)
        self.prepared: dict[str, dict] = {}
        print(
            f"frame: {len(self.features):,} rows in {self.build_seconds:.0f} s, "
            f"peak {self.build_peak_rss_kib / 1048576:.1f} GiB"
        )

    def prepare(self, name: str) -> dict:
        """One leg's record-quality model, fit rows and encoded frame.

        Exactly what ``run_baselines.run`` does inside its split loop: the
        record-quality model on this leg's training rows, ``fit_rows`` over
        train, unflagged and the four-hour exclusion, and the shrunk encoding
        recomputed from those rows and overlaid.
        """
        if name in self.prepared:
            return self.prepared[name]
        leg = self.legs[name]
        t0 = time.time()
        tail_model = self.tails.fit(self.dep.loc[leg.train])
        flagged = (
            tail_model.predict(self.dep).notna()
            if tail_model is not None
            else pd.Series(False, index=self.dep.index)
        )
        rows = fitting_rows(leg.train, flagged, self.y)
        encoding = encode_split(self.features, self.y, rows)
        got = {
            "leg": leg,
            "flagged": flagged,
            "rows": rows,
            "frame": with_encoding(self.features, encoding),
            # The pair-frequency reference, fitted on the same rows the
            # booster is, which is what ``run_baselines`` fits it on. Only
            # December's is used, for the ranking mix.
            "reference": fit_pru_reference(self.features, self.y, mask=rows),
            "seconds": time.time() - t0,
        }
        print(
            f"  {name:>8s}: {int(leg.train.sum()):,} train, "
            f"{int(leg.valid.sum()):,} validate, {int(rows.sum()):,} fit rows "
            f"({got['seconds']:.0f} s)"
        )
        self.prepared[name] = got
        return got

    def sets(self, name: str) -> dict:
        leg = self.legs[name]
        return {"valid": leg.valid, **leg.extra}


def digest(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def model_digest(model: Booster) -> str | None:
    """The fitted model's own dump, for the digest check. ``None`` off LightGBM."""
    inner = getattr(model.model, "booster_", None)
    if inner is None:
        return None
    return hashlib.sha256(inner.model_to_string().encode("utf-8")).hexdigest()


def fit_once(frame: Frame, leg: str, *, seed: int, schedule: str, features: str) -> dict:
    """One fit of the submission rung, scored on every set the leg carries."""
    prep = frame.prepare(leg)
    rung = LADDER_BY_KEY[SUBMISSION_KEY]
    columns = tuple(rung.features)
    if features == "no-season":
        columns = tuple(c for c in columns if c not in SEASON_COLUMNS)
    params = dict(BOOSTER_PARAMS)
    params["random_state"] = int(BOOSTER_PARAMS["random_state"]) + seed
    weight = None
    if schedule != "none":
        weight = recency_weights(
            frame.month, prep["rows"], LAST_TRAIN_MONTH[leg], int(schedule[2:])
        )
    t0 = time.time()
    model = Booster(columns, params).fit(
        prep["frame"].loc[prep["rows"]], frame.y[prep["rows"]], sample_weight=weight
    )
    wall = time.time() - t0
    record = {
        "leg": leg,
        "seed": seed,
        "schedule": schedule,
        "features": features,
        "columns": len(columns),
        "fit_rows": int(prep["rows"].sum()),
        "wall_s": wall,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "sets": {},
    }
    if weight is not None:
        record["model_sha256"] = model_digest(model)
        record["weight_sum"] = float(weight.sum())
    kept: dict[str, pd.DataFrame] = {}
    for name, mask in frame.sets(leg).items():
        pred = model.predict(prep["frame"].loc[mask])
        ev = evaluate(
            frame.y[mask],
            pred,
            icao=frame.features.loc[mask, "icao"],
            flagged=prep["flagged"][mask],
        )
        entry = {
            "rows": int(mask.sum()),
            "flagged": int(prep["flagged"][mask].sum()),
            "pooled_ex_flagged": _pooled(ev),
            "per_airport": {
                str(r.icao): float(r.rmse_ex_flagged_s)
                for r in ev.itertuples()
                if r.icao != "all"
            },
            "sha256": digest(pred),
        }
        # The three July airports, where the leg is July's own validation set.
        entry["three"] = (
            rb._three_ex_flagged({"eval": ev})
            if leg == "july" and name == "valid"
            else None
        )
        # The ranking mix, where the rows are December's own leg. The gap's
        # rows are December's too, but its pair counts come from January to
        # June, and the pre-registration fixes no reference for that mix.
        mix = None
        buckets = None
        if leg == "december" and name == "valid":
            mix, buckets = ranking_mix(frame, prep, mask, pred)
        entry["ranking_mix"] = mix
        record["sets"][name] = entry
        got = pd.DataFrame(
            {
                "mvt_id": frame.dep.loc[mask, "MVT_ID_mvt"].to_numpy(),
                "icao": frame.features.loc[mask, "icao"].astype(str).to_numpy(),
                "takeoff": pd.to_datetime(
                    frame.dep.loc[mask, "MVT_TIME_UTC_mvt"], utc=True
                ).to_numpy(),
                "flagged": np.asarray(prep["flagged"][mask], dtype=bool),
                "y": np.asarray(frame.y[mask], dtype=float),
                "pred": np.asarray(pred, dtype=float),
            }
        )
        # The pair-frequency bucket, kept beside the prediction so that
        # ``--report`` can resample the mix without refitting anything. It is
        # a property of the leg and not of the fit, and every December fit
        # therefore carries the same column.
        if buckets is not None:
            got["bucket"] = np.asarray(buckets)
        kept[name] = got
    return record, kept


def _pooled(ev: pd.DataFrame) -> float:
    return float(ev.loc[ev["icao"] == "all", "rmse_ex_flagged_s"].iloc[0])


def ranking_mix(frame: Frame, prep: dict, mask, pred) -> tuple[float, np.ndarray]:
    """December's bucket figures pooled at the ranking set's bucket weights.

    The same arithmetic ``run_baselines.section_frequency`` prints, over the
    same rows: the pair counts from the leg's own reference, the buckets
    scored over the rows the record-quality term leaves to the physical
    model, and ``_pooled_at`` at ``RANKING_MIX``.
    """
    counts = prep["reference"].pair_counts(frame.features[mask])
    by = evaluate_by_bucket(
        frame.y[mask], pred, counts, keep=~prep["flagged"][mask]
    ).set_index("bucket")
    pooled = rb._pooled_at(
        {b: float(by.loc[b, "rmse_s"]) for b in by.index}, rb.RANKING_MIX
    )
    return pooled, frequency_bucket(counts).to_numpy()


# --------------------------------------------------------------------------
# the records on disk


def write_record(record: dict, kept: dict[str, pd.DataFrame]) -> Path:
    """One fit's figures and its kept predictions, under the fit's own name.

    The name is the record's, not a description of it: ``--report`` reads the
    frames back by the name the summary gives, so a file named any other way
    would be a set of predictions no section could find.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    name = record["name"]
    for set_name, frame in kept.items():
        frame.to_parquet(WORK / f"{name}__{set_name}.parquet", index=False)
    path = WORK / f"{name}.json"
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return path


def read_records() -> list[dict]:
    if not WORK.exists():
        return []
    out = []
    for path in sorted(WORK.glob("*.json")):
        got = json.loads(path.read_text(encoding="utf-8"))
        got["_stem"] = path.stem
        out.append(got)
    return out


def read_kept(stem_name: str, set_name: str) -> pd.DataFrame:
    return pd.read_parquet(WORK / f"{stem_name}__{set_name}.parquet")


def movements_on_disk() -> set[str]:
    """Every movement identifier this run touched, from the kept frames.

    Read back rather than carried in the summary: they are hundreds of
    thousands of numbers, the frames already hold them, and a summary file
    carrying them would be a second copy of the one thing the report must
    never print.
    """
    out: set[str] = set()
    for path in sorted(WORK.glob("*__*.parquet")):
        ids = pd.read_parquet(path, columns=["mvt_id"])["mvt_id"].dropna()
        out |= {f"{int(v)}" for v in ids.unique()}
    return out


def find(records: list[dict], **want) -> dict | None:
    for got in records:
        if all(got.get(k) == v for k, v in want.items()):
            return got
    return None


# --------------------------------------------------------------------------
# reading rules


def printed(x: float | None) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.1f}"


def no_worse(arm: float, reference: float) -> bool:
    """Not worse at the precision the reports print. Higher RMSE is worse."""
    return round(arm, 1) <= round(reference, 1) + 1e-9


def clears_floor(arm: float, reference: float, floor: float) -> bool:
    """A gain on the unrounded figure larger than the leg's floor."""
    return (reference - arm) > floor


def floor_of(singles) -> float:
    values = np.asarray(list(singles), dtype=float)
    return float(NOISE_FLOOR_FACTOR * np.std(values, ddof=1))


def day_of(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(frame["takeoff"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()


def paired_interval(arm: pd.DataFrame, base: pd.DataFrame, rows=None):
    """Arm minus base, day-clustered, the same draw for both arms.

    Both sides are the same rows of the same evaluation set, so they go into
    one row set and the draw that resamples its days resamples both arms
    together. That is what keeps the contrast paired, and it is the reason
    this reads the two predictions rather than two independent residual sets.
    """
    if rows is None:
        rows = ~arm["flagged"].to_numpy(dtype=bool)
    a = arm.loc[rows]
    b = base.loc[rows]
    if not len(a):
        return None
    sums = cluster_sums(
        {
            "arm": (a["y"] - a["pred"]).to_numpy(dtype=float),
            "base": (b["y"] - b["pred"]).to_numpy(dtype=float),
        },
        day_of(a),
    )
    return cluster_bootstrap(
        [{"s": sums}],
        lambda r: r["s"]["arm"] - r["s"]["base"],
        draws=BOOTSTRAP_DRAWS,
    )


def mix_interval(arm: pd.DataFrame, base: pd.DataFrame):
    """The same contrast at the ranking mix, days still the clusters.

    Each bucket is reduced to per-day sums over one shared day index, so a
    single draw of days resamples every bucket and both arms at once. Pooling
    the resampled bucket RMSEs at ``RANKING_MIX`` then gives the mix under
    that draw. ``cluster_bootstrap`` draws once per row set and applies it to
    every series in it, which is exactly what is wanted here.
    """
    rows = ~arm["flagged"].to_numpy(dtype=bool)
    a, b = arm.loc[rows], base.loc[rows]
    labels = arm["bucket"].to_numpy()[rows]
    codes, _ = pd.factorize(pd.Series(day_of(a)))
    days = int(codes.max()) + 1 if codes.size else 0
    if not days:
        return None
    series: dict[str, ClusterSums] = {}
    present = []
    for name in rb.FREQUENCY_BUCKETS:
        m = labels == name
        if not m.any():
            continue
        present.append(name)
        n = np.bincount(codes[m], minlength=days).astype(float)
        for side, frame in (("arm", a), ("base", b)):
            residual = (frame["y"] - frame["pred"]).to_numpy(dtype=float)[m]
            series[f"{side}:{name}"] = ClusterSums(
                sse=np.bincount(codes[m], weights=residual**2, minlength=days), n=n
            )

    def statistic(r):
        got = r["s"]
        pooled = {}
        for side in ("arm", "base"):
            pooled[side] = rb._pooled_at(
                {b: got[f"{side}:{b}"] for b in present}, rb.RANKING_MIX
            )
        return pooled["arm"] - pooled["base"]

    return cluster_bootstrap([{"s": series}], statistic, draws=BOOTSTRAP_DRAWS)


def step(arm: float, base: float) -> float:
    return arm - base


def inside_floor(difference: float | None, floor: float) -> str:
    """Whether a step is smaller than the floor of the leg it was taken on.

    The floors in this run run from 0.19 s to 1.02 s and printed precision is
    0.1 s, so a screen read at printed precision can fire on a step the same
    run calls seed noise. Saying which is which is not a change to any rule:
    every screen here is still decided at printed precision, and this column
    only says what the step was.
    """
    if difference is None:
        return "n/a"
    return "inside it" if abs(difference) < floor else "outside it"


def shown(value) -> str:
    return "n/a" if value is None else str(value)


# --------------------------------------------------------------------------
# the checks, each with the expectation it is read against


class Checks:
    """Every check this run makes, with its expectation, printed as it runs."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, what: str, expected: str, got: str, ok: bool, source: str) -> bool:
        self.rows.append(
            {
                "what": what,
                "expected": expected,
                "got": got,
                "ok": bool(ok),
                "source": source,
            }
        )
        print(f"  [{'ok ' if ok else 'STOP'}] {what}: expected {expected}, got {got}")
        return bool(ok)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if not r["ok"]]


def reproduces(kept: pd.DataFrame, stored: Path) -> tuple[str, bool, int]:
    """Largest absolute difference against a stored residual file.

    The stored file holds ``y - pred`` with the movement identifier beside it.
    The residual is formed the same way from this fit and joined on that
    identifier, which is read and never printed. A row count that disagrees,
    or an identifier on one side and not the other, is itself a failure: the
    check is over every row and not over the rows that happen to match.
    """
    if not stored.exists():
        return (f"{stored.name} is not on disk", False, 0)
    left = pd.read_parquet(stored, columns=["mvt_id", "residual"])
    right = pd.DataFrame(
        {
            "mvt_id": kept["mvt_id"].to_numpy(),
            "mine": (kept["y"] - kept["pred"]).to_numpy(),
        }
    )
    if len(left) != len(right):
        return (f"{len(right):,} rows against {len(left):,} stored", False, 0)
    merged = left.merge(right, on="mvt_id", how="inner", validate="one_to_one")
    if len(merged) != len(left):
        return (
            f"{len(merged):,} of {len(left):,} identifiers joined",
            False,
            len(merged),
        )
    worst = float(
        np.max(np.abs(merged["residual"].to_numpy() - merged["mine"].to_numpy()))
    )
    return (f"{worst:.6g} s over {len(merged):,} rows", worst == 0.0, len(merged))


def reproduction_checks(checks: Checks, records: dict, kept: dict) -> bool:
    """The four the pre-registration names, before any experiment is read."""
    print("\nreproduction, before anything is read")
    ok = True

    got, passed, _ = reproduces(
        kept["december_seed0"]["valid"], rb.RESIDUALS / "december_plan-diffs.parquet"
    )
    ok &= checks.add(
        "December first seed against december_plan-diffs.parquet",
        "a largest absolute difference of 0 on every row",
        got,
        passed,
        "the pre-registration, Reproduction",
    )
    got, passed, _ = reproduces(
        kept["july_seed0"]["valid"], rb.RESIDUALS / "july_plan-diffs.parquet"
    )
    ok &= checks.add(
        "July first seed against july_plan-diffs.parquet",
        "a largest absolute difference of 0 on every row",
        got,
        passed,
        "the pre-registration, Reproduction",
    )

    dec = records["december_seed0"]["sets"]["valid"]
    ok &= checks.add(
        "December first seed, pooled ex-flagged",
        f"{REF_DECEMBER_ALL:.1f} s",
        f"{printed(dec['pooled_ex_flagged'])} s",
        printed(dec["pooled_ex_flagged"]) == f"{REF_DECEMBER_ALL:.1f}",
        "baseline.md line 234",
    )
    ok &= checks.add(
        "December first seed, the ranking mix",
        f"{REF_RANKING_MIX:.1f} s",
        f"{printed(dec['ranking_mix'])} s",
        printed(dec["ranking_mix"]) == f"{REF_RANKING_MIX:.1f}",
        "baseline.md line 481",
    )

    jul = records["july_seed0"]["sets"]["valid"]
    ok &= checks.add(
        "July first seed, pooled ex-flagged",
        f"{REF_JULY_ALL:.1f} s",
        f"{printed(jul['pooled_ex_flagged'])} s",
        printed(jul["pooled_ex_flagged"]) == f"{REF_JULY_ALL:.1f}",
        "baseline.md line 270",
    )
    ok &= checks.add(
        "July first seed, the three",
        f"{REF_JULY_THREE:.1f} s",
        f"{printed(jul['three'])} s",
        printed(jul["three"]) == f"{REF_JULY_THREE:.1f}",
        "baseline.md line 894",
    )

    pooled = [
        records[f"july_seed{i}"]["sets"]["valid"]["pooled_ex_flagged"]
        for i in range(SEEDS)
    ]
    three = [records[f"july_seed{i}"]["sets"]["valid"]["three"] for i in range(SEEDS)]
    span = f"{min(pooled):.2f} to {max(pooled):.2f} s"
    want = f"{JULY_SINGLES_POOLED[0]:.2f} to {JULY_SINGLES_POOLED[1]:.2f} s"
    ok &= checks.add(
        "the July five singles, pooled", want, span, span == want, "baseline.md line 1062"
    )
    span = f"{min(three):.2f} to {max(three):.2f} s"
    want = f"{JULY_SINGLES_THREE[0]:.2f} to {JULY_SINGLES_THREE[1]:.2f} s"
    ok &= checks.add(
        "the July five singles, the three",
        want,
        span,
        span == want,
        "baseline.md line 1063",
    )

    average = _july_average_pooled(kept)
    ok &= checks.add(
        "the July five-seed average, pooled",
        f"{JULY_AVERAGE_POOLED:.2f} s",
        f"{average:.2f} s",
        f"{average:.2f}" == f"{JULY_AVERAGE_POOLED:.2f}",
        "baseline.md line 1023",
    )
    return ok


def averaged(kept: dict, names: list[str], set_name: str) -> pd.DataFrame:
    """The seed average as a prediction frame, scored exactly as a single is."""
    base = kept[names[0]][set_name].copy()
    stack = np.mean(
        [kept[n][set_name]["pred"].to_numpy(dtype=float) for n in names], axis=0
    )
    base["pred"] = stack
    return base


def score(frame: pd.DataFrame, three: bool = False) -> float:
    """Pooled ex-flagged RMSE from a kept frame, or the three-airport pooling."""
    keep = ~frame["flagged"].to_numpy(dtype=bool)
    part = frame.loc[keep]
    if not three:
        residual = (part["y"] - part["pred"]).to_numpy(dtype=float)
        return float(np.sqrt(np.mean(residual**2)))
    total, weighted = 0, 0.0
    for icao in JULY_AIRPORTS:
        rows = part.loc[part["icao"] == icao]
        if not len(rows):
            return float("nan")
        residual = (rows["y"] - rows["pred"]).to_numpy(dtype=float)
        weighted += float((residual**2).sum())
        total += len(rows)
    return float(np.sqrt(weighted / total)) if total else float("nan")


def _july_average_pooled(kept: dict) -> float:
    return score(averaged(kept, [f"july_seed{i}" for i in range(SEEDS)], "valid"))


# --------------------------------------------------------------------------
# Experiment 1: seed averaging on the three and on October


def experiment_1(records: dict, kept: dict, checks: Checks) -> dict:
    """The ten singles, the two averages, October's floor, and the verdicts.

    The rule of 19 September reads December and July pooled, which
    ``baseline.md`` already prints and which ``RULE_LEGS`` quotes. What is
    measured here is the pair of readings it asks for before the average may
    ship, and the guard of 20 September over them: on the three and on
    October, is the average better than every single, and is it worse than the
    ladder's seed by more than that leg's floor. Both verdicts are written,
    because the pre-registration asks Experiment 1 for both and the guard
    alone is half of it.
    """
    july_names = [f"july_seed{i}" for i in range(SEEDS)]
    october_names = [f"october_seed{i}" for i in range(SEEDS)]

    july_pooled = [records[n]["sets"]["valid"]["pooled_ex_flagged"] for n in july_names]
    july_three = [records[n]["sets"]["valid"]["three"] for n in july_names]
    gap_pooled = [records[n]["sets"]["gap"]["pooled_ex_flagged"] for n in july_names]
    october_pooled = [
        records[n]["sets"]["valid"]["pooled_ex_flagged"] for n in october_names
    ]

    average = {
        "july_pooled": score(averaged(kept, july_names, "valid")),
        "july_three": score(averaged(kept, july_names, "valid"), three=True),
        "gap_pooled": score(averaged(kept, july_names, "gap")),
        "october_pooled": score(averaged(kept, october_names, "valid")),
    }
    floors = {
        "October, all ten": floor_of(october_pooled),
        "the gap": floor_of(gap_pooled),
    }

    verdict = {}
    for label, singles, avg, floor, ladder in (
        ("the three", july_three, average["july_three"], FLOOR_JULY_THREE, july_three[0]),
        (
            "October",
            october_pooled,
            average["october_pooled"],
            floors["October, all ten"],
            october_pooled[0],
        ),
    ):
        worse_by = avg - ladder
        verdict[label] = {
            "beats_every_single": bool(avg < min(singles)),
            "best_single": float(min(singles)),
            "average": float(avg),
            "ladder_seed": float(ladder),
            "change": float(worse_by),
            "floor": float(floor),
            # The guard: worse than the ladder's seed by more than the floor.
            "guard_trips": bool(worse_by > floor),
        }
    tripped = [k for k, v in verdict.items() if v["guard_trips"]]

    # The rule of 19 September, which reads the two forward legs. Its figures
    # are ``baseline.md``'s, quoted as constants, so this is the rule applied
    # rather than the rule recomputed; the pre-registration asks Experiment 1
    # to write the verdict of the rule and the guard, and the guard alone is
    # half of what it asks for.
    rule = []
    for leg in RULE_LEGS:
        rule.append(
            {
                **leg,
                "beats_every_single": bool(leg["average"] < leg["best_single"]),
                "clears_floor": bool(leg["gain"] > leg["floor"]),
            }
        )
    rule_holds = all(r["beats_every_single"] and r["clears_floor"] for r in rule)

    checks.add(
        "Experiment 1, the rule of 19 September",
        "on both forward legs the average beats every single seed and its gain"
        " clears that leg's floor",
        (
            "holds on both: "
            + ", ".join(
                f"{r['leg']} {r['gain']:.2f} s against {r['floor']:.2f}" for r in rule
            )
            if rule_holds
            else "fails on "
            + " and ".join(
                r["leg"]
                for r in rule
                if not (r["beats_every_single"] and r["clears_floor"])
            )
        ),
        True,  # the rule's verdict is a result, not a failed expectation
        "baseline.md line 1023, quoted",
    )
    checks.add(
        "Experiment 1, the guard of 20 September",
        "the average is not worse than the ladder's seed by more than the floor,"
        " on the three or on October",
        "holds" if not tripped else "trips on " + " and ".join(tripped),
        True,  # a tripped guard is a result, not a failed expectation
        "the pre-registration, Experiment 1",
    )
    return {
        "july_pooled": july_pooled,
        "july_three": july_three,
        "gap_pooled": gap_pooled,
        "october_pooled": october_pooled,
        "average": average,
        "floors": {k: float(v) for k, v in floors.items()},
        "verdict": verdict,
        "guard_trips_on": tripped,
        "rule": rule,
        "rule_holds": bool(rule_holds),
    }


# --------------------------------------------------------------------------
# Experiment 2: recency weighting, in the sequence the pre-registration fixed


def experiment_2(records: dict, checks: Checks, floors: dict) -> dict:
    """The digest check, the October screen, the gap screen, the forward legs.

    Every step is the one the pre-registration fixed, in its order, and a
    schedule that fails a screen is not fitted on anything further. The
    sequence is written out rather than folded into a loop over criteria
    because the point of it is that each stop is visible.
    """
    out: dict = {"schedules": {}, "digest_check": {}}
    october_reference = records["october_seed0"]["sets"]["valid"]["pooled_ex_flagged"]
    gap_reference = records["july_seed0"]["sets"]["gap"]["pooled_ex_flagged"]
    out["references"] = {
        "october": float(october_reference),
        "gap": float(gap_reference),
        "december_all": REF_DECEMBER_ALL,
        "july_three": REF_JULY_THREE,
        "ranking_mix": REF_RANKING_MIX,
        "december_all_unrounded": float(
            records["december_seed0"]["sets"]["valid"]["pooled_ex_flagged"]
        ),
        "july_three_unrounded": float(records["july_seed0"]["sets"]["valid"]["three"]),
        "ranking_mix_unrounded": float(
            records["december_seed0"]["sets"]["valid"]["ranking_mix"]
        ),
        "july_all_unrounded": float(
            records["july_seed0"]["sets"]["valid"]["pooled_ex_flagged"]
        ),
    }

    # The digest check, before any figure is read. The mild schedule's October
    # fit, made twice: the predictions and the model dumps must be identical.
    print("\nExperiment 2, the digest check")
    first = records["october_H12_seed0"]
    second = records["october_H12_seed0_repeat"]
    same_pred = first["sets"]["valid"]["sha256"] == second["sets"]["valid"]["sha256"]
    same_model = first.get("model_sha256") is not None and first.get(
        "model_sha256"
    ) == second.get("model_sha256")
    out["digest_check"] = {
        "prediction_sha256": first["sets"]["valid"]["sha256"],
        "repeat_prediction_sha256": second["sets"]["valid"]["sha256"],
        "model_sha256": first.get("model_sha256"),
        "repeat_model_sha256": second.get("model_sha256"),
        "identical": bool(same_pred and same_model),
    }
    ok = checks.add(
        "Experiment 2, the digest check (H=12 on October, fitted twice)",
        "identical prediction vectors and identical model dumps",
        (
            "identical"
            if same_pred and same_model
            else f"predictions {'agree' if same_pred else 'differ'}, "
            f"model dumps {'agree' if same_model else 'differ'}"
        ),
        same_pred and same_model,
        "the pre-registration, Experiment 2",
    )
    if not ok:
        out["stopped"] = "the digest check"
        return out

    # 1. The October screen.
    print("\nExperiment 2, step 1: the October screen")
    survivors = []
    for h in HALF_LIVES:
        name = f"october_H{h}_seed0"
        arm = records[name]["sets"]["valid"]["pooled_ex_flagged"]
        passes = no_worse(arm, october_reference)
        out["schedules"][f"H={h}"] = {
            "october": float(arm),
            "october_step": float(step(arm, october_reference)),
            "october_passes": bool(passes),
            "stopped_at": None if passes else "the October screen",
        }
        print(
            f"  H={h}: October {printed(arm)} s against "
            f"{printed(october_reference)} s, "
            f"{'no worse' if passes else 'worse: it stops here'}"
        )
        if passes:
            survivors.append(h)

    # 2. The July fit, read on the gap first.
    print("\nExperiment 2, step 2: the July fit, read on the gap")
    still = []
    for h in survivors:
        name = f"july_H{h}_seed0"
        if name not in records:
            # Only reachable where a run did not finish: the run fits a
            # survivor before this is read. Said rather than raised, so that
            # --report over a partial run prints what it has.
            out["schedules"][f"H={h}"]["stopped_at"] = "not fitted"
            continue
        got = records[name]["sets"]
        arm = got["gap"]["pooled_ex_flagged"]
        passes = no_worse(arm, gap_reference)
        entry = out["schedules"][f"H={h}"]
        entry.update(
            gap=float(arm),
            gap_step=float(step(arm, gap_reference)),
            gap_passes=bool(passes),
            july_all=float(got["valid"]["pooled_ex_flagged"]),
            july_three=float(got["valid"]["three"]),
        )
        print(
            f"  H={h}: the gap {printed(arm)} s against {printed(gap_reference)} s, "
            f"{'no worse' if passes else 'worse: it stops here'}; "
            f"July all ten {printed(entry['july_all'])} s, "
            f"the three {printed(entry['july_three'])} s"
        )
        if passes:
            still.append(h)
        else:
            entry["stopped_at"] = "the gap screen"

    # 3. The forward legs.
    print("\nExperiment 2, step 3: the forward legs")
    for h in still:
        name = f"december_H{h}_seed0"
        if name not in records:
            out["schedules"][f"H={h}"]["stopped_at"] = "not fitted"
            continue
        got = records[name]["sets"]["valid"]
        entry = out["schedules"][f"H={h}"]
        entry.update(
            december_all=float(got["pooled_ex_flagged"]),
            ranking_mix=float(got["ranking_mix"]),
        )
        print(
            f"  H={h}: December {printed(entry['december_all'])} s, "
            f"the mix {printed(entry['ranking_mix'])} s, "
            f"the three {printed(entry['july_three'])} s"
        )

    # Adoption, by the rule and by nothing else.
    for h in HALF_LIVES:
        entry = out["schedules"][f"H={h}"]
        entry["adopted"] = False
        entry["why"] = None
        if entry.get("stopped_at"):
            # The screen it failed is a reading it was worse on, and the table
            # says so rather than printing "none" beside a schedule that
            # stopped because it was worse.
            entry["worse_on"] = {
                "the October screen": ["October"],
                "the gap screen": ["the gap"],
            }.get(entry["stopped_at"], [])
            entry["clears_floor_on"] = []
            entry["why"] = f"stopped at {entry['stopped_at']}"
            continue
        readings = {
            "October": (entry["october"], october_reference, None),
            "the gap": (entry["gap"], gap_reference, None),
            "December all ten": (
                entry["december_all"],
                REF_DECEMBER_ALL,
                FLOOR_DECEMBER_ALL,
            ),
            "July the three": (entry["july_three"], REF_JULY_THREE, FLOOR_JULY_THREE),
            "the ranking mix": (entry["ranking_mix"], REF_RANKING_MIX, None),
        }
        worse_on = [k for k, (a, r, _) in readings.items() if not no_worse(a, r)]
        gains = {
            "December all ten": clears_floor(
                entry["december_all"],
                out["references"]["december_all_unrounded"],
                FLOOR_DECEMBER_ALL,
            ),
            "July the three": clears_floor(
                entry["july_three"],
                out["references"]["july_three_unrounded"],
                FLOOR_JULY_THREE,
            ),
        }
        entry["worse_on"] = worse_on
        entry["clears_floor_on"] = [k for k, v in gains.items() if v]
        entry["adopted"] = bool(not worse_on and entry["clears_floor_on"])
        if worse_on:
            entry["why"] = "worse at printed precision on " + ", ".join(worse_on)
        elif not entry["clears_floor_on"]:
            entry["why"] = (
                "no worse on all five readings, and no gain clearing the floor on"
                " December all ten or July the three: a change to how the model is"
                " fitted that moves nothing outside seed noise is not adopted"
            )
        else:
            entry["why"] = "no worse on all five readings, and the gain clears a floor"

    qualifying = [h for h in HALF_LIVES if out["schedules"][f"H={h}"]["adopted"]]
    if len(qualifying) == 2:
        mild, strong = out["schedules"]["H=12"], out["schedules"]["H=3"]
        # "beats it on all five readings by more than the respective floors".
        # October's and the gap's floors are this run's measured ones; the
        # ranking mix has no floor of its own, so December's is used for it,
        # the leg the mix reweights, and the report says so.
        beats_all = all(
            clears_floor(strong[key], mild[key], floor)
            for key, floor in (
                ("october", floors["October, all ten"]),
                ("gap", floors["the gap"]),
                ("december_all", FLOOR_DECEMBER_ALL),
                ("july_three", FLOOR_JULY_THREE),
                ("ranking_mix", FLOOR_DECEMBER_ALL),
            )
        )
        out["adopted"] = "H=3" if beats_all else "H=12"
        out["tie_break"] = (
            "both qualify; the strong schedule beats the mild on all five readings"
            " by more than the respective floors"
            if beats_all
            else "both qualify; the mild schedule is adopted, the strong one not"
            " beating it on all five readings by more than the respective floors"
        )
    elif qualifying:
        out["adopted"] = f"H={qualifying[0]}"
        out["tie_break"] = None
    else:
        out["adopted"] = None
        out["tie_break"] = None
    return out


# --------------------------------------------------------------------------
# Experiment 3: the season columns


def experiment_3(records: dict, checks: Checks) -> dict:
    december = records["december_noseason_seed0"]["sets"]["valid"]
    july = records["july_noseason_seed0"]["sets"]["valid"]
    return {
        "december_all": float(december["pooled_ex_flagged"]),
        "ranking_mix": float(december["ranking_mix"]),
        "july_all": float(july["pooled_ex_flagged"]),
        "july_three": float(july["three"]),
        "columns_removed": list(SEASON_COLUMNS),
        "steps": {
            "December all ten": float(december["pooled_ex_flagged"] - REF_DECEMBER_ALL),
            "July the three": float(july["three"] - REF_JULY_THREE),
            "the ranking mix": float(december["ranking_mix"] - REF_RANKING_MIX),
            "July all ten": float(july["pooled_ex_flagged"] - REF_JULY_ALL),
        },
    }


def load_records() -> dict:
    """Every fit's record, keyed by the fit's name, read back from disk."""
    out = {}
    for got in read_records():
        if got.get("name"):
            out[got["name"]] = got
    return out


def derive(records: dict, checks: Checks) -> dict:
    """Every check and every verdict, from the records on disk alone.

    Called once by the fitting run and again by ``--report``, over the same
    records read back from disk, so the report is a function of what was
    written and not of what the fitting process happened to be holding. A
    check or a verdict that came out differently on the second pass would be
    one the records do not support, which is the reason the fitting run
    derives its own report this way rather than from the frames in memory.
    """
    kept = {}
    for name, record in records.items():
        if record["schedule"] == "none" and record["features"] == "submission":
            kept[name] = {s: read_kept(name, s) for s in record["sets"]}
    if not reproduction_checks(checks, records, kept):
        return {"stopped": "reproduction"}
    one = experiment_1(records, kept, checks)
    return {
        "stopped": None,
        "experiment_1": one,
        "experiment_2": experiment_2(records, checks, one["floors"]),
        "experiment_3": experiment_3(records, checks),
    }


# --------------------------------------------------------------------------
# the run


# The twenty fits, named and in the order they are made. The reproduction
# fits come first because nothing else may be read until they pass, and the
# digest check's repeat sits beside its own fit rather than at the end.
def run(frame: Frame) -> dict:
    checks = Checks()
    records: dict[str, dict] = {}
    kept: dict[str, dict] = {}

    def fit(name: str, leg: str, *, seed=0, schedule="none", features="submission"):
        print(f"\nfitting {name}")
        record, frames = fit_once(
            frame, leg, seed=seed, schedule=schedule, features=features
        )
        record["name"] = name
        records[name] = record
        kept[name] = frames
        write_record(record, frames)
        for set_name, entry in record["sets"].items():
            print(
                f"  {set_name:>5s}: {printed(entry['pooled_ex_flagged'])} s pooled"
                + (f", {printed(entry['three'])} s the three" if entry["three"] else "")
                + (
                    f", {printed(entry['ranking_mix'])} s the mix"
                    if entry["ranking_mix"]
                    else ""
                )
                + f"  ({record['wall_s']:.0f} s,"
                f" peak {record['peak_rss_kib'] / 1048576:.1f} GiB)"
            )
        return record

    # The reproduction fits.
    fit("december_seed0", "december")
    for i in range(SEEDS):
        fit(f"july_seed{i}", "july", seed=i)
    if not reproduction_checks(checks, records, kept):
        return {"checks": checks.rows, "stopped": "reproduction", "records": records}

    # Experiment 1's October seeds.
    for i in range(SEEDS):
        fit(f"october_seed{i}", "october", seed=i)

    # Experiment 2. The digest check's two fits first, then the sequence.
    fit("october_H12_seed0", "october", schedule="H=12")
    fit("october_H12_seed0_repeat", "october", schedule="H=12")
    fit("october_H3_seed0", "october", schedule="H=3")
    # The screen decides what is fitted next, so it is read here rather than
    # after a loop that has already fitted everything.
    october_reference = records["october_seed0"]["sets"]["valid"]["pooled_ex_flagged"]
    survivors = [
        h
        for h in HALF_LIVES
        if no_worse(
            records[f"october_H{h}_seed0"]["sets"]["valid"]["pooled_ex_flagged"],
            october_reference,
        )
    ]
    gap_reference = records["july_seed0"]["sets"]["gap"]["pooled_ex_flagged"]
    for h in survivors:
        fit(f"july_H{h}_seed0", "july", schedule=f"H={h}")
    for h in survivors:
        arm = records[f"july_H{h}_seed0"]["sets"]["gap"]["pooled_ex_flagged"]
        if no_worse(arm, gap_reference):
            fit(f"december_H{h}_seed0", "december", schedule=f"H={h}")

    # Experiment 3.
    fit("december_noseason_seed0", "december", features="no-season")
    fit("july_noseason_seed0", "july", features="no-season")

    # Every check and verdict again, from the records on disk rather than
    # from what this process is holding, so that --report over the same
    # records reaches the same report. The reproduction checks above ran on
    # the frames in memory and these run on the frames as written, so the two
    # passes also say that nothing was lost on the way to disk.
    derived = Checks()
    got = derive(load_records(), derived)
    return {
        "checks": derived.rows,
        "records": records,
        "fits": len(records),
        # Summed from the per-fit walls, which is the figure the report's own
        # table prints. The run's wall clock covers the scoring, the paired
        # bootstraps and the reproduction joins as well, and the two are not
        # the same number.
        "fit_seconds": float(sum(r["wall_s"] for r in records.values())),
        **got,
    }


# --------------------------------------------------------------------------
# the report


def table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    bar = ["---" if a == "-" else ("---:" if a == "r" else ":---") for a in align]
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(bar) + " |"] + [
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows
    ]


def interval_phrase(got) -> str:
    return "n/a" if got is None else f"[{got.low:+,.1f}, {got.high:+,.1f}]"


def maybe_interval(arm_frame, base_frame, difference: float, three: bool = False) -> str:
    """The day-clustered interval, where the step is inside a second."""
    if abs(difference) >= INTERVAL_BELOW_S:
        return "not inside a second"
    rows = ~arm_frame["flagged"].to_numpy(dtype=bool)
    if three:
        rows = rows & arm_frame["icao"].isin(JULY_AIRPORTS).to_numpy()
    return interval_phrase(paired_interval(arm_frame, base_frame, rows=rows))


def section_checks(summary: dict) -> list[str]:
    rows = [
        [r["what"], r["expected"], r["got"], "ok" if r["ok"] else "**stop**", r["source"]]
        for r in summary["checks"]
    ]
    return [
        "## The checks, and what each was read against",
        "",
        "Each row is a check this run made, its expectation as the"
        " pre-registration or `baseline.md` fixed it, and what the run found."
        " The reproduction rows come before any experiment's figure is read:"
        " a fit that does not reproduce the ladder's stored residuals would"
        " make every figure below a reading of a model that cannot be"
        " repeated.",
        "",
        *table(
            ["check", "expected", "found", "verdict", "source"],
            rows,
            "-----",
        ),
        "",
    ]


def section_fits(summary: dict) -> list[str]:
    rows = []
    for name, record in summary["records"].items():
        for set_name, entry in record["sets"].items():
            rows.append(
                [
                    # The record's own name, **with its underscores**. That is
                    # what matters, and the code span is only for reading: a
                    # name rendered with spaces instead puts the bare token
                    # ``H12`` in the report, and that is a stand identifier at
                    # one of the ten airports, so the guard below refuses the
                    # whole report. The backtick does not help, because it is
                    # not a word character and ``(?<![\w])H12(?![\w])`` matches
                    # inside a code span; the underscore is a word character
                    # and is what defeats the match. Any label of the form
                    # letter plus digits can collide the same way, and the
                    # answer to a refusal is always to rename the label, never
                    # to loosen the guard.
                    f"`{name}`",
                    set_name,
                    record["schedule"],
                    record["features"],
                    f"{entry['rows']:,}",
                    f"{entry['flagged']:,}",
                    printed(entry["pooled_ex_flagged"]),
                    printed(entry["three"]),
                    printed(entry["ranking_mix"]),
                    f"{record['wall_s']:.0f}",
                    f"{record['peak_rss_kib'] / 1048576:.1f}",
                ]
            )
    return [
        "## Every fit",
        "",
        f"{len(summary['records'])} fits of the submission rung on one feature"
        " frame, sequential, at the ladder's eight threads. The peak is the"
        " process's own high-water mark at the end of that fit and not that"
        " fit's alone: the frame is held for the whole run, so the first fit"
        " carries most of it. The run's own peak is in the header above."
        " `the three` is printed where the evaluation set is July's own, and"
        " `the mix` where it is December's own; the gap's rows are December's"
        " but its pair counts come from January to June, and the"
        " pre-registration fixes no reference for that mix, so none is"
        " computed.",
        "",
        *table(
            [
                "fit",
                "set",
                "schedule",
                "features",
                "scored rows",
                "flagged scored rows",
                "pooled ex-flagged (s)",
                "the three (s)",
                "the mix (s)",
                "wall (s)",
                "peak (GiB)",
            ],
            rows,
            "----rrrrrrr",
        ),
        "",
    ]


def section_experiment_1(summary: dict) -> list[str]:
    one = summary["experiment_1"]
    singles = table(
        [
            "leg",
            "seed 0",
            "seed 1",
            "seed 2",
            "seed 3",
            "seed 4",
            "the 5 averaged",
            "floor (s)",
        ],
        [
            ["July, all ten"]
            + [f"{v:.2f}" for v in one["july_pooled"]]
            + [f"{one['average']['july_pooled']:.2f}", f"{FLOOR_JULY_ALL:.2f}"],
            ["July, the three"]
            + [f"{v:.2f}" for v in one["july_three"]]
            + [f"{one['average']['july_three']:.2f}", f"{FLOOR_JULY_THREE:.2f}"],
            ["October, all ten"]
            + [f"{v:.2f}" for v in one["october_pooled"]]
            + [
                f"{one['average']['october_pooled']:.2f}",
                f"{one['floors']['October, all ten']:.2f}",
            ],
            ["the gap (July's fits on December)"]
            + [f"{v:.2f}" for v in one["gap_pooled"]]
            + [f"{one['average']['gap_pooled']:.2f}", f"{one['floors']['the gap']:.2f}"],
        ],
        "-rrrrrrr",
    )
    rows = []
    for label, got in one["verdict"].items():
        rows.append(
            [
                label,
                f"{got['ladder_seed']:.2f}",
                f"{got['average']:.2f}",
                f"{got['change']:+.2f}",
                f"{got['floor']:.2f}",
                "yes" if got["beats_every_single"] else "no",
                "**trips**" if got["guard_trips"] else "holds",
            ]
        )
    rule_rows = [
        [
            r["leg"],
            f"{r['ladder_seed']:.1f}",
            f"{r['average']:.2f}",
            f"{r['gain']:.2f}",
            f"{r['floor']:.2f}",
            "yes" if r["beats_every_single"] else "no",
            "yes" if r["clears_floor"] else "no",
            r["source"],
        ]
        for r in one["rule"]
    ]
    return [
        "## What Experiment 1 measured: seed averaging on the three and on October",
        "",
        "The five July seeds and the five October seeds, singly and averaged."
        " The first July seed and the first December seed are the reproduction"
        " fits above, so the averages here are averages of fits that reproduce"
        " the ladder's. Two floors are measured here, each from its own five"
        " singles by the rule `baseline.md`'s *The noise floor* states, root"
        " two times the sample standard deviation: October's from the five"
        " October fits, and the gap's from the five July fits scored on"
        " December. December's, July's and the three's are the printed ones.",
        "",
        *singles,
        "",
        "**The rule of 19 September,** in the words it was fixed in"
        ' (`baseline.md` line 1010): "Adopt it for the submission if, on both'
        " forward legs, the five-seed average beats every one of the five"
        " single seeds and its gain over the ladder's seed clears that leg's"
        " seed floor; before it ships it is also measured on the three July"
        ' airports and on October, and both are printed."',
        "",
        "**The guard of 20 September,** added before either measurement"
        " existed: the average is not adopted if, on the three or on October,"
        " it is worse than the ladder's seed by more than that leg's floor.",
        "",
        *table(
            [
                "column",
                "the ladder's seed (s)",
                "the 5 averaged (s)",
                "change (s)",
                "floor (s)",
                "beats every single",
                "the guard",
            ],
            rows,
            "-rrrr--",
        ),
        "",
        (
            "**The guard holds on both columns.**"
            if not one["guard_trips_on"]
            else "**The guard trips on " + " and on ".join(one["guard_trips_on"]) + ".**"
        ),
        "",
        "**What the floor does on these two columns, and what it does not.** On"
        " the three and on October the floor is here for the guard alone, and"
        " the guard sizes a *regression*: it asks whether the average is worse"
        " than the ladder's seed by more than the floor. No test on either"
        " column asks the gain to clear anything, so a gain of 1.01 s beside a"
        " floor of 1.02 s is not a gain that failed a test. The rule's"
        " clears-the-floor test lives on the two forward legs below, and is"
        " read against those legs' floors.",
        "",
        "**Changes are computed on the unrounded figures**, as the"
        " pre-registration requires, while the columns beside them print to two"
        " decimals. On the three that is visible: 237.80 less 236.80 reads as"
        " 1.00 in the printed columns where the change is 1.01, because neither"
        " figure is exactly what it prints.",
        "",
        "**The rule of 19 September, on the two forward legs it reads.** This"
        " run does not fit them for the rule: December is one seed here, and"
        " July's five are fitted for the columns above. The figures are"
        " `baseline.md`'s, quoted as constants with their lines, and the rule is"
        " applied to them rather than recomputed. July's are reproduced by this"
        " run's own five seeds, which is what the reproduction checks above"
        " establish; December's are not measured here at all.",
        "",
        *table(
            [
                "leg",
                "the ladder's seed (s)",
                "the 5 averaged (s)",
                "gain (s)",
                "floor (s)",
                "beats every single",
                "clears the floor",
                "source",
            ],
            rule_rows,
            "-rrrr---",
        ),
        "",
        (
            "**At the present parameters the rule holds on both forward legs and"
            " the guard holds on both columns.**"
            if one["rule_holds"] and not one["guard_trips_on"]
            else "**The rule "
            + ("holds" if one["rule_holds"] else "fails")
            + " on the forward legs, and the guard "
            + (
                "holds on both columns.**"
                if not one["guard_trips_on"]
                else "trips on " + " and on ".join(one["guard_trips_on"]) + ".**"
            )
        )
        + " The decision is still deferred and nothing ships from this run. The"
        " parameters may move in the wider sweep, and the pre-registration fixes"
        " that the generator's `SEED_AVERAGE_PENDING` is emptied after the"
        " ladder run that follows the sweep and not before. Until then the"
        " submission is fitted from a single seed and"
        " `taxiout.models.SEED_AVERAGE_ADOPTED` stays `False`.",
        "",
    ]


def section_experiment_2(summary: dict) -> list[str]:
    two = summary["experiment_2"]
    lines = [
        "## What Experiment 2 measured: recency weighting",
        "",
        "Two schedules fixed in advance, `w = 0.5 ** (age / H)` over the"
        " booster's fitting rows alone, with `age` the whole number of"
        " calendar months back from the leg's last training month. The"
        " record-quality model, `fit_rows`, the encoding and the scoring are"
        " unweighted and unchanged.",
        "",
    ]
    digest_check = two["digest_check"]
    lines += [
        "**The digest check, before any figure was read.** `deterministic` was"
        " measured without sample weights, so the mild schedule's October fit"
        " was made twice and the two prediction vectors and the two model"
        " dumps compared.",
        "",
        *table(
            ["what", "first fit", "repeat", "verdict"],
            [
                [
                    "prediction vector",
                    f"`{digest_check['prediction_sha256'][:16]}...`",
                    f"`{digest_check['repeat_prediction_sha256'][:16]}...`",
                    "identical"
                    if digest_check["prediction_sha256"]
                    == digest_check["repeat_prediction_sha256"]
                    else "**differ**",
                ],
                [
                    "model dump",
                    f"`{str(digest_check['model_sha256'])[:16]}...`",
                    f"`{str(digest_check['repeat_model_sha256'])[:16]}...`",
                    "identical"
                    if digest_check["model_sha256"] == digest_check["repeat_model_sha256"]
                    else "**differ**",
                ],
            ],
            "----",
        ),
        "",
    ]
    if two.get("stopped"):
        return lines + [
            f"**The experiment stopped at {two['stopped']}.** No schedule was"
            " fitted on any leg beyond it and nothing is read from the fits"
            " that were made.",
            "",
        ]
    reference = two["references"]
    floors = summary["experiment_1"]["floors"]
    rows = []
    for h in HALF_LIVES:
        got = two["schedules"][f"H={h}"]
        rows.append(
            [
                f"H = {h}",
                printed(got["october"]),
                f"{got['october_step']:+.2f}",
                f"{floors['October, all ten']:.2f}",
                inside_floor(got["october_step"], floors["October, all ten"]),
                "no worse" if got["october_passes"] else "**worse: stops**",
                printed(got.get("gap")),
                f"{got['gap_step']:+.2f}" if "gap_step" in got else "not fitted",
                f"{floors['the gap']:.2f}",
                inside_floor(got.get("gap_step"), floors["the gap"]),
                (
                    "not reached"
                    if "gap_passes" not in got
                    else ("no worse" if got["gap_passes"] else "**worse: stops**")
                ),
            ]
        )
    lines += [
        f"**Step 1, the October screen**, against the unweighted October first"
        f" seed at {printed(reference['october'])} s. **Step 2, the gap**, the"
        " July leg's own fit scored on December, against the unweighted gap"
        f" first seed at {printed(reference['gap'])} s. A schedule worse at"
        " printed precision stops where it is worse and is not fitted"
        " further.",
        "",
        "Each step is printed beside the floor of the leg it was taken on, both"
        " of them measured in this run from that leg's own five single-seed"
        " fits. The screen is fixed at printed precision and reads nothing"
        " else; the floor column is there so that a stop can be read for what"
        " it is, because printed precision is 0.1 s and every floor in this run"
        " is larger than that. A step inside its leg's floor is a step the same"
        " run calls seed noise.",
        "",
        *table(
            [
                "schedule",
                "October (s)",
                "step",
                "October's floor",
                "against it",
                "the October screen",
                "the gap (s)",
                "step",
                "the gap's floor",
                "against it",
                "the gap screen",
            ],
            rows,
            "-rrr--rrr--",
        ),
        "",
    ]
    reached = [h for h in HALF_LIVES if "december_all" in two["schedules"][f"H={h}"]]
    if reached:
        rows = []
        for h in reached:
            got = two["schedules"][f"H={h}"]
            rows.append(
                [
                    f"H = {h}",
                    printed(got["december_all"]),
                    f"{got['december_all'] - REF_DECEMBER_ALL:+.2f}",
                    printed(got["july_three"]),
                    f"{got['july_three'] - REF_JULY_THREE:+.2f}",
                    printed(got["ranking_mix"]),
                    f"{got['ranking_mix'] - REF_RANKING_MIX:+.2f}",
                    printed(got["july_all"]),
                ]
            )
        lines += [
            "**Step 3, the forward legs**, against 217.6 s (December all ten),"
            " 237.8 s (July the three) and 218.6 s (the ranking mix), with July"
            " all ten beside them. The step is against the printed reference;"
            " the floor test below is on the unrounded figure of this run's own"
            " unweighted fit, which reproduces it.",
            "",
            *table(
                [
                    "schedule",
                    "December all ten (s)",
                    "step",
                    "July the three (s)",
                    "step",
                    "the ranking mix (s)",
                    "step",
                    "July all ten (s)",
                ],
                rows,
                "-rrrrrrr",
            ),
            "",
        ]
    lines += section_experiment_2_intervals(summary)
    rows = []
    for h in HALF_LIVES:
        got = two["schedules"][f"H={h}"]
        rows.append(
            [
                f"H = {h}",
                got.get("stopped_at") or "reached step 3",
                ", ".join(got.get("worse_on") or []) or "none",
                ", ".join(got.get("clears_floor_on") or []) or "none",
                "**adopted**" if got["adopted"] else "not adopted",
            ]
        )
    lines += [
        '**Adoption, by the rule and by nothing else.** "A schedule is adopted'
        " for the submission only if it is no worse at printed precision on all"
        " five readings (October, the gap, December all ten, July the three,"
        " the ranking mix) and its gain clears the floor on at least one of"
        " December all ten and July the three. A change to how the model is"
        " fitted that moves nothing outside seed noise is not adopted. If both"
        " schedules qualify, the mild one is adopted unless the strong one"
        ' beats it on all five readings by more than the respective floors."',
        "",
        *table(
            [
                "schedule",
                "how far it got",
                "worse at printed precision on",
                "clears a floor on",
                "verdict",
            ],
            rows,
            "-----",
        ),
        "",
        (
            f"**Adopted: {two['adopted']}.**"
            if two["adopted"]
            else "**Neither schedule is adopted.**"
        )
        + " "
        + " ".join(f"H = {h}: {two['schedules'][f'H={h}']['why']}." for h in HALF_LIVES)
        + (f" {two['tie_break']}." if two.get("tie_break") else "")
        + " Nothing in `taxiout.models` or `scripts/make_submission.py` is"
        " changed by this run either way.",
        "",
        "**What stopped each schedule, read against the floors above.** The"
        " strong schedule is worse on October by more than October's own floor,"
        " so that stop is a regression the run can measure. The mild schedule"
        " is a different case, and the two should not be read as one. It gains"
        " on the July forward leg by more than that leg's floor and with an"
        " interval clear of zero, and it moves nothing outside seed noise on"
        " October or on the gap: both of those steps are smaller than their own"
        " leg's floor and both intervals span zero. It is not adopted because"
        " the screen is fixed at printed precision and, at that precision, the"
        " gap reading is worse. **The stop is correct and the screen worked as"
        " it was fixed. The reason is the screen, not a measured loss on the"
        " gap.**",
        "",
        "That is what the gap leg can and cannot say here. It was built so that"
        " the recent training months are the seasonally furthest from the month"
        " being scored, which is the July 2026 geometry as nearly as one"
        " training year permits, and on it the mild schedule did nothing"
        " measurable in either direction. So this run does not separate recency"
        " from season; it shows that the schedule which helps where the two"
        " agree does not carry to the leg where they do not. The confound"
        " stands as a limitation that no leg inside one training year can"
        " resolve, which is what the pre-registration says of it, and no"
        " recency term ships.",
        "",
        *section_experiment_2_correction(),
    ]
    return lines


# The correction of 20 September, recorded in the file it corrects.
#
# The first published version of this report read the gap step as a loss. It
# is not one, and a reader of the report alone should not have to diff the
# history to find that out, because the paper's limitation section is written
# from this file. The superseded phrase is quoted exactly so that the two can
# be told apart, and the verdict is unchanged either way: neither schedule was
# adopted then and neither is adopted now.
SUPERSEDED_GAP_READING = (
    "the mild schedule gains where recency and season agree and loses on the"
    " one leg where they do not, which is the sign the gap leg was built to"
    " expose"
)
CORRECTION_COMMIT = "16a4f03"


def section_experiment_2_correction() -> list[str]:
    return [
        "### A correction to this section, 20 September",
        "",
        "The first published version of this file, committed as"
        f" `{CORRECTION_COMMIT}`, read the gap step as a measured loss. That"
        f' commit\'s message put it as: "{SUPERSEDED_GAP_READING}." It is'
        " superseded by the reading above, and this paragraph is here because"
        " the paper's limitation section is written from this report and a"
        " reader of the report alone should not have to go to the history to"
        " learn that the reading changed.",
        "",
        "What is wrong with it: the gap step is +0.10 s against that leg's own"
        " floor of 0.40 s, measured in the same run from the same five July"
        " fits, with a day-clustered interval of [-0.4, +0.6] that spans zero."
        " A step a quarter of its leg's floor is not a loss the run can see."
        " The superseded sentence also reads the July gains and the gap step as"
        " one symmetric finding, and they are not: the July gains are outside"
        " their floors and the gap step is inside its own.",
        "",
        "**No figure moved.** Every number in the first version is in this one"
        " and unchanged, and the verdict is the same: neither schedule was"
        " adopted then and neither is adopted now. What changed is the"
        " description of why the mild schedule stopped, which is the screen"
        " reading printed precision rather than a loss on the gap.",
        "",
    ]


def section_experiment_2_intervals(summary: dict) -> list[str]:
    """The day-clustered interval on every step inside a second."""
    two = summary["experiment_2"]
    floors = summary["experiment_1"]["floors"]
    rows = []
    for h in HALF_LIVES:
        got = two["schedules"][f"H={h}"]
        arms = [
            (
                "October",
                f"october_H{h}_seed0",
                "october_seed0",
                "valid",
                False,
                floors["October, all ten"],
            ),
            (
                "the gap",
                f"july_H{h}_seed0",
                "july_seed0",
                "gap",
                False,
                floors["the gap"],
            ),
            (
                "December all ten",
                f"december_H{h}_seed0",
                "december_seed0",
                "valid",
                False,
                FLOOR_DECEMBER_ALL,
            ),
            (
                "July the three",
                f"july_H{h}_seed0",
                "july_seed0",
                "valid",
                True,
                FLOOR_JULY_THREE,
            ),
            (
                "July all ten",
                f"july_H{h}_seed0",
                "july_seed0",
                "valid",
                False,
                FLOOR_JULY_ALL,
            ),
        ]
        for label, arm_name, base_name, set_name, three, floor in arms:
            if not (WORK / f"{arm_name}__{set_name}.parquet").exists():
                continue
            arm = read_kept(arm_name, set_name)
            base = read_kept(base_name, set_name)
            difference = score(arm, three=three) - score(base, three=three)
            rows.append(
                [
                    f"H = {h}",
                    label,
                    f"{difference:+.2f}",
                    f"{floor:.2f}",
                    inside_floor(difference, floor),
                    maybe_interval(arm, base, difference, three=three),
                ]
            )
        if "ranking_mix" in got:
            arm = read_kept(f"december_H{h}_seed0", "valid")
            base = read_kept("december_seed0", "valid")
            difference = got["ranking_mix"] - two["references"]["ranking_mix_unrounded"]
            phrase = "not inside a second"
            if abs(difference) < INTERVAL_BELOW_S:
                phrase = interval_phrase(mix_interval(arm, base))
            rows.append(
                [
                    f"H = {h}",
                    "the ranking mix",
                    f"{difference:+.2f}",
                    f"{FLOOR_DECEMBER_ALL:.2f}",
                    inside_floor(difference, FLOOR_DECEMBER_ALL),
                    phrase,
                ]
            )
    if not rows:
        return []
    return [
        "**The interval on every step inside a second, and the floor beside"
        " it.** The day is the cluster, and the two arms are the same rows of"
        " the same evaluation set, so one draw of days resamples both and the"
        " contrast stays paired. The mix's interval pools the resampled bucket"
        " figures at `RANKING_MIX` under that same draw. A step of a second or"
        " more is outside what the standing rule asks an interval for and none"
        " is printed for it. The floor column is the floor of the leg the step"
        " was taken on, so that a step can be read against the two instruments"
        " this repository has for the size of a difference.",
        "",
        *table(
            [
                "schedule",
                "reading",
                "step (s)",
                "that leg's floor (s)",
                "against it",
                "95% interval",
            ],
            rows,
            "--rr--",
        ),
        "",
        "The ranking mix has no floor of its own, because nothing is refitted"
        " five times at the mix. December's is printed beside it, that being"
        " the leg the mix reweights, and the column says which floor it is"
        " rather than implying the mix was measured five times.",
        "",
    ]


def section_experiment_3(summary: dict) -> list[str]:
    three = summary["experiment_3"]
    rows = []
    outside_floor = []
    for label, arm_name, base_name, set_name, is_three, reference, floor in (
        (
            "December all ten",
            "december_noseason_seed0",
            "december_seed0",
            "valid",
            False,
            REF_DECEMBER_ALL,
            FLOOR_DECEMBER_ALL,
        ),
        (
            "July the three",
            "july_noseason_seed0",
            "july_seed0",
            "valid",
            True,
            REF_JULY_THREE,
            FLOOR_JULY_THREE,
        ),
        (
            "July all ten",
            "july_noseason_seed0",
            "july_seed0",
            "valid",
            False,
            REF_JULY_ALL,
            FLOOR_JULY_ALL,
        ),
    ):
        arm = read_kept(arm_name, set_name)
        base = read_kept(base_name, set_name)
        value = score(arm, three=is_three)
        difference = value - score(base, three=is_three)
        phrase = maybe_interval(arm, base, difference, three=is_three)
        if abs(value - reference) >= floor:
            outside_floor.append(label)
        rows.append(
            [
                label,
                f"{reference:.1f}",
                printed(value),
                f"{value - reference:+.2f}",
                f"{floor:.2f}",
                inside_floor(value - reference, floor),
                phrase,
            ]
        )
    difference = three["ranking_mix"] - REF_RANKING_MIX
    phrase = "not inside a second"
    if abs(difference) < INTERVAL_BELOW_S:
        phrase = interval_phrase(
            mix_interval(
                read_kept("december_noseason_seed0", "valid"),
                read_kept("december_seed0", "valid"),
            )
        )
    if abs(difference) >= FLOOR_DECEMBER_ALL:
        outside_floor.append("the ranking mix")
    rows.append(
        [
            "the ranking mix",
            f"{REF_RANKING_MIX:.1f}",
            printed(three["ranking_mix"]),
            f"{difference:+.2f}",
            f"{FLOOR_DECEMBER_ALL:.2f}",
            inside_floor(difference, FLOOR_DECEMBER_ALL),
            phrase,
        ]
    )
    printed_intervals = [r for r in rows if r[-1].startswith("[")]
    crossing = [r for r in printed_intervals if r[-1].startswith("[-") and "+" in r[-1]]
    return [
        "## What Experiment 3 measured: the season columns",
        "",
        "The submission rung refitted on December and on July without"
        f" {', '.join('`' + c + '`' for c in SEASON_COLUMNS)}, first seed,"
        " everything else unchanged. Positive is worse, as everywhere in this"
        " report: the column is the arm minus the thing it is read against.",
        "",
        *table(
            [
                "reading",
                "the reference (s)",
                "without the season columns (s)",
                "step",
                "that leg's floor (s)",
                "against it",
                "95% interval",
            ],
            rows,
            "-rrrr--",
        ),
        "",
        "**The sign is not the same on the two legs.** Removing the columns"
        " improves December and the mix and costs July, on both of July's"
        " columns.",
        "",
        f"**The two instruments disagree here, and the report says so rather"
        f" than leaving it to be noticed.** {len(outside_floor)} of the"
        f" {len(rows)} steps are outside the floor of the leg they were taken"
        f" on ({', '.join(outside_floor)}), while"
        + (
            f" every one of the {len(printed_intervals)} intervals that print spans zero."
            if len(crossing) == len(printed_intervals)
            else f" {len(crossing)} of the {len(printed_intervals)} intervals"
            " that print span zero."
        )
        + " July on the three is outside a second and carries no"
        " interval by the standing rule, and the ranking mix, read against"
        " December's floor because it has none of its own, is the one step"
        " inside it. So on three contrasts the floor calls the difference real"
        " and the day-clustered interval does not. The paper's §7.1 will have"
        " to state which of the two it is quoting; nothing here chooses between"
        " them, and nothing needs to, because no adoption follows either way.",
        "",
        "**No adoption follows from this in either direction, whatever its"
        " sign.** Both forward legs validate on a month and a day-of-year"
        " range the training side never held: December after January to"
        " November, July after January to June. The ranking months do not."
        " January 2026 and July 2026 each have their twin in the training"
        " year, so the final fit interpolates on these columns where the legs"
        " extrapolate. The measurement answers a question the ranking months do"
        " not ask, and it is here for the paper's §7.1 as the validation"
        " design's limitation. A held-out-month split would answer the question"
        " the ranking months do ask, and it breaks the forward discipline every"
        " leg here keeps; it is named in the pre-registration below and is not"
        " run.",
        "",
    ]


def section_substitutions() -> list[str]:
    """What was changed when the pre-registration was copied in, and how much.

    Every count here is summed from ``SUBSTITUTIONS`` and every verdict is
    computed by ``substitution_audit``. There is no sentence in this section
    that a reader could find disagreeing with the table beside it, because
    both are the same object read twice.
    """
    audit = substitution_audit()
    rows = [
        [
            place.where,
            place.kind,
            str(place.tokens),
            place.widens
            or (
                "renders a name"
                if place.kind == WORK_BLOCK
                else "points at the repository's own file instead"
            ),
        ]
        for place in SUBSTITUTIONS
    ]
    lines = [
        "## What was changed when the pre-registration was copied in",
        "",
        "The pre-registration quoted below is a working note and is not"
        " published. Two things in it cannot be carried into a public"
        " repository: the numbered work blocks of the working program, which"
        " name sessions, and the file names of other working notes. Every place"
        " where the carried copy differs from the note is listed here, and the"
        " list is the record of it. **No criterion, schedule, grid value,"
        " reference, floor or reading rule differs**, which is the claim the"
        " checks under the table make good.",
        "",
        f"**{audit['places']} places, {audit['tokens']} substitutions.**"
        f" {audit['work_block_tokens']} work-block names across"
        f" {audit['work_block_places']} places, and"
        f" {audit['note_references']} note file names across"
        f" {audit['note_reference_places']} places. The counts are summed from"
        " the list below rather than written down, so a count and a list cannot"
        " drift apart. A place can hold more than one name: the floor cells of"
        " the legs table hold two and Experiment 1's outcome paragraph holds"
        " five.",
        "",
        *table(
            ["where", "what was replaced", "names", "what the change does"],
            rows,
            "--r-",
        ),
        "",
        "Only the published side of each place is recorded, in"
        " `scripts/preregistered.py`. Writing a work block's name into this"
        " repository in order to record that one was taken out would defeat the"
        " point of taking it out, so the note's own wording is not here and"
        " the note is the record of it.",
        "",
    ]
    checks = [
        [
            "every recorded place is in the carried text exactly once",
            "yes"
            if audit["markers_found_once"]
            else "**no: " + ", ".join(audit["stray_markers"]) + "**",
        ],
        [
            "the carried text carries no work-block name and no note file name",
            "yes" if audit["carried_is_clean"] else "**no**",
        ],
        [
            "the note, where it is on hand, rebuilds the carried text",
            audit["note_verdict"],
        ],
    ]
    if audit.get("note_checked"):
        checks.append(["the note's sha256", f"`{audit['note_sha256']}`"])
    lines += [
        "The checks behind that paragraph, computed each time this report is built:",
        "",
        *table(["check", "verdict"], checks, "--"),
        "",
    ]
    if not audit.get("note_checked"):
        lines += [
            "The note is not on this machine, so the byte-for-byte check was"
            " **not run** and no digest is printed. The first two checks read"
            " the published text alone and were made, which is what a clean"
            " checkout can establish: that the list is complete against this"
            " file and that nothing it was meant to remove is still here.",
            "",
        ]
    return lines


def section_fingerprint(summary: dict) -> list[str]:
    report_phase = summary["fingerprint"]["report"]
    rows = [
        [
            "the fitting run",
            f"{summary['fingerprint']['fitting']['files']:,}",
            "none exempt",
            "unchanged"
            if not summary["fingerprint"]["fitting"]["touched"]
            else "**"
            + ", ".join(summary["fingerprint"]["fitting"]["touched"])
            + " changed**",
        ],
        [
            "writing this report",
            f"{report_phase['files']:,}",
            f"`{Path(report_phase['exempt']).name}`",
            "checked after this file is written"
            if report_phase.get("pending")
            else (
                "unchanged"
                if not report_phase["touched"]
                else "**" + ", ".join(report_phase["touched"]) + " changed**"
            ),
        ],
    ]
    return [
        "## What this run was allowed to write",
        "",
        "`reports/` and `data/processed/residuals/` are fingerprinted by size"
        " and modification time before this script does anything and verified"
        " after it, as `scripts/fast_lane.py` verifies them. The fitting phase"
        " exempts nothing: it may write only under"
        " `data/processed/preregistered/`, which `.gitignore` already covers."
        " The report phase exempts this one file by name, so that exempting it"
        " cannot be done by leaving a whole directory out of the check. No"
        " stored residual was read for anything but the reproduction check and"
        " none was written.",
        "",
        *table(["phase", "files checked", "exempt", "verdict"], rows, "-r--"),
        "",
    ]


def build_report(summary: dict, generated: str, environment: dict) -> str:
    stopped = summary.get("stopped")
    head = [
        f"# The pre-registered experiments, {generated}",
        "",
        "Three questions whose criteria were fixed in writing on"
        " 20 September 2026, before any of the figures below existed. The"
        " criteria are quoted in full first and are constants in"
        " `scripts/preregistered.py`; every figure under them is computed by"
        " that script from the fits it made, and none is estimated. A dated"
        " entry in [`../docs/plan.md`](../docs/plan.md) carries the same"
        " criteria. That entry is older on disk than the earliest record this"
        " run wrote, which is the relation the commit script checks and refuses"
        " to run without, and it was committed first of the three, as"
        f" `{PLAN_ENTRY_COMMIT}`.",
        "",
        "**Nothing is decided here and nothing ships from it.** No parameter,"
        " no feature and no rule in `taxiout.models` is changed by this run;"
        " `SEED_AVERAGE_ADOPTED`, the feature registry and `BOOSTER_PARAMS`"
        " stand as they were, and no submission is written. Each experiment"
        " prints the verdict its own pre-registered rule computes, and what"
        " is done with that verdict is a later decision made under rules"
        " already fixed.",
        "",
        f"Generated by `scripts/preregistered.py` from the data under"
        f" `data/challenge/` on {generated}."
        f" {summary['fits']} fits of the submission rung"
        f" (`{SUBMISSION_KEY}`) on one feature frame, built in"
        f" {summary['frame']['build_seconds']:.0f} s, sequential, at the"
        f" ladder's eight threads. The fitting run took"
        f" {summary['run']['wall_seconds']:.0f} s of wall clock, of which"
        f" {summary['fit_seconds']:.0f} s were the fits themselves, summed from"
        " the per-fit walls in the table below; the rest is the scoring, the"
        " paired bootstraps and the reproduction joins. The run's own peak"
        f" resident size was {summary['run']['peak_rss_kib'] / 1048576:.1f} GiB."
        f" {environment['python']}, {environment['lightgbm']}.",
        "",
        "**Twenty fits were counted in advance and"
        f" {summary['fits']} were made.** The pre-registration counts twenty for"
        " the case where neither recency schedule stops early. Both stopped, so"
        " three fits that only a surviving schedule would have needed were never"
        " made: the strong schedule's July and December fits, and the mild"
        " schedule's December fit. The sequence below says where each stopped.",
        "",
        "**The sign convention, once, for the whole report.** Every step and"
        " every change below is the arm minus the thing it is read against, in"
        " seconds of RMSE, so **positive is worse and negative is better**"
        " throughout, including in the seed-averaging columns, where the arm is"
        " the average and the thing it is read against is the ladder's own"
        " seed. One column is the exception and is labelled for it: the rule of"
        " 19 September is stated in terms of a *gain*, so the gain column in"
        " Experiment 1's second table is the size of the improvement and"
        " positive there means better, as `baseline.md` states the rule.",
        "",
    ]
    if stopped:
        head += [
            f"**This run stopped at {stopped}, and no experiment's figure was"
            " read.** The checks below say what disagreed with its"
            " expectation. Nothing was adopted, nothing was changed, and the"
            " fits that were made are recorded so the disagreement can be"
            " looked at rather than repeated.",
            "",
        ]
    legs = table(
        ["leg", "train rows", "validate rows", "fit rows", "flagged training rows"],
        [
            [
                name,
                f"{got['train']:,}",
                f"{got['valid']:,}",
                f"{got['fit_rows']:,}",
                f"{got['flagged_train']:,}",
            ]
            for name, got in summary["frame"]["legs"].items()
        ],
        "-rrrr",
    )
    body = (
        head
        + [
            "## The legs",
            "",
            "The three fitted legs and the gap, which is not a fourth fit: it"
            " is the July leg's own fitted models scored on December's rows,"
            " through `Split.extra`, the mechanism the held-out-pair folds"
            " already use for a second evaluation set from one fit. December's"
            " flags there come from the July leg's record-quality model, as"
            " they must, since that is the model the fit was made without. The"
            " flagged column here counts training rows the record-quality term"
            " covers, which are the rows the booster is not fitted on; the fit"
            " table below counts flagged rows of the set being scored, which"
            " are the rows the ex-flagged figures leave out. They are different"
            " quantities and the headings say which.",
            "",
            *legs,
            "",
        ]
        + section_checks(summary)
    )
    if not stopped:
        body += (
            section_experiment_1(summary)
            + section_experiment_2(summary)
            + section_experiment_3(summary)
        )
    body += section_fits(summary) + section_fingerprint(summary)
    body += section_substitutions()
    body += [
        "## The pre-registration, as it was fixed on 20 September",
        "",
        "Quoted from the note it was written in, from its *Definitions, fixed"
        " here* heading onward, and carried as a constant in"
        " `scripts/preregistered.py`. The section above lists every place where"
        " the carried copy differs from the note, with the counts computed from"
        " that list; no criterion, schedule, grid value or reading rule differs."
        " Experiments 4, 5 and 6 are quoted with the rest because they are what"
        " was decided not to run, or to run later, at the same time and on the"
        " same evidence; nothing in this run touches them.",
        "",
        PREREGISTRATION,
        "",
        "Distance matrices derived from OpenStreetMap: data (c)"
        " OpenStreetMap contributors, ODbL 1.0. Movement and flight data"
        " supplied by EUROCONTROL PRC and the OpenSky Network under the terms"
        " of the PRC Data Challenge 2026.",
        "",
    ]
    return "\n".join(body)


def environment() -> dict:
    try:
        import lightgbm

        lgbm = f"LightGBM {lightgbm.__version__}"
    except ImportError:
        lgbm = "no LightGBM"
    return {
        "python": f"Python {sys.version.split()[0]}",
        "lightgbm": lgbm,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def write_report(
    summary: dict, generated: str, stands: set[str], movements: set[str]
) -> Path:
    out = report_path(generated)
    before = fingerprint(exempt=out)
    # The verdict is about the write that has not happened yet, so the first
    # build prints that it is pending and the second prints the answer. Both
    # are inside the same fingerprint, so neither can be the write that got
    # away with something.
    summary.setdefault("fingerprint", {})["report"] = {
        "files": len(before),
        "exempt": str(out),
        "touched": [],
        "pending": True,
    }

    def once() -> None:
        text = build_report(summary, generated, environment())
        if ATTRIBUTION not in text:
            raise RuntimeError("the report lost its OpenStreetMap attribution")
        check_no_identifiers(text, stands, movements)
        out.write_text(text, encoding="utf-8")

    # The fingerprint verdict has to appear in the file it is a verdict about,
    # so the report is written once to be checked and once with the check's
    # own answer in it. Both writes are inside the same fingerprint, and the
    # second is verified again, so neither can be the one that wrote elsewhere.
    once()
    moved = touched(before, fingerprint(exempt=out))
    summary["fingerprint"]["report"] = {
        "files": len(before),
        "exempt": str(out),
        "touched": [str(Path(m).relative_to(REPO)) for m in moved],
        "pending": False,
    }
    once()
    moved_again = touched(before, fingerprint(exempt=out))
    if moved or moved_again:
        raise SystemExit(
            "writing the report touched a file it must not: "
            + ", ".join(sorted(set(moved) | set(moved_again)))
        )
    print(f"wrote {out.relative_to(REPO)}")
    return out


# --------------------------------------------------------------------------
# main


def summary_path() -> Path:
    return WORK / "run.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build the frame, print each leg's row counts, fit nothing",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="fit nothing; regenerate the report from the records already on disk",
    )
    ap.add_argument(
        "--note",
        type=Path,
        default=None,
        metavar="PATH",
        help="the working note of 20 September, to check the published text against"
        " byte for byte (default: ${PREREGISTRATION_NOTES}/" + NOTE_NAME + ", if set)",
    )
    ap.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="the date in the report's file name (default: today, UTC)",
    )
    args = ap.parse_args()
    if args.note is not None:
        global NOTE
        NOTE = args.note
    generated = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if args.report:
        path = summary_path()
        if not path.exists():
            print(f"no run recorded under {WORK.relative_to(REPO)}", file=sys.stderr)
            return 1
        summary = json.loads(path.read_text(encoding="utf-8"))
        records = load_records()
        summary["records"] = records
        summary["fits"] = len(records)
        summary["fit_seconds"] = float(sum(r["wall_s"] for r in records.values()))
        derived = Checks()
        summary.update(derive(records, derived))
        summary["checks"] = derived.rows
        write_report(
            summary, generated, set(summary.get("stands", [])), movements_on_disk()
        )
        summary_path().write_text(
            json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
        )
        return 0

    before = fingerprint()
    print(
        f"writing only to {WORK.relative_to(REPO)}; "
        + " and ".join(str(p.relative_to(REPO)) for p in FORBIDDEN)
        + f" are checked untouched ({len(before)} files)"
    )

    frame = Frame()
    legs = {}
    for name in ("december", "july", "october"):
        prep = frame.prepare(name)
        legs[name] = {
            "train": int(prep["leg"].train.sum()),
            "valid": int(prep["leg"].valid.sum()),
            "fit_rows": int(prep["rows"].sum()),
            "flagged_train": int((prep["leg"].train & prep["flagged"]).sum()),
        }
    july = frame.prepare("july")
    legs["the gap (July's fit on December)"] = {
        "train": int(july["leg"].train.sum()),
        "valid": int(july["leg"].extra["gap"].sum()),
        "fit_rows": int(july["rows"].sum()),
        "flagged_train": int((july["leg"].train & july["flagged"]).sum()),
    }
    if args.dry_run:
        print("\nthe legs, and nothing fitted:")
        for name, got in legs.items():
            print(
                f"  {name:>34s}: {got['train']:>9,} train  "
                f"{got['valid']:>9,} validate  {got['fit_rows']:>9,} fit rows"
            )
        after = fingerprint()
        moved = touched(before, after)
        print("\n" + ("nothing was touched" if not moved else f"TOUCHED: {moved}"))
        return 0 if not moved else 1

    t0 = time.time()
    summary = run(frame)
    summary["run"] = {
        "wall_seconds": time.time() - t0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    summary["frame"] = {
        "build_seconds": frame.build_seconds,
        "rows": int(len(frame.features)),
        "legs": legs,
    }
    summary.setdefault("fits", len(summary["records"]))

    after = fingerprint()
    moved = touched(before, after)
    summary["fingerprint"] = {
        "fitting": {
            "files": len(before),
            "touched": [str(Path(m).relative_to(REPO)) for m in moved],
        }
    }
    if moved:
        print(
            "this run wrote where it must not, which makes its numbers a report's:\n  "
            + "\n  ".join(str(Path(m).relative_to(REPO)) for m in moved),
            file=sys.stderr,
        )
        return 1
    print(f"\nno report and no stored residual was touched ({len(after)} files checked)")

    # Every identifier the report could have leaked, from the rows it read.
    stands = {
        str(v).strip() for v in frame.dep["STAND_mvt"].dropna().astype(str).unique()
    }
    summary["stands"] = sorted(stands)
    WORK.mkdir(parents=True, exist_ok=True)
    summary_path().write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )

    write_report(summary, generated, stands, movements_on_disk())
    summary_path().write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )

    failed = [r for r in summary["checks"] if not r["ok"]]
    if failed:
        print(
            f"\n{len(failed)} check(s) disagreed with the expectation; "
            "the report says which.",
            file=sys.stderr,
        )
        return 1
    print(f"\n{summary['fits']} fits, every check as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
