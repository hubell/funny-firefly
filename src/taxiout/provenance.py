# SPDX-License-Identifier: GPL-3.0-only
"""The provenance register: where every number in this repository comes from.

The standing rule is that every number reaching a report, a note or the paper
is computed by the run that prints it, quoted from a named source with that
source beside it, or marked an estimate with an instrument and an owner. There
is no fourth kind. Until this module the repository obeyed that by convention,
and a convention does not fail a build.

**What this is for, in one case.** ``HEADROOM_TRIGGER_S`` was chosen with a
stated reason and no recorded basis, and it turned out to be an absolute line
on a quantity that ``docs/review_headroom.md`` §7 had already characterised as
a lower bound a refit exceeds. It went from missing to firing when the
booster's capacity grew 3.2 times, and nothing in the tree connected the
threshold to the review that had already undermined it. **A register covering
quoted measurements and not chosen thresholds would not have caught that**, so
this one covers both, and ``chosen`` is the class that must carry a basis.

Four states, and ``scripts/check_provenance.py`` proves there is no fifth:

1. **Generated** by the run that prints it. Needs no entry here.
2. **Checked** against a named source, so drift fails the gate.
3. **Registered** below with a one-line basis: quoted measurements, chosen
   thresholds, frozen records.
4. **Exempt**, on an explicit list with a reason: mechanism and definition.

Exhaustive here means *nothing is uncategorised*, not that everything carries a
paragraph. Most entries below are one word. The property the repository gains
is that adding a constant without categorising it fails the build.

**Nothing here is a data file.** A register in YAML would need a reader, a
reader is a dependency, and a dependency to answer "where did this number come
from" is the wrong trade. ``docs/provenance.md`` is generated from these lists
with every count summed from them rather than written down, which is the
arrangement ``scripts/preregistered.py`` uses for ``SUBSTITUTIONS`` and for the
same reason: two counts of one quantity cannot then disagree.

**What the register does not reach**, stated so its silence is not read as a
pass. ``SCOPE`` below carries this in full and the gate prints it.
"""

from __future__ import annotations

from typing import NamedTuple

# --------------------------------------------------------------------------
# the four classes

# Definitional, or a unit conversion. Nobody chose it and nothing could have
# been chosen differently: ``HOUR = 3600`` is not a threshold.
MECHANISM = "mechanism"

# Somebody picked it. The basis is REQUIRED and must say what it was picked
# against, because that is the half of the register that failed.
CHOSEN = "chosen"

# Computed from the data and frozen into the source. The basis names the file
# and the quantity, so the figure can be recomputed.
DERIVED = "derived"

# A measurement from a named run or document, not recomputable here. The basis
# names the source, its date, and whether it is regenerable today.
QUOTED = "quoted"

CLASSES: tuple[str, ...] = (MECHANISM, CHOSEN, DERIVED, QUOTED)

# What a basis says when the code records none. Written as a marker rather than
# left empty, because an empty basis and an unrecorded one are different facts
# and only the second is a finding. ``PROVENANCE_PENDING`` carries the class and
# ``check_provenance.py`` counts them from here rather than from a written-down
# number, so the count cannot drift from the register.
NO_BASIS = "NOT RECORDED"

# What a basis says when the record holds the method and not the value: the
# design is written down and prints this number, or the same number carries a
# recorded reason in a neighbouring use, and **no reason for this value rather
# than a neighbouring one is recorded anywhere**. A different fact from
# ``NO_BASIS`` and counted separately, because "nothing is written down" and
# "the method is written down and the value is not argued for" are different
# states, and collapsing them would either overstate the hole or hide it.
VALUE_NOT_JUSTIFIED = "VALUE NOT JUSTIFIED"

# The half of the two fetchers' bases that is the same for all six of their
# constants, and the half that is checkable. A pacing or robustness value is a
# choice, so these are ``chosen`` and not ``mechanism``; what makes the choice
# unimportant is that nothing measured depends on it, and that is asserted
# where it can be proved rather than promised.
FETCH_BASIS = (
    " No measurement lies behind the value and no reported figure depends on"
    " it, only on whether the fetch succeeded: the script writes nothing"
    " outside `data/raw/`, and check 13 proves that no generator of a report"
    " imports it. What is not recorded is why this value rather than a"
    " neighbouring one, and nothing rests on the answer."
)


# This module. Its own aggregate assignments -- ``CONSTANTS`` and the two lists
# spliced into it -- carry every registered literal in the repository between
# them, so an entry for them would be a copy of the whole register inside
# itself, and it would drift the moment any other entry changed. **The register
# cannot be an entry in itself**, which is a different exclusion from the
# authored artifact's and is stated separately for that reason. Excluded by name
# so the exclusion is one module and not a pattern, and recorded in ``SCOPE``.
REGISTER_MODULE = "src/taxiout/provenance.py"


class Constant(NamedTuple):
    """One module-level numeric assignment, classified.

    ``literals`` is every numeric literal in the assignment's value, in source
    order, as ``scripts/check_provenance.py``'s ``module_constants`` collects
    them. It is the whole value's numeric content and not the evaluated object:
    that makes one instrument serve a scalar, a tuple, a dict and a
    comprehension alike, it needs no import and so no import side effect, and
    it catches a literal edited inside a comprehension that an evaluated value
    would hide.
    """

    name: str
    module: str
    literals: tuple[float, ...]
    cls: str
    basis: str = ""


def _mod(module: str, *rows: tuple) -> tuple[Constant, ...]:
    """The entries for one module, so the path is written once."""
    return tuple(
        Constant(name, module, lits, cls, basis) for name, lits, cls, basis in rows
    )


# Every script resolves the repository root the same way, and the ``1`` the
# scanner finds in each is a subscript index into ``Path.parents`` rather than a
# quantity. Registered as one decision applied to a named list of modules, which
# is a person's judgement stated once and not a class the scanner guessed.
_REPO_ROOT_MODULES: tuple[str, ...] = (
    "scripts/build_taxi_graph.py",
    "scripts/check_provenance.py",
    "scripts/check_determinism.py",
    "scripts/check_licence_hygiene.py",
    "scripts/check_penalty_sensitivity.py",
    "scripts/check_runways_ourairports.py",
    "scripts/fast_lane.py",
    "scripts/fetch_metar.py",
    "scripts/fetch_osm.py",
    "scripts/fetch_ourairports.py",
    "scripts/make_submission.py",
    "scripts/pooled_tails.py",
    "scripts/preregistered.py",
    "scripts/report_congestion.py",
    "scripts/report_coverage.py",
    "scripts/report_geometry_gaps.py",
    "scripts/report_headroom.py",
    "scripts/report_occupancy.py",
    "scripts/report_residual.py",
    "scripts/report_submission.py",
    "scripts/report_synthetic_airport.py",
    "scripts/report_tails.py",
    "scripts/report_transfer.py",
    "scripts/review_challenge_data.py",
    "scripts/run_baselines.py",
    "scripts/sweep_params.py",
)

_REPO_ROOT: tuple[Constant, ...] = tuple(
    Constant(
        "REPO",
        m,
        (1,),
        MECHANISM,
        "a subscript index into `Path(__file__).resolve().parents`, not a quantity",
    )
    for m in _REPO_ROOT_MODULES
)


# ---- scripts/, the generators and the gates ---------------------------
#
# Spliced in above rather than written inline so that the library half reads as
# the library half. Same records, same four classes.
_SCRIPTS: tuple[Constant, ...] = (
    *_mod(
        "scripts/check_determinism.py",
        (
            "SAME_TREES_S",
            (1e-06,),
            MECHANISM,
            "a floating-point equality tolerance on a per-flight prediction,"
            " below any difference a report prints",
        ),
    ),
    *_mod(
        "scripts/check_licence_hygiene.py",
        (
            "HISTORY_DEPTH",
            (40,),
            CHOSEN,
            "line 204: deep enough to cover a batch of commits made by one"
            " script, shallow enough to stay fast on every run",
        ),
        (
            "LEAK_THRESHOLD",
            (3,),
            CHOSEN,
            "line 208, and the module docstring: one match is a coincidence and"
            " a list is a leak, so a file is flagged only at three or more",
        ),
    ),
    *_mod(
        "scripts/check_runways_ourairports.py",
        (
            "LENGTH_TOL",
            (0.1,),
            CHOSEN,
            "line 50: OSM ways are traced to the marked ends and OurAirports"
            " publishes the declared distance, so a few percent is normal and"
            " ten is not",
        ),
        (
            "HEADING_TOL_DEG",
            (15.0,),
            CHOSEN,
            "line 53: magnetic variation across these airports reaches about"
            " seven degrees and designators quantise to ten, so fifteen is"
            " generous without being useless",
        ),
    ),
    *_mod(
        "scripts/compare_runs.py",
        (
            "SAME_TREES_S",
            (1e-06,),
            MECHANISM,
            "the same floating-point equality tolerance as `check_determinism.py`",
        ),
    ),
    *_mod(
        "scripts/fast_lane.py",
        (
            "TREES_DIFFER",
            (1e-06,),
            MECHANISM,
            "the same floating-point equality tolerance as `check_determinism.py`",
        ),
        ("REFERENCE", (1,), MECHANISM, "an index into a run list, not a quantity"),
    ),
    *_mod(
        "scripts/fetch_metar.py",
        (
            "PAUSE_S",
            (2.0,),
            CHOSEN,
            "politeness toward a public archive: the delay between requests"
            " to the Iowa State Mesonet, paced so a year of stations is not"
            " fetched as fast as the client can ask." + FETCH_BASIS,
        ),
        (
            "RETRIES",
            (4,),
            CHOSEN,
            "robustness against a public archive's transient failures: how many"
            " times one request is retried before the station-period is"
            " reported as unfetched." + FETCH_BASIS,
        ),
        (
            "TIMEOUT_S",
            (120,),
            CHOSEN,
            "robustness: the socket timeout on one archive request, above the"
            " time a year of one station takes to serve and below a hang." + FETCH_BASIS,
        ),
    ),
    *_mod(
        "scripts/fetch_ourairports.py",
        (
            "PAUSE_S",
            (2.0,),
            CHOSEN,
            "politeness toward a public dataset's host, as in"
            " `fetch_metar.py`." + FETCH_BASIS,
        ),
        (
            "RETRIES",
            (3,),
            CHOSEN,
            "robustness: how many times the one table request is retried." + FETCH_BASIS,
        ),
        (
            "TIMEOUT_S",
            (120,),
            CHOSEN,
            "robustness: the socket timeout on that request." + FETCH_BASIS,
        ),
    ),
    *_mod(
        "scripts/pooled_tails.py",
        (
            "SE_MULTIPLE",
            (2.0,),
            MECHANISM,
            "line 99 states it: two standard errors of a proportion is"
            " arithmetic and not a chosen threshold, being what the premise"
            " test's question costs",
        ),
        (
            "ILLUSTRATIVE_LAGS",
            (5, 14),
            CHOSEN,
            "line 116: the only chosen numbers in that script, labelled as"
            " chosen wherever they are printed, and nothing is concluded from"
            " them because the table beside the worked one replaces them with"
            " the case airport's own rows",
        ),
    ),
    *_mod(
        "scripts/preregistered.py",
        (
            "REF_DECEMBER_ALL",
            (217.6,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 32, the digest"
            " table, which records it as read from `reports/baseline.md`"
            " *The ladder* as that file then stood. **The citation was moved"
            " here because the original no longer holds it**: the wider sweep"
            " of 21 September regenerated `baseline.md` and this figure is now"
            " nowhere in it. Quoted in the pre-registration of"
            " 20 September 2026. Regenerable only by the ladder",
        ),
        (
            "REF_JULY_THREE",
            (237.8,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 35, the digest table,"
            " which records it as read from `reports/baseline.md` before the"
            " sweep of 21 September; same pre-registration, and the figure is"
            " no longer in `baseline.md`",
        ),
        (
            "REF_RANKING_MIX",
            (218.6,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 33, the digest table,"
            " as for `REF_DECEMBER_ALL`; no longer in `baseline.md`",
        ),
        (
            "REF_JULY_ALL",
            (325.8,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 34, the digest table,"
            " as for `REF_DECEMBER_ALL`; no longer in `baseline.md`",
        ),
        (
            "FLOOR_DECEMBER_ALL",
            (0.34,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 39, read from"
            " `reports/baseline.md` *The noise floor* as that file stood on"
            " 20 September. **On 21 September, under the sweep's cell, that"
            " report printed 0.26 for this leg, and on 24 September, under the"
            " cell adopted the day before, 0.23**: every change of parameters"
            " moves every floor, and the constant is a dated quotation rather"
            " than a reading of the current file",
        ),
        (
            "FLOOR_JULY_ALL",
            (0.23,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 49, read from"
            " `reports/baseline.md` *The noise floor* as that file stood on"
            " 20 September. **On 21 September, under the sweep's cell, that"
            " report printed 0.61 for this leg, and the verdict was not in"
            " question:** July gained 0.63 s against that 0.61 and cleared it,"
            " which is why `reports/baseline.md` computed *Adopted* then. On"
            " 24 September, under the cell adopted the day before, it printed"
            " 0.54, against a July gain of 1.46 s. Each figure is right about"
            " its own moment, because the rule is re-read under the parameters"
            " in force, and reading the 20 September gain against a later floor"
            " would mix two moments. What was wrong here was the citation and"
            " not the figure",
        ),
        (
            "FLOOR_JULY_THREE",
            (1.02,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 50, read from"
            " `reports/baseline.md` *The noise floor* as that file stood on"
            " 20 September. **On 21 September, under the sweep's cell, that"
            " report printed 0.63 for this leg, and on 24 September, under the"
            " cell adopted the day before, 1.02**, the figure quoted here from"
            " 20 September: two moments that happen to agree, not one reading",
        ),
        (
            "JULY_SINGLES_POOLED",
            (325.81, 326.23),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 36, the digest table,"
            " read from `reports/baseline.md` before the sweep of"
            " 21 September: the span the five July single-seed fits must"
            " reproduce before any experiment is read",
        ),
        (
            "JULY_SINGLES_THREE",
            (237.02, 238.83),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 37, the digest table,"
            " the same reproduction gate on the three airports, read from"
            " `reports/baseline.md` before the sweep of 21 September",
        ),
        (
            "JULY_AVERAGE_POOLED",
            (325.29,),
            QUOTED,
            "`reports/preregistered-2026-09-20.md` line 38, the digest table,"
            " the five-seed average the reproduction gate requires, read from"
            " `reports/baseline.md` before the sweep of 21 September",
        ),
        (
            "HALF_LIVES",
            (3, 12),
            CHOSEN,
            "the two recency schedules fixed in the pre-registration of"
            " 20 September before any figure under them existed",
        ),
        (
            "SEEDS",
            (5,),
            CHOSEN,
            "`taxiout.models.BAGGING_SEEDS` reused, so the experiment measures"
            " the estimator the submission would ship",
        ),
        (
            "INTERVAL_BELOW_S",
            (1.0,),
            CHOSEN,
            "the pre-registered rule: any step inside a second prints its"
            " day-clustered interval, so a small step is never reported as a"
            " bare number",
        ),
        (
            "RULE_LEGS",
            (217.6, 217.11, 217.59, 0.48, 325.8, 325.29, 325.81, 0.52),
            QUOTED,
            "the figures the seed-averaging rule is read on, quoted from"
            " `reports/baseline.md` of 19 September 2026",
        ),
        (
            "SUBSTITUTIONS",
            (2, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            MECHANISM,
            "token counts per substituted place, and `substitution_audit` checks"
            " each against the carried text rather than trusting it",
        ),
        ("OCTOBER_TRAIN", (1, 10), MECHANISM, "a month range, not a quantity"),
    ),
    *_mod(
        "scripts/report_geometry_gaps.py",
        (
            "MIN_PAIR_ROWS",
            (30,),
            QUOTED,
            "line 90's own comment: the PRU reference's own floor for a pair",
        ),
        (
            "MIN_CLASS_ROWS",
            (20,),
            CHOSEN,
            "line 91's comment states the rule: per (pair, wake class) for the"
            " paired gap",
        ),
        (
            "MIN_DESIGNATOR_PAIRS",
            (5,),
            CHOSEN,
            f"{NO_BASIS}: the floor on pairs per runway designator before the"
            " report names that designator. The constants either side carry"
            " inline comments and this one carries none",
        ),
        (
            "USABLE_ENTRY_SHARE",
            (0.6,),
            CHOSEN,
            "line 93's comment: a junction in the first 60% of the runway is a"
            " usable entry",
        ),
        (
            "MIN_ENTRY_OFFSET_M",
            (50.0,),
            CHOSEN,
            "line 94's comment: nearer than this to the threshold is the threshold",
        ),
        ("HOUR", (3600.0,), MECHANISM, "seconds in an hour"),
        (
            "MIN_BETWEEN_FRACTION",
            (0.05,),
            CHOSEN,
            "lines 97 to 99: `taxiout.encoding`'s floor reused, and it says so",
        ),
        ("RANKING_MONTHS", (1, 7), MECHANISM, "the ranking set's two months"),
        (
            "FORWARD_SPLITS",
            (11, 12, 6, 7),
            MECHANISM,
            "the two forward splits as `taxiout.validation` fixes them",
        ),
        (
            "THIN_BUCKETS",
            (3,),
            MECHANISM,
            "a slice of `FREQUENCY_BUCKETS`; line 106's inline comment reads"
            " `seen a hundred times or fewer`",
        ),
    ),
    *_mod(
        "scripts/report_headroom.py",
        (
            "HEADROOM_BOOSTER_PARAMS",
            (1.0, 1.0, 0, 32, 10.0, 20260902, 8, 1),
            CHOSEN,
            "lines 104 to 126: the smallest value on each of the four axes of"
            " `sweep_params.FIRST_SWEEP`, the grid live when"
            " `docs/review_headroom.md` §2 was written, read out of that"
            " constant. A reconstruction of what line 11 calls a small booster"
            " and not a recovery of it, and the report says so and says the rule"
            " is not pre-registered",
        ),
        (
            "HEADROOM_FOLDS",
            (5,),
            MECHANISM,
            "`docs/review_headroom.md` line 11 fixes five folds grouped by day",
        ),
        (
            "HEADROOM_FOLD_SEED",
            (20260902,),
            MECHANISM,
            "a fixed seed, the repository's own. Any value serves; it has to be fixed",
        ),
        ("LEGS", (0, 1), MECHANISM, "indices into `EXPECTED_HEADROOM_S`'s pair"),
        (
            "SECTION_2_DECEMBER_THREE",
            (5.0,),
            QUOTED,
            "`docs/review_headroom.md` §2's table, the one column"
            " `EXPECTED_HEADROOM_S` does not carry, 8 September 2026. Made by a"
            " script held outside this repository and not regenerable as it"
            " stood; `make headroom` recomputes the quantity at a named capacity",
        ),
        (
            "SECTION_7_READINGS",
            (18.3, 10.8, 13.8, 6.1),
            QUOTED,
            "`docs/review_headroom.md` §7's two chain-level readings, 8 September 2026",
        ),
        (
            "LADDER_STEP_QUOTED_S",
            (15.98,),
            QUOTED,
            "`reports/baseline.md` line 978, as the ladder of 23 September"
            " regenerated it under the adopted cell; it read 17.60 under the"
            " cell before that. Cross-checked in the generator"
            " against the same figure computed from the stored residuals, which"
            " refuses to write the report if the two disagree by 0.005 s",
        ),
        (
            "CAPACITY_CHECK",
            (100, 0.1, 31, 20),
            CHOSEN,
            "LightGBM's own defaults on the four capacity axes, as the third row"
            " of the report's capacity table; the second row imports"
            " `BOOSTER_PARAMS` rather than writing it down",
        ),
    ),
    *_mod(
        "scripts/report_occupancy.py",
        (
            "GAIN_BAND_EDGES_MIN",
            (0, 20, 30, 45, 60, 120),
            CHOSEN,
            "line 103: coarser than `BAND_EDGES_MIN` because a sum of squares"
            " over a five-minute band of a 1% population is noise",
        ),
        (
            "LONG_GAP_MIN",
            (60.0,),
            CHOSEN,
            "line 111: past this the gap bounds nothing, the stand having"
            " possibly been re-occupied and released, or the aircraft towed",
        ),
        (
            "BAND_EDGES_MIN",
            (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120),
            CHOSEN,
            "line 138: five-minute bands of the gap, with the two wide ones the"
            " review used at the top, because past an hour the bound is not a"
            " bound and the bands only need to show that",
        ),
        (
            "PRESSURE_MIN",
            (20,),
            CHOSEN,
            "line 159: below twenty minutes the bound holds essentially always,"
            " so the share under it summarises stand pressure",
        ),
        (
            "FLOOR_CAPS_MIN",
            (20, 30, 45, 60),
            CHOSEN,
            "line 163: the caps the hard floor is tried at, on the reasoning"
            " that the bound is sound at short gaps and collapses at long ones",
        ),
    ),
    *_mod(
        "scripts/report_residual.py",
        (
            "TAIL_CUTS_S",
            (600.0, 1200.0),
            CHOSEN,
            "line 128 names them as where the tail is cut, in seconds of"
            " absolute residual",
        ),
        (
            "REOCCUPIED_BAND_MIN",
            (20.0, 45.0),
            CHOSEN,
            "line 131: the band where the stand-occupancy bound is the"
            " mechanism, the same band `reports/baseline.md` decomposes on",
        ),
        (
            "LAG_EDGES_MIN",
            (0, 15, 30, 45, 60, 120),
            CHOSEN,
            "line 135: plan-lag bands in minutes, the last open and nulls their own row",
        ),
        (
            "LARGEST_ROWS",
            (5,),
            CHOSEN,
            "line 139: how many of the largest rows to print per leg",
        ),
        (
            "WORST_QUANTILE",
            (0.99,),
            CHOSEN,
            "line 142: the share of absolute residual that counts as the worst"
            " one percent",
        ),
        (
            "PEAK_HOURS",
            (6,),
            CHOSEN,
            VALUE_NOT_JUSTIFIED + ": how many take-off hours the section"
            " names. The method is recorded in the comment above it, lines 158"
            " to 160: both counts are read off the leg's own squared error"
            " rather than fixed in advance, and `reports/residual.md` prints"
            " the hours it found. That is how the figure is used; no reason"
            " for six rather than five or eight is recorded",
        ),
        (
            "NAMED_PREFIXES",
            (10,),
            CHOSEN,
            VALUE_NOT_JUSTIFIED + ": how many callsign prefixes are named"
            " before pooling the rest. The same comment at lines 158 to 160"
            " covers it on the same terms as `PEAK_HOURS` above",
        ),
        (
            "MIN_INTERVAL_DAYS",
            (5,),
            CHOSEN,
            "line 169: below this many distinct take-off days a day-clustered"
            " resample has too few clusters to say anything, and reports no"
            " variation at all, which prints as a narrow interval and means the"
            " opposite",
        ),
        (
            "LAG_LABELS",
            (1, 1),
            MECHANISM,
            "indices into `LAG_EDGES_MIN` while building the band labels",
        ),
    ),
    *_mod(
        "scripts/report_submission.py",
        ("DAY_S", (86400,), MECHANISM, "seconds in a day"),
        ("MEANS", (3600,), MECHANISM, "seconds in an hour, in a unit conversion"),
    ),
    *_mod(
        "scripts/report_tails.py",
        ("HOUR", (3600.0,), MECHANISM, "seconds in an hour"),
        ("DAY", (86400.0,), MECHANISM, "seconds in a day"),
        (
            "STAND_IN_MIN_PAIR",
            (5,),
            CHOSEN,
            f"{NO_BASIS}: the minimum rows a (stand, runway) pair needs"
            " before the stand-in physical model reads its mean. Nothing in"
            " the script records why five, though `taxiout.tails.MIN_COUNT` is"
            " also five and does carry a reason",
        ),
        (
            "PREFIX_ROWS",
            (15,),
            CHOSEN,
            "line 105: enough to show the shape of the claim, with the sentence"
            " under each table computed over every row",
        ),
    ),
    *_mod(
        "scripts/review_challenge_data.py",
        ("HOUR", (3600,), MECHANISM, "seconds in an hour"),
        ("DAY", (86400,), MECHANISM, "seconds in a day"),
        (
            "FLT_SAME_RATE",
            (0.0002,),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: how far apart, as a share, the `_flt`"
            " columns' null rates may sit on every side of the data before"
            " `reports/data_review.md`'s *Null rates* stops calling them the"
            " same rate. It was written inside the function as a bare"
            " threshold with the prose restating it as a separate figure, two"
            " copies of one number; the prose is now formatted from it. The"
            " table beside the sentence prints the rates as percentages to two"
            " decimals, so the threshold is two steps of its last digit, and"
            " no reason for two rather than one or five is recorded",
        ),
        (
            "DISRUPTION_MIN_ROWS",
            (20,),
            CHOSEN,
            "line 565 states the rule both sides of the data are read at, so the"
            " two tables can be set against each other",
        ),
        (
            "DISRUPTION_RATE_MULTIPLE",
            (10,),
            CHOSEN,
            "line 565, covering this too: a share this many times the airport's"
            " own training-year rate before the day is called out",
        ),
    ),
    *_mod(
        "scripts/run_baselines.py",
        (
            "FAMILYWISE_ALPHA",
            (0.05,),
            CHOSEN,
            "lines 1234 to 1245: three shots at a 5% test is a 14% chance of one"
            " clearing zero by luck, and Bonferroni is the crude correction"
            " because it assumes nothing about how the arms' errors are related",
        ),
        (
            "K_SENSITIVITY_SCALES",
            (0.5, 2.0),
            CHOSEN,
            f"{NO_BASIS}: the halving and doubling the encoding's shrinkage"
            " is re-read at in the k-sensitivity diagnostic. The comment above"
            " belongs to `FAMILYWISE_ALPHA` and covers Bonferroni, not this",
        ),
        (
            "RANKING_MIX",
            (1115.0, 2946.0, 17153.0, 194662.0),
            DERIVED,
            "the ranking set's own pair-frequency mix, as ranking rows per"
            " bucket, counted in `reports/data_review.md` and summing to the"
            " 215,876 ranking departures. Recomputable from the challenge data",
        ),
        (
            "EXPECTED_HEADROOM_S",
            (3.9, 10.3, 2.1, 2.8, 1.9, 1.7, 1.0, 1.0),
            QUOTED,
            "`docs/review_headroom.md` §2, 8 September 2026, made by a script"
            " held outside this repository. NOT regenerable as it stood;"
            " `scripts/report_headroom.py` recomputes the quantity at a named"
            " capacity and prints it beside this constant rather than in place"
            " of it. On 22 September, on the residuals of 21 September, it came"
            " within 0.05 s of the July-three figure; on 25 September, on those"
            " of 23 September, it read +10.00, 0.30 s under it",
        ),
        (
            "HEADROOM_TRIGGER_S",
            (5.0,),
            CHOSEN,
            "a line on a DIAGNOSTIC and not a leakage criterion. Lines 2581 to"
            " 2586 give the reason for an absolute figure and"
            " `docs/prior_art.md` records that absolute thresholds transfer"
            " badly between targets. An absolute line on a quantity that scales"
            " with the measured side is why this went from missing to firing"
            " when the booster's capacity grew 3.2 times, with the block"
            " untouched: the readings are 5.56, 4.971, 5.37, 4.345 and 5.681 s"
            " across five regenerations. `docs/review_headroom.md` §7 of 8 September,"
            " before it fired, records that a residual fit bounds a refit from"
            " below and the gap is interaction value rather than evidence of"
            " anything wrong",
        ),
        (
            "HEADROOM_NEAR_MISS_S",
            (1.0,),
            CHOSEN,
            "lines 2612 to 2616: a threshold test reported only as pass or fail"
            " hides the case that missed by a hundredth, and this report already"
            " has a subsection about a rule that turned on nine thousandths",
        ),
        (
            "HEADROOM_READINGS",
            (15.86, 5.56, 2, 15.27, 4.971, 3, 15.67, 5.37, 2, 14.64, 4.345, 3),
            QUOTED,
            "the overshoots this test read on earlier regenerations, quoted from"
            " the reports committed at `ad1eecf`, `ff59892` and `067c4af` and,"
            " for the fourth, from `reports/migration-2026-09-10.md` lines 536"
            " to 541, which record the regeneration of 19 September at a step"
            " of +14.64 s and an overshoot of 4.345 s, which missed. That"
            " fourth reading was restored on 23 September by the run that"
            " regenerated `reports/baseline.md`; before it the published"
            " history read fired, missed, fired where the record reads fired,"
            " missed, fired, missed",
        ),
        (
            "HEADROOM_REFERENCE_READING",
            (2,),
            MECHANISM,
            "a position in `HEADROOM_READINGS`, not a quantity: which of its"
            " readings was taken under the reference configuration, whose seed"
            " floor is the one `HEADROOM_FLOOR_BEFORE_S` carries. Named rather"
            " than taken as the last reading because the readings are in date"
            " order and the fourth was added after that floor was measured, so"
            " `[-1]` would have attributed one configuration's floor to a run"
            " fitted under another",
        ),
        (
            "HEADROOM_UNCOMMITTED_S",
            (15.43,),
            QUOTED,
            "lines 2681 to 2683: the first run on the reference machine,"
            " 10 September, under the booster's earlier setting and never"
            " committed. The migration record gives its gain and nothing finer",
        ),
        (
            "HEADROOM_FLOOR_BEFORE_S",
            (1.28,),
            QUOTED,
            "lines 2685 to 2688: the seed floor on the July three under the"
            " reference configuration, from the migration record's noise-floor"
            " table of 11 September",
        ),
        (
            "HEADROOM_RECOMPUTED",
            (10.0,),
            QUOTED,
            "lines 2690 to 2696: `reports/headroom.md`, *The recomputed"
            " headroom, beside the quoted figures*, the stand-occupancy row on"
            " July, the three, as `make headroom` regenerated it on"
            " 25 September 2026 on the residuals the ladder wrote on"
            " 23 September. A dated quotation with its date in the constant;"
            " check 11 fails the sentence that quotes it if that report stops"
            " carrying the figure",
        ),
        (
            "EXPECTED_DID_S",
            (1.0,),
            CHOSEN,
            "lines 2792 to 2798: none of the four blocks is a property of the"
            " (airport, stand, runway) pair, so each should be within about a"
            " second of zero on the held-out-pair contrast. Not a gate, and the"
            " section says so",
        ),
        (
            "PRINTED_PRECISION_S",
            (0.1,),
            CHOSEN,
            "lines 2801 to 2813: two rungs printing the same number are a tie,"
            " and a rule that ranks them is ranking the random-number generator."
            " The comment also records that adopting it there was decided after"
            " seeing the number it turns on, and that `_tolerance_audit` prints"
            " every other margin so a reader can check it moves one step only",
        ),
        (
            "GAIN_BAND_EDGES_MIN",
            (0, 20, 30, 45, 60, 120),
            CHOSEN,
            "lines 2872 to 2876: wider than the five-minute bands"
            " `reports/occupancy.md` uses, because the question here is where"
            " the gain lands and a band must hold enough rows for a sum of"
            " squares to mean anything",
        ),
        (
            "PER_AIRPORT_INTERVAL_S",
            (2.0,),
            CHOSEN,
            "lines 2967 to 2969: below this a per-airport step is printed but"
            " not bracketed, a thousand draws on a difference of a second or two"
            " being a thousand draws spent on noise",
        ),
        (
            "BAGGING_GAIN_S",
            (1.0,),
            CHOSEN,
            "lines 4101 to 4111: one second on a 230-second error is four tenths"
            " of a percent, about the run-to-run variation the ladder showed"
            " when the rule was set. The comment records that the basis no"
            " longer holds and that the rule was retired on 19 September",
        ),
        ("OCTOBER_TRAIN", (1, 10), MECHANISM, "a month range, not a quantity"),
        (
            "BAGGING_BEFORE_TAIL_GAINS_S",
            (1.1, 1.37),
            QUOTED,
            "lines 4152 to 4156: the same measurement on this rung before the"
            " recorded-taxi tail came out of the fit, when it cleared the"
            " threshold. The run that made it fitted rows this report no longer"
            " fits, under the booster's earlier setting",
        ),
        (
            "BAGGING_EARLIER_GAINS_S",
            (0.78, 0.98),
            QUOTED,
            "lines 4159 to 4166: the same measurement on the rung below, which"
            " did not clear the threshold. Recorded in `docs/plan.md` and"
            " `taxiout.models`; recomputing it would cost ten fits to reproduce"
            " a number already on the record",
        ),
        (
            "TAIL_EXCLUSION_H",
            (4.0, 2.0),
            CHOSEN,
            "lines 4319 to 4325: both thresholds fixed before the run and both"
            " reported, so that reporting the one that came out better is not"
            " available as a move. Four hours is the mechanism, two the wider"
            " reading",
        ),
        (
            "NOISE_FLOOR_FACTOR",
            (2.0,),
            MECHANISM,
            "root two, as `_noise_floors` computes the floor: the standard"
            " deviation of a difference of two independent fits",
        ),
    ),
    *_mod(
        "scripts/sweep_params.py",
        (
            "GRID",
            (0.015, 0.03, 0.06, 63, 127, 255, 20, 50, 200),
            CHOSEN,
            "lines 119 to 123: fixed in the pre-registration of 20 September"
            " 2026 and not reopened, each new value one step past the old grid's"
            " edge at its own spacing",
        ),
        (
            "TREES",
            (300, 600, 1000, 1500, 2000, 3000),
            CHOSEN,
            "lines 119 to 123, covering this too, with the module docstring's"
            " reading rule: the first k trees of a boosted model are the same"
            " model whatever the total, so one fit scores every count",
        ),
        (
            "FIRST_SWEEP",
            (
                0.03,
                0.06,
                0.1,
                31,
                63,
                127,
                50,
                200,
                500,
                300,
                600,
                1000,
                235.0,
                235.0,
                244.8,
            ),
            QUOTED,
            "the first sweep's terms and its October figures, 3 September 2026,"
            " kept as the record of a run that happened and no longer run by"
            " that file. Its numbers are under `data/processed/`, gitignored",
        ),
        (
            "LARGEST_CELL",
            (0.03, 255, 20),
            MECHANISM,
            "line 147: the widest tree at the smallest leaf floor, which"
            " is the grid's cost bound by construction rather than a choice",
        ),
        (
            "SEEDS",
            (5,),
            CHOSEN,
            "`taxiout.models.BAGGING_SEEDS` reused, so the floor is measured"
            " with the estimator the submission would ship",
        ),
        (
            "BLOCK_07_OCTOBER_PRINTED",
            (217.28, 217.65, 217.43, 217.37, 217.43),
            QUOTED,
            "`reports/preregistered-2026-09-20.md`, Experiment 1: the five"
            " October single-seed fits the sweep's gate must reproduce",
        ),
        (
            "BLOCK_07_OCTOBER_FLOOR",
            (0.19,),
            QUOTED,
            "the same report's seed floor on that leg",
        ),
        (
            "BLOCK_07_OCTOBER_ROWS",
            (185674,),
            QUOTED,
            "the same report's October row count",
        ),
        (
            "BLOCK_07_OCTOBER_FLAGGED",
            (115,),
            QUOTED,
            "the same report's flagged-row count on October",
        ),
        (
            "BLOCK_07_OCTOBER_FIT_ROWS",
            (1570127,),
            QUOTED,
            "the same report's October fitting-sample size",
        ),
    ),
)


# --------------------------------------------------------------------------
# the register

CONSTANTS: tuple[Constant, ...] = (
    *_REPO_ROOT,
    # ---- src/taxiout, the library -------------------------------------
    *_mod(
        "src/taxiout/airports.py",
        (
            "AIRPORTS",
            (
                50.0333,
                8.5706,
                5500,
                48.3538,
                11.7861,
                5500,
                51.47,
                0.4543,
                4500,
                52.3105,
                4.7683,
                6500,
                41.2974,
                2.0833,
                4500,
                40.4719,
                3.5626,
                6500,
                49.0097,
                2.5479,
                6500,
                41.8003,
                12.2389,
                5500,
                36.8987,
                30.8005,
                5000,
                41.2753,
                28.7519,
                8500,
                47.4647,
                8.5492,
                4500,
            ),
            QUOTED,
            "reference coordinates and a bounding radius per aerodrome, from"
            " published aerodrome data; a bounding box for an Overpass query and"
            " never a model input. Regenerable from OurAirports today",
        ),
    ),
    *_mod(
        "src/taxiout/callsign.py",
        (
            "MIN_LEVEL_ROWS",
            (30,),
            CHOSEN,
            "`taxiout.unimpeded.MIN_CLASS_ROWS` reused, and the code says why at"
            " line 64: a group this thin is a handful of flights and not an"
            " operator's practice",
        ),
    ),
    *_mod(
        "src/taxiout/ceiling.py",
        (
            "GAP_BUCKETS_MIN",
            (0, 2, 2, 5, 5, 15),
            CHOSEN,
            f"{NO_BASIS}: the take-off gap bands, in minutes, that the lag-1"
            " autocorrelation between consecutive same-runway residuals is read"
            " in. The module docstring describes the reading and names no"
            " edges; nothing anywhere records why 0-2, 2-5 and 5-15",
        ),
        (
            "HALF_WINDOW_S",
            (15, 60),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: the oracle regression's window"
            " half-width. The method is recorded twice and printed once: the"
            " module docstring states the regression is on the mean residual"
            " of the other same-runway departures within fifteen minutes"
            " either side, and `reports/congestion.md` line 486 prints the"
            " same sentence under *The ceiling*. Both state the width as the"
            " instrument's definition; no reason for fifteen minutes rather"
            " than ten or twenty is recorded",
        ),
        (
            "MIN_ROWS",
            (30,),
            CHOSEN,
            f"{NO_BASIS}: the row floor, used in two places and not one --"
            " below it a gap band's correlation is not reported, and below it"
            " `oracle` declines to fit and returns a zero slope. Nothing"
            " records why thirty",
        ),
    ),
    *_mod(
        "src/taxiout/congestion.py",
        (
            "WINDOWS_MIN",
            (5, 15, 30, 60),
            CHOSEN,
            "line 59: five is the runway's immediate queue, sixty is the regime,"
            " and the two between are where the literature's counts sit",
        ),
        (
            "DEMAND_HALF_MIN",
            (15, 30, 60),
            CHOSEN,
            "line 63 names them as the half-widths of the centred demand"
            " windows, at the same spacing as `WINDOWS_MIN` above",
        ),
        (
            "CONFIGURATION_WINDOW_S",
            (3600.0,),
            CHOSEN,
            "line 67: the trailing hour over which a runway counts as in use",
        ),
        (
            "CONFIGURATION_MIN_MOVEMENTS",
            (2,),
            CHOSEN,
            "line 67, covering this too: the floor below which a designator is a"
            " stray rather than a runway in use",
        ),
        (
            "CONFIGURATION_MAX_SINCE_S",
            (6, 3600.0),
            CHOSEN,
            f"{NO_BASIS}: six hours as the cap on how stale a configuration"
            " reading may be carried forward. The comment above it covers the"
            " trailing window and the movement floor and not this",
        ),
        (
            "MOVEMENT_COLUMNS",
            (1, 1),
            MECHANISM,
            "subscript indices in a `split('.', 1)[1]`, not quantities",
        ),
    ),
    *_mod(
        "src/taxiout/corridor.py",
        (
            "CORRIDOR_WINDOWS_MIN",
            (5, 15),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: the shared-corridor block's trailing"
            " windows. The module docstring records a reason for the ceiling"
            ' on them -- "Windows of 5 and 15 minutes only: at an hour,'
            ' "shared an edge" means nothing" -- and'
            " `reports/congestion.md` line 370 prints the same sentence under"
            " *The shared corridor*. So the record rules out an hour and does"
            " not record why five and fifteen rather than three and ten",
        ),
    ),
    *_mod(
        "src/taxiout/coverage.py",
        *(
            (
                f"AirportCoverage.{f}",
                (0,),
                MECHANISM,
                "a dataclass field's zero default, not a quantity",
            )
            for f in (
                "parking_positions",
                "with_any_id",
                "taxiways",
                "taxilanes",
                "runways",
                "aprons",
                "holding_positions",
            )
        ),
    ),
    *_mod(
        "src/taxiout/destination.py",
        (
            "EARTH_RADIUS_KM",
            (6371.0088,),
            MECHANISM,
            "the IUGG mean radius, (2a + b) / 3; line 79 records why it differs"
            " from `taxiout.geo`'s fitted radius and by how little",
        ),
    ),
    *_mod(
        "src/taxiout/encoding.py",
        (
            "INNER_FOLDS",
            (5,),
            CHOSEN,
            "line 169: five leaves each row's encoding built from four fifths of"
            " its group, enough to be the group's mean and never the row itself",
        ),
        (
            "SEED",
            (20260904,),
            MECHANISM,
            "a fixed seed. Any value serves; what matters is that it is fixed,"
            " which is what makes the fold assignment a function of the code",
        ),
        (
            "LENGTH_RATIO_MATERIAL",
            (1.5,),
            CHOSEN,
            "line 181 names it the pre-registered threshold for the"
            " length-dependence check and states the test it decides",
        ),
        (
            "MIN_VARIANCE_FRACTION",
            (0.05,),
            CHOSEN,
            "line 186: a guard and not a fitted quantity, because a linear fit"
            " extrapolated to a very short route can go negative",
        ),
        (
            "MIN_BETWEEN_FRACTION",
            (0.05,),
            CHOSEN,
            "line 192: at the floor the encoding already shrinks twenty times"
            " harder than the uncorrected estimate, which is shrink-to-prior in"
            " every practical sense",
        ),
        (
            "WithinVariance.groups",
            (0,),
            MECHANISM,
            "a dataclass field's zero default, not a quantity",
        ),
    ),
    *_mod(
        "src/taxiout/exposure.py",
        (
            "LAG_LO_S",
            (30, 60.0),
            CHOSEN,
            f"{NO_BASIS}: the plausibility floor on a plan lag, minus thirty"
            " minutes -- a pushback this far ahead of the plan is not a lag."
            " Nothing in the module or the reports records why thirty minutes",
        ),
        (
            "LAG_HI_S",
            (2, 3600.0),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: the plausibility ceiling on a plan lag,"
            " two hours. The same two hours as `taxiout.tails.LONG_S`, whose"
            " reason is recorded at `src/taxiout/tails.py` line 88 -- two hours is above"
            " every plausible taxi-out and below every artifact. **Cited and"
            " not imported**: that sentence is about the length of a taxi, and"
            " this bounds the gap between a plan and a pushback, so an import"
            " would tie two decisions the record never ties together and would"
            " move this window silently if the record-quality threshold moved."
            " What the record does not hold is why two hours bounds a plan lag",
        ),
        (
            "TAXI_IN_LO_S",
            (60.0,),
            CHOSEN,
            f"{NO_BASIS}: the plausibility floor on an arrival's taxi-in, one"
            " minute. Nothing records why",
        ),
        (
            "TAXI_IN_HI_S",
            (3600.0,),
            CHOSEN,
            f"{NO_BASIS}: the plausibility ceiling on an arrival's taxi-in,"
            " one hour. Unlike `LAG_HI_S` and `ARRIVAL_LATE_S` this one is not"
            " two hours and has no counterpart carrying a reason anywhere",
        ),
        (
            "ARRIVAL_LATE_S",
            (2, 3600.0),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: the plausibility bound on an arrival's"
            " lateness, the same two hours as `LAG_HI_S` above and as"
            " `taxiout.tails.LONG_S`, whose reason is at `src/taxiout/tails.py` line 88."
            " Cited on the same terms and for the same reason: that sentence"
            " bounds a taxi duration and this bounds a lateness, so the"
            " citation places the magnitude and does not justify this use",
        ),
        (
            "MIN_ROWS",
            (100,),
            CHOSEN,
            "line 83: a cell with fewer rows than this on either side is not compared",
        ),
    ),
    *_mod(
        "src/taxiout/features.py",
        (
            "REQUIRED_COLUMNS",
            (1, 0),
            MECHANISM,
            "subscript indices in a source-prefix split, not quantities",
        ),
        (
            "MOVEMENT_COLUMNS",
            (1, 0),
            MECHANISM,
            "subscript indices in a source-prefix split, not quantities",
        ),
        ("_DAY", (86400.0,), MECHANISM, "seconds in a day"),
        ("_MINUTE", (60.0,), MECHANISM, "seconds in a minute"),
    ),
    *_mod(
        "src/taxiout/geo.py",
        ("WGS84_A", (6378137.0,), MECHANISM, "a WGS84 defining parameter"),
        ("WGS84_F", (1.0, 298.257223563), MECHANISM, "a WGS84 defining parameter"),
        ("WGS84_E2", (2.0,), MECHANISM, "derived from the two WGS84 parameters above"),
        (
            "SIMPLIFY_M",
            (3.0,),
            CHOSEN,
            "lines 137 to 145 carry the measurement that chose it: six of seven"
            " airports flatten by 2.5 to 3.0 m, and 1.5 m produced the anomaly"
            " that prompted the check",
        ),
    ),
    *_mod(
        "src/taxiout/graph.py",
        (
            "DEDUP_M",
            (0.5,),
            CHOSEN,
            "line 64: below the positional accuracy of the imagery these ways"
            " are traced from, so merging within it cannot join two genuinely"
            " distinct junctions",
        ),
        (
            "SPLIT_M",
            (1.5,),
            CHOSEN,
            "line 68: kept tight because taxiway centrelines at a hub can pass"
            " within a few metres without connecting",
        ),
        (
            "MAX_SNAP_M",
            (150.0,),
            CHOSEN,
            "lines 73 to 86 carry the measured bimodality that chose it: real"
            " attachments, then a jump to 105-137 m with no stands in between",
        ),
        (
            "DIRECT_SNAP_M",
            (60.0,),
            CHOSEN,
            "line 89: attachments at or below this are real, the stand meeting"
            " mapped pavement",
        ),
        (
            "CELL_M",
            (50.0,),
            MECHANISM,
            "a spatial index's cell size; line 92 records that any value above"
            " `SPLIT_M` and `MAX_SNAP_M` works and that it affects speed only",
        ),
        (
            "TaxiGraph._next_node",
            (0,),
            MECHANISM,
            "a counter's initial value, not a quantity",
        ),
        (
            "TaxiGraph._deduped_m",
            (0.0,),
            MECHANISM,
            "an accumulator's initial value, not a quantity",
        ),
    ),
    *_mod(
        "src/taxiout/join.py",
        ("STAGES", (1, 2, 3), MECHANISM, "the join stages' ordinals, not quantities"),
        ("UNMATCHED", (0,), MECHANISM, "a sentinel for the unmatched stage"),
    ),
    *_mod(
        "src/taxiout/metar.py",
        (
            "VIS_LOW_M",
            (1500.0,),
            CHOSEN,
            "line 78: where low-visibility procedures are typically in force",
        ),
        (
            "VIS_HIGH_M",
            (8000.0,),
            CHOSEN,
            "line 78, covering this too: 8 km is ordinary daylight visibility",
        ),
        (
            "STATUTE_MILE_M",
            (1609.344,),
            MECHANISM,
            "metres in a statute mile, the unit the archive reports visibility in",
        ),
        (
            "MAX_CARRY_H",
            (3,),
            CHOSEN,
            "line 88 states the rule it fixes: how many hours a report may be"
            " carried forward into hours without one",
        ),
    ),
    *_mod(
        "src/taxiout/models.py",
        (
            "BOOSTER_PARAMS",
            (6000, 0.03, 127, 20, 0.9, 0.8, 1, 64, 20.0, 20260902, 8, 1),
            CHOSEN,
            "the cell left standing by the extension of the second sweep, 23"
            " September 2026, scored 214.4391 s on October against 214.9045 s"
            " for the cell it replaced, a gain of 0.4654 s at 2.56 times the"
            " 0.1815 s seed floor of that run. The trees axis is the only one"
            " of the three reopened that won, and it won by about 0.59 s on"
            " three paired contrasts. The leaf and rate setting was taken on"
            " cost between two cells this instrument cannot separate:"
            " (0.015, 255) at 6,000 trees is the better reading by 0.1350 s,"
            " which is 0.74 of the floor, and was not taken because it costs"
            " 1.35x the October fit time. Lines 147 to 262 carry both sweeps'"
            " terms and the extension's, and the reason the two sweeps'"
            " October figures are not comparable; `scripts/sweep_params.py` is"
            " the instrument for the sweeps, and the extension was run from"
            " outside this tree with its per-cell records under"
            " `data/processed/`, gitignored",
        ),
        (
            "BAGGING_SEEDS",
            (5,),
            CHOSEN,
            "lines 337 to 374: five fits of one rung with nothing changed but the"
            " seed, adopted for the submission by the rule of 19 September as"
            " `reports/baseline.md` reads it under the parameters that ship, with"
            " the earlier readings against the rule of their day kept as history",
        ),
        (
            "TAIL_EXCLUSION_ADOPTED_H",
            (4.0,),
            CHOSEN,
            "lines 811 to 832: fixed at four hours on the mechanism and not by a sweep,"
            " because no aircraft taxis for four hours, and reported against an"
            " un-excluded fit and a two-hour arm both fixed before the run",
        ),
        ("_NET_CHAIN", (0,), MECHANISM, "an index into a rung list, not a quantity"),
        ("ABOVE_SUBMISSION", (1,), MECHANISM, "a slice offset, not a quantity"),
    ),
    *_mod(
        "src/taxiout/occupancy.py",
        (
            "_FAR_S",
            (10, 366, 86400),
            MECHANISM,
            "ten years in seconds: further back than any covered block reaches,"
            " which line 84 states is the point of not having a window at all",
        ),
        (
            "COVERAGE_MARGIN_S",
            (86400,),
            CHOSEN,
            "line 88: how far into its covered block a take-off must sit before"
            " no-arrival-at-this-stand is a fact about the stand and not about"
            " the data's edge",
        ),
        (
            "MOVEMENT_COLUMNS",
            (1, 1),
            MECHANISM,
            "subscript indices in a source-prefix split, not quantities",
        ),
    ),
    *_mod(
        "src/taxiout/osm.py",
        (
            "POLITE_DELAY_S",
            (5.0,),
            CHOSEN,
            "line 51: the Overpass free tier allows about two concurrent slots"
            " per address and throttles hard, and a short pause is cheaper than"
            " absorbing backoffs",
        ),
    ),
    *_mod(
        "src/taxiout/planlags.py",
        (
            "HALF_WINDOW_MIN",
            (15,),
            CHOSEN,
            f"{NO_BASIS}: the half-width of the runway plan-lag neighbourhood,"
            " which names the feature columns as well as cutting the window."
            " Nothing in the module records why fifteen minutes",
        ),
        (
            "MOVEMENT_COLUMNS",
            (1, 1),
            MECHANISM,
            "subscript indices in a source-prefix split, not quantities",
        ),
    ),
    *_mod(
        "src/taxiout/routing.py",
        (
            "RUNWAY_PENALTY",
            (4.0,),
            CHOSEN,
            "line 56 states the mechanism and names the sensitivity check in"
            " `docs/modelling_notes.md`: crossing a runway stays cheap, running"
            " a kilometre along one does not",
        ),
    ),
    *_mod(
        "src/taxiout/runways.py",
        (
            "Runway.displaced_m",
            (0.0,),
            MECHANISM,
            "a dataclass field's zero default, not a quantity",
        ),
        (
            "DISPLACED_ALIGN_DEG",
            (10.0,),
            CHOSEN,
            "line 128 states why both conditions are required: alignment alone"
            " would capture a parallel runway, adjacency alone a crossing one",
        ),
        (
            "DISPLACED_JOIN_M",
            (80.0,),
            CHOSEN,
            f"{NO_BASIS}: how close a displaced-threshold way must start to"
            " where a runway ends before it is joined to it. The comment above"
            " gives the rule -- alignment and adjacency both required -- and"
            " not this distance",
        ),
    ),
    *_mod(
        "src/taxiout/saturation.py",
        (
            "GAP_S",
            (90.0,),
            CHOSEN,
            "the module docstring names it the wake-turbulence floor, which is a"
            " mechanism and not a fitted value",
        ),
        (
            "WINDOWS_MIN",
            (5, 15),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: the trailing windows. The method is"
            " recorded in the module docstring, which reads the gap share"
            ' "over the trailing five and fifteen minutes", and in'
            " `docs/plan.md` line 374, the dated record of the block's design."
            " **Not in `reports/congestion.md`**: that report prints the"
            " block's feature list and its `sat_span3_s` distribution and"
            " states no window. They match `congestion.WINDOWS_MIN`'s first"
            " two, which is a resemblance and not a recorded reason, and no"
            " reason for five and fifteen is recorded",
        ),
        (
            "MIN_GAP_WINDOW_MIN",
            (5,),
            CHOSEN,
            f"{NO_BASIS}: the trailing window the smallest consecutive"
            " take-off gap is read over, which names the feature column too."
            " Nothing records why five minutes",
        ),
        (
            "SPAN_GAPS",
            (3,),
            CHOSEN,
            f"{NO_BASIS}: how many consecutive take-off gaps the span reading"
            " spans. Nothing records why three",
        ),
        (
            "SPAN_CAP_S",
            (1800.0,),
            CHOSEN,
            f"{NO_BASIS}: thirty minutes, used twice -- the cap on the span,"
            " and the value the span takes when fewer than three take-offs"
            " precede the subject in the window, which `sat_span3_short` flags."
            " The docstring states both uses and records no reason for the"
            " value",
        ),
        (
            "MOVEMENT_COLUMNS",
            (1, 1),
            MECHANISM,
            "subscript indices in a source-prefix split, not quantities",
        ),
        (
            "_FAR_S",
            (10, 366, 86400),
            MECHANISM,
            "ten years in seconds, as in `taxiout.occupancy`",
        ),
    ),
    *_mod(
        "src/taxiout/tails.py",
        ("HOUR", (3600.0,), MECHANISM, "seconds in an hour"),
        (
            "NEAR_S",
            (60.0,),
            CHOSEN,
            "line 85: the 60 s absorbs the schedule's minute resolution, and 83%"
            " of the rows it catches are within 5 s",
        ),
        (
            "LONG_S",
            (2,),
            CHOSEN,
            "line 85, covering this too: two hours is above every plausible"
            " taxi-out and below every artifact",
        ),
        (
            "MIN_COUNT",
            (5,),
            CHOSEN,
            "line 92: below five rows a cell's defaulted share is a coin toss,"
            " and above ten the rule discards the cells the term exists for --"
            " the Fiumicino `>12 h` cell has 29 rows in a year",
        ),
        (
            "MIN_SHARE",
            (0.05,),
            CHOSEN,
            "line 92, covering this too: five percent separates cells whose"
            " artifact share is a property of the feed from cells with a stray"
            " row",
        ),
    ),
    *_mod(
        "src/taxiout/unimpeded.py",
        (
            "PRU_PERCENTILE",
            (0.1,),
            QUOTED,
            "the PRU reference method's own percentile, named in"
            " `docs/prior_art.md`; not this project's choice",
        ),
        (
            "MIN_PAIR_COUNT",
            (30,),
            QUOTED,
            "the PRU reference method's own floor for a pair",
        ),
        (
            "MAX_SPEED_FACTOR",
            (2.0,),
            CHOSEN,
            "line 76 states the guard it imposes on the fitted speed",
        ),
        (
            "MIN_CLASS_ROWS",
            (30,),
            CHOSEN,
            "line 91: below this a wake class is a handful of flights and is"
            " pooled; `taxiout.callsign.MIN_LEVEL_ROWS` reuses it and says so",
        ),
        (
            "MIN_AIRPORT_PAIRS",
            (5,),
            CHOSEN,
            "line 99 states the rule: below this an airport has too few pairs"
            " for its own fit",
        ),
        (
            "MIN_UNIMPEDED_S",
            (30.0,),
            CHOSEN,
            "line 103: a floor below which an unimpeded time is not a taxi",
        ),
        (
            "GeometricUnimpeded.default_intercept",
            (0.0,),
            MECHANISM,
            "a dataclass field's zero default, not a quantity",
        ),
        (
            "GeometricUnimpeded.n_fit",
            (0,),
            MECHANISM,
            "a dataclass field's zero default, not a quantity",
        ),
    ),
    *_mod(
        "src/taxiout/validation.py",
        (
            "DECEMBER_TRAIN",
            (1, 12),
            MECHANISM,
            "a month range, fixed 2 September 2026 once the ranking set's shape"
            " was known, and written out so changing one is a visible edit"
            " (line 67)",
        ),
        ("JULY_TRAIN", (1, 7), MECHANISM, "a month range, as `DECEMBER_TRAIN` above"),
        (
            "PAIR_FOLDS",
            (3,),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: how many folds the held-out-pair"
            " contrast uses. The design is recorded and printed:"
            " `reports/baseline.md` line 282, *Held-out pairs*, states"
            ' "3 folds" and explains the dealing -- whole (airport, stand,'
            " runway) pairs round-robin within each airport, so every fold"
            " holds out a volume-representative third. No reason for three"
            " rather than five or ten is recorded",
        ),
        (
            "SEEN_FRACTION",
            (0.05,),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: the share of rows sampled as the matched"
            " seen-pair evaluation set. The design is recorded and printed:"
            " `reports/baseline.md` line 287 states the sample is of rows"
            " whose pairs stayed in training and which were themselves"
            " excluded from the fit, which is what the sample is for. No"
            " reason for five percent rather than one or ten is recorded",
        ),
        (
            "SEED",
            (20260902,),
            MECHANISM,
            "a fixed seed. Any value serves; what matters is that it is fixed",
        ),
        (
            "BOOTSTRAP_DRAWS",
            (1000,),
            CHOSEN,
            "line 83: a thousand puts the 2.5th and 97.5th percentiles on about"
            " 25 draws each, coarse but honest for an interval read to the"
            " nearest second, and it costs nothing since no model is refitted",
        ),
        (
            "BOOTSTRAP_LEVEL",
            (0.95,),
            CHOSEN,
            f"{VALUE_NOT_JUSTIFIED}: the interval's coverage. The"
            ' conventional level, printed as "95% interval" throughout the'
            " reports and the paper, and visible in the code as well: the"
            " comment above `BOOTSTRAP_DRAWS` at line 83 sizes the draws by"
            " where they put the 2.5th and 97.5th percentiles, which are this"
            " level's own tails. **Nothing records why 95 rather than any"
            " other level.** This is the most load-bearing of the constants"
            " whose value is unargued, since every interval in the paper rests"
            " on it, and the one where the absence matters least, because 95%"
            " is the universal default and departing from it is what would"
            " have needed a reason",
        ),
    ),
    *_mod(
        "src/taxiout/windows.py",
        (
            "_TIME_SPAN",
            (1, 40),
            MECHANISM,
            "a sentinel span guarding the stamp-resolution check, not a quantity",
        ),
    ),
    *_SCRIPTS,
)


# --------------------------------------------------------------------------
# authored artifacts


class Authored(NamedTuple):
    """A module whose numbers are authored rather than measured.

    ``condition`` is what makes that safe, and it is a claim the gate proves
    rather than a claim this record makes. A fixture nothing reads is a
    verification aid; a fixture a model figure reads is filler.
    """

    module: str
    what: str
    condition: str


AUTHORED_ARTIFACTS: tuple[Authored, ...] = (
    Authored(
        "src/taxiout/synthetic.py",
        "a fictional airfield where every distance, turn and repair is"
        " hand-derived, with worked traffic in the challenge's own schema so the"
        " pipeline runs end to end without the challenge data. Its numeric"
        " constants ARE the field's definition, which is why the module is"
        " registered once here rather than constant by constant: there is no"
        " provenance question to answer about the coordinates of an invented"
        " stand",
        "no generator of a model figure imports it",
    ),
)

# The generators whose output is a model figure, and which therefore may not
# import the authored field. Named here rather than inferred, so adding a
# generator is a visible edit to this list.
MODEL_FIGURE_GENERATORS: tuple[str, ...] = (
    "scripts/run_baselines.py",
    "scripts/make_submission.py",
    "scripts/sweep_params.py",
    "scripts/pooled_tails.py",
    "scripts/preregistered.py",
    "scripts/report_headroom.py",
    "scripts/report_transfer.py",
    "scripts/report_residual.py",
    "scripts/report_congestion.py",
    "scripts/report_occupancy.py",
)


# --------------------------------------------------------------------------
# frozen documents


class Frozen(NamedTuple):
    """A dated record that must not be brought up to date.

    These need the OPPOSITE gate from every other file here. The risk on a
    currency-checked report is that it goes stale; the risk on a dated record is
    that somebody edits it, and a currency check over one would demand
    destroying the record. The Block 07c decision was explicit that
    ``docs/plan.md``'s criteria are never edited and that anything later added
    is additive and dated.
    """

    path: str
    sha256: str
    why: str


FROZEN_DOCUMENTS: tuple[Frozen, ...] = (
    Frozen(
        "docs/plan.md",
        "9c48b07cc136eaf14d09fec4e8b2b969b4427dea54e3f72430745913607ae0b8",
        "the gated plan, and the record of every criterion fixed before the"
        " measurement it governs. Additions are additive and dated; nothing"
        " already in it is revised",
    ),
    Frozen(
        "docs/review_headroom.md",
        "1600a26cded733bfb2bf2e982717af2567b508f88c6eff13799a743ad9672910",
        "the review of 8 September 2026 whose §2 produced"
        " `EXPECTED_HEADROOM_S` and whose §7 characterised it as a floor. Its"
        " value as evidence is entirely in its date, so bringing its figures up"
        " to date would destroy what it is for",
    ),
    Frozen(
        "reports/migration-2026-09-10.md",
        "76e557901ae48f0fb360f459f5db160d6b969195cfbbca275b1aa7d2178549b6",
        "the record of the move between machines, and the source of"
        " `HEADROOM_UNCOMMITTED_S`, `HEADROOM_FLOOR_BEFORE_S` and the fourth"
        " audit reading",
    ),
    Frozen(
        "reports/reproduction-2026-09-11.md",
        "9c45be94071cdef75120cb79df834185d618cb20f765992e00c968df6e20de8e",
        "a dated reproduction of one run on one machine",
    ),
    Frozen(
        "reports/row-wise-2026-09-11.md",
        "a2f22f037d94c05a336567b30511d240df3a8905df030ac4bdd30b7ebc9916eb",
        "a dated measurement of one booster setting",
    ),
    Frozen(
        "reports/threads-2026-09-12.md",
        "087c2569a095ab5ebf3a3037789fb30401f022c0eb2ae83f85d4367a1de44291",
        "a dated measurement of what the thread count changes",
    ),
    Frozen(
        "reports/container-2026-09-12.md",
        "c5191c26c9578dd89ca4c9862b50b051a78964f882b55394766189044b2fd583",
        "a dated record of the reference container's pins",
    ),
    Frozen(
        "reports/submission-2026-09-19.md",
        "62fadca6eb1508aebb66145128d33cfe14e9e0323816b203bb4d0a44bba34437",
        "a dated record of one scored file, written at the tree it was built from",
    ),
    Frozen(
        "reports/preregistered-2026-09-20.md",
        "aeb94fb6cbe4580ba78a7284a71aafa7c801ec80dc0d9493a414e34b1ff7b9fa",
        "the pre-registered experiments and their verdicts, and the source of"
        " `sweep_params.BLOCK_07_OCTOBER_*`. Its criteria were fixed before its"
        " numbers existed and the document is the proof of that order",
    ),
)


# --------------------------------------------------------------------------
# what is still open


# --------------------------------------------------------------------------
# what a rebuild of an external input can be pinned to


class OsmPin(NamedTuple):
    """One airport's OSM pin: the Overpass database state its cache came from."""

    icao: str
    timestamp_osm_base: str


# The basis of the eleven entries below, stated once because it is the same for
# each: every entry's source is its own airport's cache file, and everything
# else about where the value came from is shared.
OSM_PINS_BASIS = (
    "each is the `osm3s.timestamp_osm_base` of that airport's cached Overpass"
    " answer, `data/raw/<ICAO>.json.gz`, which is the database state the"
    " Overpass instance that served it had ingested when it answered, on"
    " 1 September 2026. The eleven were fetched one after another and"
    " unpinned, so the database moved under the fetch and eleven answers carry"
    " seven moments rather than one. Read from the cache files on 25 September"
    " 2026, and the same eleven values Block 09e read from them on 22"
    " September. **The cache header cannot be the source of truth, because a"
    " rebuild overwrites it.**"
    " Overpass stamps every answer with the answering instance's current"
    " database state and never with the `[date:]` pin, which Block 09h measured"
    " on three answers carrying a 2026 pin, a 2020 pin and none, so a pinned"
    " rebuild writes each file stamped with the day of the rebuild. And"
    " `data/raw/` is gitignored, so a clean checkout has no cache and would"
    " have no pins. Recorded here, they survive both"
)

# One entry per airport, in the order the cache files were written on
# 1 September, which is the order of ``taxiout.airports.AIRPORTS``. The stamps
# never go backwards down the list, which is what a sequential fetch against a
# moving database looks like. ``scripts/fetch_osm.py --pin-from-register`` pins
# each airport's query to its entry.
OSM_PINS: tuple[OsmPin, ...] = (
    OsmPin("EDDF", "2026-09-01T13:23:05Z"),
    OsmPin("EDDM", "2026-09-01T13:23:05Z"),
    OsmPin("EGLL", "2026-09-01T13:23:05Z"),
    OsmPin("EHAM", "2026-09-01T13:24:06Z"),
    OsmPin("LEBL", "2026-09-01T13:26:06Z"),
    OsmPin("LEMD", "2026-09-01T13:27:05Z"),
    OsmPin("LFPG", "2026-09-01T13:27:05Z"),
    OsmPin("LIRF", "2026-09-01T13:29:10Z"),
    OsmPin("LTAI", "2026-09-01T13:30:11Z"),
    OsmPin("LTFM", "2026-09-01T13:30:11Z"),
    OsmPin("LSZH", "2026-09-01T13:32:15Z"),
)


class Pinned(NamedTuple):
    what: str
    how: str
    recoverable: str
    caveats: tuple[str, ...]


# **A pin is a claim about what a rebuild would return.** Exercising one costs
# the network and an Overpass instance willing to answer an attic query, so the
# record says how the pin is applied, how a rebuild proves it was, and what
# would make it fail. The OSM pin was exercised on LIRF on 25 September 2026,
# and on all eleven airports on 26 September under the rule the fourth caveat
# states, when every answer came back identical to the cache the matrices were
# built from and the matrices came out byte for byte.
REBUILD_PINS: tuple[Pinned, ...] = (
    Pinned(
        "the OSM movement-area geometry for the eleven airports, under"
        " `data/raw/*.json.gz`",
        "`python scripts/fetch_osm.py --pin-from-register --force`, which pins"
        " each airport's Overpass query to its own entry in `OSM_PINS`, refuses"
        " an answer from a database older than its pin, and records, in the"
        " sidecar `data/raw/PINS.json` beside the cache, the pin each airport"
        " was fetched at, the instance that answered, the time of the fetch and"
        " the database state the answer came from",
        "per airport, and only per airport, and from `OSM_PINS` rather than"
        " from the cache. The eleven entries carry **seven** distinct stamps"
        " spanning 2026-09-01T13:23:05Z to 13:32:15Z, because the fetch was"
        " sequential and unpinned and each airport was answered from whatever"
        " state its instance had ingested when it was served; one `--date` would"
        " pin eight of the eleven to a state their matrices were not built from."
        " **The cache cannot carry them**: every answer is stamped with the"
        " answering instance's own current database state and never with the"
        " pin, so a pinned rebuild re-stamps each file with the rebuild day. The"
        " mode that read the pins from the cache, `--pin-from-cache`, would have"
        " worked exactly once and then pinned every airport to that day with no"
        " error, and it now refuses. What a rebuilt cache carries in place of"
        " the header is the sidecar, and `taxiout.osm.pin_agreement` reads a"
        " cache by whichever evidence it has: an original answer by its header"
        " stamp, a rebuilt one by its sidecar entry, which must equal"
        " `OSM_PINS` and the answer's own fetch record, and whose database"
        " stamp must not precede its pin",
        (
            "attic retention is not guaranteed. An instance may decline a query"
            " at a date this old, and one that declines is a failure to report."
            " The mode refuses to fetch an airport the register has no pin for,"
            " because **a silently unpinned fetch is a stop**: it would replace"
            " an answer whose OSM state is recorded with one whose is not.",
            "**nothing in an answer's header proves the pin was applied.** The"
            " proof is the elements: `--compare FETCHED REFERENCE` matches each"
            " airport's parsed elements on type and id against the cache the"
            " matrices were built from and compares every field, and the"
            " matrices built from them are compared after that. Not the payload"
            " bytes, which differ in the header between any two answers.",
            "an airport unchanged since its pin answers the same pinned or not,"
            " so a match alone does not show the pin at work. The control does:"
            " `--control-date` asks the instance that answered for one airport"
            " at a date whose answer is known to differ, and reports whether it"
            " does. LIRF at 2020-01-01T00:00:00Z differs, 355 elements against"
            " 494, measured on overpass-api.de on 25 September 2026.",
            "the client falls back to two other mirrors when the first fails,"
            " and **on 26 September 2026 both held databases from May 2026 and"
            " could not serve a September pin**: `overpass.private.coffee` stood"
            " at 2026-05-06T03:25:00Z and `overpass.kumi.systems` at"
            " 2026-05-31T22:37:44Z. An instance cannot serve a state it has not"
            " ingested, so an answer whose `osm3s.timestamp_osm_base` precedes"
            " its pin is not the pinned answer, whatever was requested; Block 11"
            " took LSZH from the first as it stood in May, and only `--compare`"
            " saw it. The header cannot show that a pin was applied, but it can"
            " show that one could not have been, so the client now refuses such"
            " an answer, says which mirror and which state it refused, asks that"
            " mirror nothing more for the airport, and gives the primary as many"
            " asks as the whole rotation had; the sidecar records each answer's"
            " stamp, and `pin_agreement` fails one that precedes its pin. **A"
            " rebuild in which every mirror fails or refuses is a stop and not a"
            " fallback**: the pin cannot be served that day.",
        ),
    ),
)


class Pending(NamedTuple):
    """One excursion not yet resolved, with a reason and an owning block.

    The same instrument as ``run_baselines.py``'s ``SEED_AVERAGE_PENDING``: **emptied
    by the work and never by editing the tuple**, and empty is the freeze
    condition. ``scripts/check_provenance.py --strict`` fails while it is not
    empty, and the ordinary run prints it and passes, because a gate that fails
    today cannot be wired into a commit path.
    """

    what: str
    why: str
    owner: str


PROVENANCE_PENDING: tuple[Pending, ...] = ()


# --------------------------------------------------------------------------
# what the register does not reach

SCOPE: tuple[tuple[str, str], ...] = (
    (
        "module-level numeric assignments in `src/` and `scripts/`",
        "covered by `CONSTANTS`, one entry each, and check 1 fails on any the"
        " scanner finds and the register does not carry",
    ),
    (
        "numeric defaults in function signatures",
        "NOT covered. The scanner counts them and the census prints the count,"
        " but they are parameters of a call and not constants of a module, and"
        " every one that matters is bound to a module constant at the call site."
        " Stated here so the silence is not read as a pass",
    ),
    (
        f"`{REGISTER_MODULE}`",
        "NOT covered, and it cannot be: its aggregate assignments hold every"
        " registered literal in the repository, so an entry for them would be a"
        " copy of the register inside itself and would drift whenever any other"
        " entry changed. Its own prose figures are the bases, each of which"
        " names its source, and check 7 excludes it for that reason",
    ),
    (
        "`scripts/check_provenance.py`",
        "its `REPO` is registered like every other script's; its prose figures"
        " are the exemption table's own data, which is a list of figures by"
        " construction, so check 7 excludes it. It writes no report",
    ),
    (
        "which generators check 9 sees",
        "those whose output path carries an `.md` string literal. Two build a"
        " dated filename with a format field -- `scripts/report_submission.py`"
        " and `scripts/preregistered.py` -- and are outside it. Both write a"
        " frozen dated record, which check 5 covers instead",
    ),
    (
        "`src/taxiout/synthetic.py`",
        "covered once by `AUTHORED_ARTIFACTS` rather than constant by constant,"
        " with the condition that makes it safe proved by check 4",
    ),
    (
        "decimal figures in generator prose strings",
        "covered by check 7: each is computed from the data or registered as"
        " deliberate prose",
    ),
    (
        "decimal figures in code comments",
        "NOT covered by an automated check. A comment is not read by any run"
        " and cannot reach a report, so a stale figure in one misleads a reader"
        " of the source and nothing else. The census counts them",
    ),
    (
        "`docs/` and `README.md`",
        "the two frozen documents are covered by check 5, which is an"
        " immutability digest and not a currency check. `docs/paper.md` is"
        " covered by check 14, whose `--strict` passed against every report the"
        " freeze regenerated on 26 September 2026; the rest carries no model"
        " figure",
    ),
    (
        "cross-report citation",
        "covered by check 11, scoped to the citation's own sentence and to"
        " decimal figures. Bare integers are out: a report says *the three"
        " airports* and *5 folds* in words a gate cannot tell from a"
        " measurement. A frozen dated record is out as a citing document,"
        " because its figures were true on its date and the file it cites has"
        " moved since, which is what makes it a record; check 5 holds it to a"
        " digest instead",
    ),
    (
        "`docs/paper.md` as a citing document",
        "covered by check 14 and not by check 11. The paper names its sources"
        " once per section, in a `Source:` line, where a report names them in"
        " the sentence, so `scripts/check_paper.py` reads each figure against"
        " the files its section cites and the sections of them it names."
        " `--strict` passes on it since the figure pass of 25 September and is"
        " run again after every regeneration; `--report` prints every figure"
        " with where it was found",
    ),
    (
        "integers under a thousand in `docs/paper.md`",
        "NOT covered by check 14. A figure there is a decimal, a percentage, or"
        " an integer of four or more digits, and a count or a whole number of"
        " seconds below a thousand is neither: the rows a term covers, a"
        " rounded gain in seconds. The paper writes them as numerals, so they"
        " can go stale, and no check here will say so",
    ),
    (
        "the paper's drafting notes",
        "no longer in `docs/paper.md`: the figure pass of 25 September moved"
        " the section that said it was not part of the paper out of the"
        " repository, so `PAPER_EXCLUDED_SECTIONS` is empty and check 14 reads"
        " every section the paper has. The mechanism stays: a section listed"
        " there is reported with its figure count rather than passed over, and"
        " an entry naming a section the paper does not have fails",
    ),
    (
        "the correctness of a figure in `docs/paper.md`",
        "NOT covered. Check 14 is a check of presence: a figure has to appear,"
        " at the paper's own precision, in the file its section cites and,"
        " where the source line names a section of that file, in that section."
        " A short figure can equal a number in a cited section by chance, and"
        " the check prints every number in scope that a figure matches so a"
        " reader can tell the row the sentence quotes from a coincidence; it"
        " cannot tell them apart itself",
    ),
    (
        "the line numbers a basis cites",
        "covered by check 12. A basis that names a line is a citation and"
        " drifts the moment anything above it is edited: the commit that"
        " created this register edited two of the generators it registers and"
        " left sixteen of its own citations pointing at blank lines and"
        " unrelated code. The cost is that an edit above a cited constant now"
        " fails the gate until the citation is moved, which is the intended"
        " trade -- a citation that fails loudly beats one that drifts quietly",
    ),
    (
        "a line citation into another file",
        "checked only for existing and for not being blank, which is all a"
        " line number in prose can be checked for. Where the cited file is"
        " regenerable the citation is pointed at a frozen dated record"
        " instead, which cannot move under it",
    ),
    (
        "the notes and prompts outside the repository",
        "unreachable by any gate here. The discipline there is that a number is"
        " never invented to stand in for one not yet measured",
    ),
)


# --------------------------------------------------------------------------
# what a report declares it was generated from

# The heading a report's provenance footer carries. Read by
# ``scripts/check_provenance.py``, written by each generator through
# ``report_inputs_footer`` below, so the format is fixed in one place.
REPORT_INPUTS_HEADING = "## What this report was generated from"

# Generators that carry the footer. The rule for who is here WAS the Makefile's
# own measured time, under five minutes, which is why the four expensive reports
# were not here: a footer is only worth anything in the file it describes, and a
# report nobody rebuilds is a report nobody can footer. What replaced it is
# simpler. A run that regenerates a report writes its footer while it is there,
# whatever the report cost, and the four that cost hours went in on 23 September
# with the ladder run that rewrote them. The last two, the routing report and
# the penalty sensitivity, went in on 26 September with the pinned rebuild that
# regenerated them, so every live report is here and ``REPORTS_WITHOUT_FOOTER``
# is empty; it stays, for any report a later run cannot regenerate.
REPORT_INPUTS_DECLARED: tuple[tuple[str, str], ...] = (
    ("scripts/build_taxi_graph.py", "reports/taxi_routing.md"),
    ("scripts/check_determinism.py", "reports/determinism.md"),
    ("scripts/check_penalty_sensitivity.py", "reports/penalty_sensitivity.md"),
    ("scripts/check_runways_ourairports.py", "reports/runway_crosscheck.md"),
    ("scripts/report_congestion.py", "reports/congestion.md"),
    ("scripts/report_coverage.py", "reports/stand_coverage.md"),
    ("scripts/report_geometry_gaps.py", "reports/geometry_gaps.md"),
    ("scripts/report_headroom.py", "reports/headroom.md"),
    ("scripts/report_occupancy.py", "reports/occupancy.md"),
    ("scripts/report_residual.py", "reports/residual.md"),
    ("scripts/report_synthetic_airport.py", "reports/synthetic_airport.md"),
    ("scripts/report_tails.py", "reports/tails.md"),
    ("scripts/report_transfer.py", "reports/transfer.md"),
    ("scripts/review_challenge_data.py", "reports/data_review.md"),
    ("scripts/run_baselines.py", "reports/baseline.md"),
)


def digest(path) -> str:
    """The sha256 of one file, or ``"absent"``."""
    import hashlib
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()


# The mark a footer writes beside an input it digests by content rather than by
# bytes. **The cached Overpass answers, and only they.** Every answer's bytes
# change on every fetch, because its header carries the answering instance's
# own database stamp and the client adds the time of the fetch, so a byte
# digest goes stale after any refetch even where every element is the same,
# and a clean checkout that rebuilt the cache could never verify a report that
# read it. What a report reads from an answer is its elements, so that is what
# is digested.
ELEMENTS_MARK = "(parsed elements)"


def is_overpass_answer(rel: str) -> bool:
    """Whether a repository-relative path is a cached Overpass answer.

    ``data/raw/<ICAO>.json.gz``, which is where ``taxiout.osm.cache_path``
    puts one; nothing else under ``data/raw/`` is one.
    """
    from pathlib import PurePosixPath

    p = PurePosixPath(rel)
    return p.parent == PurePosixPath("data/raw") and p.name.endswith(".json.gz")


def elements_digest(path) -> str:
    """The sha256 of one cached Overpass answer's elements, or ``"absent"``.

    **The one function a footer writes with and check 8 reads with.** The
    canonical form: the payload's ``elements`` and nothing else, so the header
    (``osm3s``, the generator, the version) and the ``taxiout:fetch`` record are
    left out; sorted by type and then id; each serialized as JSON with its keys
    sorted and no whitespace, the list hashed as UTF-8. Two answers that
    ``taxiout.osm.compare_elements`` finds to hold the same elements digest the
    same whatever their headers say; the order they were served in does not
    count here, and ``--compare`` is what holds the order.
    """
    import gzip
    import hashlib
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return "absent"
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        elements = json.load(fh).get("elements", [])
    ordered = sorted(elements, key=lambda e: (str(e.get("type", "")), e.get("id", 0)))
    canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def report_inputs_footer(repo, paths) -> list[str]:
    """The footer naming every input a report was generated from, with digests.

    **Content and not modification time, and the distinction is the whole
    point.** Block 11 reproduces from a clean checkout, where every file
    carries the checkout time, so an mtime ordering is meaningless exactly
    where the check matters most. A digest is not.

    A cached Overpass answer is digested by its parsed elements and marked so
    beside its path, because its bytes change with every fetch and its
    elements need not; every other input by its bytes. See
    ``elements_digest``.

    An input under a gitignored directory is the ordinary case on a clean
    checkout, and it reads ``absent`` rather than being left out: a footer that
    omitted what it could not find would turn an unverifiable report into one
    that looks verified.
    """
    from pathlib import Path

    rows = []
    for p in paths:
        rel = Path(p).resolve().relative_to(Path(repo).resolve()).as_posix()
        if is_overpass_answer(rel):
            rows.append(f"| `{rel}` {ELEMENTS_MARK} | `{elements_digest(p)}` |")
        else:
            rows.append(f"| `{rel}` | `{digest(p)}` |")
    marked = any(ELEMENTS_MARK in row for row in rows)
    return [
        REPORT_INPUTS_HEADING,
        "",
        "Every path this run read, with the sha256 of each as it was read.",
        "`scripts/check_provenance.py` recomputes them and reports this report as",
        "verified, stale or unverifiable; a digest reading `absent` is the",
        "ordinary case on a clean checkout, where the data this repository does",
        "not redistribute is not present.",
        *(
            [
                "",
                f"A path marked {ELEMENTS_MARK} is a cached Overpass answer, and its",
                "digest is of the answer's elements in canonical form and not of the",
                "file's bytes: every answer's header carries the answering instance's",
                "own database stamp and the time of the fetch, so a refetch changes",
                "every byte of a file whose elements are all the same.",
            ]
            if marked
            else []
        ),
        "",
        "| input | sha256 |",
        "| --- | --- |",
        *rows,
    ]


# --------------------------------------------------------------------------
# what a report quotes from another report


# Reports whose generator writes no provenance footer, each with what it costs
# to rebuild and who owns it. **Listed and not silently skipped**: check 8
# fails if a report carries no footer and is not named here, so the set of
# undeclared reports is a decision and not an oversight. A report belongs here
# only while no run regenerates it, because a footer is worth nothing in a file
# that does not carry it. Empty since 26 September 2026, when the pinned
# rebuild regenerated the last two.
class Unfootered(NamedTuple):
    report: str
    generator: str
    cost: str
    owner: str


REPORTS_WITHOUT_FOOTER: tuple[Unfootered, ...] = ()


# A figure in a citation's sentence that the cited report does not carry, and
# the reason it does not have to. **An explicit list with a reason each**, so
# an exemption is a recorded judgement rather than a hole in the instrument.
# The ordinary case is a sentence that sets a quoted figure against one of the
# citing report's own, which is a contrast and not a quotation.
class Uncited(NamedTuple):
    source: str
    cited: str
    figures: tuple[str, ...]
    why: str


REPORT_CITATION_EXEMPTIONS: tuple[Uncited, ...] = (
    Uncited(
        "reports/baseline.md",
        "reports/geometry_gaps.md",
        ("0.29", "0.64", "7.88", "91.19"),
        "the sentence quotes the ranking set's bucket mix from"
        " `geometry_gaps.md` -- 0.52 / 1.36 / 7.95 / 90.17, all four of which"
        " the cited file carries -- and sets it against December's own mix,"
        " which is this report's own figure and is not in the cited file by"
        " construction",
    ),
    Uncited(
        "reports/headroom.md",
        "reports/baseline.md",
        ("15.9815",),
        "the sentence prints this report's own recomputation of the ladder's"
        " step from the stored residuals *against* the +15.98 s"
        " `baseline.md` prints, which the cited file does carry. The point of"
        " the sentence is that the two agree, so the recomputed figure cannot"
        " be in the cited file. Both figures moved with the cell adopted on"
        " 23 September; they read 15.9815 and +15.98 where they read 17.6011"
        " and +17.60 under the cell before it",
    ),
    Uncited(
        "reports/synthetic_airport.md",
        "reports/tails.md",
        ("6.0",),
        "the 6.0 m/s is the authored field's own taxi speed, stated in"
        " `taxiout.synthetic`; the citation of `tails.md` sits in the same"
        " long sentence and is about where the record-quality term is"
        " documented, not about this number",
    ),
)


# --------------------------------------------------------------------------
# the paper's figures

# The paper whose figures ``scripts/check_paper.py`` checks against the files
# its sections cite. Check 14 of ``scripts/check_provenance.py`` runs it.
PAPER = "docs/paper.md"

# The fewest digits an integer needs to be read as a figure in the paper. A
# decimal and a percentage are figures at any size. Below a thousand the
# paper's numerals are dates, list numbers and gate names as often as they are
# measurements, so the line is drawn where every integer is a quantity, and
# ``SCOPE`` records what it leaves out. Held here, beside the patterns that
# narrow the definition, because it is the other half of it.
PAPER_INTEGER_MIN_DIGITS = 4


class PaperPattern(NamedTuple):
    """A kind of token shaped like a figure that is not a measurement."""

    name: str
    pattern: str
    why: str


# Matched by pattern rather than listed one by one, because each is a whole
# kind: the paper refers to its own sections by number throughout, and a list
# of them would grow with every cross-reference. A token inside a match is
# counted and reported by the pattern's name, never silently dropped.
PAPER_EXEMPT_PATTERNS: tuple[PaperPattern, ...] = (
    PaperPattern(
        "section reference",
        r"§\s*\d+(?:\.\d+)*",
        "the paper's reference to one of its own sections. It names a place in"
        " the document and measures nothing",
    ),
    PaperPattern(
        "heading number",
        r"^#{1,6}\s+\d+(?:\.\d+)*\.?(?=\s)",
        "a section's own number at the start of its heading: the same thing as"
        " a section reference, seen from the section itself",
    ),
    PaperPattern(
        "year",
        r"(?<![\w.,])(?:19|20)\d{2}(?![\w%]|\.\d|,\d{3})",
        "a bare four-digit integer from 1900 to 2099. The paper writes every"
        " count of a thousand or more with a thousands separator, so a bare"
        " four-digit integer in it is a year: the training year, the ranking"
        " year, or the year of a cited work",
    ),
    PaperPattern(
        "version string",
        r"(?i:\b(?:v|version|release|python|cpython|odbl|gplv?)\s?)\d+(?:\.\d+)+",
        "a software or license version names a release and measures nothing."
        " A dotted number with three or more parts is never read as a figure"
        " in the first place, so this covers the two-part form after a name",
    ),
)


class PaperSection(NamedTuple):
    """A section of the paper named by its key, with the reason it is listed.

    The key is the section's number with a section sign, or, for a section
    with no number, its heading without the bracketed status tag, so an edit
    to the tag or the title does not orphan the entry.
    """

    section: str
    why: str


# Sections that carry figures and no source line by design. **Checked against
# every live report rather than a named one**, so a figure no live report
# carries still fails; what the entry excuses is the missing line, not the
# figures. Not every section without a source line belongs here: one that
# quotes figures and names nothing is a finding, and the figure pass gives it a
# source line rather than an entry.
PAPER_NO_SOURCE_SECTIONS: tuple[PaperSection, ...] = (
    PaperSection(
        "Abstract",
        "its figures summarize the body, whose sections name their sources, so"
        " it is read against every live report rather than a named one",
    ),
    PaperSection(
        "§1.2",
        "the three claims, each pointing at the section that carries its"
        " measurement. Its figures summarize the body the same way the"
        " abstract's do, and are read against every live report",
    ),
)

# Sections excluded from the check by name. **Excluded and reported**, never
# skipped in silence: a section that says it is not part of the paper is still
# in `docs/paper.md` and ships with it, so the report prints it with its figure
# count, and an entry naming a section the paper no longer has fails. **Empty
# since the figure pass of 25 September**, which moved the one such section,
# the drafting notes, out of the paper and into the notes folder outside the
# repository; every section the paper has is checked.
PAPER_EXCLUDED_SECTIONS: tuple[PaperSection, ...] = ()


class PaperFigure(NamedTuple):
    """One figure in one section, with the reason it stands as it does.

    For a computed figure, ``arithmetic`` is the expression the paper's figure
    is computed by, over numbers, and ``operands`` names the numbers in it that
    are quoted from the section's own sources. ``scripts/check_paper.py``
    verifies both: each operand has to be in the expression and in a cited
    source, and the value has to round to the figure. A number in the
    expression that is not an operand is a constant of the arithmetic, and the
    entry's ``why`` says where it comes from.
    """

    section: str
    figure: str
    why: str
    arithmetic: str = ""
    operands: tuple[str, ...] = ()


# One reason for the twenty-three keys of §7.1.1, which quote one record.
_PREREGISTERED = (
    "Experiment 2 or 3 of 20 September, run under the parameters then in force"
    " and not repeated under the parameters that ship, so the figure is that"
    " day's and matches only the dated record the experiments wrote. The"
    " subsection's heading, its first paragraph and the sentence quoting the"
    " figure all say so"
)

# Figures that match only in a frozen dated record and stand anyway, each with
# its section and the reason it is quoted as history: a figure that matches
# only in a record frozen before the model moved was true of that model and
# not of this one, so it is stale until the entry says why the paper quotes
# the past. `figure` is written as the paper writes it, sign included with a
# minus as a hyphen, so one entry covers a figure the section quotes more than
# once the same way and never a magnitude and a signed bound that share their
# digits. Transcribed on
# 25 September from the figure pass's report and checked against the checker's
# own verdicts, which reach every entry; an entry nothing reaches fails.
PAPER_HISTORICAL_FIGURES: tuple[PaperFigure, ...] = (
    PaperFigure(
        "§6.4",
        "10.0",
        "the saturation rung's oracle ceiling on December as measured on 3 and"
        " 4 September under the parameters then in force, from the entry of"
        " 4 September in `docs/plan.md`: the measurement Gate 3 was stopped on."
        " The sentence dates it, and the rung's ceiling under the parameters"
        " that ship is given two sentences later from `reports/congestion.md`",
    ),
    PaperFigure(
        "§6.4",
        "8.5",
        "the same rung's ceiling on July on the same day, from the same entry;"
        " dated in the same sentence",
    ),
    PaperFigure(
        "§7.4",
        "1.29",
        "the route-length check's ratio before 19 September, with the encoding"
        " fitted on the unflagged training rows, from the dated record of"
        " 10 September. The sentence dates it and gives beside it the reading"
        " since then, 1.56, from the live ladder report",
    ),
    *(
        PaperFigure("§7.1.1", figure, _PREREGISTERED)
        for figure in (
            "0.40",
            "0.30",
            "1.15",
            "0.77",
            "-0.40",
            "0.34",
            "-1.6",
            "+0.9",
            "+0.77",
            "0.23",
            "-0.2",
            "+2.3",
            "+0.10",
            "-0.4",
            "+0.6",
            "-0.84",
            "-1.5",
            "-1.61",
            "1.02",
            "224.8",
            "224.7",
            "+0.37",
            "0.19",
        )
    ),
)

# The ladder steps the paper states as differences, each computed from two
# ex-flagged RMSE figures in `reports/baseline.md`, *The ablation ladder*. The
# four in §1.2 repeat four of §6.3's, and §1.2 is read against every live
# report, so the operands are found there too.
_PLAN_LAG_DECEMBER = (
    "the plan-lag block's step on the December split: `identifiers + time +"
    " geometry + aircraft` to `+ plan lags`, ex-flagged"
)
_PLAN_LAG_JULY = "the same step on the July split, ex-flagged"
_EXCESS_DECEMBER = (
    "what every congestion and weather block is worth net of the plan lags on"
    " the December split: `+ plan lags` to `+ weather`, ex-flagged"
)
_EXCESS_JULY = "the same, on the July split"

# Figures the paper computes from report figures rather than quotes, a
# difference stated in words, each with the arithmetic the checker verifies.
# Transcribed on 25 September from the figure pass's report. 5.8 also equals
# an interval bound in *Held-out pairs*, which §6.3 cites, by chance; the
# entry stands first so the check reads the figure as what it is, and the
# report prints the chance match beside it.
PAPER_FIGURE_EXEMPTIONS: tuple[PaperFigure, ...] = (
    PaperFigure("§1.2", "79.3", _PLAN_LAG_DECEMBER, "320.2 - 240.9", ("320.2", "240.9")),
    PaperFigure("§1.2", "57.4", _PLAN_LAG_JULY, "407.3 - 349.9", ("407.3", "349.9")),
    PaperFigure("§1.2", "7.7", _EXCESS_DECEMBER, "240.9 - 233.2", ("240.9", "233.2")),
    PaperFigure("§1.2", "5.8", _EXCESS_JULY, "349.9 - 344.1", ("349.9", "344.1")),
    PaperFigure("§6.3", "79.3", _PLAN_LAG_DECEMBER, "320.2 - 240.9", ("320.2", "240.9")),
    PaperFigure("§6.3", "57.4", _PLAN_LAG_JULY, "407.3 - 349.9", ("407.3", "349.9")),
    PaperFigure(
        "§6.3",
        "59.3",
        "the plan-lag block measured at the `+ weather` rung by removing it:"
        " `+ weather (without plan lags)` against `+ weather`, December,"
        " ex-flagged",
        "292.5 - 233.2",
        ("292.5", "233.2"),
    ),
    PaperFigure(
        "§6.3",
        "51.7",
        "the same measurement on the July split",
        "395.8 - 344.1",
        ("395.8", "344.1"),
    ),
    PaperFigure("§6.3", "7.7", _EXCESS_DECEMBER, "240.9 - 233.2", ("240.9", "233.2")),
    PaperFigure("§6.3", "5.8", _EXCESS_JULY, "349.9 - 344.1", ("349.9", "344.1")),
    PaperFigure(
        "§7.1.2",
        "0.27",
        "the winner's curse of the sweep, an expectation and not a measurement,"
        " as the sentence says: the standard deviation one seed carries, the"
        " floor of 0.19 s divided by root two, times the expected value of the"
        " largest of twenty-seven independent standard normal draws, 1.998,"
        " which is the integral of x n phi(x) Phi(x) to the power n minus one"
        " over the real line at n equal to 27, computed by numerical"
        " integration and reproducible by anyone. The expectation is the"
        " largest the shortfall can be, reached when every cell sits at one"
        " level, and the sentence says that too. 0.19 is quoted from"
        " `src/taxiout/models.py`; 1.998 is a constant of the arithmetic",
        "0.19 / 2 ** (1 / 2) * 1.998",
        ("0.19",),
    ),
)


class DeclaredInput(NamedTuple):
    """One row of a report's footer: the path, its digest, and which digest.

    ``by_elements`` is true where the row carries ``ELEMENTS_MARK``, so the
    digest is ``elements_digest``'s. A footer written before that mark existed
    carries the byte digest of every input, cached answers included, and reads
    with ``by_elements`` false: identical bytes are identical elements, so the
    older form verifies the file it was taken from and a footer from either era
    reads.
    """

    path: str
    digest: str
    by_elements: bool = False


def parse_report_inputs(text: str) -> tuple[DeclaredInput, ...]:
    """The inputs a report's footer declares; empty if it has none."""
    import re

    if REPORT_INPUTS_HEADING not in text:
        return ()
    body = text[text.index(REPORT_INPUTS_HEADING) :]
    mark = re.escape(ELEMENTS_MARK)
    row = rf"^\| `([^`]+)`( {mark})? \| `([0-9a-f]{{64}}|absent)` \|$"
    return tuple(
        DeclaredInput(m.group(1), m.group(3), bool(m.group(2)))
        for m in re.finditer(row, body, re.M)
    )


def challenge_inputs(data):
    """The challenge files a generator reading the movements actually read.

    The twelve monthly movement files and the ranking file, sorted, so the
    footer names each rather than the directory: a directory has no digest, and
    "the data under `data/challenge/`" is the declaration this check exists to
    replace.
    """
    from pathlib import Path

    data = Path(data)
    return sorted((data / "movements").glob("*.parquet")) + [
        p for p in (data / "ranking.parquet", data / "submitting.parquet") if p.exists()
    ]
