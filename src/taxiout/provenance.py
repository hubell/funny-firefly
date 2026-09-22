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
            "line 49: OSM ways are traced to the marked ends and OurAirports"
            " publishes the declared distance, so a few percent is normal and"
            " ten is not",
        ),
        (
            "HEADING_TOL_DEG",
            (15.0,),
            CHOSEN,
            "line 52: magnetic variation across these airports reaches about"
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
            NO_BASIS + ": politeness delay between archive requests. Nothing"
            " records why two seconds, where `taxiout.osm.POLITE_DELAY_S` does"
            " for its own",
        ),
        ("RETRIES", (4,), CHOSEN, NO_BASIS + ": how many times a request is retried"),
        ("TIMEOUT_S", (120,), CHOSEN, NO_BASIS + ": the request timeout"),
    ),
    *_mod(
        "scripts/fetch_ourairports.py",
        (
            "PAUSE_S",
            (2.0,),
            CHOSEN,
            NO_BASIS + ": politeness delay between requests, as in `fetch_metar.py`",
        ),
        ("RETRIES", (3,), CHOSEN, NO_BASIS + ": how many times a request is retried"),
        ("TIMEOUT_S", (120,), CHOSEN, NO_BASIS + ": the request timeout"),
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
            "`reports/baseline.md` line 234, quoted in the pre-registration of"
            " 20 September 2026. Regenerable only by the ladder",
        ),
        (
            "REF_JULY_THREE",
            (237.8,),
            QUOTED,
            "`reports/baseline.md` line 894, same pre-registration",
        ),
        (
            "REF_RANKING_MIX",
            (218.6,),
            QUOTED,
            "`reports/baseline.md` line 481, same pre-registration",
        ),
        (
            "REF_JULY_ALL",
            (325.8,),
            QUOTED,
            "`reports/baseline.md` line 270, same pre-registration",
        ),
        (
            "FLOOR_DECEMBER_ALL",
            (0.34,),
            QUOTED,
            "`reports/baseline.md` line 1061, the seed floor on that leg",
        ),
        (
            "FLOOR_JULY_ALL",
            (0.23,),
            QUOTED,
            "`reports/baseline.md` line 1062, the seed floor on that leg",
        ),
        (
            "FLOOR_JULY_THREE",
            (1.02,),
            QUOTED,
            "`reports/baseline.md` line 1063, the seed floor on that leg",
        ),
        (
            "JULY_SINGLES_POOLED",
            (325.81, 326.23),
            QUOTED,
            "`reports/baseline.md` line 1023: the span the five July single-seed"
            " fits must reproduce before any experiment is read",
        ),
        (
            "JULY_SINGLES_THREE",
            (237.02, 238.83),
            QUOTED,
            "`reports/baseline.md` lines 1062 to 1063, the same reproduction"
            " gate on the three airports",
        ),
        (
            "JULY_AVERAGE_POOLED",
            (325.29,),
            QUOTED,
            "`reports/baseline.md` line 1023, the five-seed average the"
            " reproduction gate requires",
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
            "line 72's own comment: the PRU reference's own floor for a pair",
        ),
        (
            "MIN_CLASS_ROWS",
            (20,),
            CHOSEN,
            "line 73's comment states the rule: per (pair, wake class) for the"
            " paired gap",
        ),
        (
            "MIN_DESIGNATOR_PAIRS",
            (5,),
            CHOSEN,
            NO_BASIS + ": the floor on pairs per runway designator. The"
            " constants either side of it carry inline comments and this one"
            " does not",
        ),
        (
            "DIRECT_SNAP_M",
            (60.0,),
            CHOSEN,
            NO_BASIS + " here: it mirrors `taxiout.graph.DIRECT_SNAP_M`, which"
            " has a basis, and this copy cites nothing",
        ),
        (
            "USABLE_ENTRY_SHARE",
            (0.6,),
            CHOSEN,
            "line 76's comment: a junction in the first 60% of the runway is a"
            " usable entry",
        ),
        (
            "MIN_ENTRY_OFFSET_M",
            (50.0,),
            CHOSEN,
            "line 77's comment: nearer than this to the threshold is the threshold",
        ),
        ("HOUR", (3600.0,), MECHANISM, "seconds in an hour"),
        (
            "MIN_BETWEEN_FRACTION",
            (0.05,),
            CHOSEN,
            "line 80: `taxiout.encoding`'s floor reused, and it says so",
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
            "a slice of `FREQUENCY_BUCKETS`; line 89's inline comment reads"
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
            (17.6,),
            QUOTED,
            "`reports/baseline.md` line 978. Cross-checked in the generator"
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
            "line 102: coarser than `BAND_EDGES_MIN` because a sum of squares"
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
            "line 136: five-minute bands of the gap, with the two wide ones the"
            " review used at the top, because past an hour the bound is not a"
            " bound and the bands only need to show that",
        ),
        (
            "PRESSURE_MIN",
            (20,),
            CHOSEN,
            "line 158: below twenty minutes the bound holds essentially always,"
            " so the share under it summarises stand pressure",
        ),
        (
            "FLOOR_CAPS_MIN",
            (20, 30, 45, 60),
            CHOSEN,
            "line 162: the caps the hard floor is tried at, on the reasoning"
            " that the bound is sound at short gaps and collapses at long ones",
        ),
    ),
    *_mod(
        "scripts/report_residual.py",
        (
            "TAIL_CUTS_S",
            (600.0, 1200.0),
            CHOSEN,
            "line 111 names them as where the tail is cut, in seconds of"
            " absolute residual",
        ),
        (
            "REOCCUPIED_BAND_MIN",
            (20.0, 45.0),
            CHOSEN,
            "line 114: the band where the stand-occupancy bound is the"
            " mechanism, the same band `reports/baseline.md` decomposes on",
        ),
        (
            "LAG_EDGES_MIN",
            (0, 15, 30, 45, 60, 120),
            CHOSEN,
            "line 118: plan-lag bands in minutes, the last open and nulls their own row",
        ),
        (
            "LARGEST_ROWS",
            (5,),
            CHOSEN,
            "line 122: how many of the largest rows to print per leg",
        ),
        (
            "WORST_QUANTILE",
            (0.99,),
            CHOSEN,
            "line 125: the share of absolute residual that counts as the worst"
            " one percent",
        ),
        (
            "PEAK_HOURS",
            (6,),
            CHOSEN,
            NO_BASIS + ": how many take-off hours the section names. Line 140"
            " says both figures are read off the leg's own squared error rather"
            " than fixed in advance, which is how they are used and not why six",
        ),
        (
            "NAMED_PREFIXES",
            (10,),
            CHOSEN,
            NO_BASIS + ": how many callsign prefixes are named before pooling"
            " the rest, for the same reason as `PEAK_HOURS` above",
        ),
        (
            "MIN_INTERVAL_DAYS",
            (5,),
            CHOSEN,
            "line 150: below this many distinct take-off days a day-clustered"
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
            NO_BASIS + ": the minimum pair count the stand-in physical model"
            " uses. Nothing in the script records why five, though"
            " `taxiout.tails.MIN_COUNT` is also five and has a basis",
        ),
        (
            "PREFIX_ROWS",
            (15,),
            CHOSEN,
            "line 78: enough to show the shape of the claim, with the sentence"
            " under each table computed over every row",
        ),
    ),
    *_mod(
        "scripts/review_challenge_data.py",
        ("HOUR", (3600,), MECHANISM, "seconds in an hour"),
        ("DAY", (86400,), MECHANISM, "seconds in a day"),
        (
            "DISRUPTION_MIN_ROWS",
            (20,),
            CHOSEN,
            "line 556 states the rule both sides of the data are read at, so the"
            " two tables can be set against each other",
        ),
        (
            "DISRUPTION_RATE_MULTIPLE",
            (10,),
            CHOSEN,
            "line 556, covering this too: a share this many times the airport's"
            " own training-year rate before the day is called out",
        ),
    ),
    *_mod(
        "scripts/run_baselines.py",
        (
            "FAMILYWISE_ALPHA",
            (0.05,),
            CHOSEN,
            "lines 1174 to 1185: three shots at a 5% test is a 14% chance of one"
            " clearing zero by luck, and Bonferroni is the crude correction"
            " because it assumes nothing about how the arms' errors are related",
        ),
        (
            "K_SENSITIVITY_SCALES",
            (0.5, 2.0),
            CHOSEN,
            NO_BASIS + ": the factors the encoding's shrinkage is re-read at."
            " The comment above belongs to `FAMILYWISE_ALPHA` and covers"
            " Bonferroni, not this",
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
            " of it, reproducing the July-three figure to 0.05 s",
        ),
        (
            "HEADROOM_TRIGGER_S",
            (5.0,),
            CHOSEN,
            "a line on a DIAGNOSTIC and not a leakage criterion. Lines 2521 to"
            " 2526 give the reason for an absolute figure and"
            " `docs/prior_art.md` records that absolute thresholds transfer"
            " badly between targets. An absolute line on a quantity that scales"
            " with the measured side is why this went from missing to firing"
            " when the booster's capacity grew 3.2 times, with the block"
            " untouched: the readings are 5.56, 4.971, 5.37 and 7.301 s across"
            " four regenerations. `docs/review_headroom.md` §7 of 8 September,"
            " before it fired, records that a residual fit bounds a refit from"
            " below and the gap is interaction value rather than evidence of"
            " anything wrong",
        ),
        (
            "HEADROOM_NEAR_MISS_S",
            (1.0,),
            CHOSEN,
            "lines 2529 to 2533: a threshold test reported only as pass or fail"
            " hides the case that missed by a hundredth, and this report already"
            " has a subsection about a rule that turned on nine thousandths",
        ),
        (
            "HEADROOM_READINGS",
            (15.86, 5.56, 2, 15.27, 4.971, 3, 15.67, 5.37, 2),
            QUOTED,
            "the overshoots this test read on earlier regenerations, quoted from"
            " the reports committed at `ad1eecf`, `ff59892` and `067c4af`."
            " INCOMPLETE: `reports/migration-2026-09-10.md` lines 536 to 541"
            " record a fourth, 19 September, step +14.64 s and overshoot"
            " 4.345 s, which missed. See `PROVENANCE_PENDING`",
        ),
        (
            "HEADROOM_UNCOMMITTED_S",
            (15.43,),
            QUOTED,
            "lines 2576 to 2580: the first run on the reference machine,"
            " 10 September, under the booster's earlier setting and never"
            " committed. The migration record gives its gain and nothing finer",
        ),
        (
            "HEADROOM_FLOOR_BEFORE_S",
            (1.28,),
            QUOTED,
            "lines 2582 to 2585: the seed floor on the July three under the"
            " reference configuration, from the migration record's noise-floor"
            " table of 11 September",
        ),
        (
            "EXPECTED_DID_S",
            (1.0,),
            CHOSEN,
            "lines 2670 to 2677: none of the four blocks is a property of the"
            " (airport, stand, runway) pair, so each should be within about a"
            " second of zero on the held-out-pair contrast. Not a gate, and the"
            " section says so",
        ),
        (
            "PRINTED_PRECISION_S",
            (0.1,),
            CHOSEN,
            "lines 2686 to 2700: two rungs printing the same number are a tie,"
            " and a rule that ranks them is ranking the random-number generator."
            " The comment also records that adopting it there was decided after"
            " seeing the number it turns on, and that `_tolerance_audit` prints"
            " every other margin so a reader can check it moves one step only",
        ),
        (
            "GAIN_BAND_EDGES_MIN",
            (0, 20, 30, 45, 60, 120),
            CHOSEN,
            "lines 2751 to 2755: wider than the five-minute bands"
            " `reports/occupancy.md` uses, because the question here is where"
            " the gain lands and a band must hold enough rows for a sum of"
            " squares to mean anything",
        ),
        (
            "PER_AIRPORT_INTERVAL_S",
            (2.0,),
            CHOSEN,
            "lines 2846 to 2848: below this a per-airport step is printed but"
            " not bracketed, a thousand draws on a difference of a second or two"
            " being a thousand draws spent on noise",
        ),
        (
            "BAGGING_GAIN_S",
            (1.0,),
            CHOSEN,
            "lines 3962 to 3969: one second on a 230-second error is four tenths"
            " of a percent, about the run-to-run variation the ladder showed"
            " when the rule was set. The comment records that the basis no"
            " longer holds and that the rule was retired on 19 September",
        ),
        ("OCTOBER_TRAIN", (1, 10), MECHANISM, "a month range, not a quantity"),
        (
            "BAGGING_BEFORE_TAIL_GAINS_S",
            (1.1, 1.37),
            QUOTED,
            "lines 4010 to 4014: the same measurement on this rung before the"
            " recorded-taxi tail came out of the fit, when it cleared the"
            " threshold. The run that made it fitted rows this report no longer"
            " fits, under the booster's earlier setting",
        ),
        (
            "BAGGING_EARLIER_GAINS_S",
            (0.78, 0.98),
            QUOTED,
            "lines 4018 to 4024: the same measurement on the rung below, which"
            " did not clear the threshold. Recorded in `docs/plan.md` and"
            " `taxiout.models`; recomputing it would cost ten fits to reproduce"
            " a number already on the record",
        ),
        (
            "TAIL_EXCLUSION_H",
            (4.0, 2.0),
            CHOSEN,
            "lines 4179 to 4190: both thresholds fixed before the run and both"
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
            "lines 146 to 151: the widest tree at the smallest leaf floor, which"
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
            f"{NO_BASIS}: the take-off gap buckets the lag-1 autocorrelation is"
            " read in. The module docstring describes them and nothing records"
            " why these edges",
        ),
        (
            "HALF_WINDOW_S",
            (15, 60),
            CHOSEN,
            f"{NO_BASIS}: the oracle regression's window half-width. The"
            " docstring states the fifteen minutes as a fact about the"
            " instrument; no basis for the width is recorded",
        ),
        (
            "MIN_ROWS",
            (30,),
            CHOSEN,
            f"{NO_BASIS}: below this a bucket's correlation is not reported."
            " Nothing records why thirty",
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
            " reading may be. The comment above covers the window and the floor"
            " and not this",
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
            f"{NO_BASIS}: the shared-corridor block's trailing windows. They"
            " match `congestion.WINDOWS_MIN`'s first two, which is probably the"
            " reason, and probably is not a basis",
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
            "line 168: five leaves each row's encoding built from four fifths of"
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
            "line 180 names it the pre-registered threshold for the"
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
            " minutes. Nothing in the module records why",
        ),
        (
            "LAG_HI_S",
            (2, 3600.0),
            CHOSEN,
            f"{NO_BASIS}: the plausibility ceiling on a plan lag, two hours."
            " Nothing records why, and `taxiout.tails.LONG_S` is the same two"
            " hours with a basis, which this could have cited",
        ),
        (
            "TAXI_IN_LO_S",
            (60.0,),
            CHOSEN,
            f"{NO_BASIS}: the plausibility floor on an arrival's taxi-in",
        ),
        (
            "TAXI_IN_HI_S",
            (3600.0,),
            CHOSEN,
            f"{NO_BASIS}: the plausibility ceiling on an arrival's taxi-in",
        ),
        (
            "ARRIVAL_LATE_S",
            (2, 3600.0),
            CHOSEN,
            f"{NO_BASIS}: the plausibility bound on an arrival's lateness",
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
            (3000, 0.015, 255, 20, 0.9, 0.8, 1, 64, 20.0, 20260902, 8, 1),
            CHOSEN,
            "the second permitted sweep's cell, 21 September 2026, scored 214.90 s"
            " on October against 217.28 s for the cell it replaced, with a seed"
            " floor of 0.19 s. Lines 147 to 193 carry both sweeps' terms and the"
            " reason their October figures are not comparable;"
            " `scripts/sweep_params.py` is the instrument",
        ),
        (
            "BAGGING_SEEDS",
            (5,),
            CHOSEN,
            "lines 265 to 301: five fits of one rung with nothing changed but the"
            " seed, measured three times against a rule fixed before the first"
            " of them",
        ),
        (
            "TAIL_EXCLUSION_ADOPTED_H",
            (4.0,),
            CHOSEN,
            "line 755: fixed at four hours on the mechanism and not by a sweep,"
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
            "line 47: the Overpass free tier allows about two concurrent slots"
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
            f"{NO_BASIS}: the half-width of the runway plan-lag window. Nothing"
            " in the module records why fifteen minutes",
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
            f"{NO_BASIS}: the adjacency tolerance for joining a displaced"
            " threshold to its runway. The comment above gives the rule and not"
            " this distance",
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
            f"{NO_BASIS}: the trailing windows. The docstring states them;"
            " nothing records why, though they match"
            " `congestion.WINDOWS_MIN`'s first two",
        ),
        (
            "MIN_GAP_WINDOW_MIN",
            (5,),
            CHOSEN,
            f"{NO_BASIS}: the window the smallest consecutive gap is read over",
        ),
        (
            "SPAN_GAPS",
            (3,),
            CHOSEN,
            f"{NO_BASIS}: how many gaps the span reading covers",
        ),
        (
            "SPAN_CAP_S",
            (1800.0,),
            CHOSEN,
            f"{NO_BASIS}: the thirty-minute cap on the span. The docstring"
            " states the cap and that a short span sits at it; no basis for the"
            " value is recorded",
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
            f"{NO_BASIS}: how many folds the held-out-pair contrast uses."
            " Nothing records why three",
        ),
        (
            "SEEN_FRACTION",
            (0.05,),
            CHOSEN,
            f"{NO_BASIS}: the share of rows sampled as the matched seen-pair"
            " evaluation set. Nothing records why five percent",
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
            f"{NO_BASIS}: the interval's coverage. The conventional 95%, and the"
            " convention is not recorded as the reason",
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


class Pending(NamedTuple):
    """One excursion not yet resolved, with a reason and an owning block.

    The same instrument as ``taxiout.models.SEED_AVERAGE_PENDING``: **emptied
    by the work and never by editing the tuple**, and empty is the freeze
    condition. ``scripts/check_provenance.py --strict`` fails while it is not
    empty, and the ordinary run prints it and passes, because a gate that fails
    today cannot be wired into a commit path.
    """

    what: str
    why: str
    owner: str


PROVENANCE_PENDING: tuple[Pending, ...] = (
    Pending(
        "chosen constants whose basis is recorded nowhere in the code",
        "each is a real choice, each is marked "
        + repr(NO_BASIS)
        + " in its own entry, and the count is summed from `CONSTANTS` rather"
        " than written here so the two cannot disagree. Writing a basis is"
        " cheap; inventing one is not available, so the entries say what the"
        " constant does and that the reason is unrecorded",
        "a basis pass before the freeze",
    ),
    Pending(
        "`docs/paper.md`'s figures match no report in the repository",
        "it was last touched at `9424d7b` on 20 September, before the wider"
        " sweep of 21 September and before the regenerations of 22 September, so"
        " essentially every model figure in it is stale. Each section already"
        " carries a `Source:` line, which is what a checker would read",
        "Block 10, the figure pass, building the paper's own checker as part of it",
    ),
    Pending(
        "a five-seed submission has never been timed under the current cell",
        "the recommendation to set `SEED_AVERAGE_ADOPTED` rests on an estimate"
        " reached by scaling Block 07's leg fits, and an estimate that supports"
        " a recommendation is measured before the recommendation is acted on."
        " `make submission` is the target that settles it",
        "a timing run before the freeze",
    ),
    Pending(
        "`HEADROOM_READINGS` is missing its fourth reading",
        "`reports/migration-2026-09-10.md` lines 536 to 541 record four"
        " regenerations and state that `reports/baseline.md` prints that"
        " history; the constant carries three, and 4.345 appears nowhere in"
        " that report. Restoring it regenerates a paragraph of `baseline.md`,"
        " and `scripts/run_baselines.py` has no report-only mode and keeps no"
        " results, so the file can only be rebuilt by the whole ladder",
        "the next block that spends a ladder run, which Block 11's pinned"
        " rebuild does anyway",
    ),
    Pending(
        "`reports/baseline.md`'s section still carries the name *The leakage"
        " audit, applied*",
        "the decision of 22 September renames it to what it measures, the"
        " headroom overshoot, leaving the name *leakage audit* to"
        " `reports/congestion.md`'s registry assertion, which is a leakage test"
        " and which passes. No threshold, constant or verdict moves. The"
        " generator prose is in `scripts/run_baselines.py` and the report cannot"
        " be rebuilt without the ladder, so the rename lands with the run that"
        " regenerates it",
        "the next block that spends a ladder run",
    ),
    Pending(
        "reports with no declared inputs",
        "`report_inputs_footer` and check 8 are in place and"
        " `REPORT_INPUTS_DECLARED` names the generators carrying a footer. The"
        " rest are generators whose reports cost between forty minutes and"
        " seven hours to rebuild, and a footer that is not in the file it"
        " describes is worth nothing, so they land with the run that"
        " regenerates them",
        "Block 11's pinned rebuild, which regenerates every report",
    ),
    Pending(
        "cross-report citation is unchecked",
        "141 citations across 75 ordered pairs, and nothing ties a quoted figure"
        " to the file it came from. This is the class that bit: the oracle"
        " ceiling at Heathrow on July moved 9.9 to 9.5 when"
        " `reports/congestion.md` was regenerated, and 9.9 was quoted in the"
        " paper drafts. A checker that passed because it could not see would be"
        " worse than none",
        "`code_09e`",
    ),
    Pending(
        "external snapshots and their pinning are unscanned",
        "`scripts/fetch_osm.py` takes a `--date` and the OSM cache is from"
        " 1 September. Whether the pins are recorded and checked is a question"
        " about a rebuild rather than about the tree, and this block registers"
        " that it did not look rather than asserting either way",
        "Block 11's pinned rebuild",
    ),
)


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
        " Block 10's and is on the pending list; the rest carries no model"
        " figure",
    ),
    (
        "cross-report citation",
        "NOT covered. On the pending list, owned by `code_09e`",
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

# Generators that carry the footer. The rest are on ``PROVENANCE_PENDING``:
# a footer is only worth anything in the file it describes, and those reports
# cost between forty minutes and seven hours to rebuild.
REPORT_INPUTS_DECLARED: tuple[tuple[str, str], ...] = (
    ("scripts/report_headroom.py", "reports/headroom.md"),
    ("scripts/report_occupancy.py", "reports/occupancy.md"),
    ("scripts/report_residual.py", "reports/residual.md"),
    ("scripts/report_tails.py", "reports/tails.md"),
)


def digest(path) -> str:
    """The sha256 of one file, or ``"absent"``."""
    import hashlib
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def report_inputs_footer(repo, paths) -> list[str]:
    """The footer naming every input a report was generated from, with digests.

    **Content and not modification time, and the distinction is the whole
    point.** Block 11 reproduces from a clean checkout, where every file
    carries the checkout time, so an mtime ordering is meaningless exactly
    where the check matters most. A digest is not.

    An input under a gitignored directory is the ordinary case on a clean
    checkout, and it reads ``absent`` rather than being left out: a footer that
    omitted what it could not find would turn an unverifiable report into one
    that looks verified.
    """
    from pathlib import Path

    rows = []
    for p in paths:
        rel = Path(p).resolve().relative_to(Path(repo).resolve()).as_posix()
        rows.append(f"| `{rel}` | `{digest(p)}` |")
    return [
        REPORT_INPUTS_HEADING,
        "",
        "Every path this run read, with the sha256 of each as it was read.",
        "`scripts/check_provenance.py` recomputes them and reports this report as",
        "verified, stale or unverifiable; a digest reading `absent` is the",
        "ordinary case on a clean checkout, where the data this repository does",
        "not redistribute is not present.",
        "",
        "| input | sha256 |",
        "| --- | --- |",
        *rows,
    ]


def parse_report_inputs(text: str) -> tuple[tuple[str, str], ...]:
    """The (path, digest) pairs a report's footer declares; empty if it has none."""
    import re

    if REPORT_INPUTS_HEADING not in text:
        return ()
    body = text[text.index(REPORT_INPUTS_HEADING) :]
    return tuple(
        (m.group(1), m.group(2))
        for m in re.finditer(r"^\| `([^`]+)` \| `([0-9a-f]{64}|absent)` \|$", body, re.M)
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
