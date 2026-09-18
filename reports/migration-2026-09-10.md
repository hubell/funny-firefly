# Migration record, 2026-09-10

## L8b criterion, set before the run

Written 2026-09-10T15:39:22Z, before `make baseline` ran on this machine.

`reports/baseline.md`, `reports/residual.md` and `reports/occupancy.md` pass
only if they regenerate byte-identical to the committed versions: every
printed figure unchanged at the precision the reports print, and the
seed-averaging reproducibility check in `baseline.md` still holding.

Below printed precision, the regenerated `data/processed/residuals/` files
are compared with the verified copy and the largest difference is recorded.
That is a measurement, not part of the gate.

If a printed figure changes, the criterion is not relaxed. The outcomes are
read as follows. Identical: the migration passes. Only final printed digits
move, with sub-precision differences small and rung order unchanged: recorded
as measured platform variation, and the original tree remains the reference
until the reference platform is chosen deliberately. Anything larger:
diagnosed in order, from package versions to LightGBM's histogram strategy
(row-wise or column-wise, chosen by timing) to the linear algebra backend.

Not regenerated: `make transfer`, for run time, and `make sweep`, whose
output is gitignored.

Data (c) OpenStreetMap contributors, ODbL 1.0.

## Where the ladder stops repeating, 2026-09-11

Measured with `scripts/check_determinism.py` on one split and one rung,
December's `identifiers`, fitted repeatedly from a cached feature frame
instead of by refitting the ladder: on 11 September, five builds of the frame
and 51 fits in 17 processes, run one after another. Every figure below is in
[`determinism.md`](determinism.md), which the script generates (regenerated
on 18 September with the container run and the two scaling runs, eight fits in
three processes, and three further builds of the frame, it now counts eight
builds and 59 fits in 20 processes), except the reproducibility column quoted
next, which is read from the committed `baseline.md` and from run 1's copy of
it, kept outside the repository.

**The report's own check already failed here.** The seed-averaging section's
reproducibility column reads `0.00e+00` on both splits in the committed
`baseline.md` and `4.55e-13` (December) and `2.27e-13` (July) in run 1's. That
check refits in the same process, on the same frame and rows, so the defect
was in the booster before any harness looked.

**The feature frame repeats.** Two builds in separate processes, each under a
random string-hash seed, agree on all 175 digests: every feature column, the
record-quality term, the fit rows, the shrunk encoding and its prior, both
geometric fits and the rung's design. Under `OPENBLAS_NUM_THREADS=1` exactly
one changes: the `geometric-class` rung's predictions, where
`np.linalg.lstsq` runs in `taxiout.unimpeded`, by 2.90e-08 s at most, which
moves that rung's December RMSE by 8.70e-11 s. Two builds at one BLAS thread
agree with each other, as three at the default of eight did on 11 September,
so the difference repeats at a fixed thread count and reaches no booster
input. Since the regeneration the report has five builds at the default, all
identical, and `ctr1`, built in the container with the environment pinned
(OMP 8, OpenBLAS 8, kernel SkylakeX, hashseed 0), identical to `a` on every
component.

**The booster does not repeat, and the thread reductions are why.** Under the
current parameters every fit was a new outcome: fifteen fits at 8 and 14
threads gave fifteen different prediction arrays, up to 75.7 s apart on a
flight, with RMSE between 442.82 and 443.79 s. LightGBM chose row-wise
histograms in all six fits that logged the choice, so the trees moved with
the strategy fixed. Forcing a strategy alone changed nothing (three outcomes
in three fits, each way). `deterministic` alone gave two outcomes in three,
3.29e-09 s apart -- exactly the column-wise against row-wise difference, so
there the timing test that picks the strategy flipped between fits. Both
hypotheses hold: the OpenMP sums that `deterministic` makes serial (the L2
initial score and the per-leaf sums) are what moves whole trees, and the
strategy chosen by timing is a second, smaller source.

| configuration | 8 threads | 14 threads | 8 against 14 |
| --- | --- | --- | --- |
| current | 9 outcomes in 9 fits | 6 in 6 | -- |
| `deterministic`, `force_col_wise` | 1 in 6, two processes | 1 in 6, two processes | identical |
| `deterministic`, `force_row_wise` | 1 in 6, two processes | 1 in 6, two processes | 19.49 s apart |

The column-wise configuration also gives the same bits as the current
parameters at one thread, so on this rung its result does not depend on the
thread count. The row-wise one repeats only at a fixed count, because its
histograms are summed over row blocks sized by the thread count. **That
independence belongs to this rung and not to the configuration**, which the
Phase 4 section below measured afterwards: fitted at one thread instead of
eight, three of December's 25 rungs grow different trees, column-wise or not,
by 158.3 s, 40.3 s and 4.97 s on a flight. Which three, and why, is not
explained by anything measured. Every reported run uses eight.

**The cost** is the median fit time of this rung: the column-wise
configuration takes 30.4 s at 8 threads against 23.0 s for the current
parameters (+33%), and 52.8 s against 41.6 s at 14 (+27%). The row-wise one
takes 22.7 s at 8 (-1%). The ladder ran at 14 threads on 8 cores, and 30.4 s
at 8 threads is less than the 41.6 s the current parameters take there.

**The harness is the ladder's fit.** The current parameters at 14 threads land
at an RMSE of 443.44 to 443.79 s, against 443.43 s for the Mac run's stored
residuals and 443.24 s for run 1's -- among the ladder's own outcomes. None
matches either bit for bit, as none of the current parameters' fits matches
another.

**Not supported: a small number of discrete outcomes.** On this rung no two
of the fifteen current-parameter fits coincided. The nine files where run 2's
delta equals run 1's cannot be examined, because run 2's residuals were not
kept.

## The reference configuration, fixed before its runs, 2026-09-11

Written before any ladder figure under the new configuration exists.

**The configuration** is `deterministic=True` with `force_col_wise=True` and
`n_jobs` 8, in `BOOSTER_PARAMS`. It is chosen on one property measured above
-- on the rung measured it alone repeated at every thread count tried, and
gave the same bits at 1, 8 and 14 threads -- and on no figure it produces.
The row-wise configuration, the one alternative that also repeats, is run
once as well, so that what the choice between the two changes is on the
record: every printed figure that differs between them, and every conclusion
that does.

**The reproduction criterion.** Two consecutive runs of `make baseline`,
`residual`, `occupancy`, `congestion` and `transfer` under that
configuration. Every tracked report they write and every residual file is
byte-identical between the two, or the configuration fails. `congestion` and
`transfer` are added to the three named before: the first reads the residual
files and the second fits with `BOOSTER_PARAMS`, so both change with it. The
one line each report stamps with its UTC generation date comes from the clock
and not from the configuration: a report whose only differing line is that
date, as `scripts/compare_runs.py` labels it, counts as identical.

**The noise floor.** The seed-averaging section already refits the submission
rung under five seeds on each forward split. For each leg -- December all
ten, July all ten, July the three -- the standard deviation of the five
single-seed ex-flagged RMSEs is the seed-to-seed spread. A comparison between
two fits is below the floor when its change in ex-flagged RMSE is smaller than
the square root of two times that spread, because every such comparison is a
difference between two single-seed fits. Every rung read against its base and
each tail-exclusion arm read against the un-excluded fit is checked. The
held-out-pair folds are not, having no seed refits, and the spread is measured
on one rung and assumed for the others.

## The reference configuration, measured, 2026-09-11

**Two runs repeat byte for byte.** Runs A and B of `make baseline`,
`residual`, `occupancy`, `congestion` and `transfer`, one after the other under
the reference configuration, agree on every report and every residual file:
68 files byte-identical, none differing and none missing -- the five reports
the pipeline writes, the other reports and the figure beside them, and the 52
residual files -- in [`reproduction-2026-09-11.md`](reproduction-2026-09-11.md),
which `scripts/compare_runs.py` writes. Both runs are dated the same day, so no
report differed even in its generation date. Nothing that generates output
changed between the two runs: the edits made in between were to comments, one
constant nothing yet read, and the Makefile's help text, and that was checked
by comparing the code with a copy taken as run A started. The seed-averaging
check in `baseline.md`, which read `4.55e-13` on this machine before, reads
`0.00e+00` on both splits.

**The prose, corrected after the runs.** Reading the regenerated reports
against the committed ones found fixed text in `scripts/run_baselines.py` that
the refit had made wrong, and some that was wrong already. It is now computed
from the figures it describes, and three passages written against the noise
floor were approved before the last run. Run C, made with that code, differs
from run B in 57 lines of `baseline.md`, none of them a table row, and in
nothing else: the other four reports and all 52 residual files are
byte-identical, and the code it ran was checked unchanged from start to end.

**What the configuration moved.** Against the committed `baseline.md`, about
460 table cells changed: by at most 2.6 s in any printed RMSE (a held-out-pair
cell), 1.4 s in the ladder's own tables, and 0.1 s for the rung the submission
is fitted from, whose December figure goes from 218.0 to 218.1 s. The printed
conclusions that moved with them:

- The leakage audit on `+ stand occupancy` fires, repeatably: +15.67 s on
  July's three airports against an estimated +10.3 s, 0.37 s past the 5 s
  trigger and inside that leg's 1.28 s noise floor. The committed report had a
  near miss at +15.27 s, and run 1 fired at +15.43 s: under the old setting the
  verdict turned on which run was read. The block is kept, on a reading adopted
  after the number was seen, and the report says so.
- The encoding arm strongest on pairs seen once to ten times is now the one that
  replaces stand and runway, not the one that ships; the decomposition is now
  of the arm that ships, as its own comment always said.
- `+ configuration (without plan lags)` changes sign on December, -0.2 s to
  +0.18 s, below the 0.19 s floor. Nothing reads it.
- Orderings inside tables that no sentence relies on.

Seed averaging gains 0.86 and 0.54 s and is still not adopted.

**Column-wise against row-wise.** The row-wise configuration was run once,
from a copy of the tree taken as run A started and differing only in
`force_row_wise`, through `baseline`, `residual`, `occupancy` and
`congestion`. Against run B, both made by the same generator,
[`row-wise-2026-09-11.md`](row-wise-2026-09-11.md) records every file
differing: 27 of the 52 residual files in the last bits and 25 by whole trees,
the largest change in a pooled residual RMSE 0.31 s. Of 164 conclusions read in
the four reports, the choice changes no verdict the report acts on: the same
encoding arm ships, the Gate 2c chain passes, the leakage audit fires, seed
averaging is not adopted, the four-hour tail exclusion is, and saturation is
worth nothing. What it does change:

- which encoding arm is strongest on pairs seen once to ten times, and so which
  arm the section decomposed -- the one conclusion the prose relied on, and the
  reason the decomposition is now of the arm that ships, whichever that is;
- four steps of the chain kept without the plan lags, which cross the noise
  floor in one configuration and not the other (the corridor and configuration
  in December, weather and saturation in July, the last by 0.02 s), and two
  steps inside the floor that change sign;
- two per-airport intervals (the callsign prefix at Heathrow in December clears
  zero column-wise and not row-wise; the plan differences at Munich reach the
  2 s line that earns an interval row-wise and not column-wise), two signs in
  the pair-frequency buckets, and one ordering in `congestion.md`.

No sentence rests on any of these. The pooled ladder moves by at most 0.4 s and
a single bucket by 1.7 s. The choice was fixed on the thread-count
independence the measured rung showed, and on this evidence it decides nothing
a reader would act on.

**The noise floor.** The five seed refits `baseline.md` already made give, per
leg, the standard deviation of five single-seed ex-flagged RMSEs:

| leg | spread of five seeds (s) | standard deviation (s) | floor (s) |
| --- | ---: | ---: | ---: |
| December, all ten | 217.75 to 218.08 | 0.13 | 0.19 |
| July, all ten | 325.58 to 326.25 | 0.28 | 0.39 |
| July, the three | 235.94 to 238.09 | 0.91 | 1.28 |

Thirteen of the thirty-one covered comparisons are below the floor on at least
one leg. The conclusions that rest on one of them:

- the leakage audit's verdict on `+ stand occupancy` (0.37 s against 1.28 s);
- the low end of what the schedule-anchored block keeps, 5% on July's three (a
  gain of 0.75 s against 1.28 s), which cannot be told from zero;
- runway saturation "worth nothing": +0.29 s in July and +0.39 s on the three,
  inside the floor, and -0.21 s in December, at its edge;
- the Gate 2c chain's "no worse" on July's three for `+ callsign prefix`
  (-1.03 s) and `+ destination and holidays` (-0.45 s), carried on the other
  legs by gains clear of the floor;
- the congestion blocks read net of the plan lags: the shared corridor on every
  leg, demand on both all-ten legs, configuration and weather on July. The
  report draws no verdict on them, and any reading of them as worth something
  would rest on noise;
- the geometry contrast on July's three (-0.30 s), carried by December
  (-2.12 s) and July (-1.13 s).

Outside what the floor covers, the encoding arms tie on July (0.1 s) and are
chosen by mechanism, not score, which the report already says. The
held-out-pair contrasts have no seed refits and carry their own intervals.

**Run times, measured on run A.** `make baseline` 2 h 06 min at a peak of
8.1 GiB, `transfer` 14 min, `congestion` 1 min 42 s, `residual` 13 s and
`occupancy` 14 s, on eight cores; run B's `baseline` took 2 h 07 min. The
Makefile's help text says so.

**Not changed here.** `docs/paper.md`, `docs/plan.md` and
`docs/review_headroom.md` quote figures that moved (218.0 and 325.6 s, the
15.27 s near miss, 0.81 and 0.56 s). Sentences already wrong in the committed
`occupancy.md`, `congestion.md` and `transfer.md` are left to a separate
change.

## The reference container, 2026-09-12

**What was run.** Two full pipeline runs -- `baseline`, `residual`,
`occupancy`, `congestion`, `transfer` -- inside a container on the reference
machine, in a reflink copy of the tree at `067c4af`, with the repository's own
interpreter and installed wheels mounted read-only and
`OPENBLAS_CORETYPE=SkylakeX`, `OPENBLAS_NUM_THREADS=8`, `OMP_NUM_THREADS=8`,
`PYTHONHASHSEED=0`, `TZ=UTC` and `LC_ALL=C.UTF-8` set. The base is Debian
trixie, pinned by digest, whose glibc is 2.41 against the host's 2.44 and whose
`libgomp` is its own: `lib_lightgbm.so` takes the OpenMP runtime from the
system, and that runtime is what `deterministic` makes order its sums.

**The result.** The two container runs agree on all 69 files byte for byte, the
five reports and the 52 residual files among them. Against the host's run C,
all 52 residual files are byte-identical and each of the five reports differs
in one line: the date it was generated on, which comes from the clock.
[`container-2026-09-12.md`](container-2026-09-12.md) records both comparisons.
Two files there differ for reasons outside the container, named so the reader
need not wonder: this record itself, which gained its Phase 2 section between
run C's snapshot and the copy, and `synthetic_airport.svg`, which the container
snapshot did not copy and which neither run writes.

The cheaper check ran first and says the same thing about the machinery rather
than the reports: in the container the feature frame's 175 digests and the
booster's predictions on December's `identifiers` rung were bit-identical to
the host's, which is what made the full runs worth the five hours.

**Times.** The ladder took 7,758 s and 7,577 s in the container against
7,550 s on the host, so containerisation costs nothing measurable; `transfer`
800 s and 808 s against 843 s; the three short reports the same 14, 14 and
102 s.

**The Containerfile is a draft and does not build.** Two pins are unfilled and
both need a machine with registry and index access, which this one is not: the
base image digest, and the package hashes in `requirements-freeze.txt`. Until
they are filled the `--require-hashes` install refuses to install anything.
That is the intended failure: a base image by tag is not a version, and a
version without a hash pins no bits, because a wheel can be rebuilt and
re-uploaded under the same version. So what the runs above establish is what
the container changes about a run -- nothing, on this machine -- and not that
the image builds.

*Both pins were filled on 2026-09-13; the paragraph above stands as what was
true when these runs were made.* The hashes came from the command the freeze
file prescribes, run where there is index access. The base image is pinned to
the published amd64 image rather than to the multi-arch index, which is the
architecture this environment commits to everywhere else. The image still has
not been built and still cannot be built here. One thing the filling did not
settle: no digest for the base these two runs used survives anywhere in this
repository or in this machine's image store, so what is pinned now is the tag's
published digest and not a verified match for what they ran on. The
Containerfile says so in its own header.

**What none of this establishes.**

- Any CPU other than the reference Ryzen 7 7800X3D. The coretype pin names an
  AVX-512 kernel, so the image requires such a host and fails on one without
  it rather than computing with another kernel. What a different kernel would
  give is unmeasured.
- arm64, and the Mac the committed reports came from.
- One pair of glibc versions crossed once, 2.44 against 2.41, and one
  `libgomp` other than the host's. No older glibc, no musl, no other OpenMP
  build.
- One container runtime, rootless podman, on this kernel. Docker, containerd,
  a cgroup CPU quota or a narrower cpuset are untested, and a memory limit
  below the ladder's 8.1 GiB peak would end a run rather than change it.
- A host whose core count is not eight -- though the core count itself is no
  longer the open question it was. Two of the rungs that move at one thread
  were refitted with `n_jobs` left at eight and the process confined first to
  four CPUs and then to two, and gave the ladder's own digests every time, so
  the row blocks the histograms are summed over follow the parameter and not
  the cores on offer (Phase 4 below). That was measured by narrowing this
  machine rather than by running on a second one, so what it settles is that
  the core count is not part of the result -- not that another host reproduces
  these figures, which only that host can say. What `n_jobs` itself is worth is
  the Phase 4 section: at one thread three of December's 25 rungs are different
  models.
- The image's own install path: the uv installer fetch, `uv python install
  3.13.5`, and a fresh resolution of the thirty packages from the index. The
  runs above used the host's interpreter and wheels.
- GPU: nothing has been fitted on one, here or anywhere in this repository.

## The booster's thread count (Phase 4), 2026-09-12 to 2026-09-13

**What was run.** The ladder's 25 non-diagnostic booster rungs on each split
(the diagnostic `occupancy-tfree` was not refitted), refitted at a chosen
`n_jobs` and each compared with the rung's stored residuals: 9 runs and 114
fits over the December and July splits, made on 2026-09-12 to 2026-09-13,
with LightGBM 4.7.0 in the 2 runs that record their versions. What is
compared is the largest absolute difference over the split's validation rows,
in seconds of taxi-out time: at or below 1e-06 s it is the last bits of a sum
taken in another order, and above it the fit grew a different set of trees.
Every figure below is in [`threads-2026-09-12.md`](threads-2026-09-12.md),
which is generated from the run records, except the wall-clock comparison,
whose two times are read from the records themselves, and those of the
paragraph on [`determinism.md`](determinism.md), which are in that report.

**December.** Fitted afresh at eight threads, one at a time, which is how
every reported figure was fitted, all 25 rungs reproduce the ladder's stored
residuals exactly. At one thread they do not. Fitted eight at once, 3 of the
25 grow different trees, `weather-nolag` by 158.3 s, `throughput` by 40.27 s
and `corridor-nolag` by 4.972 s on a flight; 13 differ in the last bits only
and 9 are bit-identical. `weather-nolag` and `throughput`, refitted at one
thread one at a time, move by the same 158.3 s and 40.27 s and give the same
digests alone as beside the others, so fitting beside other fits is not what
moves them; `corridor-nolag` was not fitted at one thread alone. At four
threads, one at a time, only `corridor-nolag` grows different trees, by
4.972 s; 7 differ in the last bits and 17 are bit-identical.

**The parameter, not the cores.** `throughput` and `weather-nolag` were
refitted with `n_jobs` left at eight and the process confined first to 4 CPUs
and then to 2, and all 4 fits gave the ladder's own digests, `6e626da12f` and
`0d84cc7335`. The row blocks the histograms are summed over are therefore cut
by `n_jobs` and not by the number of CPUs the process can see.

**July.** Fitted at one thread, one at a time, none of July's 25 rungs grows
different trees: 4 differ in the last bits only and 21 are bit-identical.

**What a reader would see.** On neither split does any rung's RMSE move at the
one decimal the reports print, under any condition run.

**What it costs.** Running fits beside each other gains 1.08x in wall clock,
which the report does not carry. December's run of all 25 rungs as eight
one-thread fits side by side took 1,409 s, and its run one at a time at eight
threads 1,528 s; those two times are the records' own, and 1.08x is their
ratio.

**Which rungs move is unexplained.** With LightGBM's dataset construction
logged, all 6 rungs measured build a multi-value bin group and then use none
of it, the 3 that move and the 3 that do not (`corridor`, `throughput-nolag`
and `weather`) alike, and the sparse rate does not separate them either:
0.865997 and 0.874267 are each reported by a rung that moves and by one that
does not. No mechanism is claimed. Logging left the fits alone: the 6
rung-and-condition pairs fitted both logged and quiet each gave one digest.

**What it establishes.** The reproduction claim is a claim at `n_jobs` 8,
which `BOOSTER_PARAMS` pins, and not a claim about the configuration at every
thread count: on December a change of thread count gives a different model
for 3 of the 25 rungs at one thread, fitted eight at once, and for 1 at four.
It is a claim about a thread count and not about an eight-core machine,
because, on the two rungs refitted with fewer CPUs, the row blocks follow the
parameter and not the CPUs on offer. And what the thread count is worth is not
the same on both splits: at one thread 3 of December's 25 rungs grow different
trees, fitted eight at once, and none of July's, fitted one at a time.

**`determinism.md`, regenerated on 18 September.** The report on December's
`identifiers` rung now carries what was recorded after 11 September: the
container run and the two scaling runs, eight fits in three processes, and
three further builds of the frame. It counts eight builds and 59 fits in 20
processes. It names each fit by the parameters it recorded rather than by the
configuration it was launched under, so the three processes launched as
`current` after `deterministic` and `force_col_wise` entered `BOOSTER_PARAMS`
appear under `deterministic-col`: the scaling runs at 2 and at 4 threads,
three fits each, and the container's two fits at 8 on a row of their own,
with the environment pinned (OMP 8, OpenBLAS 8, kernel SkylakeX, hashseed 0).
Each gives one outcome, identical to `deterministic-col` at 8 and 14 threads
and to `current` at one thread. The three builds are identical to `a` on
every component. Every table entry the report carried on 11 September is
unchanged. Beside them, the component table gains a column for each new
build, the setup line names the architecture where it named the CPU model,
the seed sentence names `ctr1`'s fixed seed, the generation date is the
regeneration's, and the totals, the fits to a process (2 to 3, where it said
3) and the low end of the load average as a fit began (0.53, where it said
1.39) moved with the new records.

**What none of this establishes.**

- Another host. Every fit here ran on the reference machine; narrowing its
  CPUs, for the two rungs refitted that way, settles that the core count is
  not part of their result, not what a second machine gives.
- Any LightGBM other than the one these runs used, 4.7.0 in the 2 that record
  it. The library's own documentation says `deterministic` should keep results
  stable across a change in the number of threads, and
  [`threads-2026-09-12.md`](threads-2026-09-12.md) quotes it; in these runs 3
  of December's 25 rungs grow different trees at one thread, fitted eight at
  once, and 1 at four.
- A split, a rung or a thread count not fitted here: four threads was run on
  December alone, July only at one thread, and `occupancy-tfree` not at all.
- The mechanism. Which rungs move at another thread count, and why, is not
  explained by anything measured.
