# SPDX-License-Identifier: GPL-3.0-only
#
# Single entry point. Each target names what it reads and what it writes, so
# the pipeline can be run stage by stage or end to end.

PYTHON ?= python3

.PHONY: all data metar ourairports reports crosscheck data-review synthetic tails \
        geometry-gaps congestion occupancy residual headroom baseline transfer sweep \
        preregistered metar-history monthly monthly-fit monthly-ladder monthly-ladder-fit \
        monthly-seeds monthly-seeds-fit monthly-pairs monthly-pairs-fit planshift \
        istanbul fiumicino submission paper-docx test lint licence-check provenance \
        clean help

help:
	@echo "make data     fetch and cache OSM geometry      -> data/raw/<ICAO>.json.gz"
	@echo "make metar    fetch and cache METARs            -> data/raw/metar/<ICAO>_<period>.csv"
	@echo "make ourairports  fetch and cache OurAirports   -> data/raw/ourairports_airports.csv"
	@echo "make reports  build graph, matrices and reports -> data/processed/, reports/"
	@echo "make synthetic    the worked airport            -> reports/synthetic_airport.md, .svg"
	@echo "make data-review  profile the challenge data   -> reports/data_review.md"
	@echo "make tails        profile the record-quality term -> reports/tails.md"
	@echo "make geometry-gaps  the unimpeded term against the data -> reports/geometry_gaps.md"
	@echo "make congestion   how the excess term was built -> reports/congestion.md"
	@echo "make occupancy    the stand as a shared resource -> reports/occupancy.md"
	@echo "make residual     what the model's remaining error is -> reports/residual.md"
	@echo "make headroom     the residual-fit headroom, recomputed -> reports/headroom.md"
	@echo "make baseline     run the ablation ladder      -> reports/baseline.md"
	@echo "make transfer     leave one airport out, ten times -> reports/transfer.md"
	@echo "make sweep        the parameter sweep, October 2025 -> data/processed/sweep_october.csv"
	@echo "make preregistered  the pre-registered experiments -> reports/preregistered-<date>.md"
	@echo "make metar-history  ten years of METARs, context only -> data/raw/metar_history/"
	@echo "make monthly       every month held out, the shipped rung -> reports/monthly.md"
	@echo "make monthly-ladder  every rung, every month held out -> reports/monthly_ladder.md"
	@echo "make monthly-seeds   five seeds per held-out month -> reports/monthly_seeds.md"
	@echo "make monthly-pairs   the geometry claim by month  -> reports/monthly_pairs.md"
	@echo "make planshift     the plan-shifted departures   -> reports/planshift.md"
	@echo "make istanbul      Istanbul's February and January 2026 -> reports/istanbul.md"
	@echo "make fiumicino     the last three blocks at Fiumicino -> reports/fiumicino.md"
	@echo "make paper-docx    the paper as a Word document  -> docs/paper.docx"
	@echo "make submission TEAM=<name>  fit and write     -> data/submissions/<name>_v<n>.parquet"
	@echo "make test     run the unit tests                (no network, no challenge data)"
	@echo "make lint     ruff check and format check"
	@echo "make licence-check  verify no OSM-derived content is tracked"
	@echo "make provenance  verify every number is categorised -> docs/provenance.md"
	@echo "make all      data + reports + test"
	@echo ""
	@echo "'make data' calls the Overpass API and takes a few minutes; it is"
	@echo "resumable and skips airports already cached. 'make reports' is"
	@echo "offline and takes about two minutes, measured at 1 min 49 s on"
	@echo "27 September 2026 in a clean clone (reports/reproduction-2026-09-27.md):"
	@echo "report_coverage.py 0.2 s, build_taxi_graph.py 27 s, and"
	@echo "check_penalty_sensitivity.py 1 min 21 s, which re-routes every stand"
	@echo "at every airport for each penalty. Until that measurement this line"
	@echo "said about an hour, from the script's own estimate."
	@echo "'make synthetic' needs neither, and is the one geometry report a"
	@echo "clean checkout can regenerate. The four targets below it need the"
	@echo "challenge data under data/challenge/, which is never committed."
	@echo ""
	@echo "Measured on the reference machine, eight cores, all re-timed on"
	@echo "23 to 25 September 2026 under the cell adopted then:"
	@echo "  make baseline    9 h 44 min   8.1 GiB"
	@echo "  make transfer      50 min     7.0 GiB"
	@echo "  make sweep       1 h 30 min   6.7 GiB   (does not run again)"
	@echo "  make congestion     1:44      6.7 GiB"
	@echo "  make headroom       2:37      3.7 GiB"
	@echo "  make geometry-gaps    37 s    2.8 GiB"
	@echo "  make residual         21 s    3.2 GiB"
	@echo "  make occupancy        15 s    2.8 GiB"
	@echo "  make submission     6:55      7.6 GiB   (one seed)"
	@echo "  check_determinism.py report  0.8 s      (refits nothing)"
	@echo ""
	@echo "The held-out year: each report target reads stored residuals and"
	@echo "refits nothing (timed 3 October 2026); each -fit target refits."
	@echo "  make monthly          3.0 s    make monthly-fit          1 h 13 min 45 s"
	@echo "  make monthly-ladder   2:54.6   make monthly-ladder-fit   23.2 h (twelve splits)"
	@echo "  make monthly-seeds    7.3 s    make monthly-seeds-fit    4 h 24 min (48 fits)"
	@echo "  make monthly-pairs    2:01.8   make monthly-pairs-fit    30 min (nine fits)"
	@echo "  make planshift        42 s     make metar-history        7 min 48 s (network)"
	@echo "  make istanbul         45 s"
	@echo "  make fiumicino        9.6 s"
	@echo ""
	@echo "A five-seed submission, measured for the first time on 24 September,"
	@echo "is 29 min 20 s at 8.0 GiB. That is minutes rather than hours, which"
	@echo "is what the estimate behind SEED_AVERAGE_ADOPTED assumed and had"
	@echo "never had measured under it."
	@echo ""
	@echo "'make baseline' took about two hours until 21 September 2026, when"
	@echo "the one permitted wider sweep moved the booster to 3,000 trees and"
	@echo "255 leaves, and 7 h 11 min until 23 September, when the extension"
	@echo "moved it to 6,000 trees and 127 leaves; under the parameters that"
	@echo "ship it is 9 h 36 min 30 s (27 September, in a clean clone,"
	@echo "reports/reproduction-2026-09-27.md). Every target that fits the"
	@echo "submission rung scales with it."
	@echo ""
	@echo "THESE ARE MONOTONIC-CLOCK FIGURES, from /usr/bin/time, and a host"
	@echo "that suspends will show a longer wall time for the same work. The"
	@echo "9 h 44 min run above spanned 30 h 19 min of wall clock because this"
	@echo "host slept for 20 h 35 min of it. Time a long run with /usr/bin/time"
	@echo "and hold a systemd-inhibit sleep lock, or the number is fiction."

all: data reports test licence-check

# Network. Resumable: delete a cache file to refetch that airport.
data:
	$(PYTHON) scripts/fetch_osm.py

# Network. The Iowa State Mesonet's ASOS/METAR archive, for the training year
# and the two ranking months at every airport; the request and the terms are
# recorded in data/raw/metar/MANIFEST.json. Resumable per station and period.
metar:
	$(PYTHON) scripts/fetch_metar.py

# Network, once. OurAirports' airport table, for the destination block's
# coordinates; the request and the terms are recorded in
# data/raw/ourairports_MANIFEST.json. Resumable: --refresh refetches.
ourairports:
	$(PYTHON) scripts/fetch_ourairports.py

# Offline. Reads data/raw/, writes the matrices and every report.
reports:
	$(PYTHON) scripts/report_coverage.py
	$(PYTHON) scripts/build_taxi_graph.py
	$(PYTHON) scripts/check_penalty_sensitivity.py

# Offline, but fetches OurAirports on first run and caches it.
crosscheck:
	$(PYTHON) scripts/check_runways_ourairports.py

# Offline and needs nothing cached: the field is authored in taxiout.synthetic.
# Reads no data at all; writes the worked-airport report and its figure.
synthetic:
	$(PYTHON) scripts/report_synthetic_airport.py

# Needs the challenge data under data/challenge/ (never committed) and the
# matrices from 'make reports'. Writes only aggregates and identifier shapes,
# and refuses to write a literal stand identifier.
data-review:
	$(PYTHON) scripts/review_challenge_data.py

# Reads data/challenge/. Writes the record-quality term's own report; carries
# aggregates only and refuses to write a challenge identifier. Two generators,
# in this order: the first writes the report from end to end, the second
# measures the pooled rule and writes its one section into the file the first
# produced. Running only the first would drop that section, so a test fails if
# the committed report has lost it.
tails:
	$(PYTHON) scripts/report_tails.py
	$(PYTHON) scripts/pooled_tails.py --report

# Reads data/challenge/, the matrices from 'make reports' and the OSM cache
# under data/raw/. Writes the diagnostics of the unimpeded term against the
# training year: the fit, its residuals, wake category, and each runway's
# entries. Aggregates only. About a minute.
geometry-gaps:
	$(PYTHON) scripts/report_geometry_gaps.py

# Reads data/challenge/, the matrices and the edge sets from 'make reports'.
# Writes how the congestion features were built and whether the construction
# holds on both sides of the data; fits no model. About two minutes.
congestion:
	$(PYTHON) scripts/report_congestion.py

# Reads data/challenge/ and the validation residuals 'make baseline' writes
# under data/processed/residuals/. Writes how the stand-occupancy block was
# built, whether the gap means the same on both sides, and where the physical
# bound behind it holds. Fits no model. Under a minute.
occupancy:
	$(PYTHON) scripts/report_occupancy.py

# Reads data/challenge/ and the validation residuals 'make baseline' writes
# under data/processed/residuals/. Writes what the model's remaining error is:
# the shape of the tail, the classes it is enriched in, the error by plan lag,
# and the largest rows. Fits no model. Under a minute.
residual:
	$(PYTHON) scripts/report_residual.py

# Reads data/challenge/ and the validation residuals 'make baseline' writes
# under data/processed/residuals/, at the rung docs/review_headroom.md line 12
# names. Recomputes that review's §2 headroom estimates for the four Gate 2c
# blocks by the method its line 11 states, at a capacity this script names, and
# writes them BESIDE the quoted EXPECTED_HEADROOM_S rather than in place of it.
# Moves no threshold, no constant and no verdict: the leakage audit in
# reports/baseline.md goes on reading the quoted figure. Fits forty small
# boosters and nothing of the ladder; about three minutes.
headroom:
	$(PYTHON) scripts/report_headroom.py

# Reads data/challenge/ and the matrices from 'make reports'. Writes the
# ablation ladder. Needs the modelling extras; 9 h 44 min on eight cores,
# with a peak of 8.1 GiB of memory, measured 23 to 24 September 2026 under
# the adopted cell. Unbuffered: this script prints one line per rung and
# block-buffers them into a file otherwise, which left a nine-hour run with
# no readable progress.
baseline:
	$(PYTHON) -u scripts/run_baselines.py

# Reads data/challenge/ and the matrices from 'make reports'. Ten fits: train
# on nine airports, predict the tenth, with no identifier of the tenth in the
# frame. Answers the question the challenge's test set cannot pose, and the one
# no identifier-based entry can answer at all. About fifty minutes on eight
# cores under the parameters that ship: 49 min 22 s on 27 September 2026
# (reports/reproduction-2026-09-27.md).
transfer:
	$(PYTHON) -u scripts/report_transfer.py

# The parameter sweeps: fit January to September 2025, score October. The
# first ran on the saturation rung, the top rung at the time; the second, the
# one this target runs, is pre-registered on the submission rung under the fit
# rules that stand, and their October figures are not comparable. Twenty-seven
# cells, one 3,000-tree fit each, read at six tree counts, after a gate that
# reproduces the five October single-seed fits on record. Its numbers stay
# under data/processed/, gitignored; the chosen parameters are written into
# taxiout.models by hand with the October number that chose them. --largest
# times the costliest cell alone, --read applies the reading rule without
# fitting anything.
sweep:
	$(PYTHON) -u scripts/sweep_params.py

# Reads data/challenge/ and the matrices from 'make reports'. Fits the
# submission rung twenty times under criteria fixed in writing before any of
# them ran, and writes what each criterion's own rule decides. Adopts nothing
# and changes no parameter, feature or rule; writes its records under
# data/processed/preregistered/ (gitignored) and one report. About an hour on
# eight cores, with the same 8 GiB peak 'make baseline' has. --report
# regenerates the report from the records without refitting.
preregistered:
	$(PYTHON) scripts/preregistered.py

# Network, once. Ten calendar years of METARs before the training year, 2015
# to 2024, for the ten airports with departures, through scripts/fetch_metar.py's
# endpoint, terms, retries and pause; for one context table of
# reports/monthly_ladder.md and nothing else. Writes data/raw/metar_history/ and
# its manifest, never data/raw/metar/. 7 min 48 s on 1 October 2026. Resumable.
metar-history:
	$(PYTHON) scripts/fetch_metar_history.py

# The held-out year. Each month of 2025 held out in turn, the model trained on
# the other eleven: not forward in time, a diagnostic that decides nothing. The
# -fit targets refit and store residuals under data/processed/ (gitignored, they
# carry the target); the report targets read them and refit nothing. In order:
# monthly, then monthly-ladder (which checks every month against monthly's
# files), then monthly-seeds and monthly-pairs, then planshift, istanbul and
# fiumicino. Fit times measured 1 and 2 October 2026 on eight cores.
#
# The submitted rung, fourteen fits: 1 h 13 min 45 s.
monthly-fit:
	$(PYTHON) -u scripts/report_monthly.py --fit
monthly:
	$(PYTHON) scripts/report_monthly.py --report

# Every rung on every held-out month: 23.2 h (December 1:58:27, the other
# eleven 21:13:23). Resumable per month. Needs metar-history for one annex.
monthly-ladder-fit:
	$(PYTHON) -u scripts/report_monthly_ladder.py --fit
monthly-ladder:
	$(PYTHON) scripts/report_monthly_ladder.py --report

# Four more seeds of the submitted rung per held-out month: the seed-path check
# on January (6 min 56 s), then the 48 fits (4 h 24 min). Resumable per month.
monthly-seeds-fit:
	$(PYTHON) -u scripts/fit_monthly_seeds.py --check
	$(PYTHON) -u scripts/fit_monthly_seeds.py --fit
monthly-seeds:
	$(PYTHON) scripts/report_monthly_seeds.py

# The three held-out-pair folds refitted for three rungs with every prediction
# stored (30 min), then checked against reports/baseline.md (6 s).
monthly-pairs-fit:
	$(PYTHON) -u scripts/fit_monthly_pairs.py --fit
	$(PYTHON) -u scripts/fit_monthly_pairs.py --check
monthly-pairs:
	$(PYTHON) scripts/report_monthly_pairs.py

# The plan-shifted departures under the rule of reports/preregistered-2026-10-02.md,
# evaluated on monthly's stored residuals; the class is counted against the
# pre-registration's stops before anything is evaluated. It also reads v6,
# data/submissions/funny-firefly_v6.parquet, for the frozen model's prediction
# on the one ranking departure of the class; without it the run stops there.
planshift:
	$(PYTHON) scripts/report_planshift.py --report

# From monthly-seeds' residuals, the training and ranking files and both METAR
# caches.
istanbul:
	$(PYTHON) scripts/report_istanbul.py

# From monthly-ladder's residuals.
fiumicino:
	$(PYTHON) scripts/report_fiumicino.py

# Reads data/challenge/ and the matrices; writes a submission file and its
# manifest to data/submissions/, which is gitignored and which the script
# verifies is ignored before writing either. Refuses to fit with any booster
# but LightGBM. TEAM is the team name as the organizers hold it.
submission:
	@test -n "$(TEAM)" || { echo "usage: make submission TEAM=<name>"; exit 1; }
	$(PYTHON) scripts/make_submission.py --team $(TEAM)

# The paper as a Word document, tracked beside docs/paper.md and regenerated
# whenever the paper moves. With pandoc installed: pandoc's GFM reader and its
# table of contents. Without it: md4c's md2html renders the same GFM to HTML,
# a contents list of the paper's headings is put at its head, and LibreOffice
# converts that. Either way the zip entries are re-stamped so a rerun writes the
# same bytes, and the file is moved into place only when every step has
# passed. Make skips it while docs/paper.docx is newer than docs/paper.md.
paper-docx: docs/paper.docx

docs/paper.docx: docs/paper.md
	@set -e; d="$$(mktemp -d)"; trap 'rm -rf "$$d"' EXIT; \
	if command -v pandoc > /dev/null; then \
	    echo "pandoc"; \
	    SOURCE_DATE_EPOCH=315532800 pandoc $< -f gfm -t docx --toc -o "$$d/paper.docx"; \
	else \
	    echo "md2html and soffice"; \
	    md2html --github --full-html --html-title="$$(head -1 $< | sed 's/^# //')" $< > "$$d/paper.html"; \
	    $(PYTHON) -c "$$PAPER_CONTENTS" "$$d/paper.html"; \
	    soffice --headless --norestore "-env:UserInstallation=file://$$d/profile" \
	        --infilter="HTML (StarWriter)" --convert-to "docx:MS Word 2007 XML" \
	        --outdir "$$d" "$$d/paper.html" > /dev/null; \
	fi; \
	$(PYTHON) -c "$$PAPER_RESTAMP" "$$d/paper.docx" "$$d/stamped.docx"; \
	mv -f "$$d/stamped.docx" $@

# A contents list of the h1 to h3 headings, as pandoc's --toc writes one.
define PAPER_CONTENTS
import re, sys
path = sys.argv[1]
html = open(path, encoding="utf-8").read()
heads = re.findall(r"<h([123])>(.*?)</h\1>", html)
items = "".join(f'<p style="margin-left:{(int(n) - 1) * 2}em">{text}</p>' for n, text in heads)
html = html.replace("<body>", "<body><h1>Contents</h1>" + items, 1)
open(path, "w", encoding="utf-8").write(html)
endef
export PAPER_CONTENTS

# The same entries in the same order, each stamped 1980-01-01, the zip minimum.
define PAPER_RESTAMP
import sys, zipfile
src, dst = sys.argv[1:]
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
    for i in zin.infolist():
        z = zipfile.ZipInfo(i.filename, (1980, 1, 1, 0, 0, 0))
        zout.writestr(z, zin.read(i), zipfile.ZIP_DEFLATED)
endef
export PAPER_RESTAMP

test:
	$(PYTHON) -m pytest -q

# Verifies the repository is GPLv3 in totality. Run before every commit.
licence-check:
	$(PYTHON) scripts/check_licence_hygiene.py

# Offline, reads the tracked tree and nothing else. Proves that every number in
# the scanned surface is categorised: generated by the run that prints it,
# checked against a named source, registered with a basis, or exempt with a
# reason. Ten checks, and --strict is the freeze condition rather than the
# ordinary run, because a gate that fails today cannot gate a commit.
# --census prints the whole enumeration; --write-doc regenerates
# docs/provenance.md from the register, whose counts are summed from it.
# Seconds.
provenance:
	$(PYTHON) scripts/check_provenance.py

lint:
	ruff check .
	ruff format --check .

clean:
	rm -rf data/processed/taxi_distance/*_features.csv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
