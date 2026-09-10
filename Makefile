# SPDX-License-Identifier: GPL-3.0-only
#
# Single entry point. Each target names what it reads and what it writes, so
# the pipeline can be run stage by stage or end to end.

PYTHON ?= python3

.PHONY: all data metar ourairports reports crosscheck data-review synthetic tails \
        geometry-gaps congestion occupancy residual baseline transfer sweep submission test \
        lint licence-check clean help

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
	@echo "make baseline     run the ablation ladder      -> reports/baseline.md"
	@echo "make transfer     leave one airport out, ten times -> reports/transfer.md"
	@echo "make sweep        the one parameter sweep, October 2025 -> data/processed/sweep_october.csv"
	@echo "make submission TEAM=<name>  fit and write     -> data/submissions/<name>_v<n>.parquet"
	@echo "make test     run the unit tests                (no network, no challenge data)"
	@echo "make lint     ruff check and format check"
	@echo "make licence-check  verify no OSM-derived content is tracked"
	@echo "make all      data + reports + test"
	@echo ""
	@echo "'make data' calls the Overpass API and takes a few minutes; it is"
	@echo "resumable and skips airports already cached. 'make reports' is"
	@echo "offline and takes two to three minutes for all 11 airports."
	@echo "'make synthetic' needs neither, and is the one geometry report a"
	@echo "clean checkout can regenerate. The four targets below it need the"
	@echo "challenge data under data/challenge/, which is never committed."

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
# aggregates only and refuses to write a challenge identifier.
tails:
	$(PYTHON) scripts/report_tails.py

# Reads data/challenge/, the matrices from 'make reports' and the OSM cache
# under data/raw/. Writes the diagnostics of the unimpeded term against the
# training year: the fit, its residuals, wake category, and each runway's
# entries. Aggregates only. About a minute.
geometry-gaps:
	$(PYTHON) scripts/report_geometry_gaps.py

# Reads data/challenge/, the matrices and the edge sets from 'make reports'.
# Writes how the congestion features were built and whether the construction
# holds on both sides of the data; fits no model. About a minute.
congestion:
	$(PYTHON) scripts/report_congestion.py

# Reads data/challenge/ and the validation residuals 'make baseline' writes
# under data/processed/residuals/. Writes how the stand-occupancy block was
# built, whether the gap means the same on both sides, and where the physical
# bound behind it holds. Fits no model. A few minutes.
occupancy:
	$(PYTHON) scripts/report_occupancy.py

# Reads data/challenge/ and the validation residuals 'make baseline' writes
# under data/processed/residuals/. Writes what the model's remaining error is:
# the shape of the tail, the classes it is enriched in, the error by plan lag,
# and the largest rows. Fits no model. A few minutes.
residual:
	$(PYTHON) scripts/report_residual.py

# Reads data/challenge/ and the matrices from 'make reports'. Writes the
# ablation ladder. Needs the modelling extras; takes a few minutes.
baseline:
	$(PYTHON) scripts/run_baselines.py

# Reads data/challenge/ and the matrices from 'make reports'. Ten fits: train
# on nine airports, predict the tenth, with no identifier of the tenth in the
# frame. Answers the question the challenge's test set cannot pose, and the one
# no identifier-based entry can answer at all. An hour or more.
transfer:
	$(PYTHON) scripts/report_transfer.py

# The one parameter sweep: fit January to September 2025, score October, on
# the saturation rung's features, the top rung when it ran. Its numbers stay under data/processed/, gitignored;
# the chosen parameters are written into taxiout.models by hand with the
# October number that chose them. Takes half an hour or more.
sweep:
	$(PYTHON) scripts/sweep_params.py

# Reads data/challenge/ and the matrices; writes a submission file to
# data/submissions/, which is gitignored and which the script verifies is
# ignored before writing. TEAM is the team name as the organizers hold it.
submission:
	@test -n "$(TEAM)" || { echo "usage: make submission TEAM=<name>"; exit 1; }
	$(PYTHON) scripts/make_submission.py --team $(TEAM)

test:
	$(PYTHON) -m pytest -q

# Verifies the repository is GPLv3 in totality. Run before every commit.
licence-check:
	$(PYTHON) scripts/check_licence_hygiene.py

lint:
	ruff check .
	ruff format --check .

clean:
	rm -rf data/processed/taxi_distance/*_features.csv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
