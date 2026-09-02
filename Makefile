# SPDX-License-Identifier: GPL-3.0-only
#
# Single entry point. Each target names what it reads and what it writes, so
# the pipeline can be run stage by stage or end to end.

PYTHON ?= python3

.PHONY: all data reports crosscheck data-review test lint licence-check clean help

help:
	@echo "make data     fetch and cache OSM geometry      -> data/raw/<ICAO>.json.gz"
	@echo "make reports  build graph, matrices and reports -> data/processed/, reports/"
	@echo "make data-review  profile the challenge data   -> reports/data_review.md"
	@echo "make test     run the unit tests                (no network, no challenge data)"
	@echo "make lint     ruff check and format check"
	@echo "make licence-check  verify no OSM-derived content is tracked"
	@echo "make all      data + reports + test"
	@echo ""
	@echo "'make data' calls the Overpass API and takes a few minutes; it is"
	@echo "resumable and skips airports already cached. 'make reports' is"
	@echo "offline and takes two to three minutes for all 11 airports."

all: data reports test licence-check

# Network. Resumable: delete a cache file to refetch that airport.
data:
	$(PYTHON) scripts/fetch_osm.py

# Offline. Reads data/raw/, writes the matrices and every report.
reports:
	$(PYTHON) scripts/report_coverage.py
	$(PYTHON) scripts/build_taxi_graph.py
	$(PYTHON) scripts/check_penalty_sensitivity.py

# Offline, but fetches OurAirports on first run and caches it.
crosscheck:
	$(PYTHON) scripts/check_runways_ourairports.py

# Needs the challenge data under data/challenge/ (never committed) and the
# matrices from 'make reports'. Writes only aggregates and identifier shapes,
# and refuses to write a literal stand identifier.
data-review:
	$(PYTHON) scripts/review_challenge_data.py

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
