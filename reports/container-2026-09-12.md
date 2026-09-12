# `container run 1` against `container run 2`

69 files byte-identical, 0 differing only in their generation date, 0 differing otherwise, 0 missing from a run.

## reports/*.md

| file | result |
| --- | --- |
| `baseline.md` | identical |
| `congestion.md` | identical |
| `data_review.md` | identical |
| `determinism.md` | identical |
| `geometry_gaps.md` | identical |
| `migration-2026-09-10.md` | identical |
| `occupancy.md` | identical |
| `penalty_sensitivity.md` | identical |
| `reproduction-2026-09-11.md` | identical |
| `residual.md` | identical |
| `row-wise-2026-09-11.md` | identical |
| `runway_crosscheck.md` | identical |
| `stand_coverage.md` | identical |
| `synthetic_airport.md` | identical |
| `tails.md` | identical |
| `taxi_routing.md` | identical |
| `transfer.md` | identical |

## residuals/*.parquet

| file | result |
| --- | --- |
| `december_arrivals-nolag.parquet` | identical |
| `december_arrivals.parquet` | identical |
| `december_callsign.parquet` | identical |
| `december_configuration-nolag.parquet` | identical |
| `december_configuration.parquet` | identical |
| `december_corridor-nolag.parquet` | identical |
| `december_corridor.parquet` | identical |
| `december_demand-nolag.parquet` | identical |
| `december_demand.parquet` | identical |
| `december_destination.parquet` | identical |
| `december_encoding-added.parquet` | identical |
| `december_encoding-for-identifiers.parquet` | identical |
| `december_encoding-for-stand.parquet` | identical |
| `december_identifiers-geometry-aircraft.parquet` | identical |
| `december_identifiers-geometry.parquet` | identical |
| `december_identifiers.parquet` | identical |
| `december_occupancy-tfree.parquet` | identical |
| `december_occupancy.parquet` | identical |
| `december_plan-diffs.parquet` | identical |
| `december_plan-lags.parquet` | identical |
| `december_saturation-nolag.parquet` | identical |
| `december_saturation.parquet` | identical |
| `december_throughput-nolag.parquet` | identical |
| `december_throughput.parquet` | identical |
| `december_weather-nolag.parquet` | identical |
| `december_weather.parquet` | identical |
| `july_arrivals-nolag.parquet` | identical |
| `july_arrivals.parquet` | identical |
| `july_callsign.parquet` | identical |
| `july_configuration-nolag.parquet` | identical |
| `july_configuration.parquet` | identical |
| `july_corridor-nolag.parquet` | identical |
| `july_corridor.parquet` | identical |
| `july_demand-nolag.parquet` | identical |
| `july_demand.parquet` | identical |
| `july_destination.parquet` | identical |
| `july_encoding-added.parquet` | identical |
| `july_encoding-for-identifiers.parquet` | identical |
| `july_encoding-for-stand.parquet` | identical |
| `july_identifiers-geometry-aircraft.parquet` | identical |
| `july_identifiers-geometry.parquet` | identical |
| `july_identifiers.parquet` | identical |
| `july_occupancy-tfree.parquet` | identical |
| `july_occupancy.parquet` | identical |
| `july_plan-diffs.parquet` | identical |
| `july_plan-lags.parquet` | identical |
| `july_saturation-nolag.parquet` | identical |
| `july_saturation.parquet` | identical |
| `july_throughput-nolag.parquet` | identical |
| `july_throughput.parquet` | identical |
| `july_weather-nolag.parquet` | identical |
| `july_weather.parquet` | identical |

Distance matrices derived from OpenStreetMap: data (c)
OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied by
EUROCONTROL PRC and the OpenSky Network under the terms of the PRC Data
Challenge 2026.
# `container run 1` against `host run C`

63 files byte-identical, 5 differing only in their generation date, 1 differing otherwise, 1 missing from a run.

## reports/*.md

| file | result |
| --- | --- |
| `baseline.md` | differs only in its generation date (1 line) |
| `congestion.md` | differs only in its generation date (1 line) |
| `data_review.md` | identical |
| `determinism.md` | identical |
| `geometry_gaps.md` | identical |
| `migration-2026-09-10.md` | **116 lines differ** |
| `occupancy.md` | differs only in its generation date (1 line) |
| `penalty_sensitivity.md` | identical |
| `reproduction-2026-09-11.md` | identical |
| `residual.md` | differs only in its generation date (1 line) |
| `row-wise-2026-09-11.md` | identical |
| `runway_crosscheck.md` | identical |
| `stand_coverage.md` | identical |
| `synthetic_airport.md` | identical |
| `tails.md` | identical |
| `taxi_routing.md` | identical |
| `transfer.md` | differs only in its generation date (1 line) |

## reports/*.svg

| file | result |
| --- | --- |
| `synthetic_airport.svg` | **only in `host run C`** |

## residuals/*.parquet

| file | result |
| --- | --- |
| `december_arrivals-nolag.parquet` | identical |
| `december_arrivals.parquet` | identical |
| `december_callsign.parquet` | identical |
| `december_configuration-nolag.parquet` | identical |
| `december_configuration.parquet` | identical |
| `december_corridor-nolag.parquet` | identical |
| `december_corridor.parquet` | identical |
| `december_demand-nolag.parquet` | identical |
| `december_demand.parquet` | identical |
| `december_destination.parquet` | identical |
| `december_encoding-added.parquet` | identical |
| `december_encoding-for-identifiers.parquet` | identical |
| `december_encoding-for-stand.parquet` | identical |
| `december_identifiers-geometry-aircraft.parquet` | identical |
| `december_identifiers-geometry.parquet` | identical |
| `december_identifiers.parquet` | identical |
| `december_occupancy-tfree.parquet` | identical |
| `december_occupancy.parquet` | identical |
| `december_plan-diffs.parquet` | identical |
| `december_plan-lags.parquet` | identical |
| `december_saturation-nolag.parquet` | identical |
| `december_saturation.parquet` | identical |
| `december_throughput-nolag.parquet` | identical |
| `december_throughput.parquet` | identical |
| `december_weather-nolag.parquet` | identical |
| `december_weather.parquet` | identical |
| `july_arrivals-nolag.parquet` | identical |
| `july_arrivals.parquet` | identical |
| `july_callsign.parquet` | identical |
| `july_configuration-nolag.parquet` | identical |
| `july_configuration.parquet` | identical |
| `july_corridor-nolag.parquet` | identical |
| `july_corridor.parquet` | identical |
| `july_demand-nolag.parquet` | identical |
| `july_demand.parquet` | identical |
| `july_destination.parquet` | identical |
| `july_encoding-added.parquet` | identical |
| `july_encoding-for-identifiers.parquet` | identical |
| `july_encoding-for-stand.parquet` | identical |
| `july_identifiers-geometry-aircraft.parquet` | identical |
| `july_identifiers-geometry.parquet` | identical |
| `july_identifiers.parquet` | identical |
| `july_occupancy-tfree.parquet` | identical |
| `july_occupancy.parquet` | identical |
| `july_plan-diffs.parquet` | identical |
| `july_plan-lags.parquet` | identical |
| `july_saturation-nolag.parquet` | identical |
| `july_saturation.parquet` | identical |
| `july_throughput-nolag.parquet` | identical |
| `july_throughput.parquet` | identical |
| `july_weather-nolag.parquet` | identical |
| `july_weather.parquet` | identical |

Distance matrices derived from OpenStreetMap: data (c)
OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied by
EUROCONTROL PRC and the OpenSky Network under the terms of the PRC Data
Challenge 2026.
