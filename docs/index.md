---
icon: lucide/house
---

# environmental-stac-generator

Converts daily environmental forecast netCDF outputs from machine learning models to **Cloud Optimized GeoTIFFs (COGs)** and **STAC catalogues** for use with [environmental-stac-dashboard](https://github.com/environmental-forecasting/environmental-stac-dashboard) and the [environmental-stac-orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator) stack.

## Features

- Converts netCDF predictions into Cloud Optimized GeoTIFFs (COGs)
- Builds STAC-compliant metadata catalogues for each forecast
- Dynamically detects northern or southern hemisphere from input data
- Outputs COGs, sliced netCDFs, and thumbnails in a standardised layout
- Ingests netCDF attributes as STAC Item metadata
- Generates config files to ensure consistent processing

## Installation

```bash
pip install -e .
```

Or with uv from this repo:

```bash
uv sync
```

## Usage overview

Designed for use with the orchestrator. Pass an orchestrator environment file with `--env-file` on **ingest** (database credentials and `FILE_SERVER_URL`).

* `preprocess` does not use it: the static catalogue stores portable cwd-relative asset paths (e.g. `data/cogs/...`).
* `ingest` prefixes `FILE_SERVER_URL` when loading into pgSTAC.

```bash
envstacgen preprocess ./results/predict/*.nc
envstacgen ingest --env-file .env.development data/stac/catalog.json -o
```

See [CLI](cli.md) for flags and a full example.
