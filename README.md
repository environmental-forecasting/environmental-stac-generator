# environmental-stac-generator

Converts daily environmental forecast netCDF outputs from machine learning models to COG files + STAC catalogs for `environmental-stac-dashboard`.

A command-line tool for generating **Cloud Optimized GeoTIFFs (COGs)** and **STAC catalogs** from environmental forecast prediction netCDF files. It outputs COGs, sliced netCDFs, and thumbnails in the necessary format to ingest them into a pgSTAC database for use via STAC APIs, and `environmental-stac-dashboard`.

## Features

- Converts netCDF predictions into Cloud Optimized GeoTIFFs (COGs)
- Builds STAC-compliant metadata catalogs for each forecast
- Supports compressed or uncompressed output
- Dynamically detects northern or southern hemisphere from input data
- Outputs COGs, sliced netCDFs, and thumbnails in standardised format
- Ingests netCDF attributes as STAC Item metadata
- Generates config files to ensure consistent processing

## Installation

To use this tool, ensure the dependencies from [pyproject.toml](pyproject.toml) are installed. Or, for an editable install, clone the repo and run the following after changing directory to the repo root:

```bash
pip install -e .
```

## Usage

This tool is designed to be used with [environmental-stac-orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator).

Pass an orchestrator environment file with `--env-file` on **ingest** (database credentials and `FILE_SERVER_URL`). Preprocess does not need it: the static catalog stores portable cwd-relative asset paths (e.g. `data/cogs/...`), and ingest prefixes `FILE_SERVER_URL` when loading into pgSTAC.

```bash
envstacgen preprocess ./results/predict/*.nc
envstacgen ingest --env-file .env.development data/stac/catalog.json -o
```

### Positional Parameters

Paths to one or more `.nc` files, directories, or wildcard patterns.

Lead valid times are inferred from the netCDF (`forecast_date` when present, otherwise `leadtime` / `lead_time` offsets). A compact frequency label is stored in `data/config.json` for consistency checks between runs - you do not pass `1days` on the CLI.

### Options

`preprocess` options:

| Flag                  | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `--overwrite`, `-o`   | Overwrite existing GeoTIFF files if they already exist.                |
| `--no-compress`, `-c` | Disable compression in generated GeoTIFFs (default is compressed).     |
| `--name`, `-n`        | Specify a collection name (default: "default")                         |
| `--workers`, `-w`     | Max concurrent workers (default: CPU count)                            |
| `--stac-only`, `-s`   | Output only the STAC files, not COGs/Thumbnails (default not enabled)  |

`ingest` options:

| Flag | Description |
| ---- | ----------- |
| `--env-file` | Path to an environment file (e.g. `.env.development`) |
| `--overwrite`, `-o` | Overwrite existing matching entries |

The ingestion step requires a reachable PostgreSQL/pgSTAC instance (typically the [environmental-stac-orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator) stack). See that repository’s README.

## Example

### Step 1

```bash
envstacgen preprocess raw_data/*.nc -o
```

This will:
* Detect the hemisphere automatically
* Convert each leadtime slice to a COG
* Process the COG outputs into `data/cogs/{collection}/{date}/`
* Build a hierarchical STAC catalog in `data/stac/` with portable asset hrefs

### Step 2

```bash
envstacgen ingest --env-file .env.development data/stac/catalog.json -o
```

This will:
* Prefix asset hrefs with `FILE_SERVER_URL` for this environment
* Ingest the catalog into the PostgreSQL database.

## License

`environmental-stac-generator` is licensed under the MIT license. See [LICENSE](https://github.com/environmental-forecasting/environmental-stac-generator/blob/main/LICENSE) for more information.

## Documentation

Docs can be built with:

```bash
make docs-install
make docs
```

Related: [environmental-stac-orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator), [environmental-stac-dashboard](https://github.com/environmental-forecasting/environmental-stac-dashboard).
