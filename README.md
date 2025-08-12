# environmental-stac-generator

Converts daily environmental forecast netCDF outputs from machine learning models to COG files + STAC catalogs for `environmental-stac-dashboard`.

A command-line tool for generating **Cloud Optimized GeoTIFFs (COGs)** and **STAC catalogs** from environmental forecast prediction netCDF files. It outputs COGs, sliced netCDFs, and thumbnails in the necessary format to ingest them into a pgSTAC database for use via STAC APIs, and `environmental-stac-dashboard`.

## Features

- Converts netCDF predictions into Cloud Optimized GeoTIFFs (COGs)
- Automatically reprojects to EPSG:4326 for Leaflet/web-map compatibility
- Builds STAC-compliant metadata catalogs for each forecast
- Supports compressed or uncompressed output
- Dynamically detects northern or southern hemisphere from input data
- Outputs COGs, sliced netCDFs, and thumbnails in standardised format
- Generates config files to ensure consistent processing

## Installation

To use this tool, ensure the dependencies from [pyproject.toml](pyproject.toml) are installed. Or, for an editable install, clone the repo and run the following after changing directory to the repo root:

```bash
pip install -e .
```

## Usage

This tool is designed to be used with [environmental-stac-orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator).

The preprocessing step can be run with the following command:

```bash
envstacgen preprocess 1days ./results/predict/*.nc
```

It expects an `.env` file where you're running the command from, with `FILE_SERVER_URL` variable pointing to the location of the file server where the outputs from this command will be served from.

e.g.

```bash
FILE_SERVER_URL=http://localhost:8002
```

### Positional Parameters

The first parameter is the forecast frequency (e.g., "6hours", "1days").

The second and further arguments are the paths to one or more `.nc` files, directories, or wildcard patterns.

### Options

The optional flags that can be used are:

| Flag                  | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `--overwrite`, `-o`   | Overwrite existing GeoTIFF files if they already exist.                |
| `--no-compress`, `-c` | Disable compression in generated GeoTIFFs (default is compressed).     |
| `--name`, `-n`        | Specify a collection name (default: "default")                         |
| `--workers`, `-w`     | Set max number of concurrent workers (default: 4)                      |
| `--not-flat`, `-nf`   | Output hierarchical STAC JSON (default is flat for pgSTAC compatibility) |
| `--stac-only`, `-s`   | Output only the STAC files, not COGs/Thumbnails (default not enabled)  |

The ingestion requires the [environmental-stac-orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator) docker compose environment to be running (since it will ingest into a PostgreSQL database). Please follow the [README](https://github.com/environmental-forecasting/environmental-stac-orchestrator/blob/main/README.md) from it.

## Example

### Step 1

```bash
envstacgen preprocess 1days raw_data/*.nc -o
```

This will:
* Detect the hemisphere automatically
* Convert each leadtime slice to a COG
* Process the COG outputs into `data/cogs/{collection}/{date}/`
* Build a hierarchical STAC catalog in `data/stac/`

### Step 2

```bash
envstacgen ingest data/stac/catalog.json -o
```

This will:
* Ingest the catalog into the PostgreSQL database.

## License

`environmental-stac-generator` is licensed under the MIT license. See [LICENSE](https://github.com/environmental-forecasting/environmental-stac-generator/blob/main/LICENSE) for more information.
