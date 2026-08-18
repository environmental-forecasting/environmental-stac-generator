# environmental-stac-generator

Command-line tool that converts environmental forecast netCDF outputs (e.g. from [icenet](https://github.com/icenet-ai/icenet)) to **Cloud Optimized GeoTIFFs (COGs)** and **STAC catalogues** for use with [environmental-stac-orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator) and [environmental-stac-dashboard](https://github.com/environmental-forecasting/environmental-stac-dashboard).

## Installation

```bash
pip install -e .
```

## Quick start

```bash
envstacgen preprocess ./results/predict/*.nc -o -n my_collection
envstacgen ingest --env-file .env.dev data/stac/catalog.json -o
```

`--env-file` is only needed for **ingest** (database credentials and `FILE_SERVER_URL`). Preprocess writes a portable static catalogue with cwd-relative asset paths.

## Documentation

Full CLI reference, netCDF layout requirements, and more:

```bash
make docs-install
make docs
```

Then open http://127.0.0.1:8000.

## License

`environmental-stac-generator` is licensed under the MIT license. See [LICENSE](https://github.com/environmental-forecasting/environmental-stac-generator/blob/main/LICENSE) for more information.
