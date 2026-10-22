# environmental-stac-generator

[![Tests](https://github.com/environmental-forecasting/environmental-stac-generator/actions/workflows/test.yml/badge.svg)](https://github.com/environmental-forecasting/environmental-stac-generator/actions/workflows/test.yml)
[![Lint](https://github.com/environmental-forecasting/environmental-stac-generator/actions/workflows/lint.yml/badge.svg)](https://github.com/environmental-forecasting/environmental-stac-generator/actions/workflows/lint.yml)
[![CodeQL](https://github.com/environmental-forecasting/environmental-stac-generator/actions/workflows/codeql.yml/badge.svg)](https://github.com/environmental-forecasting/environmental-stac-generator/actions/workflows/codeql.yml)
[![Docs](https://github.com/environmental-forecasting/environmental-stac-generator/actions/workflows/docs.yml/badge.svg)](https://environmental-forecasting.github.io/environmental-stac-generator/)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/environmental-forecasting/environmental-stac-generator/badge)](https://scorecard.dev/viewer/?uri=github.com/environmental-forecasting/environmental-stac-generator)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


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
