---
icon: lucide/terminal
---

# CLI

`envstacgen` has two subcommands: **preprocess** (netCDF to COGs and a static STAC catalogue) and **ingest** (load that catalogue into pgSTAC).

## preprocess

Generate COGs and a static JSON STAC catalogue. Input files must follow the [netCDF layout](netcdf-layout.md).

### Arguments

Paths to one or more `.nc` files, directories, or wildcard patterns.

Lead valid times are inferred from the netCDF (`forecast_date` when present, otherwise `leadtime` / `lead_time` offsets). A compact frequency label is stored in `data/config.json` for consistency checks between runs.

### Options

| Flag | Description |
| ---- | ----------- |
| `--overwrite`, `-o` | Overwrite existing GeoTIFF files if they already exist |
| `--no-compress`, `-c` | Disable compression in generated GeoTIFFs (default is compressed) |
| `--name`, `-n` | STAC collection name (default: `default`) |
| `--workers`, `-w` | Max concurrent workers (default: CPU count) |
| `--stac-only`, `-s` | Output only the STAC files, not COGs/thumbnails |

### Example

```bash
envstacgen preprocess raw_data/*.nc -o -n icenet_0.2
```

This will:

- Detect the hemisphere automatically
- Convert each leadtime slice to a COG
- Write COGs under `data/cogs/{collection}/{date}/`
- Build a hierarchical STAC catalogue in `data/stac/` with portable asset hrefs

## ingest

Load a generated JSON STAC catalogue from the `envstacgen preprocess` command into a PostgreSQL/pgSTAC database (Database set-up by the [orchestrator](https://github.com/environmental-forecasting/environmental-stac-orchestrator) stack).

### Arguments

| Argument | Description |
| -------- | ----------- |
| `catalog` | Path to the STAC catalogue JSON file (e.g. `data/stac/catalog.json`) |

### Options

| Flag | Description |
| ---- | ----------- |
| `--env-file` | Path to an environment file (e.g. `.env.development`) for database credentials and `FILE_SERVER_URL`. Falls back to `.env` if present, otherwise process environment variables. |
| `--overwrite`, `-o` | Overwrite existing matching entries (default is to skip matches) |

### Example

```bash
envstacgen ingest --env-file .env.development data/stac/catalog.json -o
```

This will:

- Prefix asset hrefs with `FILE_SERVER_URL` for this environment
- Ingest the catalogue into PostgreSQL/pgSTAC
