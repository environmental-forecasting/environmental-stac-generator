# icenet-dashboard-preprocessor

Converts daily IceNet netCDF prediction outputs to COG files + STAC catalogs for `icenet-dashboard`.

A command-line tool for generating **Cloud Optimized GeoTIFFs (COGs)** and **STAC catalogs** from IceNet forecast NetCDF files. It is outputs COGs in the necessary format to serve them using [icenet-dashboard-preprocessor](https://github.com/icenet-ai/icenet-dashboard-preprocessor) for use with the `icenet-dashboard`.

## Features

- Converts IceNet NetCDF predictions into Cloud Optimized GeoTIFFs (COGs)
- Automatically reprojects to EPSG:3857 for Leaflet/web-map compatibility
- Builds STAC-compliant metadata catalogs for each forecast
- Supports compressed or uncompressed output
- Dynamically detects northern or southern hemisphere from input data

## Installation

To use this tool, ensure the dependencies from [pyproject.toml](pyproject.toml) are installed. Or, for an editable install, clone the repo and run the following after changing directory to the repo root:

```bash
pip install -e .
```

## Usage

```bash
dashboard preprocess 1days ./results/predict/*.nc
```

### Positional Parameters

The first parameter is the forecast frequency.

The second and further arguments are the paths to one or more `.nc` files.

### Options

The optional flags that can be used are:

| Flag                  | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `--overwrite`, `-o`   | Overwrite existing GeoTIFF files if they already exist.                |
| `--no_compress`, `-c` | Disable compression in generated GeoTIFFs (default is compressed).     |

## Example

```bash
dashboard preprocess 1days raw_data/*.nc -o
```

This will:
* Detect the hemisphere automatically
* Convert each leadtime slice to a COG
* Process the COG outputs into `data/cogs/{hemisphere}/{date}/`
* Build a hierarchical STAC catalog in `data/stac/`

## License

`icenet-dashboard-preprocessor` is licensed under the MIT license. See [LICENSE](https://github.com/icenet-ai/icenet-dashboard-preprocessor/blob/main/LICENSE) for more information.
