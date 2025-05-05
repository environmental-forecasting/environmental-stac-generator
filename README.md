# icenet-geotiff-generator

Converts daily IceNet netCDF prediction outputs to Cloud Optimised GeoTIFF files.

A command-line tool for generating **Cloud Optimized GeoTIFFs (COGs)** and **STAC catalogs** from IceNet forecast NetCDF files. It is outputs COGs in the necessary format to serve them using [icenet-geotiff-generator](https://github.com/icenet-ai/icenet-geotiff-generator) for use with the `icenet-dashboard`.

## Features

- Converts IceNet NetCDF predictions into Cloud Optimized GeoTIFFs (COGs)
- Automatically reprojects to EPSG:3857 for Leaflet/web-map compatibility
- Builds STAC-compliant metadata catalogs for each forecast
- Supports compressed or uncompressed output
- Dynamically detects northern or southern hemisphere from input data

## Installation

To use this tool, ensure the following dependencies are installed:

```bash
pip install xarray rioxarray pystac tqdm
```

## Usage

```bash
icenet_geotiff_generator_gen_cloud_tiffs --input ./results/predict/*.nc
```

| Flag                  | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `--input`, `-i`       | One or more paths to `.nc` files or directories (wildcards supported). |
| `--overwrite`, `-o`   | Overwrite existing GeoTIFF files if they already exist.                |
| `--no_compress`, `-c` | Disable compression in generated GeoTIFFs (default is compressed).     |

## Example

```bash
icenet_geotiff_generator_gen_cloud_tiffs -i results/predict/fc.2024-11-11_north.nc -o
```

This will:
* Detect the hemisphere automatically
* Convert each leadtime slice to a COG
* Process the COG outputs into `data/cogs/{hemisphere}/{date}/`
* Build a hierarchical STAC catalog in `data/stac/`

## License

icenet-geotiff-generator is licensed under the MIT license. See [LICENSE](https://github.com/icenet-ai/icenet-geotiff-generator/blob/main/LICENSE) for more information.
