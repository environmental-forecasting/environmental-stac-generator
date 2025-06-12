import argparse
import datetime as dt
import json
import logging
import logging.config
import time
from pathlib import Path, PosixPath

import numpy as np
import pystac
import rioxarray
import xarray as xr
from pystac import Asset, Catalog, Collection, Item
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension
from shapely.geometry import box, mapping
from tqdm import tqdm

from .stac import IceNetSTAC
from .utils import flatten_list

logger = logging.getLogger(__name__)


def get_hemisphere(netcdf_file: str) -> str:
    """
    Get the hemisphere (either "north" or "south") of the given netCDF file based on its minimum latitude value.
    Args:
        netcdf_file: Path to a netCDF file.
                     It must have geospatial information in its attributes.
    Returns:
        The hemisphere associated with the given netCDF file ("north" or "south").
    Raises:
        ValueError: If the minimum latitude value is not within the expected range (-90 to 90).
    Examples:
        >>> get_hemisphere("results/predict/fc.2024-11-11_north.nc")
        'north'

        >>> get_hemisphere("results/predict/fc.2024-11-11_south.nc")
        'south'
    """
    with xr.open_dataset(netcdf_file) as ds:
        # Extract the minimum latitude value from the dataset's attributes
        lat_min = ds.attrs.get("geospatial_lat_min", None)

        if lat_min is None:
            raise ValueError("NetCDF file does not contain geospatial information.")

        if 0 <= lat_min <= 90:
            return "north"
        elif -90 <= lat_min < 0:
            return "south"
        else:
            raise ValueError(f"Unexpected minimum latitude value: {lat_min}")


def get_nc_files(
    location: str | PosixPath, extension="nc"
) -> list[PosixPath] | PosixPath:
    """Get a list of NetCDF files located at the given `location`.

    Args:
        location: The path to check for NetCDF files.
        extension: The file extension to filter by.
                    Defaults to "nc".
    Returns:
        A list of NetCDF file paths if `location` is a directory, or the single NetCDF file
             path if `location` is a file. If `location` is invalid, returns None.
    Raises:
        FileNotFoundError: If `location` does not exist.
        NotADirectoryError: If `location` is a file and no matching file with the given extension exists.
    Examples:
        >>> get_nc_files("/path/to/netcdf/files")
        [<PosixPath('/path/to/netcdf/files/file1.nc')>, <PosixPath('/path/to/netcdf/files/file2.nc')>]

        >>> get_nc_files("/path/to/single/file.nc")
        <PosixPath('/path/to/single/file.nc')>
    """
    p = Path(location)

    if p.is_dir():
        # Return all NetCDF files if directory specified
        return list(p.glob(f"*.{extension}"))
    elif p.is_file() and p.suffix.lower() == f".{extension}":
        # Return file path if file is specified and matches the given extension.
        return p.resolve()
    else:
        logger.error(
            f"Location {location} is invalid or does not contain a matching file with the given extension."
        )
        # raise FileNotFoundError if not p.exists() else NotADirectoryError


def get_or_create_catalog(stac_catalog_path: str | Path, catalog_defs: dict) -> Catalog:
    if stac_catalog_path.exists():
        return Catalog.from_file(stac_catalog_path)
    return Catalog(
        id=catalog_defs["id"],
        description=catalog_defs["description"],
        title=catalog_defs["title"],
    )


def get_or_create_collection(parent, collection_id, description, bbox, temporal_extent) -> Collection:
    collection = next((c for c in parent.get_children() if c.id == collection_id), None)
    if not collection:
        collection = Collection(
            id=collection_id,
            description=description,
            extent=pystac.Extent(
                pystac.SpatialExtent([bbox]),
                pystac.TemporalExtent([temporal_extent]),
            ),
        )
        parent.add_child(collection)
    return collection


def generate_cloud_tiff(
    nc_file: str | Path, compress: bool = True, overwrite=False, freq="D"
) -> None:
    """Generates Cloud Optimized GeoTIFFs and STAC Catalogs from IceNet prediction netCDF files.

    Args:
        compress: Whether to compress the output GeoTIFFs.
                    Default is True.
        overwrite: Whether to overwrite existing outputs.
                    Default is False.
    """
    compress = "DEFLATE" if compress else "NONE"
    cogs_output_dir = Path("data") / "cogs"
    stac_output_dir = Path("data") / "stac"
    nc_file = Path(nc_file).resolve()
    hemisphere = get_hemisphere(nc_file)

    with xr.open_dataset(nc_file) as ds:
        n_leadtime = ds["leadtime"].values
        lat = ds["lat"].values
        lon = ds["lon"].values

        # Compute bounding box and geometry from lat/lon
        bbox = [float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))]
        geometry = mapping(box(*bbox))

        # Filter 4D variables with dims (time, yc, xc, leadtime)
        valid_bands = [
            var for var in ds.data_vars
            if set(ds[var].dims) >= {'time', 'yc', 'xc', 'leadtime'}
        ]

        ds_bands = xr.concat([ds[var] for var in valid_bands], dim='band')

        # Assign band names as a coordinate (optional but useful for metadata)
        ds_bands = ds_bands.assign_coords(band=("band", valid_bands))

        # Convert eastings and northings from kilometers to metres.
        ds_bands = ds_bands.assign_coords(xc=ds.coords["xc"] * 1000, yc=ds.coords["yc"] * 1000)
        ds_bands = ds_bands.rename({"xc": "x", "yc": "y"})

        # Initialise STAC Catalog
        stac_catalog_path = stac_output_dir / "catalog.json"
        catalog_defs = {
                "id": "forecast-data",
                "description": "Catalog of IceNet Forecast Data",
                "title": "IceNet Forecast STAC Catalog",
        }
        catalog = get_or_create_catalog(stac_catalog_path, catalog_defs)

        # Get attributes from NetCDF file
        nc_attrs = ds.attrs
        crs = nc_attrs["geospatial_bounds_crs"]
        ds_bands.rio.write_crs(crs, inplace=True)

        for time in ds["time"].values:
            # Rearrange array dimension to match rioxarray expectation
            ds_variable = (
                ds_bands.sel(time=time)
            )

            # Get attributes from NetCDF file
            forecast_start_time_str = nc_attrs["time_coverage_start"]
            forecast_end_time_str = nc_attrs["time_coverage_end"]

            # Convert forecast_start_time from "2024-08-31T00:00:00" string to datetime object
            forecast_start_time = dt.datetime.strptime(
                forecast_start_time_str, "%Y-%m-%dT%H:%M:%S"
            )
            forecast_start_date = forecast_start_time.date()
            forecast_end_time = dt.datetime.strptime(
                forecast_end_time_str, "%Y-%m-%dT%H:%M:%S"
            )
            forecast_end_date = forecast_end_time.date()

            cog_dir = Path(cogs_output_dir / f"{hemisphere}/{forecast_start_date}")
            cog_dir.mkdir(parents=True, exist_ok=True)

            # Create (or retrieve) a hemisphere collection within the catalog
            hemisphere_collection = get_or_create_collection(
                parent=catalog,
                collection_id=hemisphere,
                description=f"{hemisphere.capitalize()} hemisphere collection",
                bbox=bbox,
                temporal_extent=[forecast_start_time, forecast_end_time],
            )

            # Create (or retrieve) a forecast collection within the catalog
            forecast_collection = get_or_create_collection(
                parent=hemisphere_collection,
                collection_id=f"forecast-{forecast_start_date}",
                description=f"Forecast data for {forecast_start_date}",
                bbox=bbox,
                temporal_extent=[forecast_start_time, forecast_end_time],
            )

            # Process each leadtime
            for i in (pbar := tqdm(range(len(n_leadtime)), desc="COGifying files", leave=True)):
                cog_file = cog_dir / f"{nc_file.stem}_day_{i}.tif"
                if cog_file.exists() and not overwrite:
                    pbar.set_description(f"File already exists, skipping: {cog_file}")
                    continue
                else:
                    pbar.set_description(f"Saving to COG: {cog_file}")

                time_slice = ds_variable.isel(leadtime=i)

                # Save as COG (Cloud Optimized GeoTIFF)
                time_slice.rio.to_raster(
                    cog_file,
                    driver="COG",
                    compress=compress,
                )

                # Add STAC Item for this file
                item = Item(
                    id=f"leadtime-{i}",
                    geometry=geometry,
                    bbox=bbox,
                    datetime=forecast_start_time + dt.timedelta(days=i),
                    properties={
                        "forecast_start_date": str(forecast_start_date),
                        "hemisphere": hemisphere,
                        "leadtime": i,
                    },
                )

                # Add projection extension
                ProjectionExtension.add_to(item)
                proj = ProjectionExtension.ext(item)
                proj.epsg = int(crs.split(":")[-1]) if "EPSG" in crs else None

                # Add COG asset
                item.add_asset(
                    "geotiff",
                    pystac.Asset(
                        href=str(cog_file),
                        media_type=pystac.MediaType.COG,
                        title=f"GeoTIFF for {hemisphere} with leadtime {i}",
                    ),
                )

                # Add item to collection
                forecast_collection.add_item(item)

        # Save catalog and collections
        stac_output_dir.mkdir(parents=True, exist_ok=True)
        catalog.normalize_hrefs(
            str(stac_output_dir),
        )
        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    return


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate Cloud Optimized GeoTIFFs (COGs) from IceNet prediction netCDF files."
    )

    # Optional argument: input file or filename path pattern with wildcard
    parser.add_argument(
        "-i",
        "--input",
        nargs="*",
        help="Input directory or filename path pattern with wildcard (e.g., ./results/predict/*.nc)",
    )

    # Optional boolean flag to force overwrite of existing files
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Enable overwriting of existing COGs",
    )

    # Optional boolean flag to disable COG compression
    parser.add_argument(
        "-c",
        "--no_compress",
        action="store_true",
        default=True,
        help="Disable COG compression (default is compressed)",
    )

    return parser.parse_args()


def main():
    """
    The main function of the Icenet Dashboard Preprocessor.

    This function processes netCDF files and generates cloud-optimized geotiffs (COGs)
    using the given CLI arguments.

    Args:
        args (Namespace): Command line argument parser. The expected keys are:
            --input (List[str]): List of input netCDF files or directories.
            --compress (bool): Whether to compress the output COG files.
            --overwrite (bool): Whether to overwrite existing COG files.
    Raises:
        FileNotFoundError: If no valid netCDF files are found for processing.
    Returns:
        None
    """
    args = get_args()
    logger.debug(f"Command line input arguments: {args}")
    if args.input is None:
        default_dir = "results/predict"
        logger.warning(f"No input specified, search default location: {default_dir}")
        nc_files = get_nc_files("results/predict/")
    elif len(args.input) == 1:
        nc_files = flatten_list(
            list(filter(None, (get_nc_files(f) for f in args.input)))
        )
    else:
        nc_files = [Path(f) for f in args.input]

    if not nc_files:
        logger.error("No files provided... Please specify which files to convert.")
        exit(1)

    for nc_file in nc_files:
        if not nc_file.exists():
            nc_files.remove(nc_file)
            logger.warning(f"File {nc_file} does not exist")

    if nc_files:
        logger.info(f"Found {len(nc_files)} netCDF files")
        logger.debug(f"Processing {nc_files}")
    else:
        logger.warning("No netCDF files found for processing")
        raise FileNotFoundError(f"{args.input} is invalid")

    # manifest_entries = []
    for nc_file in (pbar := tqdm(nc_files, desc="COGifying files", leave=True)):
        pbar.set_description(f"Processing {nc_file}")
        generate_cloud_tiff(nc_file, args.no_compress, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
