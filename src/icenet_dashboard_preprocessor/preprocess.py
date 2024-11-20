import argparse
import datetime as dt
import logging
import logging.config
import time
from pathlib import Path, PosixPath

import rioxarray
import xarray as xr
from tqdm import tqdm

from .utils import flatten_list

logger = logging.getLogger(__name__)


def get_hemisphere(netcdf_file: str) -> str:
    """
    Get the hemisphere (either "north" or "south") of the given netCDF file based on its minimum latitude value.
    Args:
        netcdf_file: Path to a netCDF file.
                     It must have geospatial information in its attributes.
    Returns:
        str: The hemisphere associated with the given netCDF file ("north" or "south").
    Raises:
        ValueError: If the minimum latitude value is not within the expected range (-90 to 90).
    Examples:
        >>> get_hemisphere("/path/to/north_atlantic_climate.nc")
        'north'

        >>> get_hemisphere("/path/to/south_pacific_ocean.nc")
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


def generate_cloud_tiff(
    nc_file: str | Path, compress: bool = True, reproject: bool = True, overwrite=False
) -> None:
    """Generates Cloud Optimised GeoTIFFs from IceNet prediction netCDF files.
    Args:
        compress: Whether to compress the output GeoTIFFs.
                    Default is True.
        reproject: Whether to re-project the output GeoTIFFs to match the default of Leaflet.
                    Default is True.
    """
    compress = "DEFLATE" if compress else "NONE"
    cogs_output_dir = Path("data") / "cogs"
    hemisphere = get_hemisphere(nc_file)

    with xr.open_dataset(nc_file) as ds:
        # Convert eastings and northings from kilometers to metres.
        ds = ds.assign_coords(xc=ds.coords["xc"] * 1000, yc=ds.coords["yc"] * 1000)

        if reproject:
            ds = ds.drop_vars(["lat", "lon"])

        ds = ds.rename({"xc": "x", "yc": "y"})

        # Rearrange array dimension to match rioxarray expectation
        sic_variable = (
            ds["sic_mean"].transpose("time", "leadtime", "y", "x").isel(time=0)
        )

        # Get attributes from NetCDF file
        nc_attrs = ds.attrs
        crs = nc_attrs["geospatial_bounds_crs"]
        forecast_start_time_str = nc_attrs["time_coverage_start"]
        n_leadtime = sic_variable.leadtime.size
        sic_variable.rio.write_crs(crs, inplace=True)

        # Convert forecast_start_time from "2024-08-31T00:00:00" string to datetime object
        forecast_start_time = dt.datetime.strptime(
            forecast_start_time_str, "%Y-%m-%dT%H:%M:%S"
        )
        forecast_start_date = forecast_start_time.date()

        cog_dir = Path(cogs_output_dir / f"{hemisphere}/{forecast_start_date}")
        cog_dir.mkdir(parents=True, exist_ok=True)

        for i in (pbar := tqdm(range(n_leadtime), desc="COGifying files", leave=True)):
            cog_file = cogs_output_dir / f"north/{forecast_start_date}/leadtime_{i}.tif"
            if cog_file.exists() and not overwrite:
                pbar.set_description(f"File already exists, skipping: {cog_file}")
                time.sleep(0.01)
                continue
            else:
                pbar.set_description(f"Saving to COG: {cog_file}")

            time_slice = sic_variable.isel(leadtime=i)

            # Reproject to EPSG:3857 (WGS 84 / Web Mercator)
            # (Using leaflet's default)
            if reproject:
                reprojected_slice = time_slice.rio.reproject("EPSG:3857")
            else:
                reprojected_slice = time_slice

            # Save as COG (Cloud Optimized GeoTIFF)
            reprojected_slice.rio.to_raster(
                cog_file,
                driver="COG",
                compress=compress,
            )


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
        "--compress",
        action="store_true",
        default=True,
        help="Enable COG compression (default is uncompressed)",
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

    for nc_file in (pbar := tqdm(nc_files, desc="COGifying files", leave=True)):
        pbar.set_description(f"Processing {nc_file}")
        generate_cloud_tiff(nc_file, args.compress, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
