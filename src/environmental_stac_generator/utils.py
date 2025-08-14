import logging
import re
from pathlib import Path
from datetime import datetime as dt

import xarray as xr
from dateutil.tz import tzutc
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)


def find_coord(ds: xr.Dataset, possible_names: list[str]) -> str | None:
    """
    Find coordinate name from a list of possible options in the given dataset.

    Args:
        ds: The dataset to search for coordinates.
        possible_names: A list of possible coordinate names.

    Returns:
        The first matching coordinate name, or None if no match is found.
    """
    for name in possible_names:
        if name in ds.coords:
            return name
    return None


def flatten_list(lst):
    """Flatten a list of lists (or tuples)"""
    return [
        item
        for sublist in lst
        for item in (
            flatten_list(sublist)
            if isinstance(sublist, list) or isinstance(sublist, tuple)
            else [sublist]
        )
    ]


def get_hemisphere(netcdf_file: Path) -> str:
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
            logger.warning("netCDF does not contain `geospatial_lat_min`, "
                        "cannot determine hemisphere")
            return ""

        if 0 <= lat_min <= 90:
            return "north"
        elif -90 <= lat_min < 0:
            return "south"
        else:
            raise ValueError(f"Unexpected minimum latitude value: {lat_min}")


def get_nc_files(location: str | Path, extension="nc") -> list[Path] | Path | None:
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


def parse_forecast_frequency(forecast_frequency: str) -> (float, str):
    """
    Parse forecast frequency strings like "2hours", "3days", "2weeks", "1months", "0.5years".

    The function extracts the numeric value and unit from the input string,
    supporting hours (hours), days (days), weeks (weeks), months (months),
    and years (years) units.

    Args:
        forecast_frequency: Frequency of the forecast leadtime in the format "<value><unit>"

    Returns:
        Tuple containing the forecast step size and unit as strings.

    Raises:
        ValueError: If the input string does not match the expected format.

    Examples:
        >>> parse_forecast_frequency("2hours")
        (2.0, 'hours')
        >>> parse_forecast_frequency("3days")
        (3.0, 'days')
        >>> parse_forecast_frequency("1months")
        (1.0, 'months')
        >>> parse_forecast_frequency("0.5years")
        (0.5, 'years')
    """
    match = re.match(
        r"^\s*([0-9]*\.?[0-9]+)\s*(hours?|days?|weeks?|months?|years?)\s*$",
        forecast_frequency.lower(),
        re.IGNORECASE,
    )
    if match:
        value, unit = match.groups()
        return float(value), unit
    else:
        raise ValueError(f"Invalid leadtime format: {forecast_frequency}")


def proj_to_geo(bbox_projected: list[float], src_crs: str) -> list[float]:
    """Convert a projection to geographic coordinates"""

    bbox = transform_bounds(src_crs, CRS.from_epsg(4326), *bbox_projected)  # type: ignore

    return bbox

def ensure_utc(datetime: dt) -> dt:
    """
    Ensures a datetime object is timezone-aware in UTC.

    If the input datetime is None, returns None. If the datetime is naive
    (no timezone info), attaches UTC timezone. If already timezone-aware,
    converts to UTC equivalent.

    Args:
        dt: A datetime object, or None.

    Returns:
        datetime: The datetime object with UTC timezone, or None if input was None.
    """
    if datetime is None:
        return None
    elif datetime.tzinfo is None:
        return datetime.replace(tzinfo=tzutc())
    return datetime.astimezone(tzutc())
