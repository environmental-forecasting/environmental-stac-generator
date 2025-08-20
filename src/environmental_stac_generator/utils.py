import logging
import math
import re
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import orjson
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


def format_time(datetime: dt, utc: bool=True, with_seconds: bool=True) -> str:
    """
    Format a datetime object into a filename-safe ISO-like string.

    This function formats the datetime using hyphens instead of colons,
    making it safe for use in filenames and S3 keys. It optionally includes
    seconds and appends a 'Z' to indicate UTC time.

    Args:
        datetime: The datetime object to format.
        utc (optional): If True, appends 'Z' to indicate UTC.
            Defaults to True.
        with_seconds (optional): If True, includes seconds in the output.
            Defaults to True.

    Returns:
        A formatted datetime string, e.g. "2025-08-14T06-00-00Z".
    """
    fmt = "%Y-%m-%dT%H-%M" + ("-%S" if with_seconds else "")
    result = datetime.strftime(fmt)
    return result + "Z" if utc else result


def get_da_statistics(da: xr.DataArray) -> dict:
    """
    Compute statistics for a given xr.DataArray.

    Calculates basic statistical values such as minimum, maximum,
    mean, standard deviation, and the percentage of valid pixels (i.e., non-NaN)
    in the input DataArray. If the DataArray is empty (size 0), corresponding
    statistics are set to `None`.

    Args:
        da: DataArray containing data for which statistics should be computed.

    Returns:
        A dictionary with the following keys and values:

        - "STATISTICS_MINIMUM": Minimum value of the array, as a float or None if empty.
        - "STATISTICS_MAXIMUM": Maximum value of the array, as a float or None if empty.
        - "STATISTICS_MEAN": Mean (average) value of the array, as a float or None if empty.
        - "STATISTICS_STDDEV": Standard deviation of the array, as a float or None if empty.
        - "STATISTICS_VALID_PERCENT": Percentage of valid pixels in the array,
          formatted as a string with two decimal places.

    Notes:
        - Valid pixels are defined as those that are finite (i.e., not NaN or infinity).
        - When the DataArray is empty, all statistics except `STATISTICS_VALID_PERCENT`
          will be set to `None`.
    """
    # Compute variable statistics
    valid_mask = np.isfinite(da.values)
    valid_pixels = valid_mask.sum()
    total_pixels = da.size
    band_min = float(da.min(skipna=True).item()) if da.size > 0 else None
    band_max = float(da.max(skipna=True).item()) if da.size > 0 else None
    band_mean = float(da.mean(skipna=True).item()) if da.size > 0 else None
    band_stddev = float(da.std(skipna=True).item()) if da.size > 0 else None
    band_valid_percent = float(100.0 * valid_pixels / total_pixels)
    # Round to 2dp
    band_valid_percent = math.floor(band_valid_percent * 100) / 100

    return {
        "STATISTICS_MINIMUM": band_min,
        "STATISTICS_MAXIMUM": band_max,
        "STATISTICS_MEAN": band_mean,
        "STATISTICS_STDDEV": band_stddev,
        # What percent of the pixels in a band are valid (i.e., non-NaN / non-nodata)
        "STATISTICS_VALID_PERCENT": band_valid_percent,
    }


def is_jsonable(x):
    """
    Check if a value can be serialised to JSON
    """
    try:
        orjson.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def get_nc_attributes(nc_attrs):
    # Add all attributes found to metadata if it can be
    # serialised.
    metadata = {}
    if nc_attrs:
        for key, attr in nc_attrs.items():
            if is_jsonable(attr):
                metadata[key] = attr
    return metadata
