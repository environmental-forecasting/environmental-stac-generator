import logging
import math
import os
import re
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import orjson
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = os.cpu_count() or 1


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


def hemisphere_from_dataset(ds: xr.Dataset) -> str:
    """
    Get the hemisphere ("north" or "south") from dataset geospatial attributes.

    Args:
        ds: Dataset with a ``geospatial_lat_min`` attribute when available.

    Returns:
        ``"north"``, ``"south"``, or ``""`` if the attribute is missing.

    Raises:
        ValueError: If the minimum latitude value is not within -90 to 90.
    """
    lat_min = ds.attrs.get("geospatial_lat_min", None)

    if lat_min is None:
        logger.warning(
            "netCDF does not contain `geospatial_lat_min`, "
            "cannot determine hemisphere"
        )
        return ""

    if 0 <= lat_min <= 90:
        return "north"
    if -90 <= lat_min < 0:
        return "south"
    raise ValueError(f"Unexpected minimum latitude value: {lat_min}")


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
        return hemisphere_from_dataset(ds)


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


_LEAD_UNIT_ALIASES = {
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days",
    "w": "weeks",
    "week": "weeks",
    "weeks": "weeks",
    "mon": "months",
    "month": "months",
    "months": "months",
    "y": "years",
    "yr": "years",
    "year": "years",
    "years": "years",
}


def _normalise_lead_unit(unit: str) -> str | None:
    token = unit.strip().lower().split()[0]
    return _LEAD_UNIT_ALIASES.get(token)


def infer_lead_unit(leadtime_coords: xr.DataArray, ds: xr.Dataset | None = None) -> str:
    """
    Infer the unit of numeric leadtime / lead_time coordinates.

    Preference order: coordinate ``units`` / encoding, long_name hints, then
    dataset ``time_coverage_resolution`` (ISO-8601), else days with a warning.
    """
    raw = (
        leadtime_coords.attrs.get("units")
        or leadtime_coords.encoding.get("units")
        or ""
    )
    raw = str(raw).strip()
    if raw and "since" not in raw.lower():
        normalised = _normalise_lead_unit(raw)
        if normalised:
            return normalised

    long_name = str(leadtime_coords.attrs.get("long_name") or "").lower()
    for needle, unit in (
        ("hour", "hours"),
        ("day", "days"),
        ("week", "weeks"),
        ("month", "months"),
        ("year", "years"),
    ):
        if needle in long_name:
            return unit

    resolution = ""
    if ds is not None:
        resolution = str(ds.attrs.get("time_coverage_resolution") or "").upper()
    if resolution.startswith("PT") and "H" in resolution:
        return "hours"
    if resolution.startswith("P") and "W" in resolution:
        return "weeks"
    if resolution.startswith("P") and "M" in resolution and "T" not in resolution[:2]:
        # P1M / P3M month periods (not PT…)
        if re.match(r"^P\d*M$", resolution):
            return "months"
    if resolution.startswith("P") and "D" in resolution:
        return "days"

    logger.warning(
        "Leadtime coordinate has no usable units; assuming days "
        "(set CF units on leadtime or provide forecast_date)"
    )
    return "days"


def _offset_timestamp(reference, value: float, unit: str):
    """Add a CF-style lead offset to a forecast reference time."""
    ref = pd.Timestamp(reference)
    if unit == "hours":
        return ref + pd.Timedelta(hours=float(value))
    if unit == "days":
        return ref + pd.Timedelta(days=float(value))
    if unit == "weeks":
        return ref + pd.Timedelta(weeks=float(value))
    if unit == "months":
        return ref + relativedelta(months=int(value))
    if unit == "years":
        return ref + relativedelta(years=int(value))
    raise ValueError(f"Unsupported leadtime unit: {unit}")


def resolve_valid_times(
    forecast_reference_time,
    leadtime_coords: xr.DataArray,
    ds_slice: xr.Dataset | None = None,
) -> list:
    """
    Resolve absolute valid times for each lead index.

    Prefer ``forecast_date`` on the init slice when present (IceNet), then
    datetime/timedelta lead coordinates, else ``init + lead_offset`` using
    inferred units.
    """
    if leadtime_coords is None:
        raise ValueError("Dataset is missing leadtime / lead_time coordinates")

    nlead = int(leadtime_coords.size)
    if nlead < 1:
        raise ValueError("Leadtime coordinate is empty")

    if ds_slice is not None and "forecast_date" in ds_slice.variables:
        fd = ds_slice["forecast_date"]
        # Drop a leftover length-1 time axis if present.
        if "time" in fd.dims and fd.sizes.get("time") == 1:
            fd = fd.isel(time=0)
        values = np.asarray(fd.values).reshape(-1)
        if values.size != nlead:
            raise ValueError(
                f"forecast_date length ({values.size}) != leadtime ({nlead})"
            )
        if not np.issubdtype(values.dtype, np.datetime64):
            units = fd.attrs.get("units") or fd.encoding.get("units")
            if units and "since" in str(units).lower():
                decoded = xr.decode_cf(xr.Dataset({"forecast_date": fd}))[
                    "forecast_date"
                ]
                values = np.asarray(decoded.values).reshape(-1)
            else:
                raise ValueError(
                    "forecast_date is numeric but missing CF 'units' (… since …)"
                )
        return [pd.Timestamp(v).to_pydatetime() for v in values]

    values = np.asarray(leadtime_coords.values).reshape(-1)
    if values.size != nlead:
        values = values.reshape(-1)[:nlead]

    if np.issubdtype(values.dtype, np.datetime64):
        return [pd.Timestamp(v).to_pydatetime() for v in values]

    if np.issubdtype(values.dtype, np.timedelta64):
        ref = pd.Timestamp(forecast_reference_time)
        return [(ref + pd.Timedelta(v)).to_pydatetime() for v in values]

    unit = infer_lead_unit(leadtime_coords, ds_slice)
    return [
        _offset_timestamp(forecast_reference_time, float(v), unit).to_pydatetime()
        for v in values
    ]


def forecast_frequency_from_valid_times(valid_times: list) -> str:
    """
    Infer a compact frequency label (e.g. ``1days``, ``6hours``) from valid times.

    Used for ``config.json`` consistency checks between preprocess runs.
    """
    if len(valid_times) < 2:
        return "1days"

    seconds = []
    for a, b in zip(valid_times, valid_times[1:]):
        delta = pd.Timestamp(b) - pd.Timestamp(a)
        seconds.append(delta.total_seconds())
    median_s = float(np.median(seconds))
    if median_s <= 0:
        raise ValueError("Lead valid times are not strictly increasing")

    hour = 3600.0
    day = 86400.0
    week = 7 * day
    month = 30 * day
    year = 365 * day

    if median_s < day * 0.9:
        step = median_s / hour
        unit = "hours"
    elif median_s < week * 0.9:
        step = median_s / day
        unit = "days"
    elif median_s < month * 0.9:
        step = median_s / week
        unit = "weeks"
    elif median_s < year * 0.9:
        step = median_s / month
        unit = "months"
    else:
        step = median_s / year
        unit = "years"

    if abs(step - round(step)) < 1e-6:
        step_str = str(int(round(step)))
    else:
        step_str = f"{step:g}"
    return f"{step_str}{unit}"


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


def get_array_statistics(values: np.ndarray) -> dict:
    """
    Compute GDAL-style statistics for a NumPy array in a single pass.

    Valid pixels are finite values (not NaN/Inf). Empty or all-invalid arrays
    return ``None`` for min/max/mean/stddev.
    """
    arr = np.asarray(values)
    total_pixels = arr.size
    if total_pixels == 0:
        return {
            "STATISTICS_MINIMUM": None,
            "STATISTICS_MAXIMUM": None,
            "STATISTICS_MEAN": None,
            "STATISTICS_STDDEV": None,
            "STATISTICS_VALID_PERCENT": 0.0,
        }

    finite = np.isfinite(arr)
    valid_pixels = int(finite.sum())
    band_valid_percent = math.floor((100.0 * valid_pixels / total_pixels) * 100) / 100

    if valid_pixels == 0:
        return {
            "STATISTICS_MINIMUM": None,
            "STATISTICS_MAXIMUM": None,
            "STATISTICS_MEAN": None,
            "STATISTICS_STDDEV": None,
            "STATISTICS_VALID_PERCENT": band_valid_percent,
        }

    valid = arr[finite]
    return {
        "STATISTICS_MINIMUM": float(valid.min()),
        "STATISTICS_MAXIMUM": float(valid.max()),
        "STATISTICS_MEAN": float(valid.mean()),
        "STATISTICS_STDDEV": float(valid.std()),
        "STATISTICS_VALID_PERCENT": band_valid_percent,
    }


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
    return get_array_statistics(da.values)


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
