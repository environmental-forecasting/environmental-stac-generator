import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


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
            raise ValueError("NetCDF file does not contain geospatial information.")

        if 0 <= lat_min <= 90:
            return "north"
        elif -90 <= lat_min < 0:
            return "south"
        else:
            raise ValueError(f"Unexpected minimum latitude value: {lat_min}")


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
