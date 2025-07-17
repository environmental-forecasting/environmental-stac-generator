import argparse
import re

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


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate Cloud Optimized GeoTIFFs (COGs) from IceNet prediction netCDF files."
    )

    # Required argument: The frequency of the forecast lead time
    # TODO: This should be picked up from `forecast_period` variable (which doesn't exist in icenet)
    parser.add_argument(
        "forecast_frequency",
        type=str,
        help="The forecast frequency (e.g., 6hours, 1days, 2months, 1years). Units: hours, days, months, years",
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