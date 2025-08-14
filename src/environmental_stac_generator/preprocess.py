import logging
from pathlib import Path

from tqdm import tqdm

from .stac.generator import STACGenerator
from .utils import (
    flatten_list,
    get_nc_files,
)

logger = logging.getLogger(__name__)


def main(
    forecast_frequency: str,
    input: list[str],
    name: str,
    workers: int,
    overwrite: bool,
    no_compress: bool,
    stac_only: bool,
):
    """
    Main function to generate COGs and generate static JSON STAC catalog.

    This function processes netCDF files and generates cloud-optimized geotiffs (COGs)
    using the given CLI arguments.

    Args:
        forecast_frequency: The frequency of forecasts.
        input: List of input netCDF files or directories.
        name: Collection name.
        workers: Max number of concurrent workers.
        overwrite: Whether to overwrite existing COG files.
        no_compress: Disable COG compression.
        stac_only: Output only the STAC files.

    Raises:
        FileNotFoundError: If no valid netCDF files are found for processing.

    Returns:
        None

    Examples:
        To output a catalog for daily forecasting:
        >>> dashboard preprocess 1days raw_data/*.nc -o -f --name icenet
    """
    if input is None:
        default_dir = "results/predict"
        logger.warning(f"No input specified, search default location: {default_dir}")
        nc_files = get_nc_files("results/predict/")
    elif len(input) == 1:
        nc_files = flatten_list(
            list(filter(None, (get_nc_files(f) for f in input)))
        )
    else:
        nc_files = [Path(f) for f in input]

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
        raise FileNotFoundError(f"{input} is invalid")

    stac_generator = STACGenerator()

    for nc_file in (pbar := tqdm(nc_files, desc="COGifying files", leave=True)):  # type: ignore
        pbar.set_description(f"Processing {nc_file}")
        stac_generator.process(
            nc_file=nc_file,
            name=name,
            compress=no_compress,
            overwrite=overwrite,
            forecast_frequency=forecast_frequency,
            stac_only=stac_only,
            workers=workers,
        )
