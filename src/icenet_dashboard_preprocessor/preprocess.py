import logging
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from .stac.generator import STACGenerator
from .utils import (
    flatten_list,
    get_nc_files,
)

logger = logging.getLogger(__name__)


def main(args: SimpleNamespace):
    """
    Main function to generate COGs and generate static JSON STAC catalog.

    This function processes netCDF files and generates cloud-optimized geotiffs (COGs)
    using the given CLI arguments.

    Args:
        args: Parsed Command line arguments. The expected keys are:
            input (List[str]): List of input netCDF files or directories (positional argument).
            --name (str): Name of collection.
            --compress (bool): Whether to compress the output COG files.
            --overwrite (bool): Whether to overwrite existing COG files.
            --flat (bool): Whether to generate a flat STAC catalog for pgSTAC.

    Raises:
        FileNotFoundError: If no valid netCDF files are found for processing.

    Returns:
        None

    Examples:
        For daily forecasting, with a flat STAC structure for pgSTAC
        >>> dashboard preprocess 1days raw_data/*.nc -o -f
    """
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

    stac_generator = STACGenerator()

    for nc_file in (pbar := tqdm(nc_files, desc="COGifying files", leave=True)):  # type: ignore
        pbar.set_description(f"Processing {nc_file}")
        stac_generator.process(
            nc_file=nc_file,
            name=args.name,
            compress=args.no_compress,
            overwrite=args.overwrite,
            forecast_frequency=args.forecast_frequency,
            flat=args.flat,
        )
