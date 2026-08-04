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
    input: list[str],
    name: str,
    workers: int,
    overwrite: bool,
    compress: bool,
    stac_only: bool,
):
    """
    Main function to generate COGs and generate static JSON STAC catalog.

    This function processes netCDF files and generates cloud-optimised GeoTIFFs (COGs)
    using the given CLI arguments. Asset hrefs in the static catalog are cwd-relative
    (portable); apply ``FILE_SERVER_URL`` at ingest.

    Args:
        input: List of input netCDF files or directories.
        name: Collection name.
        workers: Max number of concurrent workers.
        overwrite: Whether to overwrite existing COG files.
        compress: Whether to compress COG output.
        stac_only: Output only the STAC files.

    Raises:
        FileNotFoundError: If no valid netCDF files are found for processing.

    Returns:
        None

    Examples:
        >>> envstacgen preprocess raw_data/*.nc -o --name icenet
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
        raise FileNotFoundError("No files provided. Please specify which files to convert.")

    missing = [f for f in nc_files if not f.exists()]
    for f in missing:
        logger.warning(f"File {f} does not exist")
    nc_files = [f for f in nc_files if f.exists()]

    if nc_files:
        logger.info(f"Found {len(nc_files)} netCDF files")
        logger.debug(f"Processing {nc_files}")
    else:
        logger.warning("No netCDF files found for processing")
        raise FileNotFoundError(f"{input} is invalid")

    stac_generator = STACGenerator()

    try:
        for nc_file in (pbar := tqdm(nc_files, desc="COGifying files", leave=True)):  # type: ignore
            pbar.set_description(f"Processing {nc_file}")
            stac_generator.process(
                nc_file=nc_file,
                name=name,
                compress=compress,
                overwrite=overwrite,
                stac_only=stac_only,
                workers=workers,
            )
    finally:
        # Persist once after the batch (not per file) to avoid O(N^2) catalog I/O.
        # Still run on failure so partial progress from earlier files is kept.
        try:
            stac_generator.save_catalog()
        finally:
            stac_generator.close_executor()
