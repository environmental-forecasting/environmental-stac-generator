import logging
from pathlib import Path

import xarray as xr
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)


def write_cog(cog_path: Path, da: xr.DataArray, compress: str = "DEFLATE") -> None:
    blocksize = 256
    overview_level = 4
    # Write the DataArray to an in-memory GeoTIFF using rioxarray
    with MemoryFile() as memfile:
        da.rio.to_raster(memfile.name, driver="GTiff", compress=compress)

        with memfile.open("r+") as src_dst:
            profile = cog_profiles.get("deflate")
            profile.update(
                {
                    "compress": compress,
                    "blockxsize": blocksize,
                    "blockysize": blocksize,
                }
            )

            # Use rio-cogeo to convert to a proper Cloud Optimized GeoTIFF
            cog_translate(
                src_dst,
                cog_path,
                profile,
                overview_level=overview_level,
                overview_resampling="average",
                in_memory=True,
                quiet=True,
            )
