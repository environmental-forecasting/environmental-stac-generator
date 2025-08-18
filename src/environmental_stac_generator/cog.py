import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

logger = logging.getLogger(__name__)


def write_cog(
    cog_path: Path,
    da: xr.DataArray,
    compress: str = "DEFLATE",
    block_size: int = 256,
    overview_level: int = 4,
) -> None:
    profile = cog_profiles.get("deflate")
    profile.update(
        {
            "compress": compress,
            "blockxsize": block_size,
            "blockysize": block_size,
        }
    )

    # Create tmp file to add stats to using rasterio (requires an actual file
    # won't work with memfile)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        da.rio.to_raster(tmp_path, driver="GTiff", compress=compress)

        # Embed band-level statistics using rasterio
        with rasterio.open(tmp_path, "r+") as src:
            for i in range(1, src.count + 1):
                data = src.read(i, masked=True)
                stats_dict = {
                    "STATISTICS_MINIMUM": float(np.nanmin(data)),
                    "STATISTICS_MAXIMUM": float(np.nanmax(data)),
                    "STATISTICS_MEAN": float(np.nanmean(data)),
                    "STATISTICS_STDDEV": float(np.nanstd(data)),
                }
                src.update_tags(i, **stats_dict)

        # Use rio-cogeo to convert to COG with internal overviews
        with rasterio.open(tmp_path) as src:
            cog_translate(
                source=src,
                dst_path=cog_path,
                dst_kwargs=profile,
                overview_level=4,
                overview_resampling="average",
                forward_band_tags=True,
                in_memory=True,
                quiet=True,
            )
    finally:
        tmp_path.unlink(missing_ok=True)
