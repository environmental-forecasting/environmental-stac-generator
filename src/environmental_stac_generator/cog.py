import logging
import shutil
import subprocess
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
    external_overviews: bool = False,
) -> None:
    """
    Write a Cloud Optimized GeoTIFF (COG) from an xarray DataArray.

    Embeds band-level statistics, then converts to a COG with internal
    overviews via ``rio-cogeo``. External ``.ovr`` sidecars are off by
    default (COG internal overviews are enough for TiTiler / STAC Browser);
    pass ``external_overviews=True`` if a workflow still needs them.

    Args:
        cog_path: Path where the final COG will be saved as a GeoTIFF file.
        da: xr.DataArray containing geospatial data. Must have a valid
            coordinate reference system (CRS) and spatial extent.
        compress: Compression method to use for the COG.
            Defaults to "DEFLATE".
        block_size: Block size (in pixels) used for tiling.
            Defaults to 256.
        overview_level: Number of overviews to generate. This defines how many
            downsampled versions of the raster will be created.
            Defaults to 4.
        external_overviews: If True, also builds external ``.ovr`` files with
            ``gdaladdo`` (extra preprocess cost; not required for COG readers).
            Defaults to False.

    Notes:
        - Band-level statistics (minimum, maximum, mean, standard deviation)
          are embedded within the output COG using rasterio.

        - ``external_overviews=True`` requires GDAL (``gdaladdo``) on PATH.
    """
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

        # Optional external overviews on the temp GeoTIFF (moved after COG write).
        if external_overviews:
            subprocess.run(
                [
                    "gdaladdo",
                    "-q",
                    "-ro",
                    str(tmp_path),
                    "2",
                    "4",
                    "8",
                    "16",
                ],
                check=True,
            )

        # Use rio-cogeo to convert to COG with internal overviews
        with rasterio.open(tmp_path) as src:
            cog_translate(
                source=src,
                dst_path=cog_path,
                dst_kwargs=profile,
                overview_level=overview_level,
                overview_resampling="average",
                forward_band_tags=True,
                in_memory=True,
                quiet=True,
            )

        if external_overviews:
            tmp_ovr_path = tmp_path.with_suffix(".tif.ovr")
            cog_ovr_path = cog_path.with_suffix(".tif.ovr")
            if tmp_ovr_path.exists():
                shutil.move(tmp_ovr_path, cog_ovr_path)
    finally:
        tmp_path.unlink(missing_ok=True)
