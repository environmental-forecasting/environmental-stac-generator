import logging
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from .utils import get_array_statistics

logger = logging.getLogger(__name__)

# Band tags written into the COG (and mirrored into STAC forecast:bands).
_STAT_TAG_KEYS = (
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
    "STATISTICS_VALID_PERCENT",
)


def _band_stat_tags(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {}
    return {
        key: stats[key]
        for key in _STAT_TAG_KEYS
        if key in stats and stats[key] is not None
    }


def _dataarray_to_raster_array(da: xr.DataArray) -> tuple[np.ndarray, int, int, int]:
    """Return (band, y, x) array plus count/height/width."""
    y_dim = da.rio.y_dim
    x_dim = da.rio.x_dim
    height = int(da.sizes[y_dim])
    width = int(da.sizes[x_dim])

    if "band" in da.dims:
        data = np.asarray(da.transpose("band", y_dim, x_dim).values)
    else:
        data = np.asarray(da.transpose(y_dim, x_dim).values)[np.newaxis, ...]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D band/y/x array, got shape {data.shape}")
    return data, int(data.shape[0]), height, width


def write_cog(
    cog_path: Path,
    da: xr.DataArray,
    compress: str = "DEFLATE",
    block_size: int = 256,
    overview_level: int = 4,
    external_overviews: bool = False,
    band_statistics: Sequence[dict[str, Any]] | None = None,
) -> None:
    """
    Write a Cloud Optimized GeoTIFF (COG) from an xarray DataArray.

    Builds an in-memory GeoTIFF, embeds band statistics tags, then converts to a
    COG with internal overviews via ``rio-cogeo`` (single disk write of the final
    COG). External ``.ovr`` sidecars are off by default; pass
    ``external_overviews=True`` if a workflow still needs them.

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
        band_statistics: Optional per-band statistics dicts (index-aligned with
            bands). When provided, avoids a second full-array stats pass.
            Expected keys match GDAL ``STATISTICS_*`` tags.

    Notes:
        - Band-level statistics (minimum, maximum, mean, standard deviation,
          valid percent) are embedded as band tags when available.

        - ``external_overviews=True`` requires GDAL (``gdaladdo``) on PATH.
    """
    dst_profile = cog_profiles.get("deflate")
    dst_profile.update(
        {
            "compress": compress,
            "blockxsize": block_size,
            "blockysize": block_size,
        }
    )

    data, count, height, width = _dataarray_to_raster_array(da)
    src_profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "count": count,
        "height": height,
        "width": width,
        "crs": da.rio.crs,
        "transform": da.rio.transform(),
        "nodata": da.rio.nodata,
    }

    # Prefer caller-supplied stats (already computed for STAC); otherwise derive
    # once from the in-memory array so COG tags stay populated.
    if band_statistics is None:
        band_statistics = [get_array_statistics(data[i]) for i in range(count)]

    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as mem:
            mem.write(data)
            for i in range(1, count + 1):
                tags = _band_stat_tags(
                    band_statistics[i - 1] if i - 1 < len(band_statistics) else None
                )
                if tags:
                    mem.update_tags(i, **tags)

        # Re-open read-only for cog_translate (write-mode datasets are deprecated).
        with memfile.open() as mem:
            cog_translate(
                source=mem,
                dst_path=cog_path,
                dst_kwargs=dst_profile,
                overview_level=overview_level,
                overview_resampling="average",
                forward_band_tags=True,
                in_memory=True,
                quiet=True,
            )

    if external_overviews:
        subprocess.run(
            [
                "gdaladdo",
                "-q",
                "-ro",
                str(cog_path),
                "2",
                "4",
                "8",
                "16",
            ],
            check=True,
        )
