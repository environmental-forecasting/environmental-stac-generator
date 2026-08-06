import logging
import subprocess
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from .utils import get_array_statistics

logger = logging.getLogger(__name__)

# Suppress rio-cogeo's legacy advisory warning regarding ZSTD in TIFF specifications.
# ZSTD is standard in libtiff 4.0.10+ and GDAL 2.3+ (used by TiTiler, QGIS, etc.).
warnings.filterwarnings(
    "ignore",
    message=r".*Non-standard compression schema: zstd.*",
    category=UserWarning,
    module=r"rio_cogeo.*",
)

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
    compress: str = "ZSTD",
    block_size: int = 256,
    overview_level: int = 5,
    overview_resampling: str = "bilinear",
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
            Defaults to "ZSTD".
        block_size: Block size (in pixels) used for tiling.
            Defaults to 256.
        overview_level: Number of overviews to generate. This defines how many
            downsampled versions of the raster will be created (powers of 2:
            2, 4, 8, 16, 32). Defaults to 5.
        overview_resampling: Resampling algorithm for internal overviews.
            Defaults to "bilinear".
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
    compress_norm = (compress or "NONE").upper()
    if compress_norm in ("NONE", "RAW"):
        dst_profile = cog_profiles.get("raw")
        dst_profile.update(
            {
                "blockxsize": block_size,
                "blockysize": block_size,
            }
        )
        dst_profile.pop("predictor", None)
        dst_profile.pop("zstd_level", None)
    else:
        profile_name = (
            compress_norm.lower()
            if compress_norm.lower() in ("deflate", "zstd", "lzw", "webp", "packbits")
            else "deflate"
        )
        dst_profile = cog_profiles.get(profile_name)
        dst_profile.update(
            {
                "compress": compress_norm,
                "blockxsize": block_size,
                "blockysize": block_size,
            }
        )
        if compress_norm == "ZSTD":
            dst_profile["zstd_level"] = 7
            dst_profile["predictor"] = 2
        elif compress_norm in ("DEFLATE", "LZW"):
            dst_profile["predictor"] = 2
        else:
            dst_profile.pop("predictor", None)

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
        allow_intermediate_compression = compress_norm not in ("NONE", "RAW")
        with memfile.open() as mem:
            cog_translate(
                source=mem,
                dst_path=cog_path,
                dst_kwargs=dst_profile,
                overview_level=overview_level,
                overview_resampling=overview_resampling,
                forward_band_tags=True,
                in_memory=True,
                allow_intermediate_compression=allow_intermediate_compression,
                temporary_compression=compress_norm if allow_intermediate_compression else "DEFLATE",
                quiet=True,
            )

    if external_overviews:
        overviews = [str(2**i) for i in range(1, overview_level + 1)]
        subprocess.run(
            [
                "gdaladdo",
                "-q",
                "-r",
                overview_resampling,
                "-ro",
                str(cog_path),
                *overviews,
            ],
            check=True,
        )
