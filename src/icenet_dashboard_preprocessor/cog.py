import subprocess
from pathlib import Path

import xarray as xr


def write_cog(cog_path: Path, da: xr.DataArray, compress: str = "DEFLATE") -> None:
    tmp_cog_path = cog_path.with_name(cog_path.stem + ".tmp.tif")

    # Convert xr.DataArray to GeoTIFF using rioxarray
    da.rio.to_raster(
        tmp_cog_path,
        driver="GTIFF",
        compress=compress,
    )

    # Add internal overviews using "gdaladdo"
    subprocess.run([
        "gdaladdo",
        "-q",
        "-r", "average",
        tmp_cog_path,
        "2", "4", "8", "16",
    ], check=True)

    # Convert to Cloud Optimized GeoTIFF using "gdal_translate"
    subprocess.run([
        "gdal_translate",
        tmp_cog_path,
        cog_path,
        "-q",
        "-of", "COG",
        "-co", f"COMPRESS={compress}",
        "-co", "BLOCKSIZE=256",
        "-co", "RESAMPLING=AVERAGE"
    ], check=True)

    if tmp_cog_path.exists():
        tmp_cog_path.unlink()