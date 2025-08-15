import mimetypes
import os

import rasterio
import xarray as xr
import zarr
from pystac import Asset
from pystac.extensions.file import FileExtension


def add_file_info_to_asset(asset: Asset, file_path: str) -> Asset:
    """
    Adds STAC File Info Extension metadata to an asset based on file type.

    Handles raster images (GeoTIFF, COG, JPG, PNG), netCDF, and Zarr stores.

    Args:
        asset: The STAC asset to update. Must have a parent Item.
        file_path: Path to the local file (or directory for Zarr).

    Returns:
        The updated asset with file extension metadata.
    """

    # Attach file extension if missing
    file_ext = FileExtension.ext(asset, add_if_missing=True)

    # Set file size
    if os.path.isdir(file_path):
        total_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(file_path)
            for f in files
        )
        file_ext.size = total_size
    else:
        file_ext.size = os.path.getsize(file_path)

    # Add media type if missing
    if asset.media_type is None:
        mime, _ = mimetypes.guess_type(file_path)
        if mime:
            asset.media_type = mime

    # Try to get type-specific metadata
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        with rasterio.open(file_path) as src:
            if src.count == 0:
                raise ValueError(f"No bands found in raster: {file_path}")
            dtype = src.dtypes[0]
            file_ext.data_type = dtype
            file_ext.byte_order = src.profile.get("endian", "little") + "-endian"
    elif ext in [".jpg", ".jpeg", ".png"]:
        # Image formats – assume 8-bit unsigned int
        file_ext.data_type = "uint8"
        file_ext.bit_depth = 8
        file_ext.byte_order = "little-endian"
    elif ext in [".nc", ".nc4"]:
        with xr.open_dataset(file_path) as ds:
            # Use first variable with dimensions
            for var in ds.data_vars:
                dtype = str(ds[var].dtype)
                file_ext.data_type = dtype
                file_ext.bit_depth = ds[var].dtype.itemsize * 8
                file_ext.byte_order = "little-endian"  # netCDF defaults
                break
    elif ext == ".zarr" or file_path.endswith(".zarr"):
        z = zarr.open(file_path, mode="r")
        array = None
        if hasattr(z, "values"):
            array = z
        else:
            for key in z.group_keys():
                a = z[key]
                if hasattr(a, "dtype"):
                    array = a
                    break
        if array is not None:
            file_ext.data_type = str(array.dtype)
            file_ext.bit_depth = array.dtype.itemsize * 8
            file_ext.byte_order = "little-endian"
    else:
        # Unknown or unsupported data type
        pass

    return asset
