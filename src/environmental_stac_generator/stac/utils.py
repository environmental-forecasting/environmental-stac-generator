import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any

import rasterio
import xarray as xr
import zarr
from pystac import Asset, Catalog, Collection, MediaType, Summaries
from multiformats import multihash
from pystac.extensions.file import FileExtension
from pystac.utils import datetime_to_str


class ConfigMismatchError(Exception):
    pass


def refresh_collection_summaries(collection: Collection) -> None:
    """
    Rebuild Collection summaries from its Items.

    Writes ``forecast:reference_time`` (sorted unique init times) and, when
    present on COG assets, ``forecast:variable`` (sorted unique band names).
    Summaries use a high enough ``maxcount`` so large forecast archives are
    not truncated by pystac's default of 25.
    """
    reference_times: set[str] = set()
    variables: set[str] = set()

    for item in collection.get_items():
        ref = item.properties.get("forecast:reference_time")
        if not ref and item.datetime is not None:
            ref = datetime_to_str(item.datetime)
        if ref:
            reference_times.add(ref)

        for asset in item.get_assets(media_type=MediaType.COG, role="data").values():
            for band in asset.extra_fields.get("forecast:bands") or []:
                name = band.get("name")
                if name:
                    variables.add(str(name))

    summary_dict: dict[str, Any] = {}
    if reference_times:
        summary_dict["forecast:reference_time"] = sorted(reference_times)
    if variables:
        summary_dict["forecast:variable"] = sorted(variables)

    maxcount = max(len(reference_times), len(variables), 25)
    collection.summaries = Summaries(summary_dict, maxcount=maxcount)


def to_cwd_relative_href(href: str, cwd: Path | None = None) -> str:
    """
    Convert a local filesystem href to a path relative to ``cwd``.

    Absolute paths under ``cwd`` become portable values such as ``data/cogs/...``.
    HTTP(S) hrefs and paths outside ``cwd`` are returned unchanged.
    """
    if not href or href.startswith(("http://", "https://")):
        return href
    root = (cwd or Path.cwd()).resolve()
    path = Path(href)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return href


def apply_file_server_url(href: str, file_server_url: str, cwd: Path | None = None) -> str:
    """
    Prefix a local/portable asset href with ``file_server_url``.

    Already-absolute HTTP(S) hrefs are left unchanged so catalogs generated for
    another environment are not silently rewritten.
    """
    if not href or href.startswith(("http://", "https://")):
        return href
    if not file_server_url:
        return href

    base = file_server_url if file_server_url.endswith("/") else file_server_url + "/"
    root = (cwd or Path.cwd()).resolve()
    path = Path(href)
    if path.is_absolute():
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            return href
    else:
        # Portable cwd-relative paths written by preprocess (e.g. data/cogs/...).
        rel = path

    return base + str(rel).lstrip("./")


def rewrite_catalog_asset_hrefs(
    catalog: Catalog,
    file_server_url: str,
    cwd: Path | None = None,
) -> None:
    """Apply ``file_server_url`` to all collection and item asset hrefs in-place."""
    for collection in catalog.get_all_collections():
        for asset in collection.assets.values():
            asset.href = apply_file_server_url(asset.href, file_server_url, cwd=cwd)
    for item in catalog.get_items(recursive=True):
        for asset in item.assets.values():
            asset.href = apply_file_server_url(asset.href, file_server_url, cwd=cwd)


def file_multihash(file_path: str) -> str:
    """Computes a multihash-encoded MD5 digest of the entire file.

    Reads the entire contents of a file into memory (should only be
    used for small files) and returns the hexadecimal representation
    of the multihash digest.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hexadecimal string of the multihash-encoded MD5 digest.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    # Compute multihash-encoded digest
    digest = multihash.digest(data, "md5")
    return digest.hex()


def file_block_multihash(file_path: str, block_size=8192) -> str:
    """Computes a multihash-encoded MD5 digest over a file in blocks.

    Reads the file in chunks and computes an MD5 digest incrementally,
    allowing processing of large files without loading the entire file into memory.

    Args:
        file_path: Path to the file to hash.
        block_size (optional): Size of each block to read from the file in bytes.
            Defaults to 8192.

    Returns:
        Hexadecimal string of the final multihash-encoded MD5 digest.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hash_md5.update(chunk)
    digest = multihash.digest(hash_md5.digest(), "md5")
    return digest.hex()


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

    file_ext.checksum = file_block_multihash(file_path)

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
