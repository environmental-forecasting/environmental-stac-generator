
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pystac
import rioxarray
import xarray as xr
from dateutil.relativedelta import relativedelta
from pystac import Asset, Catalog, Collection, Item
from pystac.extensions.projection import ProjectionExtension
from shapely.geometry import box, mapping
from tqdm import tqdm

from .cog import write_cog
from .stac import IceNetSTAC
from .utils import find_coord, flatten_list, get_hemisphere, get_nc_files, parse_forecast_frequency

logger = logging.getLogger(__name__)


def get_or_create_catalog(stac_catalog_path: Path, catalog_defs: dict) -> Catalog:
    if stac_catalog_path.exists():
        return Catalog.from_file(stac_catalog_path)
    return Catalog(
        id=catalog_defs["id"],
        description=catalog_defs["description"],
        title=catalog_defs["title"],
    )


def get_or_create_collection(parent, collection_id, title, description, bbox, temporal_extent) -> Collection:
    collection = next((c for c in parent.get_children() if c.id == collection_id), None)
    if not collection:
        collection = Collection(
            id=collection_id,
            title=title,
            description=description,
            extent=pystac.Extent(
                pystac.SpatialExtent([bbox]),
                pystac.TemporalExtent([temporal_extent]),
            ),
        )
        parent.add_child(collection)
    return collection


def generate_cloud_tiff(
    nc_file: Path,
    name: str,
    compress: bool = True,
    overwrite=False,
    forecast_frequency="1days",
) -> None:
    """Generates Cloud Optimized GeoTIFFs and STAC Catalogs from IceNet prediction netCDF files.

    Args:
        nc_file: The path to the prediction netCDF file.
        name: High-level collection name.
        compress (optional): Whether to compress the output GeoTIFFs.
                    Default is True.
        overwrite (optional): Whether to overwrite existing outputs.
                    Default is False.
        forecast_frequency (optional): The forecast frequency of the data.
                    Default is "1days".
    """
    compress_method = "DEFLATE" if compress else "NONE"
    ncdf_output_dir = Path("data") / "netcdf" / name
    cogs_output_dir = Path("data") / "cogs" / name
    stac_output_dir = Path("data") / "stac" # This has dir with `name` created by itself
    stac_output_dir.mkdir(parents=True, exist_ok=True)
    nc_file = Path(nc_file).resolve()
    hemisphere = get_hemisphere(nc_file)

    # Get input time delta options to compute forecast times
    leadtime_step, leadtime_unit = parse_forecast_frequency(forecast_frequency)

    with xr.open_dataset(nc_file, decode_coords="all") as ds:
        # Determine spatial coordinates
        x_coord = find_coord(ds, ["xc", "x", "lon", "longitude"])
        y_coord = find_coord(ds, ["yc", "y", "lat", "latitude"])

        # Get time-related coordinate information
        time_coords: xr.DataArray = ds.coords.get("time", ds.coords.get("forecast_time"))
        leadtime_coords: xr.DataArray = ds.coords.get("leadtime", ds.coords.get("lead_time"))
        leadtime = len(leadtime_coords)

        if x_coord is None or y_coord is None:
            raise ValueError("Spatial coordinates not found in dataset")

        # Convert eastings and northings from kilometers to metres (if need to).
        if ds.coords[y_coord].attrs.get("units", None) in ["1000 meter", "km"]: # `1000 meter` is legacy support for `icenet < v0.4.0``
            ds = ds.assign_coords({y_coord: ds.coords[y_coord] * 1000})
        if ds.coords[x_coord].attrs.get("units", None) in ["1000 meter", "km"]:
            ds = ds.assign_coords({x_coord: ds.coords[x_coord] * 1000})

        # Compute bounding box and geometry from coordinates
        x_min, x_max = float(ds[x_coord].min()), float(ds[x_coord].max())
        y_min, y_max = float(ds[y_coord].min()), float(ds[y_coord].max())
        bbox = [x_min, y_min, x_max, y_max]
        geometry = mapping(box(*bbox)) # type: ignore

        # Filter 4D variables - these are variables of interest for COGs
        # Assuming other vars shouldn't be converted to COGs
        valid_bands = [
            var for var in ds.data_vars
            if len(ds[var].dims) == 4
        ]

        # Initialise STAC Catalog
        stac_catalog_path = stac_output_dir / "catalog.json"
        catalog_defs = {
                "id": "forecast-data",
                "description": "Catalog of IceNet Forecast Data",
                "title": "IceNet Forecast STAC Catalog",
        }
        catalog = get_or_create_catalog(stac_catalog_path, catalog_defs)

        # Get attributes from NetCDF file
        nc_attrs = ds.attrs
        crs = nc_attrs["geospatial_bounds_crs"]
        ds.rio.write_crs(crs, inplace=True)

        # Get temporal extent range from input netCDF
        time_coords_start = pd.to_datetime(time_coords.isel(time=0).values)
        time_coords_end = pd.to_datetime(time_coords.isel(time=-1).values)

        # Create (or retrieve) highest level collection (model name) within the catalog
        main_collection = get_or_create_collection(
            parent=catalog,
            collection_id=name,
            title=f"Model Collection: {name}",
            description=f"{name} collection",
            bbox=bbox,
            temporal_extent=[time_coords_start, time_coords_end],
        )

        # Create (or retrieve) a hemisphere collection within the catalog
        hemisphere_collection = get_or_create_collection(
            parent=main_collection,
            collection_id=hemisphere,
            title=f"Hemisphere Collection: {hemisphere.capitalize()}",
            description=f"{hemisphere.capitalize()} hemisphere collection",
            bbox=bbox,
            temporal_extent=[time_coords_start, time_coords_end],
        )

        for time_idx, time_val in enumerate(time_coords):
            # Rearrange array dimension to match rioxarray expectation
            ds_time_slice = ds.sel(time=time_val)

            # The forecast initialisation time (CF Convention: `forecast_reference_time`) is the first forecast
            forecast_reference_time = pd.to_datetime(time_val.values)
            forecast_reference_date = forecast_reference_time.date()
            forecast_reference_time_str = forecast_reference_time.strftime("%Y%m%d_%H%M")
            forecast_reference_time_str_fmt = forecast_reference_time.strftime("%Y-%m-%d %H:%M")
            forecast_end_time = forecast_reference_time + relativedelta(**{leadtime_unit: leadtime - 1})

            # Create (or retrieve) a forecast collection within the catalog
            forecast_collection = get_or_create_collection(
                parent=hemisphere_collection,
                collection_id=f"{forecast_reference_date}",
                title=f"Forecast Collection: {forecast_reference_date}",
                description=f"Forecast data for {forecast_reference_date}",
                bbox=bbox,
                temporal_extent=[forecast_reference_time, forecast_end_time],
            )

            # Create output dirs
            ncdf_dir = Path(ncdf_output_dir / f"{hemisphere}/{forecast_reference_date}")
            ncdf_dir.mkdir(parents=True, exist_ok=True)
            cog_dir = Path(cogs_output_dir / f"{hemisphere}/{forecast_reference_date}")
            cog_dir.mkdir(parents=True, exist_ok=True)

            # Save the forecast init slice as a netcdf file
            item_id_nc = f"forecast_init_{forecast_reference_time_str}"
            nc_path = ncdf_dir / f"{item_id_nc}.nc"
            encoding = {
                var: {
                    "zlib": True,
                    "complevel": 5,
                } for var in ds_time_slice.data_vars
            }
            ds_time_slice.to_netcdf(
                nc_path,
                engine="h5netcdf",
                encoding=encoding,
            )

            # Add STAC Item for this netCDF file
            item = Item(
                id=item_id_nc,
                geometry=geometry,
                bbox=bbox,
                datetime=forecast_reference_time,
                properties={
                    "forecast_reference_time": str(forecast_reference_time),
                    "hemisphere": hemisphere,
                },
            )

            # Add netCDF asset to item
            item.add_asset(
                "netcdf",
                Asset(
                    href=str(nc_path),
                    media_type=pystac.MediaType.COG,
                    title=f"netCDF for {forecast_reference_time_str_fmt}",
                    description=f"netCDF file container forecast variables for forecast initialised at: {forecast_reference_time_str_fmt}",
                    roles=["data"],
                ),
            )

            # Process each leadtime
            for i in (pbar := tqdm(range(leadtime), desc="COGifying files", leave=True)):
                valid_time = forecast_reference_time + relativedelta(**{leadtime_unit: i*leadtime_step})
                ds_leadtime_slice = ds_time_slice.isel(leadtime=i)

                # Set spatial dimensions for rioxarray
                ds_leadtime_slice.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)

                valid_time_str = valid_time.strftime("%Y%m%d_%H%M")
                valid_time_str_fmt = valid_time.strftime("%Y-%m-%d %H:%M")

                # Add STAC Item for this file
                item_id_cog = f"forecast_init_{forecast_reference_time_str}_lead_{valid_time_str}"
                item = Item(
                    id=item_id_cog,
                    geometry=geometry,
                    bbox=bbox,
                    datetime=valid_time,
                    properties={
                        "forecast_reference_time": str(forecast_reference_time),
                        "hemisphere": hemisphere,
                        "leadtime": i,
                    },
                )

                # Add projection extension
                ProjectionExtension.add_to(item)
                proj = ProjectionExtension.ext(item)
                proj.code = crs

                # Add item to collection
                forecast_collection.add_item(item)

                # Save each variable as separate COG (Cloud Optimized GeoTIFF) & JPG (for thumbnail)
                for var_name in valid_bands:
                    da_variable = ds_leadtime_slice[var_name]
                    cog_path = cog_dir / f"{item_id_cog}_{var_name}.tif"
                    thumbnail_path = cog_dir / f"{item_id_cog}_{var_name}.jpg"
                    if cog_path.exists() and not overwrite:
                        pbar.set_description(f"File already exists, skipping: {cog_path}")
                        continue
                    else:
                        pbar.set_description(f"Saving to COG: {cog_path}")

                    # Add metadata to extracted variable so `to_raster` includes them in the output GeoTIFF
                    da_variable.rio.write_crs(crs, inplace=True)
                    da_variable.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)
                    # da_variable.rio.to_raster(cog_path, driver="COG", compress=compress_method)
                    write_cog(cog_path, da_variable, compress=compress_method)

                    # Add COG asset to item
                    item.add_asset(
                        f"{var_name}",
                        Asset(
                            href=str(cog_path),
                            media_type=pystac.MediaType.COG,
                            title=f"{var_name} at {valid_time_str_fmt}",
                            description=ds[var_name].long_name or None,
                            roles=["data"],
                            extra_fields={
                                "variable": var_name
                            },
                        ),
                    )

                    # Create a thumbnail plot of the variable
                    fig = plt.figure(figsize=(5, 5), dpi=100, constrained_layout=True)
                    da_variable.plot(cmap='RdBu_r', add_colorbar=True) # type: ignore
                    plt.axis('off')
                    plt.title(f"Init: {forecast_reference_time}\nLeadtime: {valid_time_str_fmt}")
                    plt.savefig(thumbnail_path, pad_inches=0, transparent=False)
                    plt.close(fig)

                    # Add thumbnail asset to item
                    # Some STAC tools may only show the first thumbnail asset
                    item.add_asset(
                        f"thumbnail_{var_name}",
                        Asset(
                            href=str(thumbnail_path),
                            media_type=pystac.MediaType.JPEG,
                            title=f"{var_name.capitalize()} Thumbnail",
                            roles=["thumbnail"],
                        ),
                    )

        # Save catalog and collections
        catalog.normalize_hrefs(
            str(stac_output_dir),
        )
        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    return


def main(args: SimpleNamespace):
    """
    Main function to generate COGs and generate static JSON STAC catalog.

    This function processes netCDF files and generates cloud-optimized geotiffs (COGs)
    using the given CLI arguments.

    Args:
        args: Parsed Command line arguments. The expected keys are:
            input (List[str]): List of input netCDF files or directories (positional argument).
            --compress (bool): Whether to compress the output COG files.
            --overwrite (bool): Whether to overwrite existing COG files.

    Raises:
        FileNotFoundError: If no valid netCDF files are found for processing.

    Returns:
        None

    Examples:
        For daily forecasting
        >>> dashboard preprocess 1days raw_data/*.nc
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

    for nc_file in (pbar := tqdm(nc_files, desc="COGifying files", leave=True)): # type: ignore
        pbar.set_description(f"Processing {nc_file}")
        generate_cloud_tiff(
            nc_file,
            name=args.name,
            compress=args.no_compress,
            overwrite=args.overwrite,
            forecast_frequency=args.forecast_frequency,
        )
