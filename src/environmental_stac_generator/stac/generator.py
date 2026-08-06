import logging
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import orjson
import pandas as pd
import pystac
import rasterio
import xarray as xr
from deepdiff import DeepDiff
from pystac import Asset, Catalog, Collection, Item
from pystac.extensions.projection import ProjectionExtension
from pystac.utils import datetime_to_str, str_to_datetime
from shapely.geometry import box, mapping
from tqdm import tqdm

from ..cog import write_cog
from ..utils import (
    DEFAULT_WORKERS,
    ensure_utc,
    find_coord,
    forecast_frequency_from_valid_times,
    format_time,
    get_da_statistics,
    get_hemisphere,
    get_nc_attributes,
    resolve_valid_times,
    proj_to_geo,
)
from .utils import (
    ConfigMismatchError,
    add_file_info_to_asset,
    refresh_collection_summaries,
    to_cwd_relative_href,
)
logger = logging.getLogger(__name__)

# Datetime format constants used for STAC asset generation
DT_FMT_FILENAME = "%Y-%m-%d_%H%M"       # Filesystem-safe format for COG filenames and item IDs
DT_FMT_DISPLAY = "%Y-%m-%d %H:%M"       # Human-readable format for asset titles
DT_FMT_ISO8601 = "%Y-%m-%dT%H:%M:%SZ"   # ISO 8601 format for STAC properties



class BaseSTAC:
    def __init__(
        self,
        data_path: Path = Path("data"),
        catalog_defs: dict | None = None,
        license: str | None = None,
    ):
        """
        Initialises the BaseSTAC class with a path to an existing STAC catalog.

        Args:
            data_path (optional): The path to the directory containing input files used
                to generate the STAC catalog.
                Defaults to `data/` if not provided.
            catalog_defs (optional): Dictionary of metadata for the root STAC catalog.
                If not provided, defaults to a standard BAS environmental forecast catalog
                definition.
            license (optional): SPDX license identifier for the items in the STAC catalog.
                Defaults to `"OGL-UK-3.0"` if not provided.
        """
        if not catalog_defs:
            catalog_defs = {
                "id": "bas-environmental-forecasts",
                "description": "Catalog of BAS Environmental Forecast Data",
                "title": "BAS Environmental Forecasting STAC Catalog",
            }

        self._data_path = data_path
        self._license = (
            license if license else "OGL-UK-3.0"
        )  # Ref: https://spdx.org/licenses/

        self._set_catalog_path()
        self.get_or_create_catalog(catalog_defs=catalog_defs)

    def _set_catalog_path(self) -> None:
        """
        Configure the STAC output directory and catalog file path.

        Creates the `stac` subdirectory within `self.data_path` if it doesn't exist,
        and uses `catalog.json` in that directory as the shared root catalog file.
        Collections from each `-n` name are added as children of this catalog.
        """
        self._stac_output_dir = self.data_path / "stac"
        self._stac_output_dir.mkdir(parents=True, exist_ok=True)
        self._stac_catalog_file = self._stac_output_dir / "catalog.json"

    def get_or_create_catalog(
        self, catalog_defs: dict, describe: bool = True
    ) -> Catalog:
        """Initialises a STAC catalog or loads an existing one if it exists.

        This method either creates a new STAC catalog with the provided metadata
        or loads an existing one from the specified path. If `describe` is set to
        True, it will print a description of the catalog after initialization.

        Args:
            id: The ID for the catalog.
            description: A detailed description of the catalog.
            title: The title of the catalog.
            describe (optional): Whether to print a description of the catalog.
                Defaults to True.
        Returns:
            Catalog: The initialised or loaded STAC catalog.
        """
        stac_catalog_file = self._stac_catalog_file
        if stac_catalog_file.exists():
            catalog = Catalog.from_file(stac_catalog_file)
        else:
            catalog = Catalog(
                id=catalog_defs["id"],
                description=catalog_defs["description"],
                title=catalog_defs["title"],
                href=str(stac_catalog_file),
            )
        self._stac_catalog = catalog
        if describe:
            print("Description:", catalog.describe())
        return catalog

    def get_or_create_collection(
        self,
        parent: Catalog | Collection,
        collection_id: str,
        title: str,
        description: str,
        bbox: list,
        temporal_extent: list,
        extra_fields: dict[str, Any] | None = None,
        license: str = "other",
    ) -> Collection:
        """
        Retrieve or create a STAC Collection within the given parent catalog.

        If a collection with the specified ID already exists as a child of the
        parent, it is returned. Otherwise, a new Collection is created with the
        provided metadata and added to the parent.

        Args:
            parent: The parent Catalog or Collection to search/add to.
            collection_id: Unique identifier for the STAC Collection.
            title: Title of the collection.
            description: Description of the collection.
            bbox: Bounding box [west, south, east, north] in WGS84 coordinates.
            temporal_extent: Temporal extent as [start_datetime, end_datetime].
            license: License type.
                Defaults to "other".

        Returns:
            Collection: The existing or newly created STAC Collection.
        """
        collection = next(
            (c for c in parent.get_children() if c.id == collection_id), None
        )
        if not collection:
            collection = Collection(
                id=collection_id,
                title=title,
                description=description,
                extra_fields=extra_fields,
                license=license,
                extent=pystac.Extent(
                    pystac.SpatialExtent([bbox]),
                    pystac.TemporalExtent([temporal_extent]),
                ),
            )
            parent.add_child(collection)
        else:
            # Update the existing collection's temporal extent
            existing_intervals = collection.extent.temporal.intervals[0]
            existing_start, existing_end = existing_intervals

            existing_start = ensure_utc(existing_start)
            existing_end = ensure_utc(existing_end)
            new_start = ensure_utc(temporal_extent[0])
            new_end = ensure_utc(temporal_extent[1])

            # Compute min start and max end (handle None values)
            updated_start = min(filter(None, [existing_start, new_start]))
            updated_end = max(filter(None, [existing_end, new_end]))

            # Update only if the extent has changed
            if (existing_start != updated_start) or (existing_end != updated_end):
                collection.extent.temporal.intervals = [[updated_start, updated_end]]

        return collection  # type: ignore

    def get_or_create_item(
        self,
        collection: Collection,
        item_id: str,
        geometry: dict,
        bbox: list,
        crs: str,
        properties: dict,
        datetime: datetime,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
    ) -> Item:
        """
        Retrieve or create a STAC Item within the given parent collection.

        If an item with the specified ID already exists as a child of the
        collection, it is returned. Otherwise, a new Item is created with the
        provided geometry, temporal/spatial extent, and properties. Also adds
        projection extension for coordinate reference system (CRS).

        Args:
            collection: Parent STAC Collection to add the Item to.
            item_id: Unique identifier for the STAC Item.
            geometry: GeoJSON-like dictionary representing item geometry.
            bbox: Bounding box [west, south, east, north] in WGS84 coordinates.
            crs: Coordinate Reference System code (e.g., "EPSG:4326").
            properties: Additional metadata properties for the item.
            datetime: Datetime object representing item temporal extent.

        Returns:
            Item: The created STAC Item with associated Asset and extensions.
        """
        item: Item | None = collection.get_item(item_id)
        if not item:
            item = Item(
                id=item_id,
                geometry=geometry,
                bbox=bbox,
                properties=properties,
                datetime=datetime,
                # # Setting following will mean that when filtering time in STAC Browser,
                # # it would show any forecast inits with leadtimes that overlap with
                # # the selected time range, so, not setting a time range.
                # start_datetime=start_datetime,
                # end_datetime=end_datetime,
            )
            # Add projection extension
            ProjectionExtension.add_to(item)
            proj = ProjectionExtension.ext(item)
            proj.code = crs # type: ignore
            collection.add_item(item)
        return item # type: ignore

    def create_multiband_raster(
        self,
        ds: xr.Dataset,
        crs: str,
        x_coord: str,
        y_coord: str,
        valid_bands: list[str],
    ) -> tuple[xr.DataArray, list[str]]:
        """
        Process and concatenate valid bands into a multiband raster.

        Processes each variable in valid_bands by setting the CRS and spatial
        dimensions, then concatenates them along the 'band' dimension to create
        a multiband xarray DataArray. Returns both the multiband array and list
        of band names.

        Args:
            ds: Input Dataset containing raster data.
            crs: Coordinate Reference System (e.g., "EPSG:4326").
            x_coord: Name of x-dimension in dataset.
            y_coord: Name of y-dimension in dataset.
            valid_bands: List of variable names to include as bands.

        Returns:
            A 3D xarray DataArray with band dimension and corresponding
            list of band names.

        Notes:
            I set the CRS and spatial dims here for supporting rioxarray's
            requirements if I use it to write out COGs, or any other
            processing on the xr.DataArray itself rather than the xr.DataSet.
        """
        da_list = []
        band_names = []

        for var_name in valid_bands:
            da = ds[var_name]
            da.rio.write_crs(crs, inplace=True)
            da.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)
            da_list.append(da)
            band_names.append(var_name)

        multiband = xr.concat(da_list, dim="band")
        multiband = multiband.assign_coords(band=("band", band_names))
        return multiband, band_names

    @property
    def catalog(self) -> Catalog:
        """Get the current STAC catalog.

        Provides access to the internal STAC catalog used for data processing.
        """
        if self._stac_catalog is None:
            raise ValueError("STAC catalog not initialised.")
        return self._stac_catalog

    def get_collection(self, collection_id):
        """Get a child collection by its ID.

        Searches the catalog's children for a collection matching the given ID.

        Args:
            collection_id: The ID of the collection to retrieve.

        Returns:
            The matching Collection, or None if not found.
        """
        collection = next(
            (
                coll
                for coll in self._stac_catalog.get_children()
                if coll.id == collection_id
            ),
            None,
        )
        return collection

    @property
    def data_path(self):
        """
        Get or set the base directory path for data processing.

        This property provides access to the internal `_data_path` value used
        for storing and retrieving STAC catalog files. The setter allows modifying
        this path during object initialisation or runtime.
        """
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        """
        Set the base directory path for data processing.

        Args:
            value: New path to use as the base directory.
        """
        self._data_path = value

    @abstractmethod
    def process(self, nc_file: Path, **kwargs):
        """
        Process a netCDF file into STAC items and collections.

        This method must be implemented by derived classes to define how netCDF
        files are converted into STAC Items and Collections. It should handle
        all necessary processing steps for ingesting data into the STAC catalog.

        Args:
            nc_file: Path to the netCDF file to process.
            **kwargs: Additional keyword arguments for custom processing logic.
        """
        pass


class STACGenerator(BaseSTAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor: ProcessPoolExecutor | None = None
        self._executor_workers: int | None = None

    def _get_executor(self, workers: int) -> ProcessPoolExecutor:
        """Return a reusable process pool, recreating it if worker count changes."""
        if self._executor is None or self._executor_workers != workers:
            self.close_executor()
            self._executor = ProcessPoolExecutor(max_workers=workers)
            self._executor_workers = workers
        return self._executor

    def close_executor(self) -> None:
        """Shut down the reusable process pool if it is running."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._executor_workers = None

    def _set_out_paths(self) -> None:
        """
        Set output directories for netCDF files, COGs, and configuration data.

        Define subdirectories under the base data path for storing processed
        netCDF files, Cloud-Optimized GeoTIFFs (COGs), and a JSON config file
        that records preprocessing parameters.
        """
        data_path = self.data_path
        collection_name = self._collection_name
        self._netcdf_output_dir = data_path / "netcdf" / collection_name
        self._cogs_output_dir = data_path / "cogs" / collection_name
        # Store a config file of how the preprocessor was run
        self._config_output_path = data_path / "config.json"

    def _validate_input_options(self):
        """
        Validate and store input processing options in configuration file.

        Checks that the current configuration matches any previously saved
        configuration for the same collection. If mismatched, logs an error
        and exits to prevent inconsistent STAC generation.
        """
        # Store input options to file
        config_data = {
            self._collection_name: {
                "forecast_frequency": self._forecast_frequency,
            }
        }
        self._store_config(config_data)

    def _store_config(self, config_data: dict):
        """
        Write or validate configuration data for STAC generation.

        If a configuration file already exists for this collection, validates
        that the new configuration matches the existing one. If the file exists
        but this collection is new, merges the new collection into the file.
        Otherwise, creates the file and stores the provided configuration.

        Args:
            config_data: Dictionary containing processing parameters to store.
        """
        collection_name = self._collection_name
        config_output_path = self._config_output_path

        # Ensure we're running with same options as any previous runs
        if config_output_path.exists():
            with open(config_output_path, "rb") as f:
                current_config_data = orjson.loads(f.read())
            if collection_name in current_config_data:
                diff = DeepDiff(
                    config_data[collection_name],
                    current_config_data[collection_name],
                )
                if diff:
                    logger.error(
                        "You are attempting to generate collection "
                        f"({collection_name}) with different options to "
                        "previous! Run with old values (below) to continue!"
                    )
                    logger.error(current_config_data[collection_name])
                    raise ConfigMismatchError("Config does not match previous run.")
                return

            # New collection in an existing config file: merge and write
            current_config_data.update(config_data)
            with open(config_output_path, "wb") as f:
                f.write(
                    orjson.dumps(current_config_data, option=orjson.OPT_INDENT_2)
                )
        else:
            config_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_output_path, "wb") as f:
                f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))

    def get_forecast_info(self, nc_file: Path) -> tuple:
        """
        Extract metadata from netCDF file for STAC generation.

        Opens the netCDF file and extracts spatial coordinates, time/forecast
        information, valid bands, and other metadata required for creating
        STAC Items and Collections.

        Args:
            nc_file: Path to input netCDF file to process.

        Returns:
            tuple containing:
                - crs (str): Coordinate Reference System of dataset.
                - bbox (list): Bounding box [west, south, east, north] in WGS84.
                - geometry (dict): GeoJSON-like representation of bounding box.
                - valid_bands (list[str]): List of 4D variables to process as bands.
                - x_coord (str): Name of X coordinate dimension.
                - y_coord (str): Name of Y coordinate dimension.
                - time_coords (xr.DataArray): Time coordinates from dataset.
                - time_coords_start (datetime): Start datetime of temporal extent.
                - time_coords_end (datetime): End datetime of temporal extent.
                - leadtime_coords (xr.DataArray): Leadtime coordinates from dataset.
        """
        with xr.open_dataset(nc_file, decode_coords="all") as ds:
            info, _ = self._forecast_info_from_dataset(ds)
            return info

    def _forecast_info_from_dataset(
        self, ds: xr.Dataset
    ) -> tuple[tuple, xr.Dataset]:
        """
        Extract forecast metadata from an already-open dataset.

        Returns:
            A pair of ``(info_tuple, ds)`` where ``info_tuple`` matches
            ``get_forecast_info`` and ``ds`` has km->m unit conversion applied.
        """
        # Determine spatial coordinates
        x_coord = find_coord(ds, ["xc", "x", "lon", "longitude"])
        y_coord = find_coord(ds, ["yc", "y", "lat", "latitude"])

        # Get time-related coordinate information
        time_coords: xr.DataArray = ds.coords.get(
            "time", ds.coords.get("forecast_time")
        )
        leadtime_coords: xr.DataArray = ds.coords.get(
            "leadtime", ds.coords.get("lead_time")
        )

        if x_coord is None or y_coord is None:
            raise ValueError("Spatial coordinates not found in dataset")

        # Convert km to m if needed
        ds = self._convert_units(ds, x_coord, y_coord)

        # Filter 4D variables - these are variables of interest for COGs
        # Assuming other vars shouldn't be converted to COGs
        valid_bands = [var for var in ds.data_vars if len(ds[var].dims) == 4]

        # Get attributes from NetCDF file
        nc_attrs = ds.attrs
        crs = nc_attrs["geospatial_bounds_crs"]
        ds.rio.write_crs(crs, inplace=True)

        # Get bounding box of dataset (in expected "EPSG:4326")
        bbox, geometry = self._get_bbox_and_geometry(ds, x_coord, y_coord, crs)

        # Get temporal bounds from input netCDF
        time_coords_start = pd.to_datetime(time_coords.isel(time=0).values)
        time_coords_end = pd.to_datetime(time_coords.isel(time=-1).values)

        info = (
            crs,
            bbox,
            geometry,
            valid_bands,
            x_coord,
            y_coord,
            time_coords,
            time_coords_start,
            time_coords_end,
            leadtime_coords,
        )
        return info, ds

    def _convert_units(self, ds: xr.Dataset, x_coord: str, y_coord: str) -> xr.Dataset:
        """
        Convert coordinate units from kilometers to meters if needed.

        Checks the units of X and Y coordinates. If they are in "km" or "1000 meter",
        converts them to meters by multiplying by 1000.

        Args:
            ds: xarray Dataset containing the data.
            x_coord: Name of the X coordinate dimension.
            y_coord: Name of the Y coordinate dimension.

        Returns:
            Modified dataset with coordinates in meters.
        """
        # Convert eastings and northings from kilometers to metres (if need to).
        if ds.coords[y_coord].attrs.get("units", None) in ["1000 meter", "km"]: # `1000 meter` is legacy support for `icenet < v0.4.0``
            ds = ds.assign_coords({y_coord: ds.coords[y_coord] * 1000})
        if ds.coords[x_coord].attrs.get("units", None) in ["1000 meter", "km"]:
            ds = ds.assign_coords({x_coord: ds.coords[x_coord] * 1000})
        return ds

    def _get_bbox_and_geometry(
        self, ds: xr.Dataset, x_coord: str, y_coord: str, crs: str
    ) -> tuple:
        """
        Calculate bounding box and geometry from dataset coordinates.

        Computes the minimum and maximum values of X/Y coordinates to form a
        bounding box. Converts this to WGS84 (EPSG:4326) if the CRS is not already
        in WGS84. Returns both the numeric bounding box and GeoJSON-like geometry.

        Args:
            ds: xarray Dataset containing coordinate data.
            x_coord: Name of X coordinate dimension.
            y_coord: Name of Y coordinate dimension.
            crs: Coordinate Reference System of dataset.

        Returns:
            tuple: (bbox, geometry), where:
                - bbox is a list [west, south, east, north] in WGS84
                - geometry is GeoJSON-like dictionary representing the bounding box
        """
        # Compute bounding box and geometry from coordinates
        x_min, x_max = float(ds[x_coord].min()), float(ds[x_coord].max())
        y_min, y_max = float(ds[y_coord].min()), float(ds[y_coord].max())
        bbox = [x_min, y_min, x_max, y_max]

        # If projected CRS, convert to WGS84
        if crs not in ["EPSG:4326", "4326"]:
            bbox = proj_to_geo(bbox_projected=bbox, src_crs=crs)
        geometry = mapping(box(*bbox)) # type: ignore
        return bbox, geometry

    def process(
        self,
        nc_file: Path,
        name: str,
        compress: bool = True,
        overwrite: bool = False,
        stac_only: bool = False,
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        """
        Process a netCDF file and generate STAC Items/Collections for forecast data.

        Processes the input netCDF file to extract metadata, create STAC Items
        representing each forecast leadtime, and generate associated COG assets.
        Creates a STAC structure with collections for model name and forecast date.
        Does not write the catalog to disk; call ``save_catalog()`` after a batch
        of files (see ``preprocess.main``).

        Args:
            nc_file: Path to the input netCDF file.
            name: Collection identifier to place processed data into.
            compress: Whether to compress COG output using ZSTD.
                Defaults to True.
            overwrite: Whether to overwrite existing files.
                Defaults to False.
            stac_only: If True, only generate STAC without writing netCDF/COG files.
                Defaults to False
            workers: Number of parallel processes for COG generation.
                Defaults to 1
        """
        self._collection_name = name
        self._compress = compress
        self._compress_method = "ZSTD" if compress else "NONE"
        self._overwrite = overwrite

        nc_file = Path(nc_file).resolve()
        hemisphere = get_hemisphere(nc_file)

        # Initialise output paths
        self._set_out_paths()

        catalog = self.catalog

        # Open once: metadata extraction and COG/netCDF writing share this handle
        ds = xr.open_dataset(nc_file, decode_coords="all")
        try:
            (
                (
                    crs,
                    bbox,
                    geometry,
                    valid_bands,
                    x_coord,
                    y_coord,
                    time_coords,
                    time_coords_start,
                    time_coords_end,
                    leadtime_coords,
                ),
                ds,
            ) = self._forecast_info_from_dataset(ds)
            if leadtime_coords is None:
                raise ValueError(
                    "Dataset is missing leadtime / lead_time coordinates"
                )
            nleadtime = len(leadtime_coords)
            lead_dim = leadtime_coords.dims[0]
            time_dim = time_coords.dims[0]

            # Infer frequency from the first init for config.json consistency checks
            # (valid times themselves come from forecast_date / leadtime values).
            first_time = time_coords.isel({time_dim: 0})
            first_slice = ds.sel({time_dim: first_time})
            first_ref = pd.to_datetime(first_time.values)
            sample_valid_times = resolve_valid_times(
                first_ref, leadtime_coords, first_slice
            )
            self._forecast_frequency = forecast_frequency_from_valid_times(
                sample_valid_times
            )
            logger.info(
                "Inferred forecast_frequency=%s from lead coordinates",
                self._forecast_frequency,
            )
            self._validate_input_options()

            # Create (or retrieve) highest level collection (model name) within the catalog
            collection = self.get_or_create_collection(
                parent=catalog,
                collection_id=name,
                title=f"{name}",
                description=f"{name.capitalize().replace('_', ' ').replace('-', ' ')} collection",
                bbox=bbox,
                extra_fields={"custom:hemisphere": hemisphere} if hemisphere else None,
                license=self._license,
                temporal_extent=[time_coords_start, time_coords_end],
            )

            for time_idx, time_val in enumerate(time_coords):
                ds_time_slice = ds.sel({time_dim: time_val})

                # The forecast initialisation time (CF Convention: `forecast_reference_time`)
                # is the first forecast being predicted
                forecast_reference_time = pd.to_datetime(time_val.values)
                forecast_reference_date = forecast_reference_time.date()
                forecast_reference_time_str = datetime_to_str(forecast_reference_time)
                # Filesystem-safe format for item IDs and netCDF filenames
                forecast_reference_time_filesafe = format_time(forecast_reference_time)

                # Absolute valid times per lead (handles irregular spacing)
                valid_times = resolve_valid_times(
                    forecast_reference_time, leadtime_coords, ds_time_slice
                )
                forecast_end_time = valid_times[-1]
                forecast_end_time_str = datetime_to_str(forecast_end_time)

                # Create output dirs
                ncdf_dir = Path(
                    self._netcdf_output_dir / f"{forecast_reference_date}"
                )
                cog_dir = Path(
                    self._cogs_output_dir / f"{forecast_reference_date}"
                )
                item_id = f"forecast_init_{forecast_reference_time_filesafe}"

                ncdf_dir.mkdir(parents=True, exist_ok=True)
                cog_dir.mkdir(parents=True, exist_ok=True)

                # Save the forecast init slice as a netcdf file
                out_nc_file = ncdf_dir / f"{forecast_reference_time_filesafe}.nc"

                # Write the netCDF file in addition to the STAC json output
                if not stac_only:
                    self._write_netcdf(ds_time_slice, out_nc_file)


                properties = {
                    "forecast:reference_time": forecast_reference_time_str,
                    "forecast:end_time": forecast_end_time_str,
                    "forecast:leadtime_length": nleadtime,
                }

                nc_metadata = get_nc_attributes(ds_time_slice.attrs)
                properties |= nc_metadata

                # Add STAC Item for this netCDF file
                item = self.get_or_create_item(
                    collection=collection,
                    item_id=item_id,
                    geometry=geometry,
                    bbox=bbox,
                    datetime=forecast_reference_time,  # Becomes "Time of Data" property, under Metadata -> General in STAC-Browser
                                                       # Used for temporal filtering of items
                    start_datetime=forecast_reference_time,
                    end_datetime=forecast_end_time,
                    crs=crs,
                    properties=properties,
                )
                # Add file extension
                item.ext.add("file")

                # Add netCDF asset to item
                nc_asset = Asset(
                    href=str(out_nc_file.resolve()),
                    media_type=pystac.MediaType.NETCDF,
                    title=f"Full forecast netCDF from {forecast_reference_time.strftime(DT_FMT_DISPLAY)}",
                    description=(
                        "netCDF file container forecast variables for forecast"
                        f" initialised at: {forecast_reference_time_str}"
                    ),
                    roles=["data"],
                    extra_fields={
                        "forecast:reference_time": forecast_reference_time_str,
                        "forecast:end_time": forecast_end_time_str,
                        "forecast:leadtime_length": nleadtime,
                    },
                )

                item.add_asset(key="netcdf", asset=nc_asset)
                nc_asset = add_file_info_to_asset(nc_asset, nc_asset.href)

                # Load the forecast-init slice once, then hand each worker only its
                # leadtime. `.load()` materialises that slice so pickling stays small
                # without an extra deep copy of the xarray structure.
                ds_time_slice = ds_time_slice.load()
                common_args = (
                    forecast_reference_time,
                    x_coord,
                    y_coord,
                    crs,
                    cog_dir,
                    stac_only,
                    item_id,
                    valid_bands,
                    overwrite,
                    self._compress_method,
                )

                # Reuse a process pool across files/inits. Use a staticmethod so
                # ProcessPoolExecutor does not pickle this STACGenerator / Catalog
                # (which mutates as assets are added and caused "dictionary changed
                # size during iteration").
                executor = self._get_executor(workers)
                with tqdm(total=nleadtime, desc="COGifying files", leave=True) as pbar:
                    futures = []
                    for i in range(nleadtime):
                        ds_leadtime_slice = ds_time_slice.isel({lead_dim: i}).load()
                        future = executor.submit(
                            STACGenerator._process_leadtime,
                            i,
                            ds_leadtime_slice,
                            valid_times[i],
                            *common_args,
                        )
                        future.add_done_callback(lambda _: pbar.update(1))
                        futures.append(future)

                    for future in futures:
                        i, cog_file, assets, pbar_description = future.result()
                        pbar.set_description(pbar_description)
                        for asset in assets:
                            item.add_asset(key=asset["key"], asset=asset["asset"])
                            add_file_info_to_asset(asset["asset"], asset["asset"].href)
                            if asset["key"] == "thumbnail" and time_idx == 0 and i == 0:
                                if not collection.get_assets(role="thumbnail"):
                                    collection.add_asset(
                                        key=asset["key"], asset=asset["asset"]
                                    )
        finally:
            ds.close()

    @staticmethod
    def _process_leadtime(
        i: int,
        ds_leadtime_slice: xr.Dataset,
        valid_time: datetime,
        forecast_reference_time: datetime,
        x_coord: str,
        y_coord: str,
        crs: str,
        cog_dir: Path,
        stac_only: bool,
        item_id: str,
        valid_bands: list[str],
        overwrite: bool,
        compress_method: str,
        reproject: bool = False,
    ):
        """
        Process a single leadtime slice to generate COG and thumbnail assets.

        Args:
            i: Index of the current leadtime.
            ds_leadtime_slice: In-memory xarray Dataset for this leadtime only.
            valid_time: Absolute valid time for this lead.
            forecast_reference_time: The forecast initialisation time.
            x_coord: X dimension coordinate name.
            y_coord: Y dimension coordinate name.
            crs: Coordinate Reference System (EPSG code).
            cog_dir: Output directory path for COG files.
            stac_only: Whether to generate only STAC metadata (no COGs or thumbnails).
            item_id: STAC Item id used to build COG/thumbnail filenames.
            valid_bands: List of valid variable names to process.
                i.e., having 4 dimensions (time, yc, xc, leadtime).
            overwrite: Whether to overwrite existing files.
            compress_method: COG compression method (e.g. ``ZSTD``, ``DEFLATE``, or ``NONE``).
            reproject: Whether to reproject to EPSG:4326.
                Defaults to False.

        Returns:
            Tuple containing:
                - cog_file: Path to the generated COG file.
                - assets: List of asset dictionaries with metadata.
                - pbar_description: Description for progress bar updates.
        """
        # Set spatial dimensions for rioxarray
        ds_leadtime_slice.rio.set_spatial_dims(
            x_dim=x_coord, y_dim=y_coord, inplace=True
        )

        valid_time = pd.Timestamp(valid_time).to_pydatetime()
        valid_time_str = datetime_to_str(valid_time)

        # Add STAC Item for this file
        item_id_cog = f"{item_id}_lead_{valid_time.strftime(DT_FMT_FILENAME)}"

        # Define cog/thumbnail output paths
        cog_file = cog_dir / f"{item_id_cog}.tif"
        thumbnail_file = cog_dir / f"{item_id_cog}.jpg"

        # Save variables as one multi-band COG (Cloud Optimized GeoTIFF) & JPG
        # (for thumbnail)
        da_list = []
        band_names = valid_bands
        band_metadata = []
        band_statistics: list[dict] = []
        for bidx, var_name in enumerate(band_names, start=1):
            da_variable = ds_leadtime_slice[var_name]
            da_variable.rio.write_crs(crs, inplace=True)
            da_variable.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)

            da_list.append(da_variable)
            metadata = {"name": var_name, "index": bidx}
            nc_attrs = da_variable.attrs

            nc_metadata = get_nc_attributes(nc_attrs)
            metadata |= nc_metadata

            # Only include statistics if not reprojecting, else stats will be different
            # would need to add after reprojecting.
            if not reproject:
                stats = get_da_statistics(da_variable)
                metadata |= stats
                band_statistics.append(stats)
            band_metadata.append(metadata)

        if not stac_only:
            # Stack variables as a single dataset
            da_multiband = xr.concat(da_list, dim="band")
            da_multiband = da_multiband.assign_coords(band=("band", band_names))

            if cog_file.exists() and not overwrite:
                pbar_description = f"File already exists, skipping: {cog_file}"
            else:
                pbar_description = f"Saving vars to multi-band COG: {cog_file}"

                STACGenerator._write_cog(
                    da_multiband,
                    x_coord,
                    y_coord,
                    crs,
                    cog_file,
                    compress_method,
                    reproject=reproject,
                    band_statistics=band_statistics or None,
                )

            # Create thumbnail plot for only the first variable for the first leadtime
            if i == 0:
                if not thumbnail_file.exists() or overwrite:
                    STACGenerator._create_and_write_thumbnail(
                        da_multiband,
                        thumbnail_file,
                        forecast_reference_time,
                        valid_time,
                    )

            with rasterio.open(cog_file) as src:
                width = src.width
                height = src.height
                transform = list(src.transform)[:6]
                epsg_code = src.crs.to_epsg()
        else:
            pbar_description = f"Processing STAC: {item_id_cog}"
            # Derive projection metadata from the in-memory array when no COG is written
            da_ref = da_list[0]
            width = int(da_ref.rio.width)
            height = int(da_ref.rio.height)
            transform = list(da_ref.rio.transform())[:6]
            epsg_code = da_ref.rio.crs.to_epsg() if da_ref.rio.crs else None

        assets = []
        # Add COG asset to item
        cog_asset = dict(
            key=valid_time_str,
            asset=Asset(
                href=str(cog_file.resolve()),
                media_type=pystac.MediaType.COG,
                title=f"Forecast at {valid_time.strftime(DT_FMT_DISPLAY)}",
                description=f"Variables: {', '.join(band_names)}",
                roles=["data"],
                extra_fields={
                    "forecast:bands": band_metadata,
                    "custom:leadtime": i,
                    "custom:valid_time": valid_time.strftime(DT_FMT_ISO8601),
                    "proj:transform": transform,  # Affine transformation
                    "proj:shape": [height, width],  # Note: shape = [rows, cols]
                    "proj:epsg": epsg_code,  # Optional but good to include
                },
            ),
        )
        assets.append(cog_asset)

        # Thumbnail asset only when we wrote (or expect) a thumbnail file
        if i == 0 and not stac_only:
            # Some STAC tools may only show the first thumbnail asset
            thumbnail_asset = dict(
                key="thumbnail",
                asset=Asset(
                    href=str(thumbnail_file.resolve()),
                    media_type=pystac.MediaType.JPEG,
                    title="Thumbnail",
                    roles=["thumbnail"],
                ),
            )
            assets.append(thumbnail_asset)

        return i, cog_file, assets, pbar_description

    def _write_netcdf(self, ds_time_slice: xr.Dataset, out_nc_file: Path):
        """
        Write a time-slice dataset to a netCDF file.

        Args:
            ds_time_slice: xarray Dataset slice containing data to write.
            out_nc_file: Path to the output netCDF file.
        """
        encoding = {
            var: {
                "zlib": True,
                "complevel": 9,
            } for var in ds_time_slice.data_vars
        }
        ds_time_slice.to_netcdf(
            out_nc_file,
            engine="h5netcdf",
            encoding=encoding,
        )

    @staticmethod
    def _write_cog(
        da_multiband: xr.DataArray,
        x_coord: str,
        y_coord: str,
        crs: str,
        cog_file: Path,
        compress_method: str,
        reproject: bool = False,
        band_statistics: list[dict] | None = None,
    ):
        """
        Write a multiband DataArray as a Cloud Optimized GeoTIFF (COG).

        Args:
            da_multiband: xarray DataArray with band dimension.
            x_coord: X dimension coordinate name.
            y_coord: Y dimension coordinate name.
            crs: Coordinate Reference System (EPSG code).
            cog_file: Path to the output COG file.
            compress_method: COG compression method (e.g. ``ZSTD``, ``DEFLATE``, or ``NONE``).
            reproject: Whether to reproject to EPSG:4326.
                Defaults to False.
            band_statistics: Optional per-band statistics already computed for
                STAC metadata; reused as COG band tags to avoid a second pass.
        """
        # Add metadata to extracted variable so `to_raster` includes them in the output
        # GeoTIFF
        da_multiband.rio.write_crs(crs, inplace=True)
        da_multiband.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)
        if reproject:
            da_multiband = da_multiband.rio.reproject("EPSG:4326", inplace=False)
            # Stats from the source grid no longer apply after reprojection.
            band_statistics = None
        write_cog(
            cog_file,
            da_multiband,
            compress=compress_method,
            band_statistics=band_statistics,
        )

    @staticmethod
    def _create_and_write_thumbnail(
        da_multiband: xr.DataArray,
        thumbnail_file: Path,
        forecast_reference_time: datetime,
        valid_time: datetime,
    ):
        """
        Generate and save a thumbnail image of the first band in the dataset.

        Args:
            da_multiband: xarray DataArray containing multiband data.
            thumbnail_file: Path to save the generated JPEG thumbnail.
            forecast_reference_time: Forecast initialization time (datetime).
            valid_time: Valid time for the leadtime being processed (datetime).
        """
        fig = plt.figure(figsize=(5, 5), dpi=300, constrained_layout=True)
        da_multiband.isel(band=0).plot(cmap="RdBu_r", add_colorbar=False)  # type: ignore
        # plt.title(f"Init: {forecast_reference_time}\nLeadtime: {valid_time}")
        plt.title("")
        plt.axis("off")
        plt.savefig(thumbnail_file, pad_inches=0, bbox_inches="tight", transparent=False)
        plt.close(fig)

    def save_catalog(self):
        """
        Save the STAC catalog with portable asset hrefs.

        Normalises catalog/collection/item links under `data/stac/`, then rewrites
        asset hrefs to cwd-relative paths (e.g. `data/cogs/...`) so the static JSON
        has no host. `FILE_SERVER_URL` is applied later at ingest.

        Collection summaries (forecast init times and variables) are refreshed
        from Items before writing so clients can build date pickers without
        listing every Item.
        """
        catalog = self.catalog
        catalog.normalize_hrefs(str(self._stac_output_dir))

        for collection in catalog.get_all_collections():
            refresh_collection_summaries(collection)
            for asset in collection.assets.values():
                asset.href = to_cwd_relative_href(asset.href)
        for item in catalog.get_items(recursive=True):
            for asset in item.assets.values():
                asset.href = to_cwd_relative_href(asset.href)

        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
