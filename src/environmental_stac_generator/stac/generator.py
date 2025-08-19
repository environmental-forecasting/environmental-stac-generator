import logging
import os
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import orjson
import pandas as pd
import pystac
import xarray as xr
from dateutil.relativedelta import relativedelta
from deepdiff import DeepDiff
from dotenv import load_dotenv
from pystac import Asset, Catalog, Collection, Item
from pystac.extensions.projection import ProjectionExtension
from pystac.utils import datetime_to_str, str_to_datetime
from shapely.geometry import box, mapping
from tqdm import tqdm

from ..cog import write_cog
from ..utils import (
    find_coord,
    get_hemisphere,
    parse_forecast_frequency,
    proj_to_geo,
    ensure_utc,
    format_time,
    get_da_statistics,
    get_nc_attributes,
)
from .utils import add_file_info_to_asset, ConfigMismatchError

logger = logging.getLogger(__name__)


class BaseSTAC:
    def __init__(
        self,
        data_path: Path = Path("data"),
        catalog_defs: dict | None = None,
        catalog_name: str = "default",
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
            catalog_name (optional): The catalog dir where the generated STAC catalog JSON
                file should be saved.
                Defaults to "default".
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

        self._load_dotenv()
        self._set_catalog_path(catalog_name=catalog_name)
        self.get_or_create_catalog(catalog_defs=catalog_defs)

    def _load_dotenv(self) -> None:
        """
        Load environment variables and configure the file server URL.

        This method loads the `.env` file and retrieves the `FILE_SERVER_URL`
        environment variable, which is used as the base URL for STAC catalog hrefs.
        """
        # The base URL for the STAC catalogs.
        # If set, the root of the STAC href links will use this url as the base path
        load_dotenv()
        self._FILE_SERVER_URL = os.getenv("FILE_SERVER_URL", None)
        logger.info(f"FILE_SERVER_URL: {self._FILE_SERVER_URL}")

    def _set_catalog_path(self, catalog_name: str) -> None:
        """
        Configure the STAC output directory and catalog file path.

        Creates the 'stac' subdirectory within `self.data_path` if it doesn't exist.
        If no custom catalog file is provided, uses 'catalog.json' as the default
        catalog file name. Otherwise, uses the specified file path.

        Args:
            catalog_file_name: Optional dirname to use where the STAC catalog file
                will be saved.
                If None, defaults to 'default' in the STAC output directory.
        """
        # This has dir with `name` created by itself
        self._stac_output_dir = self.data_path / "stac" / catalog_name
        self._stac_output_dir.mkdir(parents=True, exist_ok=True)

        self._stac_catalog_file = Path(self._stac_output_dir) / "catalog.json"

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

    def save_catalog(self, stac_output_dir):
        """
        Save the STAC catalog to disk with normalized HREFs.

        Normalises all href references in the STAC catalog to use absolute paths
        relative to the output directory and writes the catalog files to disk.

        Args:
            stac_output_dir: Directory path where STAC catalog files will be saved.
        """
        self._stac_catalog.normalize_hrefs(str(self._stac_catalog_file))
        self._stac_catalog.save()

    @property
    def catalog(self) -> Catalog:
        """Get the current STAC catalog.

        Provides access to the internal STAC catalog used for data processing.
        """
        if self._stac_catalog is None:
            raise ValueError("STAC catalog not initialised.")
        return self._stac_catalog

    def get_collection(self, collection_id):
        """Get the current STAC catalog.

        Provides access to the internal STAC catalog used for data processing.
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

    # @abstractmethod
    # def create_collection(self) -> None:
    #     pass

    # @abstractmethod
    # def add_item_to_collection(self) -> None:
    #     pass


class STACGenerator(BaseSTAC):
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
        that the new configuration matches the existing one. Otherwise, creates
        the file and stores the provided configuration.

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

            return (
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
        forecast_frequency: str = "1days",
        compress: bool = True,
        overwrite: bool = False,
        stac_only: bool = False,
        workers: int = 1,
    ) -> None:
        """
        Process a netCDF file and generate STAC Items/Collections for forecast data.

        Processes the input netCDF file to extract metadata, create STAC Items
        representing each forecast leadtime, and generate associated COG assets.
        Creates a STAC structure with collections for model name and forecast date.

        Args:
            nc_file: Path to the input netCDF file.
            name: Collection identifier to place processed data into.
            forecast_frequency: Frequency of forecasts (e.g., "1days").
                Defaults to "1days".
            compress: Whether to compress COG output using DEFLATE.
                Defaults to True.
            overwrite: Whether to overwrite existing files.
                Defaults to False.
            stac_only: If True, only generate STAC without writing netCDF/COG files.
                Defaults to False
            workers: Number of parallel processes for COG generation.
                Defaults to 1
        """
        self._collection_name = name
        self._forecast_frequency = forecast_frequency
        self._compress_method = "DEFLATE" if compress else "NONE"
        nc_file = Path(nc_file).resolve()
        hemisphere = get_hemisphere(nc_file)

        # Initialise output paths
        self._set_out_paths()

        self._validate_input_options()

        # Get input time delta options to compute forecast times
        leadtime_step, leadtime_unit = parse_forecast_frequency(forecast_frequency)

        catalog = self.catalog

        # Get required coords and metadata from forecast netCDF file
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
        ) = self.get_forecast_info(nc_file)
        nleadtime = len(leadtime_coords)

        # Create (or retrieve) highest level collection (model name) within the catalog
        collection = self.get_or_create_collection(
            parent=catalog,
            collection_id=name,
            title=f"{name}",
            description=f"{name.capitalize().replace("_", " ").replace("-", " ")} collection",
            bbox=bbox,
            extra_fields = {"custom:hemisphere": hemisphere} if hemisphere else None,
            license=self._license,
            temporal_extent=[time_coords_start, time_coords_end],
        )

        ds = xr.open_dataset(nc_file, decode_coords="all")
        # Convert km to m if needed
        ds = self._convert_units(ds, x_coord, y_coord)
        for time_idx, time_val in enumerate(time_coords):
            ds_time_slice = ds.sel(time=time_val)

            # The forecast initialisation time (CF Convention: `forecast_reference_time`)
            # is the first forecast being predicted
            forecast_reference_time = pd.to_datetime(time_val.values)
            forecast_reference_date = forecast_reference_time.date()
            forecast_reference_time_str = datetime_to_str(forecast_reference_time)
            forecast_reference_time_str_1 = forecast_reference_time.strftime(
                "%Y-%m-%d_%H:%M"
            )
            forecast_reference_time_str_2 = forecast_reference_time.strftime(
                "%Y-%m-%d %H:%M"
            )
            forecast_reference_time_str_3 = format_time(forecast_reference_time)

            forecast_end_time = forecast_reference_time + relativedelta(
                **{leadtime_unit: nleadtime - 1} # type: ignore
            )
            forecast_end_time_str = datetime_to_str(forecast_end_time)
            # forecast_end_time_str_1 = forecast_end_time.strftime("%Y-%m-%d_%H:%M")
            # forecast_end_time_str_2 = forecast_end_time.strftime("%Y-%m-%d %H:%M")
            # forecast_end_time_str_3 = forecast_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Create output dirs
            ncdf_dir = Path(
                self._netcdf_output_dir / f"{forecast_reference_date}"
            )
            cog_dir = Path(
                self._cogs_output_dir / f"{forecast_reference_date}"
            )
            item_id = f"forecast_init_{forecast_reference_time_str_3}"

            ncdf_dir.mkdir(parents=True, exist_ok=True)
            cog_dir.mkdir(parents=True, exist_ok=True)

            # Save the forecast init slice as a netcdf file
            out_nc_file = ncdf_dir / f"{forecast_reference_time_str_3}.nc"

            # Write the netCDF file in addition to the STAC json output
            if not stac_only:
                # with xr.open_dataset(nc_file, decode_coords="all") as ds:
                #     ds_time_slice = ds.sel(time=time_val)
                    self._write_netcdf(ds_time_slice, out_nc_file)


            properties={
                "forecast:reference_time": forecast_reference_time_str,
                "forecast:end_time": forecast_end_time_str,
                "forecast:leadtime_length": nleadtime,
            }

            nc_metadata = get_nc_attributes(ds_time_slice.attrs)
            properties |=  nc_metadata

            # Add STAC Item for this netCDF file
            item = self.get_or_create_item(
                collection=collection,
                item_id=item_id,
                geometry=geometry,
                bbox=bbox,
                datetime=forecast_reference_time, # Becomes "Time of Data" property, under Metadata -> General in STAC-Browser
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
                    href=str(out_nc_file),
                    media_type=pystac.MediaType.NETCDF,
                    title=f"Full forecast netCDF from {forecast_reference_time_str_2}",
                    description="netCDF file container forecast variables for forecast"
                                f" initialised at: {forecast_reference_time_str}",
                    roles=["data"],
                    extra_fields={
                        "forecast:reference_time": forecast_reference_time_str,
                        "forecast:end_time": forecast_end_time_str,
                        "forecast:leadtime_length": nleadtime,
                    }
                )

            item.add_asset(key="netcdf", asset=nc_asset)
            nc_asset = add_file_info_to_asset(nc_asset, nc_asset.href)

            process_args = (
                forecast_reference_time,
                leadtime_unit,
                leadtime_step,
                ds_time_slice,
                x_coord,
                y_coord,
                crs,
                cog_dir,
                stac_only,
                item,
                valid_bands,
                overwrite,
            )

            # for i in range(nleadtime):
            #     i, cog_file, assets, pbar_description = self._process_leadtime(i, *process_args)
            #     for asset in assets:
            #         item.add_asset(key=asset["key"], asset=asset["asset"])
            #         add_file_info_to_asset(asset["asset"], asset["asset"].href)
            #         # Use the first thumbnail generated for this item as the
            #         # thumbnail for the collection as well.
            #         if asset["key"] == "thumbnail" and time_idx == 0 and i == 0:
            #             # Skip if the collection already has a thumbnail asset
            #             if not collection.get_assets(role="thumbnail"):
            #                 collection.add_asset(key=asset["key"], asset=asset["asset"])

            # Process each leadtime
            with ProcessPoolExecutor(max_workers=workers) as executor:
                with tqdm(total=nleadtime, desc="COGifying files", leave=True) as pbar:
                    futures = []
                    for i in range(nleadtime):
                        future = executor.submit(
                            self._process_leadtime, i, *process_args
                        )
                        future.add_done_callback(lambda _: pbar.update(1))
                        futures.append(future)

                    # Wait for all futures to complete
                    for future in futures:
                        i, cog_file, assets, pbar_description = future.result()
                        pbar.set_description(pbar_description)
                        for asset in assets:
                            item.add_asset(key=asset["key"], asset=asset["asset"])
                            add_file_info_to_asset(asset["asset"], asset["asset"].href)
                            # Use the first thumbnail generated for this item as the
                            # thumbnail for the collection as well.
                            if asset["key"] == "thumbnail" and time_idx == 0 and i == 0:
                                # Skip if the collection already has a thumbnail asset
                                if not collection.get_assets(role="thumbnail"):
                                    collection.add_asset(key=asset["key"], asset=asset["asset"])

        ds.close()

        # Save catalog and collections
        self.save_catalog()


    def _process_leadtime(
        self,
        i: int,
        forecast_reference_time: datetime,
        leadtime_unit: str,
        leadtime_step: int,
        ds_time_slice: xr.Dataset,
        x_coord: str,
        y_coord: str,
        crs: str,
        cog_dir: Path,
        stac_only: bool,
        item: pystac.Item,
        valid_bands: list[str],
        overwrite: bool,
        reproject: bool=False,
    ):
        """
        Process a single leadtime slice to generate COG and thumbnail assets.

        Args:
            i: Index of the current leadtime.
            forecast_reference_time: The forecast initialisation time.
            leadtime_unit: Unit of leadtime (e.g., 'days').
            leadtime_step: Step size for leadtimes.
            ds_time_slice: xarray Dataset slice for the current forecast time.
            x_coord: X dimension coordinate name.
            y_coord: Y dimension coordinate name.
            crs: Coordinate Reference System (EPSG code).
            cog_dir: Output directory path for COG files.
            stac_only: Whether to generate only STAC metadata (no COGs or thumbnails).
            item: Item to which assets will be added.
            valid_bands: List of valid variable names to process.
                i.e., having 4 dimensions (time, yc, xc, leadtime).
            overwrite: Whether to overwrite existing files.
            reproject: Whether to reproject to EPSG:4326.
                Defaults to False.

        Returns:
            Tuple containing:
                - cog_file: Path to the generated COG file.
                - assets: List of asset dictionaries with metadata.
                - pbar_description: Description for progress bar updates.
        """
        valid_time = forecast_reference_time + relativedelta(
            **{leadtime_unit: i * leadtime_step} # type: ignore
        )
        ds_leadtime_slice = ds_time_slice.isel(leadtime=i)

        # Set spatial dimensions for rioxarray
        ds_leadtime_slice.rio.set_spatial_dims(
            x_dim=x_coord, y_dim=y_coord, inplace=True
        )

        valid_time_str = datetime_to_str(valid_time)
        valid_time_str_1 = valid_time.strftime("%Y-%m-%d_%H%M")
        valid_time_str_2 = valid_time.strftime("%Y-%m-%d %H:%M")
        valid_time_str_3 = valid_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Add STAC Item for this file
        item_id_cog = f"{item.id}_lead_{valid_time_str_1}"

        # Define cog/thumbnail output paths
        cog_file = cog_dir / f"{item_id_cog}.tif"
        thumbnail_file = cog_dir / f"{item_id_cog}.jpg"

        # Save variables as one multi-band COG (Cloud Optimized GeoTIFF) & JPG
        # (for thumbnail)
        da_list = []
        band_names = valid_bands
        band_metadata = []
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
            band_metadata.append(metadata)

        if not stac_only:
            # Stack variables as a single dataset
            da_multiband = xr.concat(da_list, dim="band")
            da_multiband = da_multiband.assign_coords(band=("band", band_names))

            if cog_file.exists() and not overwrite:
                pbar_description = f"File already exists, skipping: {cog_file}"
            else:
                pbar_description = f"Saving vars to multi-band COG: {cog_file}"

                self._write_cog(da_multiband, x_coord, y_coord, crs, cog_file, reproject=reproject)

            # Create thumbnail plot for only the first variable for the first leadtime
            if i == 0:
                if not thumbnail_file.exists() or overwrite:
                    self._create_and_write_thumbnail(
                        da_multiband,
                        thumbnail_file,
                        forecast_reference_time,
                        valid_time,
                    )
        else:
            pbar_description = f"Processing STAC: {item_id_cog}"

        assets = []
        # Add COG asset to item
        cog_asset = dict(
            key=valid_time_str,
            asset=Asset(
                href=str(cog_file),
                media_type=pystac.MediaType.COG,
                title=f"Forecast at {valid_time_str_2}",
                description=f"Variables: {', '.join(band_names)}",
                roles=["data"],
                extra_fields={
                    "forecast:bands": band_metadata,
                    "custom:leadtime": i,
                    "custom:valid_time": valid_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                },
            ),
        )
        assets.append(cog_asset)

        # Create a thumbnail plot of the first variable for the first leadtime
        if i == 0:
            # Add thumbnail asset to item
            # Some STAC tools may only show the first thumbnail asset
            thumbnail_asset = dict(
                key="thumbnail",
                asset=Asset(
                    href=str(thumbnail_file),
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

    def _write_cog(
        self,
        da_multiband: xr.DataArray,
        x_coord: str,
        y_coord: str,
        crs: str,
        cog_file: Path,
        reproject: bool=False,
    ):
        """
        Write a multiband DataArray as a Cloud Optimized GeoTIFF (COG).

        Args:
            da_multiband: xarray DataArray with band dimension.
            x_coord: X dimension coordinate name.
            y_coord: Y dimension coordinate name.
            crs: Coordinate Reference System (EPSG code).
            cog_file: Path to the output COG file.
            reproject: Whether to reproject to EPSG:4326.
                Defaults to False.
        """
        # Add metadata to extracted variable so `to_raster` includes them in the output
        # GeoTIFF
        da_multiband.rio.write_crs(crs, inplace=True)
        da_multiband.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)
        if reproject:
            da_multiband = da_multiband.rio.reproject("EPSG:4326", inplace=False)
        # da_multiband.rio.to_raster(cog_path, driver="COG", compress=self._compress_method)
        write_cog(cog_file, da_multiband, compress=self._compress_method)

    def _create_and_write_thumbnail(
        self,
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
        Save the STAC catalog and update asset URLs with a base server URL.

        This method normalizes all HREFs in the catalog, replaces local file paths
        with a base URL if specified, and writes the final catalog to disk.
        """
        catalog = self.catalog
        ## Normalize HREFs for the catalog and save
        catalog.normalize_hrefs(str(self._stac_output_dir))

        ## Replace file path prefix in "href" with URL base
        FILE_SERVER_URL = self._FILE_SERVER_URL
        if FILE_SERVER_URL:
            if not FILE_SERVER_URL.endswith("/"):
                FILE_SERVER_URL += "/"

            for item in catalog.get_all_items():
                for asset_key, asset in item.assets.items():
                    # Replace relative file path with URL
                    if asset.href.startswith("./"):
                        asset.href = FILE_SERVER_URL + asset.href.lstrip("./")

        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
