import logging
import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

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
from shapely.geometry import box, mapping
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from ..cog import write_cog
from ..utils import (
    find_coord,
    get_hemisphere,
    parse_forecast_frequency,
    proj_to_geo,
)

logger = logging.getLogger(__name__)

class BaseSTAC:
    def __init__(
        self,
        data_path: Path = Path("data"),
        catalog_defs: dict | None = None,
        stac_catalog_file: Path | None = None,
        license: str | None = None,
    ):
        """
        Initialises the BaseSTAC class with a path to an existing STAC catalog.
        Args:
            stac_catalog_path (str|Path): The path to the STAC catalog.
                If the path does not exist, a new catalog will be created at this location.
        """
        if not catalog_defs:
            catalog_defs = {
                    "id": "bas-environmental-forecasts",
                    "description": "Catalog of BAS Environmental Forecast Data",
                    "title": "BAS Environmental Forecasting STAC Catalog",
            }

        self._data_path = Path("data")
        self._license = license if license else "OGL-UK-3.0"   # Ref: https://spdx.org/licenses/

        self._load_dotenv()
        self._set_catalog_path()
        self.get_or_create_catalog(catalog_defs=catalog_defs)

    def _load_dotenv(self) -> None:
        # The base URL for the STAC catalogs.
        # If set, the root of the STAC href links will use this url as the base path
        load_dotenv()
        self._FILE_SERVER_URL = os.getenv("FILE_SERVER_URL", None)
        logger.info(f"FILE_SERVER_URL: {self._FILE_SERVER_URL}")

    def _set_catalog_path(self, stac_catalog_file=None) -> None:
        # This has dir with `name` created by itself
        self._stac_output_dir = self.data_path / "stac"
        self._stac_output_dir.mkdir(parents=True, exist_ok=True)

        if not stac_catalog_file:
            self._stac_catalog_file = Path(self._stac_output_dir) / "catalog.json"
        else:
            self._stac_catalog_file = stac_catalog_file

    def get_or_create_catalog(self, catalog_defs: dict, describe: bool = True) -> Catalog:
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
        license: str = "other",
    ) -> Collection:
        collection = next(
            (c for c in parent.get_children() if c.id == collection_id), None
        )
        if not collection:
            collection = Collection(
                id=collection_id,
                title=title,
                description=description,
                license=license,
                extent=pystac.Extent(
                    pystac.SpatialExtent([bbox]),
                    pystac.TemporalExtent([temporal_extent]),
                ),
            )
            parent.add_child(collection)
        return collection

    def create_stac_item(
        self,
        item_id: str,
        geometry: dict,
        bbox: list,
        datetime: datetime,
        crs: str,
        properties: dict,
        netcdf_path: Path,
        title: str,
        description: str,
    ) -> Item:
        item = Item(
            id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=datetime,
            properties=properties,
        )
        ProjectionExtension.add_to(item).code = crs
        item.add_asset(
            "netcdf",
            Asset(
                href=str(netcdf_path),
                media_type=pystac.MediaType.NETCDF,
                title=title,
                description=description,
                roles=["data"],
            ),
        )
        return item

    def create_multiband_raster(
        self,
        ds: xr.Dataset,
        crs: str,
        x_coord: str,
        y_coord: str,
        valid_bands: list[str],
    ) -> tuple[xr.DataArray, list[str]]:
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
        self._stac_catalog.normalize_hrefs(str(stac_output_dir))
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
        return self._data_path

    @data_path.getter
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        self._data_path = value

    @abstractmethod
    def process(self, nc_file: Path, **kwargs):
        pass

    # @abstractmethod
    # def create_collection(self) -> None:
    #     pass

    # @abstractmethod
    # def add_item_to_collection(self) -> None:
    #     pass


class STACGenerator(BaseSTAC):
    def _set_out_paths(self) -> None:
        data_path = self.data_path
        collection_name = self._collection_name
        self._netcdf_output_dir = data_path / "netcdf" / collection_name
        self._cogs_output_dir = data_path / "cogs" / collection_name
        # Store a config file of how the preprocessor was run
        self._config_output_path = data_path / "config.json"

    def _validate_input_options(self):
        # Store input options to file
        config_data = {
            self._collection_name: {
                "forecast_frequency": self._forecast_frequency,
                "flat": self._flat,
            }
        }
        self._store_config(config_data)

    def _store_config(self, config_data: dict):
        collection_name = self._collection_name
        config_output_path = self._config_output_path

        # Ensure we're running with same options as any previous runs
        if config_output_path.exists():
            with open(config_output_path, "rb") as f:
                current_config_data = orjson.loads(f.read())
                if collection_name in current_config_data:
                    diff = DeepDiff(config_data[collection_name], current_config_data[collection_name])
                    if diff:
                        logger.error(
                            f"You are attempting to generate collection ({collection_name}) with "
                            "different options to previous! Run with old values (below) to continue!"
                        )
                        logger.error(current_config_data[collection_name])
                        exit(1)
        else:
            config_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_output_path, "wb") as f:
                f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))

    def get_forecast_info(self, nc_file: Path):
        with xr.open_dataset(nc_file, decode_coords="all") as ds:
            # Determine spatial coordinates
            x_coord = find_coord(ds, ["xc", "x", "lon", "longitude"])
            y_coord = find_coord(ds, ["yc", "y", "lat", "latitude"])

            # Get time-related coordinate information
            time_coords: xr.DataArray = ds.coords.get("time", ds.coords.get("forecast_time"))
            leadtime_coords: xr.DataArray = ds.coords.get("leadtime", ds.coords.get("lead_time"))

            if x_coord is None or y_coord is None:
                raise ValueError("Spatial coordinates not found in dataset")

            # Convert km to m if needed
            ds = self._convert_units(ds, x_coord, y_coord)

            # Filter 4D variables - these are variables of interest for COGs
            # Assuming other vars shouldn't be converted to COGs
            valid_bands = [
                var for var in ds.data_vars
                if len(ds[var].dims) == 4
            ]

            # Get attributes from NetCDF file
            nc_attrs = ds.attrs
            crs = nc_attrs["geospatial_bounds_crs"]
            ds.rio.write_crs(crs, inplace=True)

            # Get bounding box of dataset (in expected "EPSG:4326")
            bbox, geometry = self._get_bbox_and_geometry(ds, x_coord, y_coord, crs)

            # Get temporal bounds from input netCDF
            time_coords_start = pd.to_datetime(time_coords.isel(time=0).values)
            time_coords_end = pd.to_datetime(time_coords.isel(time=-1).values)

            return crs, bbox, geometry, valid_bands, x_coord, y_coord, time_coords, time_coords_start, time_coords_end, leadtime_coords

    def _convert_units(self, ds: xr.Dataset, x_coord: str, y_coord: str):
            # Convert eastings and northings from kilometers to metres (if need to).
            if ds.coords[y_coord].attrs.get("units", None) in ["1000 meter", "km"]: # `1000 meter` is legacy support for `icenet < v0.4.0``
                ds = ds.assign_coords({y_coord: ds.coords[y_coord] * 1000})
            if ds.coords[x_coord].attrs.get("units", None) in ["1000 meter", "km"]:
                ds = ds.assign_coords({x_coord: ds.coords[x_coord] * 1000})
            return ds

    def _get_bbox_and_geometry(self, ds: xr.Dataset, x_coord: str, y_coord: str, crs: str):
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
        flat: bool = False,
        stac_only: bool = False,
        workers: int = 1,
    ):
        self._collection_name = name
        self._forecast_frequency = forecast_frequency
        self._compress_method = "DEFLATE" if compress else "NONE"
        self._flat = flat
        nc_file = Path(nc_file).resolve()
        hemisphere = get_hemisphere(nc_file)

        # Initialise output paths
        self._set_out_paths()

        self._validate_input_options()

        # Get input time delta options to compute forecast times
        leadtime_step, leadtime_unit = parse_forecast_frequency(forecast_frequency)

        catalog = self.catalog

        # Get required coords and metadata from forecast netCDF file
        crs, bbox, geometry, valid_bands, x_coord, y_coord, time_coords, time_coords_start, time_coords_end, leadtime_coords = self.get_forecast_info(nc_file)
        leadtime = len(leadtime_coords)

        # Create (or retrieve) highest level collection (model name) within the catalog
        main_collection = self.get_or_create_collection(
            parent=catalog,
            collection_id=name,
            title=f"Model Collection: {name}",
            description=f"{name} collection",
            bbox=bbox,
            license=self._license,
            temporal_extent=[time_coords_start, time_coords_end],
        )

        if not flat:
            # Create (or retrieve) a hemisphere collection within the main collection
            hemisphere_collection = self.get_or_create_collection(
                parent=main_collection,
                collection_id=hemisphere,
                title=f"Hemisphere Collection: {hemisphere.capitalize()}",
                description=f"{hemisphere.capitalize()} hemisphere collection",
                bbox=bbox,
                temporal_extent=[time_coords_start, time_coords_end],
            )

        ds = xr.open_dataset(nc_file, decode_coords="all")
        for time_idx, time_val in enumerate(time_coords):
            ds_time_slice = ds.sel(time=time_val)

            # The forecast initialisation time (CF Convention: `forecast_reference_time`) is the first forecast
            forecast_reference_time = pd.to_datetime(time_val.values)
            forecast_reference_date = forecast_reference_time.date()
            forecast_reference_time_str = forecast_reference_time.strftime("%Y%m%d_%H%M")
            forecast_reference_time_str_fmt = forecast_reference_time.strftime("%Y-%m-%d %H:%M")
            forecast_end_time = forecast_reference_time + relativedelta(**{leadtime_unit: leadtime - 1})
            forecast_end_time_str_fmt = forecast_end_time.strftime("%Y-%m-%d %H:%M")

            if not flat:
                # Create (or retrieve) a forecast collection within the catalog
                forecast_collection = self.get_or_create_collection(
                    parent=hemisphere_collection,
                    collection_id=f"{forecast_reference_date}",
                    title=f"Forecast Collection: {forecast_reference_date}",
                    description=f"Forecast data for {forecast_reference_date}",
                    bbox=bbox,
                    temporal_extent=[forecast_reference_time, forecast_end_time],
                )

            # Create output dirs
            ncdf_dir = Path(self._netcdf_output_dir / f"{hemisphere}/{forecast_reference_date}")
            ncdf_dir.mkdir(parents=True, exist_ok=True)
            cog_dir = Path(self._cogs_output_dir / f"{hemisphere}/{forecast_reference_date}")
            cog_dir.mkdir(parents=True, exist_ok=True)

            item_id = f"{hemisphere}_forecast_init_{forecast_reference_time_str}"

            # Save the forecast init slice as a netcdf file
            out_nc_file = ncdf_dir / f"{item_id}.nc"

            # Write the netCDF file in addition to the STAC json output
            if not stac_only:
                # with xr.open_dataset(nc_file, decode_coords="all") as ds:
                #     ds_time_slice = ds.sel(time=time_val)
                    self._write_netcdf(ds_time_slice, out_nc_file)

            # Add STAC Item for this netCDF file
            item = Item(
                id=item_id,
                geometry=geometry,
                bbox=bbox,
                datetime=forecast_reference_time,
                properties={
                    "forecast:reference_time": forecast_reference_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "forecast:end_time": forecast_end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "custom:hemisphere": hemisphere,
                },
            )

            # Add projection extension
            ProjectionExtension.add_to(item)
            proj = ProjectionExtension.ext(item)
            proj.code = crs

            # Add netCDF asset to item
            item.add_asset(
                "netcdf",
                Asset(
                    href=str(out_nc_file),
                    media_type=pystac.MediaType.NETCDF,
                    title=f"Forecast -> netCDF from {forecast_reference_time_str_fmt}",
                    description=f"netCDF file container forecast variables for forecast initialised at: {forecast_reference_time_str_fmt}",
                    roles=["data"],
                ),
            )
            if flat:
                main_collection.add_item(item)
            else:
                forecast_collection.add_item(item)

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

            # Process each leadtime
            with ProcessPoolExecutor(max_workers=workers) as executor:
                with tqdm(total=leadtime, desc="COGifying files", leave=True) as pbar:
                    futures = []
                    for i in range(leadtime):
                        future = executor.submit(self._process_leadtime, i, *process_args)
                        future.add_done_callback(lambda _: pbar.update(1))
                        futures.append(future)

                    # Wait for all futures to complete
                    for future in futures:
                        cog_file, assets, pbar_description = future.result()
                        pbar.set_description(pbar_description)
                        for asset in assets:
                            item.add_asset(key=asset["key"], asset=asset["asset"])

        ds.close()

        # Save catalog and collections
        self.save_catalog()

        if not flat:
            logger.warning(
                "Run without `-f`/`--flat` flag, this is not supported for ingestion into pgSTAC database, only stac-browser"
            )

    def _process_leadtime(self, i, forecast_reference_time, leadtime_unit, leadtime_step, ds_time_slice, x_coord, y_coord, crs, cog_dir, stac_only, item, valid_bands, overwrite):
        valid_time = forecast_reference_time + relativedelta(**{leadtime_unit: i*leadtime_step})
        ds_leadtime_slice = ds_time_slice.isel(leadtime=i)

        # Set spatial dimensions for rioxarray
        ds_leadtime_slice.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)

        valid_time_str = valid_time.strftime("%Y%m%d_%H%M")
        valid_time_str_fmt = valid_time.strftime("%Y-%m-%d %H:%M")

        # Add STAC Item for this file
        item_id_cog = f"{item.id}_lead_{valid_time_str}"

        # Define cog/thumbnail output paths
        cog_file = cog_dir / f"{item_id_cog}.tif"
        thumbnail_file = cog_dir / f"{item_id_cog}.jpg"

        # Save variables as one multi-band COG (Cloud Optimized GeoTIFF) & JPG (for thumbnail)
        da_list = []
        band_names = valid_bands
        for var_name in band_names:
            da_variable = ds_leadtime_slice[var_name]
            da_variable.rio.write_crs(crs, inplace=True)
            da_variable.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)
            da_list.append(da_variable)

        if not stac_only:
            # Stack variables as a single dataset
            da_multiband = xr.concat(da_list, dim="band")
            da_multiband = da_multiband.assign_coords(band=("band", band_names))

            if cog_file.exists() and not overwrite:
                pbar_description = f"File already exists, skipping: {cog_file}"
            else:
                pbar_description = f"Saving vars to multi-band COG: {cog_file}"

                self._write_cog(da_multiband, x_coord, y_coord, crs, cog_file)

            # Create thumbnail plot for only the first variable for the first leadtime
            if i == 0:
                if not thumbnail_file.exists() or overwrite:
                    self._create_and_write_thumbnail(da_multiband, thumbnail_file, forecast_reference_time, valid_time)
        else:
            pbar_description = f"Processing STAC: {item_id_cog}"

        assets = []
        # Add COG asset to item
        cog_asset = dict(
            key=valid_time_str_fmt,
            asset=Asset(
                href=str(cog_file),
                media_type=pystac.MediaType.COG,
                title=f"Forecast -> Multi-band COG at {valid_time_str_fmt}",
                description=f"Variables: {', '.join(band_names)}",
                roles=["data"],
                extra_fields={
                    "forecast:bands": [{"name": name} for name in band_names],
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

        return cog_file, assets, pbar_description

    def _write_netcdf(self, ds_time_slice: xr.Dataset, out_nc_file: Path):
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

    def _write_cog(self, da_multiband, x_coord, y_coord, crs, cog_file: Path):
        # Add metadata to extracted variable so `to_raster` includes them in the output GeoTIFF
        da_multiband.rio.write_crs(crs, inplace=True)
        da_multiband.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)
        # da_multiband.rio.to_raster(cog_path, driver="COG", compress=self._compress_method)
        write_cog(cog_file, da_multiband, compress=self._compress_method)

    def _create_and_write_thumbnail(self, da_multiband: xr.DataArray, thumbnail_path, forecast_reference_time, valid_time):
        fig = plt.figure(figsize=(5, 5), dpi=100, constrained_layout=True)
        # da_multiband.sel(band=band_names[0]).plot(cmap='RdBu_r', add_colorbar=True) # type: ignore
        da_multiband.isel(band=0).plot(cmap='RdBu_r', add_colorbar=True) # type: ignore
        plt.axis('off')
        plt.title(f"Init: {forecast_reference_time}\nLeadtime: {valid_time}")
        plt.savefig(thumbnail_path, pad_inches=0, transparent=False)
        plt.close(fig)

    def save_catalog(self):
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
