import datetime as dt
from abc import abstractmethod
from pathlib import Path

import pystac
from pystac import Catalog, Collection, Item
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension


class BaseSTAC:
    def __init__(self, stac_catalog_path: str | Path = None):
        """
        Initialises the BaseSTAC class with a path to an existing STAC catalog.
        Args:
            stac_catalog_path (str|Path): The path to the STAC catalog.
                If the path does not exist, a new catalog will be created at this location.
        """

        if stac_catalog_path is None:
            self._stac_catalog_path = Path("data/stac/") / "catalog.json"
        else:
            self._stac_catalog_path = Path(self._stac_catalog) / "catalog.json"
        self.initialise()

    def initialise(
        self, id: str, description: str, title: str, describe: bool = True
    ) -> Catalog:
        """Initialises a STAC catalog or loads an existing one if it exists.

        This method either creates a new STAC catalog with the provided metadata
        or loads an existing one from the specified path. If `describe` is set to
        True, it will print a description of the catalog after initialization.

        Args:
            id: The ID for the catalog.
            description: A detailed description of the catalog.
            title: The title of the catalog.
            describe (optional): Whether to print a description of thecatalog.
                Defaults to False.
        Returns:
            Catalog: The initialised or loaded STAC catalog.
        """
        stac_catalog_path = self._stac_catalog_path
        if not stac_catalog_path.exists():
            catalog = Catalog(
                id=id,
                description=description,
                title=title,
            )
        else:
            catalog = Catalog.from_file(stac_catalog_path)

        self._stac_catalog = catalog

        if describe:
            print(catalog.describe())

    @abstractmethod
    def create_collection(self) -> None:
        pass

    @abstractmethod
    def add_item_to_collection(self) -> None:
        pass

    def save_catalog(self, stac_output_dir):
        self._stac_catalog.normalize_hrefs(str(stac_output_dir))
        self._stac_catalog.save()

    @property
    def get_stac_catalog(self):
        """Get the current STAC catalog.

        Provides access to the internal STAC catalog used for data processing.
        """
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


class IceNetSTAC(BaseSTAC):
    """Initialises an IceNetSTAC object with customisable metadata.

    This class inherits from BaseSTAC and initialises a Catalog with specific ID,
    description, and title using super().initialise().

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    Returns:
        Catalog: A Catalog instance initialized with the provided metadata.
    Example:
        >>> icenet_stac = IceNetSTAC()
        >>> print(icenet_stac.initialise().id)
        forecast-data
    """

    def __init__(self, *args, **kwargs):
        """Initialise the Catalog instance with default metadata.
        Example:
            >>> icenet_stac = IceNetSTAC()
        """
        super().__init__(*args, **kwargs)
        return

    def initialise(
        self,
        id: str = "forecast-data",
        description: str = "Catalog of IceNet Forecast Data",
        title: str = "IceNetForecast STAC Catalog",
    ) -> Catalog:
        """Initialise or update the Catalog instance with custom metadata.
        Args:
            id: Catalog ID, default is 'forecast-data'.
            description: Catalog description, default is 'Catalog of IceNet Forecast Data'.
            title: Catalog title, default is 'IceNetForecast STAC Catalog'.
        Returns:
            Catalog: The updated or initializsed Catalog instance.
        Example:
            >>> icenet_stac = IceNetSTAC()
            >>> custom_id = "new-forecast-data"
            >>> new_catalog = icenet_stac.initialise(id=custom_id)
            >>> print(new_catalog.id)
            new-forecast-data
        """
        kwargs = {
            "id": id,
            "description": description,
            "title": title,
        }
        super().initialise(**kwargs)

    def create_collection(
        self,
        collection_id: str,
        description: str,
        spatial_extent: list,
        temporal_extent: list,
    ) -> None:
        """Create or retrieve an existing STAC Collection based on the provided parameters.

        This method first attempts to find an existing `Collection` in the catalog with the given ID.
        If found, it sets this as the active collection. If not found, it creates a new `Collection`
        and adds it to the catalog. The new `Collection` is then set as the active one.
        Args:
            collection_id: Unique identifier for the Collection.
            description: Human-readable description of the Collection.
            spatial_extent: Spatial extent of the Collection, as per GeoJSON format.
            temporal_extent: Temporal extent of the Collection, in ISO 8601 format.
        """
        # Find the first existing collection for this forecast date
        # If it doesn't exist yet, create a new one, else, skip since it already exists
        collection = next(
            (
                coll
                for coll in self._stac_catalog.get_children()
                if coll.id == collection_id
            ),
            None,
        )
        if not collection:
            collection = Collection(
                id=collection_id,
                description=description,
                extent=pystac.Extent(
                    pystac.SpatialExtent(spatial_extent),
                    pystac.TemporalExtent(temporal_extent),
                ),
            )
            self.collection = collection
            self._stac_catalog.add_child(collection)

    def add_item_to_collection(
        self,
        hemisphere: str,
        geo_bounds: (tuple[float, float, float, float]),
        forecast_start_time: dt.datetime,
        forecast_start_date: dt.date,
        leadtime: int,
        shape: (tuple[int, int]),
        cog_file: Path,
        collection_id: str,
    ) -> None:
        """Add a STAC Item to the specified collection for a given COG file.

        This method creates a new STAC Item with the provided metadata, adds EO and Projection
        extensions, includes the COG asset, and then adds the item to the specified collection.
        Args:
            hemisphere: Hemisphere of the data.
            geo_bounds : Geo-spatial bounds (west, south, east, north) of the data.
            forecast_start_time: Forecast start time.
            forecast_start_date: Forecast start date.
            leadtime: Lead time in days.
            shape: Shape of the COG file as a tuple (height, width).
            cog_file: Path to the COG file.
            collection_id: ID of the collection where the item will be added.
        """
        # Add STAC Item for this file
        item = Item(
            id=f"{hemisphere}-leadtime-{leadtime}",
            geometry=None,
            bbox=geo_bounds,
            datetime=forecast_start_time + dt.timedelta(days=leadtime),
            properties={
                "forecast_start_date": str(forecast_start_date),
                "hemisphere": hemisphere,
                "leadtime": leadtime,
            },
        )

        # Add EO and Projection extensions
        eo_ext = EOExtension.ext(item, add_if_missing=True)

        projection_ext = ProjectionExtension.ext(item, add_if_missing=True)
        projection_ext.epsg = 3857
        projection_ext.shape = shape
        projection_ext.bbox = geo_bounds

        # Add COG asset
        item.add_asset(
            "geotiff",
            pystac.Asset(
                href=str(cog_file),
                media_type=pystac.MediaType.COG,
                title=f"GeoTIFF for {hemisphere} with leadtime {leadtime}",
            ),
        )

        # Add item to collection
        self.get_collection(collection_id).add_item(item)
