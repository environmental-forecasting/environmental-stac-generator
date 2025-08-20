from pathlib import Path
from unittest.mock import mock_open

import numpy as np
import orjson
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from environmental_stac_generator.stac.generator import STACGenerator
from environmental_stac_generator.stac.utils import ConfigMismatchError
from shapely.geometry import box, mapping


@pytest.fixture
def stac():
    """
    Create instance of STACGenerator before each test.
    """
    return STACGenerator()


@pytest.fixture
def sample_sic_ds():
    """
    Mimicking IceNet sea ice concentration file
    """
    return xr.Dataset(
        {
            "sic_mean": (
                ("time", "yc", "xc", "leadtime"),
                np.random.rand(1, 432, 432, 93),
            ),
            "sic_stddev": (
                ("time", "yc", "xc", "leadtime"),
                np.random.rand(1, 432, 432, 93),
            ),
        },
        coords={
            "time": pd.date_range("2025-01-01", periods=1),
            "leadtime": np.random.rand((93)),
            "xc": np.random.rand((432)),
            "yc": np.random.rand((432)),
        },
        attrs={"geospatial_bounds_crs": "EPSG:6931"},
    )


def test_set_out_paths(stac, tmp_path):
    """
    Test that `STACGenerator._set_out_paths` correctly initialises output directories.

    Verifies that the method sets netcdf_output_dir, cogs_output_dir,
    and config_output_path based on data_path and collection name.

    Args:
        stac: STAC generator instance with test configuration
        tmp_path: pytest temporary directory fixture for output paths
    """
    stac.data_path = tmp_path
    stac._collection_name = "test_collection"

    stac._set_out_paths()

    assert stac._netcdf_output_dir == tmp_path / "netcdf" / "test_collection"
    assert stac._cogs_output_dir == tmp_path / "cogs" / "test_collection"
    assert stac._config_output_path == tmp_path / "config.json"


def test_store_config_new_file(stac, tmp_path, mocker):
    """
    Test that `STACGenerator._store_config` creates a new config file when it doesn't exist.

    Verifies that the method properly writes configuration data to a new JSON file
    when the target file does not already exist. Uses mocks for
    file system operations and checks that the correct content is written.

    Args:
        stac: STAC generator instance with test configuration
        tmp_path: pytest temporary directory fixture for output paths
        mocker: pytest-mocker fixture for mocking file operations
    """
    # Note: I'm storing one collection in one catalog file for simplicity
    # so, catalog and collection name are the same.
    stac._collection_name = "test_collection"
    stac._config_output_path = tmp_path / "config.json"
    config = {stac._collection_name: {"forecast_frequency": "1days"}}

    # Mock writing to a config file that does not exist
    m = mocker.patch("builtins.open", mock_open())
    mocker.patch("pathlib.Path.exists", return_value=False)

    stac._store_config(config)

    handle = m()
    handle.write.assert_called_once()
    written = handle.write.call_args[0][0]
    assert orjson.loads(written)[stac._collection_name]["forecast_frequency"] == "1days"


def test_store_config_existing_mismatch(stac, tmp_path, mocker):
    """
    Test that `STACGenerator._store_config` raises `ConfigMismatchError` when existing config doesn't match.

    Verifies that the method correctly identifies a mismatch between an existing configuration file
    and the new configuration being written. This scenario occurs when the collection name or
    configuration content in the existing file differs from what is attempted to be stored.

    Args:
        stac: STAC generator instance with test configuration
        tmp_path: pytest temporary directory fixture for output paths
        mocker: pytest-mocker fixture for mocking file operations

    Raises:
        ConfigMismatchError: Expected when attempting to overwrite an existing config that doesn't match.
    """
    stac._collection_name = "test_collection"
    stac._config_output_path = tmp_path / "config.json"
    config = {stac._collection_name: {"forecast_frequency": "1days"}}

    # Mocking an existing config file being run with forecast freq of 2 days
    existing_config = {stac._collection_name: {"forecast_frequency": "2days"}}
    mock_open_read = mock_open(read_data=orjson.dumps(existing_config))

    mocker.patch("builtins.open", mock_open_read)
    mocker.patch("pathlib.Path.exists", return_value=True)

    # Since the existing config and the new one don't match,
    # it should raise an error upon trying to validation
    # and store the updated config
    with pytest.raises(ConfigMismatchError):
        stac._store_config(config)


def test_convert_units_km(stac):
    """
    Test that `STACGenerator._convert_units` correctly converts coordinate units from kilometers to meters.

    Verifies that the method properly transforms `x` (in "km") and `y`
    (in "1000 meter") coordinates into their equivalent values in meters. The
    resulting dataset should reflect the converted values, with x and y scaled
    by a factor of 1000.

    Args:
        stac: STAC generator instance with unit conversion functionality

    Asserts:
        - `x` coordinate values are converted from km to m (multiplied by 1000)
        - `y` coordinate values are converted from "1000 meter" to m (treated as meters)
    """
    ds = xr.Dataset(
        coords={
            "x": ("x", [1, 2, 3], {"units": "km"}),
            "y": ("y", [4, 5, 6], {"units": "1000 meter"}),
        }
    )
    new_ds = stac._convert_units(ds, "x", "y")

    assert all(new_ds["x"].values == [1000, 2000, 3000])
    assert all(new_ds["y"].values == [4000, 5000, 6000])


def test_get_bbox_and_geometry_epsg4326(stac):
    """
    Test that `STACGenerator._get_bbox_and_geometry` correctly computes the bounding box and geometry using EPSG:4326 (WGS84) coordinates.

    Verifies that when a dataset is provided with longitude (`lon`) and latitude (`lat`)
    in degrees_east and degrees_north, respectively, the method returns a proper bounding box
    and GeoJSON geometry representing the geographic extent of the data.

    Args:
        stac: STAC generator instance

    Asserts:
        - The returned `bbox` matches expected [min_lon, min_lat, max_lon, max_lat]
        - The returned `geometry` is a GeoJSON representation of the bounding box
    """
    x = np.array([10, 20, 30])
    y = np.array([40, 50, 60])
    ds = xr.Dataset(coords={"lon": ("lon", x), "lat": ("lat", y)})

    # Add coordinate data
    ds["lon"].attrs["units"] = "degrees_east"
    ds["lat"].attrs["units"] = "degrees_north"

    x_coord = "lon"
    y_coord = "lat"
    crs = "EPSG:4326"

    returned_bbox, geometry = stac._get_bbox_and_geometry(ds, x_coord, y_coord, crs)

    expected_bbox = [10.0, 40.0, 30.0, 60.0]
    expected_geometry = mapping(box(*expected_bbox))

    assert returned_bbox == expected_bbox
    assert geometry == expected_geometry


def test_get_bbox_and_geometry_with_projection(stac, mocker):
    """
    Test that `STACGenerator._get_bbox_and_geometry` correctly computes the bounding box and geometry using a projected coordinate system (EPSG:6931).

    Verifies that when a dataset is provided with xc/yc coordinates in a projected system,
    such as EPSG:6931, the method uses the `proj_to_geo` utility to convert these into WGS84
    and returns the correct geographic bounding box and GeoJSON geometry.

    Args:
        stac: STAC generator instance
        mocker: pytest-mocker fixture for mocking projection conversion

    Asserts:
        - The returned `bbox` matches the expected geographic bounds after projection conversion
        - The returned `geometry` is a GeoJSON representation of the projected bounding box
    """
    # Create dataset with EPSG:6931 projected coordinates (meters)
    # Though, technically, icenet stores in km instead of m.
    # Using EPSG:6931 proj extents for test, Ref: https://epsg.io/6931
    xc = np.array([-8918256.31, 8918256.31])
    yc = np.array([-9009964.76, 9009964.76])
    ds = xr.Dataset(coords={"xc": ("xc", xc), "yc": ("yc", yc)})

    x_coord = "xc"
    y_coord = "yc"
    crs = "EPSG:6931"  # Projection used for IceNet northern hemisphere

    # from environmental_stac_generator.utils import proj_to_geo
    # bbox = [xc.min(), yc.min(), xc.max(), yc.max()]
    # test = proj_to_geo(bbox, src_crs=crs)
    # print("test bbox:", test)

    # Expected output from proj_to_geo
    expected_bbox = (-180.0, -78.49911570449875, 180.0, 90.0)

    # Mock proj_to_geo
    mocker.patch(
        "environmental_stac_generator.utils.proj_to_geo", return_value=expected_bbox
    )

    returned_bbox, geometry = stac._get_bbox_and_geometry(ds, x_coord, y_coord, crs)

    np.testing.assert_allclose(returned_bbox, expected_bbox, atol=1e-5)
    assert geometry == mapping(box(*expected_bbox))


def test_get_forecast_info(stac, sample_sic_ds, mocker):
    """
    Test that `STACGenerator.get_forecast_info` returns the correct metadata from a forecast dataset.

    This test verifies that when given a path to a NetCDF file representing a sea ice forecast,
    the method correctly parses and returns a tuple containing key metadata such as:
        - Coordinate reference system (CRS)
        - Bounding box
        - GeoJSON geometry
        - Valid bands
        - Time-related information

    Args:
        stac: STAC generator instance
        sample_sic_ds: Mock dataset fixture for testing forecast data
        mocker: pytest-mocker fixture for mocking `xarray.open_dataset`

    Asserts:
        - The returned object is a tuple
        - `crs` is set to "EPSG:6931"
        - Expected bands are included in the `valid_bands` list
    """
    mocker.patch("xarray.open_dataset", return_value=sample_sic_ds)

    result = stac.get_forecast_info(Path("/test_forecast_north.nc"))

    assert isinstance(result, tuple)

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
    ) = result

    assert crs == "EPSG:6931", "`crs` should be 'EPSG:6931'"
    for band in ["sic_mean", "sic_stddev"]:
        assert band in valid_bands, f"'{band}' should be in `valid_bands`"
