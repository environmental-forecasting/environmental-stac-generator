from unittest.mock import mock_open

import orjson
import pytest
import xarray as xr
from environmental_stac_generator.stac.generator import STACGenerator
from environmental_stac_generator.stac.utils import ConfigMismatchError


@pytest.fixture
def stac():
    """
    Create instance of STACGenerator before each test.
    """
    return STACGenerator()


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
