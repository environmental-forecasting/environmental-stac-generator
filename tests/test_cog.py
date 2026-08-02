"""Tests for COG writing defaults."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from environmental_stac_generator.cog import write_cog


def _small_da(size: int = 256) -> xr.DataArray:
    # Large enough for default overview_level without 1x1 overflow.
    data = np.arange(size * size, dtype=np.float32).reshape(size, size)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.linspace(1, 0, size), "x": np.linspace(0, 1, size)},
    )
    return da.rio.write_crs("EPSG:4326")


def test_write_cog_default_has_internal_overviews_only(tmp_path: Path) -> None:
    cog_path = tmp_path / "demo.tif"
    with patch("environmental_stac_generator.cog.subprocess.run") as gdaladdo:
        write_cog(cog_path, _small_da())
        gdaladdo.assert_not_called()

    assert cog_path.is_file()
    assert not cog_path.with_suffix(".tif.ovr").exists()


def test_write_cog_external_overviews_calls_gdaladdo(tmp_path: Path) -> None:
    cog_path = tmp_path / "demo.tif"

    def _fake_gdaladdo(cmd, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        Path(cmd[3]).with_suffix(".tif.ovr").write_bytes(b"ovr")
        return None

    with patch(
        "environmental_stac_generator.cog.subprocess.run",
        side_effect=_fake_gdaladdo,
    ) as gdaladdo:
        write_cog(cog_path, _small_da(), external_overviews=True)
        gdaladdo.assert_called_once()
        assert gdaladdo.call_args.args[0][0] == "gdaladdo"

    assert cog_path.is_file()
    assert cog_path.with_suffix(".tif.ovr").is_file()
