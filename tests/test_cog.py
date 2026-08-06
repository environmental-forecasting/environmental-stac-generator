"""Tests for COG writing defaults."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
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

    with rasterio.open(cog_path) as src:
        assert src.tags(1).get("STATISTICS_MINIMUM") is not None
        assert src.tags(1).get("STATISTICS_MAXIMUM") is not None
        overviews = src.overviews(1)
        assert len(overviews) == 5
        assert overviews == [2, 4, 8, 16, 32]


def test_write_cog_uses_provided_band_statistics(tmp_path: Path) -> None:
    cog_path = tmp_path / "demo.tif"
    stats = [
        {
            "STATISTICS_MINIMUM": -1.5,
            "STATISTICS_MAXIMUM": 2.5,
            "STATISTICS_MEAN": 0.25,
            "STATISTICS_STDDEV": 0.75,
            "STATISTICS_VALID_PERCENT": 99.0,
        }
    ]
    write_cog(cog_path, _small_da(), band_statistics=stats)

    with rasterio.open(cog_path) as src:
        tags = src.tags(1)
        assert float(tags["STATISTICS_MINIMUM"]) == -1.5
        assert float(tags["STATISTICS_MAXIMUM"]) == 2.5
        assert float(tags["STATISTICS_MEAN"]) == 0.25


def test_write_cog_external_overviews_calls_gdaladdo(tmp_path: Path) -> None:
    cog_path = tmp_path / "demo.tif"

    def _fake_gdaladdo(cmd, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        path_arg = next(arg for arg in cmd if arg.endswith(".tif"))
        Path(path_arg).with_suffix(".tif.ovr").write_bytes(b"ovr")
        return None

    with patch(
        "environmental_stac_generator.cog.subprocess.run",
        side_effect=_fake_gdaladdo,
    ) as gdaladdo:
        write_cog(cog_path, _small_da(), external_overviews=True)
        gdaladdo.assert_called_once()
        cmd = gdaladdo.call_args.args[0]
        assert cmd[0] == "gdaladdo"
        assert "-r" in cmd
        assert "bilinear" in cmd
        assert str(cog_path) in cmd
        assert "32" in cmd

    assert cog_path.is_file()
    assert cog_path.with_suffix(".tif.ovr").is_file()
