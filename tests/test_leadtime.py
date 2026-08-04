from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from environmental_stac_generator.utils import (
    forecast_frequency_from_valid_times,
    infer_lead_unit,
    resolve_valid_times,
)


def test_infer_lead_unit_from_attrs():
    lead = xr.DataArray([0, 6, 12], dims=["leadtime"], attrs={"units": "hours"})
    assert infer_lead_unit(lead) == "hours"


def test_infer_lead_unit_from_resolution():
    lead = xr.DataArray([1, 2, 3], dims=["leadtime"])
    ds = xr.Dataset({"x": 1}, attrs={"time_coverage_resolution": "P1D"})
    assert infer_lead_unit(lead, ds) == "days"


def test_resolve_valid_times_from_numeric_leadtime():
    ref = datetime(2026, 7, 15)
    lead = xr.DataArray(
        np.array([1, 2, 3], dtype="int64"),
        dims=["leadtime"],
        attrs={"units": "days"},
    )
    times = resolve_valid_times(ref, lead)
    assert times[0].date() == datetime(2026, 7, 16).date()
    assert times[-1].date() == datetime(2026, 7, 18).date()
    assert forecast_frequency_from_valid_times(times) == "1days"


def test_resolve_valid_times_from_forecast_date():
    ref = datetime(2026, 7, 15)
    lead = xr.DataArray(np.arange(1, 4), dims=["leadtime"])
    fd = xr.DataArray(
        pd.date_range("2026-07-16", periods=3, freq="D"),
        dims=["leadtime"],
    )
    ds = xr.Dataset({"forecast_date": fd})
    times = resolve_valid_times(ref, lead, ds)
    assert [t.date() for t in times] == [
        datetime(2026, 7, 16).date(),
        datetime(2026, 7, 17).date(),
        datetime(2026, 7, 18).date(),
    ]


def test_forecast_frequency_six_hours():
    times = list(pd.date_range("2026-01-01", periods=5, freq="6h"))
    assert forecast_frequency_from_valid_times(times) == "6hours"
