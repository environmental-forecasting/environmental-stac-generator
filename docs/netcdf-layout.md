---
icon: lucide/database
---

# netCDF layout

`envstacgen preprocess` expects IceNet-style forecast netCDFs. This page describes the required and optional pieces of the input file.

Catalogue vs collection: the root **catalogue** is always `data/stac/catalog.json`. This was an architectural decision based on STAC browser support of stac-fastapi only of one main catalog - json based approach doesn't seem to have this limitation (this was the initial way this codebase used to work). The `-n` / `--name` flag sets the **collection** id under that catalogue (see [CLI](cli.md)).

## Structure at a glance

The input netCDF file must have a certain structure for it to work seamlessly. As a simplified view (names in parentheses are accepted aliases):

```text
forecast.nc
├── dimensions / coordinates
│   ├── time (or forecast_time)                 [required]  init times for STAC Items
│   ├── leadtime (or lead_time)                 [required]  one COG per lead
│   ├── yc (or y, lat, latitude)                [required]  spatial Y
│   └── xc (or x, lon, longitude)               [required]  spatial X
│
├── data variables
│   ├── <any_4d_var> (time, yc, xc, leadtime)   [required*] each 4D var becomes a COG band
│   ├── forecast_date (leadtime)                [optional]  valid times (not a band)
│   └── <other non-4D vars>                     [optional]  kept in sliced netCDF only
│
└── global attributes
    ├── geospatial_bounds_crs                   [required]  e.g. EPSG:6931
    ├── geospatial_lat_min                      [optional]  Collection hemisphere
    └── <other JSON-able attrs>                 [optional]  STAC Item properties
```

\* At least one data variable with exactly four dimensions is required for COGs (`--stac-only` can skip raster output). Names are arbitrary (IceNet examples: `sic_mean`, `sic_stddev`). Typical shape: `(time, yc, xc, leadtime)`.

### Extra notes

- If X/Y `units` are `km` or `1000 meter` (which is the case with IceNet forecast outputs), values are scaled to metres before writing COGs.
- Bounding boxes come from the spatial coordinates (warped to WGS84 when needed); a separate `geospatial_bounds` attribute is not required.
- `geospatial_lat_min`: north if `0` to `90`, south if `-90` to below `0`; otherwise hemisphere is left unset (with a warning if the attr is missing).
- Lead `units` / `long_name`, or dataset `time_coverage_resolution`, help interpret numeric leads when `forecast_date` is absent.
- Missing spatial or lead coordinates raise a `ValueError` (for example `Spatial coordinates not found in dataset`).

## Valid times

In forecasting, **init time** is when the forecast was produced; **valid time** is the real-world date/time that lead is predicting for (for example init 1 January, lead 3 days, valid time 4 January).

Each lead therefore needs a calendar valid time. Preprocess attempts to derive it as follows:

1. Prefer a `forecast_date` coordinate along the lead axis (common in IceNet files).
2. If that is missing, use the `leadtime` / `lead_time` values when they are already datetimes or timedeltas.
3. Otherwise treat the lead values as numeric offsets from the forecast init time. Units are taken from the lead coordinate (or `time_coverage_resolution` if needed); if nothing is set, **days** are assumed and a warning is logged.

Valid times must increase with lead. Setting proper CF `units` on `forecast_date` or `leadtime` avoids the default-days fallback.
