from datetime import datetime, timezone

from pystac import (
    Asset,
    Collection,
    Extent,
    Item,
    MediaType,
    SpatialExtent,
    TemporalExtent,
)

from environmental_stac_generator.stac.utils import refresh_collection_summaries

_BBOX = [-10.0, 50.0, 10.0, 60.0]
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [-10.0, 50.0],
            [10.0, 50.0],
            [10.0, 60.0],
            [-10.0, 60.0],
            [-10.0, 50.0],
        ]
    ],
}


def _make_collection() -> Collection:
    return Collection(
        id="demo",
        description="demo collection",
        extent=Extent(
            SpatialExtent([_BBOX]),
            TemporalExtent(
                [
                    [
                        datetime(2026, 1, 1, tzinfo=timezone.utc),
                        datetime(2026, 1, 3, tzinfo=timezone.utc),
                    ]
                ]
            ),
        ),
    )


def _add_forecast_item(
    collection: Collection,
    *,
    item_id: str,
    day: int,
    reference_time: str,
    leadtime_length: int = 93,
) -> None:
    item = Item(
        id=item_id,
        geometry=_GEOMETRY,
        bbox=_BBOX,
        datetime=datetime(2026, 1, day, tzinfo=timezone.utc),
        properties={
            "forecast:reference_time": reference_time,
            "forecast:leadtime_length": leadtime_length,
        },
    )
    item.add_asset(
        "cog",
        Asset(
            href=f"data/cogs/demo/{reference_time}.tif",
            media_type=MediaType.COG,
            roles=["data"],
            extra_fields={
                "forecast:bands": [
                    {"name": "sic", "index": 1},
                    {"name": "sit", "index": 2},
                ]
            },
        ),
    )
    collection.add_item(item)


def test_refresh_collection_summaries_empty_collection():
    collection = _make_collection()

    refresh_collection_summaries(collection)

    assert collection.summaries.to_dict() == {}


def test_refresh_collection_summaries_from_items():
    collection = _make_collection()

    for day, ref in (
        (1, "2026-01-01T00:00:00Z"),
        (2, "2026-01-02T00:00:00Z"),
        (3, "2026-01-01T00:00:00Z"),  # duplicate init
    ):
        _add_forecast_item(
            collection,
            item_id=f"item-{day}",
            day=day,
            reference_time=ref,
            leadtime_length=93,
        )

    refresh_collection_summaries(collection)

    summaries = collection.summaries.to_dict()
    assert summaries["forecast:reference_time"] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]
    assert summaries["forecast:leadtime_length"] == [93]
    assert summaries["forecast:variable"] == ["sic", "sit"]
