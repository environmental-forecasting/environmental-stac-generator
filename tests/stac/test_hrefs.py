from pathlib import Path

from pystac import Asset, Catalog, Collection, Item, Extent, SpatialExtent, TemporalExtent
from datetime import datetime, timezone

from environmental_stac_generator.stac.utils import (
    apply_file_server_url,
    rewrite_catalog_asset_hrefs,
    to_cwd_relative_href,
)


def test_to_cwd_relative_href_from_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asset_path = tmp_path / "data" / "cogs" / "thumb.jpg"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"x")

    assert to_cwd_relative_href(str(asset_path)) == "data/cogs/thumb.jpg"


def test_to_cwd_relative_href_leaves_http():
    url = "http://example.com/data/cogs/thumb.jpg"
    assert to_cwd_relative_href(url) == url


def test_apply_file_server_url_portable_path():
    href = apply_file_server_url(
        "data/cogs/thumb.jpg",
        "http://127.0.0.1:8001",
    )
    assert href == "http://127.0.0.1:8001/data/cogs/thumb.jpg"


def test_apply_file_server_url_leaves_http():
    url = "https://files.example/data/cogs/thumb.jpg"
    assert apply_file_server_url(url, "http://127.0.0.1:8001") == url


def test_rewrite_catalog_asset_hrefs():
    catalog = Catalog(id="root", description="root")
    collection = Collection(
        id="col",
        description="c",
        extent=Extent(
            spatial=SpatialExtent([[-180, -90, 180, 90]]),
            temporal=TemporalExtent(
                [[datetime(2026, 1, 1, tzinfo=timezone.utc), None]]
            ),
        ),
    )
    item = Item(
        id="item",
        geometry=None,
        bbox=[-180, -90, 180, 90],
        datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
        properties={},
    )
    collection.add_asset(
        "thumbnail",
        Asset(href="data/cogs/thumb.jpg", media_type="image/jpeg"),
    )
    item.add_asset(
        "data",
        Asset(href="data/cogs/file.tif", media_type="image/tiff"),
    )
    catalog.add_child(collection)
    collection.add_item(item)

    rewrite_catalog_asset_hrefs(catalog, "http://files.example")

    assert (
        collection.assets["thumbnail"].href
        == "http://files.example/data/cogs/thumb.jpg"
    )
    assert item.assets["data"].href == "http://files.example/data/cogs/file.tif"
