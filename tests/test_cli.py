from unittest.mock import patch

from typer.testing import CliRunner

from environmental_stac_generator.cli import app

runner = CliRunner()


def test_preprocess_default_enables_compression():
    with patch("environmental_stac_generator.cli.preprocess_main") as mock_main:
        result = runner.invoke(app, ["preprocess", "dummy.nc"])

    assert result.exit_code == 0
    assert mock_main.call_args.kwargs["compress"] is True


def test_preprocess_no_compress_disables_compression():
    with patch("environmental_stac_generator.cli.preprocess_main") as mock_main:
        result = runner.invoke(
            app, ["preprocess", "dummy.nc", "--no-compress"]
        )

    assert result.exit_code == 0
    assert mock_main.call_args.kwargs["compress"] is False


def test_preprocess_no_compress_short_flag():
    with patch("environmental_stac_generator.cli.preprocess_main") as mock_main:
        result = runner.invoke(app, ["preprocess", "dummy.nc", "-c"])

    assert result.exit_code == 0
    assert mock_main.call_args.kwargs["compress"] is False
