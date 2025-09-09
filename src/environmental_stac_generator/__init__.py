"""Top-level package for environmental_stac_generator."""

__author__ = """Bryn Noel Ubald"""
__email__ = 'bryald@bas.ac.uk'
__version__ = '0.0.1'

import json
import logging
from logging.config import dictConfig
from pathlib import Path

# Get package root by getting the directory this script is in
package_root = Path(__spec__.origin).resolve().parent
logging_config = package_root / "logging_config.json"

try:
    with open(logging_config, "r") as f:
        config = json.load(f)
    dictConfig(config)
except FileNotFoundError:
    logging.error(f"File `{logging_config}` not found.")
    logging.error("Please ensure that the logging_config.jsonfile is defined.")
