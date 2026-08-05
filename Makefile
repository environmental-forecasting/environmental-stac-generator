.PHONY: docs-install docs docs-build

docs-install:
	uv sync --group docs --no-install-project

docs:
	uv run --group docs zensical serve

docs-build:
	uv run --group docs zensical build
