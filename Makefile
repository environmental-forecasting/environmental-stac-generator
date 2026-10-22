.PHONY: docs-install docs docs-build

audit: ## Check locked dependencies for known vulnerabilities
	uv audit --preview-features audit-command

docs-install:
	uv sync --group docs --no-install-project

docs:
	uv run --group docs zensical serve

docs-build:
	uv run --group docs zensical build

test: ## Run the test suite with coverage
	uv run pytest

lint: ## Check code style and static rules
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .