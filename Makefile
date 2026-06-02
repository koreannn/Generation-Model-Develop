set-dev:
	uv sync --extra dev

set-test:
	uv run pytest tests/

set-style:
	uv run ruff check --fix .
	uv run ruff format .