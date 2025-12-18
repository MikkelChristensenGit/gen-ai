.PHONY: setup lint test fmt qdrant-up qdrant-down

setup:
	uv sync --dev

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

test:
	uv run pytest
