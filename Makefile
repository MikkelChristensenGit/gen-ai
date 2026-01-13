.PHONY: setup lint test fmt qdrant-up qdrant-down

setup:
	uv sync --dev

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

test:
	uv run pytest

qdrant-up:
	@docker start qdrant 2>/dev/null || \
	docker run -d --name qdrant \
		-p 6333:6333 -p 6334:6334 \
		-v qdrant_storage:/qdrant/storage \
		qdrant/qdrant:latest

qdrant-down:
	docker stop qdrant
