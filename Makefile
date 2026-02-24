.PHONY: dev lint test typecheck

dev:
	docker compose -f infra/dev/docker-compose.yml up

lint:
	ruff check services/orchestrator

test:
	pytest services/orchestrator/tests

typecheck:
	mypy services/orchestrator/app
