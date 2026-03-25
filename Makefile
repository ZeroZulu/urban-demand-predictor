.PHONY: setup ingest train serve test lint clean help

## Show this help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

## Spin up all containers (DB + MLflow + API + Dashboard)
setup:
	docker-compose up -d --build
	@echo ""
	@echo "⏳ Waiting for Postgres to be ready..."
	@sleep 8
	@echo ""
	@echo "✅ All services running:"
	@echo "   API docs  → http://localhost:8000/docs"
	@echo "   Dashboard → http://localhost:8501"
	@echo "   MLflow    → http://localhost:5000"

## Download raw data and load into PostgreSQL
ingest:
	python -m src.ingest.taxi
	python -m src.ingest.weather
	python -m src.ingest.events
	python -m src.ingest.economic
	@echo "✅ All data loaded into PostgreSQL"

## Refresh the materialized view after ingestion
refresh-view:
	psql $${DATABASE_URL} -c "REFRESH MATERIALIZED VIEW ml_features;"

## Train all models and log to MLflow
train:
	python -m src.models.trainer
	@echo "✅ Training complete — view runs at http://localhost:5000"

## Restart API container to pick up newly trained model
serve:
	docker-compose restart api
	@echo "✅ API restarted with latest model"

## Run full test suite
test:
	pytest tests/ -v --tb=short

## Lint all Python files
lint:
	flake8 src/ api/ dashboard/ tests/ --max-line-length 100
	black --check src/ api/ dashboard/ tests/

## Format code in place
fmt:
	black src/ api/ dashboard/ tests/
	isort src/ api/ dashboard/ tests/

## Generate drift monitoring report
monitor:
	python -m monitoring.drift_report
	@echo "✅ Drift report saved to outputs/drift_reports/"

## Tear down all containers and volumes
clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Environment cleaned"
