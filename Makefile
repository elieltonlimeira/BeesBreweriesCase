# ---------------------------------------------------------------------------
# BeesBreweriesCase — Makefile
# ---------------------------------------------------------------------------
# Targets:
#   up            Start all Docker services (detached)
#   down          Stop and remove containers (keeps volumes/data)
#   down-v        Stop and remove containers + volumes (DELETES ALL DATA)
#   build         Build / rebuild the Airflow image
#   logs          Follow logs for all services (ctrl+c to stop)
#   test          Run the full pytest suite (unit + integration)
#   test-unit     Run only unit tests (fast, no Docker needed)
#   lint          Check style with ruff
#   format        Auto-fix style issues with ruff
#   clean         Remove __pycache__ dirs and coverage artefacts

PYTHON   ?= python
PYTEST   ?= $(PYTHON) -m pytest
RUFF     ?= $(PYTHON) -m ruff

.PHONY: up down down-v build logs test test-unit lint format clean

# ── Docker ─────────────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

down-v:
	docker compose down -v

build:
	docker compose build airflow-webserver airflow-scheduler

logs:
	docker compose logs -f

# ── Tests ───────────────────────────────────────────────────────────────────

test:
	$(PYTEST) tests/ -v --tb=short

test-unit:
	$(PYTEST) tests/unit/ -v --tb=short

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	$(RUFF) check src/ tests/ dags/

format:
	$(RUFF) check --fix src/ tests/ dags/
	$(RUFF) format src/ tests/ dags/

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage
	rm -rf .pytest_cache htmlcov
