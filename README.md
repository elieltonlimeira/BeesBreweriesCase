# BEES Breweries Data Pipeline

End-to-end data pipeline that fetches brewery data from the [Open Brewery DB API](https://www.openbrewerydb.org/), stores raw data in a **Bronze** layer, transforms and partitions it into a **Silver** layer, and aggregates it into a **Gold** layer. Orchestrated by Apache Airflow and deployed via Docker Compose with MinIO as S3-compatible local storage.

---

## Architecture

```
Open Brewery DB API (~9,383 records · 47 pages × 200 records)
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│               Airflow DAG: brewery_data_pipeline          │
│  Schedule: 0 6 * * *  ·  catchup=False  ·  max_runs=1   │
│                                                           │
│  fetch_meta ──► fetch_bronze_pages[0..46]                 │
│                         │                                 │
│                 validate_bronze                            │
│                         │                                 │
│              transform_silver (spark-submit)              │
│                         │                                 │
│                 validate_silver                            │
│                         │                                 │
│              aggregate_gold   (spark-submit)              │
│                         │                                 │
│                 validate_gold                             │
└──────────────────────────────────────────────────────────┘
           │                    │                    │
    MinIO bronze          MinIO silver          MinIO gold
  raw/dt=YYYY-MM-DD/    breweries/           brewery_counts/
  page=NNN.json         country=.../         dt=YYYY-MM-DD/
                        state_province=.../
                        *.parquet
```

### Layers

| Layer  | Format  | Partitioned by             | Contents |
|--------|---------|----------------------------|----------|
| Bronze | JSON    | `dt=YYYY-MM-DD/page=NNN`   | Raw API responses, one file per page |
| Silver | Parquet | `country / state_province` | Deduplicated, normalized, backfilled; null-id rows quarantined |
| Gold   | Parquet | `dt=YYYY-MM-DD`            | Aggregated counts per `(brewery_type, country, state_province)` |

### Silver Transformations

1. `dropDuplicates(["id"])` — keep first occurrence per id
2. Null routing — rows with `id IS NULL` → quarantine path
3. `state_province = COALESCE(state_province, state)`
4. Drop redundant columns: `state`, `street`, `address_2`, `address_3`
5. `brewery_type = LOWER(TRIM(brewery_type))`
6. `country = INITCAP(TRIM(country))`
7. Cast `longitude` / `latitude` → `DoubleType`
8. Add `pipeline_run_date` metadata column

### Gold Aggregations

```sql
SELECT brewery_type, country, state_province,
       COUNT(id)                                   AS brewery_count,
       COUNT(DISTINCT city)                        AS distinct_city_count,
       SUM(CASE WHEN latitude IS NOT NULL THEN 1 ELSE 0 END) AS geocoded_count,
       MAX(pipeline_run_date)                      AS last_updated
FROM silver.breweries
GROUP BY brewery_type, country, state_province
```

---

## Prerequisites

| Tool          | Version   |
|---------------|-----------|
| Docker        | 24+       |
| Docker Compose| V2 (built-in) |
| Git           | any       |

No local Python or Java installation needed — everything runs inside containers.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/elieltonlimeira/BeesBreweriesCase.git
cd BeesBreweriesCase

# 2. Configure environment
cp .env.example .env
# Edit .env — set strong passwords for POSTGRES_PASSWORD, FERNET_KEY,
# WEBSERVER_SECRET_KEY, AIRFLOW_ADMIN_PASSWORD

# 3. Start all services
make up
# or: docker compose up -d

# 4. Wait ~60s for Airflow to initialize, then open:
#   Airflow UI : http://localhost:8080  (admin / from .env)
#   MinIO UI   : http://localhost:9001  (minioadmin / minioadmin by default)

# 5. Trigger the DAG manually from the Airflow UI
#    or via CLI:
docker compose exec airflow-scheduler airflow dags trigger brewery_data_pipeline

# 6. Stop everything
make down
```

---

## Running Tests

Tests require Python 3.11 (for Airflow) or 3.13 (unit/integration only).

```bash
# Install dev dependencies (Anaconda example):
pip install -r requirements-dev.txt

# Unit tests only (fast, no Docker)
make test-unit

# Full test suite (unit + integration)
make test

# Check style
make lint
```

> **Note:** DAG integrity tests (`tests/integration/test_dag_integrity.py`) are
> automatically skipped when `apache-airflow` is not installed. They are designed
> to run inside the Docker container (Python 3.11).

---

## Project Structure

```
BeesBreweriesCase/
├── dags/
│   └── brewery_pipeline_dag.py   # Airflow DAG (TaskFlow API)
├── src/
│   ├── ingestion/
│   │   └── brewery_api_client.py # Paginated API client (tenacity retry)
│   ├── bronze/
│   │   └── bronze_writer.py      # JSON → S3 (idempotent)
│   ├── silver/
│   │   └── silver_transformer.py # PySpark transformations + quarantine
│   ├── gold/
│   │   └── gold_aggregator.py    # PySpark aggregation
│   ├── quality/
│   │   └── data_quality.py       # Hard/soft quality checks per layer
│   └── utils/
│       ├── config.py             # Centralized env-var config (fail-fast)
│       ├── logger.py             # structlog (JSON output)
│       ├── spark_session.py      # SparkSession with S3A config
│       └── storage_client.py     # boto3 S3 client (lru_cache + retry)
├── tests/
│   ├── conftest.py               # Spark Connect session + shared fixtures
│   ├── unit/                     # 98 tests, 97% coverage
│   └── integration/              # End-to-end + DAG integrity
├── docker/airflow/Dockerfile     # OpenJDK 17 JRE + Hadoop AWS JARs
├── docker-compose.yml
├── .env.example
├── Makefile
├── requirements.txt
└── requirements-dev.txt
```

---

## Monitoring & Alerting

| Layer | Mechanism | What's Monitored |
|-------|-----------|-----------------|
| Task failures | Airflow `on_failure_callback` | Any task exception |
| SLA | Airflow SLA miss | Not complete by 08:00 UTC |
| Bronze | `BronzeQualityChecker` | Page count + record count ≥95% of API total |
| Silver | `SilverQualityChecker` | Null ID rate, brewery_type null rate, quarantine rate, coordinate bounds |
| Gold | `GoldQualityChecker` | `SUM(brewery_count)` == Silver `COUNT(*)` |
| Structured logs | structlog (JSON) → mounted volume | Per-task events with timestamps |
| Trends | Gold layer metrics over time | Quarantine rate, total counts |
| CI | GitHub Actions | Test failures + lint on every push |

---

## Design Decisions

**Why partition Silver by `(country, state_province)` instead of `city`?**
There are ~3,000+ distinct cities but only ~70 state/province values. Partitioning by city would create thousands of tiny Parquet files (small file problem). State-level partitioning gives balanced files and matches the most common query pattern (filter by location).

**Why Spark Connect (`remote("local")`) in tests?**
PySpark's classic `py4j` socket mode is broken on Python 3.13. Spark Connect uses gRPC/Arrow instead, which works reliably on all supported Python versions.

**Why separate `valid_df` and `quarantine_df`?**
Rows with a null `id` cannot be linked to any business entity. Writing them to a quarantine path (rather than discarding them) preserves the data for investigation and audit while keeping the main dataset clean.

**Why custom `_log_before_sleep` instead of tenacity's `before_sleep_log`?**
`before_sleep_log` calls `logger.log(level, msg)` with a string log level, which is incompatible with structlog's `BoundLoggerFilteringAtInfo` that expects an integer.
