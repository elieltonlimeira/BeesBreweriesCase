"""
DAG integrity tests — verify the DAG loads correctly and has the expected structure.

These tests do NOT run tasks; they only inspect the DAG object.
No Airflow database connection is required.

NOTE: Airflow 2.9.1 requires Python <3.13. Run these tests inside the Docker
container (Python 3.11) or a virtualenv with the correct Python version.
They are automatically skipped when Airflow is not installed.
"""

import pytest

# Skip entire module if airflow is not installed (e.g., local Python 3.13 dev env)
airflow = pytest.importorskip("airflow", reason="apache-airflow not installed")


@pytest.fixture(scope="module")
def brewery_dag():
    """Load the brewery_pipeline DAG from the module."""
    from airflow.models import DagBag

    bag = DagBag(dag_folder="dags", include_examples=False)
    assert "brewery_data_pipeline" in bag.dags, (
        f"DAG not found. Import errors: {bag.import_errors}"
    )
    return bag.dags["brewery_data_pipeline"]


class TestDagLoads:
    def test_dag_has_no_import_errors(self):
        from airflow.models import DagBag

        bag = DagBag(dag_folder="dags", include_examples=False)
        assert not bag.import_errors, f"DAG import errors: {bag.import_errors}"

    def test_dag_id(self, brewery_dag):
        assert brewery_dag.dag_id == "brewery_data_pipeline"

    def test_schedule(self, brewery_dag):
        # DAG runs daily at 06:00 UTC
        assert str(brewery_dag.schedule_interval) == "0 6 * * *"

    def test_catchup_disabled(self, brewery_dag):
        assert brewery_dag.catchup is False

    def test_max_active_runs(self, brewery_dag):
        assert brewery_dag.max_active_runs == 1


class TestTaskIds:
    EXPECTED_TASK_IDS = {
        "fetch_meta",
        "fetch_bronze_page",
        "validate_bronze",
        "transform_silver",
        "validate_silver",
        "aggregate_gold",
        "validate_gold",
    }

    def test_all_expected_tasks_present(self, brewery_dag):
        actual = set(brewery_dag.task_ids)
        assert self.EXPECTED_TASK_IDS == actual

    def test_task_count(self, brewery_dag):
        assert len(brewery_dag.tasks) == 7


class TestDependencies:
    def test_fetch_meta_has_no_upstream(self, brewery_dag):
        task = brewery_dag.get_task("fetch_meta")
        assert not task.upstream_task_ids

    def test_fetch_bronze_page_depends_on_fetch_meta(self, brewery_dag):
        task = brewery_dag.get_task("fetch_bronze_page")
        assert "fetch_meta" in task.upstream_task_ids

    def test_validate_bronze_depends_on_fetch_bronze_page(self, brewery_dag):
        task = brewery_dag.get_task("validate_bronze")
        assert "fetch_bronze_page" in task.upstream_task_ids

    def test_transform_silver_depends_on_validate_bronze(self, brewery_dag):
        task = brewery_dag.get_task("transform_silver")
        assert "validate_bronze" in task.upstream_task_ids

    def test_validate_silver_depends_on_transform_silver(self, brewery_dag):
        task = brewery_dag.get_task("validate_silver")
        assert "transform_silver" in task.upstream_task_ids

    def test_aggregate_gold_depends_on_validate_silver(self, brewery_dag):
        task = brewery_dag.get_task("aggregate_gold")
        assert "validate_silver" in task.upstream_task_ids

    def test_validate_gold_depends_on_aggregate_gold(self, brewery_dag):
        task = brewery_dag.get_task("validate_gold")
        assert "aggregate_gold" in task.upstream_task_ids


class TestTaskTypes:
    def test_transform_silver_is_bash_operator(self, brewery_dag):
        from airflow.operators.bash import BashOperator

        task = brewery_dag.get_task("transform_silver")
        assert isinstance(task, BashOperator)

    def test_aggregate_gold_is_bash_operator(self, brewery_dag):
        from airflow.operators.bash import BashOperator

        task = brewery_dag.get_task("aggregate_gold")
        assert isinstance(task, BashOperator)

    def test_transform_silver_uses_spark_submit(self, brewery_dag):
        task = brewery_dag.get_task("transform_silver")
        assert "spark-submit" in task.bash_command
        assert "src.silver.silver_transformer" in task.bash_command

    def test_aggregate_gold_uses_spark_submit(self, brewery_dag):
        task = brewery_dag.get_task("aggregate_gold")
        assert "spark-submit" in task.bash_command
        assert "src.gold.gold_aggregator" in task.bash_command
