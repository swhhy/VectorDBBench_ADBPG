from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from click.testing import CliRunner
from pydantic import SecretStr, ValidationError

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.adbpg import adbpg as adbpg_module
from vectordb_bench.backend.clients.adbpg import cli as adbpg_cli
from vectordb_bench.backend.clients.adbpg.adbpg import Adbpg, AdbpgTimeoutError
from vectordb_bench.backend.clients.adbpg.config import (
    AdbpgAutotuneParameters,
    AdbpgConfig,
    AdbpgIndexConfig,
    AdbpgParameterGroup,
)
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.task_runner import CaseRunner, RunningStatus
from vectordb_bench.cli.batch_cli import build_sub_cmd_args
from vectordb_bench.models import CaseConfig, CaseType, TaskConfig, TaskStage, TestResult

if TYPE_CHECKING:
    from pathlib import Path


def make_index_config(**overrides: Any) -> AdbpgIndexConfig:
    base = {
        "metric_type": MetricType.COSINE,
        "build_parameters": {
            "reloption": {
                "algorithm": "novamt",
                "note": "quotes ', comma, and $$ stay exact",
            },
            "guc": {},
        },
        "search_parameters": {
            "reloption": {
                "nova_autotune_topk": 10,
                "nova_autotune_recall": 0.95,
            },
            "guc": {"search_path": "foo,bar"},
        },
        "autotune_parameters": {
            "n_samples": 1000,
            "n_trials": 200,
            "n_threads": 8,
        },
    }
    base.update(overrides)
    return AdbpgIndexConfig(**base)


def applied_status(**overrides: Any) -> dict[str, Any]:
    result = {
        "index_relid": "42",
        "index_relfilenode": "99",
        "is_current": True,
        "applied": True,
        "achieved": True,
        "updated_at": "2026-10-30T12:00:00+08:00",
        "topk": 10,
        "target_recall": .95,
    }
    result.update(overrides)
    return result


class FakeCursor:
    def __init__(
        self,
        fail_on: str | None = None,
        error: Exception | None = None,
        fetchone_result: tuple[Any, ...] = ("utility", "utility"),
        progress: list[list[dict[str, Any]]] | None = None,
        statuses: list[list[dict[str, Any]]] | None = None,
        handle: Any = 123,
    ):
        self.executions: list[tuple[str, Any]] = []
        self.fail_on = fail_on
        self.error = error or RuntimeError("database error")
        self.fetchone_result = fetchone_result
        self.progress = [list(rows) for rows in progress] if progress is not None else [[]]
        self.statuses = [list(rows) for rows in statuses] if statuses is not None else [[], [applied_status()]]
        self.handle = handle
        self.connection: FakeConnection | None = None
        self.closed = False
        self._one = None
        self._all = []
        self.index_options = [
            "nova_ef_search=250",
            "nova_rescore_amp=2.20",
            "nova_max_scan_points=11480",
            "nova_adaptive_gamma=0.22",
            "nova_neighbor_cap=72",
        ]

    def execute(self, query: Any, params: Any = None, **kwargs: Any):
        query = query.as_string() if isinstance(query, psycopg.sql.Composable) else str(query)
        self.executions.append((query, params))
        if self.connection is not None and self.connection.aborted:
            raise psycopg.errors.InFailedSqlTransaction
        if self.fail_on and self.fail_on in query:
            if self.connection is not None:
                self.connection.aborted = True
            raise self.error
        self._one, self._all = None, []
        if "pg_cancel_backend" in query:
            self._all = [(True,)]
            self.progress = [[]]
        elif "nova_autotune_progress" in query:
            rows = self.progress.pop(0) if self.progress else []
            self._all = [(row,) for row in rows]
        elif "nova_autotune_status" in query:
            rows = self.statuses.pop(0) if self.statuses else [applied_status()]
            self._all = [(row,) for row in rows]
        elif query.startswith("SELECT fastann.nova_autotune("):
            self._one = (self.handle,)
        elif "JOIN pg_index" in query:
            self._one = (42, 99)
        elif "SELECT reloptions" in query:
            self._one = (self.index_options,)
        elif "current_setting('gp_role'" in query:
            self._one = self.fetchone_result
        return self

    def fetchall(self):
        return self._all

    def fetchone(self):
        return self._one

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.aborted = False
        self.closed = False
        cursor.connection = self

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1
        self.aborted = False

    def close(self) -> None:
        self.closed = True


def make_client(*, with_scalar_labels: bool = False, topk: int = 10) -> Adbpg:
    client = Adbpg.__new__(Adbpg)
    client.name = "Adbpg"
    client.case_config = make_index_config()
    client.case_config.search_parameters.reloption["nova_autotune_topk"] = topk
    client._autotune_topk = topk
    client.table_name = "vector$table"
    client.connect_config = {}
    client.dim = 768
    client.with_scalar_labels = with_scalar_labels
    client._primary_field = "id"
    client._vector_field = "embedding"
    client._scalar_label_field = "label"
    client._index_name = "vector$table_novamt_index"
    client.where_clause = ""
    client.autotune_elapsed_seconds = None
    client.autotune_status = None
    return client


class TestAdbpgConfig:
    def test_defaults_allow_construction_without_password(self):
        cfg = AdbpgConfig(db_label="", version="", note="")
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.db_name == "postgres"
        assert cfg.password.get_secret_value() == ""

    def test_to_dict_uses_normal_coordinator_session(self):
        cfg = AdbpgConfig(
            user_name=SecretStr("u"),
            password=SecretStr("pw"),
            host="h.example.com",
            port=5432,
            db_name="postgres",
        )
        connect_config = cfg.to_dict()["connect_config"]
        assert connect_config == {
            "host": "h.example.com",
            "port": 5432,
            "dbname": "postgres",
            "user": "u",
            "password": "pw",
        }


class TestAdbpgStructuredConfig:
    def test_free_form_names_and_scalar_types_are_preserved(self):
        group = AdbpgParameterGroup(
            reloption={"future.knob": 7, "enabled": True, "reset_me": None},
            guc={"search_path": "foo,bar", "ratio": 0.5},
        )
        assert group.model_dump() == {
            "reloption": {"future.knob": 7, "enabled": True, "reset_me": None},
            "guc": {"search_path": "foo,bar", "ratio": 0.5},
        }

    @pytest.mark.parametrize(
        "value",
        [
            {"reloption": {"bad": [1, 2]}},
            {"guc": {"bad": {"nested": True}}},
            {"unsupported": {}},
        ],
    )
    def test_invalid_parameter_structures_are_rejected(self, value: dict[str, Any]):
        with pytest.raises(ValidationError):
            AdbpgParameterGroup(**value)

    def test_build_defaults_are_merged_with_user_parameters(self):
        build = AdbpgIndexConfig(
            build_parameters={"reloption": {"algorithm": "novamt"}},
        ).build_parameters
        assert build.model_dump() == {
            "reloption": {
                "algorithm": "novamt",
                "hnsw_m": 48,
                "hnsw_ef_construction": 600,
                "turbovec_bits": 4,
                "auto_reduction": False,
                "max_key_len": 1,
            },
            "guc": {"fastann.build_parallel_processes": 32, "fastann.nova_build_optimize_level": 3},
        }

    def test_default_algorithm_is_used_for_every_entry_point(self):
        assert AdbpgIndexConfig().algorithm == "novamt"

    def test_autotune_defaults_are_only_the_three_native_controls(self):
        autotune = AdbpgAutotuneParameters()
        assert autotune.model_dump(exclude_none=True) == {"n_samples": 1000, "n_trials": 200, "n_threads": 8}
        assert AdbpgAutotuneParameters(n_samples=0).n_samples == 0

    def test_autotune_is_disabled_when_configuration_is_omitted(self):
        assert AdbpgIndexConfig().autotune_parameters is None
        assert AdbpgIndexConfig(autotune_parameters="").autotune_parameters is None

    @pytest.mark.parametrize(
        "value",
        [
            {"n_samples": -1},
            {"n_trials": 0},
            {"n_threads": 0},
            {"timeout": 3600},
            {"topk": [10]},
            {"target_recall": [0.99]},
            {"query_table": "queries"},
            {"query_column": "embedding"},
            {"noise_scale": 0.5},
            {"synthetic_query_mode": "random"},
        ],
    )
    def test_unsupported_or_invalid_autotune_parameters_are_rejected(self, value: dict[str, Any]):
        with pytest.raises(ValidationError):
            AdbpgAutotuneParameters(**value)

    def test_legacy_enable_switch_is_rejected(self):
        with pytest.raises(ValidationError, match="omit autotune_parameters"):
            AdbpgAutotuneParameters(enable=False)

    def test_old_flattened_fields_are_migrated_to_structured_parameters(self):
        config = AdbpgIndexConfig(
            algorithm="novad",
            hnsw_m=32,
            ef_construction=256,
            nlist=2048,
            rabitq_bits=6,
            auto_reduction=True,
            pca_dim=384,
            build_parallel_processes=16,
            ef_search=77,
            max_scan_points=900,
            quantize_rescore_amp=2.0,
            nova_adaptive_gamma=0.4,
            index_scan_mode="streaming",
            nprobe=9,
        )
        assert config.build_parameters.reloption == {
            "algorithm": "novad",
            "hnsw_m": 32,
            "hnsw_ef_construction": 256,
            "rabitq_bits": 6,
            "auto_reduction": True,
            "nlist": 2048,
            "pca_dim": 384,
            "max_key_len": 1,
        }
        assert config.build_parameters.guc == {
            "fastann.build_parallel_processes": 16,
            "fastann.nova_build_optimize_level": 3,
        }
        assert config.search_parameters.guc == {
            "fastann.hnsw_ef_search": 77,
            "fastann.hnsw_max_scan_points": 900,
            "fastann.quantize_rescore_amp": 2.0,
            "fastann.nova_adaptive_gamma": 0.4,
            "fastann.index_scan_mode": "streaming",
            "fastann.nova_nprobe": 9,
        }

    def test_structured_parameters_take_precedence_over_legacy_fields(self):
        config = AdbpgIndexConfig(
            algorithm="novad",
            ef_search=77,
            build_parameters={"reloption": {"algorithm": "novamt"}},
            search_parameters={"guc": {"fastann.hnsw_ef_search": 130}},
        )
        assert config.algorithm == "novamt"
        assert config.search_parameters.guc == {"fastann.hnsw_ef_search": 130}

    def test_unknown_fields_are_still_rejected(self):
        with pytest.raises(ValidationError, match="unknown"):
            AdbpgIndexConfig(unknown=1)


class TestAdbpgNativeSQL:
    @staticmethod
    def attach(client: Adbpg | None = None, **cursor_options: Any):
        client = client or make_client()
        cursor = FakeCursor(**cursor_options)
        conn = FakeConnection(cursor)
        client.conn, client.cursor = conn, cursor
        return client, conn, cursor

    @pytest.mark.parametrize("topk", [10, 100])
    def test_optimize_builds_natively_then_tunes_with_scenario_topk_and_three_controls(self, topk: int):
        client, _conn, cursor = self.attach(make_client(topk=topk), statuses=[[], [applied_status(topk=topk)], [applied_status(topk=topk)]])
        client.optimize()
        statements = [query for query, _ in cursor.executions]
        build = next(query for query in statements if "CREATE INDEX" in query)
        assert 'USING ann ("embedding") INCLUDE ("id")' in build
        assert '"max_key_len" = 1' in build
        assert "quotes '', comma, and $$ stay exact" in build
        assert ("SELECT set_config(%s, %s, true)", ("fastann.nova_build_optimize_level", "3")) in cursor.executions
        submissions = [
            (query, params) for query, params in cursor.executions if query.startswith("SELECT fastann.nova_autotune(")
        ]
        assert submissions == [
            (
                (
                    "SELECT fastann.nova_autotune(%s::regclass, topk => %s, "
                    "target_recall => %s::double precision, "
                    "n_samples => %s, n_trials => %s, n_threads => %s)"
                ),
                ('"public"."vector$table_novamt_index"', topk, 0.95, 1000, 200, 8),
            )
        ]
        preflight = [(query, params) for query, params in cursor.executions if query.startswith("EXPLAIN ")]
        assert preflight == []
        assert statements.index(build) < statements.index(submissions[0][0])
        assert any("ALTER INDEX" in query for query in statements)
        assert not any("vectordbbench_" in query for query in statements)
        assert client.autotune_status["applied"] is True
        assert client.autotune_elapsed_seconds is not None

    @pytest.mark.parametrize("before,after", [(True, False), (False, True), (True, True)])
    def test_build_flags_preserve_constructor_and_post_insert_order(self, monkeypatch, before, after):
        cursor = FakeCursor()
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))
        config = make_index_config(create_index_before_load=before, create_index_after_load=after)
        client = Adbpg(dim=768, db_config={"table_name": "vector", "connect_config": {}},
                       db_case_config=config, drop_old=True)
        assert sum("CREATE INDEX" in q for q, _ in cursor.executions) == int(before)
        assert not any("nova_autotune(" in q for q, _ in cursor.executions)
        client.conn, client.cursor = FakeConnection(cursor), cursor
        client.optimize()
        assert sum("CREATE INDEX" in q for q, _ in cursor.executions) == int(before) + int(after)
        assert sum(q.startswith("SELECT fastann.nova_autotune(") for q, _ in cursor.executions) == 1

    def test_at_least_one_index_creation_stage_is_required(self):
        with pytest.raises(ValidationError):
            make_index_config(create_index_before_load=False, create_index_after_load=False)

    def test_reported_worker_failure_cancels_owned_task(self):
        client, _conn, cursor = self.attach(progress=[[{"stage": "failed"}]])
        with pytest.raises(RuntimeError, match="task 123 failed"):
            client._run_autotune()
        assert any("pg_cancel_backend" in q and p == (123, 42, 99) for q, p in cursor.executions)
        assert not any("ALTER INDEX" in q for q, _ in cursor.executions)

    def test_optimize_without_autotune_only_builds_and_applies_explicit_options(self):
        client = make_client()
        client.case_config = make_index_config(autotune_parameters=None, search_parameters={"reloption": {"nova_ef_search": 80}})
        client, _conn, cursor = self.attach(client)
        client.optimize()
        assert sum("CREATE INDEX" in query for query, _ in cursor.executions) == 1
        assert not any("SELECT fastann.nova_autotune" in query for query, _ in cursor.executions)

    def test_waits_for_async_task_before_accepting_the_result(self, monkeypatch: pytest.MonkeyPatch):
        client, _conn, cursor = self.attach(progress=[[{"stage": "searching"}], []])
        sleeps = []
        monkeypatch.setattr(adbpg_module.time, "sleep", sleeps.append)
        client.optimize()
        assert sleeps == [adbpg_module.AUTOTUNE_POLL_SECONDS]
        assert sum("nova_autotune_progress" in query for query, _ in cursor.executions) == 2
        assert client.autotune_status == applied_status()

    @pytest.mark.parametrize(
        "bad",
        [
            applied_status(is_current=False),
            applied_status(index_relid="777"),
            applied_status(index_relfilenode="888"),
        ],
    )
    def test_disappeared_progress_does_not_accept_a_foreign_result(self, bad: dict[str, Any]):
        client, _conn, cursor = self.attach(statuses=[[], [bad]])
        with pytest.raises(RuntimeError, match="without a fresh applied result"):
            client._run_autotune()
        assert client.autotune_status is None
        assert not any("DROP INDEX" in query for query, _ in cursor.executions)

    def test_a_stale_success_from_an_earlier_run_is_not_a_new_result(self):
        old = applied_status()
        client, _conn, _cursor = self.attach(statuses=[[old], [old]])
        with pytest.raises(RuntimeError, match="without a fresh applied result"):
            client._run_autotune()

    def test_native_achieved_false_is_not_silently_rewritten(self):
        client, _conn, _cursor = self.attach(statuses=[[], [applied_status(achieved=False)]])
        client._run_autotune()
        assert client.autotune_status["achieved"] is False

    @pytest.mark.parametrize("handle", [None, True, 0, "123"])
    def test_invalid_task_handle_fails(self, handle: Any):
        client, _conn, _cursor = self.attach(handle=handle)
        with pytest.raises(RuntimeError, match="invalid task handle"):
            client._run_autotune()

    def test_timeout_cancels_only_the_owned_task_and_preserves_the_index(self, monkeypatch: pytest.MonkeyPatch):
        client, _conn, cursor = self.attach(progress=[[{"stage": "searching"}]])
        clock = {"now": 0.0}
        monkeypatch.setattr(adbpg_module.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(adbpg_module.time, "sleep", lambda _: clock.update(now=3601.0))
        with pytest.raises(AdbpgTimeoutError, match="timed out after 3600 seconds"):
            client._run_autotune()
        cancels = [(query, params) for query, params in cursor.executions if "pg_cancel_backend" in query]
        assert len(cancels) == 1
        assert cancels[0][1] == (123, 42, 99)
        assert "p.index_relid = %s::oid AND p.index_relfilenode = %s::oid" in cancels[0][0]
        assert not any("DROP INDEX" in query for query, _ in cursor.executions)

    def test_failed_build_does_not_start_autotune(self):
        client, conn, cursor = self.attach(fail_on="CREATE INDEX")
        with pytest.raises(RuntimeError, match="database error"):
            client.optimize()
        assert not any(query.startswith("SELECT fastann.nova_autotune(") for query, _ in cursor.executions)

    @pytest.mark.parametrize("overrides", [{"dim": 3}, {"distancemeasure": "l2"}])
    def test_conflicting_build_dimension_or_metric_is_rejected(self, overrides: dict[str, Any]):
        client = make_client()
        client.case_config.build_parameters.reloption.update(overrides)
        client, _conn, cursor = self.attach(client)
        with pytest.raises(ValueError, match="must match"):
            client.optimize()
        assert not any("CREATE INDEX" in query for query, _ in cursor.executions)

    def test_missing_native_api_fails_during_optimize(self):
        client, _conn, cursor = self.attach(
            fail_on="SELECT fastann.nova_autotune(",
            error=psycopg.errors.UndefinedFunction("function unavailable"),
        )
        with pytest.raises(psycopg.errors.UndefinedFunction):
            client.optimize()
        assert any("CREATE INDEX" in query for query, _ in cursor.executions)
        assert not any(query.startswith("EXPLAIN") for query, _ in cursor.executions)

    @pytest.mark.parametrize("topk", [10, 100])
    def test_task_runner_routes_scenario_topk_to_the_client(self, monkeypatch: pytest.MonkeyPatch, topk: int):
        config = make_index_config()
        config.search_parameters.reloption["nova_autotune_topk"] = topk
        task = TaskConfig(
            db=DB.Adbpg,
            db_config=AdbpgConfig(),
            db_case_config=config,
            case_config=CaseConfig(case_id=CaseType.Performance768D1M, k=topk),
        )
        runner = CaseRunner(
            run_id="run-id",
            config=task,
            ca=task.case_config.case,
            status=RunningStatus.PENDING,
            dataset_source=DatasetSource.S3,
        )
        cursor = FakeCursor(statuses=[[applied_status(topk=topk)]])
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))
        runner.init_db(drop_old=False)
        assert runner.db._autotune_topk == topk

    @pytest.mark.parametrize("topk", [0, 50, 1000])
    def test_unsupported_topk_fails_before_connecting(self, topk: int):
        with pytest.raises(ValueError, match="only benchmark K=10 or K=100"):
            Adbpg(
                dim=768,
                db_config={"table_name": "vector", "connect_config": {}},
                db_case_config=make_index_config(),
                autotune_topk=topk,
            )

    @pytest.mark.parametrize("autotune", [None, {}, {"nova_autotune_topk": [10, 100], "nova_autotune_recall": [.95]}])
    def test_batch_reuse_uses_original_framework_key(self, autotune):
        runners = []
        for topk in (10, 100):
            task = TaskConfig(
                db=DB.Adbpg,
                db_config=AdbpgConfig(),
                db_case_config=make_index_config(autotune_parameters=autotune),
                case_config=CaseConfig(case_id=CaseType.Performance768D1M, k=topk),
            )
            runners.append(
                CaseRunner(
                    run_id="run-id",
                    config=task,
                    ca=task.case_config.case,
                    status=RunningStatus.PENDING,
                    dataset_source=DatasetSource.S3,
                )
            )
        assert runners[0].load_reuse_key() == runners[1].load_reuse_key()

    def test_manual_search_does_not_restrict_topk(self, monkeypatch: pytest.MonkeyPatch):
        cursor = FakeCursor()
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))
        client = Adbpg(
            dim=768,
            db_config={"table_name": "vector", "connect_config": {}},
            db_case_config=make_index_config(autotune_parameters=None, search_parameters={}),
            autotune_topk=50,
        )
        assert client._autotune_topk == 50

    def test_recomputed_applied_flag_does_not_make_an_old_result_fresh(self):
        client, _conn, _cursor = self.attach(statuses=[[applied_status(applied=False)], [applied_status()]])
        with pytest.raises(RuntimeError, match="without a fresh applied result"):
            client._run_autotune()

    def test_existing_index_selects_without_tuning_or_rebuilding(self, monkeypatch: pytest.MonkeyPatch):
        cursor = FakeCursor(statuses=[[applied_status()]])
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, "_create_connection", staticmethod(lambda **_: (conn, cursor)))
        client = Adbpg(
            dim=768, db_config={"table_name": "vector", "connect_config": {}}, db_case_config=make_index_config()
        )
        client.copy_for_thread()
        client.copy_for_thread()
        assert sum(q.startswith("SELECT fastann.nova_autotune(") for q, _ in cursor.executions) == 0
        assert any("ALTER INDEX" in q for q, _ in cursor.executions)
        assert not any("CREATE INDEX" in q for q, _ in cursor.executions)
        assert cursor.closed
        assert conn.closed

    def test_fresh_search_worker_only_applies_explicit_gucs(self):
        client, _conn, cursor = self.attach()
        client.case_config.search_parameters.guc["fastann.hnsw_ef_search"] = 999
        client.prepare_filter(non_filter)
        sets = dict(params for query, params in cursor.executions if query == "SELECT set_config(%s, %s, false)")
        assert sets == {
            "search_path": "foo,bar",
            "fastann.hnsw_ef_search": "999",
        }
        assert not any("ALTER INDEX" in query or "fastann.nova_autotune(" in query for query, _ in cursor.executions)

    def test_normal_search_without_autotune_only_applies_requested_gucs(self):
        client = make_client()
        client.case_config = make_index_config(autotune_parameters=None)
        client, _conn, cursor = self.attach(client)
        client.prepare_filter(non_filter)
        assert cursor.executions == [("SELECT set_config(%s, %s, false)", ("search_path", "foo,bar"))]

    @pytest.mark.parametrize("roles", [("execute", "dispatch"), ("utility", "dispatch"), (None, None)])
    def test_master_only_connection_cannot_silently_fall_back(
        self, monkeypatch: pytest.MonkeyPatch, roles: tuple[str | None, str | None]
    ):
        cursor = FakeCursor(fetchone_result=roles)
        conn = FakeConnection(cursor)
        monkeypatch.setattr(adbpg_module.psycopg, "connect", lambda **_: conn)
        monkeypatch.setattr(adbpg_module, "register_vector", lambda _: None)
        with pytest.raises(RuntimeError, match="did not enter utility mode"):
            Adbpg._create_connection(options="-c gp_session_role=utility")
        assert conn.closed

    def test_master_only_option_is_explicit_and_checked_on_every_connection(self, monkeypatch: pytest.MonkeyPatch):
        options = AdbpgConfig(master_only=True).to_dict()["connect_config"]
        assert options["options"] == "-c gp_session_role=utility"
        cursor = FakeCursor()
        conn = FakeConnection(cursor)
        monkeypatch.setattr(adbpg_module.psycopg, "connect", lambda **_: conn)
        monkeypatch.setattr(adbpg_module, "register_vector", lambda _: None)
        assert Adbpg._create_connection(**options) == (conn, cursor)
        assert any("current_setting('gp_role'" in query for query, _ in cursor.executions)

    def test_optimize_failure_prevents_the_search_runner(self, monkeypatch: pytest.MonkeyPatch):
        task = TaskConfig(
            db=DB.Adbpg,
            db_config=AdbpgConfig(),
            db_case_config=make_index_config(),
            case_config=CaseConfig(case_id=CaseType.Performance768D1M, k=10),
            stages=[TaskStage.DROP_OLD, TaskStage.LOAD, TaskStage.SEARCH_SERIAL],
        )
        runner = CaseRunner(
            run_id="run-id",
            config=task,
            ca=task.case_config.case,
            status=RunningStatus.PENDING,
            dataset_source=DatasetSource.S3,
        )
        search_started = []
        monkeypatch.setattr(CaseRunner, "_load_data", lambda _self: (1_000_000, 1.0))
        monkeypatch.setattr(
            CaseRunner, "_optimize", lambda _self: (_ for _ in ()).throw(RuntimeError("native build failed"))
        )
        monkeypatch.setattr(CaseRunner, "_init_search_runners", lambda _self: search_started.append(True))
        with pytest.raises(RuntimeError, match="native build failed"):
            runner._run_perf_case(drop_old=True)
        assert search_started == []


def test_single_and_batch_cli_preserve_equivalent_structured_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = []
    monkeypatch.setattr(
        adbpg_cli,
        "run",
        lambda **kwargs: captured.append((kwargs["db_config"], kwargs["db_case_config"])),
    )
    values = {
        "build_parameters": {"reloption": {"algorithm": "novamt", "hnsw_m": 48}, "guc": {}},
        "search_parameters": {"reloption": {"reset_me": None}, "guc": {"search_path": "foo,bar"}},
        "autotune_parameters": {"n_samples": 1000, "n_trials": 200, "n_threads": 8},
    }
    config_path = tmp_path / "adbpg.yml"
    config_path.write_text(
        "adbpgnova:\n"
        "  db_label: cohere1m-novamt\n"
        "  user_name: tester\n"
        "  host: localhost\n"
        "  db_name: postgres\n"
        f"  build_parameters: {json.dumps(values['build_parameters'])}\n"
        f"  search_parameters: {json.dumps(values['search_parameters'])}\n"
        f"  autotune_parameters: {json.dumps(values['autotune_parameters'])}\n"
    )
    result = CliRunner().invoke(adbpg_cli.AdbpgNova, ["--config-file", str(config_path), "--dry-run"])
    assert result.exit_code == 0, result.output

    batch_args = build_sub_cmd_args(
        {
            "adbpgnova": [
                {
                    "db_label": "cohere1m-novamt",
                    "user_name": "tester",
                    "host": "localhost",
                    "db_name": "postgres",
                    "dry_run": True,
                    **values,
                }
            ]
        }
    )[0]
    result = CliRunner().invoke(adbpg_cli.AdbpgNova, batch_args[1:])
    assert result.exit_code == 0, result.output
    assert captured[0][0].db_label == captured[1][0].db_label == "cohere1m-novamt"
    assert captured[0][1].model_dump(mode="json") == captured[1][1].model_dump(mode="json")


def test_cli_omits_autotune_to_disable_it(monkeypatch: pytest.MonkeyPatch):
    captured = []
    monkeypatch.setattr(adbpg_cli, "run", lambda **kwargs: captured.append(kwargs["db_case_config"]))

    result = CliRunner().invoke(
        adbpg_cli.AdbpgNova,
        ["--user-name", "tester", "--host", "localhost", "--db-name", "postgres"],
    )

    assert result.exit_code == 0, result.output
    assert captured[0].autotune_parameters is None

    batch_args = build_sub_cmd_args(
        {
            "adbpgnova": [
                {
                    "user_name": "tester",
                    "host": "localhost",
                    "db_name": "postgres",
                }
            ]
        }
    )[0]
    result = CliRunner().invoke(adbpg_cli.AdbpgNova, batch_args[1:])
    assert result.exit_code == 0, result.output
    assert captured[1].autotune_parameters is None


@pytest.mark.parametrize(
    ("topk", "autotune", "accepted"),
    [(10, True, True), (100, True, True), (50, True, False), (50, False, True)],
)
def test_cli_limits_only_autotune_to_the_two_scenarios(
    monkeypatch: pytest.MonkeyPatch, topk: int, autotune: bool, accepted: bool
):
    captured = []
    monkeypatch.setattr(adbpg_cli, "run", lambda **kwargs: captured.append(kwargs["k"]))
    args = ["--user-name", "tester", "--host", "localhost", "--db-name", "postgres", "--k", str(topk)]
    if autotune:
        args.extend(["--autotune-parameters", "{}"])
    result = CliRunner().invoke(adbpg_cli.AdbpgNova, args)
    if accepted:
        assert result.exit_code == 0, result.output
        assert captured == [topk]
    else:
        assert result.exit_code == 2
        assert "--k" in result.output
        assert "only 10 or 100" in result.output
        assert captured == []


def test_cli_accepts_formatted_multiline_json_parameters(monkeypatch: pytest.MonkeyPatch):
    captured = []
    monkeypatch.setattr(adbpg_cli, "run", lambda **kwargs: captured.append(kwargs["db_case_config"]))
    build_parameters = """{
  "reloption": {
    "algorithm": "novamt",
    "hnsw_m": 48,
    "hnsw_ef_construction": 600
  },
  "guc": {}
}"""
    search_parameters = """{
  "reloption": {},
  "guc": {
    "fastann.hnsw_ef_search": 130,
    "fastann.hnsw_max_scan_points": 5000,
    "fastann.quantize_rescore_amp": 2.0
  }
}"""

    result = CliRunner().invoke(
        adbpg_cli.AdbpgNova,
        [
            "--case-type",
            "Performance1024D1M",
            "--k",
            "10",
            "--host",
            "localhost",
            "--db-name",
            "postgres",
            "--user-name",
            "tester",
            "--password",
            "password",
            "--build-parameters",
            build_parameters,
            "--search-parameters",
            search_parameters,
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured[0]
    assert config.build_parameters.reloption["algorithm"] == "novamt"
    assert config.build_parameters.reloption["hnsw_m"] == 48
    assert config.build_parameters.reloption["hnsw_ef_construction"] == 600
    assert config.search_parameters.guc == {
        "fastann.hnsw_ef_search": 130,
        "fastann.hnsw_max_scan_points": 5000,
        "fastann.quantize_rescore_amp": 2.0,
    }


class TestResultRoundTrip:
    def test_read_file_rehydrates_structured_adbpg_config(self, tmp_path: Path):
        result_dir = tmp_path / "AnalyticDB for PostgreSQL"
        result_dir.mkdir()
        result_file = result_dir / "result_test_run.json"
        payload = {
            "run_id": "round-trip",
            "task_label": "round-trip",
            "results": [
                {
                    "metrics": {
                        "max_load_count": 0,
                        "insert_duration": 0.0,
                        "optimize_duration": 0.0,
                        "load_duration": 0.0,
                        "qps": 1.0,
                        "serial_latency_p99": 0.0,
                        "serial_latency_p95": 0.0,
                        "recall": 1.0,
                        "ndcg": 1.0,
                        "conc_num_list": [],
                        "conc_qps_list": [],
                        "conc_latency_p99_list": [],
                        "conc_latency_p95_list": [],
                        "conc_latency_avg_list": [],
                    },
                    "task_config": {
                        "db": DB.Adbpg.value,
                        "db_config": {"db_label": "", "version": "", "note": ""},
                        "db_case_config": make_index_config().model_dump(mode="json"),
                        "case_config": {"case_id": 5, "custom_case": {}, "k": 10},
                        "stages": ["search_serial"],
                        "load_concurrency": 0,
                    },
                    "label": ":)",
                }
            ],
            "timestamp": 0.0,
        }
        result_file.write_text(json.dumps(payload))

        result = TestResult.read_file(result_file, trans_unit=False)
        rehydrated = result.results[0].task_config.db_case_config
        assert isinstance(rehydrated, AdbpgIndexConfig)
        assert rehydrated.model_dump(mode="json") == make_index_config().model_dump(mode="json")

class TestAutoTuneGrid:
    @staticmethod
    def configure(client):
        client.case_config.autotune_parameters = AdbpgAutotuneParameters(
            n_trials=500, nova_autotune_topk=[10, 100],
            nova_autotune_recall=[round(0.90 + i * .01, 2) for i in range(10)])
        return client

    @staticmethod
    def rows():
        return [applied_status(topk=k, target_recall=round(.90 + i * .01, 2), applied=False)
                for k in [10, 100] for i in range(10)]

    def test_array_request_and_no_pre_tune_selector(self):
        client = self.configure(make_client())
        query, params = client._autotune_request(None)
        assert 'topk => %s::integer[]' in query
        assert 'target_recall => %s::double precision[]' in query
        assert params == (None, [10, 100], [.90,.91,.92,.93,.94,.95,.96,.97,.98,.99], 1000, 500, 8)

    def test_grid_accepts_saved_rows_without_maximum_target_achieved(self):
        client = self.configure(make_client())
        rows = self.rows()
        rows[-1]['achieved'] = False
        applied = applied_status(topk=10, target_recall=.95)
        cursor = FakeCursor(statuses=[[], rows])
        client.cursor = cursor
        client.conn = FakeConnection(cursor)
        client._run_autotune()
        assert client.autotune_status == rows[0]
        assert not any(q.startswith("ALTER INDEX") for q, _ in cursor.executions)

    def test_incomplete_grid_fails_before_applying(self):
        client = self.configure(make_client())
        cursor = FakeCursor(statuses=[[], self.rows()[:-1]])
        client.cursor = cursor
        client.conn = FakeConnection(cursor)
        with pytest.raises(RuntimeError, match='grid is incomplete'):
            client._run_autotune()
        assert not any(q.startswith('ALTER INDEX') for q, _ in cursor.executions)

    @pytest.mark.parametrize("bad", [
        applied_status(achieved=False), applied_status(applied=False),
        applied_status(index_relfilenode="100"), applied_status(is_current=False),
        applied_status(topk=100), applied_status(target_recall=.99),
    ])
    def test_search_selector_rejects_wrong_or_unapplied_status(self, bad):
        client = self.configure(make_client())
        client.cursor = FakeCursor(statuses=[[bad]])
        client.conn = FakeConnection(client.cursor)
        with pytest.raises(RuntimeError, match="No achieved AutoTune result applied"):
            client._apply_search_reloptions()

    def test_unachieved_requested_search_pair_rejected(self):
        client = self.configure(make_client())
        client.case_config.search_parameters.reloption['nova_autotune_recall'] = .99
        rows = self.rows()
        for row in rows:
            if row['target_recall'] == .99:
                row['achieved'] = False
        cursor = FakeCursor(statuses=[rows])
        client.cursor = cursor
        client.conn = FakeConnection(cursor)
        with pytest.raises(RuntimeError, match='AutoTune'):
            client._apply_search_reloptions()

    def test_reused_index_selects_without_retuning(self, monkeypatch):
        config = make_index_config()
        holder = self.configure(make_client())
        config.autotune_parameters = holder.case_config.autotune_parameters
        applied = applied_status(topk=10, target_recall=.95)
        cursor = FakeCursor(statuses=[[applied]])
        conn = FakeConnection(cursor)
        monkeypatch.setattr(Adbpg, '_create_connection', staticmethod(lambda **_: (conn, cursor)))
        client = Adbpg(dim=768, db_config={'table_name':'vector','connect_config':{}}, db_case_config=config)
        assert client.autotune_status == applied
        assert not any(q.startswith('SELECT fastann.nova_autotune(') for q, _ in cursor.executions)

    @pytest.mark.parametrize('values', [
        {'nova_autotune_topk':[10]},
        {'nova_autotune_topk':[], 'nova_autotune_recall':[.99]},
        {'nova_autotune_topk':[10], 'nova_autotune_recall':[.89]},
    ])
    def test_invalid_grid(self, values):
        with pytest.raises(ValidationError):
            AdbpgAutotuneParameters(**values)
