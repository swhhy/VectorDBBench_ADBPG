"""Wrapper around the Aliyun ADBPG (AnalyticDB for PostgreSQL) vector database."""

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from copy import copy
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AdbpgConfigDict, AdbpgIndexConfig

log = logging.getLogger(__name__)

AUTOTUNE_POLL_SECONDS = 0.5
AUTOTUNE_TIMEOUT_SECONDS = 3600
NATIVE_AUTOTUNE_SQL = (
    "SELECT fastann.nova_autotune(%s::regclass, topk => %s, target_recall => %s::double precision, "
    "n_samples => %s, n_trials => %s, n_threads => %s)"
)



class AdbpgTimeoutError(RuntimeError):
    """ADBPG canceled a database statement after its configured timeout."""


class Adbpg(VectorDB):
    """ADBPG vector database client, using psycopg."""

    # psycopg Cursor is not thread-safe and the COPY protocol cannot be
    # interleaved on a shared connection. Match PgVector/VectorChord and
    # let ConcurrentInsertRunner clamp max_workers=1.
    thread_safe: bool = False

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: AdbpgConfigDict,
        db_case_config: AdbpgIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "Adbpg"
        self.case_config = db_case_config
        self._autotune_topk = kwargs.get("autotune_topk", 10)
        if self.case_config.autotune_parameters is not None and self._autotune_topk not in (10, 100):
            raise ValueError("Native AutoTune supports only benchmark K=10 or K=100")
        # Allow the framework layer (task_runner) to inject a case-specific table
        # name via the `collection_name` kwarg (see Doris for the same pattern).
        override_name = kwargs.get("collection_name")
        self.table_name = override_name if override_name else db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"
        # Index name derives from the table name + algorithm, e.g. vector_1024d_10m_novamt_index.
        self._index_name = f"{self.table_name}_{self.case_config.algorithm}_index"

        self.where_clause = ""
        self.autotune_elapsed_seconds: float | None = None
        self.autotune_status: dict[str, Any] | None = None

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        log.info("%s case config: %s", self.name, self.case_config)
        try:
            if drop_old:
                self._drop_index()
                self._drop_table()
                self._create_table(dim)
                if self.case_config.create_index_before_load:
                    self._create_index()
            else:
                self._apply_search_reloptions()
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        try:
            register_vector(conn)
            conn.autocommit = False
            cursor = conn.cursor()
            cursor.execute("SET adbpg_ai_disable_destroy_reader_gang = on")
            conn.commit()
            if "gp_session_role=utility" in kwargs.get("options", ""):
                row = cursor.execute(
                    "SELECT current_setting('gp_role', true), current_setting('gp_session_role', true)",
                ).fetchone()
                if not row or any(str(value).lower() != "utility" for value in row):
                    # This error must also close the connection in the handler below.
                    raise RuntimeError("ADBPG master-only connection did not enter utility mode")  # noqa: TRY301
                conn.commit()
        except Exception:
            conn.close()
            raise

        return conn, cursor

    def copy_for_thread(self) -> "VectorDB":
        # psycopg holds an open socket that can't be deep-copied; shallow-copy
        # and drop the connection so init() reconnects inside the worker thread.
        db_copy = copy(self)
        db_copy.conn = None
        db_copy.cursor = None
        return db_copy

    def _generate_search_query(self) -> sql.Composed:
        search_param = self.case_config.search_param()
        distance_operator = {
            "l2": "<->",
            "ip": "<#>",
            "cosine": "<=>",
        }.get(search_param["metric"], "<->")

        where_clause = sql.SQL(self.where_clause) if self.where_clause else sql.SQL("")

        return sql.Composed(
            [
                sql.SQL(
                    "SELECT {primary_field} FROM public.{table_name} {where_clause} ORDER BY {vector_field} ",
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    primary_field=sql.Identifier(self._primary_field),
                    where_clause=where_clause,
                    vector_field=sql.Identifier(self._vector_field),
                ),
                sql.SQL(distance_operator),
                sql.SQL(" {search_vector}::vector({dim}) LIMIT %s::int").format(
                    search_vector=sql.Placeholder(),
                    dim=self.dim,
                ),
            ],
        )

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Open a database session, yield, then close it."""
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        self._post_insert()
        if self.case_config.autotune_parameters is not None:
            self._run_autotune()
        self._apply_search_reloptions()

    def _post_insert(self):
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    @property
    def _index_regclass(self) -> str:
        return sql.Identifier("public", self._index_name).as_string()

    @property
    def _has_autotune_grid(self) -> bool:
        options = self.case_config.autotune_parameters
        return options is not None and options.nova_autotune_topk is not None

    def _autotune_request(self, index):
        options = self.case_config.autotune_parameters
        if self._has_autotune_grid:
            query = (
                "SELECT fastann.nova_autotune(%s::regclass, topk => %s::integer[], "
                "target_recall => %s::double precision[], n_samples => %s, n_trials => %s, n_threads => %s)"
            )
            return query, (index, options.nova_autotune_topk, options.nova_autotune_recall,
                           options.n_samples, options.n_trials, options.n_threads)
        return NATIVE_AUTOTUNE_SQL, (index, self._autotune_topk, self._autotune_target_recall,
                                     options.n_samples, options.n_trials, options.n_threads)

    @property
    def _autotune_target_recall(self) -> float:
        value = self.case_config.search_parameters.reloption.get("nova_autotune_recall")
        target = 0.99 if value is None else float(value)
        if not 0 < target <= 1:
            raise ValueError("nova_autotune_recall must be in (0, 1]")
        return target

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {schema}.{index_name}").format(
            schema=sql.Identifier("public"),
            index_name=sql.Identifier(self._index_name),
        )
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _create_index(self) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        metric = self.case_config.parse_metric()
        options = dict(self.case_config.build_parameters.reloption)
        if options.get("dim", self.dim) != self.dim or options.get("distancemeasure", metric) != metric:
            raise ValueError("Build dim/distancemeasure must match the dataset dimension and metric")
        options.update(dim=self.dim, distancemeasure=metric)
        for name, value in self.case_config.build_parameters.guc.items():
            if value is None:
                self.cursor.execute(sql.SQL("SET LOCAL {} TO DEFAULT").format(sql.Identifier(name)))
            else:
                self.cursor.execute("SELECT set_config(%s, %s, true)", (name, str(value)))
        assignments = sql.SQL(", ").join(
            sql.SQL("{} = {}").format(sql.Identifier(name), sql.Literal(value))
            for name, value in options.items()
            if value is not None
        )
        command = sql.SQL(
            "CREATE INDEX {index} ON public.{table} USING ann ({vector}) INCLUDE ({id}) WITH ({options})",
        ).format(
            index=sql.Identifier(self._index_name),
            table=sql.Identifier(self.table_name),
            vector=sql.Identifier(self._vector_field),
            id=sql.Identifier(self._primary_field),
            options=assignments,
        )
        self.cursor.execute(command)
        self.conn.commit()

    def _index_identity(self) -> tuple[int, int]:
        row = self.cursor.execute(
            "SELECT c.oid, c.relfilenode FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid "
            "WHERE c.oid = %s::regclass AND i.indisvalid AND i.indisready",
            (self._index_regclass,),
        ).fetchone()
        if not row:
            raise RuntimeError("Native AutoTune requires a valid, ready index")
        return int(row[0]), int(row[1])

    @staticmethod
    def _decode_rows(rows: list[Any]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            value = row[0]
            value = json.loads(value) if isinstance(value, str) else value
            if not isinstance(value, dict):
                raise TypeError("Native AutoTune returned an invalid progress/status row")
            result.append(value)
        return result

    def _autotune_progress(self, handle: int) -> list[dict[str, Any]]:
        rows = self.cursor.execute(
            "SELECT row_to_json(p) FROM fastann.nova_autotune_progress(%s::bigint) p",
            (handle,),
        ).fetchall()
        return self._decode_rows(rows)

    def _autotune_results(self) -> list[dict[str, Any]]:
        rows = self.cursor.execute(
            "SELECT row_to_json(s) FROM fastann.nova_autotune_status(%s::regclass) s",
            (self._index_regclass,),
        ).fetchall()
        return self._decode_rows(rows)

    def _cancel_autotune(self, handle: int, identity: tuple[int, int]) -> None:
        try:
            self.conn.rollback()
            # Resolve the PID from this exact task at cancellation time, never
            # from a stale PID saved by an earlier progress poll.
            self.cursor.execute(
                "SELECT pg_cancel_backend(p.pid) FROM fastann.nova_autotune_progress(%s::bigint) p "
                "WHERE p.index_relid = %s::oid AND p.index_relfilenode = %s::oid",
                (handle, *identity),
            ).fetchall()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            log.exception("Could not cancel native AutoTune task %s; its index was preserved", handle)

    def _run_autotune(self) -> None:
        autotune = self.case_config.autotune_parameters
        assert autotune is not None
        identity = self._index_identity()
        previous = self._autotune_results()
        self.conn.commit()
        started = time.monotonic()
        self.cursor.execute("SELECT set_config('max_parallel_workers', %s, false)", (str(autotune.n_threads),))
        query, parameters = self._autotune_request(self._index_regclass)
        row = self.cursor.execute(query, parameters).fetchone()
        self.conn.commit()
        if not row or isinstance(row[0], bool) or not isinstance(row[0], int) or row[0] <= 0:
            raise RuntimeError("Native AutoTune returned an invalid task handle")
        handle = row[0]
        try:
            self._wait_for_autotune(handle, identity, previous, started)
        except BaseException:
            self._cancel_autotune(handle, identity)
            raise

    def _wait_for_autotune(
        self,
        handle: int,
        identity: tuple[int, int],
        previous: list[dict[str, Any]],
        started: float,
    ) -> None:
        previous_times = {status.get("updated_at") for status in previous}
        while time.monotonic() - started < AUTOTUNE_TIMEOUT_SECONDS:
            progress = self._autotune_progress(handle)
            self.conn.commit()
            if not progress:
                current = self._index_identity()
                statuses = self._autotune_results()
                self.conn.commit()
                fresh = [
                    status
                    for status in statuses
                    if current == identity
                    and str(status.get("index_relid")) == str(identity[0])
                    and str(status.get("index_relfilenode")) == str(identity[1])
                    and status.get("is_current") is True
                    and status.get("updated_at") is not None
                    and status["updated_at"] not in previous_times
                ]
                if self._has_autotune_grid:
                    options = self.case_config.autotune_parameters
                    expected = {(k, round(r, 6)) for k in options.nova_autotune_topk
                                for r in options.nova_autotune_recall}
                    found = {(int(row["topk"]), round(float(row["target_recall"]), 6)) for row in fresh}
                    if not expected <= found:
                        raise RuntimeError(f"AutoTune grid is incomplete: missing {sorted(expected - found)}")
                if not fresh:
                    raise RuntimeError("Native AutoTune task ended without a fresh applied result for this index")
                self.autotune_status = fresh[0]
                self.autotune_elapsed_seconds = time.monotonic() - started
                log.info(
                    "Native AutoTune task %s completed in %.3fs: %s",
                    handle,
                    self.autotune_elapsed_seconds,
                    self.autotune_status,
                )
                return
            if any(row.get("stage") in {"failed", "error"} for row in progress):
                msg = f"Native AutoTune task {handle} failed"
                raise RuntimeError(msg)
            time.sleep(AUTOTUNE_POLL_SECONDS)
        msg = f"Native AutoTune task {handle} timed out after {AUTOTUNE_TIMEOUT_SECONDS} seconds"
        raise AdbpgTimeoutError(msg)

    def _apply_search_reloptions(self) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        reloptions = dict(self.case_config.search_parameters.reloption)
        if reloptions.get("nova_autotune_recall") is not None or self.case_config.autotune_parameters is not None:
            selected_topk = reloptions.get("nova_autotune_topk", self._autotune_topk)
            if int(selected_topk) != self._autotune_topk:
                raise ValueError("nova_autotune_topk must match benchmark --k")
            reloptions["nova_autotune_topk"] = self._autotune_topk
            reloptions["nova_autotune_recall"] = self._autotune_target_recall
        set_options = [(name, value) for name, value in reloptions.items() if value is not None]
        reset_options = [name for name, value in reloptions.items() if value is None]
        if set_options:
            assignments = sql.SQL(", ").join(
                sql.SQL("{name} = {value}").format(
                    name=sql.Identifier(name),
                    value=sql.Literal(value),
                )
                for name, value in set_options
            )
            command = sql.SQL("ALTER INDEX {schema}.{index} SET ({assignments})").format(
                schema=sql.Identifier("public"),
                index=sql.Identifier(self._index_name),
                assignments=assignments,
            )
            self.cursor.execute(command)
        if reset_options:
            names = sql.SQL(", ").join(sql.Identifier(name) for name in reset_options)
            command = sql.SQL("ALTER INDEX {schema}.{index} RESET ({names})").format(
                schema=sql.Identifier("public"),
                index=sql.Identifier(self._index_name),
                names=names,
            )
            self.cursor.execute(command)
        if set_options or reset_options:
            self.conn.commit()
        if reloptions.get("nova_autotune_recall") is not None:
            identity = self._index_identity()
            rows = self._autotune_results()
            self.conn.commit()
            selected = [r for r in rows if r.get("is_current") and r.get("achieved") and r.get("applied")
                        and (int(r["index_relid"]), int(r["index_relfilenode"])) == identity
                        and r.get("topk") == self._autotune_topk
                        and abs(float(r["target_recall"]) - self._autotune_target_recall) < 1e-6]
            if not selected:
                raise RuntimeError("No achieved AutoTune result applied for the requested TopK/Recall")
            self.autotune_status = selected[0]

    def _apply_search_gucs(self) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        search_gucs = dict(self.case_config.search_parameters.guc)
        for name, value in search_gucs.items():
            if value is None:
                command = sql.SQL("RESET {setting_name}").format(setting_name=sql.Identifier(name))
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            else:
                self.cursor.execute(
                    "SELECT set_config(%s, %s, false)",
                    (name, str(value)),
                )
        if search_gucs:
            self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            if self.with_scalar_labels:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding vector({dim}), {label_field} VARCHAR(64));
                        """,
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        primary_field=sql.Identifier(self._primary_field),
                        dim=dim,
                        label_field=sql.Identifier(self._scalar_label_field),
                    ),
                )
            else:
                self.cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS public.{table_name}
                        ({primary_field} BIGINT PRIMARY KEY, embedding vector({dim}));
                        """,
                    ).format(
                        table_name=sql.Identifier(self.table_name),
                        primary_field=sql.Identifier(self._primary_field),
                        dim=dim,
                    ),
                )

            self.cursor.execute(
                sql.SQL(
                    "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;",
                ).format(table_name=sql.Identifier(self.table_name)),
            )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create adbpg table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        if self.with_scalar_labels:
            assert labels_data is not None, "labels_data should be provided if with_scalar_labels is set to True"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                ),
            ) as copy:
                for i, row in enumerate(metadata_arr):
                    if self.with_scalar_labels:
                        copy.set_types(["bigint", "vector", "varchar"])
                        copy.write_row((row, embeddings_arr[i], labels_data[i]))
                    else:
                        copy.set_types(["bigint", "vector"])
                        copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into adbpg table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Adbpg - {filters}"
            raise ValueError(msg)

        self._apply_search_gucs()
        self._search = self._generate_search_query()

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        q = np.asarray(query)
        result = self.cursor.execute(
            self._search,
            (q, k),
            prepare=True,
            binary=True,
        )
        return [int(i[0]) for i in result.fetchall()]
