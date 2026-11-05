from ast import literal_eval
from collections.abc import Mapping
from typing import Any, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from yaml import safe_load

from ..api import DBCaseConfig, DBConfig, MetricType

AdbpgParameterValue: TypeAlias = str | int | float | bool | None

DEFAULT_BUILD_RELOPTIONS: dict[str, AdbpgParameterValue] = {
    "algorithm": "novamt",
    "hnsw_m": 48,
    "hnsw_ef_construction": 600,
    "turbovec_bits": 4,
    "auto_reduction": False,
    "max_key_len": 1,
}
DEFAULT_BUILD_GUCS: dict[str, AdbpgParameterValue] = {
    "fastann.build_parallel_processes": 32,
    "fastann.nova_build_optimize_level": 3,
}

LEGACY_BUILD_RELOPTIONS = {
    "algorithm": "algorithm",
    "hnsw_m": "hnsw_m",
    "ef_construction": "hnsw_ef_construction",
    "nlist": "nlist",
    "rabitq_bits": "rabitq_bits",
    "turbovec_bits": "turbovec_bits",
    "auto_reduction": "auto_reduction",
    "pca_dim": "pca_dim",
}
LEGACY_BUILD_GUCS = {"build_parallel_processes": "fastann.build_parallel_processes"}
LEGACY_SEARCH_GUCS = {
    "ef_search": "fastann.hnsw_ef_search",
    "max_scan_points": "fastann.hnsw_max_scan_points",
    "quantize_rescore_amp": "fastann.quantize_rescore_amp",
    "nova_adaptive_gamma": "fastann.nova_adaptive_gamma",
    "index_scan_mode": "fastann.index_scan_mode",
    "nprobe": "fastann.nova_nprobe",
}


class AdbpgConfigDict(TypedDict):
    """Keys passed directly to psycopg.connect()."""

    user: str
    password: str
    host: str
    port: int
    dbname: str


class AdbpgConfig(DBConfig):
    user_name: SecretStr = SecretStr("tester")
    password: SecretStr = SecretStr("")
    host: str = "localhost"
    port: int = 5432
    db_name: str = "postgres"
    master_only: bool = False

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value() if isinstance(self.user_name, SecretStr) else self.user_name
        pwd_str = self.password.get_secret_value()
        result = {
            "table_name": "vector",
            "connect_config": {
                "host": self.host,
                "port": self.port,
                "dbname": self.db_name,
                "user": user_str,
                "password": pwd_str,
            },
        }
        if self.master_only:
            result["connect_config"]["options"] = "-c gp_session_role=utility"
        return result


def parse_adbpg_parameter_group(value: Any, field_name: str) -> Mapping[str, Any]:
    """Parse a structured CLI/UI value while preserving YAML scalar types."""
    if isinstance(value, str):
        try:
            value = literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                value = safe_load(value)
            except Exception as exc:
                msg = f"{field_name} must be valid YAML or JSON"
                raise ValueError(msg) from exc
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a mapping"
        raise ValueError(msg)  # noqa: TRY004 - Pydantic validators must raise ValueError.
    return value


class AdbpgParameterGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reloption: dict[str, AdbpgParameterValue] = Field(default_factory=dict)
    guc: dict[str, AdbpgParameterValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse_mapping(cls, value: Any) -> Any:
        return parse_adbpg_parameter_group(value, cls.__name__)

    @field_validator("reloption", "guc", mode="before")
    @classmethod
    def validate_parameter_map(cls, value: Any, info) -> Any:  # noqa: ANN001
        if not isinstance(value, Mapping):
            msg = f"{info.field_name} must be a mapping"
            raise ValueError(msg)  # noqa: TRY004 - Pydantic validators must raise ValueError.
        for name, setting in value.items():
            if not isinstance(name, str) or not name:
                msg = f"{info.field_name} parameter names must be non-empty strings"
                raise ValueError(msg)
            if setting is not None and not isinstance(setting, (str, int, float, bool)):
                msg = f"{info.field_name}.{name} must be a YAML scalar"
                raise ValueError(msg)
        return dict(value)


class AdbpgBuildParameters(AdbpgParameterGroup):
    reloption: dict[str, AdbpgParameterValue] = Field(default_factory=DEFAULT_BUILD_RELOPTIONS.copy)
    guc: dict[str, AdbpgParameterValue] = Field(default_factory=DEFAULT_BUILD_GUCS.copy)

    @model_validator(mode="after")
    def merge_defaults(self):
        defaults = DEFAULT_BUILD_RELOPTIONS.copy()
        if self.reloption.get("algorithm", defaults["algorithm"]) != "novamt":
            defaults.pop("turbovec_bits")
        self.reloption = {**defaults, **self.reloption}
        self.guc = {**DEFAULT_BUILD_GUCS, **self.guc}
        return self


class AdbpgAutotuneParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(default=1000, ge=0)
    n_trials: int = Field(default=200, ge=1)
    n_threads: int = Field(default=8, ge=1)
    nova_autotune_topk: list[int] | None = None
    nova_autotune_recall: list[float] | None = None

    @model_validator(mode="after")
    def validate_grid(self):
        if (self.nova_autotune_topk is None) != (self.nova_autotune_recall is None):
            raise ValueError("AutoTune grid requires both nova_autotune_topk and nova_autotune_recall")
        if self.nova_autotune_topk is not None:
            if not self.nova_autotune_topk or any(k not in (10, 100) for k in self.nova_autotune_topk):
                raise ValueError("AutoTune grid TopK must contain 10 or 100")
            if not self.nova_autotune_recall or any(not 0.90 <= r <= 1.0 for r in self.nova_autotune_recall):
                raise ValueError("AutoTune grid Recall must be between 0.90 and 1.0")
            self.nova_autotune_topk = sorted(set(self.nova_autotune_topk))
            self.nova_autotune_recall = sorted(set(self.nova_autotune_recall))
        return self

    @model_validator(mode="before")
    @classmethod
    def parse_mapping(cls, value: Any) -> Any:
        value = parse_adbpg_parameter_group(value, cls.__name__)
        if "enable" in value:
            msg = "omit autotune_parameters to disable autotune; enable is not supported"
            raise ValueError(msg)
        return value


class AdbpgIndexConfig(BaseModel, DBCaseConfig):
    model_config = ConfigDict(extra="forbid")

    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True
    build_parameters: AdbpgBuildParameters = Field(default_factory=AdbpgBuildParameters)
    search_parameters: AdbpgParameterGroup = Field(default_factory=AdbpgParameterGroup)
    autotune_parameters: AdbpgAutotuneParameters | None = None

    @model_validator(mode="after")
    def validate_index_creation(self):
        if not (self.create_index_before_load or self.create_index_after_load):
            raise ValueError("Enable create_index_before_load or create_index_after_load")
        return self

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_parameters(cls, value: Any) -> Any:
        """Keep existing UI payloads working while accepting structured config."""
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        cls._move_legacy_parameters(data, "build_parameters", "reloption", LEGACY_BUILD_RELOPTIONS)
        cls._move_legacy_parameters(data, "build_parameters", "guc", LEGACY_BUILD_GUCS)
        cls._move_legacy_parameters(data, "search_parameters", "guc", LEGACY_SEARCH_GUCS)
        return data

    @staticmethod
    def _move_legacy_parameters(
        data: dict[str, Any],
        group_name: str,
        parameter_type: str,
        mapping: Mapping[str, str],
    ) -> None:
        legacy = {name: data[name] for name in mapping if name in data}
        if not legacy:
            return

        group = data.get(group_name)
        if group is None:
            group = {}
        elif isinstance(group, BaseModel):
            group = group.model_dump()
        elif isinstance(group, Mapping):
            group = dict(group)
        else:
            return

        parameters = group.get(parameter_type)
        if parameters is None:
            parameters = {}
        elif isinstance(parameters, Mapping):
            parameters = dict(parameters)
        else:
            return

        for name, target in mapping.items():
            if name in legacy:
                parameters.setdefault(target, legacy[name])
                data.pop(name)
        group[parameter_type] = parameters
        data[group_name] = group

    @field_validator("autotune_parameters", mode="before")
    @classmethod
    def empty_autotune_is_disabled(cls, value: Any) -> Any:
        return None if value == "" else value

    @property
    def algorithm(self) -> str:
        value = self.build_parameters.reloption["algorithm"]
        return str(value)

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.IP:
            return "ip"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            **self.build_parameters.model_dump(mode="json"),
        }

    def search_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            **self.search_parameters.model_dump(mode="json"),
        }
