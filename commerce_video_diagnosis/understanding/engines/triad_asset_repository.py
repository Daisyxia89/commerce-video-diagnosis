from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:  # pragma: no cover - import success depends on runtime image
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:  # pragma: no cover
    pymysql = None
    DictCursor = None


SCHEMA_SQLITE_PATH = Path(__file__).parent / "sql" / "triad_assets_schema.sql"
SCHEMA_MYSQL_PATH = Path(__file__).parent / "sql" / "triad_assets_schema_mysql.sql"
SUPPORTED_DB_ENGINES = {"sqlite", "mysql"}

PRODUCT_SNAPSHOT_HASH_FIELDS = [
    "source_product_id",
    "leaf_category_id",
    "leaf_category_name",
    "product_name",
    "brand_name",
    "shop_name",
    "brand_asset_level",
    "price_band",
    "price_source",
    "financial_risk_level",
    "core_jtbd",
    "trust_barrier_level",
    "cognitive_barrier_level",
    "habit_switch_barrier_level",
    "diagnosis_version",
]


class TriadAssetPersistenceError(RuntimeError):
    """Physical persistence failed and must crash early."""


@dataclass(frozen=True)
class MySQLConnectionConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"
    connect_timeout: int = 5
    read_timeout: int = 10
    write_timeout: int = 10

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "MySQLConnectionConfig":
        host = str(payload.get("host") or "").strip()
        user = str(payload.get("user") or "").strip()
        password = str(payload.get("password") or "")
        database = str(payload.get("database") or payload.get("db") or "").strip()
        if not host or not user or not database:
            raise TriadAssetPersistenceError("MySQL 连接配置缺失：host/user/database 必填")
        port = _coerce_positive_int(payload.get("port"), field_name="port", default=3306)
        connect_timeout = _coerce_positive_int(payload.get("connect_timeout"), field_name="connect_timeout", default=5)
        read_timeout = _coerce_positive_int(payload.get("read_timeout"), field_name="read_timeout", default=10)
        write_timeout = _coerce_positive_int(payload.get("write_timeout"), field_name="write_timeout", default=10)
        charset = str(payload.get("charset") or "utf8mb4").strip() or "utf8mb4"
        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset=charset,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

    def safe_dsn(self) -> str:
        return f"mysql://{self.user}:***@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class ProductSnapshotWriteResult:
    product_snapshot_id: int
    snapshot_hash: str
    inserted: bool


@dataclass(frozen=True)
class BlueprintWriteResult:
    blueprint_id: str
    idempotency_key: str
    inserted: bool
    segment_count: int


@dataclass(frozen=True)
class TriadAssetCounts:
    product_snapshot_count: int
    video_blueprint_count: int
    video_segment_count: int


@dataclass(frozen=True)
class TriadAssetPersistenceSummary:
    product_snapshot_id: int
    snapshot_hash: str
    blueprint_id: str
    idempotency_key: str
    segment_count: int
    table_counts: TriadAssetCounts
    product_snapshot_inserted: bool
    blueprint_inserted: bool


class TriadAssetRepository:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        engine: Literal["sqlite", "mysql"] = "sqlite",
        mysql_config: MySQLConnectionConfig | dict[str, Any] | None = None,
    ) -> None:
        self.engine = _normalize_db_engine(engine)
        if self.engine == "sqlite":
            if db_path is None:
                raise TriadAssetPersistenceError("SQLite 落库必须提供 db_path")
            self.db_path = Path(db_path)
            self.mysql_config = None
        else:
            self.db_path = None
            if mysql_config is None:
                raise TriadAssetPersistenceError("MySQL 落库必须提供 mysql_config")
            self.mysql_config = (
                mysql_config if isinstance(mysql_config, MySQLConnectionConfig) else MySQLConnectionConfig.from_mapping(mysql_config)
            )

    @property
    def locator(self) -> str:
        if self.engine == "sqlite":
            return str(self.db_path)
        if self.mysql_config is None:
            raise TriadAssetPersistenceError("MySQL 连接配置缺失")
        return self.mysql_config.safe_dsn()

    def initialize_schema(self) -> None:
        if self.engine == "sqlite":
            if self.db_path is None:
                raise TriadAssetPersistenceError("SQLite db_path 缺失")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            schema_sql = SCHEMA_SQLITE_PATH.read_text(encoding="utf-8")
            conn = self._connect()
            try:
                conn.executescript(schema_sql)
                conn.commit()
            finally:
                conn.close()
            return

        schema_sql = SCHEMA_MYSQL_PATH.read_text(encoding="utf-8")
        conn = self._connect()
        try:
            for statement in _split_sql_statements(schema_sql):
                self._execute(conn, statement)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> Any:
        if self.engine == "sqlite":
            if self.db_path is None:
                raise TriadAssetPersistenceError("SQLite db_path 缺失")
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            return conn

        if pymysql is None or DictCursor is None:
            raise TriadAssetPersistenceError("MySQL 适配依赖缺失：请先安装 PyMySQL")
        if self.mysql_config is None:
            raise TriadAssetPersistenceError("MySQL 连接配置缺失")
        return pymysql.connect(
            host=self.mysql_config.host,
            port=self.mysql_config.port,
            user=self.mysql_config.user,
            password=self.mysql_config.password,
            database=self.mysql_config.database,
            charset=self.mysql_config.charset,
            connect_timeout=self.mysql_config.connect_timeout,
            read_timeout=self.mysql_config.read_timeout,
            write_timeout=self.mysql_config.write_timeout,
            cursorclass=DictCursor,
            autocommit=False,
        )

    def upsert_product_snapshot(
        self,
        snapshot: dict[str, Any],
        provenance: dict[str, Any],
        *,
        created_at: str | None = None,
    ) -> ProductSnapshotWriteResult:
        self.initialize_schema()
        snapshot_hash = build_product_snapshot_hash(snapshot)
        now = created_at or _nullable_text(snapshot.get("created_at")) or _utc_now_text()
        columns = [
            "source_product_id",
            "leaf_category_id",
            "leaf_category_name",
            "product_name",
            "brand_name",
            "shop_name",
            "brand_asset_level",
            "price_band",
            "price_source",
            "financial_risk_level",
            "core_jtbd",
            "trust_barrier_level",
            "cognitive_barrier_level",
            "habit_switch_barrier_level",
            "diagnosis_version",
            "diagnosis_generated_at",
            "snapshot_hash",
            "provenance_json",
            "created_at",
        ]
        values = (
            _required_text(snapshot, "source_product_id"),
            _required_text(snapshot, "leaf_category_id"),
            _required_text(snapshot, "leaf_category_name"),
            _required_text(snapshot, "product_name"),
            _required_text(snapshot, "brand_name"),
            _nullable_text(snapshot.get("shop_name")),
            _required_text(snapshot, "brand_asset_level"),
            _required_text(snapshot, "price_band"),
            _required_text(snapshot, "price_source"),
            _required_text(snapshot, "financial_risk_level"),
            _required_text(snapshot, "core_jtbd"),
            _required_text(snapshot, "trust_barrier_level"),
            _required_text(snapshot, "cognitive_barrier_level"),
            _required_text(snapshot, "habit_switch_barrier_level"),
            _required_text(snapshot, "diagnosis_version"),
            _required_text(snapshot, "diagnosis_generated_at"),
            snapshot_hash,
            _json_text(provenance),
            now,
        )
        insert_sql = (
            f"INSERT OR IGNORE INTO product_master_snapshot ({', '.join(columns)}) VALUES ({self._placeholders(len(columns))})"
            if self.engine == "sqlite"
            else f"INSERT IGNORE INTO product_master_snapshot ({', '.join(columns)}) VALUES ({self._placeholders(len(columns))})"
        )
        conn = self._connect()
        try:
            rowcount = self._execute(conn, insert_sql, values)
            row = self._fetchone(
                conn,
                f"SELECT product_snapshot_id, snapshot_hash FROM product_master_snapshot WHERE source_product_id = {self._placeholder()} AND snapshot_hash = {self._placeholder()}",
                (_required_text(snapshot, "source_product_id"), snapshot_hash),
            )
            conn.commit()
        finally:
            conn.close()
        if row is None:
            raise TriadAssetPersistenceError("product_master_snapshot upsert 后未查询到记录")
        return ProductSnapshotWriteResult(
            product_snapshot_id=int(row["product_snapshot_id"]),
            snapshot_hash=str(row["snapshot_hash"]),
            inserted=rowcount == 1,
        )

    def persist_blueprint_with_segments(
        self,
        *,
        product_snapshot_id: int,
        request_id: str,
        video_id: str,
        source_product_id: str,
        generator_version: str,
        workflow_version: str,
        blueprint: dict[str, Any],
        segment_records: list[dict[str, Any]],
        created_at: str | None = None,
    ) -> BlueprintWriteResult:
        self.initialize_schema()
        blueprint_id = _required_text(blueprint, "blueprint_id")
        idempotency_key = build_blueprint_idempotency_key(
            request_id=request_id,
            video_id=video_id,
            source_product_id=source_product_id,
            product_snapshot_id=product_snapshot_id,
            blueprint=blueprint,
            segment_records=segment_records,
            generator_version=generator_version,
            workflow_version=workflow_version,
        )
        now = created_at or _nullable_text(blueprint.get("created_at")) or _utc_now_text()
        conn = self._connect()
        try:
            self._begin(conn)
            existing = self._fetchone(
                conn,
                f"SELECT blueprint_id FROM video_blueprint_master WHERE idempotency_key = {self._placeholder()}",
                (idempotency_key,),
            )
            if existing is None:
                existing = self._fetchone(
                    conn,
                    f"SELECT blueprint_id, idempotency_key FROM video_blueprint_master WHERE blueprint_id = {self._placeholder()}",
                    (blueprint_id,),
                )
                if existing is not None and str(existing["idempotency_key"]) != idempotency_key:
                    raise TriadAssetPersistenceError("blueprint_id 已存在但 idempotency_key 不一致，拒绝覆盖历史蓝图")
            if existing is not None:
                existing_blueprint_id = str(existing["blueprint_id"])
                segment_count = int(
                    self._fetchone(
                        conn,
                        f"SELECT COUNT(1) AS cnt FROM video_segment_fact_table WHERE blueprint_id = {self._placeholder()}",
                        (existing_blueprint_id,),
                    )["cnt"]
                )
                conn.commit()
                return BlueprintWriteResult(
                    blueprint_id=existing_blueprint_id,
                    idempotency_key=idempotency_key,
                    inserted=False,
                    segment_count=segment_count,
                )

            blueprint_columns = [
                "blueprint_id",
                "video_id",
                "source_product_id",
                "product_snapshot_id",
                "request_id",
                "generator_version",
                "workflow_version",
                "storyboard_source",
                "semantic_bundle_count",
                "primary_hec_json",
                "secondary_effects_json",
                "slider_signature_json",
                "risk_flags_json",
                "semantic_bundles_json",
                "segment_to_bundle_map_json",
                "bundle_to_segment_range_json",
                "provenance_json",
                "idempotency_key",
                "created_at",
            ]
            blueprint_values = (
                blueprint_id,
                video_id,
                source_product_id,
                product_snapshot_id,
                request_id,
                generator_version,
                workflow_version,
                _required_text(blueprint, "storyboard_source"),
                int(blueprint.get("semantic_bundle_count") or len(list(blueprint.get("semantic_bundles") or []))),
                _json_text(blueprint.get("primary_hec") or {}),
                _json_text(list(blueprint.get("secondary_effects") or [])),
                _json_text(blueprint.get("slider_signature") or {}),
                _json_text(blueprint.get("risk_flags") or {}),
                _json_text(list(blueprint.get("semantic_bundles") or [])),
                _json_text(dict(blueprint.get("segment_to_bundle_map") or {})),
                _json_text(dict(blueprint.get("bundle_to_segment_range") or {})),
                _json_text(list(blueprint.get("provenance") or [])),
                idempotency_key,
                now,
            )
            self._execute(
                conn,
                f"INSERT INTO video_blueprint_master ({', '.join(blueprint_columns)}) VALUES ({self._placeholders(len(blueprint_columns))})",
                blueprint_values,
            )
            self._insert_segment_rows(
                conn=conn,
                blueprint_id=blueprint_id,
                video_id=video_id,
                source_product_id=source_product_id,
                segment_records=segment_records,
                created_at=now,
            )
            segment_count = int(
                self._fetchone(
                    conn,
                    f"SELECT COUNT(1) AS cnt FROM video_segment_fact_table WHERE blueprint_id = {self._placeholder()}",
                    (blueprint_id,),
                )["cnt"]
            )
            if segment_count != len(segment_records):
                raise TriadAssetPersistenceError(
                    f"video_segment_fact_table 条数不一致：expected={len(segment_records)} actual={segment_count}"
                )
            conn.commit()
            return BlueprintWriteResult(
                blueprint_id=blueprint_id,
                idempotency_key=idempotency_key,
                inserted=True,
                segment_count=segment_count,
            )
        except Exception as exc:  # pragma: no cover - exercised via caller tests
            conn.rollback()
            if isinstance(exc, TriadAssetPersistenceError):
                raise
            raise TriadAssetPersistenceError(f"video_blueprint_master / video_segment_fact_table 事务失败：{exc}") from exc
        finally:
            conn.close()

    def get_table_counts(self) -> TriadAssetCounts:
        self.initialize_schema()
        conn = self._connect()
        try:
            product_snapshot_count = int(self._fetchone(conn, "SELECT COUNT(1) AS cnt FROM product_master_snapshot")["cnt"])
            video_blueprint_count = int(self._fetchone(conn, "SELECT COUNT(1) AS cnt FROM video_blueprint_master")["cnt"])
            video_segment_count = int(self._fetchone(conn, "SELECT COUNT(1) AS cnt FROM video_segment_fact_table")["cnt"])
        finally:
            conn.close()
        return TriadAssetCounts(
            product_snapshot_count=product_snapshot_count,
            video_blueprint_count=video_blueprint_count,
            video_segment_count=video_segment_count,
        )

    def fetch_product_snapshots_by_source_product_id(self, source_product_id: str) -> list[dict[str, Any]]:
        self.initialize_schema()
        conn = self._connect()
        try:
            rows = self._fetchall(
                conn,
                f"SELECT * FROM product_master_snapshot WHERE source_product_id = {self._placeholder()} ORDER BY diagnosis_generated_at DESC, product_snapshot_id DESC",
                (source_product_id,),
            )
        finally:
            conn.close()
        return rows

    def fetch_blueprints_by_video_id(self, video_id: str) -> list[dict[str, Any]]:
        self.initialize_schema()
        conn = self._connect()
        try:
            rows = self._fetchall(
                conn,
                f"SELECT * FROM video_blueprint_master WHERE video_id = {self._placeholder()} ORDER BY created_at DESC, blueprint_id DESC",
                (video_id,),
            )
        finally:
            conn.close()
        return rows

    def fetch_segments_by_blueprint_id(self, blueprint_id: str) -> list[dict[str, Any]]:
        self.initialize_schema()
        conn = self._connect()
        try:
            rows = self._fetchall(
                conn,
                f"SELECT * FROM video_segment_fact_table WHERE blueprint_id = {self._placeholder()} ORDER BY segment_order ASC, segment_record_id ASC",
                (blueprint_id,),
            )
        finally:
            conn.close()
        return rows

    def fetch_all_product_snapshots(self) -> list[dict[str, Any]]:
        self.initialize_schema()
        conn = self._connect()
        try:
            rows = self._fetchall(
                conn,
                "SELECT * FROM product_master_snapshot ORDER BY product_snapshot_id ASC",
            )
        finally:
            conn.close()
        return rows

    def fetch_all_blueprints(self) -> list[dict[str, Any]]:
        self.initialize_schema()
        conn = self._connect()
        try:
            rows = self._fetchall(
                conn,
                "SELECT * FROM video_blueprint_master ORDER BY created_at ASC, blueprint_id ASC",
            )
        finally:
            conn.close()
        return rows

    def _insert_segment_rows(
        self,
        *,
        conn: Any,
        blueprint_id: str,
        video_id: str,
        source_product_id: str,
        segment_records: list[dict[str, Any]],
        created_at: str,
    ) -> None:
        columns = [
            "segment_record_id",
            "blueprint_id",
            "video_id",
            "source_product_id",
            "segment_id",
            "segment_order",
            "start_sec",
            "end_sec",
            "bundle_id",
            "shot_size",
            "camera_movement",
            "lighting_tone",
            "visual_subject",
            "key_objects_json",
            "actions_json",
            "ocr_facts_json",
            "audio_facts_json",
            "rhythm_facts_json",
            "annotation_json",
            "provenance_json",
            "created_at",
        ]
        sql = f"INSERT INTO video_segment_fact_table ({', '.join(columns)}) VALUES ({self._placeholders(len(columns))})"
        for index, record in enumerate(segment_records):
            self._execute(
                conn,
                sql,
                (
                    _required_text(record, "segment_record_id"),
                    blueprint_id,
                    video_id,
                    source_product_id,
                    _required_text(record, "segment_id"),
                    int(record.get("segment_order") if record.get("segment_order") is not None else index),
                    float(record.get("start_sec") or 0.0),
                    float(record.get("end_sec") or 0.0),
                    _nullable_text(record.get("bundle_id")),
                    _required_text(record, "shot_size"),
                    _required_text(record, "camera_movement"),
                    _required_text(record, "lighting_tone"),
                    _required_text(record, "visual_subject"),
                    _json_text(list(record.get("key_objects") or [])),
                    _json_text(list(record.get("actions") or [])),
                    _json_text(list(record.get("ocr_facts") or [])),
                    _json_text(dict(record.get("audio_facts") or {})),
                    _json_text(dict(record.get("rhythm_facts") or {})),
                    _json_text(dict(record.get("annotation") or {})),
                    _json_text(dict(record.get("provenance") or {})),
                    _nullable_text(record.get("created_at")) or created_at,
                ),
            )

    def _placeholder(self) -> str:
        return "?" if self.engine == "sqlite" else "%s"

    def _placeholders(self, count: int) -> str:
        return ", ".join([self._placeholder()] * count)

    def _begin(self, conn: Any) -> None:
        if self.engine == "sqlite":
            conn.execute("BEGIN")
            return
        conn.begin()

    def _execute(self, conn: Any, sql: str, params: tuple[Any, ...] = ()) -> int:
        if self.engine == "sqlite":
            cursor = conn.execute(sql, params)
            return int(cursor.rowcount)
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return int(cursor.rowcount)

    def _fetchone(self, conn: Any, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        if self.engine == "sqlite":
            row = conn.execute(sql, params).fetchone()
            return None if row is None else _row_to_plain_dict(row)
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
        return None if row is None else _row_to_plain_dict(row)

    def _fetchall(self, conn: Any, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if self.engine == "sqlite":
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_plain_dict(row) for row in rows]
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [_row_to_plain_dict(row) for row in rows]


class MySQLTriadAssetRepository(TriadAssetRepository):
    def __init__(self, mysql_config: MySQLConnectionConfig | dict[str, Any]) -> None:
        super().__init__(engine="mysql", mysql_config=mysql_config)


def build_product_snapshot_hash(snapshot: dict[str, Any]) -> str:
    payload = {field: snapshot.get(field) for field in PRODUCT_SNAPSHOT_HASH_FIELDS}
    return _sha256_hex(payload)


def build_blueprint_idempotency_key(
    *,
    request_id: str,
    video_id: str,
    source_product_id: str,
    product_snapshot_id: int,
    blueprint: dict[str, Any],
    segment_records: list[dict[str, Any]],
    generator_version: str,
    workflow_version: str,
) -> str:
    payload = {
        "request_id": request_id,
        "video_id": video_id,
        "source_product_id": source_product_id,
        "product_snapshot_id": product_snapshot_id,
        "generator_version": generator_version,
        "workflow_version": workflow_version,
        "storyboard_source": blueprint.get("storyboard_source"),
        "semantic_bundle_count": blueprint.get("semantic_bundle_count") or len(list(blueprint.get("semantic_bundles") or [])),
        "segment_ids": [record.get("segment_id") for record in segment_records],
        "segment_count": len(segment_records),
    }
    return _sha256_hex(payload)


def build_blueprint_id(*, request_id: str, video_id: str, source_product_id: str) -> str:
    return f"BP_{_sha256_hex({'request_id': request_id, 'video_id': video_id, 'source_product_id': source_product_id})[:10]}"


def build_segment_record_id(*, blueprint_id: str, segment_id: str) -> str:
    return f"SEGREC_{_sha256_hex({'blueprint_id': blueprint_id, 'segment_id': segment_id})[:10]}"


def _normalize_db_engine(engine: Any) -> Literal["sqlite", "mysql"]:
    normalized = str(engine or "sqlite").strip().lower()
    if normalized not in SUPPORTED_DB_ENGINES:
        raise TriadAssetPersistenceError(f"不支持的 triad_assets db engine：{engine}")
    return normalized  # type: ignore[return-value]


def _split_sql_statements(schema_sql: str) -> list[str]:
    statements: list[str] = []
    for chunk in schema_sql.split(";"):
        statement = chunk.strip()
        if statement:
            statements.append(statement)
    return statements


def _coerce_positive_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TriadAssetPersistenceError(f"MySQL 连接配置非法：{field_name} 必须是正整数") from exc
    if parsed <= 0:
        raise TriadAssetPersistenceError(f"MySQL 连接配置非法：{field_name} 必须是正整数")
    return parsed


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _row_to_plain_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        items = [(key, row[key]) for key in row.keys()]
    elif isinstance(row, dict):
        items = list(row.items())
    else:
        raise TriadAssetPersistenceError(f"未知数据库行类型：{type(row)}")

    result: dict[str, Any] = {}
    for key, value in items:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if str(key).endswith("_json") and isinstance(value, str):
            result[str(key)] = json.loads(value)
        else:
            result[str(key)] = value
    return result


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise TriadAssetPersistenceError(f"字段缺失：{key}")
    return value


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _utc_now_text() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
