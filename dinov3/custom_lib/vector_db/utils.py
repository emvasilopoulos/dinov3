from __future__ import annotations

import pathlib
import sqlite3
from typing import Any

from . import schema


METADATA_COLUMNS = ("product_name", "brand", "category", "subcategory", "image_name")


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def connect_readonly(db_path: str | pathlib.Path) -> sqlite3.Connection:
    path = pathlib.Path(db_path)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def table_names(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def database_summary(connection: sqlite3.Connection, db_path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(db_path)
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    return {
        "path": path.resolve(),
        "file_size": path.stat().st_size,
        "page_count": page_count,
        "page_size": page_size,
        "freelist_count": int(connection.execute("PRAGMA freelist_count").fetchone()[0]),
        "user_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
        "journal_mode": str(connection.execute("PRAGMA journal_mode").fetchone()[0]),
    }


def table_summaries(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    summaries = []
    for name in table_names(connection):
        quoted_name = quote_identifier(name)
        count = connection.execute(f"SELECT COUNT(*) AS total FROM {quoted_name}").fetchone()
        columns = connection.execute(f"PRAGMA table_info({quoted_name})").fetchall()
        summaries.append(
            {
                "name": name,
                "record_count": int(count["total"]),
                "columns": columns,
                "column_names": tuple(str(column["name"]) for column in columns),
            }
        )
    return summaries


def vector_table_exists(connection: sqlite3.Connection) -> bool:
    return schema.VECTOR_SCHEMA.name in set(table_names(connection))


def vector_metrics(connection: sqlite3.Connection) -> dict[str, Any]:
    vector_table = schema.VECTOR_SCHEMA.name
    if not vector_table_exists(connection):
        return {"table_exists": False, "table_name": vector_table}

    quoted_table = quote_identifier(vector_table)
    total = connection.execute(schema.COUNT_VECTORS_SQL).fetchone()
    dimensions = connection.execute(
        f"""
        SELECT dimensions, COUNT(*) AS total
        FROM {quoted_table}
        GROUP BY dimensions
        ORDER BY total DESC, dimensions
        """
    ).fetchall()
    dtypes = connection.execute(
        f"""
        SELECT dtype, COUNT(*) AS total
        FROM {quoted_table}
        GROUP BY dtype
        ORDER BY total DESC, dtype
        """
    ).fetchall()
    blob_stats = connection.execute(
        f"""
        SELECT
            MIN(LENGTH(vector)) AS min_bytes,
            AVG(LENGTH(vector)) AS avg_bytes,
            MAX(LENGTH(vector)) AS max_bytes
        FROM {quoted_table}
        """
    ).fetchone()
    byte_mismatches = connection.execute(
        f"""
        SELECT COUNT(*) AS total
        FROM {quoted_table}
        WHERE dtype = 'float32'
          AND LENGTH(vector) != dimensions * 4
        """
    ).fetchone()
    distinct_values = [
        f"COUNT(DISTINCT {quote_identifier(column)}) AS {quote_identifier(column)}"
        for column in METADATA_COLUMNS
    ]
    distinct_metadata = connection.execute(f"SELECT {', '.join(distinct_values)} FROM {quoted_table}").fetchone()
    missing_metadata = {}
    for column in METADATA_COLUMNS:
        quoted_column = quote_identifier(column)
        row = connection.execute(
            f"""
            SELECT COUNT(*) AS total
            FROM {quoted_table}
            WHERE {quoted_column} IS NULL
               OR TRIM({quoted_column}) = ''
            """
        ).fetchone()
        missing_metadata[column] = int(row["total"])

    duplicate_images = connection.execute(
        f"""
        SELECT image_name, COUNT(*) AS total
        FROM {quoted_table}
        WHERE image_name IS NOT NULL
          AND TRIM(image_name) != ''
        GROUP BY image_name
        HAVING COUNT(*) > 1
        ORDER BY total DESC, image_name
        LIMIT 10
        """
    ).fetchall()
    return {
        "table_exists": True,
        "table_name": vector_table,
        "record_count": int(total["total"]),
        "dimensions": dimensions,
        "dtypes": dtypes,
        "blob_stats": blob_stats,
        "float32_byte_mismatches": int(byte_mismatches["total"]),
        "distinct_metadata": {
            column: int(distinct_metadata[column]) for column in METADATA_COLUMNS
        },
        "missing_metadata": missing_metadata,
        "duplicate_images": duplicate_images,
    }


def sample_vector_rows(connection: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    if limit <= 0 or not vector_table_exists(connection):
        return []

    return connection.execute(
        f"""
        SELECT
            id,
            dimensions,
            dtype,
            LENGTH(vector) AS vector_bytes,
            product_name,
            brand,
            category,
            subcategory,
            image_name
        FROM {quote_identifier(schema.VECTOR_SCHEMA.name)}
        ORDER BY id
        LIMIT :sample_limit
        """,
        {"sample_limit": limit},
    ).fetchall()
