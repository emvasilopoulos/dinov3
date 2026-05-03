from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys
from collections.abc import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov3.custom_lib.vector_db import utils


DEFAULT_DB_PATH = "backbar_ai_products_embeddings.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print schema and data quality metrics for a SQLite vector database."
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path to inspect. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of sample vector rows to print. Defaults to 5.",
    )
    return parser


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_rows(rows: Iterable[sqlite3.Row], columns: tuple[str, ...]) -> None:
    for row in rows:
        print("  " + ", ".join(f"{column}={row[column]}" for column in columns))


def print_database_summary(connection: sqlite3.Connection, db_path: pathlib.Path) -> None:
    summary = utils.database_summary(connection, db_path)

    print_section("Database")
    print(f"Path: {summary['path']}")
    print(f"File size: {summary['file_size']:,} bytes")
    print(f"SQLite pages: {summary['page_count']:,} x {summary['page_size']:,} bytes")
    print(f"Free pages: {summary['freelist_count']:,}")
    print(f"User version: {summary['user_version']}")
    print(f"Journal mode: {summary['journal_mode']}")


def print_table_summary(connection: sqlite3.Connection) -> None:
    print_section("Tables")
    summaries = utils.table_summaries(connection)
    if not summaries:
        print("No application tables found.")
        return

    for summary in summaries:
        print(f"{summary['name']}: {summary['record_count']:,} records")
        print(f"  Columns: {', '.join(summary['column_names'])}")
        for column in summary["columns"]:
            primary_key = " primary_key" if column["pk"] else ""
            required = " required" if column["notnull"] else ""
            default = f" default={column['dflt_value']}" if column["dflt_value"] is not None else ""
            print(
                f"  - {column['name']}: {column['type'] or 'untyped'}"
                f"{required}{primary_key}{default}"
            )


def print_vector_metrics(connection: sqlite3.Connection) -> None:
    print_section("Vector Metrics")
    metrics = utils.vector_metrics(connection)
    if not metrics["table_exists"]:
        print(f"Table not found: {metrics['table_name']}")
        return

    print(f"Records: {metrics['record_count']:,}")
    print("Dimensions:")
    print_rows(metrics["dimensions"], ("dimensions", "total"))

    print("Dtypes:")
    print_rows(metrics["dtypes"], ("dtype", "total"))

    blob_stats = metrics["blob_stats"]
    print(
        "Vector blob bytes: "
        f"min={blob_stats['min_bytes']}, "
        f"avg={float(blob_stats['avg_bytes'] or 0):.2f}, "
        f"max={blob_stats['max_bytes']}"
    )

    print(f"Float32 byte-size mismatches: {metrics['float32_byte_mismatches']:,}")

    print("Distinct metadata values:")
    for column, count in metrics["distinct_metadata"].items():
        print(f"  {column}: {count:,}")

    print("Missing metadata values:")
    for column, count in metrics["missing_metadata"].items():
        print(f"  {column}: {count:,}")

    print("Duplicate image names:")
    if metrics["duplicate_images"]:
        print_rows(metrics["duplicate_images"], ("image_name", "total"))
    else:
        print("  None")


def print_sample_rows(connection: sqlite3.Connection, sample_limit: int) -> None:
    if sample_limit <= 0:
        return

    print_section("Sample Rows")
    rows = utils.sample_vector_rows(connection, sample_limit)
    if not rows:
        print("No rows to sample.")
        return

    print_rows(
        rows,
        (
            "id",
            "dimensions",
            "dtype",
            "vector_bytes",
            "product_name",
            "brand",
            "category",
            "subcategory",
            "image_name",
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    db_path = pathlib.Path(args.db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    connection = utils.connect_readonly(db_path)
    try:
        print_database_summary(connection, db_path)
        print_table_summary(connection)
        print_vector_metrics(connection)
        print_sample_rows(connection, args.sample_limit)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
