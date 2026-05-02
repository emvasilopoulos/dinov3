from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Column:
    name: str
    definition: str


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[Column, ...]
    primary_key: str

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def non_key_column_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.column_names if name != self.primary_key)

    def create_statement(self) -> str:
        column_definitions = ",\n                ".join(
            column.definition for column in self.columns
        )
        return f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                {column_definitions}
            )
            """

    def upsert_statement(self) -> str:
        insert_columns = ", ".join(self.column_names)
        insert_values = ", ".join(f":{column}" for column in self.column_names)
        update_values = ", ".join(
            f"{column} = excluded.{column}" for column in self.non_key_column_names
        )
        return f"""
            INSERT INTO {self.name} ({insert_columns})
            VALUES ({insert_values})
            ON CONFLICT({self.primary_key}) DO UPDATE SET
                {update_values}
            """

    def select_statement(self, columns: Sequence[str], *, where: str | None = None) -> str:
        statement = f"SELECT {', '.join(columns)} FROM {self.name}"
        if where is not None:
            statement = f"{statement} WHERE {where}"
        return statement

    def delete_statement(self, *, where: str) -> str:
        return f"DELETE FROM {self.name} WHERE {where}"

    def count_statement(self) -> str:
        return f"SELECT COUNT(*) AS total FROM {self.name}"


VECTOR_SCHEMA = TableSchema(
    name="vectors",
    primary_key="id",
    columns=(
        Column("id", "id TEXT PRIMARY KEY"),
        Column("dimensions", "dimensions INTEGER NOT NULL"),
        Column("dtype", "dtype TEXT NOT NULL"),
        Column("vector", "vector BLOB NOT NULL"),
        Column("product_name", "product_name TEXT"),
        Column("brand", "brand TEXT"),
        Column("category", "category TEXT"),
        Column("subcategory", "subcategory TEXT"),
        Column("image_name", "image_name TEXT"),
    ),
)
DATABASE_SCHEMA = (VECTOR_SCHEMA,)

VECTOR_SELECT_COLUMNS = VECTOR_SCHEMA.column_names
VECTOR_PAYLOAD_COLUMNS = VECTOR_SCHEMA.non_key_column_names

INSERT_VECTOR_SQL = VECTOR_SCHEMA.upsert_statement()
SELECT_VECTOR_SQL = VECTOR_SCHEMA.select_statement(
    VECTOR_PAYLOAD_COLUMNS,
    where=f"{VECTOR_SCHEMA.primary_key} = :{VECTOR_SCHEMA.primary_key}",
)
SELECT_ALL_VECTORS_SQL = VECTOR_SCHEMA.select_statement(VECTOR_SELECT_COLUMNS)
DELETE_VECTOR_SQL = VECTOR_SCHEMA.delete_statement(
    where=f"{VECTOR_SCHEMA.primary_key} = :{VECTOR_SCHEMA.primary_key}"
)
COUNT_VECTORS_SQL = VECTOR_SCHEMA.count_statement()