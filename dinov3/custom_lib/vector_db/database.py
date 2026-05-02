from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable, Mapping, Sequence

import sqlite3
import numpy as np

from . import schema

SqlValue = str | int | bytes | None
VectorMetadata = Mapping[str, str | None]


@dataclasses.dataclass(frozen=True)
class QueryResult:
    id: str
    score: float
    vector: np.ndarray
    metadata: dict[str, str | None] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: np.ndarray
    metadata: dict[str, str | None]


class VectorDatabase:
    def __init__(self, path: str | pathlib.Path):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        for table in schema.DATABASE_SCHEMA:
            self.connection.execute(table.create_statement())
            self._add_missing_columns(table)
        self.connection.commit()

    def _add_missing_columns(self, table: schema.TableSchema) -> None:
        existing_columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table.name})")
        }
        for column in table.columns:
            if column.name not in existing_columns:
                self.connection.execute(
                    f"ALTER TABLE {table.name} ADD COLUMN {column.definition}"
                )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "VectorDatabase":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def add_vector(
        self,
        vector_id: str,
        vector: Sequence[float] | np.ndarray,
        metadata: VectorMetadata | None = None,
        *,
        product_name: str | None = None,
        brand: str | None = None,
        category: str | None = None,
        subcategory: str | None = None,
        image_name: str | None = None,
    ) -> None:
        array = self._normalize_vector(vector)
        self.connection.execute(
            schema.INSERT_VECTOR_SQL,
            self._vector_row(
                vector_id,
                array,
                metadata,
                product_name=product_name,
                brand=brand,
                category=category,
                subcategory=subcategory,
                image_name=image_name,
            ),
        )
        self.connection.commit()

    def add_vectors(
        self,
        items: Iterable[
            tuple[str, Sequence[float] | np.ndarray]
            | tuple[str, Sequence[float] | np.ndarray, VectorMetadata]
        ],
    ) -> None:
        rows = []
        for item in items:
            vector_id, vector, metadata = self._unpack_vector_item(item)
            array = self._normalize_vector(vector)
            rows.append(self._vector_row(vector_id, array, metadata))

        self.connection.executemany(schema.INSERT_VECTOR_SQL, rows)
        self.connection.commit()

    def get_vector(self, vector_id: str) -> np.ndarray | None:
        record = self.get_record(vector_id)
        if record is None:
            return None
        return record.vector

    def get_record(self, vector_id: str) -> VectorRecord | None:
        row = self.connection.execute(
            schema.SELECT_VECTOR_SQL,
            {schema.VECTOR_SCHEMA.primary_key: vector_id},
        ).fetchone()
        if row is None:
            return None
        return VectorRecord(
            id=vector_id,
            vector=self._deserialize_vector(row),
            metadata=self._metadata_from_row(row),
        )

    def delete_vector(self, vector_id: str) -> None:
        self.connection.execute(
            schema.DELETE_VECTOR_SQL,
            {schema.VECTOR_SCHEMA.primary_key: vector_id},
        )
        self.connection.commit()

    def query(
        self,
        vector: Sequence[float] | np.ndarray,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[QueryResult]:
        """Search for similar vectors in the database."""
        if top_k <= 0:
            return []
        if metric not in {"cosine", "euclidean", "dot"}:
            raise ValueError(
                "Unsupported metric. Use 'cosine', 'euclidean', or 'dot'."
            )

        query_vector = self._normalize_vector(vector)
        rows = self.connection.execute(schema.SELECT_ALL_VECTORS_SQL).fetchall()

        results: list[QueryResult] = []
        for row in rows:
            candidate = self._deserialize_vector(row)
            if candidate.shape != query_vector.shape:
                continue

            if metric == "cosine":
                score = self._cosine_similarity(query_vector, candidate)
            elif metric == "euclidean":
                score = -float(np.linalg.norm(query_vector - candidate))
            else:
                score = float(np.dot(query_vector, candidate))

            results.append(
                QueryResult(
                    id=row["id"],
                    score=score,
                    vector=candidate.copy(),
                    metadata=self._metadata_from_row(row),
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def count(self) -> int:
        row = self.connection.execute(schema.COUNT_VECTORS_SQL).fetchone()
        return int(row["total"])

    def _vector_row(
        self,
        vector_id: str,
        array: np.ndarray,
        metadata: VectorMetadata | None = None,
        *,
        product_name: str | None = None,
        brand: str | None = None,
        category: str | None = None,
        subcategory: str | None = None,
        image_name: str | None = None,
    ) -> dict[str, SqlValue]:
        row: dict[str, SqlValue] = {
            "id": vector_id,
            "dimensions": array.shape[0],
            "dtype": str(array.dtype),
            "vector": array.tobytes(),
        }
        row.update({column: None for column in self._metadata_columns()})

        if metadata is not None:
            unknown_columns = set(metadata) - set(self._metadata_columns())
            if unknown_columns:
                names = ", ".join(sorted(unknown_columns))
                raise ValueError(f"Unknown metadata column(s): {names}")
            row.update(metadata)

        explicit_metadata = {
            "product_name": product_name,
            "brand": brand,
            "category": category,
            "subcategory": subcategory,
            "image_name": image_name,
        }
        row.update(
            {
                column: value
                for column, value in explicit_metadata.items()
                if value is not None
            }
        )
        return row

    def _unpack_vector_item(
        self,
        item: tuple[str, Sequence[float] | np.ndarray]
        | tuple[str, Sequence[float] | np.ndarray, VectorMetadata],
    ) -> tuple[str, Sequence[float] | np.ndarray, VectorMetadata | None]:
        if len(item) == 2:
            vector_id, vector = item
            return vector_id, vector, None
        if len(item) == 3:
            vector_id, vector, metadata = item
            return vector_id, vector, metadata
        raise ValueError("Vector items must be (id, vector) or (id, vector, metadata).")

    def _normalize_vector(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("Vectors must be one-dimensional.")
        if array.size == 0:
            raise ValueError("Vectors cannot be empty.")
        return np.ascontiguousarray(array)

    def _deserialize_vector(self, row: sqlite3.Row) -> np.ndarray:
        array = np.frombuffer(row["vector"], dtype=np.dtype(row["dtype"]))
        return array.reshape((row["dimensions"],))

    def _metadata_from_row(self, row: sqlite3.Row) -> dict[str, str | None]:
        return {column: row[column] for column in self._metadata_columns()}

    def _metadata_columns(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in schema.VECTOR_PAYLOAD_COLUMNS
            if column not in {"dimensions", "dtype", "vector"}
        )

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
