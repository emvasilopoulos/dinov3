from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable, Sequence

import sqlite3
import numpy as np
import numpy.typing as npt

from . import schema

SqlValue = str | int | bytes

@dataclasses.dataclass(frozen=True)
class QueryResult:
    id: str
    score: float
    vector: np.ndarray
    product_name: str
    brand: str
    category: str
    subcategory: str
    image_name: str


@dataclasses.dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: np.ndarray
    product_name: str
    brand: str
    category: str
    subcategory: str
    image_name: str


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
        product_name: str,
        brand: str,
        category: str,
        subcategory: str,
        image_name: str,
    ) -> None:
        array = self._normalize_vector(vector)
        self.connection.execute(
            schema.INSERT_VECTOR_SQL,
            self._vector_row(
                vector_id,
                array,
                product_name=product_name,
                brand=brand,
                category=category,
                subcategory=subcategory,
                image_name=image_name,
            ),
        )
        self.connection.commit()

    def _check_item(
        self,
        item: VectorRecord,
    ) -> None:
        if not isinstance(item, VectorRecord):
            raise ValueError("Each item must be a VectorRecord instance.")
        if not isinstance(item.id, str):
            raise ValueError("Vector ID must be a string.")
        if not isinstance(item.vector, np.ndarray):
            raise ValueError("Vector must be a numpy array.")
        if item.vector.dtype != np.float32:
            raise ValueError("Vector dtype must be np.float32.")
        if item.vector.ndim != 1:
            raise ValueError("Vector must be one-dimensional.")
        if not isinstance(item.product_name, str):
            raise ValueError("Product name must be a string.")
        if not isinstance(item.brand, str):
            raise ValueError("Brand must be a string.")
        if not isinstance(item.category, str):
            raise ValueError("Category must be a string.")
        if not isinstance(item.subcategory, str):
            raise ValueError("Subcategory must be a string.")
        if not isinstance(item.image_name, str):
            raise ValueError("Image name must be a string.")
        
    def add_vectors(
        self,
        items: Iterable[VectorRecord],
    ) -> None:
        rows = []
        for item in items:
            self._check_item(item)
            array = self._normalize_vector(item.vector)
            rows.append(
                self._vector_row(
                    item.id,
                    array,
                    product_name=item.product_name,
                    brand=item.brand,
                    category=item.category,
                    subcategory=item.subcategory,
                    image_name=item.image_name,
                )
            )

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
            product_name=row["product_name"],
            brand=row["brand"],
            category=row["category"],
            subcategory=row["subcategory"],
            image_name=row["image_name"],
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
                    product_name=row["product_name"],
                    brand=row["brand"],
                    category=row["category"],
                    subcategory=row["subcategory"],
                    image_name=row["image_name"],
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def count(self) -> int:
        row = self.connection.execute(schema.COUNT_VECTORS_SQL).fetchone()
        return int(row["total"])

    def _vector_row(
        self,
        vector_id: str,
        array: npt.NDArray[np.float32],
        product_name: str,
        brand: str,
        category: str,
        subcategory: str,
        image_name: str,
    ) -> dict[str, SqlValue]:
        return {
            "id": vector_id,
            "dimensions": array.shape[0],
            "dtype": str(array.dtype),
            "vector": array.tobytes(),
            "product_name": product_name,
            "brand": brand,
            "category": category,
            "subcategory": subcategory,
            "image_name": image_name,
        }

    def _unpack_vector_item(
        self,
        item: tuple[str, Sequence[float] | np.ndarray],
    ) -> tuple[str, Sequence[float] | np.ndarray]:
        if len(item) == 2:
            vector_id, vector = item
            return vector_id, vector
        raise ValueError("Vector items must be (id, vector).")

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

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
