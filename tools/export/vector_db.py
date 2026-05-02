import argparse
import pathlib
import time
from pathlib import Path

import dinov3.custom_lib.inference.coreml
import dinov3.custom_lib.utils
from dinov3.custom_lib.export.models import ConvNextWithPreProcess
import numpy as np
import dinov3.custom_lib.vector_db.database



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill a SQLite vector database with random vectors and run demo queries."
    )
    parser.add_argument(
        "--excel-path",
        required=True,
        help="Path to Excel file containing product data.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Path to directory containing images.",
    )
    parser.add_argument(
        "--dino-weights",
        help="Path to DINOv3 pretrained weights",
    )
    parser.add_argument(
        "--db-path",
        default="backbar_ai_products_embeddings.db",
        help="SQLite database path to create or reuse.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=768,
        help="Number of dimensions per vector. Defaults to 768, DINOv3's embedding size.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many nearest matches to print per query.",
    )
    parser.add_argument(
        "--metric",
        choices=("cosine", "euclidean", "dot"),
        default="cosine",
        help="Similarity metric used for demo queries.",
    )
    return parser

def _load_excel_data(excel_path: pathlib.Path) -> list[dict]:
    import pandas as pd
    # sheet "production-partial"
    # columns of interest: "product_name", "brand", "category", "subcategory", "image_name"
    df = pd.read_excel(excel_path.as_posix(), sheet_name="production-partial")
    return df.to_dict(orient="records")

def _compute_image_embeddings(excel_path: pathlib.Path, images_dir: pathlib.Path, model: dinov3.custom_lib.inference.coreml.CoreMLInferenceModel,
                              image_width: int = 256, image_height: int = 512) -> list[dict]:
    records = _load_excel_data(excel_path)
    good_records = []
    inference_times = []
    for record in records:
        image = images_dir.glob(f"{record['image_name']}*")
        image_path = next(image, None)
        if image_path is None:
            print(f"Warning: Image not found for record {record['product_name']} at {record['image_name']}")
            continue
            
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_tensor = dinov3.custom_lib.utils.load_image_for_pretrained_model(
            image_path, normalize=False).unsqueeze(0)
        image_tensor = ConvNextWithPreProcess.resize_pad_image(
            image_tensor, new_width=image_width, new_height=image_height)
        t0 = time.time()
        embedding = model.inference(image_tensor).squeeze(0).cpu().numpy()
        t1 = time.time()
        inference_times.append(t1 - t0)
        record["embedding"] = embedding
        good_records.append(record)
    print(f"Computed embeddings for {len(good_records)}/{len(records)} records. Average inference time: {np.mean(inference_times):.4f} seconds.")
    return good_records

def main() -> None:
    args = build_parser().parse_args()
    db_path = Path(args.db_path)

    if args.dimensions <= 0:
        raise ValueError("--dimensions must be greater than 0.")

    model_path = pathlib.Path(args.dino_weights)
    model = dinov3.custom_lib.inference.coreml.CoreMLInferenceModel(model_path)
    images_dir = pathlib.Path(args.images_dir)
    excel_path = pathlib.Path(args.excel_path)
    vectors = _compute_image_embeddings(excel_path, images_dir, model)
    items = (
        (
            str(index),
            record["embedding"],
            {
                "product_name": record.get("product_name"),
                "brand": record.get("brand"),
                "category": record.get("category"),
                "subcategory": record.get("subcategory"),
                "image_name": record.get("image_name"),
            },
        )
        for index, record in enumerate(vectors)
    )

    with dinov3.custom_lib.vector_db.database.VectorDatabase(db_path) as db:
        db.add_vectors(items)
        print(f"Database: {db_path.resolve()}")

if __name__ == "__main__":
    main()
