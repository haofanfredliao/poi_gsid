import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from embedding.trainer import train_poi_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train POI embedding model")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Path to CSV input data",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="",
        help="Path to parquet input data (.parquet / .parquet.gz)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing cover images",
    )
    parser.add_argument("--output_dir", type=str, default="./poi_encoder_output")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def _load_dataframe(csv_path: str, parquet_path: str) -> pd.DataFrame:
    if csv_path and parquet_path:
        raise ValueError("Please provide only one of --csv_path or --parquet_path")

    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return pd.read_csv(csv_path)

    if parquet_path:
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        return pd.read_parquet(parquet_path)

    raise ValueError("Please provide input data via --csv_path or --parquet_path")


def train_embedding_main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise NotADirectoryError(f"Invalid --image_dir: {args.image_dir}")

    df = _load_dataframe(args.csv_path, args.parquet_path)

    print(f"Dataset size: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Val set size: {len(val_df)}")

    train_poi_encoder(
        train_df=train_df,
        val_df=val_df,
        image_base_path=str(image_dir),
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        is_gcs=False,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    train_embedding_main()