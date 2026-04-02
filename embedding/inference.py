import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import gc
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.poi_dataloader import POIDataset, collate_fn
from embedding.layers import MultiModalPOIEncoder

def generate_all_poi_embeddings(
    model_path: str,
    parquet_paths: list,  # 所有batch的parquet文件路径列表
    image_base_paths: list,  # 对应的图像路径列表
    output_path: str = "/home/jupyter/poi_embeddings.pkl",
    batch_size: int = 128, 
    device: str = 'cuda'
):
    """
    generate embeddings for all POIs and save as a poi_id -> embedding dictionary
    """
    
    print("=" * 60)
    print("Starting full POI Embedding generation")
    print("=" * 60)
    
    # 1. Load Model
    print("\n Loading model...")
    model = MultiModalPOIEncoder(use_lora=False)
    
    from safetensors.torch import load_file
    checkpoint = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    print(f"Model Loaded, Device: {device}")
    
    # 2. Prepare Storage
    poi_embeddings = {}
    total_processed = 0
    
    # 3. Process each batch
    for batch_idx, (parquet_path, image_base_path) in enumerate(zip(parquet_paths, image_base_paths)):
        print(f"\n{'='*60}")
        print(f"   Processing Batch {batch_idx + 1}/{len(parquet_paths)}")
        print(f"   Data source: {parquet_path}")
        print(f"   Image path: {image_base_path}")
        print(f"{'='*60}")
        
        # Read Data
        df_batch = pd.read_parquet(parquet_path)
        
        print(f"   Data size: {len(df_batch):,}")
        
        # Create Dataset and DataLoader
        dataset = POIDataset(
            df_batch,
            image_base_path,
            model.clip_processor,
            model.text_tokenizer,
            is_gcs=False  
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  
            collate_fn=collate_fn,
            pin_memory=True  
        )
        
        # Inference
        batch_embeddings = []
        batch_poi_ids = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=f"   Batch {batch_idx+1} Inference Progress"):
                # Prepare inputs
                inputs = {
                    'image': batch_data['image'].to(device) if batch_data['image'] is not None else None,
                    'text_inputs': {
                        'input_ids': batch_data['text_inputs']['input_ids'].to(device),
                        'attention_mask': batch_data['text_inputs']['attention_mask'].to(device)
                    },
                    'lat': batch_data['lat'].to(device),
                    'lon': batch_data['lon'].to(device),
                    'admin_ids': batch_data['admin_ids'].to(device)
                }
                
                # Forward pass
                embeddings = model(**inputs)
                
                # Collect results
                batch_embeddings.append(embeddings.cpu().numpy())
                batch_poi_ids.extend(batch_data['labels'])
                
                # Periodically clear GPU cache
                if len(batch_embeddings) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Merge results of the current batch
        all_embeddings = np.vstack(batch_embeddings)
        
        print(f"\n    Batch {batch_idx+1} Completed")
        print(f"      Generated embeddings: {all_embeddings.shape}")
        
        # Build poi_id -> embedding mapping
        for poi_id, embedding in zip(batch_poi_ids, all_embeddings):
            poi_embeddings[poi_id] = embedding
        
        total_processed += len(batch_poi_ids)
        
        print(f"      Total processed: {total_processed:,} POIs")
        print(f"      GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        
        # Clear memory
        del df_batch, dataset, dataloader, batch_embeddings, batch_poi_ids, all_embeddings
        gc.collect()
        torch.cuda.empty_cache()
    
    # 4. Save Results
    print(f"\n{'='*60}")
    print(f"   Saving results to: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(poi_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"   File Size: {file_size:.2f} GB")
    print(f"   Total POIs: {len(poi_embeddings):,}")
    print(f"   Embedding Dimension: {list(poi_embeddings.values())[0].shape}")
    
    print(f"\n All Done!")
    print(f"{'='*60}")
    
    return poi_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Generate POI embeddings")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--parquet_path", type=str, default="")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./poi_embeddings.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _load_df(csv_path: str, parquet_path: str) -> pd.DataFrame:
    if csv_path and parquet_path:
        raise ValueError("Provide only one of --csv_path or --parquet_path")
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        return pd.read_csv(csv_path)
    if parquet_path:
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")
        return pd.read_parquet(parquet_path)
    raise ValueError("Provide input data with --csv_path or --parquet_path")


def generate_embeddings_single_file(
    model_path: str,
    df: pd.DataFrame,
    image_base_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
):
    model = MultiModalPOIEncoder(use_lora=False)

    from safetensors.torch import load_file

    checkpoint = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    dataset = POIDataset(
        df,
        image_base_path,
        model.clip_processor,
        model.text_tokenizer,
        is_gcs=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    poi_embeddings = {}
    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Embedding inference"):
            inputs = {
                "image": batch_data["image"].to(device) if batch_data["image"] is not None else None,
                "text_inputs": {
                    "input_ids": batch_data["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch_data["text_inputs"]["attention_mask"].to(device),
                },
                "lat": batch_data["lat"].to(device),
                "lon": batch_data["lon"].to(device),
                "admin_ids": batch_data["admin_ids"].to(device),
            }
            embeddings = model(**inputs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(batch_data["labels"])

    merged = np.vstack(all_embeddings)
    for poi_id, emb in zip(all_ids, merged):
        poi_embeddings[poi_id] = emb

    with open(output_path, "wb") as f:
        pickle.dump(poi_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved embeddings to: {output_path}")
    print(f"Total POIs: {len(poi_embeddings)}")
    print(f"Embedding dim: {merged.shape[1]}")
    return poi_embeddings


if __name__ == "__main__":
    args = parse_args()
    df = _load_df(args.csv_path, args.parquet_path)
    generate_embeddings_single_file(
        model_path=args.model_path,
        df=df,
        image_base_path=args.image_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )