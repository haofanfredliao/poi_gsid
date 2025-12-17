import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import gc
from dataset.poi_dataloader import *
from embedding.layers import *

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


if __name__ == "__main__":
    # Configure paths for all batches
    parquet_paths = [   
        "/Users/fred/HKU/data/pois_pandas/batch_0001.parquet.gz",
        "/Users/fred/HKU/data/pois_pandas/batch_0002.parquet.gz",
        "/Users/fred/HKU/data/pois_pandas/batch_0003.parquet.gz",
        "/Users/fred/HKU/data/pois_pandas/batch_0004.parquet.gz",
        "/Users/fred/HKU/data/pois_pandas/batch_0005.parquet.gz",
        "/Users/fred/HKU/data/pois_pandas/batch_0006.parquet.gz",
    ]

    image_base_paths = [
        "/home/jupyter/image_data",      
        "/home/jupyter/image_data_b2",   
        "/home/jupyter/image_data_b3",   
        "/my_data/image_data_b4",        
        "/my_data/image_data_b5",
        "/my_data/image_data_b6"
    ]

    # Generate embeddings
    poi_embeddings = generate_all_poi_embeddings(
        model_path="/home/jupyter/poi_encoder_output/final_model",
        parquet_paths=parquet_paths,
        image_base_paths=image_base_paths,
        output_path="/home/jupyter/poi_embeddings_full_v2.pkl",
        batch_size=128,  
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n  POI Embeddings Generation Completed!")
    print(f"   Statistics:")
    print(f"   Total POIs: {len(poi_embeddings):,}")
    print(f"   Embedding Dimension: {poi_embeddings[list(poi_embeddings.keys())[0]].shape[0]}")
    print(f"   Memory Usage: {sum(emb.nbytes for emb in poi_embeddings.values()) / 1024**3:.2f} GB")