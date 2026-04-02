import torch
import numpy as np
import pickle
import os
import argparse
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rqvae.layers import RQVAE, Config
from dataset.embedding_dataset import POIEmbeddingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate semantic IDs with trained RQ-VAE")
    parser.add_argument("--embedding_pkl_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="./rqvae_checkpoints")
    parser.add_argument("--save_path", type=str, default="poi_semantic_ids.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def generate_and_save_semantic_ids(model, dataset, config, save_path="poi_semantic_ids.pkl"):
    """
    Generate semantic IDs for all POIs and save as a dict.

    Args:
        model:     Trained RQVAE model.
        dataset:   POIEmbeddingDataset instance.
        config:    Config instance (uses config.device, config.batch_size,
                   config.num_workers, config.ckpt_dir, config.num_emb_list).
        save_path: File name (relative to config.ckpt_dir) for the output .pkl.

    Returns:
        semantic_ids_dict: {poi_id: (code_layer1, code_layer2, ...)}
    """
    model.eval()

    semantic_ids_dict = {}

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    print("Generating Semantic IDs...")

    with torch.no_grad():
        for poi_ids, embeddings in tqdm(loader, desc="Generating IDs"):
            embeddings = embeddings.to(config.device)

            indices = model.get_semantic_ids(embeddings)   # (batch_size, num_quantizers)
            indices = indices.cpu().numpy()

            for poi_id, semantic_id in zip(poi_ids, indices):
                semantic_ids_dict[poi_id] = tuple(semantic_id.tolist())

    # Save
    os.makedirs(config.ckpt_dir, exist_ok=True)
    save_full_path = os.path.join(config.ckpt_dir, save_path)
    with open(save_full_path, "wb") as f:
        pickle.dump(semantic_ids_dict, f)

    print(f"Semantic IDs saved to: {save_full_path}")
    print(f"  Total POIs:   {len(semantic_ids_dict):,}")
    print(f"  ID depth:     {len(list(semantic_ids_dict.values())[0])} layers")

    # Usage stats
    all_ids = list(semantic_ids_dict.values())
    for i in range(len(config.num_emb_list)):
        layer_ids = [id_tuple[i] for id_tuple in all_ids]
        unique_count = len(set(layer_ids))
        usage_rate = unique_count / config.num_emb_list[i]
        print(f"  Layer {i+1}: {unique_count}/{config.num_emb_list[i]} codes used ({usage_rate:.2%})")

    unique_combinations = len(set(all_ids))
    total_possible = int(np.prod(config.num_emb_list))
    print(f"  Total unique combinations: {unique_combinations:,}/{total_possible:,} "
          f"({unique_combinations/total_possible:.2%})")

    return semantic_ids_dict


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    config.embedding_pkl_path = args.embedding_pkl_path
    config.ckpt_dir = args.ckpt_dir
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device

    if not os.path.exists(config.embedding_pkl_path):
        raise FileNotFoundError(f"Embedding PKL not found: {config.embedding_pkl_path}")

    dataset = POIEmbeddingDataset(config.embedding_pkl_path)

    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        import glob

        checkpoints = sorted(
            glob.glob(os.path.join(config.ckpt_dir, "checkpoint_epoch_*.pt")),
            key=os.path.getmtime,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {config.ckpt_dir}")
        ckpt_path = checkpoints[-1]

    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)

    model = RQVAE(
        in_dim=dataset.get_embedding_dim(),
        num_emb_list=config.num_emb_list,
        e_dim=config.e_dim,
        layers=config.layers,
        dropout_prob=config.dropout_prob,
        bn=config.bn,
        loss_type=config.loss_type,
        quant_loss_weight=config.quant_loss_weight,
        kmeans_init=False,   # no re-init during inference
        kmeans_iters=config.kmeans_iters,
        sk_epsilons=config.sk_epsilons,
        sk_iters=config.sk_iters,
        use_linear=config.use_linear,
        use_sk=config.use_sk,
        beta=config.beta,
        diversity_loss=config.lamda,
    ).to(config.device)

    model.load_state_dict(checkpoint["model_state_dict"])

    semantic_ids_dict = generate_and_save_semantic_ids(
        model=model,
        dataset=dataset,
        config=config,
        save_path=args.save_path,
    )

    print("\nSample Semantic IDs:")
    for poi_id, semantic_id in list(semantic_ids_dict.items())[:5]:
        print(f"  {poi_id}: {semantic_id}")
