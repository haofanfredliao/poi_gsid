import torch
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from rqvae.layers import RQVAE, Config
from dataset.embedding_dataset import POIEmbeddingDataset


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
    config = Config()
    # Set the path to your pre-generated POI embeddings .pkl file:
    # config.embedding_pkl_path = "/path/to/poi_embeddings.pkl"

    if not config.embedding_pkl_path:
        raise ValueError(
            "config.embedding_pkl_path is empty. "
            "Set it to the path of your POI embeddings .pkl file before running inference."
        )

    dataset = POIEmbeddingDataset(config.embedding_pkl_path)

    # Load model from the latest checkpoint in config.ckpt_dir
    import glob
    checkpoints = sorted(
        glob.glob(os.path.join(config.ckpt_dir, "checkpoint_epoch_*.pt")),
        key=os.path.getmtime
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {config.ckpt_dir}")

    ckpt_path = checkpoints[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=config.device)

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
        save_path="poi_semantic_ids.pkl",
    )

    print("\nSample Semantic IDs:")
    for poi_id, semantic_id in list(semantic_ids_dict.items())[:5]:
        print(f"  {poi_id}: {semantic_id}")
