import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
from rqvae.layers import Config
config = Config()

def plot_training_history(history):
    """plotting loss curve"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 1].plot(history['epoch'], history['recon_loss'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quantization Loss
    axes[1, 0].plot(history['epoch'], history['quant_loss'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Quantization Loss')
    axes[1, 0].set_title('Quantization Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Collision Rate
    axes[1, 1].plot(history['epoch'], history['collision_rate'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Collision Rate')
    axes[1, 1].set_title('Collision Rate (Lower is Better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.ckpt_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("training history plot saved")

# call instance
# plot_training_history(trainer.history)

from sklearn.manifold import TSNE
import matplotlib.cm as cm

def evaluate_tsne(model, dataset, config, num_samples=3000, perplexity=30):
    """
    use t-SNE to visualize original vs reconstructed embeddings,
    colored by learned semantic IDs.
    """
    print(f"Starting t-SNE evaluation (num_samples: {num_samples})...")
    
    model.eval()
    
    # 1. 随机采样数据
    # 使用 DataLoader 配合 shuffle=True 来随机获取数据
    eval_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    original_embs = []
    recon_embs = []
    layer1_codes = [] # 用于着色：使用第一层量化器的 ID
    
    collected_samples = 0
    
    with torch.no_grad():
        for _, batch_emb in eval_loader:
            if collected_samples >= num_samples:
                break
                
            batch_emb = batch_emb.to(config.device)
            
            # 获取重构向量和 Semantic IDs
            # 注意：我们需要调用 forward 来获取重构，调用 get_semantic_ids 获取 ID
            recon, _, _, _, indices = model(batch_emb)
            
            original_embs.append(batch_emb.cpu().numpy())
            recon_embs.append(recon.cpu().numpy())
            layer1_codes.append(indices[:, 0].cpu().numpy()) # 取第一层的 ID 作为颜色标签
            
            collected_samples += batch_emb.shape[0]
    
    # 拼接数据并截断到 num_samples
    original_embs = np.concatenate(original_embs, axis=0)[:num_samples]
    recon_embs = np.concatenate(recon_embs, axis=0)[:num_samples]
    layer1_codes = np.concatenate(layer1_codes, axis=0)[:num_samples]
    
    print("Data ready, running t-SNE dimensionality reduction...")
    print(f"  Data shape: {original_embs.shape}")
    
    # 2. 运行 t-SNE
    # 将原始数据和重构数据拼接在一起进行降维，以保证在同一个坐标系下比较
    combined_data = np.concatenate([original_embs, recon_embs], axis=0)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, init='pca', learning_rate='auto')
    combined_2d = tsne.fit_transform(combined_data)
    
    # 分离数据
    orig_2d = combined_2d[:num_samples]
    recon_2d = combined_2d[num_samples:]
    
    print("t-SNE complete, plotting...")
    
    # 3. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    
    # 设置颜色映射 (根据第一层的 Code ID)
    num_codes = config.num_emb_list[0]
    cmap = cm.get_cmap('tab20', num_codes) # 使用 tab20 颜色板
    
    # 图 1: 原始数据的分布 (按 Semantic ID 着色)
    # 这可以验证 RQ-VAE 是否学到了数据的聚类结构
    scatter1 = axes[0].scatter(orig_2d[:, 0], orig_2d[:, 1], c=layer1_codes, cmap=cmap, s=10, alpha=0.6)
    axes[0].set_title(f'Original Embeddings\n(Colored by Layer-1 Semantic ID)')
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    axes[0].grid(True, alpha=0.3)
    
    # 图 2: 重构数据的分布 (按 Semantic ID 着色)
    # 这展示了模型输出的流形结构
    # scatter2 = axes[1].scatter(recon_2d[:, 0], recon_2d[:, 1], c=layer1_codes, cmap=cmap, s=10, alpha=0.6)
    # axes[1].set_title(f'Reconstructed Embeddings\n(Colored by Layer-1 Semantic ID)')
    # axes[1].set_xlabel('t-SNE Dim 1')
    # axes[1].set_ylabel('t-SNE Dim 2')
    # axes[1].grid(True, alpha=0.3)
    scatter2 = axes[1].scatter(recon_2d[:, 0], recon_2d[:, 1], c=layer1_codes, cmap=cmap, s=10, alpha=0.6)

    # 添加 Colorbar，并调整位置
    # ax=axes[1]: 指定 colorbar 属于哪个子图
    # fraction=0.046: 控制 colorbar 的宽度（相对于图的大小），数值越小越细
    # pad=0.04: 控制 colorbar 与图之间的距离，数值越大距离越远
    cbar = fig.colorbar(scatter2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Layer 1 Code Index')

    axes[1].set_title(f'Reconstructed Embeddings\n(Colored by Layer-1 Semantic ID)')
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    axes[1].grid(True, alpha=0.3)
    
    # 图 3: 原始 vs 重构 (对比分布重合度)
    # 蓝色是原始，红色是重构。理想情况下两者应该高度重合。
    axes[2].scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', label='Original', s=10, alpha=0.3)
    axes[2].scatter(recon_2d[:, 0], recon_2d[:, 1], c='red', label='Reconstructed', s=10, alpha=0.3)
    axes[2].set_title('Distribution Overlap\n(Original vs Reconstructed)')
    axes[2].set_xlabel('t-SNE Dim 1')
    axes[2].set_ylabel('t-SNE Dim 2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 添加颜色条
    # cbar = plt.colorbar(scatter1, ax=axes[:2], fraction=0.02, pad=0.04)
    # cbar.set_label('Layer 1 Code Index')
    
    plt.tight_layout()
    save_path = os.path.join(config.ckpt_dir, 'tsne_evaluation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"t-SNE visualization saved to: {save_path}")

# Usage:
# evaluate_tsne(model, dataset, config, num_samples=3000)


# =====================================================================
# validate_reconstruction
# =====================================================================

import torch.nn.functional as F
from tqdm import tqdm


def validate_reconstruction(model, dataset, config, num_samples=100):
    """
    Validate embedding reconstruction quality via semantic ID round-trip.

    Args:
        model:       Trained RQVAE.
        dataset:     POIEmbeddingDataset.
        config:      Config (uses config.device, config.ckpt_dir).
        num_samples: Number of random samples to evaluate.

    Returns:
        dict with keys: mse, mae, avg_cos_sim, l2_distances
    """
    model.eval()

    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    original_embeddings = []
    reconstructed_embeddings = []

    print(f"Validating reconstruction quality (n={len(sample_indices)})...")

    with torch.no_grad():
        for idx in tqdm(sample_indices):
            poi_id, original_emb = dataset[idx]
            original_emb = original_emb.unsqueeze(0).to(config.device)

            semantic_id = model.get_semantic_ids(original_emb)
            recon_emb = model.reconstruct_from_ids(semantic_id)

            original_embeddings.append(original_emb.cpu())
            reconstructed_embeddings.append(recon_emb.cpu())

    original_embeddings = torch.cat(original_embeddings, dim=0)
    reconstructed_embeddings = torch.cat(reconstructed_embeddings, dim=0)

    mse = F.mse_loss(reconstructed_embeddings, original_embeddings).item()
    mae = F.l1_loss(reconstructed_embeddings, original_embeddings).item()
    cos_sim = F.cosine_similarity(original_embeddings, reconstructed_embeddings, dim=1)
    avg_cos_sim = cos_sim.mean().item()

    print(f"\nReconstruction quality:")
    print(f"  MSE:                {mse:.6f}")
    print(f"  MAE:                {mae:.6f}")
    print(f"  Avg Cosine Sim:     {avg_cos_sim:.4f}")

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].hist(cos_sim.numpy(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(avg_cos_sim, color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {avg_cos_sim:.4f}')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reconstruction Quality: Cosine Similarity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    l2_distances = torch.norm(original_embeddings - reconstructed_embeddings, dim=1).numpy()
    axes[1].hist(l2_distances, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(l2_distances.mean(), color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {l2_distances.mean():.4f}')
    axes[1].set_xlabel('L2 Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Reconstruction Quality: L2 Distance Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.ckpt_dir, 'reconstruction_quality.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return {'mse': mse, 'mae': mae, 'avg_cos_sim': avg_cos_sim, 'l2_distances': l2_distances}


# =====================================================================
# analyze_codebook_usage
# =====================================================================


def analyze_codebook_usage(semantic_ids_dict, config):
    """
    Plot per-layer codebook utilisation frequency.

    Args:
        semantic_ids_dict: {poi_id: (code0, code1, ...)}
        config:            Config (uses config.num_emb_list, config.ckpt_dir).
    """
    print(f"\n{'='*60}")
    print("Codebook Usage Analysis")
    print(f"{'='*60}\n")

    all_ids = list(semantic_ids_dict.values())
    num_layers = len(config.num_emb_list)

    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    for i in range(num_layers):
        layer_ids = [id_tuple[i] for id_tuple in all_ids]
        unique, counts = np.unique(layer_ids, return_counts=True)

        axes[i].bar(range(len(unique)), counts, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel('Code Index')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(
            f'Layer {i+1} - Codebook Usage\n'
            f'(Size: {config.num_emb_list[i]}, Used: {len(unique)})'
        )
        axes[i].grid(True, alpha=0.3, axis='y')

        print(f"Layer {i+1}:")
        print(f"  Codebook size:     {config.num_emb_list[i]}")
        print(f"  Codes used:        {len(unique)} ({len(unique)/config.num_emb_list[i]:.2%})")
        print(f"  Avg frequency:     {counts.mean():.1f}")
        print(f"  Max frequency:     {counts.max()}")
        print(f"  Min frequency:     {counts.min()}")
        print()

    plt.tight_layout()
    plt.savefig(os.path.join(config.ckpt_dir, 'codebook_usage.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("Codebook usage analysis complete.")


# =====================================================================
# save_final_model
# =====================================================================

import json


def save_final_model(model, config, trainer, semantic_ids_dict, validation_results):
    """
    Save the final model weights, config JSON, and a plain-text training summary.

    Args:
        model:              Trained RQVAE.
        config:             Config instance.
        trainer:            RQVAETrainer instance (for best_loss / best_collision_rate).
        semantic_ids_dict:  Output of generate_and_save_semantic_ids().
        validation_results: Output of validate_reconstruction().
    """
    os.makedirs(config.ckpt_dir, exist_ok=True)

    # --- model weights --------------------------------------------------
    final_save_path = os.path.join(config.ckpt_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'semantic_ids': semantic_ids_dict,
        'validation_results': validation_results,
        'model_info': {
            'input_dim': model.in_dim,
            'latent_dim': model.e_dim,
            'num_quantizers': len(model.num_emb_list),
            'codebook_sizes': model.num_emb_list,
            'total_params': sum(p.numel() for p in model.parameters()),
        }
    }, final_save_path)
    print(f"Final model saved to: {final_save_path}")

    # --- config JSON ----------------------------------------------------
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    config_dict['device'] = str(config_dict['device'])
    config_json_path = os.path.join(config.ckpt_dir, "config.json")
    with open(config_json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {config_json_path}")

    # --- training summary ------------------------------------------------
    summary_path = os.path.join(config.ckpt_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RQ-VAE Training Summary\n")
        f.write("="*60 + "\n\n")
        f.write("Model Configuration:\n")
        f.write(f"  Input Dimension:    {model.in_dim}\n")
        f.write(f"  Latent Dimension:   {model.e_dim}\n")
        f.write(f"  Num Quantizers:     {len(model.num_emb_list)}\n")
        f.write(f"  Codebook Sizes:     {model.num_emb_list}\n")
        f.write(f"  Total Parameters:   {sum(p.numel() for p in model.parameters()):,}\n\n")
        f.write("Training Results:\n")
        f.write(f"  Best Loss:          {trainer.best_loss:.6f}\n")
        f.write(f"  Best Collision Rate:{trainer.best_collision_rate:.4%}\n\n")
        f.write("Validation Results:\n")
        f.write(f"  MSE:                {validation_results['mse']:.6f}\n")
        f.write(f"  MAE:                {validation_results['mae']:.6f}\n")
        f.write(f"  Avg Cosine Sim:     {validation_results['avg_cos_sim']:.4f}\n\n")
        f.write("Semantic IDs:\n")
        f.write(f"  Total POIs:         {len(semantic_ids_dict):,}\n")
        f.write(f"  Unique IDs:         {len(set(semantic_ids_dict.values())):,}\n")
    print(f"Training summary saved to: {summary_path}")
    print("All files saved.")


# =====================================================================
# plot_semantic_clusters  (PCA visualisation with prefix matching)
# =====================================================================

from sklearn.decomposition import PCA


def plot_semantic_clusters(
    query_semantics,
    embeddings_2d,
    all_poi_ids,
    semantic_df,
    alpha_fg=0.8,
    point_size=15,
):
    """
    Visualise PCA-reduced embeddings, highlighting POIs that match the
    given semantic ID queries (supports full or prefix matching).

    Args:
        query_semantics: list of queries, each can be:
                         - int            → match first layer only
                         - tuple/list     → match as many layers as the length
                         e.g. [(12,25,3), (17,17), 30]
        embeddings_2d:   numpy array (N, 2) – PCA-reduced coordinates
                         (must align 1-to-1 with all_poi_ids).
        all_poi_ids:     list/array of POI IDs (same order as embeddings_2d).
        semantic_df:     DataFrame with columns 'poi_id' and 'semantic_id' (tuple).
        alpha_fg:        Opacity for highlighted points.
        point_size:      Marker size.
    """
    print(f"Plotting Semantic Clusters: {query_semantics}")

    poi_to_index = {pid: i for i, pid in enumerate(all_poi_ids)}

    # Ensure semantic_id column contains tuples
    if len(semantic_df) > 0 and isinstance(semantic_df['semantic_id'].iloc[0], str):
        from ast import literal_eval
        semantic_df = semantic_df.copy()
        semantic_df['semantic_id'] = semantic_df['semantic_id'].apply(literal_eval)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(query_semantics)))
    total_found = 0

    for idx, query in enumerate(query_semantics):
        query_tuple = (query,) if isinstance(query, int) else tuple(query)
        query_len = len(query_tuple)

        matched_rows = semantic_df[
            semantic_df['semantic_id'].apply(
                lambda sid: isinstance(sid, tuple)
                and len(sid) >= query_len
                and sid[:query_len] == query_tuple
            )
        ]
        matched_poi_ids = matched_rows['poi_id'].values
        valid_indices = [poi_to_index[pid] for pid in matched_poi_ids if pid in poi_to_index]

        if not valid_indices:
            print(f"  Warning: no matching POIs found in dataset for query {query}")
            continue

        selected = embeddings_2d[valid_indices]
        plt.scatter(
            selected[:, 0], selected[:, 1],
            color=colors[idx], alpha=alpha_fg,
            s=point_size * 1.5, edgecolors='white', linewidth=0.5,
            label=f'Query: {query} (n={len(valid_indices)})'
        )
        total_found += len(valid_indices)

    plt.title(
        f'PCA – Embeddings by Semantic ID Group\nTotal highlighted: {total_found}',
        fontsize=14
    )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
