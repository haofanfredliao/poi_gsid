import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from dataset.embedding_dataset import *
from rqvae.trainer import *
from rqvae.layers import *

if __name__ == "__main__":
    # 设置随机种子
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config()
    print(config)

    # 创建checkpoint目录
    os.makedirs(config.ckpt_dir, exist_ok=True)
    dataset = POIEmbeddingDataset(config.embedding_pkl_path)

    # 初始化模型
    model = RQVAE(
        in_dim=dataset.get_embedding_dim(),
        num_emb_list=config.num_emb_list,
        e_dim=config.e_dim,
        layers=config.layers,
        dropout_prob=config.dropout_prob,
        bn=config.bn,
        loss_type=config.loss_type,
        quant_loss_weight=config.quant_loss_weight,
        kmeans_init=config.kmeans_init,
        kmeans_iters=config.kmeans_iters,
        sk_epsilons=config.sk_epsilons,
        sk_iters=config.sk_iters,
        use_linear=config.use_linear,
        use_sk=config.use_sk,
        beta=config.beta,
        diversity_loss=config.lamda,
    ).to(config.device)

    # 创建DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 创建训练器
    trainer = RQVAETrainer(config, model, train_loader)

    # 开始训练
    best_loss, best_collision_rate = trainer.fit()
        