import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from sklearn.cluster import KMeans
import os

class Config:
    """RQ-VAE训练配置"""
    
    def __init__(self):
        # 数据路径
        self.embedding_pkl_path = "/home/jupyter/poi_embeddings_full_v2.pkl"
        
        
        # 模型参数
        self.num_emb_list = [32, 32, 32]
        self.e_dim = 16
        self.layers = [512, 256, 128]
        
        # VQ参数
        self.beta = 0.25
        self.lamda = 0.0
        self.quant_loss_weight = 2.0
        self.kmeans_init = True
        self.kmeans_iters = 100
        self.use_sk = True
        self.sk_epsilons = [0.01, 0.01, 0.01]
        self.sk_iters = 50
        self.use_linear = 0
        
        # 训练参数
        self.lr = 1e-4
        self.epochs = 300
        self.batch_size = 128
        self.num_workers = 0
        self.eval_step = 50
        self.warmup_epochs = 100
        
        # 正则化
        self.weight_decay = 1e-4
        self.dropout_prob = 0.1
        self.bn = False
        
        # 损失函数
        self.loss_type = "mse"
        
        # 优化器和调度器
        self.learner = "AdamW"
        self.lr_scheduler_type = "constant"
        
        # 保存
        self.ckpt_dir = "./rqvae_checkpoints"
        self.save_limit = 5
        
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __repr__(self):
        """打印配置"""
        config_str = "=" * 60 + "\n"
        config_str += "RQ-VAE Configuration\n"
        config_str += "=" * 60 + "\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"{key:25s}: {value}\n"
        config_str += "=" * 60
        return config_str

config = Config()
print(config)

# 创建checkpoint目录
os.makedirs(config.ckpt_dir, exist_ok=True)

def activation_layer(activation_name="relu", emb_dim=None):
    """创建激活函数层"""
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            f"activation function {activation_name} is not implemented"
        )
    return activation


class MLPLayers(nn.Module):    
    def __init__(self, layers, dropout=0.0, activation="relu", bn=False, weight_init="xavier"):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.weight_init = weight_init

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.weight_init == 'xavier':
                init.xavier_uniform_(module.weight)
            elif self.weight_init == 'he':
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif self.weight_init == 'normal':
                init.normal_(module.weight, mean=0.0, std=0.01)
            elif self.weight_init == 'uniform':
                init.uniform_(module.weight, a=-0.1, b=0.1)
            
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)
    
def kmeans(samples, num_clusters, num_iters=10):
    """使用sklearn的K-means初始化codebook"""
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """Sinkhorn算法用于软分配"""
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q

class VectorQuantizer(nn.Module):
    """单层向量量化器（VQ层）"""
    
    def __init__(
            self,
            n_e,
            e_dim,
            beta=0.25,
            kmeans_init=False,
            kmeans_iters=10,
            sk_epsilon=0.01,
            sk_iters=100,
            use_linear=0,
            use_sk=False,
            diversity_loss=0.0
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.use_sk = use_sk
        self.diversity_loss = diversity_loss

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, epoch_idx):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(self.embedding.weight)
        else:
            embeddings_weight = self.embedding.weight

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, embeddings_weight.t())
       
        indices = torch.argmin(d, dim=-1)
        if self.use_linear == 1:
            x_q = F.embedding(indices, embeddings_weight).view(x.shape)
        else:
            x_q = self.embedding(indices).view(x.shape)

        if self.use_sk and self.sk_epsilon > 0:
            d_soft = self.center_distance_for_constraint(d)
            d_soft = d_soft.double()
            Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)
        else:
            Q = F.softmax(-d, dim=-1)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        if epoch_idx >= 1000:        
            if self.diversity_loss > 0:
                soft_counts = Q.sum(0)
                mean_soft_count = soft_counts.mean()
                mean_count_loss = torch.mean((soft_counts - mean_soft_count) ** 2) / (mean_soft_count ** 2 + 1e-5)
                
                diversity_loss = 0.05 * mean_count_loss
                loss = codebook_loss + self.beta * commitment_loss + self.diversity_loss * diversity_loss
            else:
                loss = codebook_loss + self.beta * commitment_loss
        else:
            loss = codebook_loss + self.beta * commitment_loss
        
        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d
    
class ResidualVectorQuantizer(nn.Module):
    """残差向量量化器（多层VQ堆叠）"""
    def __init__(
            self,
            n_e_list,
            e_dim,
            sk_epsilons,
            kmeans_init=False,
            kmeans_iters=100,
            sk_iters=100,
            use_linear=0,
            use_sk=False,
            beta=0.25,
            diversity_loss=0.0,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.use_linear = use_linear
        self.use_sk = use_sk
        self.sk_iters = sk_iters
        
        self.vq_layers = nn.ModuleList([VectorQuantizer(
            n_e,
            e_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilon=sk_epsilon,
            sk_iters=sk_iters,
            use_linear=use_linear,
            use_sk=use_sk,
            beta=beta,
            diversity_loss=diversity_loss,
        ) for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, epoch_idx):
        all_losses = []
        all_indices = []
        all_distances = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, epoch_idx)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, mean_losses, all_indices, all_distances
    
class RQVAE(nn.Module):
    """
    残差量化变分自编码器（RQ-VAE）
    用于将POI embeddings压缩成semantic IDs
    """
    
    def __init__(
            self,
            in_dim,
            num_emb_list,
            e_dim,
            layers,
            dropout_prob=0.1,
            bn=False,
            loss_type="mse",
            quant_loss_weight=1.0,
            kmeans_init=False,
            kmeans_iters=100,
            sk_epsilons=None,
            sk_iters=100,
            use_linear=0,
            use_sk=False,
            beta=0.25,
            diversity_loss=0.0,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.e_dim = e_dim
        self.num_emb_list = num_emb_list
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        
        # Encoder: embedding_dim -> e_dim
        encoder_layers = [in_dim] + layers + [e_dim]
        self.encoder = MLPLayers(
            encoder_layers,
            dropout=dropout_prob,
            activation="relu",
            bn=bn
        )
        
        # Residual Vector Quantizer
        if sk_epsilons is None:
            sk_epsilons = [0.0] * len(num_emb_list)
            
        self.quantizer = ResidualVectorQuantizer(
            n_e_list=num_emb_list,
            e_dim=e_dim,
            sk_epsilons=sk_epsilons,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_iters=sk_iters,
            use_linear=use_linear,
            use_sk=use_sk,
            beta=beta,
            diversity_loss=diversity_loss,
        )
        
        # Decoder: e_dim -> embedding_dim
        decoder_layers = [e_dim] + layers[::-1] + [in_dim]
        self.decoder = MLPLayers(
            decoder_layers,
            dropout=dropout_prob,
            activation="relu",
            bn=bn
        )
        
        print(f" RQVAE模型构建完成:")
        print(f" 输入维度: {in_dim}")
        print(f" Latent维度: {e_dim}")
        print(f" Quantizer层数: {len(num_emb_list)}")
        print(f" Codebook大小: {num_emb_list}")
        print(f" 总参数量: {sum(p.numel() for p in self.parameters()):,}")

    def encode(self, x):
        """编码: 原始embedding -> latent"""
        return self.encoder(x)
    
    def decode(self, z):
        """解码: latent -> 重构的embedding"""
        return self.decoder(z)
    
    def quantize(self, z, epoch_idx):
        """量化: latent -> quantized latent + indices"""
        z_q, quant_loss, indices, distances = self.quantizer(z, epoch_idx)
        return z_q, quant_loss, indices, distances
    
    def forward(self, x, epoch_idx=0):
        """
        前向传播
        
        Args:
            x: 输入embeddings, shape (batch_size, embedding_dim)
            epoch_idx: 当前epoch索引（用于diversity loss）
        
        Returns:
            recon: 重构的embeddings
            total_loss: 总损失
            quant_loss: 量化损失
            recon_loss: 重构损失
            indices: semantic IDs, shape (batch_size, num_quantizers)
        """
        # Encode
        z = self.encode(x)
        
        # Quantize
        z_q, quant_loss, indices, distances = self.quantize(z, epoch_idx)
        
        # Decode
        recon = self.decode(z_q)
        
        # Reconstruction loss
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(recon, x)
        elif self.loss_type == "l1":
            recon_loss = F.l1_loss(recon, x)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Total loss
        total_loss = recon_loss + self.quant_loss_weight * quant_loss
        
        return recon, total_loss, quant_loss, recon_loss, indices
    
    def get_semantic_ids(self, x, epoch_idx=0):
        """
        获取semantic IDs（推理模式）
        
        Args:
            x: 输入embeddings
        
        Returns:
            indices: semantic IDs, shape (batch_size, num_quantizers)
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
            _, _, indices, _ = self.quantize(z, epoch_idx)
        return indices
    
    def reconstruct_from_ids(self, indices):
        """
        从semantic IDs重构embeddings
        
        Args:
            indices: semantic IDs, shape (batch_size, num_quantizers)
        
        Returns:
            recon: 重构的embeddings
        """
        self.eval()
        with torch.no_grad():
            # 从每层quantizer获取embeddings并相加
            z_q = 0
            for i, quantizer in enumerate(self.quantizer.vq_layers):
                z_q = z_q + quantizer.get_codebook_entry(indices[:, i])
            
            # Decode
            recon = self.decode(z_q)
        return recon
