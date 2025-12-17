import torch
from torch.utils.data import Dataset
import pickle


class POIEmbeddingDataset(Dataset):
    """
    从PKL文件加载POI embeddings
    PKL文件格式应该是:
    {
        'poi_id_1': embedding_tensor,  # shape: (embedding_dim,)
        'poi_id_2': embedding_tensor,
        ...
    }
    """
    
    def __init__(self, pkl_path):
        print(f"加载embeddings from: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 提取POI IDs和embeddings
        self.poi_ids = list(self.data.keys())
        self.embeddings = [self.data[poi_id] for poi_id in self.poi_ids]
        
        # 转换为tensor（如果还不是）
        if not isinstance(self.embeddings[0], torch.Tensor):
            self.embeddings = [torch.FloatTensor(emb) for emb in self.embeddings]
        
        # 检查维度一致性
        embedding_dims = [emb.shape[0] for emb in self.embeddings]
        assert len(set(embedding_dims)) == 1, "所有embeddings维度必须一致！"
        
        self.embedding_dim = embedding_dims[0]
        self.num_pois = len(self.poi_ids)
        
        print(f"加载完成:")
        print(f"POI数量: {self.num_pois:,}")
        print(f"Embedding维度: {self.embedding_dim}")
        print(f"示例POI ID: {self.poi_ids[0]}")
        print(f"示例Embedding shape: {self.embeddings[0].shape}")
    
    def __len__(self):
        return self.num_pois
    
    def __getitem__(self, idx):
        """
        Returns:
            poi_id: POI的ID
            embedding: POI的embedding tensor
        """
        return self.poi_ids[idx], self.embeddings[idx]
    
    def get_embedding_dim(self):
        """返回embedding维度"""
        return self.embedding_dim