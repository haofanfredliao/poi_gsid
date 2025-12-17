import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Optional
import numpy as np
import io
import os

def parse_admin_id(admin_str: Optional[str]) -> int:
    """
    解析行政区划ID
    例如: '3_1' -> 3, '16_1' -> 16
    """
    if admin_str is None or pd.isna(admin_str):
        return 0
    
    try:
        # 如果包含下划线，取下划线前的部分
        if '_' in str(admin_str):
            admin_num = int(str(admin_str).split('_')[0])
            return min(63,admin_num)
        else:
            return min(63,int(admin_str))
    except:
        return 0
    
class TripletContrastiveLoss(nn.Module):
    """三元组对比学习损失（可选）"""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class POIDataset(Dataset):
    """POI数据集"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_base_path: str,
        clip_processor,
        text_tokenizer,
        max_text_length: int = 512,
        is_gcs: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.image_base_path = image_base_path
        self.clip_processor = clip_processor
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.is_gcs = is_gcs
        
        if is_gcs:
            from google.cloud import storage
            self.gcs_bucket = storage.Client(project=PROJECT_ID).bucket('qpon-recommend-samples')
    
    def __len__(self):
        return len(self.df)
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图片（支持本地和GCS）"""
        image_name = Path(image_path).name
        try:
            if self.is_gcs:
                # GCS路径
                blob_path = os.path.join(GCS_RELATIVE, image_name)
                blob = self.gcs_bucket.blob(blob_path)
                image_data = blob.download_as_bytes()
                return Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                # 本地路径
                full_path = Path(self.image_base_path) / image_name
                return Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ===== 图像处理 (只处理cover_image) =====
        cover_image = None
        if pd.notna(row.get('cover_image_url')):
            cover = self.load_image(row['cover_image_url'])
            if cover:
                # 处理图片
                processed = self.clip_processor(images=cover, return_tensors="pt")
                cover_image = processed['pixel_values'].squeeze(0)
        
        # ===== 文本处理 (使用预处理的text_for_embedding) =====
        text_for_embedding = row.get('text_for_embedding', '')
        
        # 如果text_for_embedding为空或None，构建一个基本的描述
        if pd.isna(text_for_embedding) or text_for_embedding == '':
            name = row.get('name', '')
            type_code = row.get('type_code', '')
            if isinstance(type_code, np.ndarray):
                type_code = ' '.join(type_code)
            text_for_embedding = f"[NAME] {name} [TYPE] {type_code}"
        
        text_inputs = self.text_tokenizer(
            text_for_embedding,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # ===== 地理信息 =====
        lat = torch.tensor(row['latitude'], dtype=torch.float32)
        lon = torch.tensor(row['longitude'], dtype=torch.float32)
        
        # 行政区划ID (解析admin_L1-4)
        admin_ids = []
        for level in range(1, 5):
            col_name = f'admin_L{level}'
            if col_name in row and pd.notna(row[col_name]):
                admin_id = parse_admin_id(row[col_name])
                admin_ids.append(admin_id)
            else:
                admin_ids.append(0)  # 缺失值用0填充
        
        admin_ids = torch.tensor(admin_ids, dtype=torch.long)
        
        return {
            'cover_image': cover_image,
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'lat': lat,
            'lon': lon,
            'admin_ids': admin_ids,
            'poi_id': row.get('poi_id', idx)
        }


def collate_fn(batch):
    """自定义collate函数"""
    
    # 分离有图像和无图像的样本
    cover_images = []
    text_input_ids = []
    text_attention_masks = []
    lats = []
    lons = []
    admin_ids_list = []
    poi_ids = []
    
    for item in batch:
        cover_images.append(item['cover_image'])
        text_input_ids.append(item['text_input_ids'])
        text_attention_masks.append(item['text_attention_mask'])
        lats.append(item['lat'])
        lons.append(item['lon'])
        admin_ids_list.append(item['admin_ids'])
        poi_ids.append(item['poi_id'])
    
    # 处理图像 - 如果有None，则整个batch的image设为None
    has_all_images = all(img is not None for img in cover_images)
    if has_all_images:
        stacked_images = torch.stack(cover_images)
    else:
        # 如果部分样本没有图像，用零张量填充
        valid_images = [img for img in cover_images if img is not None]
        if valid_images:
            img_shape = valid_images[0].shape
            stacked_images = []
            for img in cover_images:
                if img is not None:
                    stacked_images.append(img)
                else:
                    # 用零张量填充
                    stacked_images.append(torch.zeros_like(valid_images[0]))
            stacked_images = torch.stack(stacked_images)
        else:
            stacked_images = None
    
    return {
        'image': stacked_images,
        'text_inputs': {
            'input_ids': torch.stack(text_input_ids),
            'attention_mask': torch.stack(text_attention_masks)
        },
        'lat': torch.stack(lats),
        'lon': torch.stack(lons),
        'admin_ids': torch.stack(admin_ids_list),
        'labels': poi_ids  
    }

class MultiModalContrastiveLoss(nn.Module):
    """多模态对比学习损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [batch_size, embedding_dim]
        """
        batch_size = embeddings.shape[0]
        
        # 计算相似度矩阵
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 对角线为正样本，其他为负样本
        labels = torch.arange(batch_size, device=embeddings.device)
        
        # 对比损失
        loss = self.criterion(similarity_matrix, labels)
        
        return loss