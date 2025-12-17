import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel, CLIPProcessor,
    AutoTokenizer, AutoModel,
)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional
import numpy as np
import os
from dataset.poi_dataloader import parse_admin_id

# Embedding Layers
class SpatialEncoder(nn.Module):
    """
    converts (lat, lon) -> high-dim vector
    using Sinusoidal Positional Encoding (Fourier Features)
    """
    def __init__(self, d_model, sigma=10.0):
        super(SpatialEncoder, self).__init__()
        self.d_model = d_model
        self.sigma = sigma
        # Randomly generate frequency matrix, fixed once generated
        self.register_buffer('freq', torch.randn(2, d_model // 2) * sigma)

    def forward(self, coords):
        # coords: [Batch, 2] (Lat, Lon)
        # Project to high-dimensional frequency space
        # [Batch, 2] @ [2, d_model/2] -> [Batch, d_model/2]
        proj = 2 * np.pi * coords @ self.freq
        # Concatenate sin and cos -> [Batch, d_model]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class GeoEncoder(nn.Module):
    """Geo Encoder"""
    
    def __init__(self, embedding_dim: int = 128, num_admin_levels: int = 4):
        super().__init__()
        
        # Coordinate encoding (using sin/cos encoding)
        # self.coord_encoder = nn.Sequential(
        #     nn.Linear(2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, embedding_dim // 2)
        # )
        self.coord_encoder = SpatialEncoder(d_model=embedding_dim//2)
        
        # GADM Country Code Administrative code encoding (one embedding table per level)
        # Set appropriate vocab size based on Indonesian administrative divisions
        self.admin_embeddings = nn.ModuleList([
            nn.Embedding(64, embedding_dim // (2 * num_admin_levels)),      # L1: Province level (34 provinces)
            nn.Embedding(64, embedding_dim // (2 * num_admin_levels)),     # L2: City/Regency level
            nn.Embedding(64, embedding_dim // (2 * num_admin_levels)),   # L3: District level
            nn.Embedding(64, embedding_dim // (2 * num_admin_levels))    # L4: Sub-district level
        ])
        
        self.fusion = nn.Linear(embedding_dim, embedding_dim)
    
    def encode_coordinates(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """Encode latitude and longitude"""
        # Normalize to [-1, 1], coordinate transformation based on Indonesian bounding box
        lat_norm = (lat + 5.5) / 17
        lon_norm = (lon - 118) / 46
        coords = torch.stack([lat_norm, lon_norm], dim=-1)
        return self.coord_encoder(coords)
    
    def forward(self, lat: torch.Tensor, lon: torch.Tensor, 
                admin_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Coordinate encoding
        coord_embed = self.encode_coordinates(lat, lon)
        
        # Administrative division encoding
        if admin_ids is not None:
            admin_embeds = []
            for i, emb_layer in enumerate(self.admin_embeddings):
                if i < admin_ids.shape[-1]:
                    admin_embeds.append(emb_layer(admin_ids[:, i]))
            admin_embed = torch.cat(admin_embeds, dim=-1)
            
            # Concatenate
            combined = torch.cat([coord_embed, admin_embed], dim=-1)
        else:
            # If no administrative divisions, pad with zeros
            batch_size = lat.shape[0]
            padding = torch.zeros(batch_size, coord_embed.shape[-1], 
                                device=lat.device)
            combined = torch.cat([coord_embed, padding], dim=-1)
        
        return self.fusion(combined)

class MultiModalPOIEncoder(nn.Module):
    """Multi-Modal POI Encoder with CLIP Vision, Text Encoder, and Geo Encoder"""
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        embedding_dim: int = 512,
        use_lora: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Image Encoder (CLIP Vision)
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Text Encoder (Multilingual Support)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        
        # Freeze Backbone Networks
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # LoRA Fine-tuning (Optional)
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],  # Targeting attention layers
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.vision_model = get_peft_model(self.vision_model, lora_config)
            self.text_model = get_peft_model(self.text_model, lora_config)
        
        # Geo Encoder
        self.geo_encoder = GeoEncoder(embedding_dim=128)
        
        # Get original dimensions of each modality
        vision_dim = self.vision_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size
        geo_dim = 128
        
        # Projection layers (map each modality to a unified space)
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.geo_projection = nn.Sequential(
            nn.Linear(geo_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention Fusion Layer
        self.fusion_attention = MultiModalAttentionFusion(embedding_dim)
        
        # Final Projection
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        
        self.embedding_dim = embedding_dim
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single cover image"""
        # image shape: [batch_size, 3, H, W]
        outputs = self.vision_model(pixel_values=image)
        
        # Use pooler_output or mean pooling of the last layer
        if hasattr(outputs, 'pooler_output'):
            image_embeds = outputs.pooler_output
        else:
            image_embeds = outputs.last_hidden_state.mean(dim=1)
        
        # Project to unified space
        image_embeds = self.vision_projection(image_embeds)
        
        return image_embeds
    
    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode preprocessed text for embedding"""
        outputs = self.text_model(**text_inputs)
        
        # Mean pooling
        attention_mask = text_inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        text_embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        
        # Projection
        text_embeds = self.text_projection(text_embeds)
        return text_embeds
    
    def encode_geo(self, lat: torch.Tensor, lon: torch.Tensor, 
                   admin_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode geographic information"""
        geo_embeds = self.geo_encoder(lat, lon, admin_ids)
        geo_embeds = self.geo_projection(geo_embeds)
        return geo_embeds
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        lat: Optional[torch.Tensor] = None,
        lon: Optional[torch.Tensor] = None,
        admin_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass, fuse all modalities
        """
        modality_embeds = []
        
        # Image modality
        if image is not None:
            image_embed = self.encode_image(image)
            modality_embeds.append(image_embed)
        
        # Text modality
        if text_inputs is not None:
            text_embed = self.encode_text(text_inputs)
            modality_embeds.append(text_embed)
        
        # Geo modality
        if lat is not None and lon is not None:
            geo_embed = self.encode_geo(lat, lon, admin_ids)
            modality_embeds.append(geo_embed)
        
        # Fuse all modalities
        if len(modality_embeds) == 0:
            raise ValueError("At least one modality input is required")
        
        # Fuse using attention mechanism
        fused_embed = self.fusion_attention(modality_embeds)
        
        # Final projection and normalization
        final_embed = self.final_projection(fused_embed)
        final_embed = F.normalize(final_embed, p=2, dim=-1)
        
        return final_embed

class MultiModalAttentionFusion(nn.Module):
    """Multi-Modal MHA Fusion"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, modality_embeds: List[torch.Tensor]) -> torch.Tensor:
        # Stack all modalities [batch_size, num_modalities, embedding_dim]
        stacked = torch.stack(modality_embeds, dim=1)
        
        # Self-attention
        attn_output, _ = self.attention(stacked, stacked, stacked)
        
        # Average pooling across modalities
        fused = attn_output.mean(dim=1)
        
        return self.layer_norm(fused)

# Inference Class
class POIEmbeddingExtractor:
    """POI Embedding提取器"""
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = MultiModalPOIEncoder(use_lora=False)
        
        # 加载模型权重 - 修改这里
        from safetensors.torch import load_file
        
        safetensors_path = f"{model_path}/model.safetensors"
        
        # 检查文件是否存在
        if os.path.exists(safetensors_path):
            try:
                print(f"✅ 找到 safetensors 文件: {safetensors_path}")
                checkpoint = load_file(safetensors_path)
            except:
                raise FileNotFoundError(
                f"未找到模型文件！\n"
                f"尝试查找的位置:\n"
                f"  - {safetensors_path}\n"
                f"请检查模型路径是否正确"
            )
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
        
        # 打印加载信息
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"Model loaded successfully")
        
        self.model = self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def get_embedding(
        self,
        cover_image_path: Optional[str] = None,
        text_for_embedding: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        admin_L1: Optional[str] = None,
        admin_L2: Optional[str] = None,
        admin_L3: Optional[str] = None,
        admin_L4: Optional[str] = None
    ) -> np.ndarray:
        """
        Get embedding for a single POI
        
        Returns: numpy array of shape (embedding_dim,)
        """
        # Process image
        cover_image = None
        if cover_image_path:
            img = Image.open(cover_image_path).convert('RGB')
            processed = self.model.clip_processor(images=img, return_tensors="pt")
            cover_image = processed['pixel_values'].to(self.device)
        
        # Process text
        if text_for_embedding:
            text_inputs = self.model.text_tokenizer(
                text_for_embedding,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
        else:
            # If no text, create empty input
            text_inputs = None
        
        # Process geographic coordinates
        lat = torch.tensor([latitude], dtype=torch.float32).to(self.device) if latitude else None
        lon = torch.tensor([longitude], dtype=torch.float32).to(self.device) if longitude else None
        
        # Process administrative divisions
        admin_ids = []
        for admin in [admin_L1, admin_L2, admin_L3, admin_L4]:
            admin_ids.append(parse_admin_id(admin))
        admin_ids_tensor = torch.tensor([admin_ids], dtype=torch.long).to(self.device)
        
        # Forward pass
        embedding = self.model(
            image=cover_image,
            text_inputs=text_inputs,
            lat=lat,
            lon=lon,
            admin_ids=admin_ids_tensor
        )
        
        return embedding.cpu().numpy()[0]
    
    def batch_get_embeddings(self, df: pd.DataFrame, image_base_path: str = '') -> np.ndarray:
        """
        批量获取embeddings
        
        Args:
            df: 包含POI数据的DataFrame
            image_base_path: 图像基础路径
        
        Returns:
            embeddings: numpy array of shape (len(df), embedding_dim)
        """
        embeddings = []
        
        for idx, row in df.iterrows():
            # 构建完整的图像路径
            cover_image_path = None
            if pd.notna(row.get('cover_image_url')) and image_base_path:
                cover_image_path = f"{image_base_path}/{row['cover_image_url']}"
            
            emb = self.get_embedding(
                cover_image_path=cover_image_path,
                text_for_embedding=row.get('text_for_embedding'),
                latitude=row.get('latitude'),
                longitude=row.get('longitude'),
                admin_L1=row.get('admin_L1'),
                admin_L2=row.get('admin_L2'),
                admin_L3=row.get('admin_L3'),
                admin_L4=row.get('admin_L4')
            )
            embeddings.append(emb)
        
        return np.array(embeddings)