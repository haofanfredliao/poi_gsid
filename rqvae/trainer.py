import torch
import torch.optim as optim
import os
from tqdm.auto import tqdm


class RQVAETrainer:
    """RQ-VAE训练器"""
    def __init__(self, config, model, dataloader):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.device = config.device
        
        # 优化器
        if config.learner == "AdamW":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
        elif config.learner == "Adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.learner}")
        
        # 学习率调度器
        if config.lr_scheduler_type == "constant":
            self.scheduler = None
        elif config.lr_scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs
            )
        elif config.lr_scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.epochs // 3,
                gamma=0.1
            )
        
        # Warmup
        self.warmup_epochs = config.warmup_epochs
        self.base_lr = config.lr
        
        # 记录
        self.history = {
            'epoch': [],
            'train_loss': [],
            'recon_loss': [],
            'quant_loss': [],
            'collision_rate': []
        }
        
        self.best_loss = float('inf')
        self.best_collision_rate = 1.0
        
        print("训练器初始化完成")
    
    def warmup_lr(self, epoch):
        """Warmup学习率"""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def calculate_collision_rate(self, indices):
        """
        计算collision rate（重复率）
        
        Args:
            indices: shape (num_samples, num_quantizers)
        
        Returns:
            collision_rate: 碰撞率
        """
        # 将multi-level indices转换为单一ID
        # 例如 [a, b, c] -> a * K2*K3 + b * K3 + c
        num_quantizers = indices.shape[1]
        codebook_sizes = self.config.num_emb_list
        
        unique_id = indices[:, 0].clone()
        multiplier = 1
        for i in range(1, num_quantizers):
            multiplier *= codebook_sizes[i-1]
            unique_id = unique_id * codebook_sizes[i] + indices[:, i]
        
        # 计算unique数量
        num_unique = len(torch.unique(unique_id))
        total = len(unique_id)
        
        collision_rate = 1.0 - (num_unique / total)
        return collision_rate
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        # Warmup
        if epoch < self.warmup_epochs:
            self.warmup_lr(epoch)
        
        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        all_indices = []
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for poi_ids, embeddings in pbar:
            embeddings = embeddings.to(self.device)
            
            # Forward
            recon, loss, quant_loss, recon_loss, indices = self.model(embeddings, epoch)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_quant_loss += quant_loss.item()
            all_indices.append(indices.cpu())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'recon': f"{recon_loss.item():.6f}",
                'quant': f"{quant_loss.item():.6f}"
            })
        
        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 计算平均值
        num_batches = len(self.dataloader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_quant_loss = total_quant_loss / num_batches
        
        # 计算collision rate
        all_indices = torch.cat(all_indices, dim=0)
        collision_rate = self.calculate_collision_rate(all_indices)
        
        return avg_loss, avg_recon_loss, avg_quant_loss, collision_rate
    
    def fit(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练 RQ-VAE")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss, recon_loss, quant_loss, collision_rate = self.train_epoch(epoch)
            
            # 记录
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['recon_loss'].append(recon_loss)
            self.history['quant_loss'].append(quant_loss)
            self.history['collision_rate'].append(collision_rate)
            
            # 更新最佳结果
            if train_loss < self.best_loss:
                self.best_loss = train_loss
            if collision_rate < self.best_collision_rate:
                self.best_collision_rate = collision_rate
            
            # 评估和保存
            if (epoch + 1) % self.config.eval_step == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print(f"{'='*60}")
                print(f"📊 Train Loss: {train_loss:.6f}")
                print(f"   - Recon Loss: {recon_loss:.6f}")
                print(f"   - Quant Loss: {quant_loss:.6f}")
                print(f"🎯 Collision Rate: {collision_rate:.4%}")
                print(f"📈 Learning Rate: {current_lr:.2e}")
                print(f"🏆 Best Loss: {self.best_loss:.6f}")
                print(f"🏆 Best Collision: {self.best_collision_rate:.4%}")
                
                # 保存checkpoint
                self.save_checkpoint(epoch, train_loss, collision_rate)
        
        print(f"\n{'='*60}")
        print(f"✅ 训练完成!")
        print(f"{'='*60}")
        print(f"🏆 Best Loss: {self.best_loss:.6f}")
        print(f"🏆 Best Collision Rate: {self.best_collision_rate:.4%}")
        
        return self.best_loss, self.best_collision_rate
    
    def save_checkpoint(self, epoch, loss, collision_rate):
        """保存checkpoint"""
        checkpoint_path = os.path.join(
            self.config.ckpt_dir,
            f"checkpoint_epoch_{epoch+1}_loss_{loss:.6f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'collision_rate': collision_rate,
            'config': self.config,
            'history': self.history
        }, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # 保留最近的N个checkpoints
        checkpoints = sorted(
            [f for f in os.listdir(self.config.ckpt_dir) if f.endswith('.pt')],
            key=lambda x: os.path.getmtime(os.path.join(self.config.ckpt_dir, x))
        )
        
        if len(checkpoints) > self.config.save_limit:
            for old_ckpt in checkpoints[:-self.config.save_limit]:
                os.remove(os.path.join(self.config.ckpt_dir, old_ckpt))