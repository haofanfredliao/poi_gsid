import torch
from transformers import Trainer, TrainingArguments
import pandas as pd
from dataset.poi_dataloader import *
from embedding.layers import *

class POIContrastiveTrainer(Trainer):
    """POI Embedding Contrastive Trainer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss = MultiModalContrastiveLoss(temperature=0.07)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Loss Computation"""
        # Forward pass
        poi_ids = inputs.pop('labels', None)  # pop 会移除这个 key

        embeddings = model(
            image=inputs['image'],
            text_inputs=inputs['text_inputs'],
            lat=inputs['lat'],
            lon=inputs['lon'],
            admin_ids=inputs['admin_ids']
        )
        
        # Contrastive loss
        loss = self.contrastive_loss(embeddings)
        
        return (loss, {'embeddings': embeddings}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Prediction step during validation/testing - key modification!"""
        
        poi_ids = inputs.pop('labels', None)
        
        # Forward pass
        with torch.no_grad():
            embeddings = model(**inputs)
            # Compute loss
            if prediction_loss_only:
                loss = self.contrastive_loss(embeddings)
                return (loss, None, None)
            else:
                loss = self.contrastive_loss(embeddings)
                return (loss, embeddings, None)

def train_poi_encoder(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_base_path: str,
    output_dir: str = "./poi_encoder_checkpoints",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    use_lora: bool = True,
    is_gcs: bool = False,
    resume_from_checkpoint = None
):
    """Train POI Encoder"""
    
    # Initialize model
    model = MultiModalPOIEncoder(
        embedding_dim=512,
        use_lora=use_lora,
        freeze_backbone=True
    )
    
    # Prepare datasets
    train_dataset = POIDataset(
        train_df,
        image_base_path,
        model.clip_processor,
        model.text_tokenizer,
        is_gcs=is_gcs
    )
    
    val_dataset = POIDataset(
        val_df,
        image_base_path,
        model.clip_processor,
        model.text_tokenizer,
        is_gcs=is_gcs
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        do_eval=False,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,  
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    # Trainer
    trainer = POIContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(f"{output_dir}/final_model")
    
    return model, trainer