"""
Main training script for the Modular Context-Specialized Network.
Supports training both the core model and specialized modules.
"""

import os
import sys
import yaml
import argparse
import logging
from typing import Dict, Any, Optional, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model.core import CoreConfig, CoreLMHeadModel
from model.modules.chat_instruct import ChatInstructConfig, ChatInstructModule, ModularWithChatInstruct
from training.data_module import ModularContextDataModule


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ModularContextModelTrainer(pl.LightningModule):
    """PyTorch Lightning module for training the models."""
    
    def __init__(self, config: Dict[str, Any], model_type: str = "core"):
        super().__init__()
        
        self.config = config
        self.model_type = model_type
        
        # Save hyperparameters for resuming training
        self.save_hyperparameters()
        
        # Initialize model based on type
        if model_type == "core":
            self.model = self._init_core_model()
        elif model_type == "chat_instruct":
            self.model = self._init_chat_instruct_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _init_core_model(self) -> CoreLMHeadModel:
        """Initialize the core transformer model."""
        logger.info("Initializing core model...")
        
        # Create config from yaml config
        core_config = CoreConfig(
            vocab_size=self.config["model"]["core"]["vocab_size"],
            hidden_size=self.config["model"]["core"]["hidden_size"],
            num_hidden_layers=self.config["model"]["core"]["num_hidden_layers"],
            num_attention_heads=self.config["model"]["core"]["num_attention_heads"],
            intermediate_size=self.config["model"]["core"]["intermediate_size"],
            hidden_dropout_prob=self.config["model"]["core"]["hidden_dropout_prob"],
            attention_dropout_prob=self.config["model"]["core"]["attention_dropout_prob"],
            max_position_embeddings=self.config["model"]["core"]["max_position_embeddings"],
            initializer_range=self.config["model"]["core"]["initializer_range"],
            layer_norm_eps=self.config["model"]["core"]["layer_norm_eps"],
            use_cache=self.config["model"]["core"]["use_cache"],
            activation_function=self.config["model"]["core"]["activation_function"],
            tie_word_embeddings=self.config["model"]["core"]["tie_word_embeddings"],
            gradient_checkpointing=self.config["model"]["core"]["gradient_checkpointing"],
        )
        
        # Create model
        model = CoreLMHeadModel(core_config)
        
        return model
    
    def _init_chat_instruct_model(self) -> ModularWithChatInstruct:
        """Initialize the chat instruction module combined with a core model."""
        logger.info("Initializing chat-instruct model...")
        
        # First, load the core model
        core_config = CoreConfig(
            vocab_size=self.config["model"]["core"]["vocab_size"],
            hidden_size=self.config["model"]["core"]["hidden_size"],
            num_hidden_layers=self.config["model"]["core"]["num_hidden_layers"],
            num_attention_heads=self.config["model"]["core"]["num_attention_heads"],
            intermediate_size=self.config["model"]["core"]["intermediate_size"],
            hidden_dropout_prob=self.config["model"]["core"]["hidden_dropout_prob"],
            attention_dropout_prob=self.config["model"]["core"]["attention_dropout_prob"],
            max_position_embeddings=self.config["model"]["core"]["max_position_embeddings"],
            initializer_range=self.config["model"]["core"]["initializer_range"],
            layer_norm_eps=self.config["model"]["core"]["layer_norm_eps"],
            use_cache=self.config["model"]["core"]["use_cache"],
            activation_function=self.config["model"]["core"]["activation_function"],
            tie_word_embeddings=self.config["model"]["core"]["tie_word_embeddings"],
            gradient_checkpointing=self.config["model"]["core"]["gradient_checkpointing"],
        )
        
        # Try to load pretrained core model
        core_model = CoreLMHeadModel(core_config)
        
        # Check if there's a pretrained core model to load
        core_checkpoint_path = self.config.get("training", {}).get("chat_instruct", {}).get("core_model_path")
        if core_checkpoint_path and os.path.exists(core_checkpoint_path):
            logger.info(f"Loading pretrained core model from {core_checkpoint_path}")
            # Load pretrained weights
            checkpoint = torch.load(core_checkpoint_path, map_location="cpu")
            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                # Lightning checkpoint
                state_dict = checkpoint["state_dict"]
                # Remove 'model.' prefix if present
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            else:
                # Direct state dict
                state_dict = checkpoint
            
            # Load weights
            core_model.load_state_dict(state_dict, strict=False)
        
        # Create chat-instruct config
        chat_config = ChatInstructConfig(
            hidden_size=self.config["model"]["chat_instruct"]["hidden_size"],
            num_hidden_layers=self.config["model"]["chat_instruct"]["num_hidden_layers"],
            num_attention_heads=self.config["model"]["chat_instruct"]["num_attention_heads"],
            intermediate_size=self.config["model"]["chat_instruct"]["intermediate_size"],
            hidden_dropout_prob=self.config["model"]["chat_instruct"]["hidden_dropout_prob"],
            attention_dropout_prob=self.config["model"]["chat_instruct"]["attention_dropout_prob"],
            layer_norm_eps=self.config["model"]["chat_instruct"]["layer_norm_eps"],
            use_cross_attention=self.config["model"]["chat_instruct"]["use_cross_attention"],
            activation_function=self.config["model"]["chat_instruct"]["activation_function"],
            role_embeddings_size=self.config["model"]["chat_instruct"]["role_embeddings_size"],
            instruction_embeddings_size=self.config["model"]["chat_instruct"]["instruction_embeddings_size"],
        )
        
        # Create chat-instruct module
        chat_instruct_module = ChatInstructModule(
            name="chat_instruct",
            config=chat_config,
            core_config=core_config,
        )
        
        # Create combined model
        model = ModularWithChatInstruct(
            core_model=core_model,
            chat_instruct_module=chat_instruct_module,
        )
        
        # Freeze core model if specified
        if self.config["training"]["chat_instruct"]["freeze_core_layers"]:
            logger.info("Freezing core model layers")
            for param in core_model.parameters():
                param.requires_grad = False
        
        return model
    
    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Forward pass
        outputs = self.model(**batch)
        
        # Get loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Log loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Forward pass
        outputs = self.model(**batch)
        
        # Get loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Log loss
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        # Forward pass
        outputs = self.model(**batch)
        
        # Get loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Log loss
        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer config based on model type
        if self.model_type == "core":
            lr = self.config["training"]["core"]["lr"]
            weight_decay = self.config["training"]["core"]["weight_decay"]
        elif self.model_type == "chat_instruct":
            lr = self.config["training"]["chat_instruct"]["lr"]
            weight_decay = self.config["training"]["chat_instruct"]["weight_decay"]
            
            # Apply different learning rate to core model if not frozen
            if not self.config["training"]["chat_instruct"]["freeze_core_layers"]:
                # Create parameter groups with different learning rates
                core_params = []
                module_params = []
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if name.startswith("core_model"):
                            core_params.append(param)
                        else:
                            module_params.append(param)
                
                # Apply learning rate multiplier to core model
                core_lr_multiplier = self.config["training"]["chat_instruct"]["lr_core_multiplier"]
                param_groups = [
                    {"params": module_params, "lr": lr},
                    {"params": core_params, "lr": lr * core_lr_multiplier},
                ]
                
                # Create optimizer with parameter groups
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=weight_decay,
                    eps=self.config["training"]["optimizer"]["eps"],
                    betas=(
                        self.config["training"]["optimizer"]["beta1"],
                        self.config["training"]["optimizer"]["beta2"],
                    ),
                )
                
                # Create scheduler
                scheduler_config = self.config["training"]["scheduler"]
                if scheduler_config["name"] == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=scheduler_config["max_steps"],
                        eta_min=0.0,
                    )
                elif scheduler_config["name"] == "linear":
                    scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1.0,
                        end_factor=0.0,
                        total_iters=scheduler_config["max_steps"],
                    )
                else:
                    raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
                
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }
                
        # Default optimizer setup (when not using parameter groups)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            eps=self.config["training"]["optimizer"]["eps"],
            betas=(
                self.config["training"]["optimizer"]["beta1"],
                self.config["training"]["optimizer"]["beta2"],
            ),
        )
        
        # Create scheduler
        scheduler_config = self.config["training"]["scheduler"]
        if scheduler_config["name"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config["max_steps"],
                eta_min=0.0,
            )
        elif scheduler_config["name"] == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=scheduler_config["max_steps"],
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def train(config_path: str, model_type: str, resume_from_checkpoint: Optional[str] = None):
    """Train a model with the given configuration."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create data module
    data_module = ModularContextDataModule(config, module_type=model_type)
    
    # Create model
    model = ModularContextModelTrainer(config, model_type=model_type)
    
    # Create logger
    logger_type = config["logging"]["logger"]
    if logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=config["logging"]["log_dir"],
            name=config["logging"]["name"],
            version=config["logging"]["version"],
            default_hp_metric=config["logging"]["tensorboard"]["default_hp_metric"],
            log_graph=config["logging"]["tensorboard"]["log_graph"],
        )
    else:
        # Default to TensorBoard if unknown logger
        logger = TensorBoardLogger(
            save_dir=config["logging"]["log_dir"],
            name=config["logging"]["name"],
        )
    
    # Create callbacks
    callbacks = [
        # Model checkpoint
        ModelCheckpoint(
            dirpath=config["checkpoint"]["dirpath"],
            filename=config["checkpoint"]["filename"],
            save_top_k=config["checkpoint"]["save_top_k"],
            monitor=config["checkpoint"]["monitor"],
            mode=config["checkpoint"]["mode"],
            auto_insert_metric_name=config["checkpoint"]["auto_insert_metric_name"],
            every_n_epochs=config["checkpoint"]["every_n_epochs"],
            save_last=config["checkpoint"]["save_last"],
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="step"),
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
        ),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        precision=config["training"]["precision"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        val_check_interval=config["training"]["val_check_interval"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )
    
    # Train model
    trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a modular context model")
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["core", "chat_instruct"], default="core",
        help="Type of model to train"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    train(args.config, args.model_type, args.resume)