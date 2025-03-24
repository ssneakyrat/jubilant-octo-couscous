"""
Chat-Instruct module for the Modular Context-Specialized Network.
Handles conversational context and instruction following capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from model.core import CoreLayer, CoreConfig
from model.modules.module_base import ModuleBase, register_module


class ChatInstructConfig:
    """Configuration class for the Chat-Instruct module."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_cross_attention: bool = True,
        activation_function: str = "gelu_new",
        role_embeddings_size: int = 4,
        instruction_embeddings_size: int = 8,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.use_cross_attention = use_cross_attention
        self.activation_function = activation_function
        self.role_embeddings_size = role_embeddings_size
        self.instruction_embeddings_size = instruction_embeddings_size


class CrossAttention(nn.Module):
    """Cross-attention layer for attending between different streams."""
    
    def __init__(self, config: ChatInstructConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_attention_heads"
        
        # Initialize query, key, value projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        """Reshape tensor for attention computation."""
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length = hidden_states.shape[:2]
        encoder_seq_length = encoder_hidden_states.shape[1]
        
        # Get query from decoder, key/value from encoder
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, encoder_seq_length, batch_size)
        value_states = self._shape(value_states, encoder_seq_length, batch_size)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.hidden_size)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs


class ChatInstructLayer(nn.Module):
    """Layer combining self-attention, cross-attention (optional), and MLP."""
    
    def __init__(self, config: ChatInstructConfig, core_config: CoreConfig):
        super().__init__()
        
        # Self-attention and MLP from core
        core_config_for_layer = CoreConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_dropout_prob=config.attention_dropout_prob,
            activation_function=config.activation_function,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.self_attention_layer = CoreLayer(core_config_for_layer)
        
        # Cross attention (optional)
        self.use_cross_attention = config.use_cross_attention
        if self.use_cross_attention:
            self.cross_attention = CrossAttention(config)
            self.ln_cross = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # Self-attention block using core layer
        self_outputs = self.self_attention_layer(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        hidden_states = self_outputs[0]
        
        # Cross-attention block (if enabled and encoder states provided)
        cross_attn_weights = None
        if self.use_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.ln_cross(hidden_states)
            
            cross_outputs = self.cross_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            
            hidden_states = residual + cross_outputs[0]
            
            if output_attentions:
                cross_attn_weights = cross_outputs[1]
                
        # Collect outputs
        outputs = (hidden_states,)
        
        if output_attentions:
            # Add attention weights
            outputs += (self_outputs[1],)
            if cross_attn_weights is not None:
                outputs += (cross_attn_weights,)
                
        return outputs


@register_module
class ChatInstructModule(ModuleBase):
    """
    Specialized module for chat and instruction contexts.
    Adds role embeddings and instruction type embeddings to better handle
    conversational data and instruction following.
    """
    
    def __init__(
        self,
        name: str,
        config: ChatInstructConfig,
        core_config: CoreConfig,
    ):
        super().__init__(name, config.hidden_size)
        
        self.config = config
        self.core_config = core_config
        
        # Role embeddings (system, user, assistant, none)
        self.role_embeddings = nn.Embedding(
            config.role_embeddings_size, config.hidden_size
        )
        
        # Instruction type embeddings
        self.instruction_embeddings = nn.Embedding(
            config.instruction_embeddings_size, config.hidden_size
        )
        
        # Transformer layers (combination of self-attention and cross-attention)
        self.layers = nn.ModuleList([
            ChatInstructLayer(config, core_config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        instruction_ids: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Process hidden states with chat-specific features.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            role_ids: Optional tensor of shape (batch_size, seq_len) with role indices
            instruction_ids: Optional tensor with instruction type indices
            encoder_hidden_states: Optional tensor from another stream for cross-attention
            encoder_attention_mask: Optional attention mask for encoder states
            output_attentions: Whether to return attention weights
            **kwargs: Additional arguments
            
        Returns:
            Processed hidden states and any additional outputs
        """
        # Add role embeddings if provided
        if role_ids is not None:
            role_embeds = self.role_embeddings(role_ids)
            hidden_states = hidden_states + role_embeds
            
        # Add instruction embeddings if provided
        if instruction_ids is not None:
            instruction_embeds = self.instruction_embeddings(instruction_ids)
            # Broadcasting to add the same instruction embedding to all tokens
            if instruction_embeds.dim() == 2:
                instruction_embeds = instruction_embeds.unsqueeze(1)
            hidden_states = hidden_states + instruction_embeds
            
        # Process through transformer layers
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.use_cross_attention else None
        
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if len(layer_outputs) > 2:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
                    
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Prepare outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (all_self_attentions,)
            if all_cross_attentions is not None:
                outputs += (all_cross_attentions,)
                
        return outputs
    
    def get_module_config(self) -> Dict:
        """Get the configuration for this module to enable serialization."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
            "hidden_dropout_prob": self.config.hidden_dropout_prob,
            "attention_dropout_prob": self.config.attention_dropout_prob,
            "layer_norm_eps": self.config.layer_norm_eps,
            "use_cross_attention": self.config.use_cross_attention,
            "activation_function": self.config.activation_function,
            "role_embeddings_size": self.config.role_embeddings_size,
            "instruction_embeddings_size": self.config.instruction_embeddings_size,
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ChatInstructModule':
        """Create a module instance from a configuration dictionary."""
        # Create module config
        module_config = ChatInstructConfig(
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attention_dropout_prob=config["attention_dropout_prob"],
            layer_norm_eps=config["layer_norm_eps"],
            use_cross_attention=config["use_cross_attention"],
            activation_function=config["activation_function"],
            role_embeddings_size=config["role_embeddings_size"],
            instruction_embeddings_size=config["instruction_embeddings_size"],
        )
        
        # Create core config (only used for layer initialization)
        core_config = CoreConfig(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attention_dropout_prob=config["attention_dropout_prob"],
        )
        
        return cls(
            name=config["name"],
            config=module_config,
            core_config=core_config,
        )


class ModularWithChatInstruct(nn.Module):
    """
    Combined model with core transformer and chat instruction module.
    This is a convenience class for training and inference.
    """
    
    def __init__(
        self,
        core_model,
        chat_instruct_module,
    ):
        super().__init__()
        
        self.core_model = core_model
        self.chat_instruct_module = chat_instruct_module
        
        # Check compatibility
        assert core_model.config.hidden_size == chat_instruct_module.config.hidden_size, \
            "Core model and chat module must have the same hidden size"
            
        # LM head (using core model's)
        if hasattr(core_model, "lm_head") and core_model.lm_head is not None:
            self.lm_head = core_model.lm_head
        else:
            # If core model doesn't have one, create one
            self.lm_head = nn.Linear(
                core_model.config.hidden_size, 
                core_model.config.vocab_size, 
                bias=False
            )
            # Tie weights if core has tied weights
            if core_model.config.tie_word_embeddings:
                self.lm_head.weight = core_model.wte.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        instruction_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both the core model and chat instruction module.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            role_ids: Role IDs for tokens (system/user/assistant)
            instruction_ids: Instruction type IDs
            labels: Optional labels for computing loss
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with model outputs
        """
        # Forward through core model
        core_outputs = self.core_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = core_outputs["last_hidden_state"]
        
        # Forward through chat instruction module
        chat_outputs = self.chat_instruct_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            role_ids=role_ids,
            instruction_ids=instruction_ids,
            **kwargs
        )
        
        enhanced_hidden_states = chat_outputs[0]
        
        # Apply LM head
        if isinstance(self.lm_head, nn.Linear):
            lm_logits = self.lm_head(enhanced_hidden_states)
        else:
            # Function-based LM head (for weight tying)
            lm_logits = self.lm_head(enhanced_hidden_states)
            
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        # Organize outputs
        outputs = {
            "loss": loss,
            "logits": lm_logits,
            "core_hidden_states": hidden_states,
            "final_hidden_states": enhanced_hidden_states,
        }
        
        # Add any attention outputs if available
        if len(chat_outputs) > 1:
            outputs["chat_self_attentions"] = chat_outputs[1]
            
        if len(chat_outputs) > 2:
            outputs["chat_cross_attentions"] = chat_outputs[2]
            
        return outputs