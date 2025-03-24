"""
Core Transformer model for the Modular Context-Specialized Network.
This is the lightweight foundation that other modules build upon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class CoreConfig:
    """Configuration class for the core transformer model."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        activation_function: str = "gelu_new",
        tie_word_embeddings: bool = True,
        gradient_checkpointing: bool = False,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.activation_function = activation_function
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing


class CoreAttention(nn.Module):
    """Multi-head attention module for the core transformer."""
    
    def __init__(self, config: CoreConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
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
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # If using past key values, only compute keys and values for the new tokens
        if past_key_values is not None:
            past_key, past_value = past_key_values
            seq_length = past_key.shape[2] + hidden_states.shape[1]
        else:
            past_key, past_value = None, None
            
        # Get query, key, value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape query, key, value for multi-head attention
        query_states = self._shape(query_states, hidden_states.shape[1], batch_size)
        key_states = self._shape(key_states, hidden_states.shape[1], batch_size)
        value_states = self._shape(value_states, hidden_states.shape[1], batch_size)
        
        # Concatenate with past key values if provided
        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
        # Cache current key and value if requested
        if use_cache:
            present = (key_states, value_states)
        else:
            present = None
            
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.hidden_size)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs


class CoreMLP(nn.Module):
    """MLP layer for transformer blocks."""
    
    def __init__(self, config: CoreConfig):
        super().__init__()
        
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if config.activation_function == "gelu_new":
            self.act_fn = self._gelu_new
        elif config.activation_function == "gelu":
            self.act_fn = F.gelu
        elif config.activation_function == "relu":
            self.act_fn = F.relu
        elif config.activation_function == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {config.activation_function}")
            
    def _gelu_new(self, x):
        """Implementation of the GELU activation function used in some newer models."""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CoreLayer(nn.Module):
    """Transformer layer combining attention and MLP."""
    
    def __init__(self, config: CoreConfig):
        super().__init__()
        
        self.attention = CoreAttention(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CoreMLP(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]  # Add attentions if output_attentions=True
        
        # Skip connection
        hidden_states = residual + attn_output
        
        # MLP block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        
        # Skip connection
        hidden_states = residual + mlp_output
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
            
        return outputs


class CoreModel(nn.Module):
    """Core transformer model serving as the foundation for modular extensions."""
    
    def __init__(self, config: CoreConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            CoreLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def get_input_embeddings(self):
        return self.wte
        
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        # Set defaults
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        # Process inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
            
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
            
        # Initialize past length
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Token type IDs (for segment embeddings if needed)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
            
        # Attention mask
        if attention_mask is not None:
            # Convert binary mask to causal attention mask
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            elif attention_mask.dim() == 3:
                extended_attention_mask = attention_mask.unsqueeze(1)
            else:
                raise ValueError(f"Attention mask should be 2D or 3D, got {attention_mask.dim()}D")
        else:
            # Create causal mask
            extended_attention_mask = None
            
        # Head mask
        if head_mask is not None:
            raise NotImplementedError("Head masking is not implemented yet")
        else:
            head_mask = [None] * self.config.num_hidden_layers
            
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            
        position_embeds = self.wpe(position_ids)
        
        # Combine token and position embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Process through transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_present_key_values = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Get past key values for this layer
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Apply gradient checkpointing if enabled
            if self.config.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    extended_attention_mask,
                    head_mask[i],
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask[i],
                    past_key_values=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
            hidden_states = layer_outputs[0]
            
            if use_cache:
                all_present_key_values = all_present_key_values + (layer_outputs[1],)
                
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)
                
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Add hidden states to outputs if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Organize outputs as a dictionary or tuple
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": all_present_key_values,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
        else:
            return tuple(filter(lambda x: x is not None, [
                hidden_states, all_present_key_values, all_hidden_states, all_attentions
            ]))
            
    def _gradient_checkpointing_func(self, *args, **kwargs):
        """Wrapper for gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(*args, **kwargs)


class CoreLMHeadModel(nn.Module):
    """Core language model with a language modeling head."""
    
    def __init__(self, config: CoreConfig):
        super().__init__()
        
        self.config = config
        self.transformer = CoreModel(config)
        
        # Add LM head
        if config.tie_word_embeddings:
            self.lm_head = lambda x: F.linear(x, self.transformer.wte.weight)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def get_output_embeddings(self):
        """Get the output embeddings."""
        if self.config.tie_word_embeddings:
            return self.transformer.wte
        else:
            return self.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings."""
        if self.config.tie_word_embeddings:
            self.transformer.wte = new_embeddings
        else:
            self.lm_head = new_embeddings
            
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=None,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=None,
        eos_token_id=None,
        use_cache=True,
        **kwargs
    ):      
        
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
    
        # Set default values if not provided
        if max_new_tokens is None:
            max_new_tokens = 20
    
        # Set up padding token
        if pad_token_id is None:
            pad_token_id = 0  # Default padding token
    
        # Initialize past key values for faster generation
        past_key_values = None
    
        # Create new attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
    
        # Start with the input_ids
        output_ids = input_ids.clone()
    
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Forward pass through model
            with torch.no_grad():
                if past_key_values is None:
                    outputs = self(
                        input_ids=output_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        **kwargs
                    )
                else:
                    # Only process the last token with cached key values
                    outputs = self(
                        input_ids=output_ids[:, -1].unsqueeze(-1),
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        **kwargs
                    )
        
        # Get logits for the next token
        if isinstance(outputs, dict):
            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs.get("past_key_values", None)
        else:
            next_token_logits = outputs[0][:, -1, :]
            past_key_values = outputs[1] if len(outputs) > 1 else None
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size):
                # Get the tokens that have been generated so far
                prev_tokens = output_ids[i].tolist()
                # Apply penalty to already seen tokens
                for prev_token in set(prev_tokens):
                    next_token_logits[i, prev_token] /= repetition_penalty
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            # Get top-k indices
            topk_values, topk_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)), dim=-1)
            # Zero out other values
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            # Scatter top-k values back
            next_token_logits.scatter_(-1, topk_indices, topk_values)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            # Sort the logits in descending order
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create a mask for indices to remove
            for i in range(batch_size):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                next_token_logits[i, indices_to_remove] = float('-inf')
        
                # Sample from the filtered distribution
                if do_sample:
                    # Use softmax to get probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    # Sample from the distribution
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append new tokens to output
                output_ids = torch.cat([output_ids, next_tokens], dim=-1)
                
                # Update attention mask to include the new token
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=-1)
                
                # Check if all sequences have reached the end token (if provided)
                if eos_token_id is not None:
                    if (next_tokens == eos_token_id).all():
                        break

        return output_ids

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else False
        
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Always use dict internally
        )
        
        # Get hidden states
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Apply LM head
        if isinstance(self.lm_head, nn.Linear):
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
            
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
        if return_dict:
            return {
                "loss": loss,
                "logits": lm_logits,
                "past_key_values": transformer_outputs.get("past_key_values", None),
                "hidden_states": transformer_outputs.get("hidden_states", None),
                "attentions": transformer_outputs.get("attentions", None),
            }
        else:
            if loss is not None:
                return (loss, lm_logits) + tuple(v for k, v in transformer_outputs.items() if k != "last_hidden_state")
            else:
                return (lm_logits,) + tuple(v for k, v in transformer_outputs.items() if k != "last_hidden_state")
            
    