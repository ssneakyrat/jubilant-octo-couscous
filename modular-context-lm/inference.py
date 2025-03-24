"""
Inference script for the Modular Context-Specialized Network.
Generates text using a trained model.
"""

import os
import argparse
import yaml
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union

from model.core import CoreConfig, CoreLMHeadModel
from model.modules.chat_instruct import ChatInstructConfig, ChatInstructModule, ModularWithChatInstruct
from transformers import AutoTokenizer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ModularContextInference:
    """Class for generating text with trained models."""
    
    def __init__(
        self,
        config_path: str,
        core_checkpoint_path: Optional[str] = None,
        chat_instruct_checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the inference class.
        
        Args:
            config_path: Path to the config file
            core_checkpoint_path: Path to the core model checkpoint
            chat_instruct_checkpoint_path: Path to the chat-instruct module checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_path"])
        
        # Load models
        self.core_model = self._load_core_model(core_checkpoint_path)
        
        # Load chat-instruct module if checkpoint is provided
        self.chat_instruct_module = None
        if chat_instruct_checkpoint_path is not None:
            self.chat_instruct_module = self._load_chat_instruct_module(chat_instruct_checkpoint_path)
            
        # Create combined model if both are loaded
        self.combined_model = None
        if self.core_model is not None and self.chat_instruct_module is not None:
            self.combined_model = ModularWithChatInstruct(
                core_model=self.core_model,
                chat_instruct_module=self.chat_instruct_module,
            ).to(self.device)
            self.combined_model.eval()
            
    def _load_core_model(self, checkpoint_path: Optional[str] = None) -> Optional[CoreLMHeadModel]:
        """Load the core model from a checkpoint."""
        # Create core config
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
            gradient_checkpointing=False,  # Disable for inference
        )
        
        # Create model
        model = CoreLMHeadModel(core_config).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.info(f"Loading core model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
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
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        elif checkpoint_path is not None:
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return None
        else:
            logger.info("Using untrained core model")
            model.eval()
            return model
            
    def _load_chat_instruct_module(self, checkpoint_path: Optional[str] = None) -> Optional[ChatInstructModule]:
        """Load the chat-instruct module from a checkpoint."""
        if checkpoint_path is None:
            return None
            
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
        
        # Create core config (for layer initialization)
        core_config = CoreConfig(
            hidden_size=self.config["model"]["core"]["hidden_size"],
            num_attention_heads=self.config["model"]["core"]["num_attention_heads"],
            intermediate_size=self.config["model"]["core"]["intermediate_size"],
            hidden_dropout_prob=self.config["model"]["core"]["hidden_dropout_prob"],
            attention_dropout_prob=self.config["model"]["core"]["attention_dropout_prob"],
        )
        
        # Create module
        module = ChatInstructModule(
            name="chat_instruct",
            config=chat_config,
            core_config=core_config,
        ).to(self.device)
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading chat-instruct module from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                # Lightning checkpoint
                state_dict = checkpoint["state_dict"]
                
                # Extract just the chat module weights
                chat_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model.chat_instruct_module."):
                        # Remove prefix
                        chat_state_dict[k.replace("model.chat_instruct_module.", "")] = v
                        
                state_dict = chat_state_dict
            
            # Load weights
            module.load_state_dict(state_dict, strict=False)
            module.eval()
            return module
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return None
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        use_chat_module: bool = True,
        role: Optional[str] = None,
        instruction_type: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt
            max_length: Maximum length of the generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            use_chat_module: Whether to use the chat-instruct module (if loaded)
            role: Role for the chat-instruct module (system, user, assistant, none)
            instruction_type: Instruction type for the chat-instruct module
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Choose which model to use
        if use_chat_module and self.combined_model is not None:
            model_to_use = self.combined_model
            
            # Prepare additional inputs for chat module
            if role is not None:
                # Convert role to IDs
                role_mapping = {
                    "system": 0,
                    "user": 1,
                    "assistant": 2,
                    "none": 3,
                }
                role_id = role_mapping.get(role.lower(), 3)
                
                # Create role_ids tensor (same role for all tokens)
                role_ids = torch.full_like(inputs["input_ids"], role_id)
                inputs["role_ids"] = role_ids
                
            if instruction_type is not None:
                # Convert instruction type to IDs
                instruction_mapping = {
                    "general": 0,
                    "summarize": 1,
                    "explain": 2,
                    "translate": 3,
                    "code": 4,
                    "create": 5,
                    "answer": 6,
                    "none": 7,
                }
                instruction_id = instruction_mapping.get(instruction_type.lower(), 7)
                
                # Create instruction_ids tensor (batch size, 1)
                instruction_ids = torch.tensor([[instruction_id]], device=self.device)
                inputs["instruction_ids"] = instruction_ids
        else:
            model_to_use = self.core_model
            
        # Set generation parameters from config or use defaults
        gen_config = self.config.get("inference", {})
        temp = gen_config.get("temperature", temperature)
        p = gen_config.get("top_p", top_p)
        k = gen_config.get("top_k", top_k)
        rep_penalty = gen_config.get("repetition_penalty", repetition_penalty)
        max_new = gen_config.get("max_new_tokens", max_length)
        do_sample = gen_config.get("do_sample", True)
        
        # Generate text
        with torch.no_grad():
            output_sequences = model_to_use.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temp,
                top_p=p,
                top_k=k,
                repetition_penalty=rep_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(
            output_sequences[0], skip_special_tokens=True
        )
        
        # Remove the prompt from the generated text if it starts with it
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
            
        return generated_text
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response in a chat format.
        
        Args:
            messages: List of chat messages, each with 'role' and 'content'
            max_length: Maximum length of the generated text
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        if self.combined_model is None:
            raise ValueError("Chat-instruct module is not loaded, cannot use chat function")
            
        # Format messages into a chat prompt
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role.capitalize()}: {content}\n"
            
        prompt += "Assistant: "
        
        # The last non-assistant message determines the role and instruction type
        for msg in reversed(messages):
            if msg.get('role') != 'assistant':
                role = msg.get('role')
                # Try to infer instruction type from content
                content = msg.get('content', '').lower()
                if "explain" in content or "how" in content:
                    instruction_type = "explain"
                elif "summarize" in content or "summary" in content:
                    instruction_type = "summarize"
                elif "code" in content or "function" in content or "program" in content:
                    instruction_type = "code"
                elif "write" in content or "create" in content:
                    instruction_type = "create"
                elif any(q in content for q in ["what", "who", "where", "when", "why"]):
                    instruction_type = "answer"
                else:
                    instruction_type = "general"
                break
        else:
            role = "user"
            instruction_type = "general"
            
        # Generate response
        response = self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            use_chat_module=True,
            role=role,
            instruction_type=instruction_type,
        )
        
        return response


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate text with a trained model")
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--core_checkpoint", type=str, default=None,
        help="Path to core model checkpoint"
    )
    parser.add_argument(
        "--chat_checkpoint", type=str, default=None,
        help="Path to chat-instruct module checkpoint"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--use_chat", action="store_true",
        help="Use chat-instruct module if available"
    )
    parser.add_argument(
        "--role", type=str, default=None,
        choices=["system", "user", "assistant", "none"],
        help="Role for chat-instruct module"
    )
    parser.add_argument(
        "--instruction_type", type=str, default=None,
        choices=["general", "summarize", "explain", "translate", "code", "create", "answer", "none"],
        help="Instruction type for chat-instruct module"
    )
    parser.add_argument(
        "--chat_mode", action="store_true",
        help="Enter an interactive chat mode"
    )
    
    args = parser.parse_args()
    
    # Create inference object
    inference = ModularContextInference(
        config_path=args.config,
        core_checkpoint_path=args.core_checkpoint,
        chat_instruct_checkpoint_path=args.chat_checkpoint,
    )
    
    if args.chat_mode:
        print("Entering chat mode. Type 'exit' to quit.")
        messages = []
        
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
                
            messages.append({"role": "user", "content": user_input})
            response = inference.chat(messages, max_length=args.max_length, temperature=args.temperature)
            print(f"Assistant: {response}")
            
            messages.append({"role": "assistant", "content": response})
    elif args.prompt:
        # Generate text from prompt
        generated_text = inference.generate_text(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            use_chat_module=args.use_chat,
            role=args.role,
            instruction_type=args.instruction_type,
        )
        
        print(f"Generated text: {generated_text}")
    else:
        print("No prompt provided. Use --prompt or --chat_mode.")


if __name__ == "__main__":
    main()