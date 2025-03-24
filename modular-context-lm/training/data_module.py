"""
PyTorch Lightning DataModule for handling the training data for all modules.
"""

import os
import json
import glob
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer


@dataclass
class ChatExample:
    """A single chat example for training the chat-instruct module."""
    
    input_text: str
    response_text: str
    role: Optional[str] = None
    instruction_type: Optional[str] = None


@dataclass
class TextExample:
    """A single text example for training the core model."""
    
    text: str


class CoreDataset(Dataset):
    """Dataset for training the core model."""
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        text_key: str = "text",
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        
        # Load examples
        self.examples = self._load_examples()
        
    def _load_examples(self) -> List[TextExample]:
        """Load examples from files."""
        examples = []
        
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if self.text_key in data:
                            examples.append(TextExample(text=data[self.text_key]))
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
                        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add labels for language modeling (same as input_ids)
        item["labels"] = item["input_ids"].clone()
        
        return item


class ChatInstructDataset(Dataset):
    """Dataset for training the chat-instruct module."""
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        input_key: str = "input",
        response_key: str = "response",
        role_key: str = "role",
        instruction_key: str = "instruction_type",
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_key = input_key
        self.response_key = response_key
        self.role_key = role_key
        self.instruction_key = instruction_key
        
        # Define mappings for role and instruction IDs
        self.role_to_id = {
            "system": 0,
            "user": 1,
            "assistant": 2,
            "none": 3,
        }
        
        self.instruction_to_id = {
            "general": 0,
            "summarize": 1,
            "explain": 2,
            "translate": 3,
            "code": 4,
            "create": 5,
            "answer": 6,
            "none": 7,
        }
        
        # Load examples
        self.examples = self._load_examples()
        
    def _load_examples(self) -> List[ChatExample]:
        """Load examples from files."""
        examples = []
        
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        # Check for required fields
                        if self.input_key in data and self.response_key in data:
                            role = data.get(self.role_key, "none")
                            instruction_type = data.get(self.instruction_key, "none")
                            
                            examples.append(ChatExample(
                                input_text=data[self.input_key],
                                response_text=data[self.response_key],
                                role=role,
                                instruction_type=instruction_type,
                            ))
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
                        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Format input and response
        full_text = f"{example.input_text}\n{example.response_text}"
        
        # Tokenize text
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add labels for language modeling (same as input_ids)
        item["labels"] = item["input_ids"].clone()
        
        # Create role_ids
        input_length = len(self.tokenizer.encode(example.input_text))
        role_id = self.role_to_id.get(example.role, self.role_to_id["none"])
        
        # All tokens in the input have the same role
        role_ids = torch.full_like(item["input_ids"], self.role_to_id["none"])
        # Set role for actual tokens (not padding)
        role_ids[:input_length] = role_id
        
        # Create instruction_ids
        instruction_id = self.instruction_to_id.get(
            example.instruction_type, self.instruction_to_id["none"]
        )
        instruction_ids = torch.tensor([instruction_id], dtype=torch.long)
        
        item["role_ids"] = role_ids
        item["instruction_ids"] = instruction_ids
        
        return item


class ModularContextDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the Modular Context-Specialized Network."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        module_type: str = "core",
    ):
        super().__init__()
        
        self.config = config
        self.module_type = module_type
        
        # Set paths based on config
        self.train_path = config["data"]["train_path"]
        self.val_path = config["data"]["val_path"]
        self.test_path = config.get("data", {}).get("test_path")
        
        # Set file pattern based on module type
        if module_type == "core":
            self.file_pattern = config["data"]["core_data"]["file_pattern"]
            self.text_key = config["data"]["core_data"]["text_key"]
            self.batch_size = config["training"]["core"]["batch_size"]
            self.max_length = config["training"]["core"]["max_length"] if "max_length" in config["training"]["core"] else config["data"]["max_seq_length"]
        elif module_type == "chat_instruct":
            self.file_pattern = config["data"]["chat_instruct_data"]["file_pattern"]
            self.input_key = config["data"]["chat_instruct_data"]["input_key"]
            self.response_key = config["data"]["chat_instruct_data"]["response_key"]
            self.role_key = config["data"]["chat_instruct_data"]["role_key"]
            self.instruction_key = config["data"]["chat_instruct_data"]["instruction_key"]
            self.batch_size = config["training"]["chat_instruct"]["batch_size"]
            self.max_length = config["training"]["chat_instruct"]["max_length"] if "max_length" in config["training"]["chat_instruct"] else config["data"]["max_seq_length"]
        else:
            raise ValueError(f"Unknown module type: {module_type}")
            
        # Other dataset parameters
        self.tokenizer_path = config["data"]["tokenizer_path"]
        self.num_workers = config["data"]["num_workers"]
        self.prefetch_factor = config["data"]["prefetch_factor"]
        
        # Create attributes for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tokenizer = None
        
    def _get_files(self, path: str) -> List[str]:
        """Get all matching files in a directory."""
        return glob.glob(os.path.join(path, self.file_pattern))
    
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for training, validation, and testing."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # Create datasets based on module type
        if self.module_type == "core":
            if stage == "fit" or stage is None:
                train_files = self._get_files(self.train_path)
                val_files = self._get_files(self.val_path)
                
                self.train_dataset = CoreDataset(
                    file_paths=train_files,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    text_key=self.text_key,
                )
                
                self.val_dataset = CoreDataset(
                    file_paths=val_files,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    text_key=self.text_key,
                )
                
            if stage == "test" and self.test_path:
                test_files = self._get_files(self.test_path)
                self.test_dataset = CoreDataset(
                    file_paths=test_files,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    text_key=self.text_key,
                )
                
        elif self.module_type == "chat_instruct":
            if stage == "fit" or stage is None:
                train_files = self._get_files(self.train_path)
                val_files = self._get_files(self.val_path)
                
                self.train_dataset = ChatInstructDataset(
                    file_paths=train_files,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    input_key=self.input_key,
                    response_key=self.response_key,
                    role_key=self.role_key,
                    instruction_key=self.instruction_key,
                )
                
                self.val_dataset = ChatInstructDataset(
                    file_paths=val_files,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    input_key=self.input_key,
                    response_key=self.response_key,
                    role_key=self.role_key,
                    instruction_key=self.instruction_key,
                )
                
            if stage == "test" and self.test_path:
                test_files = self._get_files(self.test_path)
                self.test_dataset = ChatInstructDataset(
                    file_paths=test_files,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    input_key=self.input_key,
                    response_key=self.response_key,
                    role_key=self.role_key,
                    instruction_key=self.instruction_key,
                )
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test DataLoader."""
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )