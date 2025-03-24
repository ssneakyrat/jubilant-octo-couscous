"""
Base class for all specialized modules in the Modular Context-Specialized Network.
This defines the common interface that all modules must implement.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

class ModuleBase(nn.Module, ABC):
    """Abstract base class for all specialized modules."""
    
    def __init__(self, name: str, hidden_size: int):
        """
        Initialize the base module.
        
        Args:
            name: A unique identifier for this module
            hidden_size: The dimension of hidden states
        """
        super().__init__()
        self.name = name
        self.hidden_size = hidden_size
        
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Process the hidden states and produce module-specific outputs.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            **kwargs: Additional module-specific arguments
            
        Returns:
            Processed hidden states and any additional module-specific outputs
        """
        pass
    
    @abstractmethod
    def get_module_config(self) -> Dict:
        """
        Get the configuration for this module to enable serialization.
        
        Returns:
            A dictionary containing the module configuration
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict) -> 'ModuleBase':
        """
        Create a module instance from a configuration dictionary.
        
        Args:
            config: A dictionary containing the module configuration
            
        Returns:
            An initialized module instance
        """
        pass
    
    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """
        Get a state dict for saving the module, potentially excluding
        specific parameters.
        
        Returns:
            A state dict for saving
        """
        return self.state_dict()
    
    def get_input_constraints(self) -> Dict:
        """
        Get constraints on the input this module accepts.
        
        Returns:
            A dictionary of constraints (e.g., expected tensor shapes)
        """
        return {
            "hidden_size": self.hidden_size,
        }
    
    def can_process(self, hidden_states: torch.Tensor) -> bool:
        """
        Check if this module can process the given hidden states.
        
        Args:
            hidden_states: Tensor to check
            
        Returns:
            True if this module can process the hidden states, False otherwise
        """
        return hidden_states.size(-1) == self.hidden_size


class ModuleRegistry:
    """Registry for all available modules."""
    
    _modules = {}
    
    @classmethod
    def register(cls, module_class: type):
        """
        Register a module class.
        
        Args:
            module_class: The module class to register
        """
        cls._modules[module_class.__name__] = module_class
        return module_class
    
    @classmethod
    def get_module_class(cls, module_name: str) -> type:
        """
        Get a module class by name.
        
        Args:
            module_name: The name of the module class
            
        Returns:
            The module class
            
        Raises:
            KeyError: If the module class is not registered
        """
        if module_name not in cls._modules:
            raise KeyError(f"Module class '{module_name}' not registered")
        return cls._modules[module_name]
    
    @classmethod
    def list_modules(cls) -> List[str]:
        """
        List all registered module classes.
        
        Returns:
            A list of module class names
        """
        return list(cls._modules.keys())
    
    @classmethod
    def create_from_config(cls, config: Dict) -> ModuleBase:
        """
        Create a module instance from a configuration dictionary.
        
        Args:
            config: A dictionary containing the module configuration
            
        Returns:
            An initialized module instance
            
        Raises:
            KeyError: If the module class is not registered
        """
        module_type = config.get("type")
        if module_type not in cls._modules:
            raise KeyError(f"Module class '{module_type}' not registered")
        
        module_class = cls._modules[module_type]
        return module_class.from_config(config)


def register_module(cls):
    """
    Decorator to register a module class.
    
    Args:
        cls: The module class to register
        
    Returns:
        The module class
    """
    return ModuleRegistry.register(cls)