"""
Logging utilities for the Modular Context-Specialized Network.
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Custom TensorBoard logger with additional features."""
    
    def __init__(
        self,
        log_dir: str,
        name: str,
        version: Optional[Union[int, str]] = None,
    ):
        """
        Initialize the TensorBoard logger.
        
        Args:
            log_dir: Directory to save logs
            name: Name of the experiment
            version: Version or run ID (if None, will be auto-generated)
        """
        self.log_dir = log_dir
        self.name = name
        
        # Auto-generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d-%H%M%S")
            
        # Create log directory
        self.version = version
        self.full_log_dir = os.path.join(log_dir, name, str(version))
        os.makedirs(self.full_log_dir, exist_ok=True)
        
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.full_log_dir)
        
        # Set up text logger
        self.logger = logging.getLogger(f"{name}_{version}")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        log_file = os.path.join(self.full_log_dir, "log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Track metrics for later plotting
        self.metrics = {}
        
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            name: Name of the scalar
            value: Value of the scalar
            step: Global step
        """
        self.writer.add_scalar(name, value, step)
        
        # Track metric for later plotting
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))
        
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars to TensorBoard.
        
        Args:
            main_tag: Main tag for the scalars
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Global step
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Track metrics for later plotting
        for tag, value in tag_scalar_dict.items():
            metric_name = f"{main_tag}/{tag}"
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append((step, value))
            
    def log_histogram(self, name: str, values: torch.Tensor, step: int, bins: str = "auto"):
        """
        Log a histogram to TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Tensor of values
            step: Global step
            bins: Number of bins for the histogram
        """
        self.writer.add_histogram(name, values, step, bins=bins)
        
    def log_text(self, name: str, text: str, step: int):
        """
        Log text to TensorBoard.
        
        Args:
            name: Name of the text
            text: Text to log
            step: Global step
        """
        self.writer.add_text(name, text, step)
        self.logger.info(f"{name} (step {step}): {text}")
        
    def log_figure(self, name: str, figure: plt.Figure, step: int):
        """
        Log a matplotlib figure to TensorBoard.
        
        Args:
            name: Name of the figure
            figure: Matplotlib figure
            step: Global step
        """
        self.writer.add_figure(name, figure, step)
        
        # Also save the figure as an image
        figure_dir = os.path.join(self.full_log_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, f"{name.replace('/', '_')}_{step}.png")
        figure.savefig(figure_path)
        
    def log_hyperparams(self, hparams: Dict, metrics: Optional[Dict] = None):
        """
        Log hyperparameters to TensorBoard.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of metric names
        """
        self.writer.add_hparams(hparams, metrics or {})
        
        # Also log to text file
        self.logger.info(f"Hyperparameters: {hparams}")
        if metrics:
            self.logger.info(f"Tracked Metrics: {metrics}")
            
    def log_model_graph(self, model, input_tensor):
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor
        """
        self.writer.add_graph(model, input_tensor)
        
    def log_embedding(self, name: str, embeddings, metadata=None, label_img=None, step: int = 0):
        """
        Log embeddings to TensorBoard.
        
        Args:
            name: Name of the embedding
            embeddings: Embedding tensor
            metadata: Metadata for the embeddings
            label_img: Images to display for the embeddings
            step: Global step
        """
        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            label_img=label_img,
            global_step=step,
            tag=name,
        )
        
    def log_pr_curve(self, name: str, labels, predictions, step: int):
        """
        Log precision-recall curve to TensorBoard.
        
        Args:
            name: Name of the curve
            labels: Ground truth labels
            predictions: Predicted probabilities
            step: Global step
        """
        self.writer.add_pr_curve(name, labels, predictions, step)
        
    def log_generation_samples(
        self,
        name: str,
        prompts: list,
        generations: list,
        step: int,
        max_samples: int = 10,
    ):
        """
        Log text generation samples to TensorBoard.
        
        Args:
            name: Name for the samples
            prompts: List of prompts
            generations: List of generated texts
            step: Global step
            max_samples: Maximum number of samples to log
        """
        # Limit number of samples to log
        num_samples = min(len(prompts), len(generations), max_samples)
        
        # Create markdown table
        md_table = "| Prompt | Generated Text |\n| --- | --- |\n"
        for i in range(num_samples):
            # Clean up text for markdown
            prompt = prompts[i].replace("\n", "<br>").replace("|", "\\|")
            generation = generations[i].replace("\n", "<br>").replace("|", "\\|")
            md_table += f"| {prompt} | {generation} |\n"
            
        # Log to TensorBoard
        self.log_text(f"{name}/generation_samples", md_table, step)
        
        # Also log to separate text file for easier viewing
        samples_dir = os.path.join(self.full_log_dir, "generation_samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        samples_path = os.path.join(samples_dir, f"{name.replace('/', '_')}_{step}_{timestamp}.txt")
        
        with open(samples_path, "w") as f:
            for i in range(num_samples):
                f.write(f"PROMPT [{i}]:\n{prompts[i]}\n\n")
                f.write(f"GENERATION [{i}]:\n{generations[i]}\n\n")
                f.write("-" * 80 + "\n\n")
                
    def plot_metrics(self, metrics: Optional[list] = None, save_path: Optional[str] = None):
        """
        Plot tracked metrics and save the figure.
        
        Args:
            metrics: List of metric names to plot (if None, plot all)
            save_path: Path to save the figure (if None, use default)
        """
        if not self.metrics:
            return
            
        # Use all metrics if none specified
        if metrics is None:
            metrics = list(self.metrics.keys())
            
        # Create figure with subplots
        n = len(metrics)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            if metric in self.metrics and self.metrics[metric]:
                steps, values = zip(*self.metrics[metric])
                axes[i].plot(steps, values)
                axes[i].set_title(metric)
                axes[i].set_ylabel("Value")
                axes[i].grid(True)
                
        # Set common x-axis label
        axes[-1].set_xlabel("Step")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            metrics_dir = os.path.join(self.full_log_dir, "metric_plots")
            os.makedirs(metrics_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(metrics_dir, f"metrics_{timestamp}.png")
            
        plt.savefig(save_path)
        
        return fig
        
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class MetricsTracker:
    """Class to track and compute metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
        
    def reset(self):
        """Reset all tracked metrics."""
        self.loss_sum = 0.0
        self.loss_count = 0
        self.perplexity_sum = 0.0
        self.perplexity_count = 0
        self.accuracy_correct = 0
        self.accuracy_total = 0
        
    def update_loss(self, loss: float, count: int = 1):
        """
        Update loss metric.
        
        Args:
            loss: Loss value
            count: Number of samples
        """
        self.loss_sum += loss * count
        self.loss_count += count
        
    def update_perplexity(self, perplexity: float, count: int = 1):
        """
        Update perplexity metric.
        
        Args:
            perplexity: Perplexity value
            count: Number of samples
        """
        self.perplexity_sum += perplexity * count
        self.perplexity_count += count
        
    def update_accuracy(self, correct: int, total: int):
        """
        Update accuracy metric.
        
        Args:
            correct: Number of correct predictions
            total: Total number of predictions
        """
        self.accuracy_correct += correct
        self.accuracy_total += total
        
    def get_avg_loss(self) -> float:
        """Get average loss."""
        return self.loss_sum / max(1, self.loss_count)
        
    def get_avg_perplexity(self) -> float:
        """Get average perplexity."""
        return self.perplexity_sum / max(1, self.perplexity_count)
        
    def get_accuracy(self) -> float:
        """Get accuracy."""
        return self.accuracy_correct / max(1, self.accuracy_total)
        
    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        metrics = {}
        
        if self.loss_count > 0:
            metrics["loss"] = self.get_avg_loss()
            
        if self.perplexity_count > 0:
            metrics["perplexity"] = self.get_avg_perplexity()
            
        if self.accuracy_total > 0:
            metrics["accuracy"] = self.get_accuracy()
            
        return metrics