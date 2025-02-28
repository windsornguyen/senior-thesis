import torch

from typing import Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig
from pathlib import Path

from thesis.experiments.synthetics.induction_heads import generate_induction_heads
from thesis.experiments.synthetics.copy import CopyDataset


def create_dataset(config: DictConfig) -> Tuple[Dataset, Dict[str, int]]:
    """Create dataset based on config.
    Also validates and returns the expected input/output shapes for model validation.

    Args:
        config (DictConfig): The experiment configuration.

    Returns:
        Tuple[Dataset, Dict[str, int]]: The dataset and expected shapes dictionary.
    """
    task_name = config.task.name.lower()
    task_module = None

    # Dynamically import the task module based on task name
    try:
        if task_name == "copy":
            from thesis.experiments.synthetics.copy import CopyDataset as TaskDataset
        elif task_name == "induction_heads":
            from thesis.experiments.synthetics.induction_heads import InductionHeadsDataset as TaskDataset
        # Add more tasks here as needed
        else:
            raise ValueError(f"Unknown task: {task_name}")
    except ImportError as e:
        raise ImportError(f"Failed to import dataset for task '{task_name}': {e}")

    # Create dataset instance
    dataset = TaskDataset(**config.task.params)

    # Get expected shapes from the dataset
    # Each dataset class should implement get_expected_shapes()
    expected_shapes = dataset.get_expected_shapes()

    return dataset, expected_shapes


def create_train_val_dataloaders(
    dataset_and_shapes: tuple, batch_size: int, val_split: float = 0.05, pin_memory: bool = True
):
    """Create train and validation dataloaders.

    Args:
        dataset_and_shapes (tuple): Tuple of (dataset, expected_shapes)
        batch_size (int): Batch size for dataloaders
        val_split (float, optional): Fraction of data to use for validation. Defaults to 0.05.
        pin_memory (bool, optional): Whether to pin memory for GPU training. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader, Dict]: Train loader, val loader, and expected shapes
    """
    dataset, expected_shapes = dataset_and_shapes
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, val_loader, expected_shapes
