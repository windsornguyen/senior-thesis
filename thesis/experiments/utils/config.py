"""Utilities for config management and experiment tracking."""

import hashlib
import json
import inspect
import ast
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional

from thesis.experiments.models.transformer import Transformer

# Map of model types to their implementation classes
MODEL_IMPLEMENTATIONS = {
    "transformer": Transformer,
}

# Register compute functions for copy task
OmegaConf.register_new_resolver(
    "copy_compute_total_vocab",
    lambda vocab_size: vocab_size + 4,  # Add 4 special tokens (BOS, Delimiter, EOS, BLANK)
)

OmegaConf.register_new_resolver(
    "copy_compute_d_in",
    lambda vocab_size, one_hot: vocab_size + 4 if one_hot else 1,
)

OmegaConf.register_new_resolver(
    "copy_compute_seq_len",
    lambda copy_len, blank_len, selective: (
        copy_len + 3  # non-selective: copy_len + BOS + Delimiter + EOS
        if not selective
        else 1 + (copy_len + blank_len) + 2  # selective: BOS + interleaved + Delimiter + EOS
    ),
)


def normalize_config(config: DictConfig) -> Dict[str, Any]:
    """Normalize config by converting to dict and sorting keys recursively."""
    # Convert to dict for easier manipulation
    config_dict = OmegaConf.to_container(config, resolve=True)

    def _normalize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively normalize a dictionary."""
        result = {}
        for k, v in sorted(d.items()):  # Sort by keys
            if isinstance(v, dict):
                result[k] = _normalize_dict(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [_normalize_dict(x) if isinstance(x, dict) else x for x in v]
            else:
                result[k] = v
        return result

    return _normalize_dict(config_dict)


def get_normalized_ast(source_code: str) -> str:
    """Convert source code to a normalized AST representation."""
    tree = ast.parse(source_code)

    # Helper to convert AST nodes to a canonical form
    def normalize_node(node):
        if isinstance(node, ast.AST):
            # Get node fields excluding metadata like line numbers
            fields = {
                field: normalize_node(value)
                for field, value in ast.iter_fields(node)
                if field not in {"lineno", "col_offset", "end_lineno", "end_col_offset", "ctx"}
            }
            return (node.__class__.__name__, sorted(fields.items()))
        elif isinstance(node, list):
            return [normalize_node(item) for item in node]
        return node

    # Convert the entire AST to a normalized form
    return str(normalize_node(tree))


def get_model_code_hash(model_type: str) -> str:
    """Generate a hash of the model's implementation code using AST."""
    if model_type not in MODEL_IMPLEMENTATIONS:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = MODEL_IMPLEMENTATIONS[model_type]
    # Get the source code and convert to normalized AST
    source_code = inspect.getsource(model_class)
    normalized_ast = get_normalized_ast(source_code)

    # Generate hash of the normalized AST
    return hashlib.sha256(normalized_ast.encode()).hexdigest()[:8]


def get_config_hash(config: DictConfig) -> str:
    """Generate a deterministic hash from a config and model implementation."""
    # Get model implementation hash
    model_hash = get_model_code_hash(config.model_type.lower())

    # Normalize config
    normalized = normalize_config(config)

    # Add model hash to config
    normalized["_model_impl_hash"] = model_hash

    # Convert to JSON string with sorted keys
    config_str = json.dumps(normalized, sort_keys=True)

    # Generate hash
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]  # first 12 chars are enough


def get_experiment_dir(config: DictConfig, base_dir: str = "results") -> Path:
    """Get the experiment directory based on config hash."""
    config_hash = get_config_hash(config)

    # Create a meaningful name combining task and model
    exp_name = f"{config.task.name}_{config.model_type}"

    # Final path: results/task_model_hash
    return Path(base_dir) / f"{exp_name}_{config_hash}"


def check_experiment_exists(config: DictConfig, base_dir: str = "results") -> Optional[Path]:
    """Check if an experiment with this config already exists.

    Returns:
        Optional[Path]: Path to existing experiment dir if found, None otherwise
    """
    exp_dir = get_experiment_dir(config, base_dir)

    # Check if directory exists and has a completed flag or results
    if exp_dir.exists():
        # You might want to check for specific files that indicate completion
        if (exp_dir / "completed.flag").exists() or list(exp_dir.glob("*.pt")):
            return exp_dir

    return None


def save_experiment_config(config: DictConfig, exp_dir: Path) -> None:
    """Save the experiment config for reproducibility."""
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save both the raw and resolved configs
    OmegaConf.save(config, exp_dir / "config.yaml")

    # Save resolved config (with all interpolations resolved)
    resolved_config = OmegaConf.to_container(config, resolve=True)
    with open(exp_dir / "config_resolved.json", "w") as f:
        json.dump(resolved_config, f, indent=2)
