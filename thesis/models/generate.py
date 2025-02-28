"""Base class for packed causal generation."""

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from thesis.models.tokenizer import Tokenizer
from safetensors.torch import load_file


@dataclass
class PackedCausalGeneratorConfig:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stream: bool = True
    max_gen_len: int = 512  # Maximum number of tokens to generate
    max_tokens: int = 1024  # Maximum number of tokens that can go through the model
    max_prompt_len: Optional[int] = None
    until: List[str] = field(default_factory=list)
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False # How many batches of prompts have been processed
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"


class BasePackedCausalGenerator(ABC):
    def __init__(
        self,
        config: PackedCausalGeneratorConfig,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        super().__init__()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        self.temperature = getattr(config, "temperature", 1.0)
        self.top_p = getattr(config, "top_p", 1.0)
        self.top_k = getattr(config, "top_k", 0)
        self.stream = getattr(config, "stream", True)

        self.max_gen_len = getattr(config, "max_gen_len", 128)
        self.max_tokens = getattr(config, "max_tokens", 2048)
        self.max_prompt_len = getattr(config, "max_prompt_len", None)
        self.until = getattr(config, "until", None)
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = getattr(config, "device", "cuda")

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not config.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not config.reduce_generation_overhead,
        )

        self.dtype = getattr(config, "dtype", "fp32")  # e.g. 'fp32' or 'bf16'
        self.show_progress = getattr(config, "show_progress", False)

        # Internal trackers
        self.prefill_doc_id = None
        self.prefill_tok_id = None
        self.padded_doc_id = None
        self.padded_tok_id = None
        self.current_doc_id = None
        self.current_tok_id = None
        self.padded_doc_start = None
        self.prefill_mask = None

    @abstractmethod
    def clear_cache(self, offset: torch.Tensor):
        pass

    @abstractmethod
    def setup_prefilling(self, lengths: torch.Tensor):
        pass

    @abstractmethod
    def setup_generation(self, lengths: torch.Tensor):
        pass

    @abstractmethod
    def prefill(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    def generate_next_token(self, current_token: torch.Tensor, *args, **kwargs):
        pass

    @torch.inference_mode()
    def generate(self, prompts: List[str], *args, **kwargs):
        """Placeholder implementation. Subclass should fill in logic."""
        generation, loglikelihood, greedy = [], [], []
        return generation, loglikelihood, greedy

def load_model(
    job_config,
    checkpoint_path: str,
    models_config: dict,
    model_name_to_cls: dict,
    use_safetensors: bool = True,
):
    model_name = job_config.model.name
    variant = job_config.model.variant
    model_dtype_str = getattr(job_config.training, "model_dtype", "bfloat16")

    if model_name not in models_config:
        raise KeyError(f"Model '{model_name}' not found in models_config.")
    if variant not in models_config[model_name]:
        raise KeyError(f"Variant '{variant}' not found for model '{model_name}'.")
    model_config = models_config[model_name][variant]

    if model_name not in model_name_to_cls:
        raise KeyError(f"No class found for model_name '{model_name}'.")
    model_cls = model_name_to_cls[model_name]
    model = model_cls.from_model_args(model_config)

    if use_safetensors:
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "model" in state_dict:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    str_to_torch_dtype = {
        "fp32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "float8": torch.float8,
    }
    param_dtype = str_to_torch_dtype.get(model_dtype_str, torch.float32)
    model.to(device=device, dtype=param_dtype)
    model.eval()

    return model


def build_tokenizer_stub():
    """
    Minimal placeholder tokenizer builder. Replace with your own logic,
    e.g. HF tokenizer or a custom built tokenizer from job_config paths.
    """
    return object()
