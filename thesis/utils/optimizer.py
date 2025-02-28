import functools
import torch
import torch.nn as nn

from typing import Iterator

from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.optim import Optimizer

# TODO: Add the low-precision optimizers from torchao

from omegaconf import DictConfig

class Optimizers(Stateful):
    """
    Utility class for calling step/zero_grad on multiple optimizers needed for
    virtual pipeline stages and saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(self, model_parts: list[nn.Module], name: str, optim_kwargs: dict[str, any]) -> None:
        self.optimizers: list[Optimizer] = []
        self.model_parts = model_parts

        for model in self.model_parts:
            if name == "Adam":
                # TODO: Make the optim options configurable by toml/cmd args
                optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
            elif name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
            else:
                raise NotImplementedError(f"Optimizer not added: {name}")
            self.optimizers.append(optimizer)
        self._validate_length(len(self.model_parts))

    def __iter__(self):
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def __getitem__(self, idx: int) -> Optimizer:
        return self.optimizers[idx]

    def _validate_length(self, expected_length) -> None:
        assert expected_length == len(
            self.optimizers
        ), "Must pass one optimizer per model part or per param if using OptimizersBwd"

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    @property
    def param_groups(self) -> list:
        return [pg for opt in self.optimizers for pg in opt.param_groups]

    def state_dict(self) -> dict[str, any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for state_dict in map(func, self.model_parts, self.optimizers) for k, v in state_dict.items()}

    def load_state_dict(self, state_dict: dict[str, any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))


class OptimizersBwd(Optimizers):
    """
    Optimizers in backward to skip .step() and .zero_grad().
    """

    def __init__(self, model_parts: list[nn.Module], name: str, optim_kwargs: dict[str, any]) -> None:
        self.optimizers: list[Optimizer] = []
        self.model_parts = model_parts
        optim_dict = {}
        for model in self.model_parts:
            if name == "Adam":
                # TODO: Make the optimizer options configurable by toml/cmd args
                optim_dict.update({param: torch.optim.Adam([param], **optim_kwargs) for param in model.parameters()})
            elif name == "AdamW":
                optim_dict.update({param: torch.optim.AdamW([param], **optim_kwargs) for param in model.parameters()})
            else:
                raise NotImplementedError(f"Optimizer not added: {name}")

        def __iter__(self) -> Iterator[Optimizer]:
            return iter(self.optimizers)

        def __len__(self) -> int:
            return len(self.optimizers)

        def __getitem__(self, idx: int) -> Optimizer:
            return self.optimizers[idx]

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

            self.optimizers.extend([optim_dict[param] for param in model.parameters()])

        self._validate_length(sum(len([param for param in model.parameters()]) for model in self.model_parts))

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


# TODO: Consider split between pp and non-pp
def build_optimizers(model_parts: list[nn.Module], config: DictConfig):
    """
    Wrap one optimizer per model part in an Optimizers class which provides
    a single step() and zero_grad() method for all the children optimizers.
    """
    # Get optimizer config from training section
    optim_config = config.training.optimizer

    # Check for early step in backward (if configured)
    optim_in_bwd = optim_config.get("early_step_in_backward", False)
    if optim_in_bwd and config.distributed.parallelism.get("pipeline_parallel_degree", 1) > 1:
        raise NotImplementedError("Optimizers in backward is not supported with pipeline parallelism.")

    name = optim_config.name
    kwargs = {
        "lr": float(optim_config.lr),  # ensure float
        "betas": tuple(optim_config.betas),
        "eps": float(optim_config.eps),
        "weight_decay": float(optim_config.weight_decay),
        "amsgrad": optim_config.get("amsgrad", False),
        "fused": optim_config.get("fused", False),
        "foreach": not optim_config.get("fused", False),  # foreach and fused are mutually exclusive
    }

    return Optimizers(model_parts, name, kwargs) if not optim_in_bwd else OptimizersBwd(model_parts, name, kwargs)
