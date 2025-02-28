from functools import partial
from typing import Callable, Iterator

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from omegaconf import DictConfig

from thesis.utils.lr_schedules import linear_warmup_linear_decay


class Schedulers(Stateful):
    """
    Utility class for calling step on multiple learning rate schedulers.
    """

    def __init__(self, optimizers: list[Optimizer], lr_lambda: Callable[[int], float]) -> None:
        self.schedulers: list[LambdaLR] = []

        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def __iter__(self) -> Iterator[LambdaLR]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def __getitem__(self, idx: int) -> LambdaLR:
        return self.schedulers[idx]

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, any]:
        # Currently, we have one scheduler per optimizer. However, when using MultiSchedule PP or optimizer-in-backward,
        # there are multiple optimizers and schedulers, but the scheduler state_dict remains the same for all.
        # Therefore, we only save the first one and later load it for all.
        assert len(self.schedulers) > 0, "Must have at least one scheduler to save state_dict"
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned with in scheduler.state_dict() is `last_epoch`,
        # which is an integer that will be automatically copied. As long as `training.steps` and `training.warmup_steps` remain
        # unchanged when resuming from a checkpoint, this approach is safe. We call `.copy()` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(state_dict.copy())


def build_lr_schedulers(optimizers: list[Optimizer], config: DictConfig) -> Schedulers:
    """Build schedulers from config."""
    scheduler_config = config.training.scheduler
    
    # Convert to int/float as needed
    warmup_steps = int(scheduler_config.num_warmup_steps)
    decay_steps = float(max(1, scheduler_config.num_steps - warmup_steps))

    lr_lambda = partial(linear_warmup_linear_decay, warmup_steps, decay_steps)
    return Schedulers(optimizers, lr_lambda)
