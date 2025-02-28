import torch
from torch.optim import Optimizer


class MultiplicativeWeights(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.1, initial_weight=1.0):
        """
        Initializes the MultiplicativeWeights optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Base learning rate. Default: 1e-3.
            alpha (float, optional): Learning rate parameter for multiplicative update. Default: 0.1.
            initial_weight (float, optional): Initial weight for each parameter group. Default: 1.0.
        """
        defaults = dict(lr=lr, alpha=alpha, weight=initial_weight)
        super(MultiplicativeWeights, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_factors):
        """
        Performs a single optimization step using multiplicative weights update.

        Args:
            loss_factors (list or tensor): A list or tensor containing performance metrics
                                           (e.g., losses) for each parameter group.
        """
        if not isinstance(loss_factors, (list, tuple, torch.Tensor)):
            raise TypeError("loss_factors should be a list, tuple, or torch.Tensor")

        if len(loss_factors) != len(self.param_groups):
            raise ValueError("Number of loss_factors must match number of parameter groups")

        # Convert loss_factors to a tensor for vectorized operations
        loss_factors = torch.tensor(loss_factors, dtype=torch.float32, device=self.param_groups[0]["params"][0].device)

        # Extract current weights
        weights = torch.tensor(
            [group["weight"] for group in self.param_groups], dtype=torch.float32, device=loss_factors.device
        )

        # Compute updated weights multiplicatively based on loss factors
        # Here, higher loss implies lower weight
        updated_weights = weights * torch.exp(-self.defaults["alpha"] * loss_factors)

        # Normalize the weights to sum to 1
        updated_weights = updated_weights / updated_weights.sum()

        # Update the weight in each parameter group
        for group, new_weight in zip(self.param_groups, updated_weights, strict=True):
            group["weight"] = new_weight.item()

        # Update the learning rate for each parameter group based on the new weights
        for group in self.param_groups:
            group_lr = self.defaults["lr"] * group["weight"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Perform standard gradient descent update scaled by group-specific learning rate
                p.add_(-group_lr * p.grad)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
