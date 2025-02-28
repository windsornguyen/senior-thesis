import torch
import torch.nn as nn
import math
from flash_stu.layers.stu_layer import STULayer
from training.losses.chunked_cross_entropy import ChunkedCrossEntropyLoss

class FLH2(nn.Module):
    def __init__(self, alpha=0.1) -> None:
        """
        Initializes the FLH2 model.

        Args:
            alpha (float): Learning rate parameter for the multiplicative weight update.
        """
        super(FLH2, self).__init__()
        self.alpha = alpha
        self.experts = nn.ModuleList([STULayer()])  # Initialize with one expert
        self.creation_time = [1]  # List to track creation time of each expert
        self.loss_fn = ChunkedCrossEntropyLoss()

    def lifetime(self, j: int) -> int:
        """
        Calculates the lifetime of expert j based on its creation time.

        Args:
            j (int): The creation time of the expert.

        Returns:
            int: The lifetime of the expert.
        """
        if j <= 0:
            return 1
        k = int(math.log2(j & -j))  # Position of the rightmost set bit
        return 2 ** (k + 2) + 1

    def is_alive(self, j: int, t: int) -> bool:
        """
        Determines if expert j is alive at time t.

        Args:
            j (int): The creation time of the expert.
            t (int): The current time step.

        Returns:
            bool: True if the expert is alive, False otherwise.
        """
        return j <= t < j + self.lifetime(j)

    def prune(self, S_t: set, t: int) -> set:
        """
        Prunes the set of experts to retain only those that are alive at time t.

        Args:
            S_t (set): The current set of active experts.
            t (int): The current time step.

        Returns:
            set: The pruned set of active experts.
        """
        return set(j for j in S_t if self.is_alive(j, t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FLH2 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, tokens, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tokens, d_in).
        """
        bsz, toks, d_in = x.size()
        device = x.device

        # Initialize the active set S and probability weights p
        S = set(range(len(self.experts)))  # Initialize the set to include all existing experts
        p = {j: 1.0 / len(S) for j in S}  # Distribute the initial probability weight equally among experts

        # Initialize the output tensor
        y = torch.zeros(bsz, toks, d_in, device=device)

        for t in range(1, toks + 1):
            # Determine the set of alive experts S_t at time t
            S_t = set()
            for j in range(1, len(self.experts) + 1):
                if self.is_alive(j, t):
                    S_t.add(j)

            if not S_t:
                raise ValueError(f"No experts are alive at time step {t}.")

            # Compute outputs from all experts in S_t
            outputs = {}
            for j in S_t:
                outputs[j] = self.experts[j - 1](x[:, t - 1, :])  # Shape: (batch_size, d_in)

            # Aggregate the outputs to get the final decision y_t
            y_t = torch.zeros(bsz, d_in, device=device)
            for j in S_t:
                y_t += p[j] * outputs[j]
            y[:, t - 1, :] = y_t

            # Compute the loss for the aggregated decision
            # Assuming loss_fn returns a tensor of shape (batch_size,)
            loss_agg = self.loss_fn(y_t, x[:, t - 1, :])  # Shape: (batch_size,)

            # Compute the loss for each expert's output
            loss_experts = {}
            for j in S_t:
                loss_experts[j] = self.loss_fn(outputs[j], x[:, t - 1, :]).mean().item()

            # Update the probability weights using the multiplicative update rule
            for j in S_t:
                p[j] *= math.exp(-self.alpha * loss_experts[j])

            # Normalize the probability weights
            sum_p = sum(p[j] for j in S_t)
            for j in S_t:
                p[j] /= sum_p

            # Prune the set of experts and add a new expert
            S_pruned = self.prune(S_t, t + 1)
            new_expert_id = len(self.experts) + 1  # Assign a new expert ID
            S_next = S_pruned.union({new_expert_id})

            # Initialize the weight for the new expert
            p[new_expert_id] = 1.0 / t

            # Normalize the weights for all experts in S_next
            sum_p_next = sum(p[j] for j in S_next)
            for j in S_next:
                p[j] /= sum_p_next

            # Update the active set S
            S = S_next

            # Add the new expert to the experts list
            self.experts.append(STULayer())
            self.creation_time.append(new_expert_id)

        return y
