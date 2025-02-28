import torch
import psutil

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from torch import Tensor
from torch.autograd.graph import saved_tensors_hooks

from thesis.utils.logger import logger


@dataclass
class OffloadConfig:
    """Configuration for activation offloading.

    Attributes:
        use_pin_memory: Whether to use pinned memory for CPU tensors
        use_streams: Whether to use multiple CUDA streams for overlap
        max_fwd_stash_size: Maximum number of tensors to keep in forward stash
        min_offload_size: Minimum tensor size in bytes to qualify for offloading
        virtual_memory_safe_pct: Maximum safe percentage of virtual memory to use
    """

    use_pin_memory: bool = True
    use_streams: Optional[bool] = None
    max_fwd_stash_size: int = 5
    min_offload_size: int = 1024
    virtual_memory_safe_pct: float = 60.0


def torch_version_ge(version: str) -> bool:
    """Check if current PyTorch version is greater than or equal to target.

    Args:
        version: Version string to compare against (e.g. "2.5.0")

    Returns:
        bool: Whether current version meets minimum
    """
    current = torch.__version__.split("+")[0]
    current_parts = [int(x) for x in current.split(".")]
    target_parts = [int(x) for x in version.split(".")]

    for c, t in zip(current_parts, target_parts, strict=True):
        if c > t:
            return True
        if c < t:
            return False
    return True


class OffloadActivations(saved_tensors_hooks):
    """Context manager for offloading activation tensors to CPU memory.

    This manager implements memory efficiency by selectively offloading large
    activation tensors to CPU during the forward pass and bringing them back
    during backward pass. It can optionally use multiple CUDA streams to overlap
    communication and computation for better performance.

    Args:
        config: Configuration object for offload behavior

    Raises:
        ValueError: If configuration parameters are invalid
        RuntimeError: If CUDA streams requested but PyTorch version insufficient

    Example:
        >>> with OffloadActivations():
        >>>     logits = model(inputs)
        >>> loss = loss_fn(logits, labels)
        >>> loss.backward()
    """

    def __init__(self, config: Optional[OffloadConfig] = None) -> None:
        self.config = config or OffloadConfig()

        # Initialize streams
        if self.config.use_streams is None:
            self.use_streams = torch_version_ge("2.5.0")
        else:
            self.use_streams = self.config.use_streams

        if self.use_streams and not torch_version_ge("2.5.0"):
            raise RuntimeError("OffloadActivations with use_streams=True requires PyTorch 2.5.0 or later")

        if self.config.max_fwd_stash_size < 1:
            raise ValueError(f"max_fwd_stash_size must be >= 1, got {self.config.max_fwd_stash_size}")

        # Initialize state
        self.tracker: Dict[int, Tuple[Tensor, bool]] = {}
        self.tensor_id: int = 0
        self.is_first_fwd_call: bool = True
        self.is_first_bwd_call: bool = True
        self.is_first_fwd_pass: bool = True

        # Stream management
        self.s0 = torch.cuda.default_stream()
        if self.use_streams:
            self.s1 = torch.cuda.Stream()
            self.fwd_stash: Dict[int, Tuple[Tensor, torch.cuda.Event]] = {}
            self.bwd_tensor_stash: Dict[int, Tensor] = {}
            self.bwd_ev_stash: Dict[int, torch.cuda.Event] = {}
            self.curr_graph_id = None
            self.curr_autograd_node = None

        # Register hooks
        super().__init__(self.pack_tensor, self.unpack_tensor)

    def get_cpu_ram_pct(self) -> float:
        """Get current CPU RAM usage percentage."""
        return psutil.virtual_memory().percent

    def verify_sufficient_virtual_memory(self) -> None:
        """Check if there's sufficient virtual memory available."""
        curr_pct = self.get_cpu_ram_pct()
        if curr_pct > self.config.virtual_memory_safe_pct:
            logger.warn(
                f"WARNING: Current RAM usage {curr_pct}% exceeds safe threshold "
                f"{self.config.virtual_memory_safe_pct}%"
            )

    def get_tensor_id(self) -> int:
        """Generate unique ID for tensor tracking."""
        self.tensor_id += 1
        return self.tensor_id

    @staticmethod
    def get_num_bytes_tensor(x: Tensor) -> int:
        """Calculate memory size of tensor in bytes."""
        try:
            return x.element_size() * x.nelement()
        except RuntimeError as e:
            logger.error(f"Failed to calculate tensor size: {e}")
            return 0

    def pack_tensor(self, activation: Tensor) -> int:
        """Pack activation tensor for offloading during forward pass.

        Args:
            activation: Tensor to potentially offload

        Returns:
            int: Unique identifier for tracking the tensor

        Note:
            Small tensors below min_offload_size are kept on GPU
        """
        if self.is_first_fwd_call:
            assert len(self.tracker) == 0, "Backward pass should clear tracker"
            self.is_first_fwd_call = False
            self.is_first_bwd_call = True

        num_bytes = self.get_num_bytes_tensor(activation)
        tensor_id = self.get_tensor_id()

        # Only offload large tensors
        if num_bytes >= self.config.min_offload_size:
            try:
                if self.use_streams:
                    # Clear old tensors from stash
                    for id in list(self.fwd_stash.keys()):
                        if id <= tensor_id - self.config.max_fwd_stash_size:
                            _, ev = self.fwd_stash[id]
                            self.s0.wait_event(ev)
                            del self.fwd_stash[id]
                        else:
                            break

                    self.s1.wait_stream(self.s0)

                stream = self.s1 if self.use_streams else self.s0
                with torch.cuda.stream(stream):
                    cpu_tensor = torch.empty_like(activation, pin_memory=self.config.use_pin_memory, device="cpu")
                    cpu_tensor.copy_(activation, non_blocking=True)
                    self.tracker[tensor_id] = (cpu_tensor, True)

                if self.use_streams:
                    event = stream.record_event()
                    self.fwd_stash[tensor_id] = (activation, event)

            except Exception as e:
                logger.error(f"Failed to offload tensor: {e}")
                self.tracker[tensor_id] = (activation, False)
        else:
            self.tracker[tensor_id] = (activation, False)

        return tensor_id

    def unpack_tensor(self, tensor_id: int) -> Tensor:
        """Retrieve tensor during backward pass.

        This method dispatches to either single-stream or multi-stream implementation
        based on configuration.

        Args:
            tensor_id: Unique identifier for tensor to retrieve

        Returns:
            Tensor: Retrieved tensor, either from CPU or GPU

        Raises:
            AssertionError: If tensor_id not found in tracker
        """
        if self.use_streams:
            return self._unpack_tensor_with_streams(tensor_id)
        return self._unpack_tensor_single_stream(tensor_id)

    def _unpack_tensor_single_stream(self, tensor_id: int) -> Tensor:
        """Simple single-stream tensor retrieval implementation."""
        if self.is_first_bwd_call:
            if self.is_first_fwd_pass:
                self.is_first_fwd_pass = False
                if self.config.use_pin_memory:
                    self.verify_sufficient_virtual_memory()

            self.is_first_bwd_call = False
            self.is_first_fwd_call = True

        assert tensor_id in self.tracker, f"Untracked tensor ID: {tensor_id}"

        maybe_gpu_tensor, modified = self.tracker[tensor_id]
        if modified:
            try:
                gpu_tensor = maybe_gpu_tensor.to("cuda", non_blocking=True)
                maybe_gpu_tensor = gpu_tensor
            except Exception as e:
                logger.error(f"Failed to move tensor back to GPU: {e}")

        del self.tracker[tensor_id]
        return maybe_gpu_tensor

    def _unpack_tensor_with_streams(self, tensor_id: int) -> Tensor:
        """Multi-stream tensor retrieval with computation/communication overlap."""
        if self.is_first_bwd_call:
            self.curr_graph_id = torch._C._current_graph_task_id()

            def cleanup_callback() -> None:
                """Clean up remaining tensor references."""
                for id in list(self.bwd_tensor_stash.keys()):
                    event = self.bwd_ev_stash[id]
                    self.s1.wait_event(event)
                    del self.bwd_tensor_stash[id]

            torch.autograd.variable.Variable._execution_engine.queue_callback(cleanup_callback)

            if self.is_first_fwd_pass:
                self.is_first_fwd_pass = False
                if self.config.use_pin_memory:
                    self.verify_sufficient_virtual_memory()

            self.is_first_bwd_call = False
            self.is_first_fwd_call = True

        assert tensor_id in self.tracker, f"Untracked tensor ID: {tensor_id}"

        maybe_gpu_tensor, modified = self.tracker[tensor_id]
        if modified:
            graph_id = torch._C._current_graph_task_id()
            node = torch._C._current_autograd_node()
            prev_node_ids = []

            if graph_id == self.curr_graph_id and self.curr_autograd_node != node:
                self.curr_autograd_node = node
                prev_node_ids = list(self.bwd_tensor_stash.keys())

            brought_back_from_cpu = True
            if tensor_id in self.fwd_stash:
                maybe_gpu_tensor = self.fwd_stash[tensor_id][0]
                brought_back_from_cpu = False
            else:
                try:
                    with torch.cuda.stream(self.s1):
                        gpu_tensor = maybe_gpu_tensor.to("cuda", non_blocking=True)
                        maybe_gpu_tensor = gpu_tensor

                    self.s0.wait_stream(self.s1)
                    self.bwd_tensor_stash[tensor_id] = maybe_gpu_tensor

                except Exception as e:
                    logger.error(f"Failed stream operation: {e}")

            def hook(outputs: Any, inputs: Any) -> Any:
                """Manage synchronization between computation and communication."""
                if brought_back_from_cpu:
                    event = self.s0.record_event()
                    self.bwd_ev_stash[tensor_id] = event

                # Clear forward stash
                for id in list(self.fwd_stash.keys()):
                    _, ev = self.fwd_stash[id]
                    self.s0.wait_event(ev)
                    del self.fwd_stash[id]

                # Sync previous node operations
                for id in prev_node_ids:
                    event = self.bwd_ev_stash[id]
                    self.s1.wait_event(event)
                    del self.bwd_tensor_stash[id]

                return outputs

            node.register_hook(hook)

        del self.tracker[tensor_id]
        return maybe_gpu_tensor

    def __enter__(self) -> "OffloadActivations":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and clean up resources."""
        if self.use_streams:
            # Ensure all operations are complete
            torch.cuda.synchronize()

            # Clear stashes
            self.fwd_stash.clear()
            self.bwd_tensor_stash.clear()
            self.bwd_ev_stash.clear()

        self.tracker.clear()
        self.tensor_id = 0
        self.is_first_fwd_call = True
        self.is_first_bwd_call = True
        self.is_first_fwd_pass = True


class NoOpManager(saved_tensors_hooks):
    """Context manager to disable activation offloading in a local scope.

    This manager overrides any previously applied saved_tensors_hooks by
    implementing no-op pack and unpack operations.

    Example:
        >>> with NoOpManager():
        >>>     # Offloading disabled in this block
        >>>     output = model(input)
    """

    def __init__(self) -> None:
        """Initialize no-op manager."""

        def noop(tensor: Any) -> Any:
            """No-op function that returns input unchanged."""
            return tensor

        super().__init__(noop, noop)
