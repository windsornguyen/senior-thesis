import os
import pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from thesis.utils.logger import logger
from thesis.distributed import is_master_process


class NDParallelStrategy:
    """
    Maps a global rank into an n-dimensional rank decomposition, e.g., [data_rank, tensor_rank, pipeline_rank, ...].
    'dim_sizes' is a list specifying the size for each dimension. The total number of ranks must be the product
    of all dim_sizes.

    Example:
        dim_sizes = [2, 2, 2, 8] -> total_ranks = 2*2*2*8 = 64

        global_rank=37 might decompose into something like [1, 0, 1, 5].
    """

    def __init__(self, dim_sizes: list[int]):
        self.dim_sizes = dim_sizes
        self.total_ranks = 1
        for size in dim_sizes:
            self.total_ranks *= size

    def decompose_rank(self, global_rank: int) -> list[int]:
        """
        Converts global_rank to an n-d index (like base conversion).
        Returns something like [data_rank, tensor_rank, pipeline_rank, ...].
        """
        if not (0 <= global_rank < self.total_ranks):
            raise ValueError(f"global_rank {global_rank} out of range (total={self.total_ranks})")

        coords = []
        r = global_rank
        for size in reversed(self.dim_sizes):
            coords.append(r % size)
            r //= size
        coords.reverse()
        return coords

    def compose_rank(self, coords: list[int]) -> int:
        """
        The inverse of decompose_rank: turns an n-D coordinate back into a global rank.

        Example:
            If coords=[1,0,1,5], we re-encode that into the single integer rank 37
            (or whichever value it corresponds to).
        """
        if len(coords) != len(self.dim_sizes):
            raise ValueError("coords length must match dim_sizes length")

        val = 0
        for c, size in zip(coords, self.dim_sizes, strict=True):
            if not (0 <= c < size):
                raise ValueError(f"coordinate {c} out of range for dimension size {size}")
            val = val * size + c
        return val


class NDOfflineDataset(IterableDataset, Stateful):
    """
    Offline, pre-tokenized dataset that splits shards among n-D ranks.
    Each shard is assumed to be a .npy (or .pt) file containing a list/array
    of token IDs.

    Yields (input, label) pairs of length seq_len, with a 1-token shift:
        input[i] = token[i : i + seq_len]
        label[i] = token[i+1 : i + seq_len + 1]

    Args:
        nd_strategy: The NDParallelStrategy instance describing the n-D rank decomposition.
        global_rank: The rank (0..nd_strategy.total_ranks-1) for this process.
        dataset_path: Directory containing .npy or .pt shards.
        seq_len: Number of tokens in each input sequence (label is shifted by 1).
        infinite: If True, loops back to shard 0 after exhausting everything.
        split: Dataset split to use ('train', 'val', 'test').
    """

    def __init__(
        self,
        nd_strategy: NDParallelStrategy,
        global_rank: int,
        dataset_path: str,
        seq_len: int = 8,
        infinite: bool = True,
        split: str = "train",
    ):
        super().__init__()
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"

        self.nd_strategy = nd_strategy
        self.global_rank = global_rank
        self.local_coords = nd_strategy.decompose_rank(global_rank)
        self.seq_len = seq_len
        self.infinite = infinite
        self.split = split

        # dimension 0 is our "data" dimension
        data_dim = 0
        data_rank = self.local_coords[data_dim]
        num_data_ranks = nd_strategy.dim_sizes[data_dim]

        # gather all shards in sorted order
        data = [
            os.path.join(dataset_path, f)
            for f in sorted(os.listdir(dataset_path))
            if f.endswith(".npy") or f.endswith(".pt")
        ]
        data = [f for f in data if self.split in os.path.basename(f)]
        if not data:
            raise ValueError(f"No shards found in {dataset_path}")

        if is_master_process:
            logger.info(f"Found {len(data)} shards for split {split}")

        # assign subset of shards to this data rank
        # e.g., data_rank=0 gets shard 0,2,4..., data_rank=1 gets shard 1,3,5..., etc.
        self._all_shards = data[data_rank::num_data_ranks]

        # iteration state
        self._shard_idx = 0
        self._local_pos = 0
        self._tokens: list[int] = []

        # load the first shard
        self._load_current_shard()

    def _load_current_shard(self):
        """Loads the shard at self._shard_idx (or loops if infinite)."""
        if self._shard_idx >= len(self._all_shards):
            if not self.infinite:
                return
            self._shard_idx = 0  # loop back

        if not self._all_shards:
            return  # no shards at all

        shard_path = self._all_shards[self._shard_idx]
        ext = os.path.splitext(shard_path)[1]
        if ext == ".npy":
            arr = np.load(shard_path)
            self._tokens = arr.tolist()
        elif ext == ".pt":
            arr = torch.load(shard_path)
            if isinstance(arr, torch.Tensor):
                self._tokens = arr.long().tolist()
            else:
                raise ValueError(f"Unsupported .pt structure in {shard_path}")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self._local_pos = 0

    def _next_sample(self):
        """
        Returns a chunk of seq_len+1 tokens as (x, y), or moves to the next shard
        if we run out in this shard.
        """
        needed = self.seq_len + 1
        while True:
            if self._local_pos + needed <= len(self._tokens):
                chunk = self._tokens[self._local_pos : self._local_pos + needed]
                self._local_pos += self.seq_len
                x = torch.LongTensor(chunk[:-1])
                y = torch.LongTensor(chunk[1:])
                return x, y
            else:
                # move to the next shard
                self._shard_idx += 1
                if self._shard_idx >= len(self._all_shards):
                    if not self.infinite:
                        return None
                    self._shard_idx = 0
                self._load_current_shard()
                if not self._all_shards or (len(self._tokens) < needed and not self.infinite):
                    return None

    def __iter__(self):
        """Iterate forever if infinite=True, else stop once shards are exhausted."""
        while True:
            sample = self._next_sample()
            if sample is None:
                # no more data
                break
            yield sample

    def state_dict(self) -> dict[str, any]:
        """Return the minimal data needed to resume iteration exactly."""
        return {
            "shard_idx": self._shard_idx,
            "local_pos": self._local_pos,
        }

    def load_state_dict(self, state: dict[str, any]) -> None:
        """Restore iteration state from the given dictionary."""
        self._shard_idx = state["shard_idx"]
        self._local_pos = state["local_pos"]
        if 0 <= self._shard_idx < len(self._all_shards):
            self._load_current_shard()
            # Re-set local_pos, because _load_current_shard overwrote it
            self._local_pos = state["local_pos"]


class NDAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around StatefulDataLoader that only stores state for the data-parallel rank
    (dimension 0). This avoids duplicating the same state for other parallel dimensions
    like tensor, pipeline, or context parallel.

    Args:
        dp_rank: The data-parallel rank (i.e., local_coords[0]).
        dataset: The NDOfflineDataset instance.
        batch_size: Number of (x, y) pairs per batch.
    """

    def __init__(self, dp_rank: int, dataset: NDOfflineDataset, batch_size: int):
        super().__init__(dataset, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, any]:
        """
        Only store the state for our data-parallel rank. That way, if we have
        multiple ranks that differ only in non-data dimensions, we don’t produce
        redundant states for them.
        """
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: dict[str, any]) -> None:
        """Restore state from the dictionary, if present for our dp_rank."""
        if not state_dict:
            return
        if self._rank_id not in state_dict:
            logger.warning(f"No state found for dp rank {self._dp_rank}")
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_nd_data_loader(
    dim_sizes: list[int],
    global_rank: int,
    dataset_path: str,
    batch_size: int = 2,
    seq_len: int = 8,
    infinite: bool = True,
    split: str = "train",
) -> NDAwareDataLoader:
    """
    Creates an n-D offline dataset, then wraps it in NDAwareDataLoader.

    If product(dim_sizes)=1 (i.e., single GPU or no real parallel dimension),
    we skip ND logic, treat dp_rank=0, dp_size=1.
    Otherwise, we do ND decomposition on dimension 0.

    Args:
        dim_sizes: The shape of your n-D rank decomposition, e.g., [2, 2, 2, 8].
        global_rank: The integer rank in [0..product(dim_sizes)-1].
        dataset_path: Directory containing shards with token IDs.
        batch_size: Number of samples per batch.
        seq_len: Number of tokens in each input sequence.
        infinite: Whether to loop shards infinitely.
        split: Dataset split to use ('train', 'val', 'test').

    Returns:
        An NDAwareDataLoader that yields (batch_x, batch_y) pairs.
    """
    # Calculate the product of dim_sizes
    product = 1
    for d in dim_sizes:
        product *= d

    if product == 1:
        # Single-rank fallback
        logger.info(f"build_nd_data_loader: single-GPU / single-rank fallback, dim_sizes={dim_sizes}")
        dim_sizes = [1]
        global_rank = 0

    nd_strategy = NDParallelStrategy(dim_sizes)
    if global_rank >= nd_strategy.total_ranks:
        raise ValueError(f"global_rank {global_rank} out of range for dims {dim_sizes}")

    ds = NDOfflineDataset(
        nd_strategy=nd_strategy,
        global_rank=global_rank,
        dataset_path=dataset_path,
        seq_len=seq_len,
        infinite=infinite,
        split=split,
    )

    # local_coords[0] is the data-parallel rank
    local_coords = nd_strategy.decompose_rank(global_rank)
    dp_rank = local_coords[0]

    loader = NDAwareDataLoader(dp_rank, ds, batch_size=batch_size)
    return loader


def make_test_data_dir(dir_name: str = "test_data", num_shards: int = 4, seed: int = 1746):
    """
    Creates a directory of random .npy files that contain tokenized data.
    Each shard is basically a small array of random IDs.

    We create num_shards shards, each containing ~80 random tokens in [100..300].

    Args:
        dir_name: Directory name to create shards in.
        num_shards: Number of shards to create.
        seed: Random seed for reproducibility.
    """
    os.makedirs(dir_name, exist_ok=True)
    rng = np.random.default_rng(seed=seed)

    for i in range(num_shards):
        shard_path = os.path.join(dir_name, f"shard_{i}.npy")
        tokens = rng.integers(low=100, high=300, size=80, dtype=np.int64)
        np.save(shard_path, tokens)
        print(f"Wrote shard {shard_path} with shape={tokens.shape}")


def main():
    """
    Demonstration script that:
      1) Creates some fake data shards in `test_data/`.
      2) Defines an n-D parallel shape with product=64, e.g., [2, 2, 2, 8].
      3) Iterates over each possible global_rank in [0..63], building a loader,
         fetching batches, and showing basic checkpoint/resume logic.
    """
    # 1) create test data shards
    make_test_data_dir("test_data", num_shards=4, seed=999)

    # 2) define an n-D parallel shape for 64 ranks:
    #    e.g., data=2, tensor=2, pipeline=2, context=8 => 2*2*2*8 = 64.
    dim_sizes = [2, 2, 2, 8]
    total_ranks = 1
    for d in dim_sizes:
        total_ranks *= d
    print(f"We have dim_sizes={dim_sizes} => total_ranks={total_ranks}\n")

    # 3) simulate running on each global_rank in [0..63].
    for global_rank in range(total_ranks):
        node_id = global_rank // 8
        local_gpu_id = global_rank % 8

        print(f"\n--- DEMO for global_rank={global_rank} (node={node_id}, gpu={local_gpu_id}) ---")
        loader = build_nd_data_loader(
            dim_sizes=dim_sizes,
            global_rank=global_rank,
            dataset_path="test_data",
            batch_size=2,
            seq_len=12,
            infinite=True,
            split="train",
        )

        ds = loader.dataset
        print(f" local_coords={ds.local_coords}, shards={ds._all_shards}")

        # Show a few batches
        all_batches = []
        checkpoint = None
        for i, (inp, lab) in enumerate(loader):
            all_batches.append((inp.clone(), lab.clone()))
            if i < 2:
                print(f"   batch {i} => inp={inp.tolist()}, lab={lab.tolist()}")
            if i == 4:
                checkpoint = loader.state_dict()
                print("   [checkpoint saved]")
            if i == 6 and checkpoint is not None:
                loader.load_state_dict(checkpoint)
                print("   [resumed from checkpoint, continuing iteration...]")
            if i > 8:
                break

        print(f"  ... total batches: {len(all_batches)}\n")

    print("\n--- done ---")


if __name__ == "__main__":
    main()
