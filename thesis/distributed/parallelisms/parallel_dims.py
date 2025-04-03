from dataclasses import dataclass
from functools import cached_property
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from thesis.distributed import validate_parallel_factors

@dataclass
class ParallelDims:
    """
    Encapsulates various parallelism factors and constructs the corresponding
    device mesh for distributed training.

    Args:
        dp_replicate: Data parallel replication factor
        dp_shard: Data parallel sharding factor (-1 indicates auto-infer)
        cp: Context parallel factor
        tp: Tensor parallel factor
        pp: Pipeline parallel factor
        world_size: Total number of ranks/devices
        enable_loss_parallel: If True, indicates a special "loss parallel" strategy
    """

    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        Validates that dp_replicate * dp_shard * cp * tp * pp == world_size,
        optionally auto-infers dp_shard if it's set to -1.
        Raises ValueError on invalid configurations.
        """
        self.dp_shard = validate_parallel_factors(
            dp_replicate=self.dp_replicate,
            dp_shard=self.dp_shard,
            cp=self.cp,
            tp=self.tp,
            pp=self.pp,
            auto_infer_shard=True,
        )

    def build_mesh(self, device_type: str) -> DeviceMesh:
        from thesis.utils.logger import logger

        mesh_shape = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
            strict=True,
        ):
            if d > 1:
                mesh_shape.append(d)
                names.append(name)

        logger.info(f"Building {len(mesh_shape)}-D device mesh with {names}, {mesh_shape}")
        names = tuple(names)
        mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=mesh_shape,
            mesh_dim_names=names,
        )

        # Mesh for data parallel (no communication on this mesh)
        dp_mesh_dim_names = []

        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []

        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")

        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names != []:
            # Flatten dp-related dims into a single dim: (dp_replicate, dp_shard) -> (dp_replicate * dp_shard)
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        return mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def loss_parallel_enabled(self) -> bool:
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self) -> int:
        return self.cp * self.tp * self.pp
