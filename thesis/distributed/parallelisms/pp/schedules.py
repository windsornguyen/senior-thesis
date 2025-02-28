from typing import Any, Dict, Callable, List, Optional, Tuple, Union

from torch.distributed.pipelining.schedules import _Action, F, I, W, PipelineScheduleMulti
from torch.distributed.pipelining.schedules.stage import _PipelineStageBase
from torch.distributed.pipelining.scheduhles.microbatch import TensorChunkSpec


class ScheduleDualPipe(PipelineScheduleMulti):
    """
    The DualPipe schedule.
    See https://arxiv.org/pdf/2412.19437v1 Section 3.2 for details.

    TODO: Add some more details here.
    """
    # WARNING: The below is the Zero Bubble schedule (ZBV variant) implementation as a placeholder.
    # DualPipe has not yet been implemented yet.
    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        stage_index_to_group_rank: Optional[Dict[int, int]] = None,
    ):
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            stage_index_to_group_rank=stage_index_to_group_rank,
        )
        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly 2 stages per rank, but got "
                f"{self.n_local_stages}."
            )

        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

    def _calculate_single_rank_operations(self, rank) -> List[Optional[_Action]]:
        # max(2 * self.pp_group_size - 1, ...) ensure the number of microbatches is at least
        # as large of the number of microbatches needed to fully utilize the pipeline
        n_micro = max(2 * self.pp_group_size - 1, self._n_microbatches)
        rank_ops: List[Optional[_Action]] = [None for _ in range(rank)]

        # Forward and backward action counts for stage chunk 0 and chunk 1
        f0_cnt, f1_cnt, b0_cnt, b1_cnt = 0, 0, 0, 0
        # warm-up phase
        warmup_n1 = 2 * (self.pp_group_size - rank) - 1
        stage_id_chunk0 = rank
        stage_id_chunk1 = self.num_stages - 1 - rank

        for _ in range(warmup_n1):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt)
            )
            f0_cnt += 1
        warmup_n2 = rank
        for _ in range(warmup_n2):
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt)
            )
            f0_cnt += 1
        warmup_n3 = self.pp_group_size - rank
        for _ in range(warmup_n3):
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        # stable phase
        while f1_cnt < f0_cnt or f0_cnt < n_micro:
            if f0_cnt < n_micro:
                rank_ops.append(
                    _Action(
                        stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt
                    )
                )
                f0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=b0_cnt)
            )
            b0_cnt += 1

            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        # cool-down phase
        w0_cnt, w1_cnt = b0_cnt, b1_cnt
        cooldown_n1 = rank
        for _ in range(cooldown_n1):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            b0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        cooldown_n2 = self.pp_group_size - rank
        for _ in range(cooldown_n2):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            b0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=w0_cnt)
            )
            w0_cnt += 1
        while w1_cnt < b1_cnt:
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=w1_cnt)
            )
            w1_cnt += 1
        while w0_cnt < b0_cnt:
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=w0_cnt)
            )
            w0_cnt += 1

        assert w0_cnt == b0_cnt and b0_cnt == f0_cnt
        assert w1_cnt == b1_cnt and b1_cnt == f1_cnt
        # We use max() in the n_micro computation above, so we may need to
        # remove redundant microbatches
        rank_ops = [
            (
                action
                if action is not None
                and action.microbatch_index is not None
                and action.microbatch_index < self._n_microbatches
                else None
            )
            for action in rank_ops
        ]
        return rank_ops
