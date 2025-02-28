from typing import List, Optional, Tuple

import torch
from thesis.utils.logger import logger
from mamba_ssm.ops.triton.ssd_combined import (
    _mamba_chunk_scan_combined_fwd,
    _mamba_chunk_scan_combined_bwd,
    mamba_chunk_scan_combined as mamba_chunk_scan_combined_ref,
)

# TODO: Make the mem_eff_path also torch compilable
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

@torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def _compiled_mamba_chunk_scan_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compiled forward dispatch to Triton-based SSM code.
    """
    logger.debug("Entering compiled forward.")
    return _mamba_chunk_scan_combined_fwd(
        x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, seq_idx=seq_idx,
        cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit
    )

@torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def _compiled_mamba_chunk_scan_combined_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    dfinal_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Compiled backward dispatch to Triton-based SSM code.
    """
    logger.debug("Entering compiled backward.")
    return _mamba_chunk_scan_combined_bwd(
        dout, x, dt, A, B, C, out, chunk_size, D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, dfinal_states=dfinal_states,
        seq_idx=seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit
    )

@torch.library.custom_op(
    "mamba_ssm::ssm_chunk_scan_combined_fwd",
    mutates_args=(),
    device_types="cuda",
)
def ssm_chunk_scan_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom-op forward: 
      Allows torch.compile to treat SSM chunk-scan as a recognized operator.
    """
    out, out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(
        x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, seq_idx=seq_idx,
        cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    return (
        out,
        out_x if out_x is not None else out.new_empty(0),
        rest[0] if cu_seqlens is not None else out.new_empty(0),
    )

@ssm_chunk_scan_combined_fwd.register_fake
def _ssm_chunk_scan_combined_fwd_fake(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake forward: only returns placeholder tensors for shape inference.
    This is crucial for graph tracing in torch.compile.
    """
    _, _, n_heads, head_dim = x.shape
    return (
        torch.empty_like(x), 
        torch.empty_like(x) if z is not None else None, 
        x.new_empty((cu_seqlens.size(0)-1, n_heads, head_dim, B.size(0))) if cu_seqlens is not None else None,
    )

@torch.library.custom_op(
    "mamba_ssm::ssm_chunk_scan_combined_bwd", 
    mutates_args=(),
    device_types="cuda",
)
def ssm_chunk_scan_combined_bwd(
    dout: torch.Tensor,
    x: torch.Tensor, 
    dt: torch.Tensor, 
    A: torch.Tensor, 
    B: torch.Tensor, 
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int, 
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Custom-op backward:
      Declared so torch.compile knows the gradient relationship for the custom op.
    """
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
        dout, x, dt, A, B, C, out, chunk_size, D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, dfinal_states=None,
        seq_idx=seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    return (
        dx,
        ddt,
        dA,
        dB,
        dC,
        dD if dD is not None else dx.new_empty(0),
        dz if dz is not None else dx.new_empty(0),
        ddt_bias if ddt_bias is not None else dx.new_empty(0),
        dinitial_states if initial_states is not None else dx.new_empty(0)
    )

@ssm_chunk_scan_combined_bwd.register_fake
def _ssm_chunk_scan_combined_bwd_fake(
    dout: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Fake backward: returns placeholder gradients with correct shapes
    for the graph tracer (AOT Autograd).
    """
    return (
        torch.empty_like(x),
        torch.empty_like(dt),
        torch.empty_like(A),
        torch.empty_like(B),
        torch.empty_like(C),
        torch.empty_like(D) if D is not None else None,
        torch.empty_like(z) if z is not None else None,
        torch.empty_like(dt_bias) if dt_bias is not None else None,
        torch.empty_like(initial_states) if initial_states is not None else None,
    )

def ssm_chunk_scan_combined_setup_context(ctx, inputs, output):
    """
    Save relevant tensors for backward and store small flags (e.g. dt_softplus).
    """
    (
        x, dt, A, B, C, chunk_size, D, z, dt_bias,
        initial_states, seq_idx, cu_seqlens, dt_softplus, dt_limit
    ) = inputs
    out, out_x, state_varlen = output
    ctx.save_for_backward(
        out if z is None else out_x,
        x, dt, A, B, C, D, z, dt_bias, initial_states, seq_idx
    )
    ctx.dt_softplus = dt_softplus
    ctx.chunk_size = chunk_size
    ctx.dt_limit = dt_limit

def ssm_chunk_scan_combined_bridge(ctx, dout, dout_x, dout_state_varlen):
    """
    Bridge function that calls the custom backward op, passing saved values.
    """
    out, x, dt, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = ssm_chunk_scan_combined_bwd(
        dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, seq_idx=seq_idx,
        dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit
    )
    return (
        dx, ddt, dA, dB, dC,
        None,  # chunk_size is an int, no gradient
        dD if D is not None else None,
        dz if z is not None else None,
        ddt_bias if dt_bias is not None else None,
        dinitial_states if initial_states is not None else None,
        None, None, None, None
    )

# Register custom autograd function
torch.library.register_autograd(
    "mamba_ssm::ssm_chunk_scan_combined_fwd",
    ssm_chunk_scan_combined_bridge,
    setup_context=ssm_chunk_scan_combined_setup_context,
)

def mamba_chunk_scan_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf"))
) -> torch.Tensor:
    """
    High-level Python entry point that calls our custom-op forward.
    Returns the forward output (and varlen_states if cu_seqlens is used).
    """
    out, _, varlen_states = ssm_chunk_scan_combined_fwd(
        x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, seq_idx=seq_idx,
        cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    return (out, varlen_states) if cu_seqlens is not None else out


if __name__ == "__main__":
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as mamba_chunk_scan_combined_ref

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn(2, 3, 4, 5).cuda()
    dt = torch.randn(2, 3, 4).cuda()
    A = torch.randn(4).cuda()
    B = torch.randn(2, 3, 4, 5).cuda()
    C = torch.randn(2, 3, 4, 5).cuda()
    chunk_size = 2
    D = torch.randn(4, 5).cuda()
    z = torch.randn(2, 3, 4, 5).cuda()
    dt_bias = torch.randn(4).cuda()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out.min(), out.max(), out.mean(), out.std())

    compiled_mamba_chunk_scan_combined = torch.compile(mamba_chunk_scan_combined)
    out = compiled_mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out.min(), out.max(), out.mean(), out.std())

    out_ref = mamba_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out_ref.min(), out_ref.max(), out_ref.mean(), out_ref.std())
