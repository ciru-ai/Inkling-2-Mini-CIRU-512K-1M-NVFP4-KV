# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Ciru for Blackwell and NVFP4 KV caching.
"""SM120 batched Triton kernels for Inkling2LoopDecoderLayer.blend.

Replaces the two tiny-M cuBLAS `torch.bmm` calls (which fall back to Ampere
`cutlass_80_wmma` + splitKreduce at ~5% of GDDR7 bandwidth on RTX 5090) and
fuses the `controller_gain_forget` projection directly into `blend_rms_norm`.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .norm import rsqrt


@triton.jit
def _batched_down_splitk_kernel(
    Pair,
    WDown,
    Partials,
    T,
    R: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Batched Split-K GEMV: Partials[s, split, t, r] = sum_k Pair[s, t, k] * WDown[s, r, k]."""
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_sk = tl.program_id(2)
    s = (pid_sk // SPLIT_K).to(tl.int64)
    sk = (pid_sk % SPLIT_K).to(tl.int64)

    rows = pid_m * MP + tl.arange(0, MP)
    cols = pid_n * BN + tl.arange(0, BN)
    row_mask = rows < T
    col_mask = cols < R

    tiles_k = tl.cdiv(K, BK)
    tiles_per_split = tl.cdiv(tiles_k, SPLIT_K)
    k_start = sk * tiles_per_split
    k_end = tl.minimum(k_start + tiles_per_split, tiles_k)

    p_base = Pair + s * T * K
    w_base = WDown + s * R * K

    acc = tl.zeros((MP, BN), tl.float32)
    for kt in range(k_start, k_end):
        ks = kt * BK + tl.arange(0, BK)
        k_mask = ks < K
        x = tl.load(
            p_base + rows[:, None].to(tl.int64) * K + ks[None, :],
            row_mask[:, None] & k_mask[None, :],
            other=0,
        )
        w = tl.load(
            w_base + cols[None, :].to(tl.int64) * K + ks[:, None],
            col_mask[None, :] & k_mask[:, None],
            other=0,
        )
        acc += tl.dot(x, w)

    out_base = Partials + (s * SPLIT_K + sk) * T * R
    tl.store(
        out_base + rows[:, None].to(tl.int64) * R + cols[None, :],
        acc,
        row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _batched_down_reduce_kernel(
    Partials,
    Out,
    T,
    R: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Reduce Partials[S, SPLIT_K, T, R] -> bf16 Out[S, T, R]."""
    tl.extra.cuda.gdc_wait()
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, R)
    stride_sk = tl.full((), T, tl.int64) * R
    base = Partials + s * SPLIT_K * stride_sk + t * R + cols
    acc = tl.load(base)
    for sk in tl.static_range(1, SPLIT_K):
        acc += tl.load(base + sk * stride_sk)
    tl.store(Out + (s * T + t) * R + cols, acc.to(Out.dtype.element_ty))


@triton.jit
def _fused_gf_blend_part_kernel(
    Hidden,
    WGF,
    Dest,
    Message,
    Blended,
    PartSq,
    stride_dest_s,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    MP: tl.constexpr,
    BD: tl.constexpr,
):
    """Fuse controller_gain_forget GEMM (K=R=64) + tanh blend + partial sum_sq.

    Computes for each (s, m_tile, d_block):
      gain   = Hidden[s, t, :] @ WGF[s, d, :].T
      forget = Hidden[s, t, :] @ WGF[s, D + d, :].T
      x      = (1 - tanh(forget)) * dest + tanh(gain) * message
    Stores fp32 x -> Blended[s, t, d] and partial sum(x*x) -> PartSq[s, t, d_block].
    """
    tl.extra.cuda.gdc_wait()
    d_blk = tl.program_id(0)
    m_blk = tl.program_id(1)
    s = tl.program_id(2).to(tl.int64)

    rows = m_blk * MP + tl.arange(0, MP)
    cols = d_blk * BD + tl.arange(0, BD)
    row_mask = rows < T
    col_mask = cols < D
    r_idx = tl.arange(0, R)

    h = tl.load(
        Hidden + (s * T + rows[:, None].to(tl.int64)) * R + r_idx[None, :],
        row_mask[:, None],
        other=0,
    )
    w_base = WGF + s * (2 * D) * R
    w_gain = tl.load(
        w_base + cols[None, :].to(tl.int64) * R + r_idx[:, None],
        col_mask[None, :],
        other=0,
    )
    w_forget = tl.load(
        w_base + (D + cols[None, :]).to(tl.int64) * R + r_idx[:, None],
        col_mask[None, :],
        other=0,
    )
    gain = tl.dot(h, w_gain)
    forget = tl.dot(h, w_forget)

    dest = tl.load(
        Dest + s * stride_dest_s + rows[:, None].to(tl.int64) * D + cols[None, :],
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    msg = tl.load(
        Message + (s * T + rows[:, None].to(tl.int64)) * D + cols[None, :],
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    x = (1.0 - tldevice.tanh(forget)) * dest + tldevice.tanh(gain) * msg
    x = tl.where(col_mask[None, :], x, 0.0)
    tl.store(
        Blended + (s * T + rows[:, None].to(tl.int64)) * D + cols[None, :],
        x,
        row_mask[:, None] & col_mask[None, :],
    )
    psq = tl.sum(x * x, axis=1)
    tl.store(
        PartSq + (s * T + rows.to(tl.int64)) * N_BLOCKS + d_blk,
        psq,
        row_mask,
    )


@triton.jit
def _blend_norm_finish_kernel(
    Blended,
    PartSq,
    NormW,
    Out,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    N_BLOCKS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Apply final RMSNorm over Blended[s, t, :] using PartSq[s, t, :]."""
    tl.extra.cuda.gdc_wait()
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    row = s * T + t

    b_idx = tl.arange(0, N_BLOCKS_POW2)
    psq = tl.load(PartSq + row * N_BLOCKS + b_idx, b_idx < N_BLOCKS, other=0.0)
    sum_sq = tl.sum(psq, axis=0)
    rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)

    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(Blended + row * D + cols, mask, other=0.0)
    w = tl.load(NormW + s * stride_w_s + cols, mask, other=0.0).to(tl.float32)
    y = (x * rstd) * w
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def fused_loop_blend_sm120(
    pair: torch.Tensor,
    controller_down: torch.Tensor,
    controller_gain_forget: torch.Tensor,
    dest: torch.Tensor,
    message: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Fast SM120 Triton path for Inkling2LoopDecoderLayer.blend (T <= 64)."""
    num_streams, num_tokens, d = message.shape
    rank = controller_down.shape[1]
    k_down = 2 * d
    if num_tokens == 0:
        return message.new_empty((num_streams, 0, d), dtype=torch.bfloat16)

    mp = 16
    bn_down = 32
    bk_down = 256
    tiles_k = triton.cdiv(k_down, bk_down)
    # Saturate 170 SMs on RTX 5090 while keeping reduce tiny.
    split_k = min(tiles_k, 18 if num_streams == 1 else 9)

    partials = torch.empty(
        (num_streams, split_k, num_tokens, rank),
        dtype=torch.float32,
        device=message.device,
    )
    hidden = torch.empty(
        (num_streams, num_tokens, rank),
        dtype=torch.bfloat16,
        device=message.device,
    )
    _batched_down_splitk_kernel[
        (triton.cdiv(rank, bn_down), triton.cdiv(num_tokens, mp), num_streams * split_k)
    ](
        pair,
        controller_down,
        partials,
        num_tokens,
        R=rank,
        K=k_down,
        SPLIT_K=split_k,
        MP=mp,
        BN=bn_down,
        BK=bk_down,
        num_warps=4,
        num_stages=3,
        launch_pdl=True,
    )
    _batched_down_reduce_kernel[(num_streams, num_tokens)](
        partials,
        hidden,
        num_tokens,
        R=rank,
        SPLIT_K=split_k,
        num_warps=2,
        launch_pdl=True,
    )

    bd = 32
    n_blocks = triton.cdiv(d, bd)
    blended = torch.empty(
        (num_streams, num_tokens, d), dtype=torch.float32, device=message.device
    )
    part_sq = torch.empty(
        (num_streams, num_tokens, n_blocks), dtype=torch.float32, device=message.device
    )
    _fused_gf_blend_part_kernel[(n_blocks, triton.cdiv(num_tokens, mp), num_streams)](
        hidden,
        controller_gain_forget,
        dest,
        message,
        blended,
        part_sq,
        dest.stride(0),
        num_tokens,
        D=d,
        R=rank,
        N_BLOCKS=n_blocks,
        MP=mp,
        BD=bd,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
        launch_pdl=True,
    )

    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    _blend_norm_finish_kernel[(num_streams, num_tokens)](
        blended,
        part_sq,
        norm_weight,
        out,
        norm_weight.stride(0),
        num_tokens,
        norm_eps,
        D=d,
        N_BLOCKS=n_blocks,
        N_BLOCKS_POW2=triton.next_power_of_2(n_blocks),
        BLOCK=triton.next_power_of_2(d),
        num_warps=8,
        launch_pdl=True,
    )
    return out
