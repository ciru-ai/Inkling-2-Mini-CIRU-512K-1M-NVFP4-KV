# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Ciru for Blackwell and NVFP4 KV caching.
"""Low-M BF16 tensor-core projections for SM120/SM121 (with Split-K & PDL)."""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

MAX_M = 16
_BN, _BK = 32, 256


@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _gemv_splitk_kernel(
    X,
    W,
    Partials,
    M,
    N,
    K,
    SX,
    SW,
    SPLIT_K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    """Split-K low-M BF16 projection to saturate all 170 SMs on RTX 5090."""
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    cols = pid_n * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    tiles_k = tl.cdiv(K, BK)
    tiles_per_split = tl.cdiv(tiles_k, SPLIT_K)
    t_start = pid_k * tiles_per_split
    t_end = tl.minimum(t_start + tiles_per_split, tiles_k)
    acc = tl.zeros((MP, BN), tl.float32)
    for t in range(t_start, t_end):
        ks = t * BK + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    base = pid_k.to(tl.int64) * M * N
    tl.store(
        Partials + base + rows[:, None].to(tl.int64) * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _gemv_splitk_reduce_kernel(
    Partials,
    B,
    Y,
    M,
    N,
    SY,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    mn = tl.full((), M, tl.int64) * N
    acc = tl.load(Partials + row * N + cols, mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(Partials + s * mn + row * N + cols, mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(B + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * SY + cols, acc.to(Y.dtype.element_ty), mask)


def _choose_split_k(n: int, k: int) -> int:
    if os.getenv("INKLING_SM120_SPLITK_GEMV", "1") == "0":
        return 1
    if k < 1024:
        return 1
    n_ctas = triton.cdiv(n, _BN)
    sms = num_compute_units()
    # Only Split-K when N-CTAs underfill the GPU (WO=72, k/v_proj=32 on 170-SM RTX 5090).
    if n_ctas >= sms:
        return 1
    target = sms * 2
    max_tiles = triton.cdiv(k, _BK)
    for sk in (2, 4, 8):
        if n_ctas * sk >= target and sk <= max_tiles:
            return sk
    return min(8, max(1, max_tiles // 2))


def sm12x_bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    assert m <= MAX_M
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    if m:
        split_k = _choose_split_k(n, k)
        stages = 3 if os.getenv("INKLING_SM120_SPLITK_GEMV", "1") != "0" else 1
        pdl = os.getenv("INKLING_SM120_SPLITK_GEMV", "1") != "0"
        if split_k == 1:
            _gemv_kernel[(triton.cdiv(n, _BN),)](
                x2,
                weight,
                bias,
                out,
                m,
                n,
                k,
                x2.stride(0),
                weight.stride(0),
                out.stride(0),
                HAS_BIAS=bias is not None,
                BN=_BN,
                BK=_BK,
                MP=16,
                num_warps=4,
                num_stages=stages,
                launch_pdl=pdl,
            )
        else:
            partials = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
            _gemv_splitk_kernel[(triton.cdiv(n, _BN), split_k)](
                x2,
                weight,
                partials,
                m,
                n,
                k,
                x2.stride(0),
                weight.stride(0),
                SPLIT_K=split_k,
                BN=_BN,
                BK=_BK,
                MP=16,
                num_warps=4,
                num_stages=stages,
                launch_pdl=pdl,
            )
            _gemv_splitk_reduce_kernel[(triton.cdiv(n, 256), m)](
                partials,
                bias,
                out,
                m,
                n,
                out.stride(0),
                SPLIT_K=split_k,
                HAS_BIAS=bias is not None,
                BLOCK=256,
                num_warps=4,
                launch_pdl=pdl,
            )
    return out.view(*lead, n)


def use_sm12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    return (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.stride(-1) == weight.stride(-1) == 1
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
        and current_platform.is_device_capability_family(120)
    )
