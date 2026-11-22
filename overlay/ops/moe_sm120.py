# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Ciru for Blackwell and NVFP4 KV caching.
"""SM120/121 expert GEMMs with native NVFP4 + native MXFP8 MMA, Split-K & PDL.

Unmodified-weight RTX 5090 (170-SM) optimizations over upstream b426b46:
1. Native SM120 MXFP8 `tl.dot_scaled(..., "e4m3", ..., "e4m3")` for coda layers
   20..23 (`INKLING_SM120_NATIVE_MXFP8=1`), replacing software BF16 dequant.
2. Tuned SM120 NVFP4 tiles (`INKLING_SM120_TUNED_NVFP4=1`): `BM=16` for decode/
   DSpark (`tile <= 16`), Split-K (`SK=4`) for low-M SM saturation, `num_stages=3`,
   `launch_pdl=True`, and `BM=128/BN=128` prefill for both w13 and w2.
"""

import os

import torch

from vllm.triton_utils import tl, triton

from .block_scales import _scale_offsets
from .trtllm_bmm import ExpertPlan


@triton.jit
def _expert_gemm(
    Act,
    ActScale,
    RowScale,
    Weight,
    WeightScale,
    ExpertScale,
    Out,
    Tiles,
    MAX_CTAS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.program_id(1) * BM + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        packed = k * (BK // 2) + tl.arange(0, BK // 2)
        groups = k * (BK // 16) + tl.arange(0, BK // 16)
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * (K // 2) + packed[None, :],
            valid[:, None] & (packed[None, :] < K // 2),
            other=0,
        )
        a_scale = tl.load(
            ActScale + _scale_offsets(rows[:, None], groups[None, :], K // 16, TILE),
            valid[:, None] & (groups[None, :] < K // 16),
            other=0.0,
        )
        w = tl.load(
            Weight + (expert * N + cols[None, :]) * (K // 2) + packed[:, None],
            (cols[None, :] < N) & (packed[:, None] < K // 2),
            other=0,
            eviction_policy="evict_first",
        )
        w_scale = tl.load(
            WeightScale + (expert * N + cols[:, None]) * (K // 16) + groups[None, :],
            (cols[:, None] < N) & (groups[None, :] < K // 16),
            other=0.0,
            eviction_policy="evict_first",
        )
        acc = tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1", acc)
    acc *= tl.load(ExpertScale + expert)
    acc *= tl.load(RowScale + rows, valid, other=0)[:, None]
    tl.store(
        Out + rows[:, None].to(tl.int64) * N + cols[None, :],
        acc,
        valid[:, None] & (cols[None, :] < N),
    )


@triton.jit
def _expert_gemm_splitk(
    Act,
    ActScale,
    Weight,
    WeightScale,
    Partials,
    Tiles,
    MAX_CTAS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Split-K NVFP4 expert GEMM so M=1..8 decode saturates all 170 RTX 5090 SMs."""
    tl.extra.cuda.gdc_wait()
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    pid_k = tl.program_id(2)
    tiles_k = tl.cdiv(K, BK)
    tiles_per_split = tl.cdiv(tiles_k, SPLIT_K)
    k_start = pid_k * tiles_per_split
    k_end = tl.minimum(k_start + tiles_per_split, tiles_k)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(k_start, k_end):
        packed = k * (BK // 2) + tl.arange(0, BK // 2)
        groups = k * (BK // 16) + tl.arange(0, BK // 16)
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * (K // 2) + packed[None, :],
            valid[:, None] & (packed[None, :] < K // 2),
            other=0,
        )
        a_scale = tl.load(
            ActScale + _scale_offsets(rows[:, None], groups[None, :], K // 16, TILE),
            valid[:, None] & (groups[None, :] < K // 16),
            other=0.0,
        )
        w = tl.load(
            Weight + (expert * N + cols[None, :]) * (K // 2) + packed[:, None],
            (cols[None, :] < N) & (packed[:, None] < K // 2),
            other=0,
            eviction_policy="evict_first",
        )
        w_scale = tl.load(
            WeightScale + (expert * N + cols[:, None]) * (K // 16) + groups[None, :],
            (cols[:, None] < N) & (groups[None, :] < K // 16),
            other=0.0,
            eviction_policy="evict_first",
        )
        acc = tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1", acc)
    base = pid_k.to(tl.int64) * TOTAL_ROWS * N
    tl.store(
        Partials + base + rows[:, None].to(tl.int64) * N + cols[None, :],
        acc,
        valid[:, None] & (cols[None, :] < N),
    )


@triton.jit
def _expert_splitk_reduce(
    Partials,
    RowScale,
    ExpertScale,
    Out,
    Tiles,
    MAX_CTAS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    row = tile * TILE + tl.program_id(1)
    if row >= tl.minimum(row_end, (tile + 1) * TILE):
        return
    cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    stride_s = tl.full((), TOTAL_ROWS, tl.int64) * N
    row64 = row.to(tl.int64)
    acc = tl.load(Partials + row64 * N + cols, mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(Partials + s * stride_s + row64 * N + cols, mask, other=0.0)
    scale = tl.load(ExpertScale + expert) * tl.load(RowScale + row)
    tl.store(Out + row64 * N + cols, (acc * scale).to(tl.bfloat16), mask)


@triton.jit
def _dequant_e4m3(values, exponents):
    exponents = exponents.to(tl.uint32)
    scales = tl.where(exponents == 0, 0x00400000, exponents << 23)
    scales = scales.to(tl.float32, bitcast=True)
    return (values.to(tl.float32) * scales).to(tl.bfloat16)


@triton.jit
def _expand_scales(scales):
    rows: tl.constexpr = scales.shape[0]
    groups: tl.constexpr = scales.shape[1]
    scales = tl.broadcast_to(scales[:, :, None], (rows, groups, 32))
    return tl.reshape(scales, (rows, groups * 32))


@triton.jit
def _mxfp8_gemm_native(
    Act,
    ActScale,
    Weight,
    WeightScale,
    ExpertScale,
    Out,
    Tiles,
    MAX_CTAS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Native SM120 hardware MXFP8 block-scaled MMA (2x BF16 Tensor Core rate)."""
    tl.extra.cuda.gdc_wait()
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.program_id(1) * BM + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    col_mask = cols < N
    weight = Weight + expert * N * K
    weight_scale = WeightScale + expert * N * (K // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        kp = k + tl.arange(0, BK)
        kg = k // 32 + tl.arange(0, BK // 32)
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * K + kp[None, :],
            valid[:, None] & (kp[None, :] < K),
            other=0.0,
        )
        a_scale = tl.load(
            ActScale + _scale_offsets(rows[:, None], kg[None, :], K // 32, TILE),
            valid[:, None] & (kg[None, :] < K // 32),
            other=0,
        )
        # w is [BK, BN] (transposed) with w_scale [BN, BK // 32] for tl.dot_scaled.
        w = tl.load(
            weight + cols[None, :].to(tl.int64) * K + kp[:, None],
            col_mask[None, :] & (kp[:, None] < K),
            other=0.0,
        )
        w_scale = tl.load(
            weight_scale + cols[:, None].to(tl.int64) * (K // 32) + kg[None, :],
            col_mask[:, None] & (kg[None, :] < K // 32),
            other=0,
        )
        acc = tl.dot_scaled(a, a_scale, "e4m3", w, w_scale, "e4m3", acc)
    acc = acc * tl.load(ExpertScale + expert)
    tl.store(
        Out + rows[:, None].to(tl.int64) * N + cols[None, :],
        acc.to(tl.bfloat16),
        valid[:, None] & col_mask[None, :],
    )


@triton.jit
def _mxfp8_gemm(
    Act,
    ActScale,
    Weight,
    WeightScale,
    ExpertScale,
    Out,
    Tiles,
    MAX_CTAS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    T_BLOCKS: tl.constexpr = tl.cdiv(TILE, BT)
    tl.extra.cuda.gdc_wait()
    pid_t = tl.program_id(1)
    tile = pid_t // T_BLOCKS
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + (pid_t % T_BLOCKS) * BT + tl.arange(0, BT)
    row_mask = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    col_mask = cols < N
    weight = Weight + expert * N * K
    weight_scale = WeightScale + expert * N * (K // 32)
    acc = tl.zeros((BN, BT), tl.float32)
    for k in range(0, K, BK):
        kp = k + tl.arange(0, BK)
        kg = k // 32 + tl.arange(0, BK // 32)
        w = tl.load(
            weight + cols[:, None].to(tl.int64) * K + kp[None, :],
            col_mask[:, None],
        )
        w_scale = _expand_scales(
            tl.load(
                weight_scale + cols[:, None] * (K // 32) + kg[None, :],
                col_mask[:, None],
            )
        )
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * K + kp[None, :],
            row_mask[:, None],
        )
        a_scale = _expand_scales(
            tl.load(
                ActScale + _scale_offsets(rows[:, None], kg[None, :], K // 32, TILE),
                row_mask[:, None],
            )
        )
        acc = tl.dot(
            _dequant_e4m3(w, w_scale), tl.trans(_dequant_e4m3(a, a_scale)), acc
        )
    acc = acc * tl.load(ExpertScale + expert)
    tl.store(
        Out + rows[None, :].to(tl.int64) * N + cols[:, None],
        acc.to(tl.bfloat16),
        row_mask[None, :] & col_mask[:, None],
    )


def bmm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan: ExpertPlan,
    projection: int,
) -> torch.Tensor:
    """Multiply quantized expert rows, returning bf16 `[plan.rows, N]`."""
    del projection
    if weight.dtype == torch.float8_e4m3fn:
        n, k = weight.shape[1:]
        assert k % 128 == 0
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
        if os.getenv("INKLING_SM120_NATIVE_MXFP8", "1") != "0":
            if plan.tile <= 16:
                bm, bn, bk, warps, stages = 16, 128, 256, 4, 2
            elif plan.tile <= 64:
                bm, bn, bk, warps, stages = 64, 128, 256, 4, 2
            else:
                bm, bn, bk, warps, stages = 128, 128, 128, 8, 3
            _mxfp8_gemm_native[
                (plan.max_ctas, triton.cdiv(plan.tile, bm), triton.cdiv(n, bn))
            ](
                act,
                act_scale,
                weight,
                weight_scale,
                ones,
                out,
                tiles,
                MAX_CTAS=plan.max_ctas,
                N=n,
                K=k,
                TILE=plan.tile,
                BM=bm,
                BN=bn,
                BK=bk,
                num_warps=warps,
                num_stages=stages,
                launch_pdl=True,
            )
            return out
        if plan.tile <= 16:
            block_n, block_t, num_warps = 128, 16, 4
        elif plan.tile <= 64:
            block_n, block_t, num_warps = 128, 64, 4
        else:
            block_n, block_t, num_warps = 256, 128, 8
        _mxfp8_gemm[
            (triton.cdiv(n, block_n), plan.max_ctas * triton.cdiv(plan.tile, block_t))
        ](
            act,
            act_scale,
            weight,
            weight_scale,
            ones,
            out,
            tiles,
            MAX_CTAS=plan.max_ctas,
            N=n,
            K=k,
            TILE=plan.tile,
            BN=block_n,
            BT=block_t,
            BK=128,
            num_warps=num_warps,
            num_stages=3,
            launch_pdl=True,
        )
        return out

    assert weight.dtype == torch.uint8
    n, k = weight.shape[1], weight.shape[2] * 2
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tuned = os.getenv("INKLING_SM120_TUNED_NVFP4", "1") != "0"

    # Low-M Split-K opt-in only (stock M=1 NVFP4 is already 23us on RTX 5090).
    if tuned and os.getenv("INKLING_SM120_NVFP4_SPLITK", "0") == "1" and plan.tile <= 8 and plan.max_ctas <= 32 and k >= 1024:
        split_k = 4 if k >= 2048 else 2
        bm, bn, bk = 16, 64, 256
        partials = torch.empty(
            (split_k, plan.rows, n), dtype=torch.float32, device=act.device
        )
        _expert_gemm_splitk[(plan.max_ctas, triton.cdiv(n, bn), split_k)](
            act,
            act_scale,
            weight,
            weight_scale,
            partials,
            tiles,
            MAX_CTAS=plan.max_ctas,
            TOTAL_ROWS=plan.rows,
            N=n,
            K=k,
            TILE=plan.tile,
            BM=bm,
            BN=bn,
            BK=bk,
            SPLIT_K=split_k,
            num_warps=4,
            num_stages=3,
            launch_pdl=True,
        )
        _expert_splitk_reduce[(plan.max_ctas, plan.tile, triton.cdiv(n, 256))](
            partials,
            row_scale,
            ones,
            out,
            tiles,
            MAX_CTAS=plan.max_ctas,
            TOTAL_ROWS=plan.rows,
            N=n,
            TILE=plan.tile,
            SPLIT_K=split_k,
            BLOCK=256,
            num_warps=4,
            launch_pdl=True,
        )
        return out

    if tuned:
        if plan.tile <= 16:
            block_m, block_n, block_k, num_warps, stages = 16, 64, 512, 4, 2
        elif plan.tile <= 64:
            block_m, block_n, block_k, num_warps, stages = 64, 64, 512, 4, 2
        else:
            block_m, block_n, block_k, num_warps, stages = 128, 128, 256, 8, 2
        pdl = True
    else:
        block_m, block_n, block_k, num_warps = 32, 64, 512, 4
        if plan.tile >= 128 and n == 2304 and k in (1152, 2304):
            block_m = 128
            block_n = 128 if plan.tile >= 256 else 64
            num_warps = 8
        stages, pdl = 2, False

    _expert_gemm[
        (plan.max_ctas, triton.cdiv(plan.tile, block_m), triton.cdiv(n, block_n))
    ](
        act,
        act_scale,
        row_scale,
        weight,
        weight_scale,
        ones,
        out,
        tiles,
        MAX_CTAS=plan.max_ctas,
        N=n,
        K=k,
        TILE=plan.tile,
        BM=block_m,
        BN=block_n,
        BK=block_k,
        num_warps=num_warps,
        num_stages=stages,
        launch_pdl=pdl,
    )
    return out
