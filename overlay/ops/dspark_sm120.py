# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Ciru for Blackwell and NVFP4 KV caching.
"""SM120 DSpark fused add_softcap_argmax & Markov head Split-K GEMV helpers."""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .nvfp4_kv import (
    _quantize_nvfp4_kv_tile,
    get_nvfp4_kv_scale_tensor,
    nvfp4_kv_head_dim,
)


@triton.jit
def _add_softcap_argmax_stage1(
    A,
    B,
    Vals,
    Idxs,
    stride_a,
    stride_b,
    softcap,
    N,
    BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse bf16 add + tanh softcap + block argmax without writing 200K logits."""
    tl.extra.cuda.gdc_wait()
    row = tl.program_id(1).to(tl.int64)
    blk = tl.program_id(0)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask, other=0).to(tl.float32)
    x = (a + b).to(tl.bfloat16)
    x = tldevice.tanh(x / softcap) * softcap
    x = x.to(tl.bfloat16).to(tl.float32)
    # Match greedy_argmax NaN-as--inf semantics.
    x = tl.where(mask & (x == x), x, float("-inf"))
    best_val, best_local = tl.max(x, axis=0, return_indices=True)
    tl.store(Vals + row * BLOCKS + blk, best_val)
    tl.store(Idxs + row * BLOCKS + blk, blk * BLOCK + best_local)


@triton.jit
def _add_softcap_argmax_stage2(
    Vals,
    Idxs,
    Out,
    BLOCKS: tl.constexpr,
    PAD_BLOCKS: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, PAD_BLOCKS)
    mask = b < BLOCKS
    vals = tl.load(Vals + row * BLOCKS + b, mask, other=float("-inf"))
    idxs = tl.load(Idxs + row * BLOCKS + b, mask, other=0)
    _, best_slot = tl.max(vals, axis=0, return_indices=True)
    best_id = tl.load(Idxs + row * BLOCKS + best_slot)
    tl.store(Out + row, best_id)


def fused_add_softcap_argmax(
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    softcap: float,
    valid_vocab: int,
) -> torch.Tensor:
    """Single-pass fused `add_softcap(base, bias)[:, :valid_vocab].argmax(-1)`."""
    rows = base_logits.shape[0]
    out = torch.empty((rows,), dtype=torch.int64, device=base_logits.device)
    if not rows:
        return out
    block = 1024
    blocks = triton.cdiv(valid_vocab, block)
    vals = torch.empty((rows, blocks), dtype=torch.float32, device=base_logits.device)
    idxs = torch.empty((rows, blocks), dtype=torch.int64, device=base_logits.device)
    _add_softcap_argmax_stage1[(blocks, rows)](
        base_logits,
        bias,
        vals,
        idxs,
        base_logits.stride(0),
        bias.stride(0),
        softcap,
        valid_vocab,
        BLOCKS=blocks,
        BLOCK=block,
        num_warps=4,
        launch_pdl=True,
    )
    _add_softcap_argmax_stage2[(rows,)](
        vals,
        idxs,
        out,
        BLOCKS=blocks,
        PAD_BLOCKS=triton.next_power_of_2(blocks),
        num_warps=4,
        launch_pdl=True,
    )
    return out


@triton.jit
def _write_kv_cache_kernel(
    K,
    V,
    KeyCache,
    ValCache,
    KeyScales,
    ValScales,
    Slots,
    KScale,
    VScale,
    num_tokens,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_kv_blk,
    stride_kv_n,
    stride_kv_h,
    stride_sc_blk,
    stride_sc_n,
    stride_sc_h,
    block_size,
    USE_SCALE: tl.constexpr,
    NVFP4_KV: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    """SM120-native Triton KV cache writer replacing reshape_and_cache_flash.

    The official inkling-2-mini container's _C_stable_libtorch.abi3.so omits
    sm_120 cubin/PTX for reshape_and_cache_flash; this Triton kernel writes
    BF16, E4M3 FP8, and NVFP4 paged KV caches natively on sm_120.
    """
    toff = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)
    is_v = tl.program_id(2) == 1
    d_idx = tl.arange(0, D)
    t_mask = toff < num_tokens
    slots = tl.load(Slots + toff, mask=t_mask, other=-1)
    valid = t_mask & (slots >= 0)
    safe = tl.maximum(slots, 0).to(tl.int64)

    if is_v:
        src = V + toff.to(tl.int64)[:, None] * stride_vt + head * stride_vh + d_idx[None, :]
        dst_base = ValCache
        sc_base = ValScales
        scale_ptr = VScale
    else:
        src = K + toff.to(tl.int64)[:, None] * stride_kt + head * stride_kh + d_idx[None, :]
        dst_base = KeyCache
        sc_base = KeyScales
        scale_ptr = KScale

    vals = tl.load(src, mask=valid[:, None], other=0.0)
    blk_id = safe // block_size
    blk_off = safe % block_size
    dst_row = (
        dst_base
        + blk_id[:, None] * stride_kv_blk
        + blk_off[:, None] * stride_kv_n
        + head * stride_kv_h
    )
    if NVFP4_KV:
        packed, scales = _quantize_nvfp4_kv_tile(vals.to(tl.float32), BT, D)
        tl.store(dst_row + tl.arange(0, D // 2)[None, :], packed, mask=valid[:, None])
        sc_row = (
            sc_base
            + blk_id[:, None] * stride_sc_blk
            + blk_off[:, None] * stride_sc_n
            + head * stride_sc_h
        )
        tl.store(
            sc_row + tl.arange(0, D // 16)[None, :],
            scales,
            mask=valid[:, None],
        )
    else:
        if USE_SCALE:
            inv_scale = 1.0 / tl.load(scale_ptr).to(tl.float32)
            vals = (vals.to(tl.float32) * inv_scale).to(dst_base.dtype.element_ty)
        else:
            vals = vals.to(dst_base.dtype.element_ty)
        tl.store(dst_row + d_idx[None, :], vals, mask=valid[:, None])


def write_kv_cache_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Write `(T, H_kv, D)` K/V tensors into vLLM's `(B, H, N, 2*kv_D)` KV cache."""
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    num_kv_heads = k.shape[1]
    head_dim = k.shape[2]
    nvfp4_dim = nvfp4_kv_head_dim(head_dim)
    nvfp4_kv = kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == 2 * nvfp4_dim
    split_dim = nvfp4_dim if nvfp4_kv else head_dim
    is_fp8 = (not nvfp4_kv) and kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)
    if is_fp8 and kv_cache.dtype == torch.uint8:
        kv_cache = kv_cache.view(torch.float8_e4m3fn)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(split_dim, dim=-1)
    if nvfp4_kv:
        key_scales = get_nvfp4_kv_scale_tensor(key_cache)
        value_scales = get_nvfp4_kv_scale_tensor(value_cache)
    else:
        key_scales = key_cache
        value_scales = value_cache
    block_size = key_cache.shape[1]
    bt = 8 if num_tokens >= 64 else 4
    use_scale = is_fp8 and k_scale is not None and v_scale is not None
    scale_k = k_scale if use_scale else slot_mapping
    scale_v = v_scale if use_scale else slot_mapping
    grid = (triton.cdiv(num_tokens, bt), num_kv_heads, 2)
    _write_kv_cache_kernel[grid](
        k,
        v,
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        slot_mapping,
        scale_k,
        scale_v,
        num_tokens,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        block_size,
        USE_SCALE=use_scale,
        NVFP4_KV=nvfp4_kv,
        D=head_dim,
        BT=bt,
        num_warps=4,
    )
