# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Ciru for Blackwell and NVFP4 KV caching.
"""Blackwell SM120 NVFP4 KV cache pack/unpack kernels (fouroversix_mae).

Storage layout for head_dim=128 (zero hybrid page-padding waste):
  - Main vLLM KV cache (`head_size=64`, uint8): 64 packed E2M1 bytes per K or V
    head (128 bytes per KV head — exact power of 2, 50.0% of FP8 KV, 128-bit
    aligned for vectorized LDG.E.128 loads).
  - Sibling shared scale buffer (`1/8` of the KV storage bytes = 12.5% of FP8):
    8 float8_e4m3fn block scales per K or V head (1 scale per 16 elements,
    unit outer scale), mapped 1-to-1 from the underlying KV cache storage so
    hybrid KV cache groups share scales with the exact same aliasing invariant.

Total KV footprint: 56.25% of FP8 KV (72 bytes per K or V head, 43.75% smaller)
with zero hybrid block-alignment bloat. Inkling-2-Mini's L2-clipped K (<=22.6)
and RMS-normalized V (<=11.3) sit squarely inside E4M3 scale range [0.0156, 448].
"""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Shared backing scale storages keyed by (kv_storage_ptr, kv_storage_nbytes),
# and cached strided views keyed by (ptr, shape, stride).
_SCALE_STORAGE_REGISTRY: dict[tuple[int, int], torch.Tensor] = {}
_SCALE_VIEW_CACHE: dict[
    tuple[int, tuple[int, ...], tuple[int, ...]], torch.Tensor
] = {}


def nvfp4_kv_enabled(cache_dtype: str) -> bool:
    """True when NVFP4 KV packing is requested over FP8 KV cache storage on SM120."""
    return (
        os.getenv("INKLING_SM120_NVFP4_KV", "0") == "1"
        and cache_dtype in ("fp8", "fp8_e4m3")
        and current_platform.is_device_capability_family(120)
    )


def nvfp4_kv_head_dim(head_dim: int) -> int:
    """Power-of-2 payload width in bytes per K or V head (64 for D=128)."""
    assert head_dim % 16 == 0
    return head_dim // 2


def nvfp4_kv_scale_dim(head_dim: int) -> int:
    """Number of E4M3 block scales per K or V head (8 for D=128)."""
    assert head_dim % 16 == 0
    return head_dim // 16


def clear_nvfp4_kv_scales() -> None:
    """Release cached sibling scale buffers (useful in tests across engines)."""
    _SCALE_VIEW_CACHE.clear()
    _SCALE_STORAGE_REGISTRY.clear()


def get_nvfp4_kv_scale_tensor(cache: torch.Tensor) -> torch.Tensor:
    """Return the `(B, N, H, D//16)` uint8 E4M3 scale view aliasing `cache`.

    Every 8 bytes of packed E2M1 payload (16 FP4 elements) maps to 1 byte of
    E4M3 block scale at offset `byte_offset // 8` in a single shared flat scale
    storage sized `kv_storage.nbytes() // 8`.
    """
    view_key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()))
    cached_view = _SCALE_VIEW_CACHE.get(view_key)
    if cached_view is not None:
        return cached_view

    storage = cache.untyped_storage()
    nbytes = storage.nbytes()
    assert nbytes % 8 == 0, f"NVFP4 KV storage size {nbytes} must be multiple of 8"
    storage_key = (storage.data_ptr(), nbytes)
    scale_buf = _SCALE_STORAGE_REGISTRY.get(storage_key)
    if scale_buf is None:
        # Evict any prior large pool if a new engine pool is created in-process.
        if nbytes >= 128 * 1024 * 1024:
            stale = [k for k in _SCALE_STORAGE_REGISTRY if k[1] >= 128 * 1024 * 1024]
            for k in stale:
                _SCALE_STORAGE_REGISTRY.pop(k, None)
            _SCALE_VIEW_CACHE.clear()
        scale_buf = torch.zeros(
            (nbytes // 8,), dtype=torch.uint8, device=cache.device
        )
        _SCALE_STORAGE_REGISTRY[storage_key] = scale_buf

    assert cache.dtype == torch.uint8 and cache.stride(-1) == 1
    packed_dim = cache.shape[-1]
    assert packed_dim % 8 == 0
    scale_dim = packed_dim // 8
    byte_offset = cache.storage_offset()
    assert byte_offset % 8 == 0, f"Unaligned NVFP4 KV offset {byte_offset}"

    scale_shape = (*cache.shape[:-1], scale_dim)
    scale_strides = []
    for dim_sz, st in zip(cache.shape[:-1], cache.stride()[:-1]):
        if dim_sz > 1:
            assert st % 8 == 0, f"Unaligned NVFP4 KV stride {st} for dim {dim_sz}"
            scale_strides.append(st // 8)
        else:
            scale_strides.append(st // 8 if st % 8 == 0 else scale_dim)
    scale_strides.append(1)

    view = scale_buf.as_strided(scale_shape, tuple(scale_strides), byte_offset // 8)
    _SCALE_VIEW_CACHE[view_key] = view
    return view


@triton.jit
def _pack_e2m1_roundtrip_sm120(x_pairs):
    """Pack [..., 2] fp32 pairs to uint8 e2m1x2 and return fp32 roundtrip values."""
    low, high = tl.split(x_pairs)
    packed, halves = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $3, $2; "
        "cvt.u32.u8 $0, b; cvt.rn.f16x2.e2m1x2 $1, b; }",
        "=r,=r,f,f",
        [low, high],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    lo_f16 = (halves & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi_f16 = (halves >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return packed.to(tl.uint8), tl.join(lo_f16, hi_f16).to(tl.float32)


@triton.jit
def _quantize_nvfp4_kv_tile(y, BT: tl.constexpr, D: tl.constexpr):
    """Quantize fp32 [BT, D] K or V tile to NVFP4 (fouroversix_mae per 16-elem group).

    Returns:
      packed: [BT, D // 2] uint8
      scales: [BT, D // 16] uint8 (bitcast float8_e4m3fn)
    """
    G: tl.constexpr = D // 16
    yg = tl.reshape(y, (BT, G, 16))
    bmax = tl.max(tl.abs(yg), axis=2)  # [BT, G]

    s6 = tl.minimum(bmax * (1.0 / 6.0), 448.0).to(tl.float8e4nv).to(tl.float32)
    s4 = tl.minimum(bmax * 0.25, 448.0).to(tl.float8e4nv).to(tl.float32)
    inv6 = tl.where(s6 > 0.0, tl.div_rn(1.0, s6), 0.0)
    inv4 = tl.where(s4 > 0.0, tl.div_rn(1.0, s4), 0.0)

    p6, r6 = _pack_e2m1_roundtrip_sm120(tl.reshape(yg * inv6[:, :, None], (BT, G, 8, 2)))
    p4, r4 = _pack_e2m1_roundtrip_sm120(tl.reshape(yg * inv4[:, :, None], (BT, G, 8, 2)))

    rec6 = tl.reshape(r6, (BT, G, 16)) * s6[:, :, None]
    rec4 = tl.reshape(r4, (BT, G, 16)) * s4[:, :, None]
    err6 = tl.sum(tl.abs(rec6 - yg), axis=2)
    err4 = tl.sum(tl.abs(rec4 - yg), axis=2)

    choose4 = err4 < err6
    packed = tl.reshape(tl.where(choose4[:, :, None], p4, p6), (BT, D // 2))
    scales = tl.where(choose4, s4, s6).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    return packed, scales


@triton.jit
def _load_nvfp4_kv(
    ptr,
    sc_ptr,
    offsets,
    sc_offsets,
    mask,
    dtype: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Vectorized SM120 load & dequant of [BLOCK_N, D//2] NVFP4 + [BLOCK_N, D//16] E4M3 scales."""
    QUARTS: tl.constexpr = D // 4  # 32 uint16 words = 64 bytes per row (LDG.E.128)
    GROUPS: tl.constexpr = D // 16  # 8 E4M3 scales = 4 uint16 pairs per row
    w16 = tl.load(
        ptr.to(tl.pointer_type(tl.uint16))
        + tl.multiple_of(offsets // 2, QUARTS)[:, None]
        + tl.arange(0, QUARTS)[None, :],
        mask=mask[:, None],
        other=0,
    )
    sc_pair = tl.load(
        sc_ptr.to(tl.pointer_type(tl.uint16))
        + tl.multiple_of(sc_offsets // 2, GROUPS // 2)[:, None]
        + tl.arange(0, GROUPS // 2)[None, :],
        mask=mask[:, None],
        other=0,
    )
    sc_lo, sc_hi = tl.inline_asm_elementwise(
        "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",
        "=h,=h,h",
        [sc_pair],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    sc = tl.interleave(sc_lo, sc_hi)  # [BLOCK_N, GROUPS] float16
    sc_q = tl.reshape(
        tl.broadcast_to(sc[:, :, None], (BLOCK_N, GROUPS, 4)), (BLOCK_N, QUARTS)
    )
    e0, e1, e2, e3 = tl.inline_asm_elementwise(
        "{ .reg .b32 w, w1, h0, h1, s2; .reg .b8 q0, q1; "
        "cvt.u32.u16 w, $4; shr.u32 w1, w, 8; "
        "cvt.u8.u32 q0, w; cvt.u8.u32 q1, w1; "
        "cvt.rn.f16x2.e2m1x2 h0, q0; cvt.rn.f16x2.e2m1x2 h1, q1; "
        "mov.b32 s2, {$5, $5}; "
        "mul.rn.f16x2 h0, h0, s2; mul.rn.f16x2 h1, h1, s2; "
        "mov.b32 {$0, $1}, h0; mov.b32 {$2, $3}, h1; }",
        "=h,=h,=h,=h,h,h",
        [w16, sc_q],
        dtype=(tl.float16, tl.float16, tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    even = tl.interleave(e0, e2)
    odd = tl.interleave(e1, e3)
    return tl.interleave(even, odd).to(dtype)
