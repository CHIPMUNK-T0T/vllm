# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for ``GemmaRMSNorm.forward_cuda`` on the eager path.

Without these, ``forward_cuda`` falls back to ``forward_native``, which under
``enforce_eager`` runs as many separate copy/cast, reduce and elementwise
launches. These kernels fuse the common decode case (2D activations) into a
single launch. Inputs outside ``can_use_gemma_rms_norm`` use the native path.

Note: these are bare Triton launches for the eager path, not registered
``torch.ops`` custom ops, so they are not captured/opchecked under
``torch.compile`` (the compiled path keeps using the native IR fusion). See
also ``fused_allreduce_gemma_rms_norm`` for the TP attention-output fusion.
"""

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

# Largest hidden size the single-block reduction handles; covers Gemma/Qwen3.5.
_MAX_HIDDEN = 8192


@triton.jit
def _gemma_rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols,
    stride_x_row,
    stride_x_col,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(
        x_ptr + row * stride_x_row + cols * stride_x_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * rstd * (1.0 + weight)

    tl.store(
        out_ptr + row * n_cols + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gemma_fused_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols,
    stride_x_row,
    stride_x_col,
    stride_residual_row,
    stride_residual_col,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(
        x_ptr + row * stride_x_row + cols * stride_x_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * stride_residual_row + cols * stride_residual_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual_out = x + residual
    tl.store(
        residual_out_ptr + row * n_cols + cols,
        residual_out.to(residual_out_ptr.dtype.element_ty),
        mask=mask,
    )

    variance = tl.sum(residual_out * residual_out, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = residual_out * rstd * (1.0 + weight)

    tl.store(
        out_ptr + row * n_cols + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _num_warps(block_n: int) -> int:
    # One program per row does a full-row reduction; widen warps with the row.
    return min(16, max(4, block_n // 512))


def can_use_gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> bool:
    """Whether the fused Triton path supports these inputs.

    The supported set mirrors the common decode shape; everything else should
    use ``GemmaRMSNorm.forward_native``.
    """
    if not HAS_TRITON or not x.is_cuda or not weight.is_cuda:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if x.dim() != 2 or weight.dim() != 1 or x.shape[-1] != weight.shape[0]:
        return False
    if x.shape[-1] > _MAX_HIDDEN:
        return False
    if residual is None:
        return True
    return residual.is_cuda and residual.shape == x.shape and residual.dtype == x.dtype


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused Gemma RMSNorm. See ``can_use_gemma_rms_norm`` for the input set."""
    n_cols = x.shape[-1]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n_cols)

    _gemma_rms_norm_kernel[(x.shape[0],)](
        x,
        weight,
        out,
        n_cols,
        x.stride(0),
        x.stride(1),
        eps,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    return out


def gemma_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + Gemma RMSNorm, returning ``(out, updated_residual)``."""
    n_cols = x.shape[-1]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    residual_out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(n_cols)

    _gemma_fused_add_rms_norm_kernel[(x.shape[0],)](
        x,
        residual,
        weight,
        out,
        residual_out,
        n_cols,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        eps,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )
    return out, residual_out
