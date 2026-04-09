"""
bit_layers.py
=============
BitLinear layer and ternary weight quantisation utilities.

This module implements the core low-precision linear projection layer
described in BitNet b1.58 [Wang et al., 2023]. ``BitLinear`` serves as
a drop-in replacement for ``torch.nn.Linear``, quantising weights to the
ternary set ``{-1, 0, +1}`` during the forward pass whilst retaining
full-precision weights for the backward pass (straight-through estimator).

References
----------
- BitNet: Scaling 1-bit Transformers for Large Language Models.
  Wang et al., 2023. arXiv:2310.11453.
- The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.
  Ma et al., 2024. arXiv:2402.17764.
- BitNet a4.8: 4-bit Activations for 1-bit LLMs.
  Wang et al., 2024. arXiv:2411.04965.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantisation functions
# ---------------------------------------------------------------------------

def quantise_weights_ternary(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Quantise a weight tensor to ternary values ``{-1, 0, +1}``.

    The quantisation scheme follows the absmean scaling approach introduced
    in BitNet b1.58: the scale factor is computed as the mean absolute value
    of the weight tensor, and each weight is rounded to the nearest integer
    within ``[-1, +1]`` after scaling.

    Parameters
    ----------
    weight : torch.Tensor
        Full-precision weight tensor of arbitrary shape.
    eps : float, optional
        Small constant for numerical stability in the scale denominator.
        Default is ``1e-8``.

    Returns
    -------
    torch.Tensor
        Ternary weight tensor of the same shape, with values in ``{-1, 0, +1}``,
        cast to the same dtype as the input.
    """
    scale: float = weight.abs().mean().clamp(min=eps).item()
    quantised = (weight / scale).round().clamp(-1, 1)
    return quantised.to(weight.dtype)


def quantise_activations_8bit(
    x: torch.Tensor,
    quant_range: int = 127,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Quantise an activation tensor to 8-bit integer range via absmax scaling.

    Parameters
    ----------
    x : torch.Tensor
        Input activation tensor.
    quant_range : int, optional
        Symmetric quantisation range (typically 127 for INT8). Default is 127.
    eps : float, optional
        Numerical stability constant. Default is ``1e-8``.

    Returns
    -------
    torch.Tensor
        Quantised activation tensor re-scaled to the original dynamic range,
        with values rounded to the nearest representable level.
    """
    scale = x.abs().max().clamp(min=eps) / quant_range
    quantised = (x / scale).round().clamp(-quant_range, quant_range) * scale
    return quantised.to(x.dtype)


# ---------------------------------------------------------------------------
# BitLinear layer
# ---------------------------------------------------------------------------

class BitLinear(nn.Linear):
    """
    Ternary linear projection layer for BitNet b1.58 architectures.

    This layer extends ``torch.nn.Linear`` with straight-through ternary
    weight quantisation and optional 8-bit activation quantisation, as
    specified in BitNet b1.58.  During training, gradients flow through
    the full-precision weights via the straight-through estimator.  During
    inference, only the quantised weights are required.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If ``False``, no additive bias is used. Default is ``False``,
        consistent with the BitNet specification which omits bias terms.
    quantise_activations : bool, optional
        If ``True``, 8-bit activation quantisation is applied to the input
        before the linear projection. Default is ``True``.
    eps : float, optional
        Numerical stability constant for quantisation scaling. Default is ``1e-8``.

    Notes
    -----
    Sub-Layer Normalisation (SubLN) should be applied *before* this layer
    by the enclosing Transformer block, as per the BitNet specification.
    Bias terms are excluded by default in accordance with that specification.

    Examples
    --------
    >>> layer = BitLinear(in_features=512, out_features=512)
    >>> x = torch.randn(4, 16, 512)
    >>> out = layer(x)
    >>> assert out.shape == (4, 16, 512)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quantise_activations: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.quantise_activations = quantise_activations
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the ternary-quantised linear projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., out_features)``.
        """
        # Quantise activations (8-bit absmax) if enabled
        if self.quantise_activations:
            x_q = quantise_activations_8bit(x, eps=self.eps)
        else:
            x_q = x

        # Quantise weights to {-1, 0, +1} with straight-through estimator:
        # the gradient is computed w.r.t. the full-precision self.weight.
        w_q = quantise_weights_ternary(self.weight, eps=self.eps)
        w_q = self.weight + (w_q - self.weight).detach()

        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quantise_activations={self.quantise_activations}"
        )


# ---------------------------------------------------------------------------
# Utility: replace nn.Linear with BitLinear in a module
# ---------------------------------------------------------------------------

def replace_linear_with_bitlinear(
    module: nn.Module,
    skip_modules: Optional[list[str]] = None,
    quantise_activations: bool = True,
) -> nn.Module:
    """
    Recursively replace all ``nn.Linear`` layers in a module with ``BitLinear``.

    This utility facilitates the conversion of an existing full-precision
    Transformer backbone to BitNet by substituting linear projections in-place.

    Parameters
    ----------
    module : nn.Module
        The root module to traverse.
    skip_modules : list of str, optional
        List of attribute names to skip during replacement (e.g. output
        heads that should remain in FP16/BF16).
    quantise_activations : bool, optional
        Whether to enable 8-bit activation quantisation in the replaced layers.
        Default is ``True``.

    Returns
    -------
    nn.Module
        The modified module (in-place modification; same object is returned).

    Notes
    -----
    The *embedding* layer and *layer-norm* layers are not of type
    ``nn.Linear`` and will not be affected by this replacement.
    """
    if skip_modules is None:
        skip_modules = []

    for name, child in module.named_children():
        if name in skip_modules:
            continue
        if isinstance(child, nn.Linear) and not isinstance(child, BitLinear):
            bit_layer = BitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                quantise_activations=quantise_activations,
            )
            # Transfer full-precision weights
            bit_layer.weight = child.weight
            if child.bias is not None:
                bit_layer.bias = child.bias
            setattr(module, name, bit_layer)
            logger.debug("Replaced nn.Linear '%s' with BitLinear.", name)
        else:
            replace_linear_with_bitlinear(child, skip_modules, quantise_activations)

    return module


# ---------------------------------------------------------------------------
# ReLU² activation (BitNet specification)
# ---------------------------------------------------------------------------

class ReLUSquared(nn.Module):
    """
    Squared ReLU activation function as specified in the BitNet architecture.

    Defined as :math:`\\text{ReLU}^2(x) = \\max(0, x)^2`.

    References
    ----------
    - Primer: Searching for Efficient Transformers for Language Modeling.
      So et al., 2021. arXiv:2109.08668.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squared ReLU activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor of the same shape.
        """
        return F.relu(x) ** 2
