"""
connector.py
============
Multimodal connector module: two-layer GeLU MLP.

The connector bridges perception encoders and the BitNet language backbone.
It is maintained in BF16 or FP16 throughout all training phases and is
never subjected to aggressive quantisation, consistent with the BitVLA
precedent [BitVLA, 2024] which attributes the connector's suitability for
high precision to its relatively small parameter contribution and its
critical role in cross-modal alignment.

Architecture
------------
::

    [encoder output — d_enc] → Linear(d_enc, d_hidden) → GeLU
                              → Linear(d_hidden, d_model) → output

References
----------
- BitVLA: Multimodal Large Language Models with 1-bit LLMs. 2024.
- Language Is Not All You Need: Aligning Perception with Language Models.
  Huang et al., 2023. arXiv:2302.14045.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModalityConnector(nn.Module):
    """
    Two-layer GeLU MLP connector between a perception encoder and the BitNet core.

    This module projects encoder output representations into the input
    embedding space of the BitNet language backbone.  It is intentionally
    kept in full precision (BF16/FP16) and is not quantised, as consistent
    with the BitVLA training protocol.

    Parameters
    ----------
    d_enc : int
        Dimensionality of the encoder output representation.
    d_model : int
        Dimensionality of the BitNet backbone input (i.e., model hidden size).
    d_hidden : int, optional
        Hidden dimension of the MLP intermediate layer.  Defaults to
        ``max(d_enc, d_model)``.
    dropout : float, optional
        Dropout probability applied after the first linear projection.
        Default is ``0.0`` (no dropout).
    dtype : torch.dtype, optional
        Computation dtype for this module. Default is ``torch.bfloat16``.

    Examples
    --------
    >>> connector = ModalityConnector(d_enc=1024, d_model=2048)
    >>> enc_out = torch.randn(2, 64, 1024)
    >>> projected = connector(enc_out)
    >>> assert projected.shape == (2, 64, 2048)
    """

    def __init__(
        self,
        d_enc: int,
        d_model: int,
        d_hidden: Optional[int] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if d_hidden is None:
            d_hidden = max(d_enc, d_model)

        self.d_enc = d_enc
        self.d_model = d_model
        self.d_hidden = d_hidden
        self._dtype = dtype

        self.proj_in = nn.Linear(d_enc, d_hidden, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.proj_out = nn.Linear(d_hidden, d_model, bias=True)

        self._init_weights()
        self.to(dtype)

        logger.info(
            "ModalityConnector initialised: d_enc=%d → d_hidden=%d → d_model=%d, dtype=%s.",
            d_enc, d_hidden, d_model, dtype,
        )

    def _init_weights(self) -> None:
        """
        Initialise connector weights using scaled Xavier uniform initialisation.
        """
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project encoder representations into the BitNet embedding space.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Encoder output tensor of shape ``(batch, seq_len, d_enc)``.

        Returns
        -------
        torch.Tensor
            Projected tensor of shape ``(batch, seq_len, d_model)`` in BF16/FP16.
        """
        assert encoder_output.shape[-1] == self.d_enc, (
            f"Expected encoder output dim {self.d_enc}, "
            f"got {encoder_output.shape[-1]}."
        )
        x = encoder_output.to(self._dtype)
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj_out(x)
        return x

    def freeze(self) -> None:
        """Freeze all connector parameters (requires_grad=False)."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("ModalityConnector frozen.")

    def unfreeze(self) -> None:
        """Unfreeze all connector parameters (requires_grad=True)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("ModalityConnector unfrozen.")

    def extra_repr(self) -> str:
        return (
            f"d_enc={self.d_enc}, d_hidden={self.d_hidden}, "
            f"d_model={self.d_model}, dtype={self._dtype}"
        )
