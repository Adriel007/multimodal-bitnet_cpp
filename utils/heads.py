"""
heads.py
========
Task-specific output heads for the hybrid BitNet architecture.

All heads are maintained in high precision (FP16/BF16) and operate on
the final hidden states produced by the BitNet backbone.  This design,
consistent with BitVLA [2024], separates the ternary-quantised backbone
from the task-sensitive output components that benefit from higher
numerical precision.

Supported heads
---------------
- ``ClassificationHead`` — Multi-class linear classifier.
- ``RegressionHead``     — Continuous output projection.
- ``ActionHead``         — Control / action output with configurable
                           activation bounds.

References
----------
- BitVLA: Multimodal Large Language Models with 1-bit LLMs. 2024.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Linear classification head operating on pooled hidden states.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input hidden states.
    num_classes : int
        Number of output classes.
    pooling : str, optional
        Sequence pooling strategy prior to classification.
        Supported: ``"mean"`` (default) or ``"cls"`` (first token).
    dropout : float, optional
        Dropout probability applied before the linear layer. Default is ``0.1``.
    dtype : torch.dtype, optional
        Computation dtype. Default is ``torch.bfloat16``.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        pooling: str = "mean",
        dropout: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert pooling in ("mean", "cls"), (
            f"Unsupported pooling strategy: '{pooling}'. Choose 'mean' or 'cls'."
        )
        self.pooling = pooling
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, num_classes, bias=True)
        self.to(dtype)
        logger.info(
            "ClassificationHead: d_model=%d, num_classes=%d, pooling=%s.",
            d_model, num_classes, pooling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute classification logits.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Backbone output of shape ``(batch, seq_len, d_model)``.
        attention_mask : torch.Tensor, optional
            Binary mask of shape ``(batch, seq_len)``. Required when
            ``pooling="mean"`` to exclude padding tokens.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            else:
                pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.linear(pooled)


class RegressionHead(nn.Module):
    """
    Linear regression head for continuous output prediction.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input hidden states.
    output_dim : int
        Dimensionality of the continuous output vector.
    pooling : str, optional
        Sequence pooling strategy. Supported: ``"mean"`` or ``"cls"``.
        Default is ``"mean"``.
    dropout : float, optional
        Dropout probability. Default is ``0.1``.
    dtype : torch.dtype, optional
        Computation dtype. Default is ``torch.bfloat16``.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        pooling: str = "mean",
        dropout: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert pooling in ("mean", "cls")
        self.pooling = pooling
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, output_dim, bias=True)
        self.to(dtype)
        logger.info(
            "RegressionHead: d_model=%d, output_dim=%d, pooling=%s.",
            d_model, output_dim, pooling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute continuous output predictions.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Backbone output of shape ``(batch, seq_len, d_model)``.
        attention_mask : torch.Tensor, optional
            Binary mask for mean pooling.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(batch, output_dim)``.
        """
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            else:
                pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.linear(pooled)


class ActionHead(nn.Module):
    """
    Action / control output head for robotics, RL, or agentic tasks.

    Maintained in full precision (BF16/FP16) with an optional output
    activation to bound the action range. Consistent with the BitVLA
    separation of backbone from task-sensitive control components.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input hidden states.
    action_dim : int
        Dimensionality of the action output vector.
    action_bounds : tuple of float, optional
        If provided, a Tanh activation is applied and actions are rescaled
        to the interval ``[action_bounds[0], action_bounds[1]]``.
    dropout : float, optional
        Dropout probability. Default is ``0.0``.
    dtype : torch.dtype, optional
        Computation dtype. Default is ``torch.float32`` for maximum precision
        in control-critical applications.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        action_bounds: Optional[tuple[float, float]] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.action_bounds = action_bounds
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_model, action_dim, bias=True)
        self.to(dtype)
        logger.info(
            "ActionHead: d_model=%d, action_dim=%d, bounds=%s, dtype=%s.",
            d_model, action_dim, action_bounds, dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute action predictions from the last hidden state.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Backbone output of shape ``(batch, seq_len, d_model)``.
            The last token position is used for action decoding.

        Returns
        -------
        torch.Tensor
            Action tensor of shape ``(batch, action_dim)``.
        """
        last_hidden = hidden_states[:, -1, :]  # last token
        last_hidden = self.dropout(last_hidden)
        actions = self.linear(last_hidden)

        if self.action_bounds is not None:
            lo, hi = self.action_bounds
            mid = (hi + lo) / 2.0
            scale = (hi - lo) / 2.0
            actions = torch.tanh(actions) * scale + mid

        return actions
