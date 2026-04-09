"""
losses.py
=========
Composite training loss functions for the hybrid BitNet pipeline.

This module implements all loss components specified in Section 6 of
the architecture specification:

  1. **Language loss** — Autoregressive cross-entropy.
  2. **Distillation loss** — KL divergence between teacher and student logits.
  3. **Alignment loss** — Distance between intermediate encoder and connector
     representations.
  4. **Task loss** — Classification, regression, or action objectives.
  5. **Stability loss** — Regularisation targeting activation outliers and
     per-channel sensitivity, following BitNet a4.8 [Wang et al., 2024].

References
----------
- BitNet a4.8: 4-bit Activations for 1-bit LLMs. Wang et al., 2024.
- BitVLA: Multimodal Large Language Models with 1-bit LLMs. 2024.
- Distilling the Knowledge in a Neural Network. Hinton et al., 2015.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Language loss (autoregressive cross-entropy)
# ---------------------------------------------------------------------------

def language_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute the autoregressive language modelling cross-entropy loss.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits of shape ``(batch, seq_len, vocab_size)``.
    labels : torch.Tensor
        Target token indices of shape ``(batch, seq_len)``.
        Positions with ``ignore_index`` are excluded from the loss.
    ignore_index : int, optional
        Index to ignore in the loss computation (typically padding or
        prompt tokens). Default is ``-100``.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss.
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


# ---------------------------------------------------------------------------
# 2. Distillation loss (KL divergence)
# ---------------------------------------------------------------------------

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Compute the knowledge distillation loss via KL divergence.

    Soft targets are obtained by scaling logits by a temperature parameter
    before applying the softmax, following Hinton et al. [2015].  The loss
    is scaled by ``temperature ** 2`` to preserve gradient magnitudes.

    Parameters
    ----------
    student_logits : torch.Tensor
        Student model logits of shape ``(batch, seq_len, vocab_size)`` or
        ``(N, vocab_size)`` for flattened sequences.
    teacher_logits : torch.Tensor
        Teacher model logits of the same shape as ``student_logits``.
        The teacher must produce logits over an identical vocabulary.
    temperature : float, optional
        Softmax temperature. Higher values produce softer distributions.
        Default is ``4.0``.

    Returns
    -------
    torch.Tensor
        Scalar KL-divergence distillation loss.
    """
    assert student_logits.shape == teacher_logits.shape, (
        "Student and teacher logits must have identical shapes. "
        f"Got {student_logits.shape} vs {teacher_logits.shape}."
    )
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)

    loss = F.kl_div(
        student_log_probs.view(-1, student_log_probs.size(-1)),
        teacher_probs.view(-1, teacher_probs.size(-1)),
        reduction="batchmean",
    ) * (temperature ** 2)
    return loss


# ---------------------------------------------------------------------------
# 3. Representation alignment loss
# ---------------------------------------------------------------------------

def alignment_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """
    Compute the representation alignment loss between teacher and student embeddings.

    This loss encourages the student connector output to remain close to the
    teacher encoder representations in embedding space, as employed by BitVLA.

    Parameters
    ----------
    student_embeddings : torch.Tensor
        Student connector output of shape ``(batch, seq_len, d_model)`` or
        ``(N, d_model)``.
    teacher_embeddings : torch.Tensor
        Teacher encoder output of shape matching ``student_embeddings``.
    mode : str, optional
        Distance metric. Supported values:
        - ``"cosine"`` — Mean negative cosine similarity (default).
        - ``"mse"``    — Mean squared error.
        - ``"l1"``     — Mean absolute error.

    Returns
    -------
    torch.Tensor
        Scalar alignment loss.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported distance metrics.
    """
    s = student_embeddings.view(-1, student_embeddings.size(-1))
    t = teacher_embeddings.view(-1, teacher_embeddings.size(-1)).detach()

    if mode == "cosine":
        return 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
    elif mode == "mse":
        return F.mse_loss(s, t)
    elif mode == "l1":
        return F.l1_loss(s, t)
    else:
        raise ValueError(
            f"Unsupported alignment loss mode: '{mode}'. "
            "Choose from 'cosine', 'mse', or 'l1'."
        )


# ---------------------------------------------------------------------------
# 4. Task-specific losses
# ---------------------------------------------------------------------------

def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy loss for classification heads.

    Parameters
    ----------
    logits : torch.Tensor
        Classification logits of shape ``(batch, num_classes)``.
    targets : torch.Tensor
        Ground-truth class indices of shape ``(batch,)``.
    label_smoothing : float, optional
        Label smoothing factor in ``[0, 1)``. Default is ``0.0``.

    Returns
    -------
    torch.Tensor
        Scalar classification loss.
    """
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    """
    Regression loss for continuous output heads.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted values of shape ``(batch, output_dim)``.
    targets : torch.Tensor
        Ground-truth values of the same shape.
    mode : str, optional
        Loss function: ``"mse"`` (default) or ``"huber"``.

    Returns
    -------
    torch.Tensor
        Scalar regression loss.
    """
    if mode == "mse":
        return F.mse_loss(predictions, targets)
    elif mode == "huber":
        return F.huber_loss(predictions, targets)
    else:
        raise ValueError(f"Unsupported regression mode: '{mode}'.")


# ---------------------------------------------------------------------------
# 5. Stability / outlier regularisation loss
# ---------------------------------------------------------------------------

def stability_loss(
    activations: torch.Tensor,
    lambda_outlier: float = 1e-3,
    outlier_percentile: float = 0.99,
) -> torch.Tensor:
    """
    Regularisation loss penalising activation outlier channels.

    This loss targets the per-channel outlier magnitudes that constitute
    the primary obstacle to aggressive activation quantisation, as
    identified in BitNet a4.8 [Wang et al., 2024].  It penalises
    activations whose absolute value exceeds the specified percentile
    threshold, encouraging the model to distribute dynamic range more
    uniformly across channels.

    Parameters
    ----------
    activations : torch.Tensor
        Activation tensor of shape ``(batch, seq_len, d_model)`` or any
        shape broadcastable to ``(..., d_model)``.
    lambda_outlier : float, optional
        Regularisation coefficient. Default is ``1e-3``.
    outlier_percentile : float, optional
        Percentile threshold above which activations are penalised.
        Default is ``0.99``.

    Returns
    -------
    torch.Tensor
        Scalar stability regularisation loss.
    """
    flat = activations.detach().abs().view(-1)
    threshold = torch.quantile(flat, outlier_percentile)
    outlier_mask = activations.abs() > threshold
    outlier_penalty = (activations[outlier_mask] ** 2).mean() if outlier_mask.any() else torch.tensor(0.0)
    return lambda_outlier * outlier_penalty


# ---------------------------------------------------------------------------
# Composite loss aggregator
# ---------------------------------------------------------------------------

class HybridLoss(nn.Module):
    """
    Aggregator for the full composite training loss.

    Combines language, distillation, alignment, task, and stability losses
    with configurable scalar weights per component.

    Parameters
    ----------
    w_language : float
        Weight for the language cross-entropy component. Default is ``1.0``.
    w_distill : float
        Weight for the KL distillation component. Default is ``0.5``.
    w_align : float
        Weight for the representation alignment component. Default is ``0.1``.
    w_task : float
        Weight for the task-specific loss component. Default is ``1.0``.
    w_stability : float
        Weight for the stability regularisation component. Default is ``1e-3``.
    temperature : float
        Distillation temperature. Default is ``4.0``.
    alignment_mode : str
        Distance metric for alignment loss. Default is ``"cosine"``.
    """

    def __init__(
        self,
        w_language: float = 1.0,
        w_distill: float = 0.5,
        w_align: float = 0.1,
        w_task: float = 1.0,
        w_stability: float = 1e-3,
        temperature: float = 4.0,
        alignment_mode: str = "cosine",
    ) -> None:
        super().__init__()
        self.w_language = w_language
        self.w_distill = w_distill
        self.w_align = w_align
        self.w_task = w_task
        self.w_stability = w_stability
        self.temperature = temperature
        self.alignment_mode = alignment_mode

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        student_embeddings: Optional[torch.Tensor] = None,
        teacher_embeddings: Optional[torch.Tensor] = None,
        task_loss: Optional[torch.Tensor] = None,
        activations: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute and aggregate all active loss components.

        Parameters
        ----------
        student_logits : torch.Tensor
            Student model output logits.
        labels : torch.Tensor
            Ground-truth token labels.
        teacher_logits : torch.Tensor, optional
            Teacher logits for distillation (omit to skip).
        student_embeddings : torch.Tensor, optional
            Student connector output embeddings (omit to skip alignment).
        teacher_embeddings : torch.Tensor, optional
            Teacher encoder output embeddings (omit to skip alignment).
        task_loss : torch.Tensor, optional
            Pre-computed task-specific loss (omit to skip).
        activations : torch.Tensor, optional
            Intermediate activations for stability regularisation (omit to skip).

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary with keys ``"total"``, ``"language"``, ``"distill"``,
            ``"align"``, ``"task"``, ``"stability"`` containing individual
            and aggregated scalar losses.
        """
        losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=student_logits.device)

        # Language loss (always active)
        l_lang = language_loss(student_logits, labels)
        losses["language"] = l_lang
        total = total + self.w_language * l_lang

        # Distillation loss
        if teacher_logits is not None:
            l_distill = distillation_loss(student_logits, teacher_logits, self.temperature)
            losses["distill"] = l_distill
            total = total + self.w_distill * l_distill
        else:
            losses["distill"] = torch.tensor(0.0)

        # Alignment loss
        if student_embeddings is not None and teacher_embeddings is not None:
            l_align = alignment_loss(student_embeddings, teacher_embeddings, self.alignment_mode)
            losses["align"] = l_align
            total = total + self.w_align * l_align
        else:
            losses["align"] = torch.tensor(0.0)

        # Task loss
        if task_loss is not None:
            losses["task"] = task_loss
            total = total + self.w_task * task_loss
        else:
            losses["task"] = torch.tensor(0.0)

        # Stability loss
        if activations is not None:
            l_stab = stability_loss(activations)
            losses["stability"] = l_stab
            total = total + self.w_stability * l_stab
        else:
            losses["stability"] = torch.tensor(0.0)

        losses["total"] = total
        return losses
