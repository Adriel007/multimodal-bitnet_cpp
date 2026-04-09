"""
training.py
===========
Training loop utilities and quantisation-aware training (QAT) helpers.

This module provides reusable infrastructure for all pipeline phases:

  - ``freeze_module`` / ``unfreeze_module`` — Granular parameter control.
  - ``get_trainable_param_count``            — Diagnostic logging.
  - ``build_optimiser``                      — AdamW with parameter groups.
  - ``build_scheduler``                      — Cosine annealing with warmup.
  - ``train_one_epoch``                      — Generic training loop.
  - ``evaluate``                             — Evaluation loop with metric accumulation.
  - ``QATScheduler``                         — Gradual quantisation strength ramp.

The continual QAT strategy implemented here follows the findings of the
continual quantisation-aware pre-training study cited in the architecture
specification: the 16-bit → 1.58-bit transition is preferred over training
entirely in low precision from scratch, and retaining the optimiser state
across phase transitions reduces loss spikes.

References
----------
- Continual quantization-aware pre-training for BitNet. 2024.
- BitNet a4.8: 4-bit Activations for 1-bit LLMs. Wang et al., 2024.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter group utilities
# ---------------------------------------------------------------------------

def freeze_module(module: nn.Module, name: str = "module") -> None:
    """
    Freeze all parameters of a module (``requires_grad = False``).

    Parameters
    ----------
    module : nn.Module
        The module to freeze.
    name : str, optional
        Descriptive name for logging purposes.
    """
    for param in module.parameters():
        param.requires_grad = False
    logger.info("Frozen: %s (%d params).", name, sum(p.numel() for p in module.parameters()))


def unfreeze_module(module: nn.Module, name: str = "module") -> None:
    """
    Unfreeze all parameters of a module (``requires_grad = True``).

    Parameters
    ----------
    module : nn.Module
        The module to unfreeze.
    name : str, optional
        Descriptive name for logging purposes.
    """
    for param in module.parameters():
        param.requires_grad = True
    logger.info("Unfrozen: %s (%d params).", name, sum(p.numel() for p in module.parameters()))


def get_trainable_param_count(model: nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.

    Returns
    -------
    tuple of (int, int)
        ``(trainable_count, total_count)``.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Parameters — trainable: %s / total: %s (%.1f%%).",
        f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1),
    )
    return trainable, total


# ---------------------------------------------------------------------------
# Optimiser and scheduler construction
# ---------------------------------------------------------------------------

def build_optimiser(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    no_decay_keywords: tuple[str, ...] = ("bias", "norm", "embedding"),
) -> AdamW:
    """
    Construct an AdamW optimiser with separate weight decay parameter groups.

    Parameters that match any of the ``no_decay_keywords`` patterns are
    excluded from weight decay, following standard practice for Transformer
    training.

    Parameters
    ----------
    model : nn.Module
        Model whose trainable parameters are to be optimised.
    lr : float, optional
        Base learning rate. Default is ``2e-4``.
    weight_decay : float, optional
        Weight decay coefficient for decayed parameters. Default is ``1e-2``.
    no_decay_keywords : tuple of str, optional
        Name substrings that identify parameters to exclude from weight decay.

    Returns
    -------
    AdamW
        Configured AdamW optimiser instance.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimiser = AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    logger.info(
        "AdamW optimiser built — lr=%.2e, wd=%.2e, "
        "decay params=%d, no-decay params=%d.",
        lr, weight_decay, len(decay), len(no_decay),
    )
    return optimiser


def build_scheduler(
    optimiser: AdamW,
    total_steps: int,
    warmup_steps: int = 100,
    eta_min: float = 1e-6,
) -> SequentialLR:
    """
    Construct a cosine annealing scheduler with linear warmup.

    Parameters
    ----------
    optimiser : AdamW
        The optimiser instance to schedule.
    total_steps : int
        Total number of training steps.
    warmup_steps : int, optional
        Number of linear warmup steps. Default is ``100``.
    eta_min : float, optional
        Minimum learning rate at the end of cosine decay. Default is ``1e-6``.

    Returns
    -------
    SequentialLR
        Combined warmup + cosine annealing scheduler.
    """
    warmup = LinearLR(optimiser, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimiser, T_max=max(total_steps - warmup_steps, 1), eta_min=eta_min
    )
    scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps])
    logger.info(
        "Scheduler: linear warmup (%d steps) + cosine annealing (%d steps).",
        warmup_steps, total_steps - warmup_steps,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimiser: AdamW,
    scheduler,
    loss_fn: Callable,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """
    Execute one training epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    dataloader : Iterable
        Training dataloader yielding batches.
    optimiser : AdamW
        Optimiser instance.
    scheduler : LRScheduler
        Learning rate scheduler (stepped per batch).
    loss_fn : Callable
        Loss function taking model outputs and returning a loss dict with
        key ``"total"``.
    device : torch.device
        Compute device.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping. Default is ``1.0``.
    scaler : GradScaler, optional
        AMP gradient scaler for mixed-precision training.
    log_interval : int, optional
        Number of steps between log emissions. Default is ``50``.

    Returns
    -------
    dict of str to float
        Epoch-level averages of all loss components.
    """
    model.train()
    accumulated: dict[str, float] = {}
    n_batches = 0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(**batch)
            losses = loss_fn(**outputs)

        total_loss: torch.Tensor = losses["total"]

        optimiser.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimiser)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimiser.step()

        scheduler.step()

        for k, v in losses.items():
            accumulated[k] = accumulated.get(k, 0.0) + float(v)
        n_batches += 1

        if (step + 1) % log_interval == 0:
            avg_total = accumulated.get("total", 0.0) / n_batches
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "Step %d | loss=%.4f | lr=%.2e.", step + 1, avg_total, current_lr
            )

    return {k: v / max(n_batches, 1) for k, v in accumulated.items()}


# ---------------------------------------------------------------------------
# QAT scheduler: gradual quantisation strength
# ---------------------------------------------------------------------------

class QATScheduler:
    """
    Gradual quantisation-aware training strength scheduler.

    Linearly ramps the quantisation strength from 0.0 (no quantisation,
    equivalent to full-precision training) to 1.0 (full ternary regime)
    over a configurable number of warmup steps.

    Parameters
    ----------
    total_steps : int
        Total number of QAT training steps.
    warmup_steps : int, optional
        Steps over which to ramp quantisation strength from 0 to 1.
        Default: ``total_steps // 5``.

    Notes
    -----
    The continual QAT study indicates that gradual introduction of
    quantisation strength offers limited but non-negligible benefits,
    particularly in reducing loss spikes during the 16-bit → 1.58-bit
    transition phase.
    """

    def __init__(self, total_steps: int, warmup_steps: Optional[int] = None) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps if warmup_steps is not None else total_steps // 5
        self._step = 0

    def get_strength(self) -> float:
        """
        Return the current quantisation strength in ``[0.0, 1.0]``.

        Returns
        -------
        float
            Quantisation strength coefficient.
        """
        if self._step >= self.warmup_steps:
            return 1.0
        return self._step / max(self.warmup_steps, 1)

    def step(self) -> None:
        """Advance the internal step counter by one."""
        self._step = min(self._step + 1, self.total_steps)
