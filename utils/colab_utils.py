"""
colab_utils.py
==============
Google Colab and Google Drive integration utilities.

This module provides functions for mounting Google Drive, constructing
persistent storage paths, and saving/loading training artefacts in a
reproducible manner across Colab sessions.

Notes
-----
All path construction assumes the following Drive layout::

    /content/drive/MyDrive/
    └── multimodal-ternary-llm/
        ├── checkpoints/
        ├── logs/
        └── metrics/
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DRIVE_ROOT: str = "/content/drive/MyDrive/multimodal-ternary-llm"
CHECKPOINT_DIR: str = os.path.join(DRIVE_ROOT, "checkpoints")
LOG_DIR: str = os.path.join(DRIVE_ROOT, "logs")
METRICS_DIR: str = os.path.join(DRIVE_ROOT, "metrics")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Fix all sources of randomness to ensure reproducible experiments.

    Parameters
    ----------
    seed : int, optional
        Integer seed value. Default is 42.

    Notes
    -----
    Sets seeds for Python's built-in ``random`` module, NumPy, and PyTorch
    (both CPU and CUDA backends). Additionally configures cuDNN for
    deterministic operation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device() -> torch.device:
    """
    Resolve the optimal available compute device.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Resolved compute device: %s.", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s | VRAM: %.2f GiB.",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        )
    return device


# ---------------------------------------------------------------------------
# Drive mounting
# ---------------------------------------------------------------------------

def mount_drive(mount_path: str = "/content/drive") -> None:
    """
    Mount Google Drive within the Colab runtime environment.

    Parameters
    ----------
    mount_path : str, optional
        Filesystem path at which to mount Google Drive.
        Default is ``/content/drive``.

    Notes
    -----
    This function is a no-op when executed outside of a Google Colab
    environment, allowing notebooks to be tested locally.
    """
    try:
        from google.colab import drive  # type: ignore[import]
        drive.mount(mount_path)
        logger.info("Google Drive mounted at %s.", mount_path)
    except ImportError:
        logger.warning(
            "google.colab not available. Skipping Drive mount "
            "(assumed local execution)."
        )


def ensure_drive_dirs() -> None:
    """
    Create the standard project directory tree on Google Drive if absent.
    """
    for directory in (CHECKPOINT_DIR, LOG_DIR, METRICS_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory: %s.", directory)


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    filename: str,
    subdir: str = "",
) -> Path:
    """
    Persist a training checkpoint to Google Drive.

    Parameters
    ----------
    state : dict
        Checkpoint dictionary (model state dict, optimiser state, epoch, etc.).
    filename : str
        Name of the output file, e.g. ``"phase1_epoch05.pt"``.
    subdir : str, optional
        Optional subdirectory within ``CHECKPOINT_DIR``.

    Returns
    -------
    Path
        Absolute path to the saved checkpoint file on Drive.
    """
    target_dir = Path(CHECKPOINT_DIR) / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    torch.save(state, target_path)
    logger.info("Checkpoint saved to %s.", target_path)
    return target_path


def load_checkpoint(
    filename: str,
    subdir: str = "",
    map_location: Optional[torch.device] = None,
) -> dict:
    """
    Load a previously persisted checkpoint from Google Drive.

    Parameters
    ----------
    filename : str
        Name of the checkpoint file.
    subdir : str, optional
        Optional subdirectory within ``CHECKPOINT_DIR``.
    map_location : torch.device, optional
        Device mapping for tensor loading.

    Returns
    -------
    dict
        Deserialised checkpoint dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified checkpoint file does not exist on Drive.
    """
    target_path = Path(CHECKPOINT_DIR) / subdir / filename
    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {target_path}")
    state = torch.load(target_path, map_location=map_location)
    logger.info("Checkpoint loaded from %s.", target_path)
    return state


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger for structured output.

    Parameters
    ----------
    level : int, optional
        Logging verbosity level. Default is ``logging.INFO``.
    log_file : str, optional
        If provided, log output is mirrored to this file on Drive.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(LOG_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logger.info("Logging initialised at level %s.", logging.getLevelName(level))
