"""
utils — Shared utility modules for the Hybrid Multimodal BitNet Architecture.

This package provides reusable components across all pipeline phases:
  - bit_layers:   BitLinear layer and ternary quantisation utilities.
  - connector:    Two-layer GeLU MLP connector between encoder and BitNet core.
  - encoders:     Perception modules (visual ViT, text, Mamba time-series).
  - losses:       Composite training loss functions (language, distillation,
                  alignment, stability regularisation).
  - heads:        Task-specific output heads (text, classification, regression, action).
  - training:     Training loop utilities, QAT helpers, optimiser state management.
  - evaluation:   Downstream evaluation and benchmark utilities.
  - colab_utils:  Google Colab / Drive persistence helpers.
"""
