"""JEPA-Cortex experiment package.

A small-LM-that-talks-and-reasons via byte-level Mamba-3 + thought-level
distillation from Qwen-2.5-1.5B + isotropy regularizer + residual primitives.

Self-contained: local copies of cortex_counting.py and mamba3_minimal.py
live alongside the new modules so the existing top-level experiments are
not affected by changes here. Run scripts as `uv run python jepa/<name>.py`.
"""
