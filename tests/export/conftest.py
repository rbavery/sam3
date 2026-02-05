from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model


@pytest.fixture(scope="session")
def sam3_model() -> torch.nn.Module:
    force_cpu = os.getenv("SAM3_EXPORT_FORCE_CPU", "0") == "1"
    device = os.getenv("SAM3_EXPORT_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    try:
        model = build_sam3_image_model(
            device=device, eval_mode=True, enable_segmentation=True
        )
    except torch.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        pytest.skip("CUDA OOM while loading SAM3 model; free GPU memory or set SAM3_EXPORT_FORCE_CPU=1")
    model.eval()
    return model
