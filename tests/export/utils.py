from __future__ import annotations

import io
import os
import re
from contextlib import contextmanager, redirect_stderr
from typing import Generator
from pathlib import Path

import torch

LOG_DIR = Path(__file__).resolve().parent / "export_logs"


def _sanitize_test_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _current_test_name() -> str:
    raw = os.getenv("PYTEST_CURRENT_TEST", "unknown")
    return _sanitize_test_name(raw.split(" ")[0])


@contextmanager
def capture_stderr_on_fail(suffix: str) -> Generator[None, None, None]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        try:
            yield
        except Exception:
            log_path = LOG_DIR / f"{_current_test_name()}-{suffix}.stderr.txt"
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write(buffer.getvalue())
            raise


def get_device() -> str:
    force_cpu = os.getenv("SAM3_EXPORT_FORCE_CPU", "0") == "1"
    device = os.getenv("SAM3_EXPORT_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    return device
