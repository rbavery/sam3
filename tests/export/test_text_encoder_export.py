from __future__ import annotations

import pytest
import torch

from sam3.model.text_encoder_ve import VETextEncoder
from tests.export.utils import capture_stderr_on_fail, get_device


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder: VETextEncoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, token_ids: torch.Tensor):
        _, text_tokens = self.text_encoder.encoder(token_ids)
        text_tokens = text_tokens.transpose(0, 1)
        text_memory = self.text_encoder.resizer(text_tokens)
        text_attention_mask = token_ids.ne(0)
        text_attention_mask = text_attention_mask.ne(1)
        return text_attention_mask, text_memory


def _make_tokens(batch: int, seq_len: int, vocab_size: int, device: str) -> torch.Tensor:
    token_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    token_ids[:, -1] = 1
    return token_ids


def _export_text_encoder(model: torch.nn.Module, token_ids: torch.Tensor):
    device = token_ids.device
    wrapper = TextEncoderWrapper(model.backbone.language_backbone).to(device).eval()  # type: ignore[arg-type]
    export_tokens = token_ids
    if token_ids.shape[0] == 1:
        export_tokens = token_ids.repeat(2, 1)
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (export_tokens,),
            dynamic_shapes={
                "token_ids": {
                    0: torch.export.Dim("batch", min=1, max=4),
                    1: torch.export.Dim.AUTO,
                }
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    return exported


def test_text_encoder_export_static(sam3_model):
    device = get_device()
    vocab_size = sam3_model.backbone.language_backbone.encoder.vocab_size
    token_ids = _make_tokens(1, 32, vocab_size, device)
    with capture_stderr_on_fail("export_static"):
        exported = _export_text_encoder(sam3_model, token_ids)
    assert exported is not None


def test_text_encoder_export_loads(sam3_model):
    device = get_device()
    vocab_size = sam3_model.backbone.language_backbone.encoder.vocab_size
    token_ids = _make_tokens(1, 32, vocab_size, device)
    with capture_stderr_on_fail("export_loads"):
        exported = _export_text_encoder(sam3_model, token_ids)
    module = exported.module()
    with torch.no_grad():
        out = module(token_ids)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_text_encoder_export_matches_eager(sam3_model):
    device = get_device()
    vocab_size = sam3_model.backbone.language_backbone.encoder.vocab_size
    token_ids = _make_tokens(1, 32, vocab_size, device)
    wrapper = TextEncoderWrapper(sam3_model.backbone.language_backbone).to(device).eval()  # type: ignore[arg-type]
    with torch.no_grad():
        eager_out = wrapper(token_ids)
    with capture_stderr_on_fail("export_match"):
        exported = _export_text_encoder(sam3_model, token_ids)
    module = exported.module()
    with torch.no_grad():
        export_out = module(token_ids)
    torch.testing.assert_close(eager_out[0], export_out[0])
    torch.testing.assert_close(eager_out[1], export_out[1], rtol=1e-3, atol=1e-3)


def test_text_encoder_export_dynamic_batch(sam3_model):
    device = get_device()
    vocab_size = sam3_model.backbone.language_backbone.encoder.vocab_size
    token_ids = _make_tokens(1, 32, vocab_size, device)
    with capture_stderr_on_fail("export_dynamic_batch"):
        exported = _export_text_encoder(sam3_model, token_ids)
    module = exported.module()
    with torch.no_grad():
        out = module(_make_tokens(2, 32, vocab_size, device))
    assert isinstance(out, tuple)


@pytest.mark.parametrize("seq_len", [32])
def test_text_encoder_export_dynamic_seq_len(sam3_model, seq_len: int):
    device = get_device()
    vocab_size = sam3_model.backbone.language_backbone.encoder.vocab_size
    token_ids = _make_tokens(1, 32, vocab_size, device)
    with capture_stderr_on_fail("export_dynamic_seq_len"):
        exported = _export_text_encoder(sam3_model, token_ids)
    module = exported.module()
    with torch.no_grad():
        out = module(_make_tokens(1, seq_len, vocab_size, device))
    assert isinstance(out, tuple)


@pytest.mark.parametrize("batch,seq_len", [(1, 32), (2, 32)])
def test_text_encoder_export_inference_shapes(
    sam3_model, batch: int, seq_len: int
):
    device = get_device()
    vocab_size = sam3_model.backbone.language_backbone.encoder.vocab_size
    token_ids = _make_tokens(1, 32, vocab_size, device)
    with capture_stderr_on_fail("export_inference_shapes"):
        exported = _export_text_encoder(sam3_model, token_ids)
    module = exported.module()
    with torch.no_grad():
        out = module(_make_tokens(batch, seq_len, vocab_size, device))
    assert isinstance(out, tuple)
