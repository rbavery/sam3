from __future__ import annotations

import pytest
import torch

from sam3.model.encoder import TransformerEncoderFusion
from tests.export.utils import capture_stderr_on_fail, get_device


class EncoderFusionWrapper(torch.nn.Module):
    def __init__(self, encoder: TransformerEncoderFusion):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        img_feats: torch.Tensor,
        img_pos: torch.Tensor,
        img_mask: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        out = self.encoder(
            src=[img_feats],
            src_pos=[img_pos],
            src_key_padding_mask=[img_mask],
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
        )
        return (
            out["memory"],
            out["pos_embed"],
            out["padding_mask"],
            out["level_start_index"],
            out["spatial_shapes"],
            out["valid_ratios"],
        )


def _make_image_tokens(batch: int, height: int, width: int, device: str):
    channels = 256
    img_feats = torch.randn(batch, channels, height, width, device=device)
    img_pos = torch.randn(batch, channels, height, width, device=device)
    img_mask = torch.zeros(batch, height, width, dtype=torch.bool, device=device)
    return img_feats, img_pos, img_mask


def _make_prompt(batch: int, seq_len: int, device: str):
    prompt = torch.randn(seq_len, batch, 256, device=device)
    prompt_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    return prompt, prompt_mask


def _export_encoder(model: torch.nn.Module, img_feats, img_pos, prompt, prompt_mask):
    device = img_feats.device
    wrapper = EncoderFusionWrapper(model.transformer.encoder).to(device).eval()  # type: ignore[arg-type]
    if img_feats.shape[0] == 1:
        img_feats = img_feats.repeat(2, 1, 1, 1)
        img_pos = img_pos.repeat(2, 1, 1, 1)
        img_mask = torch.zeros(
            2, img_feats.shape[2], img_feats.shape[3], dtype=torch.bool, device=device
        )
        prompt = prompt.repeat(1, 2, 1)
        prompt_mask = prompt_mask.repeat(2, 1)
    else:
        img_mask = torch.zeros(
            img_feats.shape[0],
            img_feats.shape[2],
            img_feats.shape[3],
            dtype=torch.bool,
            device=device,
        )
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (img_feats, img_pos, img_mask, prompt, prompt_mask),
            dynamic_shapes={
                "img_feats": {
                    0: torch.export.Dim.AUTO,
                },
                "img_pos": {
                    0: torch.export.Dim.AUTO,
                },
                "img_mask": {
                    0: torch.export.Dim.AUTO,
                },
                "prompt": {
                    0: torch.export.Dim("seq", min=1, max=64),
                    1: torch.export.Dim.AUTO,
                },
                "prompt_mask": {
                    0: torch.export.Dim.AUTO,
                    1: torch.export.Dim("seq", min=1, max=64),
                },
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    return exported


def test_encoder_export_static(sam3_model):
    device = get_device()
    img_feats, img_pos, img_mask = _make_image_tokens(1, 72, 72, device)
    prompt, prompt_mask = _make_prompt(1, 4, device)
    with capture_stderr_on_fail("export_static"):
        exported = _export_encoder(sam3_model, img_feats, img_pos, prompt, prompt_mask)
    assert exported is not None


def test_encoder_export_loads(sam3_model):
    device = get_device()
    img_feats, img_pos, img_mask = _make_image_tokens(1, 72, 72, device)
    prompt, prompt_mask = _make_prompt(1, 4, device)
    with capture_stderr_on_fail("export_loads"):
        exported = _export_encoder(sam3_model, img_feats, img_pos, prompt, prompt_mask)
    module = exported.module()
    with torch.no_grad():
        out = module(img_feats, img_pos, img_mask, prompt, prompt_mask)
    assert isinstance(out, tuple)
    assert len(out) == 6


def test_encoder_export_matches_eager(sam3_model):
    device = get_device()
    img_feats, img_pos, img_mask = _make_image_tokens(1, 72, 72, device)
    prompt, prompt_mask = _make_prompt(1, 4, device)
    wrapper = EncoderFusionWrapper(sam3_model.transformer.encoder).to(device).eval()  # type: ignore[arg-type]
    with torch.no_grad():
        eager_out = wrapper(img_feats, img_pos, img_mask, prompt, prompt_mask)
    with capture_stderr_on_fail("export_match"):
        exported = _export_encoder(sam3_model, img_feats, img_pos, prompt, prompt_mask)
    module = exported.module()
    with torch.no_grad():
        export_out = module(img_feats, img_pos, img_mask, prompt, prompt_mask)
    for eager, compiled in zip(eager_out, export_out):
        torch.testing.assert_close(eager, compiled, rtol=1e-3, atol=1e-3)


def test_encoder_export_dynamic_batch(sam3_model):
    device = get_device()
    img_feats, img_pos, img_mask = _make_image_tokens(1, 72, 72, device)
    prompt, prompt_mask = _make_prompt(1, 4, device)
    with capture_stderr_on_fail("export_dynamic_batch"):
        exported = _export_encoder(sam3_model, img_feats, img_pos, prompt, prompt_mask)
    module = exported.module()
    img_feats2, img_pos2, img_mask2 = _make_image_tokens(2, 72, 72, device)
    prompt2, prompt_mask2 = _make_prompt(2, 4, device)
    with torch.no_grad():
        out = module(img_feats2, img_pos2, img_mask2, prompt2, prompt_mask2)
    assert isinstance(out, tuple)


def test_encoder_export_dynamic_spatial(sam3_model):
    device = get_device()
    img_feats, img_pos, img_mask = _make_image_tokens(1, 72, 72, device)
    prompt, prompt_mask = _make_prompt(1, 4, device)
    with capture_stderr_on_fail("export_dynamic_spatial"):
        exported = _export_encoder(sam3_model, img_feats, img_pos, prompt, prompt_mask)
    module = exported.module()
    img_feats2, img_pos2, img_mask2 = _make_image_tokens(1, 72, 72, device)
    with torch.no_grad():
        out = module(img_feats2, img_pos2, img_mask2, prompt, prompt_mask)
    assert isinstance(out, tuple)


@pytest.mark.parametrize("batch,seq_len", [(1, 4), (2, 8)])
def test_encoder_export_inference_shapes(sam3_model, batch: int, seq_len: int):
    device = get_device()
    img_feats, img_pos, img_mask = _make_image_tokens(1, 72, 72, device)
    prompt, prompt_mask = _make_prompt(1, 4, device)
    with capture_stderr_on_fail("export_inference_shapes"):
        exported = _export_encoder(sam3_model, img_feats, img_pos, prompt, prompt_mask)
    module = exported.module()
    img_feats2, img_pos2, img_mask2 = _make_image_tokens(batch, 72, 72, device)
    prompt2, prompt_mask2 = _make_prompt(batch, seq_len, device)
    with torch.no_grad():
        out = module(img_feats2, img_pos2, img_mask2, prompt2, prompt_mask2)
    assert isinstance(out, tuple)
