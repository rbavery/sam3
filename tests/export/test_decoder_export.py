from __future__ import annotations

import pytest
import torch

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from tests.export.utils import capture_stderr_on_fail, get_device


class FullInferenceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        img_ids: torch.Tensor,
        text_ids: torch.Tensor,
        box_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
    ):
        model = self.model  # type: ignore[assignment]
        backbone_out = model.backbone.forward_image(images)
        text_encoder = model.backbone.language_backbone
        _, text_tokens = text_encoder.encoder(token_ids)
        text_tokens = text_tokens.transpose(0, 1)
        text_memory = text_encoder.resizer(text_tokens)
        text_attention_mask = token_ids.ne(0)
        text_attention_mask = text_attention_mask.ne(1)
        backbone_out["language_features"] = text_memory
        backbone_out["language_mask"] = text_attention_mask

        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=box_embeddings,
            input_boxes_mask=box_mask,
            input_boxes_label=box_labels,
            input_points=torch.zeros(
                0, int(token_ids.shape[0]), 2, device=images.device
            ),
            input_points_mask=torch.zeros(
                int(token_ids.shape[0]), 0, device=images.device, dtype=torch.bool
            ),
        )

        geometric_prompt = Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
        )

        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )

        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["pred_masks"],
            out.get("presence_logit_dec"),
        )


def _make_inputs(batch: int, height: int, width: int, device: str, num_boxes: int = 0):
    images = torch.randn(batch, 3, height, width, device=device)
    token_ids = torch.ones(batch, 32, device=device, dtype=torch.long)
    token_ids[:, -1] = 0
    img_ids = torch.arange(batch, device=device, dtype=torch.long)
    text_ids = torch.zeros(batch, device=device, dtype=torch.long)
    box_embeddings = torch.rand(num_boxes, batch, 4, device=device)
    if num_boxes > 0:
        box_embeddings[..., 2:] = box_embeddings[..., 2:] * 0.5
    box_mask = torch.zeros(batch, num_boxes, device=device, dtype=torch.bool)
    box_labels = torch.zeros(num_boxes, batch, device=device, dtype=torch.long)
    return images, token_ids, img_ids, text_ids, box_embeddings, box_mask, box_labels


def _export_decoder(model: torch.nn.Module, inputs):
    (
        images,
        token_ids,
        img_ids,
        text_ids,
        box_embeddings,
        box_mask,
        box_labels,
    ) = inputs
    device = images.device
    wrapper = FullInferenceWrapper(model).to(device).eval()  # type: ignore[arg-type]
    if images.shape[0] == 1:
        images = images.repeat(2, 1, 1, 1)
        token_ids = token_ids.repeat(2, 1)
        img_ids = img_ids.repeat(2)
        text_ids = text_ids.repeat(2)
        box_embeddings = box_embeddings.repeat(1, 2, 1)
        box_mask = box_mask.repeat(2, 1)
        box_labels = box_labels.repeat(1, 2)
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (
                images,
                token_ids,
                img_ids,
                text_ids,
                box_embeddings,
                box_mask,
                box_labels,
            ),
            dynamic_shapes={
                "images": {
                    0: torch.export.Dim.AUTO,
                    2: 1008,
                    3: 1008,
                },
                "token_ids": {
                    0: torch.export.Dim.AUTO,
                    1: 32,
                },
                "img_ids": {0: torch.export.Dim.AUTO},
                "text_ids": {0: torch.export.Dim.AUTO},
                "box_embeddings": {
                    0: 1,
                    1: torch.export.Dim.AUTO,
                },
                "box_mask": {
                    0: torch.export.Dim.AUTO,
                    1: 1,
                },
                "box_labels": {
                    0: 1,
                    1: torch.export.Dim.AUTO,
                },
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    return exported


def test_decoder_export_static(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    with capture_stderr_on_fail("export_static"):
        exported = _export_decoder(sam3_model, inputs)
    assert exported is not None


def test_decoder_export_loads(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    with capture_stderr_on_fail("export_loads"):
        exported = _export_decoder(sam3_model, inputs)
    module = exported.module()
    with torch.no_grad():
        out = module(*inputs)
    assert isinstance(out, tuple)
    assert len(out) == 4


def test_decoder_export_matches_eager(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    wrapper = FullInferenceWrapper(sam3_model).to(device).eval()
    with torch.no_grad():
        eager_out = wrapper(*inputs)
    with capture_stderr_on_fail("export_match"):
        exported = _export_decoder(sam3_model, inputs)
    module = exported.module()
    with torch.no_grad():
        export_out = module(*inputs)
    for eager, compiled in zip(eager_out, export_out):
        if eager is None:
            assert compiled is None
        else:
            torch.testing.assert_close(eager, compiled, rtol=1e-3, atol=1e-3)


def test_decoder_export_dynamic_batch(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    with capture_stderr_on_fail("export_dynamic_batch"):
        exported = _export_decoder(sam3_model, inputs)
    module = exported.module()
    new_inputs = _make_inputs(2, 1008, 1008, device, num_boxes=1)
    with torch.no_grad():
        out = module(*new_inputs)
    assert isinstance(out, tuple)


def test_decoder_export_dynamic_spatial(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    with capture_stderr_on_fail("export_dynamic_spatial"):
        exported = _export_decoder(sam3_model, inputs)
    module = exported.module()
    new_inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    with torch.no_grad():
        out = module(*new_inputs)
    assert isinstance(out, tuple)


def test_decoder_export_full_dynamic(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    with capture_stderr_on_fail("export_full_dynamic"):
        exported = _export_decoder(sam3_model, inputs)
    module = exported.module()
    new_inputs = _make_inputs(2, 1008, 1008, device, num_boxes=1)
    with torch.no_grad():
        out = module(*new_inputs)
    assert isinstance(out, tuple)


@pytest.mark.parametrize("batch", [1, 2])
def test_decoder_export_inference_shapes(sam3_model, batch: int):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device, num_boxes=1)
    with capture_stderr_on_fail("export_inference_shapes"):
        exported = _export_decoder(sam3_model, inputs)
    module = exported.module()
    new_inputs = _make_inputs(batch, 1008, 1008, device, num_boxes=1)
    with torch.no_grad():
        out = module(*new_inputs)
    assert isinstance(out, tuple)
