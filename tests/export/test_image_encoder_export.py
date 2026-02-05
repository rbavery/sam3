from __future__ import annotations

import pytest
import torch

from sam3.model.vl_combiner import SAM3VLBackbone
from tests.export.utils import capture_stderr_on_fail, get_device


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, backbone: SAM3VLBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor):
        out = self.backbone._forward_image_no_act_ckpt(images)
        return (
            out["vision_features"],
            out["vision_pos_enc"],
            out["backbone_fpn"],
        )


def _make_images(batch: int, height: int, width: int, device: str) -> torch.Tensor:
    return torch.randn(batch, 3, height, width, device=device, dtype=torch.float32)


def _export_image_encoder(model: torch.nn.Module, images: torch.Tensor):
    device = images.device
    wrapper = ImageEncoderWrapper(model.backbone).to(device).eval()  # type: ignore[arg-type]
    export_images = images
    if images.shape[0] == 1:
        export_images = images.repeat(2, 1, 1, 1)
    with torch.no_grad():
        height = torch.export.Dim("height", min=3, max=4)
        width = torch.export.Dim("width", min=3, max=4)
        exported = torch.export.export(
            wrapper,
            (export_images,),
            dynamic_shapes={
                "images": {
                    0: torch.export.Dim("batch", min=1, max=4),
                    2: 336 * height,
                    3: 336 * width,
                }
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    return exported


def test_image_encoder_export_static(sam3_model):
    device = get_device()
    images = _make_images(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_static"):
        exported = _export_image_encoder(sam3_model, images)
    assert exported is not None


def test_image_encoder_export_loads(sam3_model):
    device = get_device()
    images = _make_images(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_loads"):
        exported = _export_image_encoder(sam3_model, images)
    module = exported.module()
    with torch.no_grad():
        out = module(images)
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_image_encoder_export_matches_eager(sam3_model):
    device = get_device()
    images = _make_images(1, 1008, 1008, device)
    wrapper = ImageEncoderWrapper(sam3_model.backbone).to(device).eval()
    with torch.no_grad():
        eager_out = wrapper(images)
    with capture_stderr_on_fail("export_match"):
        exported = _export_image_encoder(sam3_model, images)
    module = exported.module()
    with torch.no_grad():
        export_out = module(images)
    for eager, compiled in zip(eager_out, export_out):
        if isinstance(eager, (list, tuple)):
            for e_item, c_item in zip(eager, compiled):
                torch.testing.assert_close(e_item, c_item, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(eager, compiled, rtol=1e-3, atol=1e-3)


def test_image_encoder_export_dynamic_batch(sam3_model):
    device = get_device()
    images = _make_images(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_dynamic_batch"):
        exported = _export_image_encoder(sam3_model, images)
    module = exported.module()
    with torch.no_grad():
        out = module(_make_images(2, 1008, 1008, device))
    assert isinstance(out, tuple)


@pytest.mark.parametrize("height,width", [(1008, 1008)])
def test_image_encoder_export_dynamic_spatial(sam3_model, height: int, width: int):
    device = get_device()
    images = _make_images(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_dynamic_spatial"):
        exported = _export_image_encoder(sam3_model, images)
    module = exported.module()
    with torch.no_grad():
        out = module(_make_images(1, height, width, device))
    assert isinstance(out, tuple)


def test_image_encoder_export_full_dynamic(sam3_model):
    device = get_device()
    images = _make_images(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_full_dynamic"):
        exported = _export_image_encoder(sam3_model, images)
    module = exported.module()
    with torch.no_grad():
        out = module(_make_images(3, 1008, 1008, device))
    assert isinstance(out, tuple)


@pytest.mark.parametrize("batch,height,width", [(2, 1008, 1008)])
def test_image_encoder_export_inference_shapes(
    sam3_model, batch: int, height: int, width: int
):
    device = get_device()
    images = _make_images(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_inference_shapes"):
        exported = _export_image_encoder(sam3_model, images)
    module = exported.module()
    with torch.no_grad():
        out = module(_make_images(batch, height, width, device))
    assert isinstance(out, tuple)
