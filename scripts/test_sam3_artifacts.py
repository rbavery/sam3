import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    np_image = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _prepare_image(image: torch.Tensor, size: int) -> torch.Tensor:
    image = image.clamp(0, 1)
    image = torch.nn.functional.interpolate(
        image, size=(size, size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.5, 0.5, 0.5], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def _make_inputs(model, image: torch.Tensor, prompts):
    device = image.device
    num_prompts = len(prompts)

    tokenizer = model.backbone.language_backbone.tokenizer
    token_ids = tokenizer(prompts, context_length=32).to(device)

    img_ids = torch.zeros(num_prompts, device=device, dtype=torch.long)
    text_ids = torch.zeros(num_prompts, device=device, dtype=torch.long)

    box_embeddings = torch.zeros(1, num_prompts, 4, device=device)
    box_mask = torch.zeros(num_prompts, 1, device=device, dtype=torch.bool)
    box_labels = torch.zeros(1, num_prompts, device=device, dtype=torch.long)

    return (
        image,
        token_ids,
        img_ids,
        text_ids,
        box_embeddings,
        box_mask,
        box_labels,
    )


def _run_full_model(model, inputs):
    (
        images,
        token_ids,
        img_ids,
        text_ids,
        box_embeddings,
        box_mask,
        box_labels,
    ) = inputs
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
        input_points=torch.zeros(0, int(token_ids.shape[0]), 2, device=images.device),
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
        out["pred_masks"],
        out["pred_boxes"],
        out["pred_logits"],
        out["pred_boxes_xyxy"],
    )


def _load_export(path: Path):
    exported = torch.export.load(str(path))
    return exported.module()


def _to_pil_image(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    np_image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(np_image)


def _color_palette(num_colors: int):
    base = [
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (238, 130, 238),
        (255, 215, 0),
        (255, 165, 0),
    ]
    return [base[i % len(base)] for i in range(num_colors)]


def _overlay_masks(
    image: Image.Image, masks: torch.Tensor, scores: torch.Tensor, out_path: Path
):
    num_prompts, num_queries = scores.shape[:2]
    best_idx = scores.squeeze(-1).argmax(dim=1)
    colors = _color_palette(num_prompts)
    base = image.copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for i in range(num_prompts):
        mask = masks[i, best_idx[i]].detach().cpu()
        mask = mask > 0
        mask_img = Image.fromarray((mask.numpy() * 255).astype(np.uint8), mode="L")
        color = colors[i]
        color_img = Image.new("RGBA", base.size, (*color, 120))
        overlay = Image.composite(color_img, overlay, mask_img)
    blended = Image.alpha_composite(base.convert("RGBA"), overlay)
    blended.convert("RGB").save(out_path)


def _draw_boxes(
    image: Image.Image, boxes_xyxy: torch.Tensor, scores: torch.Tensor, out_path: Path
):
    num_prompts, num_queries = scores.shape[:2]
    best_idx = scores.squeeze(-1).argmax(dim=1)
    colors = _color_palette(num_prompts)
    draw = ImageDraw.Draw(image)
    for i in range(num_prompts):
        box = boxes_xyxy[i, best_idx[i]].detach().cpu().tolist()
        color = colors[i]
        draw.rectangle(box, outline=color, width=3)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("assets/images/cat_dog.jpg"),
        help="Path to input image",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="cat,dog",
        help="Comma-separated text prompts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/export"),
        help="Directory with exported artifacts",
    )
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise ValueError("Provide at least one prompt")

    model = build_sam3_image_model(
        device=args.device, eval_mode=True, enable_segmentation=True
    )
    model.eval()

    image = _load_image(args.image, torch.device(args.device))
    image = _prepare_image(image, size=1008)
    inputs = _make_inputs(model, image, prompts)

    with torch.no_grad():
        eager_masks, eager_boxes, eager_logits, eager_boxes_xyxy = _run_full_model(
            model, inputs
        )

    image_module = _load_export(args.artifact_dir / "image_encoder.pt2")
    text_module = _load_export(args.artifact_dir / "text_encoder.pt2")
    encoder_module = _load_export(args.artifact_dir / "encoder_fusion.pt2")
    decoder_module = _load_export(args.artifact_dir / "decoder.pt2")

    with torch.no_grad():
        vision_features, vision_pos_enc, _ = image_module(inputs[0])
        text_attention_mask, text_memory = text_module(inputs[1])
        img_mask = [
            torch.zeros(
                feat.shape[0],
                feat.shape[2],
                feat.shape[3],
                device=feat.device,
                dtype=torch.bool,
            )
            for feat in vision_features
        ]
        enc_out = encoder_module(
            vision_features, vision_pos_enc, img_mask, text_memory, text_attention_mask
        )
        assert isinstance(enc_out, tuple)
        pred_logits, pred_boxes, pred_masks, pred_boxes_xyxy = decoder_module(*inputs)

    print("Prompt count:", len(prompts))
    print("Pred logits shape:", pred_logits.shape)
    print("Pred boxes shape:", pred_boxes.shape)
    print("Pred masks shape:", pred_masks.shape)
    torch.testing.assert_close(pred_logits, eager_logits, rtol=1e-3, atol=1e-3)
    print("Eager vs export logits match")

    out_dir = args.artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_image = _to_pil_image(inputs[0][0])
    _overlay_masks(
        base_image.copy(),
        eager_masks,
        eager_logits,
        out_dir / "eager_masks_overlay.jpg",
    )
    _overlay_masks(
        base_image.copy(),
        pred_masks,
        pred_logits,
        out_dir / "export_masks_overlay.jpg",
    )
    _draw_boxes(
        base_image.copy(),
        eager_boxes_xyxy,
        eager_logits,
        out_dir / "eager_boxes_overlay.jpg",
    )
    _draw_boxes(
        base_image.copy(),
        pred_boxes_xyxy,
        pred_logits,
        out_dir / "export_boxes_overlay.jpg",
    )
    print("Saved overlays to", out_dir)


if __name__ == "__main__":
    main()
