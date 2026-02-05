import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model
from tests.export.test_decoder_export import _export_decoder
from tests.export.test_encoder_export import EncoderFusionWrapper
from tests.export.test_image_encoder_export import _export_image_encoder
from tests.export.test_text_encoder_export import _export_text_encoder


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


def _save_export(exported, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(path))


def _export_encoder_multilevel(
    model, img_feats, img_pos, img_mask, prompt, prompt_mask
):
    device = prompt.device
    wrapper = EncoderFusionWrapper(model.transformer.encoder).to(device).eval()
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (img_feats, img_pos, img_mask, prompt, prompt_mask),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    return exported


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
        "--out-dir",
        type=Path,
        default=Path("artifacts/export"),
        help="Directory to write exported artifacts",
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

    print("Exporting image encoder...")
    image_encoder = _export_image_encoder(model, inputs[0])
    print("Exporting text encoder...")
    text_encoder = _export_text_encoder(model, inputs[1])
    print("Exporting encoder fusion...")
    with torch.no_grad():
        image_module = image_encoder.module()
        text_module = text_encoder.module()
        vision_features, vision_pos_enc, _ = image_module(inputs[0])
        text_attention_mask, text_memory = text_module(inputs[1])
        prompt = text_memory
        prompt_mask = text_attention_mask
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
    encoder = _export_encoder_multilevel(
        model, vision_features, vision_pos_enc, img_mask, prompt, prompt_mask
    )
    print("Exporting decoder...")
    decoder = _export_decoder(model, inputs)

    _save_export(image_encoder, args.out_dir / "image_encoder.pt2")
    _save_export(text_encoder, args.out_dir / "text_encoder.pt2")
    _save_export(encoder, args.out_dir / "encoder_fusion.pt2")
    _save_export(decoder, args.out_dir / "decoder.pt2")
    print("Saved exports to", args.out_dir)


if __name__ == "__main__":
    main()
