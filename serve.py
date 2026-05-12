"""
Minimal Flask server for SAM 3 image inference.

Start with:
    uv run python serve.py
"""

import io
import json
import sys

import torch
from PIL import Image
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.train.masks_ops import rle_encode

print("Loading SAM 3 model...", flush=True)
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model loaded.", flush=True)

from flask import Flask, jsonify, request

app = Flask(__name__)


def _format_predictions(inference_state, orig_w, orig_h, score_threshold):
    boxes = inference_state["boxes"]
    scores = inference_state["scores"]
    masks = inference_state["masks"]

    boxes_norm = torch.stack(
        [
            boxes[:, 0] / orig_w,
            boxes[:, 1] / orig_h,
            boxes[:, 2] / orig_w,
            boxes[:, 3] / orig_h,
        ],
        dim=-1,
    )
    boxes_xywh = box_xyxy_to_xywh(boxes_norm).tolist()

    rle_masks = rle_encode(masks.squeeze(1))
    rle_masks = [m["counts"] for m in rle_masks]

    scores_list = scores.tolist()

    return sorted(
        [
            {"box": b, "mask_rle": m, "score": s}
            for b, m, s in zip(boxes_xywh, rle_masks, scores_list)
            if s >= score_threshold
        ],
        key=lambda r: r["score"],
        reverse=True,
    )


def _parse_components(raw_components):
    if not raw_components:
        raise ValueError("missing 'components' field")

    try:
        components = json.loads(raw_components)
    except json.JSONDecodeError as exc:
        raise ValueError("'components' must be valid JSON") from exc

    if not isinstance(components, list) or not components:
        raise ValueError("'components' must be a non-empty JSON array")

    parsed = []
    for index, component in enumerate(components):
        if isinstance(component, str):
            label = component.strip()
            prompt = label
        elif isinstance(component, dict):
            component_data = {str(key): value for key, value in component.items()}
            label = str(component_data.get("label", "")).strip()
            prompt = str(component_data.get("prompt", label)).strip()
        else:
            raise ValueError(
                f"component {index} must be a string or an object with label/prompt"
            )

        if not label:
            raise ValueError(f"component {index} is missing a label")
        if not prompt:
            raise ValueError(f"component {index} is missing a prompt")

        parsed.append({"label": label, "prompt": prompt})

    return parsed


@app.post("/predict")
def predict():
    """Run SAM 3 segmentation on an uploaded image with a text prompt."""
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file"}), 400
    prompt = request.form.get("prompt")
    if not prompt:
        return jsonify({"error": "missing 'prompt' field"}), 400
    score_threshold = float(request.form.get("score_threshold", 0.0))

    pil_image = Image.open(request.files["image"].stream).convert("RGB")
    orig_w, orig_h = pil_image.size

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(pil_image)
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=prompt
        )

    results = _format_predictions(
        inference_state=inference_state,
        orig_w=orig_w,
        orig_h=orig_h,
        score_threshold=score_threshold,
    )

    return jsonify(
        {
            "image_width": orig_w,
            "image_height": orig_h,
            "prompt": prompt,
            "num_predictions": len(results),
            "predictions": results,
        }
    )


@app.post("/predict_multi")
def predict_multi():
    """Run SAM 3 once per labeled component and return labeled masks."""
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file"}), 400

    try:
        components = _parse_components(request.form.get("components"))
        score_threshold = float(request.form.get("score_threshold", 0.0))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    pil_image = Image.open(request.files["image"].stream).convert("RGB")
    orig_w, orig_h = pil_image.size

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(pil_image)
        segment_groups = []
        segments = []

        for component in components:
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=component["prompt"]
            )
            predictions = _format_predictions(
                inference_state=inference_state,
                orig_w=orig_w,
                orig_h=orig_h,
                score_threshold=score_threshold,
            )

            labeled_predictions = [
                {
                    "label": component["label"],
                    "prompt": component["prompt"],
                    "instance_id": f"{component['label']}_{index}",
                    **prediction,
                }
                for index, prediction in enumerate(predictions)
            ]
            segments.extend(labeled_predictions)
            segment_groups.append(
                {
                    "label": component["label"],
                    "prompt": component["prompt"],
                    "num_predictions": len(labeled_predictions),
                    "predictions": labeled_predictions,
                }
            )

    return jsonify(
        {
            "image_width": orig_w,
            "image_height": orig_h,
            "components": components,
            "num_segments": len(segments),
            "segments": segments,
            "segment_groups": segment_groups,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=False)
