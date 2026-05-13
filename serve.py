"""
Minimal Flask server for SAM 3 image inference.

Start with:
    uv run python serve.py
"""

import json

import torch
from flask import Flask, jsonify, request
from PIL import Image
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.train.masks_ops import rle_encode

print("Loading SAM 3 model...", flush=True)
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model loaded.", flush=True)


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


def _json_form_field(form, name):
    raw_value = form.get(name)
    if not raw_value:
        return None

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"'{name}' must be valid JSON") from exc


def _parse_box(box, field_name):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"'{field_name}' must be a JSON array of four numbers")

    try:
        return [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must contain only numbers") from exc


def _box_to_normalized_cxcywh(box, box_format, orig_w, orig_h, field_name):
    if box_format == "xywh":
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
    elif box_format == "xyxy":
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
    elif box_format == "cxcywh":
        cx, cy, w, h = box
    else:
        raise ValueError(f"'{field_name}_format' must be one of: xywh, xyxy, cxcywh")

    return [cx / orig_w, cy / orig_h, w / orig_w, h / orig_h]


def _parse_visual_prompts(form, orig_w, orig_h):
    visual_prompts = []

    for field_name in ("bbox", "exemplar"):
        raw_box = _json_form_field(form, field_name)
        if raw_box is None:
            continue

        box_format = form.get(f"{field_name}_format", "xywh").lower()
        box = _parse_box(raw_box, field_name)
        visual_prompts.append(
            {
                "type": field_name,
                "box": box,
                "box_format": box_format,
                "normalized_cxcywh": _box_to_normalized_cxcywh(
                    box=box,
                    box_format=box_format,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    field_name=field_name,
                ),
            }
        )

    return visual_prompts


def _apply_prompts(inference_state, prompt, visual_prompts):
    if prompt:
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=prompt
        )

    for visual_prompt in visual_prompts:
        inference_state = processor.add_geometric_prompt(
            state=inference_state,
            box=visual_prompt["normalized_cxcywh"],
            label=True,
        )

    return inference_state


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
    """Run SAM 3 segmentation on an uploaded image with text or visual prompts."""
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file"}), 400
    prompt = request.form.get("prompt")
    score_threshold = float(request.form.get("score_threshold", 0.0))

    pil_image = Image.open(request.files["image"].stream).convert("RGB")
    orig_w, orig_h = pil_image.size

    try:
        visual_prompts = _parse_visual_prompts(request.form, orig_w, orig_h)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not prompt and not visual_prompts:
        return jsonify({"error": "missing 'prompt', 'bbox', or 'exemplar' field"}), 400

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(pil_image)
        inference_state = _apply_prompts(
            inference_state=inference_state,
            prompt=prompt,
            visual_prompts=visual_prompts,
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
            "visual_prompts": visual_prompts,
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
