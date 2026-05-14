"""
Minimal Flask server for SAM 3 image inference.

Start with:
    uv run python serve.py

Exposes a single endpoint:

    POST /predict

multipart/form-data fields:
    image    (file, required)  - the image to segment
    prompts  (form, required)  - JSON object keyed by user-chosen keys. Each
                                  value must be an object containing at least
                                  one of: ``text`` (alias: ``prompt``) and/or
                                  ``bbox`` (aliases: ``geometric_prompt``,
                                  ``geometric``). Optional ``bbox_format`` /
                                  ``geometric_format`` defaults to ``xywh``;
                                  allowed: ``xywh``, ``xyxy``, ``cxcywh``.
    score_threshold (form, optional, default 0.0)

The response is JSON with ``image_width``, ``image_height`` and a ``results``
object keyed by the same keys as the request (insertion order preserved).
"""

import json
import math
from collections import OrderedDict

import torch
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.train.masks_ops import rle_encode

print("Loading SAM 3 model...", flush=True)
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model loaded.", flush=True)


app = Flask(__name__, static_folder=None)
setattr(app.json, "sort_keys", False)


_TEXT_FIELDS = ("text", "prompt")
_BOX_FIELDS = ("bbox", "geometric_prompt", "geometric")
_BOX_FORMAT_FIELDS = ("bbox_format", "geometric_format")
_ALLOWED_BOX_FORMATS = ("xywh", "xyxy", "cxcywh")


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


def _parse_box(box, field_name):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"'{field_name}' must be a JSON array of four finite numbers")

    try:
        parsed = [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must contain only numbers") from exc

    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"'{field_name}' must contain only finite numbers")

    return parsed


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

    if w <= 0 or h <= 0:
        raise ValueError(f"'{field_name}' width and height must be positive")

    return [cx / orig_w, cy / orig_h, w / orig_w, h / orig_h]


def _pick_first(mapping, fields):
    """Return (field_name, value) for the first field present with a value, else (None, None)."""
    for field in fields:
        if field in mapping:
            value = mapping[field]
            if value is None:
                continue
            return field, value
    return None, None


def _parse_prompts(raw_value, orig_w, orig_h):
    """Parse the ``prompts`` form field into a list of (key, entry) pairs.

    Each entry has shape:
        {
            "text": str | None,
            "geometric_prompt": [x, y, w, h] | None  # original units, original format
            "bbox_format": str | None,
            "normalized_cxcywh": [cx, cy, w, h] | None,
        }
    """
    if raw_value is None:
        raise ValueError("missing 'prompts' field")

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("missing 'prompts' field")

    try:
        prompts = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("'prompts' must be valid JSON") from exc

    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("'prompts' must be a non-empty JSON object")

    parsed = []
    for key, entry in prompts.items():
        key_str = str(key)
        if not isinstance(entry, dict) or not entry:
            raise ValueError(f"prompts[{key_str!r}] must be a non-empty JSON object")

        _, text_value = _pick_first(entry, _TEXT_FIELDS)
        text = None
        if text_value is not None:
            if not isinstance(text_value, str):
                raise ValueError(f"prompts[{key_str!r}].text must be a string")
            text = text_value.strip() or None
        box_field, raw_box = _pick_first(entry, _BOX_FIELDS)
        box_format = None
        normalized = None
        parsed_box = None
        if raw_box is not None:
            box_format_field, fmt_value = _pick_first(entry, _BOX_FORMAT_FIELDS)
            box_format = str(fmt_value).lower() if fmt_value is not None else "xywh"
            if box_format not in _ALLOWED_BOX_FORMATS:
                raise ValueError(
                    f"'prompts[{key_str!r}].{box_format_field or 'bbox_format'}' "
                    f"must be one of: xywh, xyxy, cxcywh"
                )
            field_label = f"prompts[{key_str!r}].{box_field}"
            parsed_box = _parse_box(raw_box, field_label)
            normalized = _box_to_normalized_cxcywh(
                box=parsed_box,
                box_format=box_format,
                orig_w=orig_w,
                orig_h=orig_h,
                field_name=field_label,
            )

        if text is None and normalized is None:
            raise ValueError(
                f"prompts[{key_str!r}] must include at least one of: "
                f"text/prompt, bbox/geometric_prompt/geometric"
            )

        parsed.append(
            (
                key_str,
                {
                    "text": text,
                    "geometric_prompt": parsed_box,
                    "bbox_format": box_format,
                    "normalized_cxcywh": normalized,
                },
            )
        )

    return parsed


@app.post("/predict")
def predict():
    """Run SAM 3 segmentation on one image with a keyed batch of prompts."""
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file"}), 400

    raw_prompts = request.form.get("prompts")
    if raw_prompts is None or not raw_prompts.strip():
        return jsonify({"error": "missing 'prompts' field"}), 400

    raw_score = request.form.get("score_threshold")
    if raw_score is not None and raw_score != "":
        try:
            score_threshold = float(raw_score)
        except (TypeError, ValueError):
            return jsonify({"error": "'score_threshold' must be a number"}), 400
    else:
        score_threshold = 0.0

    try:
        pil_image = Image.open(request.files["image"].stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "'image' must be a valid image file"}), 400
    orig_w, orig_h = pil_image.size

    try:
        parsed_entries = _parse_prompts(raw_prompts, orig_w, orig_h)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    results = OrderedDict()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(pil_image)

        for key, entry in parsed_entries:
            processor.reset_all_prompts(inference_state)

            if entry["text"] is not None:
                inference_state = processor.set_text_prompt(
                    state=inference_state, prompt=entry["text"]
                )

            if entry["normalized_cxcywh"] is not None:
                inference_state = processor.add_geometric_prompt(
                    state=inference_state,
                    box=entry["normalized_cxcywh"],
                    label=True,
                )

            predictions = _format_predictions(
                inference_state=inference_state,
                orig_w=orig_w,
                orig_h=orig_h,
                score_threshold=score_threshold,
            )

            results[key] = {
                "text": entry["text"],
                "geometric_prompt": entry["geometric_prompt"],
                "bbox_format": entry["bbox_format"],
                "num_predictions": len(predictions),
                "predictions": predictions,
            }

    return jsonify(
        {
            "image_width": orig_w,
            "image_height": orig_h,
            "results": results,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=False)
