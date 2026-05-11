"""
Minimal Flask server for SAM 3 image inference.

Start with:
    uv run python serve.py
"""

import io
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

    results = sorted(
        [
            {"box": b, "mask_rle": m, "score": s}
            for b, m, s in zip(boxes_xywh, rle_masks, scores_list)
            if s >= score_threshold
        ],
        key=lambda r: r["score"],
        reverse=True,
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


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=False)
