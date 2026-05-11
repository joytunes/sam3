"""
Minimal FastAPI server for SAM 3 image inference.

Start with:
    uv run uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import io
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.train.masks_ops import rle_encode

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    model_state["processor"] = processor
    print("Model loaded.")
    yield
    model_state.clear()


app = FastAPI(title="SAM 3 Inference", lifespan=lifespan)


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    score_threshold: float = Form(0.0),
):
    """Run SAM 3 segmentation on an uploaded image with a text prompt."""
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_w, orig_h = pil_image.size

    processor: Sam3Processor = model_state["processor"]
    inference_state = processor.set_image(pil_image)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    boxes = inference_state["boxes"]
    scores = inference_state["scores"]
    masks = inference_state["masks"]

    # Normalize boxes to [0, 1] and convert to xywh
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

    # RLE-encode masks
    rle_masks = rle_encode(masks.squeeze(1))
    rle_masks = [m["counts"] for m in rle_masks]

    scores_list = scores.tolist()

    # Filter by score threshold and sort descending
    results = sorted(
        [
            {"box": b, "mask_rle": m, "score": s}
            for b, m, s in zip(boxes_xywh, rle_masks, scores_list)
            if s >= score_threshold
        ],
        key=lambda r: r["score"],
        reverse=True,
    )

    return JSONResponse(
        content={
            "image_width": orig_w,
            "image_height": orig_h,
            "prompt": prompt,
            "num_predictions": len(results),
            "predictions": results,
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "processor" in model_state}
