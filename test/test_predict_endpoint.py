# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for the single ``POST /predict`` endpoint in serve.py.

These tests avoid loading real SAM 3 weights or a GPU. We import ``serve.py``
with ``build_sam3_image_model`` patched to a sentinel, then replace the
module-level ``processor`` with a fake whose methods record calls and write
the small tensors that ``_format_predictions`` needs.
"""

from __future__ import annotations

import importlib.util
import io
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from PIL import Image

SERVE_PATH = Path(__file__).resolve().parents[1] / "serve.py"


def _png_bytes(size: tuple[int, int] = (32, 24)) -> io.BytesIO:
    image = Image.new("RGB", size, "black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class _FakeProcessor:
    """Records calls and produces deterministic predictions for tests."""

    def __init__(self) -> None:
        self.set_image_calls = 0
        self.reset_calls = 0
        self.text_prompts: list[str] = []
        self.geometric_prompts: list[list[float]] = []
        # Sequence of operations in the order they happen, for leak detection.
        self.events: list[tuple[str, Any]] = []

    def _populate(self, state: dict) -> dict:
        # One fake detection, score=0.9, box covers a corner of the image.
        h = state["original_height"]
        w = state["original_width"]
        state["boxes"] = torch.tensor([[0.0, 0.0, w / 2, h / 2]], dtype=torch.float32)
        state["scores"] = torch.tensor([0.9], dtype=torch.float32)
        state["masks"] = torch.zeros(1, 1, h, w, dtype=torch.bool)
        state["masks"][0, 0, : h // 2, : w // 2] = True
        return state

    # --- methods used by serve.predict ------------------------------------
    def set_image(self, image):
        self.set_image_calls += 1
        self.events.append(("set_image", None))
        return {"original_width": image.width, "original_height": image.height}

    def reset_all_prompts(self, state):
        self.reset_calls += 1
        self.events.append(("reset", None))
        for key in ("boxes", "scores", "masks", "_text", "_box"):
            state.pop(key, None)
        return None  # mutates in place; explicit None per processor contract

    def set_text_prompt(self, state, prompt):
        state["_text"] = prompt
        self.text_prompts.append(prompt)
        self.events.append(("text", prompt))
        return self._populate(state)

    def add_geometric_prompt(self, state, box, label=True):
        state["_box"] = (list(box), label)
        self.geometric_prompts.append(list(box))
        self.events.append(("geometric", list(box)))
        return self._populate(state)


def _import_serve_with_fake_processor() -> tuple[Any, _FakeProcessor]:
    with patch("sam3.model_builder.build_sam3_image_model", return_value=object()):
        spec = importlib.util.spec_from_file_location(
            "serve_under_test_" + Path(__file__).stem, SERVE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    fake = _FakeProcessor()
    setattr(module, "processor", fake)
    return module, fake


class TestPredictEndpointRoutes(unittest.TestCase):
    def test_exactly_one_predict_route(self) -> None:
        serve, _ = _import_serve_with_fake_processor()

        non_static = [
            rule for rule in serve.app.url_map.iter_rules() if rule.endpoint != "static"
        ]
        self.assertEqual(len(non_static), 1)
        rule = non_static[0]
        self.assertEqual(rule.rule, "/predict")
        self.assertIn("POST", rule.methods or set())
        # No leftover endpoints.
        endpoints = {r.rule for r in non_static}
        self.assertNotIn("/predict_multi", endpoints)
        self.assertNotIn("/predict_exemplars", endpoints)
        self.assertNotIn("/health", endpoints)


class TestPredictRequestValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.serve, self.fake = _import_serve_with_fake_processor()
        self.client = self.serve.app.test_client()

    def _post(self, **fields) -> Any:
        return self.client.post(
            "/predict", data=fields, content_type="multipart/form-data"
        )

    def test_missing_image_returns_400(self) -> None:
        response = self._post(prompts=json.dumps({"a": {"text": "cat"}}))
        self.assertEqual(response.status_code, 400)
        self.assertIn("image", response.get_json()["error"])

    def test_missing_prompts_returns_400(self) -> None:
        response = self._post(image=(_png_bytes(), "img.png"))
        self.assertEqual(response.status_code, 400)
        self.assertIn("prompts", response.get_json()["error"])

    def test_invalid_image_returns_400(self) -> None:
        response = self._post(
            image=(io.BytesIO(b"not an image"), "img.txt"),
            prompts=json.dumps({"a": {"text": "cat"}}),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid image", response.get_json()["error"])

    def test_invalid_json_prompts_returns_400(self) -> None:
        response = self._post(image=(_png_bytes(), "img.png"), prompts="not-json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", response.get_json()["error"])

    def test_non_object_prompts_returns_400(self) -> None:
        response = self._post(
            image=(_png_bytes(), "img.png"), prompts=json.dumps(["a", "b"])
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("prompts", response.get_json()["error"])

    def test_empty_prompts_returns_400(self) -> None:
        response = self._post(image=(_png_bytes(), "img.png"), prompts=json.dumps({}))
        self.assertEqual(response.status_code, 400)
        self.assertIn("prompts", response.get_json()["error"])

    def test_non_object_entry_returns_400(self) -> None:
        response = self._post(
            image=(_png_bytes(), "img.png"),
            prompts=json.dumps({"a": "just a string"}),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("a", response.get_json()["error"])

    def test_empty_entry_returns_400(self) -> None:
        response = self._post(
            image=(_png_bytes(), "img.png"), prompts=json.dumps({"a": {}})
        )
        self.assertEqual(response.status_code, 400)

    def test_entry_with_neither_text_nor_box_returns_400(self) -> None:
        response = self._post(
            image=(_png_bytes(), "img.png"),
            prompts=json.dumps({"a": {"unrelated": 1}}),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("text", response.get_json()["error"])

    def test_malformed_box_returns_400(self) -> None:
        response = self._post(
            image=(_png_bytes(), "img.png"),
            prompts=json.dumps({"a": {"bbox": [1, 2, 3]}}),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("bbox", response.get_json()["error"])

    def test_invalid_box_format_returns_400(self) -> None:
        response = self._post(
            image=(_png_bytes(), "img.png"),
            prompts=json.dumps({"a": {"bbox": [1, 2, 3, 4], "bbox_format": "polar"}}),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("xywh", response.get_json()["error"])

    def test_invalid_score_threshold_returns_400(self) -> None:
        response = self._post(
            image=(_png_bytes(), "img.png"),
            prompts=json.dumps({"a": {"text": "cat"}}),
            score_threshold="not-a-number",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("score_threshold", response.get_json()["error"])


class TestPredictEndpointBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.serve, self.fake = _import_serve_with_fake_processor()
        self.client = self.serve.app.test_client()

    def test_set_image_called_once_for_multiple_prompts(self) -> None:
        prompts = {
            "cat": {"text": "cat"},
            "dog": {"prompt": "dog"},
            "thing": {"bbox": [4, 4, 8, 8], "bbox_format": "xywh"},
        }
        response = self.client.post(
            "/predict",
            data={
                "image": (_png_bytes((40, 30)), "img.png"),
                "prompts": json.dumps(prompts),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(self.fake.set_image_calls, 1)

    def test_reset_called_before_every_entry(self) -> None:
        prompts = {
            "a": {"text": "a"},
            "b": {"text": "b"},
            "c": {"bbox": [0, 0, 10, 10]},
        }
        self.client.post(
            "/predict",
            data={
                "image": (_png_bytes(), "img.png"),
                "prompts": json.dumps(prompts),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(self.fake.reset_calls, len(prompts))
        # Each "reset" must precede the corresponding text/geometric event.
        non_image_events = [e for e in self.fake.events if e[0] != "set_image"]
        # First event must be a reset for the first entry, before any prompt.
        self.assertEqual(non_image_events[0][0], "reset")
        # Between every two resets there should be at least one prompt event
        # (otherwise we'd reset twice with no prompt applied in between).
        reset_indices = [i for i, e in enumerate(non_image_events) if e[0] == "reset"]
        for start, end in zip(reset_indices, reset_indices[1:]):
            kinds = {e[0] for e in non_image_events[start + 1 : end]}
            self.assertTrue(kinds & {"text", "geometric"})

    def test_response_keys_preserve_input_order(self) -> None:
        prompts = {
            "zeta": {"text": "z"},
            "alpha": {"text": "a"},
            "middle": {"text": "m"},
        }
        response = self.client.post(
            "/predict",
            data={
                "image": (_png_bytes(), "img.png"),
                "prompts": json.dumps(prompts),
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        self.assertEqual(list(payload["results"].keys()), list(prompts.keys()))

    def test_per_entry_text_and_geometry_applied(self) -> None:
        prompts = {
            "cat": {"text": "a cat"},
            "with_box": {"text": "a thing", "bbox": [0, 0, 10, 10]},
            "box_only": {"geometric_prompt": [2, 2, 4, 4], "geometric_format": "xywh"},
        }
        response = self.client.post(
            "/predict",
            data={
                "image": (_png_bytes((40, 40)), "img.png"),
                "prompts": json.dumps(prompts),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

        # Text prompts were applied for "cat" and "with_box" only.
        self.assertEqual(self.fake.text_prompts, ["a cat", "a thing"])
        # Geometric prompts applied for "with_box" and "box_only".
        self.assertEqual(len(self.fake.geometric_prompts), 2)

        payload = response.get_json()
        self.assertEqual(payload["image_width"], 40)
        self.assertEqual(payload["image_height"], 40)
        cat = payload["results"]["cat"]
        self.assertEqual(cat["text"], "a cat")
        self.assertIsNone(cat["geometric_prompt"])
        self.assertIsNone(cat["bbox_format"])
        self.assertEqual(cat["num_predictions"], len(cat["predictions"]))
        self.assertGreaterEqual(cat["num_predictions"], 1)

        with_box = payload["results"]["with_box"]
        self.assertEqual(with_box["text"], "a thing")
        self.assertEqual(with_box["geometric_prompt"], [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(with_box["bbox_format"], "xywh")

        box_only = payload["results"]["box_only"]
        self.assertIsNone(box_only["text"])
        self.assertEqual(box_only["geometric_prompt"], [2.0, 2.0, 4.0, 4.0])
        self.assertEqual(box_only["bbox_format"], "xywh")

    def test_no_prompt_leakage_between_entries(self) -> None:
        """An entry with no text must not inherit the previous entry's text."""
        prompts = {
            "first": {"text": "leaky"},
            "second": {"bbox": [1, 1, 2, 2]},
        }
        self.client.post(
            "/predict",
            data={
                "image": (_png_bytes(), "img.png"),
                "prompts": json.dumps(prompts),
            },
            content_type="multipart/form-data",
        )

        # Walk events; ensure between the reset that *starts* "second" and the
        # end, no text event occurs.
        events = self.fake.events
        # Sequence should be: set_image, reset, text(leaky), reset, geometric(..)
        kinds = [e[0] for e in events]
        self.assertEqual(
            kinds,
            ["set_image", "reset", "text", "reset", "geometric"],
        )


if __name__ == "__main__":
    unittest.main()
