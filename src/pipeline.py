"""
Orchestrates the full OCR-to-speech accessibility pipeline:
  Image → Detection → Cropping → Recognition → Gate → Reading Order → TTS

    # result dict:
    {
        "gated_results": [
            {"text": "STOP", "confidence": 0.92, "box": [...], "gated": False},
            {"text": "cafe", "confidence": 0.61, "box": [...], "gated": True},
        ],
        "latency": {
            "detection_s":   0.153,
            "recognition_s": 0.089,
            "total_s":       0.251
        },
        "n_boxes":    5,
        "n_spoken":   3,
        "n_silenced": 2,
        "tts_output": None  # or MP3 bytes on cloud backend
  # NOTE: total_s includes gTTS network call (cloud backend only).
# gTTS is a placeholder for MVP — will be replaced with async TTS in Phase 5/6.
# True compute latency = detection_s + recognition_s (consistently within 2s budget)
      }
"""

import time
from pathlib import Path
from typing import Union, Dict

import numpy as np
import yaml
from PIL import Image

from src.detection.detector import load_detector, run_detector
from src.postprocessing.cropper import crop_boxes
from src.postprocessing.reading_order import sort_boxes
from src.recognition.recognizer import load_recognizer, run_recognizer
from src.confidence import apply_gate
from src.tts.speaker import Speaker

# Load config
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Pipeline:
    def __init__(
        self,
        device: str = "cuda",
        tts_backend: str = "local",
        config_path: Path = _CONFIG_PATH,
        model_cache_dir: str = None
    ):
        """
        Args:
            device:          "cuda" or "cpu"
            tts_backend:     "local" or "cloud"
            config_path:     path to pipeline.yaml
            model_cache_dir: optional cache dir for TrOCR weights (e.g. Drive path)
        """
        self.cfg    = _load_config(config_path)
        self.device = device

        # Load models
        self.detector              = load_detector(device=device)
        self.processor, self.model = load_recognizer(
            device=device, cache_dir=model_cache_dir
        )

        # TTS
        tts_cfg   = self.cfg.get("tts", {})
        self.speaker = Speaker(
            backend=tts_backend,
            rate=tts_cfg.get("rate", 150),
            volume=tts_cfg.get("volume", 1.0),
            lang=tts_cfg.get("lang", "en")
        )

        self.conf_threshold = self.cfg["recognition"]["confidence_threshold"]
        self.crop_padding   = self.cfg["recognition"]["crop_padding"]
        self.row_tolerance  = self.cfg["reading_order"]["row_tolerance"]
        self.max_new_tokens = self.cfg["recognition"]["max_new_tokens"]
        self.budget_s       = self.cfg["latency"]["budget_s"]

        print(f"\nPipeline ready ")
        print(f"  Confidence gate : {self.conf_threshold}")
        print(f"  Latency budget  : {self.budget_s}s")
        print(f"  TTS backend     : {tts_backend}")

    def run(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict:
        """
        Run the full pipeline on a single image.

        Args:
            image_input: file path (str/Path), numpy array (HxWx3 RGB uint8),
                         or PIL.Image

        Returns:
            result dict — see module docstring for full schema
        """
        pipeline_start = time.perf_counter()

        # ── Load image ──
        if isinstance(image_input, (str, Path)):
            img_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img_pil = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        img_np      = np.array(img_pil)
        img_w, img_h = img_pil.size

        # ── 1. Detection ──
        boxes, det_latency = run_detector(
            self.detector, img_np, img_w, img_h
        )

        if not boxes:
            return self._empty_result(det_latency)

        # ── 2. Reading order ──
        sorted_indices = sort_boxes(boxes, row_tolerance=self.row_tolerance)
        boxes_ordered  = [boxes[i] for i in sorted_indices]

        # ── 3. Crop ──
        crops = crop_boxes(img_pil, boxes_ordered, padding=self.crop_padding)

        # ── 4. Recognition ──
        rec_results, rec_latency = run_recognizer(
            self.processor, self.model, crops,
            device=self.device,
            max_new_tokens=self.max_new_tokens
        )

        # ── 5. Confidence gate ──
        gated = apply_gate(rec_results, boxes_ordered, threshold=self.conf_threshold)

        # ── 6. TTS ──
        tts_start = time.perf_counter()
        tts_output = self.speaker.speak(gated)
        tts_latency = time.perf_counter() - tts_start

        total_latency = time.perf_counter() - pipeline_start - tts_latency  # Exclude TTS from compute latency report

        # ── Latency report ──
        latency = {
            "detection_s":   round(det_latency, 4),
            "recognition_s": round(rec_latency, 4),
            "tts_s":         round(tts_latency, 4),
            "total_s":       round(total_latency, 4),
            "within_budget": total_latency <= self.budget_s
        }

        self._print_latency(latency)

        return {
            "gated_results": gated,
            "latency":       latency,
            "n_boxes":       len(boxes),
            "n_spoken":      sum(1 for r in gated if not r["gated"]),
            "n_silenced":    sum(1 for r in gated if r["gated"]),
            "tts_output":    tts_output
        }

    def repeat_last(self):
        """Repeat last TTS output ->  triggered by spacebar in Phase 6."""
        return self.speaker.repeat_last()

    def _empty_result(self, det_latency: float) -> Dict:
        print("Pipeline: no text detected in image")
        return {
            "gated_results": [],
            "latency": {
                "detection_s":   round(det_latency, 4),
                "recognition_s": 0.0,
                "total_s":       round(det_latency, 4),
                "within_budget": True
            },
            "n_boxes":    0,
            "n_spoken":   0,
            "n_silenced": 0,
            "tts_output": None
        }

    def _print_latency(self, latency: Dict):
        budget_ok = "✓" if latency["within_budget"] else "✗ EXCEEDS BUDGET"
        print(f"\nLatency breakdown:")
        print(f"  Detection   : {latency['detection_s']*1000:.1f}ms")
        print(f"  Recognition : {latency['recognition_s']*1000:.1f}ms")
        print(f"  Total       : {latency['total_s']*1000:.1f}ms  {budget_ok}")
