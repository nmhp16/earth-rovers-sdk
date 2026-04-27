"""Vision pipeline: VLM caption + depth (metric or relative) + optional
open-vocabulary object detection.

Depth output:
- USE_METRIC_DEPTH=1 (default): Depth-Anything-V2-Metric-Outdoor-Small returns
  depth in METERS. Region values are the MIN (closest point) — that's what the
  brain actually cares about for collision avoidance.
- USE_METRIC_DEPTH=0: legacy relative depth, normalized 0..1 (1 = closest in
  frame). Region values are the MEAN. The brain prompt and FSM both handle
  either mode via the unified obstacle_* booleans.

ObjectDetector (optional, USE_OBJECT_DETECTION=1):
- grounding-dino-tiny: open-vocabulary, accepts a text prompt
- yolov8n: closed-set COCO classes via the ultralytics package (if installed)
- Both return the same DetectedObject schema so callers don't branch.
"""

import base64
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

try:
    from config import (
        logger,
        VLM_MODEL_NAME,
        VLM_MAX_NEW_TOKENS,
        DEPTH_MODEL_NAME,
        DEPTH_MODEL_FALLBACK,
        USE_METRIC_DEPTH,
        OBSTACLE_AHEAD_THRESHOLD,
        SAFE_FORWARD_CLEARANCE_M,
        CLIFF_DETECTION_ENABLED,
        CLIFF_GRADIENT_DROP,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        USE_OBJECT_DETECTION,
        OBJECT_DETECTOR_KIND,
        GROUNDING_DINO_MODEL,
        YOLO_MODEL,
        DETECTION_TEXT_PROMPT,
        DETECTION_THRESHOLD,
        DETECTION_MAX_OBJECTS,
        CAMERA_HFOV_DEG,
    )
    from utils import pixel_to_bearing_deg
except ImportError:
    from .config import (
        logger,
        VLM_MODEL_NAME,
        VLM_MAX_NEW_TOKENS,
        DEPTH_MODEL_NAME,
        DEPTH_MODEL_FALLBACK,
        USE_METRIC_DEPTH,
        OBSTACLE_AHEAD_THRESHOLD,
        SAFE_FORWARD_CLEARANCE_M,
        CLIFF_DETECTION_ENABLED,
        CLIFF_GRADIENT_DROP,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        USE_OBJECT_DETECTION,
        OBJECT_DETECTOR_KIND,
        GROUNDING_DINO_MODEL,
        YOLO_MODEL,
        DETECTION_TEXT_PROMPT,
        DETECTION_THRESHOLD,
        DETECTION_MAX_OBJECTS,
        CAMERA_HFOV_DEG,
    )
    from .utils import pixel_to_bearing_deg


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class DepthReading:
    """Structured depth analysis output.

    Semantics depend on `is_metric`:
    - is_metric=True  -> ground/mid_ahead/sky/left/right are METERS to the
      closest point in that region (min over pixels).
    - is_metric=False -> values are normalized 0..1 (mean over region),
      where 1.0 = closest object in the frame.

    Boolean obstacle_* flags are mode-aware: a True value always means
    "obstacle close enough to be a problem" regardless of mode.
    """
    summary: str = "depth unavailable"
    is_metric: bool = False
    ground: float = 0.0
    mid_ahead: float = 0.0
    sky: float = 0.0
    left: float = 0.0
    right: float = 0.0
    obstacle_ahead: bool = False
    obstacle_left: bool = False
    obstacle_right: bool = False
    cliff: bool = False
    error: Optional[str] = None
    # Heavy fields excluded from repr / structured logs
    depth_map: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def path_clear(self) -> bool:
        return not (self.obstacle_ahead or self.obstacle_left or self.obstacle_right or self.cliff)

    def is_blocked_ahead(self) -> bool:
        return self.obstacle_ahead or self.cliff

    def clearance_ahead_m(self) -> Optional[float]:
        """Best-effort metric clearance ahead. Only meaningful when metric."""
        if self.is_metric and self.mid_ahead > 0:
            return self.mid_ahead
        return None


DEPTH_UNAVAILABLE = DepthReading(summary="unavailable")


@dataclass
class DetectedObject:
    """Single detection with spatial info."""
    label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # in image pixel coords
    bearing_deg: float = 0.0     # +left, -right, 0 = ahead
    distance_m: Optional[float] = None  # only set when metric depth was available

    def to_brain_line(self) -> str:
        if self.distance_m is not None:
            return f"{self.label} at {self.bearing_deg:+.0f}°, {self.distance_m:.1f}m (conf {self.confidence:.2f})"
        return f"{self.label} at {self.bearing_deg:+.0f}° (conf {self.confidence:.2f})"


# ==============================================================================
# VLM + DEPTH
# ==============================================================================

DEFAULT_VLM_MODEL = "microsoft/git-large-coco"
SMOLVLM_MODELS = ["smolvlm", "smol"]


class VisionSystem:
    """Captioning + metric/relative depth + optional object detection."""

    def __init__(self):
        model_name = VLM_MODEL_NAME

        if "florence" in model_name.lower() or "moondream" in model_name.lower():
            logger.warning(
                f"{model_name} has compatibility issues with transformers 5.x. "
                f"Using GIT model instead."
            )
            model_name = DEFAULT_VLM_MODEL

        self.device, self.dtype = self._select_device()
        self.model_name = model_name

        self._load_vision_model()
        self._load_depth_model()

        # Optional object detector
        self.detector: Optional[ObjectDetector] = None
        if USE_OBJECT_DETECTION:
            try:
                self.detector = ObjectDetector(self.device, self.dtype)
            except Exception as e:
                logger.error(f"Object detector init failed: {e}. Continuing without detection.")
                self.detector = None

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    def _select_device(self) -> tuple:
        if torch.cuda.is_available():
            return "cuda", torch.float16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_vision_model(self):
        logger.info(f"Loading VLM from {self.model_name}...")
        model_lower = self.model_name.lower()
        if any(s in model_lower for s in SMOLVLM_MODELS):
            self._load_smolvlm()
        elif "git" in model_lower:
            self._load_git()
        else:
            self._load_blip_fallback()

    def _load_git(self):
        from transformers import AutoProcessor, AutoModelForCausalLM
        logger.info(f"Loading GIT from {self.model_name}...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_git = True
            self.is_smolvlm = False
            logger.info(f"GIT loaded on {self.device} with {self.dtype}")
        except Exception as e:
            logger.error(f"Failed to load GIT: {e}")
            self._load_blip_fallback()

    def _load_smolvlm(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        logger.info(f"Loading SmolVLM from {self.model_name}...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                _attn_implementation="eager",
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_git = False
            self.is_smolvlm = True
            logger.info(f"SmolVLM loaded on {self.device} with {self.dtype}")
        except Exception as e:
            logger.warning(f"Failed to load SmolVLM: {e}. Falling back to GIT.")
            self.model_name = DEFAULT_VLM_MODEL
            self._load_git()

    def _load_blip_fallback(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        fallback = "Salesforce/blip-image-captioning-base"
        logger.info(f"Loading BLIP fallback from {fallback}...")
        try:
            self.processor = BlipProcessor.from_pretrained(fallback)
            self.model = BlipForConditionalGeneration.from_pretrained(
                fallback, torch_dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_git = False
            self.is_smolvlm = False
            logger.info(f"BLIP loaded on {self.device} with {self.dtype}")
        except Exception as e:
            logger.error(f"Failed to load BLIP: {e}")
            raise RuntimeError("No vision model available") from e

    def _load_depth_model(self):
        """Load depth model. Try metric first, fall back to relative if needed."""
        self.depth_is_metric = USE_METRIC_DEPTH
        primary = DEPTH_MODEL_NAME
        try:
            logger.info(f"Loading Depth-Anything from {primary}...")
            self.depth_processor = AutoImageProcessor.from_pretrained(primary)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                primary, torch_dtype=self.dtype
            )
            self.depth_model.to(self.device)
            self.depth_model.eval()
            self.depth_enabled = True
            logger.info(
                f"Depth ({'metric' if self.depth_is_metric else 'relative'}) "
                f"loaded on {self.device} with {self.dtype}"
            )
        except Exception as e:
            logger.warning(f"Primary depth model {primary} failed: {e}. Trying fallback.")
            try:
                self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_FALLBACK)
                self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                    DEPTH_MODEL_FALLBACK, torch_dtype=self.dtype
                )
                self.depth_model.to(self.device)
                self.depth_model.eval()
                self.depth_enabled = True
                self.depth_is_metric = False  # fallback is relative
                logger.info(f"Depth fallback (relative) loaded: {DEPTH_MODEL_FALLBACK}")
            except Exception as e2:
                logger.error(f"Both depth models failed: {e2}")
                self.depth_enabled = False

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------
    def caption_b64(self, image_b64: str) -> str:
        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((384, 384))
            if getattr(self, "is_smolvlm", False):
                cap = self._caption_smolvlm(img)
            elif self.is_git:
                cap = self._caption_git(img)
            else:
                cap = self._caption_blip(img)
            return self._clean_caption(cap)
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return "camera malfunction"

    def _caption_smolvlm(self, img: Image.Image) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Describe this scene briefly for robot navigation. "
                         "Focus on obstacles, paths, and terrain."}
            ]
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS, do_sample=False)
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _caption_git(self, img: Image.Image) -> str:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=inputs.pixel_values, max_length=50, num_beams=4,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def _caption_blip(self, img: Image.Image) -> str:
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def _clean_caption(self, caption: str) -> str:
        caption = re.sub(r"(.)\1{3,}", r"\1", caption)
        caption = re.sub(r"\[\s*unused\d+\s*\]", "", caption)
        caption = " ".join(caption.split())
        if len(caption) > 150:
            caption = caption[:150].rsplit(" ", 1)[0] + "..."
        return caption.strip()

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------
    def analyze_depth_b64(self, image_b64: str) -> DepthReading:
        if not self.depth_enabled:
            return DepthReading(summary="depth unavailable")

        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

            inputs = self.depth_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth

            depth = predicted_depth.squeeze().cpu().float().numpy()

            if self.depth_is_metric:
                # Metric: depth IS in meters. Region values = MIN (closest point).
                # Clamp at 0.05m to avoid zero-distance noise from black pixels.
                depth_m = np.clip(depth, 0.05, 50.0)
                return self._summarize_metric(depth_m)
            else:
                # Relative: normalize 0..1, region values = MEAN.
                d_min, d_max = float(depth.min()), float(depth.max())
                if d_max - d_min > 0:
                    depth_norm = (depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = depth
                return self._summarize_relative(depth_norm)

        except Exception as e:
            logger.error(f"Depth analysis error: {e}")
            return DepthReading(summary="depth error", error=str(e))

    def _summarize_metric(self, depth_m: np.ndarray) -> DepthReading:
        h, w = depth_m.shape
        ground_r = depth_m[2 * h // 3:, w // 3:2 * w // 3]
        mid_r    = depth_m[h // 3:2 * h // 3, w // 3:2 * w // 3]
        sky_r    = depth_m[:h // 3, w // 3:2 * w // 3]
        left_r   = depth_m[h // 3:2 * h // 3, :w // 3]
        right_r  = depth_m[h // 3:2 * h // 3, 2 * w // 3:]

        # Use a low percentile rather than strict min — robust to single-pixel noise
        ground = float(np.percentile(ground_r, 50))   # ground average
        mid    = float(np.percentile(mid_r,    10))   # closest 10% in mid
        sky    = float(np.percentile(sky_r,    50))
        left   = float(np.percentile(left_r,  10))
        right  = float(np.percentile(right_r, 10))

        ahead = mid    < SAFE_FORWARD_CLEARANCE_M
        obs_l = left   < SAFE_FORWARD_CLEARANCE_M
        obs_r = right  < SAFE_FORWARD_CLEARANCE_M

        cliff = self._detect_cliff_metric(depth_m) if CLIFF_DETECTION_ENABLED else False

        obstacles = []
        if ahead: obstacles.append(f"ahead<{mid:.1f}m")
        if obs_l: obstacles.append(f"left<{left:.1f}m")
        if obs_r: obstacles.append(f"right<{right:.1f}m")
        if cliff: obstacles.append("cliff")

        depth_str = f"ahead={mid:.2f}m, left={left:.2f}m, right={right:.2f}m, ground={ground:.2f}m"
        obs_str = ", ".join(obstacles) if obstacles else "path clear"

        return DepthReading(
            summary=f"{depth_str} | {obs_str}",
            is_metric=True,
            ground=ground, mid_ahead=mid, sky=sky, left=left, right=right,
            obstacle_ahead=ahead, obstacle_left=obs_l, obstacle_right=obs_r,
            cliff=cliff,
            depth_map=depth_m,
        )

    def _summarize_relative(self, depth_n: np.ndarray) -> DepthReading:
        h, w = depth_n.shape
        ground = float(depth_n[2 * h // 3:, w // 3:2 * w // 3].mean())
        mid    = float(depth_n[h // 3:2 * h // 3, w // 3:2 * w // 3].mean())
        sky    = float(depth_n[:h // 3, w // 3:2 * w // 3].mean())
        left   = float(depth_n[h // 3:2 * h // 3, :w // 3].mean())
        right  = float(depth_n[h // 3:2 * h // 3, 2 * w // 3:].mean())

        ahead = mid > OBSTACLE_AHEAD_THRESHOLD
        obs_l = left > OBSTACLE_AHEAD_THRESHOLD
        obs_r = right > OBSTACLE_AHEAD_THRESHOLD
        cliff = self._detect_cliff_relative(depth_n) if CLIFF_DETECTION_ENABLED else False

        obstacles = []
        if ahead: obstacles.append("obstacle ahead")
        if obs_l: obstacles.append("obstacle left")
        if obs_r: obstacles.append("obstacle right")
        if cliff: obstacles.append("cliff")

        depth_str = f"Close(ground)={ground:.2f}, Mid(ahead)={mid:.2f}, Far(sky)={sky:.2f}"
        obs_str = ", ".join(obstacles) if obstacles else "path clear"

        return DepthReading(
            summary=f"{depth_str} | {obs_str}",
            is_metric=False,
            ground=ground, mid_ahead=mid, sky=sky, left=left, right=right,
            obstacle_ahead=ahead, obstacle_left=obs_l, obstacle_right=obs_r,
            cliff=cliff,
            depth_map=depth_n,
        )

    def _detect_cliff_metric(self, depth_m: np.ndarray) -> bool:
        """Cliff = bottom strip of frame should monotonically get CLOSER as we
        move down the image (i.e. depth decreases). A sudden depth INCREASE
        between adjacent rows = a downstep (curb / stairs / drop)."""
        h, w = depth_m.shape
        bottom = depth_m[int(h * 0.75):, int(w * 0.3):int(w * 0.7)]
        if bottom.size == 0:
            return False
        row_means = bottom.mean(axis=1)
        if len(row_means) < 2:
            return False
        # Walking from middle to bottom: row index increases, distance should decrease.
        # If a later row jumps significantly farther than an earlier one => cliff.
        max_jump = float(np.max(np.diff(row_means)))
        baseline = float(np.mean(row_means)) + 1e-3
        return max_jump > CLIFF_GRADIENT_DROP * baseline

    def _detect_cliff_relative(self, depth_n: np.ndarray) -> bool:
        """In normalized form, bottom of frame should have HIGH depth values
        (close = high). A sudden DROP in normalized depth = the ground
        suddenly looks further away = cliff."""
        h, w = depth_n.shape
        bottom = depth_n[int(h * 0.75):, int(w * 0.3):int(w * 0.7)]
        if bottom.size == 0:
            return False
        row_means = bottom.mean(axis=1)
        if len(row_means) < 2:
            return False
        max_drop = float(np.max(-np.diff(row_means)))
        return max_drop > CLIFF_GRADIENT_DROP

    # ------------------------------------------------------------------
    # Object detection
    # ------------------------------------------------------------------
    def detect_b64(self, image_b64: str, depth: Optional[DepthReading] = None) -> List[DetectedObject]:
        """Run object detection and attach bearing/distance per detection.

        Returns [] when detection is disabled or unavailable.
        """
        if self.detector is None:
            return []
        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Detector resizes internally for its own preprocessing
            raw_dets = self.detector.detect(img)
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []

        # Map bbox center -> bearing; sample depth_map at the center for distance
        out: List[DetectedObject] = []
        det_w, det_h = img.size
        for det in raw_dets[:DETECTION_MAX_OBJECTS]:
            x1, y1, x2, y2 = det.bbox_xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bearing = pixel_to_bearing_deg(cx, det_w, CAMERA_HFOV_DEG)
            distance: Optional[float] = None
            if depth is not None and depth.depth_map is not None and depth.is_metric:
                # Map detection-space coords -> depth-map coords
                dh, dw = depth.depth_map.shape
                dx = int(cx / det_w * dw)
                dy = int(cy / det_h * dh)
                dx = max(0, min(dw - 1, dx))
                dy = max(0, min(dh - 1, dy))
                # Use a 5x5 patch around center
                patch = depth.depth_map[
                    max(0, dy - 2): dy + 3,
                    max(0, dx - 2): dx + 3,
                ]
                if patch.size > 0:
                    distance = float(np.percentile(patch, 25))
            out.append(DetectedObject(
                label=det.label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                bearing_deg=bearing,
                distance_m=distance,
            ))
        return out

    def process_image(self, image_b64: str) -> dict:
        """Convenience: run all enabled stages on a single frame."""
        depth = self.analyze_depth_b64(image_b64)
        return {
            "caption": self.caption_b64(image_b64),
            "depth": depth,
            "objects": self.detect_b64(image_b64, depth),
        }


# ==============================================================================
# OBJECT DETECTOR
# ==============================================================================

@dataclass
class _RawDet:
    label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


class ObjectDetector:
    """Thin wrapper around grounding-dino-tiny or YOLO that returns _RawDet."""

    def __init__(self, device: str, dtype):
        self.device = device
        self.dtype = dtype
        self.kind = OBJECT_DETECTOR_KIND
        self.processor = None
        self.model = None
        self.yolo = None

        if self.kind == "grounding-dino":
            self._load_grounding_dino()
        elif self.kind == "yolo":
            self._load_yolo()
        else:
            raise ValueError(f"Unknown OBJECT_DETECTOR_KIND={self.kind}")

    def _load_grounding_dino(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        logger.info(f"Loading grounding-dino from {GROUNDING_DINO_MODEL}...")
        self.processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL, torch_dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Grounding-DINO loaded on {self.device}")

    def _load_yolo(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ultralytics not installed. `pip install ultralytics` or set "
                "OBJECT_DETECTOR_KIND=grounding-dino"
            ) from e
        logger.info(f"Loading YOLO from {YOLO_MODEL}...")
        self.yolo = YOLO(YOLO_MODEL)
        logger.info("YOLO loaded")

    def detect(self, img: Image.Image) -> List[_RawDet]:
        if self.kind == "grounding-dino":
            return self._detect_grounding_dino(img)
        return self._detect_yolo(img)

    def _detect_grounding_dino(self, img: Image.Image) -> List[_RawDet]:
        text = DETECTION_TEXT_PROMPT
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            box_threshold=DETECTION_THRESHOLD,
            text_threshold=DETECTION_THRESHOLD,
            target_sizes=[img.size[::-1]],
        )[0]

        out: List[_RawDet] = []
        labels = results.get("labels") or results.get("text_labels") or []
        scores = results.get("scores") or []
        boxes = results.get("boxes") or []
        for label, score, box in zip(labels, scores, boxes):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            out.append(_RawDet(
                label=str(label),
                confidence=float(score),
                bbox_xyxy=(x1, y1, x2, y2),
            ))
        # Sort by confidence desc, return all (caller caps count)
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out

    def _detect_yolo(self, img: Image.Image) -> List[_RawDet]:
        results = self.yolo.predict(img, conf=DETECTION_THRESHOLD, verbose=False)
        out: List[_RawDet] = []
        if not results:
            return out
        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            out.append(_RawDet(
                label=names.get(cls_id, str(cls_id)),
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2),
            ))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out
