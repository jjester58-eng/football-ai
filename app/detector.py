"""
detector.py
-----------
YOLOv8 player and ball detection with ByteTrack multi-object tracking.

Outputs per-frame detection results as structured dicts for downstream
team classification and effort scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class indices used by yolov8*.pt
_CLASS_PERSON = 0
_CLASS_SPORTS_BALL = 32


@dataclass
class Detection:
    track_id: int
    label: str          # "person" | "ball"
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coords)
    confidence: float
    frame_idx: int


@dataclass
class FrameResult:
    frame_idx: int
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)

    @property
    def persons(self) -> list[Detection]:
        return [d for d in self.detections if d.label == "person"]

    @property
    def balls(self) -> list[Detection]:
        return [d for d in self.detections if d.label == "ball"]

    def ball_center(self) -> tuple[float, float] | None:
        """Return (cx, cy) of the first detected ball, or None."""
        for d in self.balls:
            x1, y1, x2, y2 = d.bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None


class PlayerDetector:
    """
    Wraps a YOLOv8 model with ByteTrack to produce per-frame detections.

    Parameters
    ----------
    model_path : str
        Path to a .pt weights file, or a model name like 'yolov8m.pt'
        (auto-downloaded from Ultralytics hub on first use).
    conf_threshold : float
        Minimum confidence to keep a detection.
    device : str
        Torch device string, e.g. 'cpu', 'cuda', '0'.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_threshold: float = 0.35,
        device: str = "cpu",
    ) -> None:
        logger.info("Loading YOLO model: %s on device=%s", model_path, device)
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.device = device

    def process_video(
        self,
        video_path: str | Path,
        skip_frames: int = 0,
    ) -> Iterator[FrameResult]:
        """
        Yield FrameResult objects for every processed frame.

        Parameters
        ----------
        video_path : path to input video file
        skip_frames : process every (skip_frames+1)-th frame to save time
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        logger.info("Tracking video: %s", video_path)
        results = self.model.track(
            source=str(video_path),
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=[_CLASS_PERSON, _CLASS_SPORTS_BALL],
            device=self.device,
            stream=True,
            verbose=False,
        )

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        for result in results:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            detections = self._parse_result(result, frame_idx)
            yield FrameResult(frame_idx=frame_idx, frame=frame, detections=detections)
            frame_idx += 1

        cap.release()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _parse_result(self, result, frame_idx: int) -> list[Detection]:
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in (_CLASS_PERSON, _CLASS_SPORTS_BALL):
                continue

            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            label = "person" if cls_id == _CLASS_PERSON else "ball"

            detections.append(
                Detection(
                    track_id=track_id,
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    frame_idx=frame_idx,
                )
            )

        return detections
