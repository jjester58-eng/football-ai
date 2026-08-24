"""
team_classifier.py
------------------
Classifies detected players into teams by jersey color using K-Means
clustering in the HSV color space.

By default:
  • Blue hue (H ≈ 100–140° in OpenCV's 0-180 range → 50–70) → "defense"
  • Everything else → "offense"

The blue range is configurable at construction time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

from app.detector import Detection

logger = logging.getLogger(__name__)

# OpenCV HSV: H is 0-180, S and V are 0-255
_BLUE_H_LOW = 90    # ~180° in standard HSV
_BLUE_H_HIGH = 130  # ~260° in standard HSV
_MIN_SATURATION = 60
_MIN_VALUE = 50

# Fraction of bounding-box height used for jersey crop (top portion)
_JERSEY_CROP_FRACTION = 0.45


@dataclass
class ClassifiedPlayer:
    detection: Detection
    team: str           # "defense" (blue) | "offense" | "unknown"
    dominant_hue: float  # mean hue of dominant cluster


class TeamClassifier:
    """
    Assigns each person Detection to a team based on jersey color.

    Parameters
    ----------
    blue_h_low / blue_h_high : OpenCV hue bounds for blue (0-180 scale).
    min_saturation / min_value : Ignore near-grey/black pixels.
    """

    def __init__(
        self,
        blue_h_low: int = _BLUE_H_LOW,
        blue_h_high: int = _BLUE_H_HIGH,
        min_saturation: int = _MIN_SATURATION,
        min_value: int = _MIN_VALUE,
    ) -> None:
        self.blue_h_low = blue_h_low
        self.blue_h_high = blue_h_high
        self.min_saturation = min_saturation
        self.min_value = min_value

    def classify(
        self,
        frame: np.ndarray,
        persons: list[Detection],
    ) -> list[ClassifiedPlayer]:
        """
        Return a ClassifiedPlayer for each person in the frame.
        """
        results: list[ClassifiedPlayer] = []
        for det in persons:
            team, hue = self._classify_one(frame, det)
            results.append(ClassifiedPlayer(detection=det, team=team, dominant_hue=hue))
        return results

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _classify_one(
        self, frame: np.ndarray, det: Detection
    ) -> tuple[str, float]:
        x1, y1, x2, y2 = det.bbox
        h = y2 - y1

        # Crop the jersey region (upper portion of the bounding box)
        jersey_y2 = y1 + int(h * _JERSEY_CROP_FRACTION)
        jersey_crop = frame[y1:jersey_y2, x1:x2]

        if jersey_crop.size == 0:
            return "unknown", -1.0

        hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)

        # Filter out low-saturation / dark pixels (grass, shadows)
        mask = (
            (hsv[:, :, 1] >= self.min_saturation)
            & (hsv[:, :, 2] >= self.min_value)
        )
        hues = hsv[:, :, 0][mask]

        if len(hues) < 20:
            # Not enough colorful pixels — fall back to mean hue
            mean_hue = float(np.mean(hsv[:, :, 0]))
            team = self._hue_to_team(mean_hue)
            return team, mean_hue

        # K-Means on hue values to find dominant color cluster
        hues_2d = hues.reshape(-1, 1).astype(np.float32)
        k = min(2, len(hues_2d))
        km = KMeans(n_clusters=k, n_init=3, random_state=0)
        km.fit(hues_2d)

        # Pick the cluster with the most pixels
        labels, counts = np.unique(km.labels_, return_counts=True)
        dominant_label = labels[np.argmax(counts)]
        dominant_hue = float(km.cluster_centers_[dominant_label][0])

        team = self._hue_to_team(dominant_hue)
        return team, dominant_hue

    def _hue_to_team(self, hue: float) -> str:
        if self.blue_h_low <= hue <= self.blue_h_high:
            return "defense"
        return "offense"
