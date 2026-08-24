"""
team_classifier.py
------------------
Classifies detected players by jersey color using K-Means in HSV space.

Logic:
  • Blue jersey (H ≈ 90-130) + blue helmet → "defense"  ← we care about these
  • White jersey (high V, low S)            → "offense"  ← ignored
  • Anything else                           → "unknown"  ← ignored

Blue helmets are used as a secondary confirmation signal since defenders
wear both blue jerseys and blue helmets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

from app.detector import Detection

logger = logging.getLogger(__name__)

# ── HSV thresholds (OpenCV: H=0-180, S=0-255, V=0-255) ───────────────
_BLUE_H_LOW  = 90
_BLUE_H_HIGH = 130
_MIN_SAT     = 60
_MIN_VAL     = 50

# White jersey: very low saturation + high value
_WHITE_MAX_SAT = 50
_WHITE_MIN_VAL = 160

# Fraction of bbox height used for jersey (middle band)
_JERSEY_TOP_FRAC    = 0.15   # skip neck/helmet shadow
_JERSEY_BOTTOM_FRAC = 0.55

# Fraction of bbox height used for helmet check (very top)
_HELMET_BOTTOM_FRAC = 0.18


@dataclass
class ClassifiedPlayer:
    detection: Detection
    team: str           # "defense" | "offense" | "unknown"
    dominant_hue: float


class TeamClassifier:
    def __init__(
        self,
        blue_h_low: int = _BLUE_H_LOW,
        blue_h_high: int = _BLUE_H_HIGH,
    ) -> None:
        self.blue_h_low  = blue_h_low
        self.blue_h_high = blue_h_high

    def classify(
        self,
        frame: np.ndarray,
        persons: list[Detection],
    ) -> list[ClassifiedPlayer]:
        return [self._classify_one(frame, d) for d in persons]

    # ------------------------------------------------------------------

    def _classify_one(self, frame: np.ndarray, det: Detection) -> ClassifiedPlayer:
        x1, y1, x2, y2 = det.bbox
        h = y2 - y1

        # ── jersey crop (middle band) ──────────────────────────────────
        j_y1 = y1 + int(h * _JERSEY_TOP_FRAC)
        j_y2 = y1 + int(h * _JERSEY_BOTTOM_FRAC)
        jersey = frame[j_y1:j_y2, x1:x2]

        if jersey.size == 0:
            return ClassifiedPlayer(det, "unknown", -1.0)

        hsv_j = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

        # ── check for white jersey first ───────────────────────────────
        if self._is_white(hsv_j):
            return ClassifiedPlayer(det, "offense", -1.0)

        # ── dominant hue of jersey ─────────────────────────────────────
        dominant_hue = self._dominant_hue(hsv_j)
        jersey_blue  = self.blue_h_low <= dominant_hue <= self.blue_h_high

        # ── helmet check (top of bbox) ─────────────────────────────────
        h_y2 = y1 + int(h * _HELMET_BOTTOM_FRAC)
        helmet = frame[y1:h_y2, x1:x2]
        helmet_blue = False
        if helmet.size > 0:
            hsv_h = cv2.cvtColor(helmet, cv2.COLOR_BGR2HSV)
            hh = self._dominant_hue(hsv_h)
            helmet_blue = self.blue_h_low <= hh <= self.blue_h_high

        # Defense = blue jersey OR (blue jersey AND blue helmet) for confidence
        if jersey_blue or helmet_blue:
            team = "defense"
        else:
            team = "unknown"

        return ClassifiedPlayer(det, team, dominant_hue)

    # ------------------------------------------------------------------

    def _is_white(self, hsv: np.ndarray) -> bool:
        """True if the majority of pixels are white/light grey."""
        white_mask = (hsv[:, :, 1] < _WHITE_MAX_SAT) & (hsv[:, :, 2] > _WHITE_MIN_VAL)
        ratio = np.sum(white_mask) / max(white_mask.size, 1)
        return ratio > 0.45

    def _dominant_hue(self, hsv: np.ndarray) -> float:
        """K-Means dominant hue of colorful pixels."""
        mask = (hsv[:, :, 1] >= _MIN_SAT) & (hsv[:, :, 2] >= _MIN_VAL)
        hues = hsv[:, :, 0][mask]
        if len(hues) < 20:
            return float(np.mean(hsv[:, :, 0]))
        hues_2d = hues.reshape(-1, 1).astype(np.float32)
        k = min(2, len(hues_2d))
        km = KMeans(n_clusters=k, n_init=3, random_state=0)
        km.fit(hues_2d)
        labels, counts = np.unique(km.labels_, return_counts=True)
        best = labels[np.argmax(counts)]
        return float(km.cluster_centers_[best][0])
