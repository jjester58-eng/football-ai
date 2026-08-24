"""
field_detector.py
-----------------
Detects the playable field area (inside the white lines) using
green grass color segmentation.

Returns a binary mask and a convex-hull polygon.
Only players whose foot position (bottom-center of bbox) falls inside
this mask should be analysed.
"""

from __future__ import annotations

import cv2
import numpy as np


# HSV range for natural grass green
# (works for both bright and shadowed turf)
_GRASS_H_LOW  = 30
_GRASS_H_HIGH = 90
_GRASS_S_LOW  = 30
_GRASS_V_LOW  = 40


class FieldDetector:
    """
    Detect the green field region in a video frame.

    Usage::
        fd = FieldDetector()
        mask = fd.detect(frame)          # binary np.ndarray H×W uint8
        inside = fd.is_on_field(mask, x, y)
    """

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a binary mask (255 = field, 0 = out of bounds).
        The mask is the convex hull of the largest grass-coloured blob.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([_GRASS_H_LOW, _GRASS_S_LOW, _GRASS_V_LOW])
        upper = np.array([_GRASS_H_HIGH, 255, 255])
        raw_mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

        # Largest contour → convex hull
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            # Fallback: full frame is the field
            return np.full(frame.shape[:2], 255, dtype=np.uint8)

        largest = max(contours, key=cv2.contourArea)
        hull    = cv2.convexHull(largest)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    @staticmethod
    def is_on_field(mask: np.ndarray, x: int, y: int) -> bool:
        """
        Returns True if pixel (x, y) is inside the field mask.
        Clamps coordinates to valid image bounds.
        """
        h, w = mask.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return bool(mask[y, x])
