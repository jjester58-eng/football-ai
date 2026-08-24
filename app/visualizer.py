"""
visualizer.py
-------------
Draws bounding boxes and Yes/No effort labels on video frames.

Only blue defensive players (inside the field) are annotated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Colors (BGR)
_COLOR_EFFORT     = (50, 220, 50)    # green  — made effort
_COLOR_NO_EFFORT  = (50, 50, 220)    # red    — no effort
_COLOR_BALL       = (0, 215, 255)    # gold
_COLOR_FIELD_EDGE = (255, 255, 255)  # white (optional field outline)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_THICKNESS  = 2


class Visualizer:
    """
    Annotates frames and writes them to an output .mp4 file.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter at {output_path}")

    def annotate_and_write(
        self,
        frame: np.ndarray,
        classified_players: list,       # list[ClassifiedPlayer]
        balls: list,                    # list[Detection]
        effort_map: dict[int, bool | None],  # {track_id: effort_bool}
        field_mask: np.ndarray | None = None,
    ) -> None:
        annotated = frame.copy()

        # Optional: faint field overlay
        if field_mask is not None:
            tint = annotated.copy()
            tint[field_mask == 0] = (tint[field_mask == 0] * 0.5).astype(np.uint8)
            cv2.addWeighted(tint, 0.25, annotated, 0.75, 0, annotated)

        # Draw ball
        for ball in balls:
            x1, y1, x2, y2 = ball.bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated, (cx, cy), 10, _COLOR_BALL, _THICKNESS)
            cv2.putText(
                annotated, "ball",
                (x1, y1 - 6), _FONT, 0.45, _COLOR_BALL, 1
            )

        # Draw blue defensive players only
        for cp in classified_players:
            if cp.team != "defense":
                continue

            x1, y1, x2, y2 = cp.detection.bbox
            tid = cp.detection.track_id
            effort = effort_map.get(tid)

            # Choose color and label based on effort (None = still computing)
            if effort is True:
                color = _COLOR_EFFORT
                label = f"#{tid} EFFORT ✓"
            elif effort is False:
                color = _COLOR_NO_EFFORT
                label = f"#{tid} NO EFFORT ✗"
            else:
                color = (180, 180, 180)
                label = f"#{tid}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _THICKNESS)
            # Label background for readability
            (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 8), (x1 + tw + 4, y1),
                color, -1
            )
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 5), _FONT, _FONT_SCALE, (0, 0, 0), _THICKNESS - 1
            )

        self._writer.write(annotated)

    def release(self) -> None:
        self._writer.release()
        logger.info("Video written to %s", self.output_path)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


def draw_summary_overlay(
    frame: np.ndarray,
    reports: list,  # list[PlayerEffortReport]
) -> np.ndarray:
    """
    Leaderboard panel in the top-right corner showing
    player ID and Yes/No effort result.
    """
    overlay = frame.copy()
    x_start = frame.shape[1] - 210
    y_start = 18
    line_h  = 24

    panel_h = len(reports) * line_h + 35
    cv2.rectangle(
        overlay,
        (x_start - 8, y_start - 18),
        (frame.shape[1] - 5, y_start + panel_h),
        (20, 20, 20), -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, overlay)

    cv2.putText(
        overlay, "DEF EFFORT",
        (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )

    for i, r in enumerate(reports):
        y = y_start + (i + 1) * line_h
        color = _COLOR_EFFORT if r.effort else _COLOR_NO_EFFORT
        text  = f"#{r.track_id}: {r.label}"
        cv2.putText(overlay, text, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return overlay
