"""
visualizer.py
-------------
Draws annotated bounding boxes, team labels, and effort scores onto
video frames. Produces the final annotated output video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Colors (BGR)
_COLOR_DEFENSE = (220, 80, 20)    # blue-ish
_COLOR_OFFENSE = (40, 180, 40)    # green
_COLOR_BALL = (0, 215, 255)       # gold
_COLOR_UNKNOWN = (180, 180, 180)  # grey
_COLOR_HIGH = (0, 200, 0)
_COLOR_MODERATE = (0, 200, 255)
_COLOR_LOW = (0, 0, 220)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS = 2


def _grade_color(score: float) -> tuple[int, int, int]:
    if score >= 80:
        return _COLOR_HIGH
    if score >= 50:
        return _COLOR_MODERATE
    return _COLOR_LOW


def _team_color(team: str) -> tuple[int, int, int]:
    if team == "defense":
        return _COLOR_DEFENSE
    if team == "offense":
        return _COLOR_OFFENSE
    return _COLOR_UNKNOWN


class Visualizer:
    """
    Annotates frames and writes them to an output video file.

    Parameters
    ----------
    output_path : destination .mp4 file path
    fps : frames per second of the output video
    width, height : frame dimensions
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
        classified_players: list,      # list[ClassifiedPlayer]
        balls: list,                   # list[Detection]
        frame_scores: dict[int, float], # {track_id: effort_score}
    ) -> None:
        """Draw all annotations and write the frame to the output file."""
        annotated = frame.copy()

        # Draw ball
        for ball in balls:
            x1, y1, x2, y2 = ball.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _COLOR_BALL, _THICKNESS)
            cv2.putText(
                annotated, "ball",
                (x1, y1 - 6), _FONT, _FONT_SCALE - 0.1, _COLOR_BALL, 1
            )

        # Draw players
        for cp in classified_players:
            x1, y1, x2, y2 = cp.detection.bbox
            color = _team_color(cp.team)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _THICKNESS)

            tid = cp.detection.track_id
            score = frame_scores.get(tid)

            if cp.team == "defense" and score is not None:
                grade_color = _grade_color(score)
                label = f"DEF #{tid} | {score:.0f}"
                cv2.putText(
                    annotated, label,
                    (x1, y1 - 8), _FONT, _FONT_SCALE, grade_color, _THICKNESS
                )
                # Effort bar below the box
                bar_w = x2 - x1
                filled = int(bar_w * score / 100)
                cv2.rectangle(
                    annotated,
                    (x1, y2 + 2), (x2, y2 + 8),
                    (60, 60, 60), -1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y2 + 2), (x1 + filled, y2 + 8),
                    grade_color, -1
                )
            else:
                label = f"OFF #{tid}" if cp.team == "offense" else f"#{tid}"
                cv2.putText(
                    annotated, label,
                    (x1, y1 - 8), _FONT, _FONT_SCALE, color, 1
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
    Draw a compact leaderboard of effort scores in the top-right corner.
    Returns a copy of the frame with the overlay applied.
    """
    overlay = frame.copy()
    x_start = frame.shape[1] - 230
    y_start = 15
    line_h = 22

    # Background panel
    panel_h = len(reports) * line_h + 30
    cv2.rectangle(
        overlay,
        (x_start - 8, y_start - 15),
        (frame.shape[1] - 5, y_start + panel_h),
        (30, 30, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, overlay)

    cv2.putText(
        overlay, "DEF EFFORT",
        (x_start, y_start), _FONT, 0.55, (255, 255, 255), 1
    )
    for i, r in enumerate(reports):
        y = y_start + (i + 1) * line_h
        color = _grade_color(r.effort_score)
        text = f"#{r.track_id}: {r.effort_score:.0f}  [{r.grade}]"
        cv2.putText(overlay, text, (x_start, y), _FONT, 0.48, color, 1)

    return overlay
