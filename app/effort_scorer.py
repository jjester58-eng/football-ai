"""
effort_scorer.py
----------------
Computes a per-player "effort score" for defensive players by measuring:

  1. Speed       — pixel displacement per frame (smoothed)
  2. Alignment   — cosine similarity between the player's velocity vector
                   and the vector pointing toward the ball

Effort Score (0-100) = 0.5 × speed_score + 0.5 × alignment_score

Grades:
  ● 80-100  High Effort   🟢
  ● 50-79   Moderate      🟡
  ● 0-49    Low Effort    🔴
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# Maximum expected player speed in pixels/frame (calibration constant).
# Tune per camera / resolution; used to normalise raw speed to [0, 1].
_MAX_SPEED_PX_PER_FRAME = 25.0

# Smoothing window (frames) for velocity estimation
_SMOOTH_WINDOW = 5


@dataclass
class PlayerTrack:
    """Accumulates frame-by-frame state for a single tracked player."""
    track_id: int
    team: str
    frame_centers: list[tuple[float, float]] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    ball_centers: list[Optional[tuple[float, float]]] = field(default_factory=list)

    def add_frame(
        self,
        frame_idx: int,
        center: tuple[float, float],
        ball_center: Optional[tuple[float, float]],
    ) -> None:
        self.frame_centers.append(center)
        self.frame_indices.append(frame_idx)
        self.ball_centers.append(ball_center)


@dataclass
class PlayerEffortReport:
    track_id: int
    team: str
    effort_score: float          # 0-100
    grade: str                   # "High" | "Moderate" | "Low"
    avg_speed_px_per_frame: float
    avg_alignment: float         # -1 to 1 (1 = perfectly toward ball)
    frame_count: int


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _grade(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 50:
        return "Moderate"
    return "Low"


class EffortScorer:
    """
    Aggregates per-frame observations and produces effort reports.

    Usage::

        scorer = EffortScorer()
        for frame_result, classified_players in zip(frame_results, players):
            ball_center = frame_result.ball_center()
            for cp in classified_players:
                if cp.team == "defense":
                    scorer.update(frame_result.frame_idx, cp, ball_center)
        reports = scorer.compute_reports()
    """

    def __init__(self, max_speed: float = _MAX_SPEED_PX_PER_FRAME) -> None:
        self.max_speed = max_speed
        self._tracks: dict[int, PlayerTrack] = {}

    def update(
        self,
        frame_idx: int,
        classified_player,
        ball_center: Optional[tuple[float, float]],
    ) -> None:
        """Record one frame of observation for a player."""
        tid = classified_player.detection.track_id
        if tid not in self._tracks:
            self._tracks[tid] = PlayerTrack(
                track_id=tid,
                team=classified_player.team,
            )
        c = _center(classified_player.detection.bbox)
        self._tracks[tid].add_frame(frame_idx, c, ball_center)

    def compute_reports(self) -> list[PlayerEffortReport]:
        """Return one PlayerEffortReport per tracked defensive player."""
        reports: list[PlayerEffortReport] = []
        for tid, track in self._tracks.items():
            if track.team != "defense":
                continue
            report = self._score_track(track)
            if report:
                reports.append(report)
        return sorted(reports, key=lambda r: r.effort_score, reverse=True)

    def per_frame_scores(self) -> dict[int, list[tuple[int, float]]]:
        """
        Returns {frame_idx: [(track_id, per_frame_effort_score), ...]}
        for live annotation during video rendering.
        """
        result: dict[int, list[tuple[int, float]]] = {}
        for tid, track in self._tracks.items():
            if track.team != "defense":
                continue
            centers = np.array(track.frame_centers)
            if len(centers) < 2:
                continue
            speeds = self._compute_speeds(centers)
            alignments = self._compute_alignments(centers, track.ball_centers)
            for i, (fi, speed, align) in enumerate(
                zip(track.frame_indices, speeds, alignments)
            ):
                spd_norm = min(speed / self.max_speed, 1.0)
                align_norm = (align + 1.0) / 2.0  # map [-1,1] → [0,1]
                score = round((spd_norm * 0.5 + align_norm * 0.5) * 100, 1)
                result.setdefault(fi, []).append((tid, score))
        return result

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _score_track(self, track: PlayerTrack) -> Optional[PlayerEffortReport]:
        centers = np.array(track.frame_centers)
        if len(centers) < 2:
            return None

        speeds = self._compute_speeds(centers)
        alignments = self._compute_alignments(centers, track.ball_centers)

        avg_speed = float(np.mean(speeds))
        avg_align = float(np.mean(alignments))

        spd_norm = min(avg_speed / self.max_speed, 1.0)
        align_norm = (avg_align + 1.0) / 2.0
        score = round((spd_norm * 0.5 + align_norm * 0.5) * 100, 1)

        return PlayerEffortReport(
            track_id=track.track_id,
            team=track.team,
            effort_score=score,
            grade=_grade(score),
            avg_speed_px_per_frame=round(avg_speed, 2),
            avg_alignment=round(avg_align, 3),
            frame_count=len(centers),
        )

    @staticmethod
    def _compute_speeds(centers: np.ndarray) -> np.ndarray:
        """Smoothed pixel displacement per frame."""
        deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        # Pad to same length as centers
        deltas = np.concatenate([[deltas[0]], deltas])
        if len(deltas) >= _SMOOTH_WINDOW:
            deltas = uniform_filter1d(deltas, size=_SMOOTH_WINDOW)
        return deltas

    @staticmethod
    def _compute_alignments(
        centers: np.ndarray,
        ball_centers: list[Optional[tuple[float, float]]],
    ) -> np.ndarray:
        """
        For each frame, compute cosine similarity between the player's
        velocity vector and the vector pointing toward the ball.
        Returns values in [-1, 1]; 1.0 = perfectly running at the ball.
        """
        alignments = []
        velocity = np.zeros(2)

        for i, (cx, cy) in enumerate(centers):
            # Update velocity estimate
            if i > 0:
                velocity = centers[i] - centers[i - 1]

            ball = ball_centers[i]
            if ball is None or np.linalg.norm(velocity) < 1e-6:
                alignments.append(0.0)
                continue

            to_ball = np.array(ball) - np.array([cx, cy])
            to_ball_norm = np.linalg.norm(to_ball)
            vel_norm = np.linalg.norm(velocity)

            if to_ball_norm < 1e-6:
                alignments.append(1.0)  # player is on the ball
                continue

            cos_sim = float(np.dot(velocity, to_ball) / (vel_norm * to_ball_norm))
            alignments.append(max(-1.0, min(1.0, cos_sim)))

        return np.array(alignments)
