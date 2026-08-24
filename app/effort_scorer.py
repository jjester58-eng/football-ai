"""
effort_scorer.py
----------------
Determines whether each blue defensive player made an EFFORT to close
distance to the ball.

Definition of effort (Yes / No):
  - Track the ball position across the play.
  - Track each defender's distance to the ball per frame.
  - Compare average distance in the FIRST third of frames vs the LAST third.
  - If distance decreased → YES (made effort).
  - If distance stayed the same or increased → NO.

Ball movement is NOT required — effort is judged regardless of whether
the ball moved forward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of frames a player must appear in to be scored
_MIN_FRAMES = 6

# How much distance must decrease to count as "closed" (pixels)
# Avoids penalising tiny noise as "no effort"
_CLOSE_THRESHOLD_PX = 5.0


@dataclass
class PlayerTrack:
    track_id: int
    frame_indices: list[int] = field(default_factory=list)
    # Distance from player foot to ball center each frame (pixels)
    distances_to_ball: list[float] = field(default_factory=list)


@dataclass
class PlayerEffortReport:
    track_id: int
    effort: bool          # True = Yes, False = No
    label: str            # "EFFORT ✓" | "NO EFFORT ✗"
    dist_start: float     # average distance in first third (px)
    dist_end: float       # average distance in last third (px)
    frame_count: int


def _foot_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Bottom-center of bounding box — where the player's feet are."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class EffortScorer:
    """
    Accumulates per-frame observations then computes Yes/No effort reports.

    Usage::
        scorer = EffortScorer()
        for frame_result, classified_players in ...:
            ball_center = frame_result.ball_center()
            for cp in classified_players:
                if cp.team == "defense":
                    scorer.update(frame_result.frame_idx, cp, ball_center)
        reports = scorer.compute_reports()
    """

    def __init__(self) -> None:
        self._tracks: dict[int, PlayerTrack] = {}

    def update(
        self,
        frame_idx: int,
        classified_player,          # ClassifiedPlayer
        ball_center: Optional[tuple[float, float]],
    ) -> None:
        tid = classified_player.detection.track_id
        if tid not in self._tracks:
            self._tracks[tid] = PlayerTrack(track_id=tid)

        foot = _foot_center(classified_player.detection.bbox)

        if ball_center is not None:
            d = _dist(foot, ball_center)
        else:
            d = float("nan")

        self._tracks[tid].frame_indices.append(frame_idx)
        self._tracks[tid].distances_to_ball.append(d)

    def compute_reports(self) -> list[PlayerEffortReport]:
        reports: list[PlayerEffortReport] = []
        for tid, track in self._tracks.items():
            r = self._score_track(track)
            if r:
                reports.append(r)
        # Sort: effort first, then by how much distance they closed
        return sorted(reports, key=lambda r: (not r.effort, r.dist_end - r.dist_start))

    def per_frame_effort(self) -> dict[int, bool | None]:
        """
        Returns {track_id: effort_bool} after compute_reports() is called.
        Used by the visualizer for per-frame annotation.
        """
        result: dict[int, bool | None] = {}
        for r in self.compute_reports():
            result[r.track_id] = r.effort
        return result

    # ------------------------------------------------------------------

    def _score_track(self, track: PlayerTrack) -> Optional[PlayerEffortReport]:
        dists = [d for d in track.distances_to_ball if not np.isnan(d)]
        if len(dists) < _MIN_FRAMES:
            return None

        n = len(dists)
        third = max(1, n // 3)

        dist_start = float(np.mean(dists[:third]))
        dist_end   = float(np.mean(dists[-third:]))

        # Closed distance by more than the noise threshold?
        effort = (dist_start - dist_end) > _CLOSE_THRESHOLD_PX

        return PlayerEffortReport(
            track_id=track.track_id,
            effort=effort,
            label="EFFORT ✓" if effort else "NO EFFORT ✗",
            dist_start=round(dist_start, 1),
            dist_end=round(dist_end, 1),
            frame_count=len(track.frame_indices),
        )
