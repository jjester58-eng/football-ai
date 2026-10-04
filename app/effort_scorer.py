"""
effort_scorer.py
----------------
Determines whether each blue defensive player made an EFFORT to close
distance to the play target.

Definition of effort (Yes / No):
  - Track the ball position across the play.
  - Track each defender's distance to the ball per frame.
  - Compare start/end distance as one signal.
  - Also measure whether the defender repeatedly moves toward the target.
  - A moving target can increase final distance even when the defender is
    pursuing correctly, so start/end distance is no longer the only test.

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
_MIN_FRAMES = 15

# How much distance must decrease to count as "closed" (pixels)
_CLOSE_THRESHOLD_PX = 5.0

# A real defensive unit has exactly 11 players.
# After all tracking is done, keep only the top N track IDs by frame count.
# This automatically discards short-lived ghost tracks caused by ID switches.
_MAX_DEFENSIVE_PLAYERS = 11


@dataclass
class PlayerTrack:
    track_id: int
    frame_indices: list[int] = field(default_factory=list)
    # Distance from player foot to ball center each frame (pixels)
    distances_to_target: list[float] = field(default_factory=list)
    foot_positions: list[tuple[float, float]] = field(default_factory=list)
    target_positions: list[Optional[tuple[float, float]]] = field(default_factory=list)


@dataclass
class PlayerEffortReport:
    track_id: int
    effort: bool          # True = Yes, False = No
    label: str            # "EFFORT ✓" | "NO EFFORT ✗"
    dist_start: float     # average distance in first third (px)
    dist_end: float       # average distance in last third (px)
    frame_count: int
    position_group: str = "Secondary"
    position_number: int = 0


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
        target_point: Optional[tuple[float, float]],
    ) -> None:
        tid = classified_player.detection.track_id
        if tid not in self._tracks:
            self._tracks[tid] = PlayerTrack(track_id=tid)

        foot = _foot_center(classified_player.detection.bbox)

        if target_point is not None:
            d = _dist(foot, target_point)
        else:
            d = float("nan")

        self._tracks[tid].frame_indices.append(frame_idx)
        self._tracks[tid].distances_to_target.append(d)
        self._tracks[tid].foot_positions.append(foot)
        self._tracks[tid].target_positions.append(target_point)

    def compute_reports(self) -> list[PlayerEffortReport]:
        """Return one PlayerEffortReport per tracked defensive player.

        Only the top _MAX_DEFENSIVE_PLAYERS track IDs (by frame count) are
        kept. Short-lived ghost tracks caused by ByteTrack ID switches are
        automatically discarded.
        """
        # Sort all tracks by how many frames they appear in (most = real player)
        sorted_tracks = sorted(
            self._tracks.values(),
            key=lambda t: len(t.frame_indices),
            reverse=True,
        )

        # Diagnostic only: show whether defensive tracks have enough usable
        # observations to reach the scoring stage. This does not change scoring.
        logger.info(
            "Scorer diagnostics | unique_defense_tracks=%d | top_tracks=%s",
            len(sorted_tracks),
            [
                {
                    "id": t.track_id,
                    "frames": len(t.frame_indices),
                    "ball_frames": sum(not np.isnan(d) for d in t.distances_to_target),
                }
                for t in sorted_tracks[:15]
            ],
        )

        # Keep only the top 11
        top_tracks = sorted_tracks[:_MAX_DEFENSIVE_PLAYERS]

        reports: list[PlayerEffortReport] = []
        for track in top_tracks:
            r = self._score_track(track)
            if r:
                reports.append(r)

        # Sort final list: effort first, then by most distance closed
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
        dists = [d for d in track.distances_to_target if not np.isnan(d)]
        if len(dists) < _MIN_FRAMES:
            return None

        n = len(dists)
        third = max(1, n // 3)

        dist_start = float(np.mean(dists[:third]))
        dist_end   = float(np.mean(dists[-third:]))

        # Primary signal: did the defender repeatedly move toward the target?
        # This handles a moving ball carrier/receiver better than a simple
        # start-vs-end distance comparison. A defender can pursue correctly
        # while the target is moving away, causing final distance to increase.
        closing_steps = 0
        meaningful_steps = 0
        for i in range(1, len(dists)):
            prev_target = track.target_positions[i - 1]
            curr_target = track.target_positions[i]
            prev_foot = track.foot_positions[i - 1]
            curr_foot = track.foot_positions[i]
            if prev_target is None or curr_target is None:
                continue

            target_vector = (
                prev_target[0] - prev_foot[0],
                prev_target[1] - prev_foot[1],
            )
            target_length = float(np.hypot(target_vector[0], target_vector[1]))
            if target_length < 1e-6:
                continue

            defender_move = (
                curr_foot[0] - prev_foot[0],
                curr_foot[1] - prev_foot[1],
            )
            movement = float(np.hypot(defender_move[0], defender_move[1]))
            if movement < 1.0:
                continue

            meaningful_steps += 1
            toward_target = (
                defender_move[0] * target_vector[0]
                + defender_move[1] * target_vector[1]
            ) / target_length
            if toward_target > 0.5:
                closing_steps += 1

        closing_ratio = (
            closing_steps / meaningful_steps if meaningful_steps else 0.0
        )

        # Sustained pursuit is now a primary signal. Clear distance closure
        # still receives credit. The 30% threshold avoids calling a player
        # effort based on only a few noisy tracking movements.
        effort = (
            closing_ratio >= 0.30
            or (dist_start - dist_end) > _CLOSE_THRESHOLD_PX
        )

        logger.debug(
            "Effort score | track=%s | start=%.1f | end=%.1f | closing_ratio=%.2f | effort=%s",
            track.track_id,
            dist_start,
            dist_end,
            closing_ratio,
            effort,
        )

        return PlayerEffortReport(
            track_id=track.track_id,
            effort=effort,
            label="EFFORT ✓" if effort else "NO EFFORT ✗",
            dist_start=round(dist_start, 1),
            dist_end=round(dist_end, 1),
            frame_count=len(track.frame_indices),
        )
