"""
effort_scorer.py
----------------
Determines whether each blue defensive player made an EFFORT to pursue
the ball / ball carrier.

Definition of effort (Yes / No):
  - The question is NOT whether the defender ended closer to the ball.
  - The question is whether the defender meaningfully pursued the ball.
  - Sustained movement toward the ball / ball carrier is the primary signal.
  - Start/end distance is supporting evidence only.
  - A moving receiver or ball carrier can increase final distance even when
    the defender is pursuing correctly.

Football grading intent:
  YES = runs/pursues to the ball.
  NO  = stands, watches, walks, or otherwise does not meaningfully pursue.

Ball movement is NOT required — effort is judged from defender movement
toward the current target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MIN_FRAMES = 15
_CLOSE_THRESHOLD_PX = 5.0
_MAX_DEFENSIVE_PLAYERS = 11

# A defender must show repeated directional movement toward the target.
# This is intentionally the primary effort test; final distance is not.
_MIN_PURSUIT_RATIO = 0.30


@dataclass
class PlayerTrack:
    track_id: int
    frame_indices: list[int] = field(default_factory=list)
    distances_to_target: list[float] = field(default_factory=list)
    foot_positions: list[tuple[float, float]] = field(default_factory=list)
    target_positions: list[Optional[tuple[float, float]]] = field(default_factory=list)


@dataclass
class PlayerEffortReport:
    track_id: int
    effort: bool
    label: str
    dist_start: float
    dist_end: float
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

    Effort means pursuit toward the ball / ball carrier, not simply getting
    closer in absolute distance.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, PlayerTrack] = {}

    def update(
        self,
        frame_idx: int,
        classified_player,
        target_point: Optional[tuple[float, float]],
    ) -> None:
        tid = classified_player.detection.track_id
        if tid not in self._tracks:
            self._tracks[tid] = PlayerTrack(track_id=tid)

        foot = _foot_center(classified_player.detection.bbox)
        d = _dist(foot, target_point) if target_point is not None else float("nan")

        track = self._tracks[tid]
        track.frame_indices.append(frame_idx)
        track.distances_to_target.append(d)
        track.foot_positions.append(foot)
        track.target_positions.append(target_point)

    def compute_reports(self) -> list[PlayerEffortReport]:
        """Return one effort report per tracked defensive player."""
        sorted_tracks = sorted(
            self._tracks.values(),
            key=lambda t: len(t.frame_indices),
            reverse=True,
        )

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

        reports: list[PlayerEffortReport] = []
        for track in sorted_tracks[:_MAX_DEFENSIVE_PLAYERS]:
            report = self._score_track(track)
            if report:
                reports.append(report)

        return sorted(reports, key=lambda r: (not r.effort, r.dist_end - r.dist_start))

    def per_frame_effort(self) -> dict[int, bool | None]:
        result: dict[int, bool | None] = {}
        for report in self.compute_reports():
            result[report.track_id] = report.effort
        return result

    def _score_track(self, track: PlayerTrack) -> Optional[PlayerEffortReport]:
        # Keep distance, defender position, and target position aligned.
        # Previously dists removed NaNs independently, which could shift the
        # indexes used for the pursuit-direction calculation.
        observations = [
            (distance, foot, target)
            for distance, foot, target in zip(
                track.distances_to_target,
                track.foot_positions,
                track.target_positions,
            )
            if not np.isnan(distance) and target is not None
        ]

        if len(observations) < _MIN_FRAMES:
            return None

        dists = [item[0] for item in observations]
        feet = [item[1] for item in observations]
        targets = [item[2] for item in observations]

        n = len(dists)
        third = max(1, n // 3)
        dist_start = float(np.mean(dists[:third]))
        dist_end = float(np.mean(dists[-third:]))

        # Primary signal:
        # Did the defender repeatedly move in the direction of the ball?
        #
        # This intentionally does NOT require the defender's final distance
        # to be smaller. A receiver/ball carrier can outrun a defender while
        # the defender is still giving full pursuit effort.
        pursuit_steps = 0
        meaningful_steps = 0
        pursuit_distance = 0.0
        away_distance = 0.0

        for i in range(1, len(feet)):
            prev_foot = feet[i - 1]
            curr_foot = feet[i]
            target = targets[i - 1]

            target_vector = (
                target[0] - prev_foot[0],
                target[1] - prev_foot[1],
            )
            target_length = float(np.hypot(target_vector[0], target_vector[1]))
            if target_length < 1e-6:
                continue

            defender_move = (
                curr_foot[0] - prev_foot[0],
                curr_foot[1] - prev_foot[1],
            )
            movement = float(np.hypot(defender_move[0], defender_move[1]))

            # Ignore detector jitter / stationary players.
            if movement < 1.0:
                continue

            meaningful_steps += 1
            projection = (
                defender_move[0] * target_vector[0]
                + defender_move[1] * target_vector[1]
            ) / target_length

            if projection > 0.5:
                pursuit_steps += 1
                pursuit_distance += projection
            elif projection < -0.5:
                away_distance += abs(projection)

        pursuit_ratio = (
            pursuit_steps / meaningful_steps if meaningful_steps else 0.0
        )

        # Pursuit direction is the football grading signal. Distance closure
        # is only a supporting signal for cases where the target is relatively
        # stationary. A defender who pursues a moving ball carrier can still
        # finish farther away and should receive effort credit.
        net_pursuit = pursuit_distance - away_distance
        effort = (
            meaningful_steps > 0
            and pursuit_ratio >= _MIN_PURSUIT_RATIO
            and net_pursuit > 0
        )

        # Preserve clear closure as a secondary signal, but do not let it
        # override the pursuit test for a defender who is moving away from
        # the ball.
        if (
            (dist_start - dist_end) > _CLOSE_THRESHOLD_PX
            and pursuit_ratio >= 0.20
            and net_pursuit > 0
        ):
            effort = True

        logger.debug(
            "Effort score | track=%s | start=%.1f | end=%.1f | "
            "pursuit_ratio=%.2f | pursuit_px=%.1f | away_px=%.1f | "
            "net_pursuit=%.1f | effort=%s",
            track.track_id,
            dist_start,
            dist_end,
            pursuit_ratio,
            pursuit_distance,
            away_distance,
            net_pursuit,
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
