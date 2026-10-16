"""
analyzer.py
-----------
Orchestrates football defensive effort analysis.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.detector import PlayerDetector
from app.effort_scorer import EffortScorer
from app.field_detector import FieldDetector
from app.team_classifier import TeamClassifier
from app.visualizer import Visualizer, draw_summary_overlay


def _foot_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


logger = logging.getLogger(__name__)



def _select_target_points(frame_store: list[dict]) -> list[tuple[float, float] | None]:
    """
    Select a stable ball-carrier/receiver target for every frame.

    Ball detections take priority. When the ball is missing, retain the last
    selected non-defensive track; if the clip never provides a ball-based
    target, use the most mobile non-defensive track as a best-effort fallback.
    """
    positions_by_track: dict[int, list[tuple[int, tuple[float, float], str]]] = {}
    for fd in frame_store:
        for cp in fd["classified"]:
            if cp.team == "defense":
                continue
            tid = cp.detection.track_id
            positions_by_track.setdefault(tid, []).append(
                (fd["frame_idx"], _foot_center(cp.detection.bbox), cp.team)
            )

    # Rank fallback candidates by movement, with a small preference for
    # tracks classified as offense. Unknown-team tracks remain eligible because
    # jersey/helmet classification can be unreliable in game footage.
    fallback_rank: list[tuple[float, int, int]] = []
    for tid, observations in positions_by_track.items():
        observations.sort(key=lambda item: item[0])
        if len(observations) < 3:
            continue
        movement = sum(
            _distance(observations[i - 1][1], observations[i][1])
            for i in range(1, len(observations))
            if observations[i][0] == observations[i - 1][0] + 1
        )
        offense_frames = sum(team == "offense" for _, _, team in observations)
        fallback_rank.append((movement + offense_frames * 2.0, len(observations), tid))

    fallback_track_id = (
        max(fallback_rank)[2] if fallback_rank else None
    )

    target_track_id: int | None = None
    last_target_point: tuple[float, float] | None = None
    selected_points: list[tuple[float, float] | None] = []
    ball_target_frames = 0
    fallback_target_frames = 0
    retained_target_frames = 0

    for fd in frame_store:
        classified = fd["classified"]
        ball_center = fd["ball_center"]
        candidates = [cp for cp in classified if cp.team != "defense"]
        by_track = {cp.detection.track_id: cp for cp in candidates}

        if ball_center is not None and candidates:
            # Prefer a confidently classified offensive player when similarly
            # close to the ball; otherwise choose the nearest non-defender.
            target = min(
                candidates,
                key=lambda cp: _distance(
                    _foot_center(cp.detection.bbox), ball_center
                ),
            )
            target_track_id = target.detection.track_id
            last_target_point = _foot_center(target.detection.bbox)
            ball_target_frames += 1
        elif target_track_id in by_track:
            last_target_point = _foot_center(
                by_track[target_track_id].detection.bbox
            )
            retained_target_frames += 1
        elif fallback_track_id in by_track:
            target_track_id = fallback_track_id
            last_target_point = _foot_center(
                by_track[fallback_track_id].detection.bbox
            )
            fallback_target_frames += 1
        # If the selected target is temporarily occluded, retain its last
        # known point rather than dropping all pursuit observations for a frame.

        if last_target_point is None and ball_center is not None:
            last_target_point = ball_center
        selected_points.append(last_target_point)

    logger.info(
        "Target selection diagnostics | ball_associated_frames=%d | "
        "retained_target_frames=%d | fallback_target_frames=%d | "
        "fallback_track_id=%s | usable_target_frames=%d/%d",
        ball_target_frames,
        retained_target_frames,
        fallback_target_frames,
        fallback_track_id,
        sum(point is not None for point in selected_points),
        len(frame_store),
    )
    return selected_points


def _detect_change_of_speed_flags(
    frame_store: list[dict],
    target_points: list[tuple[float, float] | None],
    fps: float,
    scored_track_ids: set[int],
) -> dict[int, list[dict]]:
    """Flag possible Rule 2 slowdowns for coach review; never deduct points."""
    observations: dict[int, list[tuple[int, tuple[float, float], tuple[float, float] | None]]] = {}
    for fd, target in zip(frame_store, target_points):
        for cp in fd["classified"]:
            tid = cp.detection.track_id
            if cp.team != "defense" or tid not in scored_track_ids:
                continue
            observations.setdefault(tid, []).append(
                (fd["frame_idx"], _foot_center(cp.detection.bbox), target)
            )

    flags: dict[int, list[dict]] = {}
    for tid, track_obs in observations.items():
        track_obs.sort(key=lambda item: item[0])
        steps: list[dict] = []
        for prev, curr in zip(track_obs, track_obs[1:]):
            prev_idx, prev_foot, prev_target = prev
            curr_idx, curr_foot, _ = curr
            if curr_idx != prev_idx + 1 or prev_target is None:
                steps.append({"frame": curr_idx, "speed": None, "pursuing": False,
                              "distance": None, "direction": None})
                continue

            dx, dy = curr_foot[0] - prev_foot[0], curr_foot[1] - prev_foot[1]
            speed = float((dx * dx + dy * dy) ** 0.5)
            tx, ty = prev_target[0] - prev_foot[0], prev_target[1] - prev_foot[1]
            target_len = float((tx * tx + ty * ty) ** 0.5)
            dot = dx * tx + dy * ty
            pursuing = target_len > 1e-6 and dot / target_len > 0.5
            direction = (dx / speed, dy / speed) if speed > 1.0 else None
            steps.append({"frame": curr_idx, "speed": speed, "pursuing": pursuing,
                          "distance": target_len, "direction": direction})

        for i, current in enumerate(steps):
            if current["speed"] is None or current["distance"] is None or current["distance"] < 20:
                continue
            prior = [
                step["speed"] for step in steps[max(0, i - 10):i]
                if step["speed"] is not None and step["pursuing"] and step["speed"] >= 3.0
            ]
            if len(prior) < 4:
                continue
            baseline = float(np.median(prior))
            slowdown_threshold = max(1.0, baseline * 0.45)
            if current["speed"] > slowdown_threshold:
                continue

            run = [current]
            for later in steps[i + 1:i + 3]:
                if (later["speed"] is None or later["distance"] is None
                        or later["distance"] < 20 or later["speed"] > slowdown_threshold):
                    break
                if later["direction"] is not None and current["direction"] is not None:
                    alignment = (later["direction"][0] * current["direction"][0]
                                 + later["direction"][1] * current["direction"][1])
                    if alignment < -0.25:
                        break
                run.append(later)
            if len(run) < 3:
                continue

            start_frame = run[0]["frame"]
            flags[tid] = [{
                "rule": 2,
                "title": "Possible change of speed during pursuit",
                "frame": start_frame,
                "time_seconds": round(start_frame / max(fps, 1.0), 1),
                "evidence": (
                    f"Tracked speed stayed at or below {slowdown_threshold:.1f} px/frame "
                    f"for {len(run)} consecutive frames after prior pursuit averaging "
                    f"{baseline:.1f} px/frame. Review the film for context."
                ),
                "status": "Needs coach review",
            }]
            break

    logger.info("Potential D-Loaf diagnostics | rule_2_flags=%d | scored_players=%d",
                len(flags), len(scored_track_ids))
    return flags


def _assign_defensive_groups(
    frame_store: list[dict],
    reports,
) -> dict[int, tuple[str, int]]:
    """Assign the 11 scored defenders to the user's 4-2-5 structure."""
    if not reports:
        return {}

    report_ids = {r.track_id for r in reports}
    sample_count = max(1, min(len(frame_store), max(30, len(frame_store) // 5)))

    defender_points: dict[int, list[tuple[float, float]]] = {
        tid: [] for tid in report_ids
    }
    offense_points: list[tuple[float, float]] = []

    for fd in frame_store[:sample_count]:
        for cp in fd["classified"]:
            foot = _foot_center(cp.detection.bbox)
            if cp.team == "defense" and cp.detection.track_id in defender_points:
                defender_points[cp.detection.track_id].append(foot)
            elif cp.team == "offense":
                offense_points.append(foot)

    if not offense_points:
        logger.warning("Position grouping: no early offense points; using y-depth fallback.")
        ordered = sorted(
            (
                (tid, sum(p[1] for p in pts) / len(pts))
                for tid, pts in defender_points.items()
                if pts
            ),
            key=lambda item: item[1],
        )
        ranked_ids = [tid for tid, _ in ordered]
    else:
        ox = sum(p[0] for p in offense_points) / len(offense_points)
        oy = sum(p[1] for p in offense_points) / len(offense_points)

        all_def_points = [p for pts in defender_points.values() for p in pts]
        if not all_def_points:
            return {}

        dx = ox - (sum(p[0] for p in all_def_points) / len(all_def_points))
        dy = oy - (sum(p[1] for p in all_def_points) / len(all_def_points))
        length = (dx * dx + dy * dy) ** 0.5

        if length < 1e-6:
            logger.warning("Position grouping: offense/defense centers overlap; grouping skipped.")
            return {}

        ux, uy = dx / length, dy / length
        projected = []
        for tid, pts in defender_points.items():
            if not pts:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            projection = (cx - ox) * ux + (cy - oy) * uy
            projected.append((tid, projection))

        ranked_ids = [
            tid for tid, _ in sorted(projected, key=lambda item: item[1], reverse=True)
        ]

    ranked_ids = ranked_ids[:11]

    group_map: dict[int, tuple[str, int]] = {}
    for index, tid in enumerate(ranked_ids):
        if index < 4:
            group_map[tid] = ("D Line", index + 1)
        elif index < 6:
            group_map[tid] = ("LB", index - 3)
        else:
            group_map[tid] = ("Secondary", index - 5)

    logger.info(
        "Defensive position groups | %s",
        {tid: group_map[tid] for tid in ranked_ids},
    )
    return group_map


def analyze_video(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    model_path: str = "yolov8m.pt",
    conf_threshold: float = 0.35,
    device: str = "cpu",
    job_id: str | None = None,
) -> dict:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if job_id is None:
        job_id = f"{int(time.time())}_{input_path.stem}"

    output_video_path = output_dir / f"{job_id}_annotated.mp4"
    output_json_path = output_dir / f"{job_id}_report.json"

    logger.info("Starting analysis | job=%s | input=%s", job_id, input_path)
    t0 = time.perf_counter()

    detector = PlayerDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
    )
    classifier = TeamClassifier()
    field_det = FieldDetector()
    scorer = EffortScorer()

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_store: list[dict] = []
    field_mask = None

    total_person_detections = 0
    total_on_field_persons = 0
    total_defense_classifications = 0
    total_offense_classifications = 0
    total_unknown_classifications = 0

    for frame_result in detector.process_video(input_path):
        frame = frame_result.frame

        if field_mask is None:
            field_mask = field_det.detect(frame)

        on_field_persons = [
            p for p in frame_result.persons
            if _foot_on_field(field_mask, p.bbox)
        ]

        classified = classifier.classify(frame, on_field_persons)
        ball_center = frame_result.ball_center()

        total_person_detections += len(frame_result.persons)
        total_on_field_persons += len(on_field_persons)
        total_defense_classifications += sum(cp.team == "defense" for cp in classified)
        total_offense_classifications += sum(cp.team == "offense" for cp in classified)
        total_unknown_classifications += sum(cp.team == "unknown" for cp in classified)

        frame_store.append({
            "frame_idx": frame_result.frame_idx,
            "frame": frame,
            "classified": classified,
            "balls": frame_result.balls,
            "ball_center": ball_center,
        })

    logger.info(
        "Pipeline diagnostics | YOLO persons=%d | on-field=%d | "
        "defense=%d | offense=%d | unknown=%d",
        total_person_detections,
        total_on_field_persons,
        total_defense_classifications,
        total_offense_classifications,
        total_unknown_classifications,
    )

    # Select targets after the full clip is available so the fallback can
    # use track movement across the entire play, even when the ball is never
    # detected by YOLO.
    target_points = _select_target_points(frame_store)
    for fd, target_point in zip(frame_store, target_points):
        for cp in fd["classified"]:
            if cp.team == "defense":
                scorer.update(fd["frame_idx"], cp, target_point)

    reports = scorer.compute_reports()

    potential_loaf_flags = _detect_change_of_speed_flags(
        frame_store, target_points, fps, {report.track_id for report in reports}
    )

    position_groups = _assign_defensive_groups(frame_store, reports)
    for report in reports:
        group, number = position_groups.get(report.track_id, ("Secondary", 0))
        report.position_group = group
        report.position_number = number

    effort_map = scorer.per_frame_effort()

    with Visualizer(output_video_path, fps, width, height) as viz:
        for fd in frame_store:
            fi = fd["frame_idx"]
            frame = fd["frame"]

            if fi == 0 or fi == len(frame_store) - 1 or fi % 30 == 0:
                frame = draw_summary_overlay(frame, reports)

            viz.annotate_and_write(
                frame,
                fd["classified"],
                fd["balls"],
                effort_map,
                field_mask,
            )

    report_data = {
        "job_id": job_id,
        "input_file": str(input_path),
        "output_video": str(output_video_path),
        "processing_time_seconds": round(time.perf_counter() - t0, 2),
        "player_reports": [
            {
                "track_id": r.track_id,
                "position_group": r.position_group,
                "position_number": r.position_number,
                "display_name": (
                    f"{r.position_group} {r.position_number}"
                    if r.position_number
                    else r.position_group
                ),
                "effort": r.effort,
                "label": r.label,
                "dist_start": r.dist_start,
                "dist_end": r.dist_end,
                "movement_px": r.movement_px,
                "avg_movement_px_per_frame": r.avg_movement_px_per_frame,
                "pursuit_ratio": r.pursuit_ratio,
                "net_pursuit_px": r.net_pursuit_px,
                "frame_count": r.frame_count,
                "potential_loaf_flags": potential_loaf_flags.get(r.track_id, []),
            }
            for r in reports
        ],
    }

    with open(output_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    report_data["output_json"] = str(output_json_path)
    logger.info(
        "Done in %.1fs — %d defenders scored",
        report_data["processing_time_seconds"],
        len(reports),
    )
    return report_data


def _foot_on_field(mask, bbox: tuple[int, int, int, int]) -> bool:
    """Check if the player's foot position (bottom-center) is on the field."""
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) // 2
    foot_y = y2
    return FieldDetector.is_on_field(mask, foot_x, foot_y)
