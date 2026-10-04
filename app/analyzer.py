"""
analyzer.py
-----------
Orchestrates the full pipeline:
  1. Detect + track players/ball (YOLOv8 + ByteTrack)
  2. Detect the green field boundary — only players inside it count
  3. Classify teams: blue jersey/helmet → defense; white → offense (ignored)
  4. Score each defender: did they close distance to the ball? → Yes / No
  5. Render annotated output video (defenders only, Yes/No labels)
  6. Return JSON report
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import cv2

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

def _assign_defensive_groups(
    frame_store: list[dict],
    reports,
) -> dict[int, tuple[str, int]]:
    """
    Assign the 11 scored defenders to the user's 4-2-5 structure.

    Uses early-play defender depth relative to the offense instead of
    exposing ByteTrack IDs to the coach:
      - 4 closest to the offense = D Line
      - next 2 = LB
      - remaining 5 = Secondary
    """
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
        logger.warning(
            "Position grouping: no early offense points; using y-depth fallback."
        )
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
            logger.warning(
                "Position grouping: offense/defense centers overlap; grouping skipped."
            )
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
            tid
            for tid, _ in sorted(
                projected, key=lambda item: item[1], reverse=True
            )
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
    """
    Run the full analysis pipeline on a football play video.

    Returns a dict with job metadata, player effort reports, and output paths.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if job_id is None:
        job_id = f"{int(time.time())}_{input_path.stem}"

    output_video_path = output_dir / f"{job_id}_annotated.mp4"
    output_json_path  = output_dir / f"{job_id}_report.json"

    logger.info("Starting analysis | job=%s | input=%s", job_id, input_path)
    t0 = time.perf_counter()

    # ── components ─────────────────────────────────────────────────────
    detector   = PlayerDetector(model_path=model_path, conf_threshold=conf_threshold, device=device)
    classifier = TeamClassifier()
    field_det  = FieldDetector()
    scorer     = EffortScorer()
    target_track_id = None

    # ── video properties ───────────────────────────────────────────────
    cap    = cv2.VideoCapture(str(input_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ── first pass: detect, classify, accumulate track data ────────────
    frame_store: list[dict] = []
    field_mask = None   # computed once from first frame

    # Diagnostic counters — these report what each pipeline stage sees
    # without changing detection, classification, or scoring behavior.
    total_person_detections = 0
    total_on_field_persons = 0
    total_defense_classifications = 0
    total_offense_classifications = 0
    total_unknown_classifications = 0

    for frame_result in detector.process_video(input_path):
        frame = frame_result.frame

        # Compute field mask once (assumes fixed camera)
        if field_mask is None:
            field_mask = field_det.detect(frame)

        # Keep only players whose feet land inside the field
        on_field_persons = [
            p for p in frame_result.persons
            if _foot_on_field(field_mask, p.bbox)
        ]

        classified  = classifier.classify(frame, on_field_persons)
        ball_center = frame_result.ball_center()

        # The football is the best target when visible. When it is not
        # visible, keep following the non-defensive player last associated
        # with the football. This makes a receiver/ball carrier the stable
        # target after a catch instead of requiring continuous ball detection.
        target_point = ball_center
        if ball_center is not None:
            candidates = [
                cp for cp in classified
                if cp.team != "defense"
            ]
            if candidates:
                target = min(
                    candidates,
                    key=lambda cp: _distance(
                        _foot_center(cp.detection.bbox), ball_center
                    ),
                )
                target_track_id = target.detection.track_id
                target_point = _foot_center(target.detection.bbox)
        elif target_track_id is not None:
            target = next(
                (
                    cp for cp in classified
                    if cp.detection.track_id == target_track_id
                    and cp.team != "defense"
                ),
                None,
            )
            if target is not None:
                target_point = _foot_center(target.detection.bbox)

        total_person_detections += len(frame_result.persons)
        total_on_field_persons += len(on_field_persons)
        total_defense_classifications += sum(cp.team == "defense" for cp in classified)
        total_offense_classifications += sum(cp.team == "offense" for cp in classified)
        total_unknown_classifications += sum(cp.team == "unknown" for cp in classified)

        for cp in classified:
            if cp.team == "defense":
                scorer.update(frame_result.frame_idx, cp, target_point)

        frame_store.append({
            "frame_idx":  frame_result.frame_idx,
            "frame":      frame,
            "classified": classified,
            "balls":      frame_result.balls,
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

    # ── compute final effort reports ───────────────────────────────────
    reports = scorer.compute_reports()

    # Replace raw ByteTrack IDs with the user's 4-2-5 defensive structure.
    position_groups = _assign_defensive_groups(frame_store, reports)
    for report in reports:
        group, number = position_groups.get(report.track_id, ("Defense", 0))
        report.position_group = group
        report.position_number = number

    effort_map = scorer.per_frame_effort()

    # ── second pass: render annotated video ────────────────────────────
    with Visualizer(output_video_path, fps, width, height) as viz:
        for fd in frame_store:
            fi    = fd["frame_idx"]
            frame = fd["frame"]

            # Leaderboard on first, last, and every 30th frame
            if fi == 0 or fi == len(frame_store) - 1 or fi % 30 == 0:
                frame = draw_summary_overlay(frame, reports)

            viz.annotate_and_write(
                frame,
                fd["classified"],
                fd["balls"],
                effort_map,
                field_mask,
            )

    # ── JSON report ────────────────────────────────────────────────────
    report_data = {
        "job_id":                    job_id,
        "input_file":                str(input_path),
        "output_video":              str(output_video_path),
        "processing_time_seconds":   round(time.perf_counter() - t0, 2),
        "player_reports": [
            {
                "track_id":    r.track_id,
                "position_group": r.position_group,
                "position_number": r.position_number,
                "display_name": (
                    f"{r.position_group} {r.position_number}"
                    if r.position_number
                    else r.position_group
                ),
                "effort":      r.effort,
                "label":       r.label,
                "dist_start":  r.dist_start,
                "dist_end":    r.dist_end,
                "frame_count": r.frame_count,
            }
            for r in reports
        ],
    }

    with open(output_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    report_data["output_json"] = str(output_json_path)
    logger.info(
        "Done in %.1fs — %d defenders scored",
        report_data["processing_time_seconds"], len(reports)
    )
    return report_data


# ── helper ─────────────────────────────────────────────────────────────

def _foot_on_field(mask, bbox: tuple[int, int, int, int]) -> bool:
    """Check if the player's foot position (bottom-center) is on the field."""
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) // 2
    foot_y = y2
    return FieldDetector.is_on_field(mask, foot_x, foot_y)
