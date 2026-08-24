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

logger = logging.getLogger(__name__)


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

    # ── video properties ───────────────────────────────────────────────
    cap    = cv2.VideoCapture(str(input_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ── first pass: detect, classify, accumulate track data ────────────
    frame_store: list[dict] = []
    field_mask = None   # computed once from first frame

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

        for cp in classified:
            if cp.team == "defense":
                scorer.update(frame_result.frame_idx, cp, ball_center)

        frame_store.append({
            "frame_idx":  frame_result.frame_idx,
            "frame":      frame,
            "classified": classified,
            "balls":      frame_result.balls,
        })

    # ── compute final effort reports ───────────────────────────────────
    reports    = scorer.compute_reports()
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
