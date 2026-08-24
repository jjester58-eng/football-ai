"""
analyzer.py
-----------
Orchestrates the full pipeline:
  1. Detect + track players/ball with YOLOv8 + ByteTrack
  2. Classify teams by jersey color (blue → defense)
  3. Score defensive player effort (speed + direction toward ball)
  4. Render annotated output video
  5. Return structured JSON report
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

    Returns
    -------
    dict with keys:
      - job_id
      - input_file
      - output_video (path)
      - output_json (path)
      - player_reports (list of per-player effort dicts)
      - processing_time_seconds
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if job_id is None:
        job_id = f"{int(time.time())}_{input_path.stem}"

    output_video_path = output_dir / f"{job_id}_annotated.mp4"
    output_json_path = output_dir / f"{job_id}_report.json"

    logger.info("Starting analysis | job_id=%s | input=%s", job_id, input_path)
    t0 = time.perf_counter()

    # ── initialise components ──────────────────────────────────────────
    detector = PlayerDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
    )
    classifier = TeamClassifier()
    scorer = EffortScorer()

    # ── first pass: collect all frames & build tracks ──────────────────
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Store per-frame data for the second (render) pass
    frame_store: list[dict] = []

    for frame_result in detector.process_video(input_path):
        classified = classifier.classify(frame_result.frame, frame_result.persons)
        ball_center = frame_result.ball_center()

        for cp in classified:
            if cp.team == "defense":
                scorer.update(frame_result.frame_idx, cp, ball_center)

        frame_store.append(
            {
                "frame_idx": frame_result.frame_idx,
                "frame": frame_result.frame,
                "classified": classified,
                "balls": frame_result.balls,
            }
        )

    # ── compute final effort reports ───────────────────────────────────
    reports = scorer.compute_reports()
    per_frame_scores = scorer.per_frame_scores()

    # ── second pass: render annotated video ────────────────────────────
    with Visualizer(output_video_path, fps, width, height) as viz:
        for fd in frame_store:
            fi = fd["frame_idx"]
            frame = fd["frame"]
            classified = fd["classified"]
            balls = fd["balls"]

            # Build {track_id → live score} for this frame
            live_scores = {
                tid: score
                for tid, score in per_frame_scores.get(fi, [])
            }

            # Add summary leaderboard on final frame or every N frames
            if fi == len(frame_store) - 1 or fi % 30 == 0:
                frame = draw_summary_overlay(frame, reports)

            viz.annotate_and_write(frame, classified, balls, live_scores)

    # ── build JSON report ──────────────────────────────────────────────
    report_data = {
        "job_id": job_id,
        "input_file": str(input_path),
        "output_video": str(output_video_path),
        "processing_time_seconds": round(time.perf_counter() - t0, 2),
        "player_reports": [asdict(r) for r in reports],
    }

    with open(output_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    report_data["output_json"] = str(output_json_path)
    logger.info(
        "Analysis complete in %.1fs | %d defensive players scored",
        report_data["processing_time_seconds"],
        len(reports),
    )
    return report_data
