# Football AI — Defensive Effort Analyzer

> Upload a football play video and let AI score each **blue defensive player's effort** toward the ball — powered by **YOLOv8 + ByteTrack**.

---

## How It Works

```
Video Upload → YOLOv8 Detection → ByteTrack → Jersey Color (K-Means HSV)
    → Blue = Defense → Speed + Direction Analysis → Effort Score (0-100)
    → Annotated Video + JSON Report
```

### Effort Score Formula
| Metric | Weight | Description |
|--------|--------|-------------|
| Speed | 50% | Smoothed pixel displacement per frame |
| Ball Alignment | 50% | Cosine similarity between velocity vector and ball direction |

**Grades:** 🟢 80-100 High · 🟡 50-79 Moderate · 🔴 0-49 Low

---

## Quick Start (Docker — recommended)

```bash
docker build -t football-ai .
docker run -p 8000:8000 football-ai
```

Open **http://localhost:8000** — drag in a `.mp4` clip and click **Analyze Play**.

---

## Quick Start (Local Python)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000**

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload video (multipart/form-data, field `file`) |
| `GET`  | `/result/{job_id}` | Poll for status / get JSON report |
| `GET`  | `/video/{job_id}` | Stream annotated output video |
| `GET`  | `/health` | Liveness check |

### Example (curl)
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@my_play.mp4"
# → {"job_id": "abc-123", "status": "queued"}

curl http://localhost:8000/result/abc-123
# → {"status": "done", "player_reports": [...], "video_url": "/video/abc-123"}
```

---

## Configuration

| Parameter | Default | Where |
|-----------|---------|-------|
| YOLO model | `yolov8m.pt` | `analyzer.py` `model_path` |
| Confidence threshold | `0.35` | `analyzer.py` `conf_threshold` |
| Blue hue range (OpenCV H) | `90–130` | `team_classifier.py` |
| Max speed (px/frame) | `25` | `effort_scorer.py` |
| Device | `cpu` | `analyzer.py` `device` — set to `"0"` for GPU |

---

## Output

- **Annotated video** — bounding boxes colored by team, effort score + bar overlay per defender
- **JSON report** — per-player `track_id`, `effort_score`, `grade`, `avg_speed_px_per_frame`, `avg_alignment`

---

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection & tracking
- OpenCV — video I/O & color analysis
- scikit-learn — K-Means jersey classification
- FastAPI — REST API
- Docker — packaging
