# app/backend/

FastAPI server that wraps the OCR pipeline and exposes it over HTTP and WebSocket.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok", "model": ..., "device": ...}` |
| `POST` | `/ocr/image` | Multipart file upload — full pipeline on a single image |
| `POST` | `/ocr/image/base64` | JSON body `{"image": "<base64>"}` — same pipeline |
| `WS` | `/ocr/stream` | WebSocket — accepts raw base64 frames, streams results back |

### Response schema (HTTP endpoints)

```json
{
  "text":     "PUSH DOOR",
  "tokens":   ["PUSH", "DOOR"],
  "boxes":    [[x, y, w, h], ...],
  "feedback": "",
  "audio":    "<base64 MP3>",
  "meta": {
    "n_spoken":   2,
    "n_silenced": 1,
    "latency":    {"detection_s": 0.15, "recognition_s": 0.09, "total_s": 0.25},
    "streaming":  false
  }
}
```

### WebSocket message types

The stream endpoint sends three types of JSON messages:

- `"text"` — new text detected (audio synthesis is in progress)
- `"audio"` — audio ready (base64 MP3 attached)
- `"feedback"` — no readable text, or text not clear enough

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `HF_REPO` | `MoCamaraData/trocr-ocr-accessibility` | HuggingFace model repo |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `PROJECT_ROOT` | two levels up from `main.py` | root path for `src/` imports |

---

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Run from the project root so that `src/` is on the Python path.

---

## Key implementation details

- **Pipeline is loaded once at startup** via the `@app.on_event("startup")` hook — not per request.
- **WebSocket frame rate** is throttled to one frame every 300 ms (`FRAME_INTERVAL_S = 0.30`).
- **Duplicate suppression** — the stream endpoint tracks the last spoken text and only triggers TTS when the new text differs by more than 20% (`SIMILARITY_THRESH = 0.80`).
- **TTS feedback cooldown** — feedback messages (`"No readable text detected"`) are suppressed for 2 seconds between sends to avoid flooding the client.
- **Token assembly** — glued words are split using `wordninja` (`"exithere"` → `"exit here"`) before being passed to edge-tts.

---

## Dependencies (`requirements.txt`)

| Package | Purpose |
|---|---|
| `fastapi` / `uvicorn` | Web server |
| `pillow` | Image loading and preprocessing |
| `transformers` | TrOCR processor and model |
| `torch` | PyTorch inference backend |
| `python-doctr[torch]` | DBNet text detector |
| `edge-tts` | Cloud text-to-speech |
| `wordninja` | Word boundary splitting |
| `pyyaml` | Config loading |
| `huggingface-hub` | Model download |
| `editdistance` | Similarity scoring |
