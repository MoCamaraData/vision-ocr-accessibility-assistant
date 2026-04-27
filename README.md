# VisionLens — OCR Accessibility Assistant

A real-time camera-to-voice reading assistant designed to help visually impaired users access printed text. Point a camera at signage, labels, or menus and the app will detect the text, recognize it, and speak it aloud — all within a 2-second latency budget.

---

## How It Works

```
Camera Frame
    └── Text Detection    (docTR — db_resnet50)
        └── Cropping + Deduplication + Reading Order
            └── Text Recognition  (TrOCR fine-tuned on COCO-Text, CER 0.1998)
                └── Confidence Gate  (threshold 0.75 — low-confidence text silenced)
                    └── Text-to-Speech  (Edge TTS — en-US-AriaNeural)
```

The pipeline is exposed as a **FastAPI backend** with HTTP and WebSocket endpoints, and consumed by a **React + Vite frontend** that handles live camera streaming, image upload, and audio playback.

---

## Project Structure

```
vision-ocr-accessibility-assistant/
├── app/
│   ├── backend/          FastAPI server (OCR endpoints + WebSocket stream)
│   └── frontend/         React + Vite UI (live camera, upload, i18n)
├── src/                  Core ML pipeline (detection, recognition, TTS, postprocessing)
│   ├── detection/        docTR db_resnet50 text detector
│   ├── recognition/      TrOCR processor + model
│   ├── postprocessing/   Cropper, reading order sorter, IoU deduplicator
│   ├── tts/              Speaker (local pyttsx3 / cloud edge-tts)
│   └── utils/            Shared metrics (IoU, CER, WER)
├── configs/              YAML configuration and environment files
├── notebooks/            Jupyter notebooks (EDA → fine-tuning → demo)
├── app.py                Gradio demo (HuggingFace Spaces entry point)
└── Dockerfile            Production image (bakes model weights at build time)
```

---

## Quick Start

### Docker (production)

```bash
docker build -t visionlens .
docker run -p 7860:7860 visionlens
```

The image downloads and bakes TrOCR weights from HuggingFace at build time via the `HF_REPO` env variable (`MoCamaraData/trocr-ocr-accessibility`).

### Local development

**Backend**

```bash
cd app/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend**

```bash
cd app/frontend
npm install
npm run dev
```

Set `VITE_API_URL` in `app/frontend/.env` to point at the backend if it is not running on the default HuggingFace Space URL.

---

## Key Configuration

All pipeline constants live in [`configs/pipeline.yaml`](configs/pipeline.yaml). Nothing is hardcoded in source files.

| Parameter | Value | Notes |
|---|---|---|
| Detection model | `db_resnet50` | docTR pretrained |
| Recognition model | `microsoft/trocr-base-printed` | fine-tuned checkpoint on Hub |
| Confidence threshold | `0.75` | below this, text is silenced |
| Max crops per image | `12` | bounds worst-case latency |
| Latency budget | `2.0s` | hard requirement |
| TTS voice | `en-US-AriaNeural` | via edge-tts |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns model and device info |
| `POST` | `/ocr/image` | Multipart image upload — returns text, boxes, audio (base64) |
| `POST` | `/ocr/image/base64` | Base64 image payload — same response schema |
| `WS` | `/ocr/stream` | WebSocket — continuous webcam frame processing |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Text detection | [docTR](https://github.com/mindee/doctr) — `db_resnet50` |
| Text recognition | [TrOCR](https://huggingface.co/microsoft/trocr-base-printed) — fine-tuned |
| Text-to-speech | [edge-tts](https://github.com/rany2/edge-tts) — `en-US-AriaNeural` |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Frontend | [React 18](https://react.dev/) + [Vite 5](https://vitejs.dev/) |
| Demo (HF Spaces) | [Gradio](https://gradio.app/) |
| Experiment tracking | [DagsHub MLflow](https://dagshub.com/MoCamaraData/vision-ocr-accessibility-assistant/experiments) |

---

## Model

The fine-tuned TrOCR model is hosted publicly on HuggingFace:
[MoCamaraData/trocr-ocr-accessibility](https://huggingface.co/MoCamaraData/trocr-ocr-accessibility)

- Base: `microsoft/trocr-base-printed`
- Fine-tuned on: COCO-Text dataset
- Character Error Rate (CER): **0.1998**

---

## Capstone Project — La Cité Collégiale, 2026
