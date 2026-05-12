# VisionLens — OCR Accessibility Assistant

> Point a camera at a sign, label, or menu. The assistant detects the text, recognizes it, and speaks it aloud — all within a 2-second latency budget.

Real-time camera-to-voice reading aid for visually impaired users. A fine-tuned TrOCR model reaches **CER 0.1998** on COCO-Text. The full pipeline — detection, postprocessing, recognition, confidence gating, and TTS — runs end-to-end in under 2 seconds.

---

## How it works

```
Camera Frame
    │
    ├─ detection/       docTR db_resnet50 — returns bounding boxes
    │
    ├─ postprocessing/  IoU deduplication → reading order sort → crop
    │
    ├─ recognition/     TrOCR fine-tuned on COCO-Text (CER 0.1998)
    │
    ├─ confidence gate  Silences results below 0.75 — avoids garbled speech
    │
    └─ TTS              edge-tts (cloud) or pyttsx3 (local)
```

The pipeline is configured entirely from `configs/pipeline.yaml` — nothing is hardcoded in source files.

---

## Results

| Metric | Value |
|---|---:|
| CER — fine-tuned TrOCR | **0.1998** |
| Confidence gate threshold | 0.75 |
| Latency budget | 2.0 s |
| Max crops per image | 12 |

Fine-tuning on COCO-Text delivers the biggest quality gain. The confidence gate and IoU deduplication together eliminate the main sources of garbled or repeated speech.

---

## Three ways to use it

**Gradio demo (HuggingFace Spaces)**
Upload an image, take a webcam snapshot, or stream in real time. The pipeline annotates detections with green (spoken) and red (silenced) boxes.

**Web app (FastAPI + React)**
Production-grade interface with a live camera tab (WebSocket), image upload tab (HTTP), and bilingual UI (English / French). See [Quick start](#quick-start) below.

**Research notebooks**
Full development trail in `notebooks/` — EDA, detection benchmark, recognition benchmark, fine-tuning, and ablation studies. Start from `01_dataset_engineering_and_eda.ipynb`.

---

## Repo structure

```
vision-ocr-accessibility-assistant/
├── src/                    Core ML pipeline (framework-agnostic)
│   ├── pipeline.py         Pipeline class — loads all models, runs pipe.run(image)
│   ├── confidence.py       Confidence gate — silences text below threshold
│   ├── detection/          docTR db_resnet50 wrapper
│   ├── recognition/        TrOCR processor + model wrapper
│   ├── postprocessing/     IoU deduplication, reading order, cropper
│   ├── tts/                Speaker — pyttsx3 (local) or edge-tts (cloud)
│   └── utils/              Shared metrics: IoU, CER, WER
├── app/
│   ├── backend/            FastAPI server — HTTP + WebSocket endpoints
│   └── frontend/           React + Vite UI — camera, upload, audio playback
├── app.py                  Gradio demo (HuggingFace Spaces entry point)
├── notebooks/              Research notebooks — EDA through fine-tuning
├── scripts/                Stratified sample extraction utilities
├── configs/
│   ├── pipeline.yaml       All pipeline hyperparameters
│   └── paths.yaml          Dataset and output paths
├── docs/                   Extended project documentation
├── Dockerfile              Production image — bakes TrOCR weights at build time
└── docker-compose.yml      Single-service backend stack
```

---

## Quick start

### Docker (production)

```bash
docker build -t visionlens .
docker run -p 8000:8000 visionlens
```

The image downloads and bakes TrOCR weights from HuggingFace at build time via the `HF_REPO` env var (`MoCamaraData/trocr-ocr-accessibility`).

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

Set `VITE_API_URL` in `app/frontend/.env` to point at the backend if it is not running on the default URL.

---

## Key configuration

All pipeline constants live in [`configs/pipeline.yaml`](configs/pipeline.yaml). Nothing is hardcoded in source files.

| Parameter | Default | Notes |
|---|---|---|
| Detection model | `db_resnet50` | docTR pretrained |
| Recognition checkpoint | `microsoft/trocr-base-printed` | overridden by `HF_REPO` in prod |
| Confidence threshold | `0.75` | validated in Phase 5 |
| Max crops per image | `12` | caps worst-case latency |
| Latency budget | `2.0 s` | hard requirement from Phase 0 |
| TTS backend | `local` | `"local"` or `"cloud"` |

---

## Model

The fine-tuned TrOCR model is hosted publicly on HuggingFace Hub:
[MoCamaraData/trocr-ocr-accessibility](https://huggingface.co/MoCamaraData/trocr-ocr-accessibility)

- Base: `microsoft/trocr-base-printed`
- Fine-tuned on: COCO-Text dataset
- CER: **0.1998**

---

## Capstone project — La Cité Collégiale, 2026

---

## Tech stack

PyTorch · docTR · Transformers (TrOCR) · edge-tts · FastAPI · Uvicorn · React 18 · Vite · Gradio · DagsHub MLflow
