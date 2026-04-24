# configs/

Configuration files for the pipeline, data paths, and experiment tracking. All `src/` modules read from `pipeline.yaml` — no values are hardcoded in source files.

---

## `pipeline.yaml`

Single source of truth for all ML pipeline constants.

```yaml
detection:
  model: "db_resnet50"
  iou_threshold: 0.5
  dedup_iou_threshold: 0.5     # NMS threshold for overlapping boxes

recognition:
  model: "microsoft/trocr-base-printed"
  max_new_tokens: 32
  confidence_threshold: 0.75   # below this, text is silenced
  crop_padding: 4              # pixels of padding around each detected box
  max_crops: 12                # cap crops per image to bound worst-case latency

reading_order:
  row_tolerance: 0.6           # fraction of median box height for row grouping

latency:
  budget_s: 2.0                # hard latency requirement

tts:
  backend: "local"             # "local" (pyttsx3) | "cloud" (edge-tts)
  rate: 150                    # words per minute (pyttsx3 only)
  volume: 1.0                  # 0.0 → 1.0 (pyttsx3 only)
  lang: "en"                   # language code (gTTS only)
```

---

## `paths.yaml`

Directory layout for data, models, and outputs. Used by notebooks to resolve paths relative to the project root.

```yaml
paths:
  raw_data:          data/raw
  processed_data:    data/processed
  splits:            data/splits
  model_weights:     models/weights
  benchmark_results: models/benchmark_results
  figures:           outputs/figures
  results:           outputs/results
```

---

## `mlflow.env`

Environment variables for MLflow experiment tracking via DagsHub. Load this file before running notebooks that log experiments:

```bash
export $(cat configs/mlflow.env | xargs)
```

Experiment dashboard: [DagsHub MLflow](https://dagshub.com/MoCamaraData/vision-ocr-accessibility-assistant/experiments)
