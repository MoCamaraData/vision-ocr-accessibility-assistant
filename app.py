"""
app.py — OCR Accessibility Pipeline Demo
Gradio interface with three modes: upload, webcam snapshot, real-time capture.

Usage on Colab (VSCode extension or browser):
    1. Mount Drive
    2. Set PROJECT_ROOT to your Drive project folder
    3. Run: !python app.py
    The app launches with share=True — a public URL is printed.
"""

import io
import os
import sys
import tempfile
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

# ── Paths — update PROJECT_ROOT if different ──────────────────────────────────

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/content/drive/MyDrive/vision-ocr-accessibility-assistant"
))

MODEL_CACHE  = PROJECT_ROOT / "models/trocr"
FINETUNE_DIR = PROJECT_ROOT / "models/trocr-finetuned"
TMP_DIR      = Path(tempfile.gettempdir())

# Add project root to path so src/ imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

assert FINETUNE_DIR.exists(), (
    f"Fine-tuned model not found at {FINETUNE_DIR}. "
    "Run 06a first to generate it."
)

print(f"Project root : {PROJECT_ROOT}")
print(f"Fine-tuned   : {FINETUNE_DIR}")

# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline():
    import torch
    from src.pipeline import Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline on {device}...")

    pipe = Pipeline(
        device=device,
        tts_backend="cloud",
        model_cache_dir=str(MODEL_CACHE),
        recognition_checkpoint=str(FINETUNE_DIR),
    )
    print("Pipeline ready ✓")
    return pipe

pipe = load_pipeline()

# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(image: Image.Image):
    if image is None:
        return None, None, "No image provided."

    tmp_path = str(TMP_DIR / "input_image.jpg")
    image.save(tmp_path)

    result = pipe.run(tmp_path)

    # Annotated image
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    for r in result["gated_results"]:
        x, y, w, h = r["box"]
        color = "#27ae60" if not r["gated"] else "#e74c3c"
        rect  = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x, max(0, y - 4),
            f"{r['text']} ({r['confidence']:.2f})",
            fontsize=7, color=color, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none")
        )
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    annotated = Image.open(buf).convert("RGB")

    # Info text
    spoken   = [r["text"] for r in result["gated_results"] if not r["gated"]]
    silenced = [r["text"] for r in result["gated_results"] if r["gated"]]
    det_ms   = result["latency"]["detection_s"] * 1000
    rec_ms   = result["latency"]["recognition_s"] * 1000
    total_ms = det_ms + rec_ms
    budget   = "within budget" if total_ms <= 2000 else "EXCEEDS budget"

    info = (
        f"Spoken     : {spoken}\n"
        f"Silenced   : {silenced}\n"
        f"Regions    : {result['n_spoken']} spoken / {result['n_silenced']} silenced\n"
        f"Detection  : {det_ms:.1f}ms\n"
        f"Recognition: {rec_ms:.1f}ms\n"
        f"Total      : {total_ms:.1f}ms  ({budget})"
    )

    # Audio
    audio = None
    if result["tts_output"]:
        audio_path = str(TMP_DIR / "tts_output.mp3")
        with open(audio_path, "wb") as f:
            f.write(result["tts_output"])
        audio = audio_path

    return annotated, audio, info


def process_upload(image):
    return run_inference(image)


def process_webcam(image):
    if image is not None:
        import PIL.ImageOps
        image = PIL.ImageOps.mirror(image)
    return run_inference(image)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="OCR Accessibility Pipeline", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # OCR Accessibility Pipeline
        **Text detection, recognition, and speech synthesis for visually impaired users.**

        - Detection    : db_resnet50 (docTR)
        - Recognition  : TrOCR base-printed, fine-tuned on COCO-Text (CER 0.1998)
        - Confidence gate : 0.75 — low-confidence text is silenced
        - Green boxes  : spoken | Red boxes : silenced
        """
    )

    with gr.Tabs():

        with gr.Tab("Upload Image"):
            with gr.Row():
                with gr.Column():
                    upload_input  = gr.Image(type="pil", label="Upload an image")
                    upload_button = gr.Button("Run Pipeline", variant="primary")
                with gr.Column():
                    upload_output_img   = gr.Image(type="pil", label="Annotated Output")
                    upload_output_audio = gr.Audio(label="TTS Output", type="filepath")
                    upload_output_info  = gr.Textbox(label="Results", lines=7)

            upload_button.click(
                fn=process_upload,
                inputs=upload_input,
                outputs=[upload_output_img, upload_output_audio, upload_output_info]
            )

        with gr.Tab("Webcam Snapshot"):
            with gr.Row():
                with gr.Column():
                    webcam_input  = gr.Image(sources=["webcam"], type="pil",
                                             label="Capture from webcam")
                    webcam_button = gr.Button("Run Pipeline", variant="primary")
                with gr.Column():
                    webcam_output_img   = gr.Image(type="pil", label="Annotated Output")
                    webcam_output_audio = gr.Audio(label="TTS Output", type="filepath")
                    webcam_output_info  = gr.Textbox(label="Results", lines=7)

            webcam_button.click(
                fn=process_webcam,
                inputs=webcam_input,
                outputs=[webcam_output_img, webcam_output_audio, webcam_output_info]
            )

        with gr.Tab("Real-Time"):
            gr.Markdown(
                "Press **Capture & Process** to take a webcam frame and run the pipeline. "
                "Re-press to process a new frame."
            )
            with gr.Row():
                with gr.Column():
                    rt_input  = gr.Image(sources=["webcam"], type="pil",
                                         label="Webcam feed — click capture")
                    rt_button = gr.Button("Capture & Process", variant="primary")
                with gr.Column():
                    rt_output_img   = gr.Image(type="pil", label="Annotated Output")
                    rt_output_audio = gr.Audio(label="TTS Output", type="filepath")
                    rt_output_info  = gr.Textbox(label="Results", lines=7)

            rt_button.click(
                fn=process_webcam,
                inputs=rt_input,
                outputs=[rt_output_img, rt_output_audio, rt_output_info]
            )

    gr.Markdown(
        """
        ---
        Experiment tracking: [DagsHub MLflow](https://dagshub.com/MoCamaraData/vision-ocr-accessibility-assistant/experiments)
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
