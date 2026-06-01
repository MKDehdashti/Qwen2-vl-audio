"""
inference.py — Inference utilities for Qwen2VLAudioForConditionalGeneration.

Covers three notebook cells:
  cell-53: core run_inference() helper
  cell-55: run_vl_inference() — VL sanity check (image + text prompt)
  cell-57: run_asr_inference() — ASR on a dataset sample before/after fine-tuning

Usage (standalone):
    source /workspace/projects/speech/.venv/bin/activate
    python /workspace/projects/speech/src/inference.py
"""

import sys
import os

sys.path.insert(0, "/workspace/projects/speech/transformers/src")
sys.path.insert(0, "/workspace/projects/speech/qwen-vl-utils/src")

import torch
from transformers import Qwen2VLProcessor
from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAudioForConditionalGeneration
from qwen_vl_utils import process_vision_info, process_audio_info

from data_utils import format_data

# ---------------------------------------------------------------------------
# Model / processor loading
# ---------------------------------------------------------------------------

MODEL_REPO = "MayaKD/qwen2-vl-audio"
PROCESSOR_DIR = "/workspace/projects/speech/processor"


def load_model_and_processor(
    model_path: str = MODEL_REPO,
    processor_path: str = PROCESSOR_DIR,
    device: str | None = None,
    torch_dtype=torch.bfloat16,
):
    """Load Qwen2VLAudioForConditionalGeneration and its processor.

    Args:
        model_path: Local directory or HF repo id.
        processor_path: Local directory or HF repo id for the processor.
        device: e.g. "cuda", "cpu". Defaults to "cuda" if available.
        torch_dtype: Weight dtype. bfloat16 recommended for A100/A6000.

    Returns:
        (model, processor) both ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor from {processor_path} …")
    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()  # attach manually

    print(f"Loading model from {model_path} …")
    model = Qwen2VLAudioForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    print("Model and processor loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    model: Qwen2VLAudioForConditionalGeneration,
    processor: Qwen2VLProcessor,
    messages: list[dict],
    device: str | None = None,
    max_new_tokens: int = 256,
) -> str:
    """Run a single forward + generate pass given an OpenAI-format message list.

    Args:
        model: The loaded Qwen2VLAudioForConditionalGeneration instance.
        processor: Matching Qwen2VLProcessor (with audio_processor attached).
        messages: OpenAI-format conversation, e.g. from format_data().
        device: Target device. Inferred from model parameters if None.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Decoded output string (assistant reply only, special tokens stripped).
    """
    if device is None:
        device = next(model.parameters()).device

    # 1. Apply chat template → raw text string
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2. Extract multimodal inputs
    image_inputs, video_inputs = process_vision_info(messages)
    audio_inputs = process_audio_info(messages)  # list of (np.ndarray, sr) or empty

    # 3. Tokenise + build model inputs
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        audios=audio_inputs if audio_inputs else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # 4. Generate
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 5. Trim prompt tokens, decode only the new tokens
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)
    return output_text[0]


# ---------------------------------------------------------------------------
# cell-55: VL sanity check
# ---------------------------------------------------------------------------

def run_vl_inference(
    model: Qwen2VLAudioForConditionalGeneration,
    processor: Qwen2VLProcessor,
    image_url: str,
    prompt: str,
    device: str | None = None,
    max_new_tokens: int = 256,
) -> str:
    """Run a vision-language inference (no audio) to verify VL capability is intact.

    Example:
        output = run_vl_inference(
            model, processor,
            image_url="https://t4.ftcdn.net/jpg/01/57/82/05/360_F_157820583_agejYX5XeczPZuWRSCDF2YYeCGwJqUdG.jpg",
            prompt="Detect the bounding box of the red car.",
        )
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return run_inference(model, processor, messages, device=device, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# cell-57: per-sample inference helpers
# ---------------------------------------------------------------------------

def run_sample_inference(
    model: Qwen2VLAudioForConditionalGeneration,
    processor: Qwen2VLProcessor,
    sample: dict,
    device: str | None = None,
    max_new_tokens: int = 256,
) -> tuple[str, str]:
    """Run inference on any dataset sample supported by format_data().

    Works for audio, image, video, and pre-formatted ('messages') samples.
    Ground truth is extracted from the assistant turn produced by format_data().

    Args:
        sample: A dataset row with a 'wav', 'image', 'video', or 'messages' key.
        (remaining args same as run_inference)

    Returns:
        (prediction, ground_truth) — both as plain strings.
    """
    messages = format_data(sample)
    ground_truth = next(
        (m["content"] for m in messages if m["role"] == "assistant"), ""
    )
    user_messages = [m for m in messages if m["role"] != "assistant"]
    prediction = run_inference(
        model, processor, user_messages, device=device, max_new_tokens=max_new_tokens
    )
    return prediction, str(ground_truth)


def run_asr_inference(
    model: Qwen2VLAudioForConditionalGeneration,
    processor: Qwen2VLProcessor,
    sample: dict,
    device: str | None = None,
    max_new_tokens: int = 256,
) -> tuple[str, str]:
    """Run ASR inference on a single LargeScaleASR dataset sample.

    Thin wrapper around run_sample_inference() kept for backward compatibility.
    For new code prefer run_sample_inference() which works for any modality.
    """
    return run_sample_inference(model, processor, sample, device=device, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# Bounding box visualisation
# ---------------------------------------------------------------------------

def draw_bounding_box(image_url: str, model_output: str) -> None:
    """Download image, parse bounding box from model output, and display with matplotlib.

    Qwen2VL returns coordinates normalised to [0, 1000]. Handles formats:
        "(x1,y1,x2,y2)"  or  "<box>(x1,y1,x2,y2)</box>"

    Args:
        image_url: URL (or local path) of the source image.
        model_output: Raw string returned by run_vl_inference().
    """
    import re
    import requests
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    from io import BytesIO

    coords = re.findall(r"\d+", model_output)
    if len(coords) < 4:
        print("Could not parse bounding box from output:", model_output)
        return
    x1, y1, x2, y2 = [int(c) for c in coords[:4]]

    if image_url.startswith("http"):
        resp = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(image_url).convert("RGB")
    w, h = img.size

    # Scale from [0, 1000] → pixel coords
    x1p, y1p = x1 * w / 1000, y1 * h / 1000
    x2p, y2p = x2 * w / 1000, y2 * h / 1000

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(np.array(img))
    rect = patches.Rectangle(
        (x1p, y1p), x2p - x1p, y2p - y1p,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    ax.add_patch(rect)
    ax.axis("off")
    ax.set_title(model_output)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Quick smoke test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, processor = load_model_and_processor()

    # ── cell-55: VL sanity check ──────────────────────────────────────────
    print("\n[VL sanity check]")
    vl_output = run_vl_inference(
        model, processor,
        image_url="https://t4.ftcdn.net/jpg/01/57/82/05/360_F_157820583_agejYX5XeczPZuWRSCDF2YYeCGwJqUdG.jpg",
        prompt="Detect the bounding box of the red car.",
    )
    print("VL output:", vl_output)

    # ── cell-57: ASR baseline ─────────────────────────────────────────────
    print("\n[ASR baseline — before fine-tuning, expect garbage output]")
    from datasets import load_dataset
    stream = load_dataset("speechbrain/LargeScaleASR", "small", split="train", streaming=True)
    sample = next(iter(stream))
    pred, gt = run_asr_inference(model, processor, sample)
    print("Ground truth:", gt)
    print("Prediction:  ", pred)
