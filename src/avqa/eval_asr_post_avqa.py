"""
eval_asr_post_avqa.py — Check ASR (WER) degradation after AVQA fine-tuning.

Loads the whisper_fullres_v2 Stage 2 model (base + LoRA adapter) and runs
LibriSpeech test-clean through the TTS/ASR audio pathway to check whether
AVQA training degraded transcription quality.

Baseline: WER 5.09% (n=500, normalized) — the Stage-2 merged root weights that initialize AVQA.
(4.71% = lora_stage3 LibriSpeech-adapted variant, never merged into root; 3.65% = its n=100
unnormalized final. Neither is the AVQA init — verified via HF root-shard commit history 2026-07-13.)

Usage:
    source /workspace/projects/speech/setup.sh
    python src/avqa/eval_asr_post_avqa.py           # n=500 (matches prior benchmark)
    python src/avqa/eval_asr_post_avqa.py --n 2620  # full test-clean set (~32 min on A100)
"""

import argparse
import re
import sys

sys.path.insert(0, "/workspace/projects/speech/transformers/src")
sys.path.insert(0, "/workspace/projects/speech/qwen-vl-utils/src")
sys.path.insert(0, "/workspace/projects/speech/src/asr")

import torch

HF_REPO       = "MayaKD/qwen2-vl-audio"
# Paths in the reorganized HF tree (2026-08-11). whisper_fullres_v2 is the headline
# Stage 1; its Stage 2 is the question-projector-tuned variant.
STAGE1_SUBDIR = "avqa/headline/stage1"
STAGE2_SUBDIR = "avqa/headline/stage2_qproj_tuned"
PROCESSOR_DIR = "/workspace/projects/speech/processor"


def load_model():
    from peft import PeftModel
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    print(f"[load] base model from {HF_REPO}/{STAGE1_SUBDIR} …")
    base = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        HF_REPO,
        subfolder=STAGE1_SUBDIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    print(f"[load] applying LoRA adapter from {HF_REPO}/{STAGE2_SUBDIR} …")
    model = PeftModel.from_pretrained(base, HF_REPO, subfolder=STAGE2_SUBDIR)
    model.eval()

    print(f"[load] processor from {PROCESSOR_DIR} …")
    processor = Qwen2VLProcessor.from_pretrained(PROCESSOR_DIR)
    processor.audio_processor = Qwen2VLAudioProcessor()

    return model, processor


def run_inference(model, processor, messages: list[dict]) -> str:
    from qwen_vl_utils import process_audio_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audio_inputs = process_audio_info(messages)

    inputs = processor(
        text=[text],
        audios=audio_inputs if audio_inputs else None,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, prompt_len:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0]


def normalize(text: str) -> str:
    try:
        from whisper.normalizers import EnglishTextNormalizer
        return EnglishTextNormalizer()(text)
    except ImportError:
        return re.sub(r"[^\w\s]", "", text.lower()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Samples to evaluate (default 500, matches prior benchmark)")
    args = parser.parse_args()

    try:
        from jiwer import wer, cer
    except ImportError:
        print("ERROR: pip install jiwer")
        sys.exit(1)

    model, processor = load_model()

    print(f"[eval] loading LibriSpeech test-clean (n={args.n}) …")
    from datasets import load_dataset
    from data_utils import format_data
    ds = load_dataset("speechbrain/LargeScaleASR", "small", split="test", streaming=True)

    references, hypotheses = [], []
    for i, sample in enumerate(ds):
        if i >= args.n:
            break

        if i == 0:
            print(f"  [debug] sample keys: {list(sample.keys())}")

        # format_data handles "wav" key (speechbrain format: wav["bytes"])
        # and extracts ground truth from text/wrd/label keys
        try:
            messages = format_data(sample)
        except Exception as e:
            print(f"  [skip] sample {i}: format_data failed: {e}")
            continue

        gt = next((m["content"] for m in messages if m["role"] == "assistant"), "").strip()
        if not gt:
            # fallback: speechbrain uses "wrd" key, not covered by format_data's label lookup
            gt = sample.get("wrd", "").strip()
        if not gt:
            print(f"  [skip] sample {i}: no ground truth found")
            continue

        user_messages = [m for m in messages if m["role"] != "assistant"]
        try:
            pred = run_inference(model, processor, user_messages)
            references.append(normalize(gt))
            hypotheses.append(normalize(pred))
            if i < 3:
                print(f"  [{i}] GT:   {gt}")
                print(f"  [{i}] PRED: {pred}")
        except Exception as e:
            print(f"  [skip] sample {i} inference failed: {e}")
            continue

        if (i + 1) % 20 == 0:
            running_wer = wer(references, hypotheses)
            print(f"  [{i+1}/{args.n}] running WER: {running_wer:.4f}")

    if not references:
        print("[eval] No samples evaluated.")
        sys.exit(1)

    wer_val = wer(references, hypotheses)
    cer_val = cer(references, hypotheses)
    delta   = wer_val - 0.0509
    print(f"\n[post-avqa WER] WER: {wer_val:.4f}  CER: {cer_val:.4f}  (n={len(references)})")
    print(f"[post-avqa WER] Baseline (pre-AVQA, Stage-2 merged root, n=500 normalized): WER 0.0509")
    print(f"[post-avqa WER] Delta: {delta:+.4f} ({'degraded' if delta > 0 else 'improved/unchanged'})")


if __name__ == "__main__":
    main()
