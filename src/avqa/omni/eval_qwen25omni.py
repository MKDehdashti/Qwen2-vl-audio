# ============================================================================
# ⚠️  DO NOT USE FOR THE PAPER — INVALID ZERO-SHOT BASELINE.  ⚠️
#
# THE BUG: this script gives the base model only 8 frames + the QUESTION TEXT.
# It passes NO AUDIO. But the fine-tuned Omni (80.91%) was given frames + the
# music audio track + the TTS-spoken question. So the published 38.42% compared
# two different input regimes — the gap measured "no audio vs audio", not
# "no fine-tuning vs fine-tuning". (A second bug in the original v1 run also left
# <LR>/<Object>/... placeholders unfilled in 69.4% of questions; that part is
# fixed here, but the no-audio flaw above remains and cannot be fixed in this
# frames+text design.)
#
# THE FIX: use  eval_qwen25omni_audiomatched.py  instead. It feeds the base model
# the EXACT same inputs as the fine-tuned run (frames + music audio + TTS question)
# via the shared AVQADatasetOmni + make_messages path, so the ONLY difference is
# the LoRA adapter. It is crash-safe (resumes) and reports exact/lenient ×
# normalized/raw, with exact_norm @ max_new_tokens=5 as the headline.
# ============================================================================
"""
Zero-shot Qwen2.5-Omni-7B evaluation on MUSIC-AVQA test set.

Passes 8 video frames + text question — no fine-tuning, no TTS.
Uses precomputed frames from video_frames/ (original mp4s were removed in repo cleanup;
the original run also passed only 8 frames + text — no audio — so the protocol is unchanged).
Filters to the same samples used in all other experiments (available frames).
Scoring: identical to eval_utils.py — _normalise() + exact string match.

Usage:
    source .venv/bin/activate
    python src/avqa/eval_qwen25omni.py

Results saved to: qwen25omni_avqa_results_v2.json
(v1 = qwen25omni_avqa_results.json — INVALID: fill_placeholders bug left <LR>/<Object>/...
placeholders unfilled in 69.4% of questions; kept only as a record of the buggy run.)
"""

import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.append(_src)
from avqa.dataset import fill_placeholders  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path('/workspace/projects/speech/music_avqa_dataset')
FRAMES_DIR  = DATASET_DIR / 'data' / 'video_frames'   # (8, H, W, 3) uint8 .npy per video
TEST_JSON   = DATASET_DIR / 'data' / 'json' / 'avqa-test.json'
OUT_FILE    = Path('/workspace/projects/speech/qwen25omni_avqa_results_v2.json')

MODEL_ID    = 'Qwen/Qwen2.5-Omni-7B'

SYSTEM_PROMPT = (
    "You are an expert at answering questions about music performance videos. "
    "Answer each question with only a single word or short phrase. "
    "Do not explain your answer."
)


def _normalise(text: str) -> str:
    """Identical to eval_utils._normalise — lowercase + strip punctuation."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def load_model():
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    print(f"Loading {MODEL_ID} ...")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def run_inference(model, processor, frames: list, question: str) -> str:
    """frames: list of 8 PIL Images (precomputed, same as fine-tuned runs)."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,     # pre-extracted frames — match our model's frame count
                },
                {
                    "type": "text",
                    "text": f"Question: {question}\nAnswer:",
                },
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors='pt',
    )
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.thinker.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    # Trim prompt tokens
    generated = output_ids[0][inputs['input_ids'].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate():
    # ── Load test data ─────────────────────────────────────────────────────────
    samples = json.loads(TEST_JSON.read_text())
    # Filter to available precomputed frames (same as all other experiments)
    samples = [s for s in samples if (FRAMES_DIR / f"{s['video_id']}.npy").exists()]
    print(f"Test samples with available frames: {len(samples)}")

    # ── Resume support ─────────────────────────────────────────────────────────
    done: dict[str, dict] = {}
    if OUT_FILE.exists():
        done = {r['question_id']: r for r in json.loads(OUT_FILE.read_text())}
        print(f"Resuming — {len(done)} already done")

    remaining = [s for s in samples if str(s['question_id']) not in done
                 and s['question_id'] not in done]

    # ── Load model ─────────────────────────────────────────────────────────────
    if remaining:
        model, processor = load_model()
    else:
        print("All samples already evaluated.")

    # ── Inference loop ─────────────────────────────────────────────────────────
    results = list(done.values())
    for sample in tqdm(remaining, desc='Qwen2.5-Omni zero-shot'):
        import numpy as np
        from PIL import Image
        frames_arr = np.load(FRAMES_DIR / f"{sample['video_id']}.npy")  # (8, H, W, 3) uint8
        frames     = [Image.fromarray(frames_arr[i]) for i in range(len(frames_arr))]
        question   = fill_placeholders(sample['question_content'], sample['templ_values'])
        gt         = _normalise(sample['anser'])

        try:
            raw     = run_inference(model, processor, frames, question)
            pred    = _normalise(raw)
            correct = int(pred == gt)
        except Exception as e:
            import traceback
            print(f"\nERROR qid={sample['question_id']}: {e}", file=sys.stderr)
            if sum(1 for r in results if r['correct'] == 0 and r['raw'] == '') <= 1:
                traceback.print_exc(file=sys.stderr)
            raw, pred, correct = '', '', 0

        results.append({
            'question_id': sample['question_id'],
            'video_id':    sample['video_id'],
            'type':        sample['type'],
            'question':    question,
            'gt':          gt,
            'raw':         raw,
            'pred':        pred,
            'correct':     correct,
        })

        # Save after every sample (resume-safe)
        OUT_FILE.write_text(json.dumps(results, indent=2))

    # ── Scoring ────────────────────────────────────────────────────────────────
    print_scores(results)


def print_scores(results: list[dict]):
    from collections import defaultdict

    overall  = sum(r['correct'] for r in results) / len(results) * 100
    per_type = defaultdict(lambda: [0, 0])  # [correct, total]

    for r in results:
        try:
            qtype = json.loads(r['type'])
            key   = ' / '.join(qtype)
        except Exception:
            key = r['type']
        per_type[key][0] += r['correct']
        per_type[key][1] += 1

    print(f"\n{'─'*50}")
    print(f"Qwen2.5-Omni-7B zero-shot — MUSIC-AVQA test (n={len(results)})")
    print(f"{'─'*50}")
    print(f"Overall: {overall:.2f}%")
    print()
    for qtype, (correct, total) in sorted(per_type.items()):
        print(f"  {qtype:<35} {correct/total*100:.2f}%  ({correct}/{total})")


if __name__ == '__main__':
    evaluate()
