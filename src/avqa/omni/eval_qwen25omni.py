"""
Zero-shot Qwen2.5-Omni-7B evaluation on MUSIC-AVQA test set.

Passes video (with embedded audio) + text question — no fine-tuning, no TTS.
Filters to the same 7,402 samples used in all other experiments (available videos).
Scoring: identical to eval_utils.py — _normalise() + exact string match.

Usage:
    source .venv/bin/activate
    python src/avqa/eval_qwen25omni.py

Results saved to: qwen25omni_avqa_results.json
"""

import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path('/workspace/projects/speech/music_avqa_dataset')
VIDEO_DIR   = DATASET_DIR / 'data' / 'video' / 'MUSIC-AVQA-videos-Real'
TEST_JSON   = DATASET_DIR / 'data' / 'json_update' / 'avqa-test.json'
OUT_FILE    = Path('/workspace/projects/speech/qwen25omni_avqa_results.json')

MODEL_ID    = 'Qwen/Qwen2.5-Omni-7B'

SYSTEM_PROMPT = (
    "You are an expert at answering questions about music performance videos. "
    "Answer each question with only a single word or short phrase. "
    "Do not explain your answer."
)


def fill_placeholders(text: str, templ_values_str: str) -> str:
    """Identical to src/avqa/dataset.py — fills <Placeholder> tokens."""
    try:
        values = json.loads(templ_values_str)
    except Exception:
        values = []
    for val in values:
        text = text.replace('<Placeholder>', val, 1)
    text = text.replace('_', ' ')
    return text


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


def run_inference(model, processor, video_path: str, question: str) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": 8,        # match our model's frame count
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
    # Filter to available videos (same as all other experiments)
    samples = [s for s in samples if (VIDEO_DIR / f"{s['video_id']}.mp4").exists()]
    print(f"Test samples with available video: {len(samples)}")

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
        video_path = str(VIDEO_DIR / f"{sample['video_id']}.mp4")
        question   = fill_placeholders(sample['question_content'], sample['templ_values'])
        gt         = _normalise(sample['anser'])

        try:
            raw     = run_inference(model, processor, video_path, question)
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
