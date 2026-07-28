"""
Audio-matched ZERO-SHOT Qwen2.5-Omni-7B evaluation on MUSIC-AVQA test set.

This is the *defensible* zero-shot baseline for the paper. Unlike the old
eval_qwen25omni.py (frames + text only — no audio, an invalid comparison), this
script feeds the base model the EXACT same inputs the fine-tuned Omni received:

    system prompt  +  8 precomputed frames  +  music audio track  +  TTS question  +  "Answer the question."

It reuses the fine-tuned eval's own building blocks verbatim:
    - AVQADatasetOmni('test')        (dataset.py)   → frames + video_audio + tts_path
    - make_messages(...)             (collator.py)  → identical chat format
    - process_vision_info / process_audio_info       → identical preprocessing
    - model.thinker.generate(greedy, max_new_tokens) → identical decoding

The ONLY difference vs the fine-tuned run is that NO LoRA adapter is applied.
=> Any accuracy gap is attributable to fine-tuning alone.

Crash-safe: the raw decoded prediction is saved per sample after every step, so
the run resumes where it left off, and ALL metrics are recomputed from the stored
raw text (the JSON is the source of truth — re-run scoring any time without GPU).

Reported metrics (2x2 grid), computed over the same samples:
    exact_norm   : _normalise(pred) == _normalise(gt)     # canonical, == fine-tuned scoring (HEADLINE)
    exact_raw    : pred.strip()     == gt.strip()         # no normalization
    lenient_norm : normalised gt is a whole-word phrase inside normalised pred
    lenient_raw  : raw gt is a whole-word phrase inside raw pred
Lenient metrics credit a zero-shot model that knows the answer but is verbose
("The answer is two") — a known zero-shot format-following weakness. The headline
number for "only LoRA differs" remains exact_norm.

Usage:
    source /workspace/projects/speech/setup.sh
    python src/avqa/omni/eval_qwen25omni_audiomatched.py
"""

import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm   # plain text bar — renders in VSCode notebooks (ipywidget bar fails there)

# Silence the per-sample "System prompt modified, audio output may not work" warning
# from the Omni processor. It concerns the talker (speech synthesis), which we never
# use — we only call model.thinker.generate() for TEXT. Harmless; just log spam.
# (The fine-tuned Omni used the same custom system prompt and got the same warning,
# so suppressing it changes nothing about comparability.)
logging.getLogger().addFilter(
    lambda record: 'System prompt modified' not in record.getMessage()
)

# ── Repo paths (transformers fork, qwen-vl-utils fork, omni package) ─────────────
for _p in (
    '/workspace/projects/speech/transformers/src',
    '/workspace/projects/speech/qwen-vl-utils/src',
    '/workspace/projects/speech/src/avqa/omni',
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collator import make_messages          # noqa: E402  identical chat format to fine-tuned
from dataset import AVQADatasetOmni          # noqa: E402  frames + music audio + TTS

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL_ID  = 'Qwen/Qwen2.5-Omni-7B'

# mp4s were uploaded here (not under music_avqa_dataset/), so override video_dir.
VIDEO_DIR = '/workspace/projects/speech/data/video/MUSIC-AVQA-videos-Real/MUSIC-AVQA-videos-Real-mp4'

OUT_FILE  = Path('/workspace/projects/speech/qwen25omni_zeroshot_audiomatched.json')

# Generation budget. Held at 5 to match EVERY other run in this project:
#   - fine-tuned Omni eval (train.py:evaluate_omni)        → 5
#   - our Qwen2-VL AVQA headline eval (eval_utils.evaluate_avqa) → 5  ("AVQA answers are ≤3 tokens")
# Identical decoding => the only difference vs fine-tuned is the LoRA adapter.
# Override for a separate lenient-friendly pass: OMNI_MAX_NEW_TOKENS=16
MAX_NEW_TOKENS = int(os.environ.get('OMNI_MAX_NEW_TOKENS', '5'))


# ── Scoring ─────────────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """Identical to eval_utils._normalise / train._normalise — lowercase + strip punctuation."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def _phrase_in(needle: str, haystack: str) -> bool:
    """True if `needle` occurs as a contiguous whole-word phrase inside `haystack`."""
    if not needle:
        return False
    return f' {needle} ' in f' {haystack} '


def score_record(raw_pred: str, gt: str) -> dict:
    """All four metrics for one (prediction, ground-truth) pair, from raw text."""
    pn, gn = _normalise(raw_pred), _normalise(gt)
    pr, gr = raw_pred.strip(),     gt.strip()
    return {
        'exact_norm':   int(pn == gn),
        'exact_raw':    int(pr == gr),
        'lenient_norm': int(_phrase_in(gn, pn)),
        'lenient_raw':  int(_phrase_in(gr, pr)),
    }


METRICS = ('exact_norm', 'exact_raw', 'lenient_norm', 'lenient_raw')


# ── Model ───────────────────────────────────────────────────────────────────────
def load_model():
    """Load BASE Qwen2.5-Omni-7B (no LoRA). Mirrors init_omni()'s loading exactly."""
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
    )
    from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor

    print(f'Loading {MODEL_ID} (base, zero-shot — no LoRA) ...')
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    model.eval()
    print('Model loaded.')
    return model, processor


def run_inference(model, processor, item: dict) -> str:
    """Decode one sample. Input construction is identical to evaluate_omni() in train.py."""
    from qwen_vl_utils import process_audio_info, process_vision_info

    msgs = make_messages(item['frames'], item['video_audio'], item['tts_path'])
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(msgs)
    audio_raw       = process_audio_info(msgs)
    audio_inputs    = [arr for arr, _ in audio_raw] if audio_raw else None

    device = next(model.thinker.parameters()).device
    inputs = processor(
        text=[text],
        videos=video_inputs if video_inputs else None,
        audio=audio_inputs,
        return_tensors='pt',
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.thinker.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    generated = out_ids[0][inputs['input_ids'].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Eval loop (crash-safe) ───────────────────────────────────────────────────────
def evaluate():
    dataset = AVQADatasetOmni('test', video_dir=VIDEO_DIR)
    print(f'Test samples (frames + video + tts all present): {len(dataset)}')

    # Resume: load any prior raw predictions, keyed by question_id.
    done: dict[int, dict] = {}
    if OUT_FILE.exists():
        done = {int(r['question_id']): r for r in json.loads(OUT_FILE.read_text())}
        print(f'Resuming — {len(done)} already done')

    remaining_idx = [i for i in range(len(dataset))
                     if int(dataset.items[i]['question_id']) not in done]
    print(f'Remaining to run: {len(remaining_idx)}')

    model = processor = None
    if remaining_idx:
        model, processor = load_model()

    results = list(done.values())
    for i in tqdm(remaining_idx, desc='Omni zero-shot (audio-matched)', unit='sample'):
        item = dataset[i]   # loads frames (.npy) + video_audio (ffmpeg) + tts path
        try:
            raw = run_inference(model, processor, item)
        except Exception as e:
            import traceback
            print(f"\nERROR qid={item['question_id']}: {e}", file=sys.stderr)
            n_blank = sum(1 for r in results if r.get('raw', '') == '')
            if n_blank <= 1:
                traceback.print_exc(file=sys.stderr)
            raw = ''

        results.append({
            'question_id': item['question_id'],
            'video_id':    item['video_id'],
            'type':        item['question_type'],
            'question':    item['question'],
            'gt':          item['answer'],   # raw ground truth (dataset 'anser' typo upstream)
            'raw':         raw,              # raw decoded prediction — source of truth for all metrics
        })

        # Resume-safe: persist after every sample.
        OUT_FILE.write_text(json.dumps(results, indent=2))

    print_scores(results)


# ── Scoring / reporting (recomputes everything from stored raw text) ──────────────
def print_scores(results: list[dict]):
    n = len(results)
    if n == 0:
        print('No results.')
        return

    totals = {m: 0 for m in METRICS}
    # per-type breakdown for the headline metric (exact_norm)
    per_type = defaultdict(lambda: [0, 0])  # qtype -> [correct_exact_norm, total]

    for r in results:
        sc = score_record(r['raw'], r['gt'])
        for m in METRICS:
            totals[m] += sc[m]
        try:
            qtype = ' / '.join(json.loads(r['type']))
        except Exception:
            qtype = str(r['type'])
        per_type[qtype][0] += sc['exact_norm']
        per_type[qtype][1] += 1

    bar = '─' * 64
    print(f'\n{bar}')
    print(f'Qwen2.5-Omni-7B ZERO-SHOT (audio-matched) — MUSIC-AVQA test (n={n})')
    print(f'inputs: frames + music audio + TTS question | base model, no LoRA | max_new_tokens={MAX_NEW_TOKENS}')
    print(bar)
    print('  metric          normalized      raw')
    print(f'  exact      {totals["exact_norm"]/n*100:9.2f}%  {totals["exact_raw"]/n*100:9.2f}%')
    print(f'  lenient    {totals["lenient_norm"]/n*100:9.2f}%  {totals["lenient_raw"]/n*100:9.2f}%')
    print(bar)
    print('HEADLINE (exact_norm, == fine-tuned scoring): '
          f'{totals["exact_norm"]/n*100:.2f}%  ({totals["exact_norm"]}/{n})')
    print(bar)
    print('Per-type (exact_norm):')
    for qtype, (c, t) in sorted(per_type.items()):
        print(f'  {qtype:<38} {c/t*100:6.2f}%  ({c}/{t})')


if __name__ == '__main__':
    evaluate()
