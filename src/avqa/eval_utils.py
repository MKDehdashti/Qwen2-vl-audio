"""
src/avqa/eval_utils.py — AVQA evaluation helpers.

Mirrors src/eval_utils.py (WER/CER for ASR) but uses accuracy (exact match).

Three evaluation moments wired into run_avqa_stage1/2():
  1. baseline  — before training, step=0    (tag="baseline")
  2. per-epoch — end of each epoch          (tag="epoch_eval", via AVQAEvalCallback)
  3. final     — after training completes   (tag="final")

Also logs per-question-type accuracy breakdown (Audio-Visual / Audio / Visual ×
Counting / Existential / Location / Comparative / Temporal).
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/workspace/projects/speech/transformers/src")
sys.path.insert(0, "/workspace/projects/speech/qwen-vl-utils/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import TrainerCallback
from qwen_vl_utils import process_audio_info, process_vision_info


def _normalise(text: str) -> str:
    """Lowercase, strip whitespace and punctuation for exact-match comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


# Extra scoring variants (additive — DO NOT change the headline exact_norm metric,
# which equals the v3 / Omni published numbers). Identical definitions to the
# Qwen2.5-Omni audio-matched baseline (eval_qwen25omni_audiomatched.py) so all
# models are scored apple-to-apple.
METRICS = ("exact_norm", "exact_raw", "lenient_norm", "lenient_raw")


def _phrase_in(needle: str, haystack: str) -> bool:
    """True if `needle` occurs as a contiguous whole-word phrase inside `haystack`."""
    return bool(needle) and f" {needle} " in f" {haystack} "


def score_record(raw_pred: str, raw_gt: str) -> dict:
    """All four metrics for one (prediction, ground-truth) pair, from raw text.

    exact_norm == the existing headline metric (_normalise + exact match).
    """
    pn, gn = _normalise(raw_pred), _normalise(raw_gt)
    pr, gr = raw_pred.strip(),     raw_gt.strip()
    return {
        "exact_norm":   int(pn == gn),
        "exact_raw":    int(pr == gr),
        "lenient_norm": int(_phrase_in(gn, pn)),
        "lenient_raw":  int(_phrase_in(gr, pr)),
    }


def run_avqa_inference(model, processor, item: dict, max_new_tokens: int = 10) -> str:
    """Run inference on a single AVQADataset item. Returns the model's answer string."""
    messages       = item["messages"]
    music_features = item["music_features"]   # np.float32 [2048]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    audio_inputs               = process_audio_info(messages)

    # Single-vector [D] → add batch dim [1, D].
    # Sequence [T, D] → keep flat [T, D] (total_T = T for 1 sample), pass music_lengths=[T].
    if music_features.ndim == 1:
        music_tensor  = torch.from_numpy(music_features[np.newaxis, :]).float()
        music_lengths = None
    else:
        T = music_features.shape[0]
        music_tensor  = torch.from_numpy(music_features).float()   # [T, D]
        music_lengths = [T]

    inputs = processor(
        text=[text],
        audios=audio_inputs  if audio_inputs  else None,
        images=image_inputs  if image_inputs  else None,
        videos=video_inputs  if video_inputs  else None,
        music_features=music_tensor,
        music_lengths=music_lengths,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = processor.tokenizer.decode(
        generated[0][prompt_len:], skip_special_tokens=True
    ).strip()
    return answer


def _preprocess_batch(processor, items: list[dict]) -> dict:
    """CPU-only: load frames, run processor, return inputs on CPU. No GPU needed.
    Designed to run in a background thread while GPU works on the previous batch."""
    all_messages = [item["messages"] for item in items]

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    images, videos = process_vision_info(all_messages)
    audios          = process_audio_info(all_messages)

    # Handle both single-vector [D] and sequence [T_i, D] music features.
    # For variable-T sequence features, concatenate along T and pass music_lengths.
    first_mf = items[0]["music_features"]
    if first_mf.ndim == 1:
        # Single-vector [D]: stack → [B, D]
        music_tensor  = torch.from_numpy(
            np.stack([item["music_features"] for item in items])
        ).float()
        music_lengths = None
    else:
        # Always pass music_lengths so processor uses actual T per sample,
        # not the self.n_music_tokens fallback (default 8).
        Ts = [item["music_features"].shape[0] for item in items]
        music_tensor  = torch.from_numpy(
            np.concatenate([item["music_features"] for item in items], axis=0)
        ).float()
        music_lengths = Ts

    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor(
            text=texts,
            audios=audios,
            images=images,
            videos=videos,
            music_features=music_tensor,
            music_lengths=music_lengths,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    finally:
        processor.tokenizer.padding_side = orig_padding_side

    return inputs  # CPU tensors — moved to GPU in evaluate_avqa just before generate


def _run_batch_inference(model, processor, items: list[dict], max_new_tokens: int = 10) -> list[str]:
    """Run batched inference on a list of AVQADataset items."""
    inputs = _preprocess_batch(processor, items)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    answers = []
    for seq in generated:
        ans = processor.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        answers.append(ans)
    return answers


def evaluate_avqa(model, processor, dataset, n=100, tag="eval", step=None,
                  batch_size=4, max_new_tokens=5, detailed=False):
    """Run inference on n samples and compute accuracy + per-type breakdown.

    Args:
        model:          Loaded Qwen2VLDualAudioForConditionalGeneration.
        processor:      Matching Qwen2VLProcessor (with audio_processor attached).
        dataset:        AVQADataset instance.
        n:              Number of samples to evaluate.
        tag:            Prefix for W&B metric keys (e.g. "baseline", "epoch_eval").
        step:           W&B global step for chart alignment.
        batch_size:     Samples per forward pass (default 4; set 1 to match old behaviour).
        max_new_tokens: Cap on generated tokens (default 5; AVQA answers are ≤3 tokens).
        detailed:       If True, also print the exact/lenient × normalized/raw grid and a
                        multi-line per-type breakdown. Purely additive — the headline
                        `accuracy` (== exact_norm) is unchanged regardless of this flag,
                        so v3 / Omni numbers stay reproducible. Default False (callers
                        during training are byte-for-byte unaffected).

    Returns:
        dict with keys: accuracy, n_correct, n_evaluated, per_type: {type_str: accuracy},
                        and metrics: {exact_norm, exact_raw, lenient_norm, lenient_raw}.
                        accuracy == metrics['exact_norm'] by construction.
    """
    model.eval()
    n = min(n, len(dataset))

    correct  = 0
    per_type: dict[str, list[bool]] = {}
    metric_tot = {m: 0 for m in METRICS}   # additive extra metrics

    def load_and_preprocess(start):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        inputs = _preprocess_batch(processor, items) if batch_size > 1 else None
        return items, inputs

    pbar = tqdm(range(0, n, batch_size), desc=f"[avqa_eval/{tag}]", unit="batch")
    with ThreadPoolExecutor(max_workers=1) as pool:
        # Kick off preprocessing of first batch in background
        future = pool.submit(load_and_preprocess, 0)

        for batch_start in pbar:
            # Wait for current batch's CPU preprocessing to finish
            batch_items, inputs_cpu = future.result()
            raw_gts = [item["answer"] for item in batch_items]
            gts     = [_normalise(g) for g in raw_gts]

            # Immediately kick off preprocessing of NEXT batch in background
            # (overlaps CPU work with GPU inference below)
            next_start = batch_start + batch_size
            if next_start < n:
                future = pool.submit(load_and_preprocess, next_start)

            try:
                if batch_size == 1:
                    raw_preds = [run_avqa_inference(model, processor, batch_items[0])]
                    preds     = [_normalise(raw_preds[0])]
                else:
                    # Pin memory → faster CPU→GPU transfer
                    inputs_gpu = {
                        k: v.pin_memory().to(model.device, non_blocking=True)
                           if isinstance(v, torch.Tensor) else v
                        for k, v in inputs_cpu.items()
                    }
                    prompt_len = inputs_gpu["input_ids"].shape[1]
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs_gpu,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=processor.tokenizer.pad_token_id,
                        )
                    raw_preds = [
                        processor.tokenizer.decode(
                            seq[prompt_len:], skip_special_tokens=True).strip()
                        for seq in generated
                    ]
                    preds = [_normalise(p) for p in raw_preds]
            except Exception as e:
                print(f"\n  [avqa_eval] batch {batch_start} failed: {e}")
                continue

            for i, (gt, pred) in enumerate(zip(gts, preds)):
                is_correct = pred == gt
                correct   += int(is_correct)
                # Additive: accumulate the 4-metric grid from raw text.
                # sc['exact_norm'] == int(is_correct) by construction.
                sc = score_record(raw_preds[i], raw_gts[i])
                for m in METRICS:
                    metric_tot[m] += sc[m]
                try:
                    q_type = json.loads(batch_items[i]["question_type"])
                    keys = q_type + [" / ".join(q_type)]
                    for k in keys:
                        per_type.setdefault(k, []).append(is_correct)
                except Exception:
                    pass

            evaluated_so_far = min(batch_start + batch_size, n)
            pbar.set_postfix(acc=f"{correct/evaluated_so_far:.3f}")

    if n == 0:
        print("[avqa_eval] No samples evaluated.")
        return {}

    accuracy = correct / n
    results  = {
        "accuracy":    accuracy,
        "n_correct":   correct,
        "n_evaluated": n,
        "per_type":    {k: sum(v) / len(v) for k, v in per_type.items()},
        "metrics":     {m: metric_tot[m] / n for m in METRICS},  # additive grid
    }

    print(
        f"[{tag}] Accuracy: {accuracy:.4f}  ({correct}/{n})  "
        + "  ".join(f"{k}: {v:.3f}" for k, v in results["per_type"].items()
                    if "/" in k)
    )

    if detailed:
        bar = "─" * 64
        print(bar)
        print(f"[{tag}] scoring grid (n={n}) | max_new_tokens={max_new_tokens}")
        print("  metric          normalized      raw")
        print(f'  exact      {metric_tot["exact_norm"]/n*100:9.2f}%  {metric_tot["exact_raw"]/n*100:9.2f}%')
        print(f'  lenient    {metric_tot["lenient_norm"]/n*100:9.2f}%  {metric_tot["lenient_raw"]/n*100:9.2f}%')
        print(bar)
        print(f'  HEADLINE (exact_norm, == accuracy): {accuracy*100:.2f}%  ({correct}/{n})')
        print(bar)
        print("Per-type (exact_norm):")
        for k in sorted(per_type):
            if "/" in k:   # joined question-type keys (matches Omni breakdown)
                c, t = sum(per_type[k]), len(per_type[k])
                print(f"  {k:<38} {c/t*100:6.2f}%  ({c}/{t})")
        print(bar)

    if wandb.run is not None:
        log = {f"{tag}/accuracy": accuracy, f"{tag}/n_evaluated": n}
        for m, v in results["metrics"].items():
            log[f"{tag}/{m}"] = v
        for k, v in results["per_type"].items():
            log[f"{tag}/type/{k}"] = v
        wandb.log(log, step=step)

    return results


class AVQAEvalCallback(TrainerCallback):
    """Runs evaluate_avqa() at the end of each training epoch."""

    def __init__(self, model, processor, dataset, n=50, tag="epoch_eval"):
        self.model     = model
        self.processor = processor
        self.dataset   = dataset
        self.n         = n
        self.tag       = tag

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        evaluate_avqa(
            self.model,
            self.processor,
            self.dataset,
            n=self.n,
            tag=self.tag,
            step=state.global_step,
        )
