"""
eval_utils.py — ASR evaluation helpers.

Three evaluation moments wired into run_stage1():
  1. baseline  — before training, step=0    (tag="baseline")
  2. per-epoch — end of each epoch          (tag="epoch_eval", via ASREvalCallback)
  3. final     — after training completes   (tag="final")

All logged to the active W&B run so everything appears on the same chart.
"""

import sys

sys.path.insert(0, "/workspace/projects/speech/transformers/src")
sys.path.insert(0, "/workspace/projects/speech/qwen-vl-utils/src")
sys.path.insert(0, "/workspace/projects/speech/src/asr")

import wandb
from transformers import TrainerCallback

from inference import run_sample_inference


def _build_normalizer(normalize: bool):
    """Return a text normalizer function.

    If normalize=True, uses Whisper's EnglishTextNormalizer (lowercase,
    strip punctuation, normalise numbers) — the same normaliser used by
    published Qwen / Whisper evals, so numbers are directly comparable.
    Falls back to basic lowercase+strip if openai-whisper is not installed.
    If normalize=False, returns identity (strip only).
    """
    if not normalize:
        return str.strip
    try:
        from whisper.normalizers import EnglishTextNormalizer
        return EnglishTextNormalizer()
    except ImportError:
        import re
        print(
            "[eval] whisper package not found — using basic lowercase+strip normalizer.\n"
            "       Install with: pip install openai-whisper"
        )
        return lambda s: re.sub(r"[^\w\s]", "", s.lower()).strip()


def evaluate_asr(model, processor, dataset, n=100, tag="eval", step=None, normalize=True):
    """Run inference on n samples and compute WER / CER.

    Args:
        model:     Loaded Qwen2VLAudioForConditionalGeneration.
        processor: Matching Qwen2VLProcessor (with audio_processor attached).
        dataset:   HF Dataset — any format_data-compatible keys ('wav', 'audio', etc.).
        n:         Number of samples to evaluate.
        tag:       Prefix for W&B metric keys, e.g. "baseline", "epoch_eval", "final".
        step:      W&B global step so metrics land at the right x-axis position.
        normalize: Apply Whisper EnglishTextNormalizer before scoring (default True).
                   Use True for benchmark comparisons (LibriSpeech, Qwen published).
                   Use False to measure raw output exactly as produced by the model.

    Returns:
        dict with keys: wer, cer, n_evaluated
        Empty dict if jiwer is not installed.
    """
    try:
        from jiwer import wer, cer
    except ImportError:
        print("jiwer not installed — skipping WER/CER.  pip install jiwer")
        return {}

    norm = _build_normalizer(normalize)
    model.eval()
    ds = dataset.select(range(min(n, len(dataset)))) if hasattr(dataset, "select") else dataset

    # Avoid torchcodec (needs libavutil.so.56 / ffmpeg) by returning raw bytes.
    # fetch_audio already handles bytes via soundfile — no behavioural change.
    try:
        import datasets as hf_datasets
        if hasattr(ds, "features") and "audio" in ds.features:
            ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))
    except Exception:
        pass

    references, hypotheses = [], []
    for idx in range(len(ds)):
        sample = ds[idx]
        try:
            pred, gt = run_sample_inference(model, processor, sample)
            references.append(norm(gt))
            hypotheses.append(norm(pred))
        except Exception as e:
            print(f"  [eval] sample {idx} failed: {e}")
            continue

    if not references:
        print("[eval] No samples evaluated successfully.")
        return {}

    results = {
        "wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses),
        "n_evaluated": len(references),
    }

    print(
        f"[{tag}] WER: {results['wer']:.4f}  "
        f"CER: {results['cer']:.4f}  "
        f"(n={results['n_evaluated']})"
    )

    if wandb.run is not None:
        wandb.log({f"{tag}/{k}": v for k, v in results.items()}, step=step)

    return results


class ASREvalCallback(TrainerCallback):
    """Runs evaluate_asr() at the end of each training epoch.

    Gives a WER/CER curve across epochs inside the same W&B run.
    Uses a smaller n than the final eval to keep it fast.
    """

    def __init__(self, model, processor, dataset, n=50, tag="epoch_eval"):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.n = n
        self.tag = tag

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        evaluate_asr(
            self.model,
            self.processor,
            self.dataset,
            n=self.n,
            tag=self.tag,
            step=state.global_step,
        )
