# Qwen-MAVL-7B — Speech and Music Audio-Visual QA on Qwen2-VL

Fine-tuning [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) with a grafted Whisper encoder for two tasks:

1. **ASR** — speech recognition via Whisper encoder + linear projection → **4.85% WER** on LibriSpeech test-clean
2. **MUSIC-AVQA** — audio-visual question answering on music performance videos → **97.31% accuracy** on the
   7,402-pair available-video test subset

In the best AVQA model, a **single frozen Whisper encoder serves both roles**: it encodes the TTS-spoken
question *and* the music track, through two separate linear projectors. PANNs CNN14 is an **ablation
baseline**, not part of the best model — swapping Whisper's frame sequence for a PANNs pooled vector at
the same token budget costs 26 points.

Paper: *Qwen-MAVL-7B: An Efficient Modular Alternative to Omni-Modal Models for Music Audio-Visual QA*.
(The repo, W&B runs and HF checkpoints predate that name and use `qwen2-vl-audio` throughout.)

Architecture diagram: [Qwen2_VL_Audio_architecture.pdf](Qwen2_VL_Audio_architecture.pdf)

---

## Architecture

### ASR Track
```
raw audio → Whisper encoder (whisper-large-v3-turbo, frozen)
          → audio_projector (Linear 1280→3584)
          → <|audio_pad|> tokens → Qwen2-VL-7B LLM
```

### AVQA Track (`whisper_fullres_v3` — best model)
```
Video frames  → Qwen2-VL vision encoder                        → visual tokens
TTS question  → Whisper encoder (live)      → audio_projector  → <|audio_pad|> tokens
Music audio   → Whisper encoder (precomputed, stride-pooled to
                32 tok / 30 s chunk)        → music_projector  → <|music_pad|> tokens
                                    ↓
              all tokens → Qwen2-VL-7B LLM → answer
```

The **same frozen Whisper encoder** produces both the question and the music representations; only the
two linear projectors are learned. Music features are precomputed offline and passed as
`music_features` at training/inference time, so the music encoder is not stored in the model.

**Ablation variant** (`PANNs-32`, for comparison only): music audio → PANNs CNN14 → one pooled 2048-d
vector → expanded to 32 tokens. This is what `panns_features/` is for.

---

## Results

### ASR (LibriSpeech test-clean)

| Stage | What's trained | WER (test-clean) |
|---|---|---|
| Stage 1 | audio_projector only | 14.95%* |
| Stage 2 | audio_projector + LLM LoRA (r=64) | **4.85%** |
| Stage 3 | + 1 epoch LibriSpeech | 4.71%† |

Stage 2 (4.85%, normalized, full 2,620-utterance test-clean) is the checkpoint the AVQA track
initializes from, and is the figure reported in the paper. \* Stage 1 is n=100, unnormalized —
not directly comparable. † n=500, normalized.

### MUSIC-AVQA (test set, n=7,402)

| Model | Test Acc. |
|---|---|
| MUSIC-AVQA / AVST baseline (2022)* | 71.6% |
| Qwen2.5-Omni-7B zero-shot (audio-matched) | 56.82% |
| Qwen2.5-Omni-7B fine-tuned | 80.91% |
| Ours — `panns32`, 8 tokens (ablation) | 66.87% |
| Ours — `panns32`, 32 tokens (ablation) | 69.9% |
| Ours — `whisper32_full` (ablation) | 70.5% |
| Ours — `whisper_fullres` (30 s) | 95.91% |
| Ours — `whisper_fullres_v2` | 95.49% |
| **Ours — `whisper_fullres_v3`** | **97.31%** |

The paper uses descriptive configuration names; this repo, W&B, and the HF checkpoints use the original
internal labels. Mapping:

| Repo / W&B / HF | Paper |
|---|---|
| `whisper_fullres_v3` | Whisper-60s-chunked (question projector frozen) — headline model |
| `whisper_fullres_v2` | Whisper-60s-chunked (question projector tuned) |
| `whisper_fullres` | Whisper-30s |
| `whisper32_full` | Whisper-60s-compressed |
| `panns32` | PANNs-8 / PANNs-32 (8- or 32-token expansion) |

\* Published baselines use the **full official train/test splits**; our rows use 8,000 training pairs
and the 7,402-pair available-video test subset, so the two blocks are not directly comparable.
Zero-shot Omni is the audio-matched re-run (`src/avqa/omni/eval_qwen25omni_audiomatched.py`, 56.82%);
an earlier 38.42% figure fed the model frames + text only, without audio, and is superseded.

**Per-modality breakdown (v3):**

| Modality | Accuracy |
|---|---|
| Audio questions | 97.4% |
| Audio-Visual questions | 97.3% |
| Visual questions | 97.3% |

---

## Checkpoints

All checkpoints are on HuggingFace: [`MayaKD/qwen2-vl-audio`](https://huggingface.co/MayaKD/qwen2-vl-audio)

| Checkpoint | Description |
|---|---|
| `avqa_stage1_whisper_fullres_v3/` | Stage 1: `music_projector` only, everything else frozen |
| `avqa_stage2_whisper_fullres_v3/` | Stage 2: best model (97.31%) |
| `avqa_init/` | Base Qwen2-VL-7B with audio/music tokens added |

---

## Repository Structure

```
src/
  train.py        # AVQA + ASR training entry points (a FUNCTION LIBRARY, not a CLI)
  asr/            # ASR pipeline
    collator.py           # data collator with audio padding
    create_processor.py   # Qwen2VLProcessor + audio_processor assembly
    init_model.py         # model init + audio token injection
    data_utils.py         # LibriSpeech loading
    eval_utils.py         # WER evaluation callback
    inference.py          # single-sample inference
    wandb_utils.py        # W&B run setup
    train.py              # legacy ASR entry points (superseded by src/train.py)
  avqa/           # AVQA pipeline
    dataset.py                      # AVQADataset + fill_placeholders()
    eval_utils.py                   # AVQA accuracy callback + evaluate_avqa()
    eval_asr_post_avqa.py           # ASR regression check after AVQA training
    whisper_preprocess_fullres.py   # Whisper music features, per-chunk pooling (BEST MODEL)
    whisper_preprocess.py           # Whisper music features, 30 s
    whisper_preprocess_full.py      # Whisper music features, 60 s concatenated then pooled
    panns_preprocess.py             # PANNs CNN14 features (ablation)
    clap_preprocess.py              # CLAP features (exploratory; not used in the paper)
    tts_preprocess.py               # edge-tts question synthesis
    video_precompute.py             # frame extraction + caching
    omni/         # Qwen2.5-Omni baseline (paper Section 4.6)
      dataset.py                        # AVQADatasetOmni + make_messages
      collator.py
      train.py                          # matched fine-tuning of Omni
      eval_qwen25omni_audiomatched.py   # zero-shot, same inputs as fine-tuned (56.82%)
      eval_qwen25omni.py                # earlier zero-shot eval (superseded)
processor/      # Qwen2VLProcessor config (tokenizer + special tokens)
music_avqa_dataset/data/json/   # MUSIC-AVQA train/val/test splits (+ MUSIC-AVQA-R head/tail samples)
```

---

## Setup

Tested on RunPod A100 80GB with PyTorch 2.4.0 + CUDA 12.4.1.

```bash
# 1. Clone with submodules (custom transformers fork + qwen-vl-utils)
git clone --recurse-submodules https://github.com/MKDehdashti/Qwen2-vl-audio.git
cd Qwen2-vl-audio

# 2. Create venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./transformers
pip install -e ./qwen-vl-utils
pip install -r requirements.txt
```

The `transformers` submodule is a fork of HuggingFace Transformers with the `Qwen2VLAudio` model class added (separate audio and music input paths). The base Qwen2-VL model is not modified.

---

## Preprocessing (AVQA)

Before training, precompute features from the [MUSIC-AVQA dataset](https://github.com/GeWu-Lab/MUSIC-AVQA):

```bash
# Precomputed features are available on HuggingFace:
# MayaKD/qwen2-vl-audio-data (whisper_features_fullres/, panns_features/, video_frames/, tts_questions/)

# Or recompute from scratch:
python src/avqa/tts_preprocess.py --out_dir music_avqa_dataset/data/tts_questions
# Whisper music features — used by the BEST model for BOTH the music and question paths
python src/avqa/whisper_preprocess_fullres.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/whisper_features_fullres

# PANNs features — only needed to reproduce the PANNs-32 ablation
python src/avqa/panns_preprocess.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/panns_features \
    --checkpoint <path/to/Cnn14_mAP=0.431.pth>
python src/avqa/video_precompute.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/video_frames
```

PANNs checkpoint (`Cnn14_mAP=0.431.pth`) available at [zenodo.org/record/3987831](https://zenodo.org/record/3987831).

---

## Training

`src/train.py` is a **function library**, not a command-line tool — there is no `argparse` and no
`__main__`. Import the stage runners and call them (this is how the runs in the paper were produced):

```python
import sys; sys.path.insert(0, "src")
from train import (
    init_avqa_whisper_model,
    freeze_for_avqa_whisper_stage1,
    run_avqa_whisper_fullres_v2_stage1,     # Stage 1: music_projector only, LLM frozen
    setup_avqa_whisper_fullres_v2_stage2,
    run_avqa_whisper_fullres_v3_stage2,     # Stage 2: LoRA, question projector FROZEN (headline)
)
from avqa.dataset import AVQADataset

# Stage 1 — trains the music projector against a frozen LLM
model, processor = init_avqa_whisper_model(seed=42)
freeze_for_avqa_whisper_stage1(model)
run_avqa_whisper_fullres_v2_stage1(
    model=model, processor=processor,
    train_dataset=AVQADataset(split="train", max_samples=8000),
    val_dataset=AVQADataset(split="val"),
    seed=42,
)

# Stage 2 — restart the kernel first (see below), then load Stage 1 and add LoRA
model, processor, lora_config = setup_avqa_whisper_fullres_v2_stage2()
run_avqa_whisper_fullres_v3_stage2(model, processor, lora_config, seed=42)
```

The stage runners set `music_dir=WHISPER_FULLRES_MUSIC_DIR` internally, so the music path reads
`whisper_features_fullres/`. Point it at `panns_features/` only to reproduce the PANNs ablation.

**Always restart the Python kernel between Stage 1 and Stage 2** — `gc.collect()` and
`torch.cuda.empty_cache()` do not reliably free enough GPU memory.

Key training decisions:
- **Stage 2**: `audio_projector` (the question path) frozen — only `music_projector` + LLM LoRA are
  trained. This is the v2→v3 change: +1.8 pp overall (95.49% → 97.31%) and +9.7 pp on Audio/Comparative
  at the shared seed. Both are single-seed differences and sit inside the cross-seed spread
  (96.0% ± 3.9% over three full retrainings), so freezing is a conservative default rather than an
  established improvement.
- `load_best_model_at_end=False` — required due to key structure mismatch between stages

---

## Citation

If you use this work, please cite:

```bibtex
@misc{dehdashti2026qwenmavl,
  title  = {Qwen-MAVL-7B: An Efficient Modular Alternative to Omni-Modal Models
            for Music Audio-Visual QA},
  author = {Dehdashti, Maryam},
  year   = {2026},
  url    = {https://github.com/MKDehdashti/Qwen2-vl-audio}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
