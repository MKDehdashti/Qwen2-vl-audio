# Qwen2-VL-Audio: Dual-Encoder Speech and Music-Audio-Visual QA

Fine-tuning [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) with a grafted Whisper encoder for two tasks:

1. **ASR** — speech recognition via Whisper encoder + linear projection → **3.65% WER** on LibriSpeech test-clean
2. **MUSIC-AVQA** — audio-visual question answering on music performance videos via dual encoder (Whisper + PANNs) → **97.31% accuracy** on MUSIC-AVQA test set (state of the art)

Architecture diagram: [Qwen2_VL_Audio_architecture.pdf](Qwen2_VL_Audio_architecture.pdf)

---

## Architecture

### ASR Track
```
raw audio → Whisper encoder (whisper-large-v3-turbo, frozen)
          → audio_projector (Linear 1280→3584)
          → <|audio_pad|> tokens → Qwen2-VL-7B LLM
```

### AVQA Track (whisper_fullres_v3 — best model)
```
Video frames  → Qwen2-VL vision encoder → visual tokens
TTS question  → Whisper encoder (fullres, var-len) → audio_projector → <|audio_pad|> tokens
Music audio   → PANNs CNN14 (precomputed, frozen) → music_projector → <|music_pad|> tokens
                                    ↓
              all tokens → Qwen2-VL-7B LLM → answer
```

The music encoder (PANNs CNN14) is not stored in the model — embeddings are precomputed offline and passed as `music_features` at training/inference time.

---

## Results

### ASR (LibriSpeech test-clean)

| Stage | What's trained | WER |
|---|---|---|
| Stage 1 | audio_projector only | ~10% |
| Stage 2 | audio_projector + LLM LoRA (r=64) | **3.65%** |

### MUSIC-AVQA (test set, n=7,402)

| Model | Test Acc. |
|---|---|
| MUSIC-AVQA baseline (2022)* | 79.42% |
| Qwen2.5-Omni-7B zero-shot (audio-matched) | 56.82% |
| Qwen2.5-Omni-7B fine-tuned | 80.91% |
| Ours — PANNs-32 | 66.87% |
| Ours — whisper_fullres_v2 | 94.78% |
| **Ours — whisper_fullres_v3** | **97.31%** |

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
| `avqa_stage1_whisper_fullres_v3/` | Stage 1: projectors trained, LLM frozen |
| `avqa_stage2_whisper_fullres_v3/` | Stage 2: best model (97.31%) |
| `avqa_init/` | Base Qwen2-VL-7B with audio/music tokens added |

---

## Repository Structure

```
src/
  asr/          # ASR training pipeline
    train.py            # Stage 1 & 2 training
    collator.py         # data collator with audio padding
    init_model.py       # model init + audio token injection
    eval_utils.py       # WER evaluation callback
    inference.py        # single-sample inference
  avqa/         # AVQA training pipeline
    train.py            # Stage 1 & 2 training (via src/train.py)
    dataset.py          # AVQADataset + fill_placeholders()
    whisper_preprocess_fullres.py   # Whisper fullres feature extraction
    panns_preprocess.py             # PANNs CNN14 feature extraction
    tts_preprocess.py               # edge-tts question synthesis
    video_precompute.py             # frame extraction + caching
    eval_utils.py                   # AVQA accuracy callback
processor/      # Qwen2VLProcessor config (tokenizer + special tokens)
music_avqa_dataset/data/json/   # MUSIC-AVQA train/val/test splits
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

The `transformers` submodule is a fork of HuggingFace Transformers with the `Qwen2VLAudio` model class added (dual-encoder support). The base Qwen2-VL model is not modified.

---

## Preprocessing (AVQA)

Before training, precompute features from the [MUSIC-AVQA dataset](https://github.com/GeWu-Lab/MUSIC-AVQA):

```bash
# Precomputed features are available on HuggingFace:
# MayaKD/qwen2-vl-audio-data (whisper_features_fullres/, panns_features/, video_frames/, tts_questions/)

# Or recompute from scratch:
python src/avqa/tts_preprocess.py --out_dir music_avqa_dataset/data/tts_questions
python src/avqa/panns_preprocess.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/panns_features \
    --checkpoint <path/to/Cnn14_mAP=0.431.pth>
python src/avqa/whisper_preprocess_fullres.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/whisper_features_fullres
python src/avqa/video_precompute.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/video_frames
```

PANNs checkpoint (`Cnn14_mAP=0.431.pth`) available at [zenodo.org/record/3987831](https://zenodo.org/record/3987831).

---

## Training

```bash
# AVQA Stage 1: train projectors, freeze LLM
python src/train.py --stage avqa1 \
    --whisper_features_dir music_avqa_dataset/data/whisper_features_fullres \
    --music_features_dir music_avqa_dataset/data/panns_features \
    --video_frames_dir music_avqa_dataset/data/video_frames

# AVQA Stage 2: unfreeze LLM with LoRA (r=64)
python src/train.py --stage avqa2 \
    --stage1_checkpoint <path/to/stage1> \
    --whisper_features_dir music_avqa_dataset/data/whisper_features_fullres \
    --music_features_dir music_avqa_dataset/data/panns_features \
    --video_frames_dir music_avqa_dataset/data/video_frames
```

Key training decisions:
- **Stage 2**: `audio_projector` frozen (only `music_projector` + LLM LoRA trained) — this was the key change from v2→v3, yielding +2.5 pp overall and +9.7 pp on Comparative questions
- `load_best_model_at_end=False` — required due to key structure mismatch between stages

---

## Citation

If you use this work, please cite:

```bibtex
@misc{dehdashti2026qwen2vlaudio,
  title  = {Dual-Encoder Qwen2-VL for Speech and Music Audio-Visual Question Answering},
  author = {Dehdashti, Maryam},
  year   = {2026},
  url    = {https://github.com/MKDehdashti/Qwen2-vl-audio}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
