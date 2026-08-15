# Qwen-MusicAVQA-7B — Speech and Music Audio-Visual QA on Qwen2-VL

Fine-tuning [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) with a grafted Whisper encoder for two tasks:

1. **ASR** — speech recognition via Whisper encoder + linear projection → **4.85% WER** on LibriSpeech test-clean
2. **MUSIC-AVQA** — audio-visual question answering on music performance videos → **96.0% ± 3.9%**
   across three full retraining seeds on the 7,402-pair available-video test subset
   (seed-42 representative run: 97.31%)

In the best AVQA model, a **single frozen Whisper encoder serves both roles**: it encodes the TTS-spoken
question *and* the music track, through two separate linear projectors. PANNs CNN14 is an **ablation
baseline**, not part of the best model — swapping Whisper's frame sequence for a PANNs pooled vector at
the same token budget costs 26 points.

📄 **Paper:** [*Qwen-MusicAVQA-7B: A Multimodal Model for Music Audio-Visual QA*](https://arxiv.org/abs/2608.11329) (arXiv:2608.11329)
🤗 **Checkpoints:** [`MayaKD/qwen2-vl-audio`](https://huggingface.co/MayaKD/qwen2-vl-audio)

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

The **same frozen Whisper encoder** produces both the question and the music representations. The only
new modality-specific modules are the two linear projectors; Whisper and the Qwen2-VL base weights stay
frozen throughout, and AVQA Stage 2 additionally trains LoRA adapters on the LLM's attention layers.
Music features are precomputed offline and passed as `music_features` at training/inference time, so the
music encoder is not stored in the model. Question audio is **not** precomputed — it is encoded by the
same frozen Whisper encoder live in the forward pass.

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
| Ours — PANNs-8 (ablation) | 66.87% |
| Ours — `panns32` (ablation) | 69.9% |
| Ours — `whisper32_full` (ablation) | 70.5% |
| Ours — `whisper32` (30 s) | 95.91% |
| Ours — `whisper_fullres_v2` | 95.49% |
| **Ours — `whisper_fullres_v3`** | **97.31%** |

The paper uses descriptive configuration names; this repo, W&B, and the HF checkpoints use the original
internal labels. Mapping:

| Repo / W&B / HF | Paper |
|---|---|
| `whisper_fullres_v3` | Whisper-60s-chunked (question projector frozen) — headline model |
| `whisper_fullres_v2` | Whisper-60s-chunked (question projector tuned); its Stage-1 checkpoint is shared with v3 |
| `whisper32` | Whisper-30s |
| `whisper32_full` | Whisper-60s-compressed |
| `panns32` | PANNs-32 (32-token expansion) |

(PANNs-8, the earlier 8-token PANNs run, predates the naming scheme: HF `avqa_stage1/` and
`avqa_stage2/`. The plain `whisper_fullres` run is an earlier 60 s-chunked variant not reported in the
paper.)

### Why the ablations differ: temporal resolution, not encoder quality

`whisper32_full` (70.5%) and `whisper32` (95.91%) use the **same frozen Whisper encoder and the same
32-token budget**. The only difference is how much time each token covers — and that accounts for the
25-point gap. This is the paper's central finding, and the reason the internal names are misleading on
their own:

| Music encoder | Representation | Audio in | # tok. | Proj. | Overall | AV/Temp. |
|---|---|---|---|---|---|---|
| *Pooled — one global vector expanded to tokens* ||||||
| PANNs-8 | classification, 2048-d | full clip | 8 | 59M | 66.9% | 69.9% |
| PANNs-32 | classification, 2048-d | full clip | 32 | 235M | 69.9% | 69.4% |
| *Whisper — a 1280-d frame sequence, one token per frame* ||||||
| Whisper-30s (`whisper32`) | **0.94 s/token** | first 30 s | 32 | 4.6M | 95.9% | 86.2% |
| **Whisper-60s-chunked** (`whisper_fullres_v3`) | **0.94 s/token** | ~60 s | 32–64 | 4.6M | **97.3%** | **92.2%** |
| Whisper-60s-compressed (`whisper32_full`) | **1.875 s/token** | ~60 s | 32 | 4.6M | 70.5% | 64.7% |

Note the 26-point PANNs-32 → Whisper-30s gap at an identical 32-token budget, with a **50× smaller
projector** (4.6M vs 235M). Seeing *more* audio does not help if the representation is pooled flat.

### On comparability

The fine-tuned Omni comparison matches data, inputs, audio duration, and Stage-2 hyperparameters, but
the systems differ in backbone and adaptation; read it as a system-level comparison rather than an
isolated encoder comparison.

\* Published baselines use the **full official train/test splits**; our rows use 8,000 training pairs
and the 7,402-pair available-video test subset, so the two blocks are not directly comparable.
Zero-shot Omni is the audio-matched re-run (`src/avqa/omni/eval_qwen25omni_audiomatched.py`, 56.82%);
an earlier 38.42% figure fed the model frames + text only, without audio, and is superseded.

**Is the available-video subset easier?** There is some evidence it is not. Qwen2.5-Omni-7B, fine-tuned
on this subset, reaches **80.9%** — *below* the 83.5% the best published system reports on the full
official split. A substantially easier subset should have pushed a strong modern 7B model above the
published numbers, not below them. This is directional rather than conclusive: our Omni row trains on
8,000 pairs against the published systems' full ~32k train split, so the two differ in training budget
as well as test set.

**Per-modality breakdown (v3):**

| Modality | Accuracy |
|---|---|
| Audio questions | 97.4% |
| Audio-Visual questions | 97.3% |
| Visual questions | 97.3% |

---

## Checkpoints

On HuggingFace: [`MayaKD/qwen2-vl-audio`](https://huggingface.co/MayaKD/qwen2-vl-audio). The **repo
root is the merged headline model** (Stage-1 projector + Stage-2 LoRA already merged), so it loads
in one line:

```python
Qwen2VLDualAudioForConditionalGeneration.from_pretrained("MayaKD/qwen2-vl-audio", torch_dtype="bfloat16")
```

| Path | Description |
|---|---|
| *(root)* | **Qwen-MusicAVQA-7B, merged — 97.31%** |
| `asr/merged_stage2/` | ASR Stage-2 merge (4.85% WER); what every AVQA run initializes from |
| `asr/stage1_only/`, `asr/lora_stage2/`, `asr/lora_stage3/` | ASR track |
| `avqa/init/` | ASR merge + untrained `music_projector` (the AVQA starting point) |
| `avqa/headline/stage1/` | `music_projector` only, LLM frozen — 96.0% |
| `avqa/headline/stage2_qproj_frozen/` | LoRA, question projector frozen — 97.31% (merged into the root) |
| `avqa/headline/stage2_qproj_tuned/` | LoRA, question projector tuned — 95.49% |
| `avqa/seeds/seed1234\|seed2026/` | the runs behind 96.0% ± 3.9% |
| `avqa/ablations/<tag>/{stage1,stage2}/` | one folder per W&B `experiment_tag` (see the mapping table above) |
| `avqa/comparison/qwen2.5-omni/` | fine-tuned Qwen2.5-Omni baseline |

`headline/stage1/` is shared — both Stage-2 variants trained from it, differing only in whether the
question projector was frozen. To reproduce training, start from `asr/merged_stage2/`, **not** the
root: the root is the finished model.

---

## Repository Structure

```
src/
  train.py        # ALL training entry points, ASR + AVQA (imported as a library — see Training)
  asr/            # ASR pipeline components used by src/train.py
    collator.py           # data collator with audio padding
    create_processor.py   # Qwen2VLProcessor + audio_processor assembly
    init_model.py         # model init + audio token injection
    data_utils.py         # LibriSpeech loading
    eval_utils.py         # WER evaluation callback
    inference.py          # single-sample inference
    wandb_utils.py        # W&B run setup
  avqa/           # AVQA pipeline components used by src/train.py
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
    omni/         # Qwen2.5-Omni baseline (paper Section 4.6) — self-contained, own train script
      dataset.py                        # AVQADatasetOmni + make_messages
      collator.py
      train.py                          # matched LoRA fine-tuning of Qwen2.5-Omni (not via src/train.py)
      eval_qwen25omni_audiomatched.py   # zero-shot, same inputs as fine-tuned (56.82%)
      eval_qwen25omni.py                # earlier zero-shot eval (superseded)
processor/      # Qwen2VLProcessor config (tokenizer + special tokens)
music_avqa_dataset/data/json/   # MUSIC-AVQA train/val/test splits (+ MUSIC-AVQA-R head/tail samples)
```

---

## Setup

Tested on a single RunPod A100 80GB with CUDA 12.4. PyTorch comes from the RunPod template
rather than pip (see the note at the top of `requirements.txt`).

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

Before training, precompute features from the [MUSIC-AVQA dataset](https://github.com/GeWu-Lab/MUSIC-AVQA).

### Already published

Every precomputed audio feature set is on
[`MayaKD/qwen2-vl-audio-data`](https://huggingface.co/datasets/MayaKD/qwen2-vl-audio-data) — download
these rather than recomputing them (they need the source videos, which are not redistributed):

| Directory | Used by |
|---|---|
| `whisper_features_fullres/` | **the headline model** (music branch) |
| `whisper_features_full/` | the Whisper-60 s-compressed ablation (70.5%) |
| `whisper_features/` | the Whisper-30 s ablation (95.91%) |
| `whisper_features_fullres_varlen/` | the variable-length ablation (78.55%, negative result) |
| `panns_features/` | the PANNs-8 / PANNs-32 ablations |
| `clap_features/` | the CLAP ablation (exploratory, not in the paper) |
| `tts_questions_r/` | the MUSIC-AVQA-R robustness evaluation |

### You must regenerate these two

`video_frames/` (42 GB) and `tts_questions/` (42,814 of its 45,629 entries are symlinks) were **too
large to upload**, so they are not in that repo and have to be produced locally:

- **`tts_questions/`** — regenerate directly; **no video needed**. The question text comes from
  `music_avqa_dataset/data/json/` (included in this repo) and is synthesized with `edge-tts`. Only
  the 2,815 unique question texts are synthesized; the remaining ~42,800 become symlinks.
- **`video_frames/`** — requires the **source videos, which are not redistributed here**. Download
  `MUSIC-AVQA-videos-Real.zip` (36.7 GB) from the MUSIC-AVQA authors, unzip it, then extract frames.
  Note that Google Drive may refuse the download with a quota error; if so, retry later or request
  the archive from the dataset authors. Roughly 1,900 of the 9,288 referenced videos are no longer
  downloadable — the loaders skip them, which is why the effective test split is 7,402 pairs and
  not the full 9,185.

```bash
# 1. TTS questions — no video required
python src/avqa/tts_preprocess.py --out_dir music_avqa_dataset/data/tts_questions

# 2. Source videos — needed for video_frames/, and for recomputing any feature set from scratch
gdown 1Ovj5Ay8rDXaPy57CNCHes0A99S43lPBy -O MUSIC-AVQA-videos-Real.zip
unzip MUSIC-AVQA-videos-Real.zip -d music_avqa_dataset/data/video

# 3. Video frames — 8 fixed frames per video (~1,196 visual tokens per sample)
python src/avqa/video_precompute.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/video_frames
```

### Recomputing the published features (optional)

Only if you do not want to download them. Both scripts need the source videos from step 2 above.

```bash
# Whisper music features. The best model uses Whisper for both paths, but this script precomputes
# only the music branch; question audio is encoded live in the forward pass.
python src/avqa/whisper_preprocess_fullres.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/whisper_features_fullres

# PANNs features — only needed to reproduce the PANNs ablations
python src/avqa/panns_preprocess.py --video_dir music_avqa_dataset/data/video \
    --out_dir music_avqa_dataset/data/panns_features \
    --checkpoint <path/to/Cnn14_mAP=0.431.pth>
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

# Stage 1 — trains the music projector against a frozen LLM.
# Let the runner build the datasets: it wires music_dir to whisper_features_fullres/ itself.
# (Constructing AVQADataset yourself without music_dir= silently defaults to panns_features/.)
model, processor = init_avqa_whisper_model(seed=42)
freeze_for_avqa_whisper_stage1(model)   # required when passing model= (the runner then skips freezing)
run_avqa_whisper_fullres_v2_stage1(model=model, processor=processor, seed=42)

# Stage 2 — restart the kernel first (see below), then load Stage 1 and add LoRA
model, processor, lora_config = setup_avqa_whisper_fullres_v2_stage2()
run_avqa_whisper_fullres_v3_stage2(model, processor, lora_config, seed=42)
```

Stage 2 produces the 97.31% model under the `whisper_fullres_v3` tag (same Stage-1 checkpoint as
v2; the v3 difference is the frozen question projector).

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
@article{dehdashti2026qwenmusicavqa,
  title   = {Qwen-MusicAVQA-7B: A Multimodal Model for Music Audio-Visual QA},
  author  = {Dehdashti, Maryam},
  journal = {arXiv preprint arXiv:2608.11329},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.11329}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
