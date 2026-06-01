# src/avqa/whisper_preprocess_fullres.py
#
# Per-chunk-pooling variant of full-duration Whisper feature extraction.
#
# Key difference from whisper_preprocess_full.py:
#   _full.py  — concatenate all chunk encoder outputs, then stride-pool the
#               entire sequence to N_FRAMES (= 32).  Resolution ∝ 1/n_chunks.
#   _fullres.py — pool each 30s chunk independently to N_FRAMES tokens, then
#               concatenate.  Resolution stays constant at 0.94s/token regardless
#               of video length.
#
# Output shape per file: float32 [n_chunks * N_FRAMES, 1280]  — VARIABLE length.
#   60s video (2 chunks) → [64, 1280]
#   30s video (1 chunk)  → [32, 1280]
#
# A durations.json sidecar is written so downstream code can compute
# n_music_tokens = n_chunks * N_FRAMES per sample.
#
# Usage (run on GPU pod):
#   python src/avqa/whisper_preprocess_fullres.py \
#       --video_dir  music_avqa_dataset/data/video/MUSIC-AVQA-videos-Real \
#       --out_dir    music_avqa_dataset/data/whisper_features_fullres \
#       --checkpoint MayaKD/qwen2-vl-audio  \
#       --n_frames   32
#
# Compare against:
#   whisper_features/      (30s cap, single chunk, 32 tokens)
#   whisper_features_full/ (full duration, concat then pool, always 32 tokens)

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers" / "src"))

WHISPER_SAMPLE_RATE = 16000
CHUNK_SECS          = 30.0          # Whisper's fixed window — do not change
N_FRAMES_DEFAULT    = 32            # tokens per chunk after pooling
D_MODEL             = 1280
N_FFT               = 400
HOP_LENGTH          = 160
MIN_TAIL_SECS       = 5.0           # drop last chunk if shorter than this
HF_REPO             = "MayaKD/qwen2-vl-audio"
DURATIONS_FILE      = "durations.json"
SAVE_EVERY          = 100


# ── audio extraction ──────────────────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install it (apt-get install ffmpeg) or "
            "pip install imageio-ffmpeg."
        )


def extract_audio(video_path: Path, target_sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Extract full mono waveform from video. Returns float32 array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                _ffmpeg_exe(), "-y", "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(target_sr),
                "-ac", "1",
                tmp_path,
            ],
            check=True,
            capture_output=True,
        )
        import soundfile as sf
        audio, sr = sf.read(tmp_path, dtype="float32")
        assert sr == target_sr
        return audio
    finally:
        os.unlink(tmp_path)


# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_audio(audio: np.ndarray, sr: int = WHISPER_SAMPLE_RATE,
                chunk_secs: float = CHUNK_SECS,
                min_tail_secs: float = MIN_TAIL_SECS) -> list[np.ndarray]:
    """Split audio into non-overlapping 30s chunks.

    Discards the last chunk if it is shorter than min_tail_secs (default 5s).
    """
    chunk_len    = int(chunk_secs * sr)
    min_tail_len = int(min_tail_secs * sr)
    chunks = []
    for start in range(0, max(len(audio), 1), chunk_len):
        chunk = audio[start : start + chunk_len]
        is_last = (start + chunk_len) >= len(audio)
        if is_last and len(chunk) < min_tail_len:
            break
        chunks.append(chunk)
    return chunks if chunks else [audio]


def _actual_enc_frames(chunk_len_samples: int) -> int:
    """Number of valid Whisper encoder frames for a chunk of given sample length."""
    mel_frames = (chunk_len_samples + N_FFT // 2) // HOP_LENGTH + 1
    return mel_frames // 2


# ── Whisper encoder ───────────────────────────────────────────────────────────

def load_whisper_encoder(checkpoint: str, device: str = "cuda"):
    """Load just the Whisper encoder + audio processor from our fine-tuned checkpoint."""
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(
        "/workspace/projects/speech/processor"
    )
    processor.audio_processor = Qwen2VLAudioProcessor()

    print(f"Loading model from {checkpoint} ...")
    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=True,
    )
    model.eval()
    encoder    = model.audio_encoder.to(device)
    audio_proc = processor.audio_processor
    del model
    torch.cuda.empty_cache()
    return encoder, audio_proc


@torch.no_grad()
def get_whisper_features_fullres(
    encoder, audio_proc, audio: np.ndarray,
    n_frames: int, device: str = "cuda"
) -> tuple[np.ndarray, float]:
    """Run Whisper encoder on full-duration audio via per-chunk pooling.

    Each 30s chunk is processed independently and pooled to n_frames tokens.
    The per-chunk pooled features are then concatenated along the time axis.

    Returns:
        features : float32 ndarray [n_chunks * n_frames, D_MODEL]
        duration : float, seconds

    Constant resolution: 0.94s / token  (= 30s / 32 frames)
    regardless of video duration.
    """
    import torch.nn.functional as F

    duration = len(audio) / WHISPER_SAMPLE_RATE
    chunks   = chunk_audio(audio)
    full_chunk_samples = int(CHUNK_SECS * WHISPER_SAMPLE_RATE)

    pooled_chunks: list[torch.Tensor] = []

    for chunk in chunks:
        out = audio_proc.preprocess(
            [{"audio": chunk, "sampling_rate": WHISPER_SAMPLE_RATE}]
        )
        input_features = torch.tensor(
            out["input_features"][0], dtype=torch.float16, device=device
        ).unsqueeze(0)  # [1, 128, T_mel]

        # Pad to Whisper's required fixed length (3000 mel frames = 30s)
        expected_len = 3000
        if input_features.shape[-1] < expected_len:
            input_features = F.pad(
                input_features, (0, expected_len - input_features.shape[-1])
            )

        enc_out = encoder(input_features).last_hidden_state  # [1, 1500, 1280]

        # Trim encoder output to frames backed by real audio (short last chunk)
        is_short = len(chunk) < full_chunk_samples
        if is_short:
            valid_frames = min(_actual_enc_frames(len(chunk)), enc_out.shape[1])
            chunk_enc = enc_out[0, :valid_frames].float()  # [valid_frames, D]
        else:
            chunk_enc = enc_out[0].float()  # [1500, D]

        # Pool this chunk independently to n_frames tokens
        T = chunk_enc.shape[0]
        if T >= n_frames:
            enc_4d = chunk_enc.unsqueeze(0).unsqueeze(0)      # [1, 1, T, D]
            pooled = F.adaptive_avg_pool2d(enc_4d, (n_frames, D_MODEL))
            pooled_chunks.append(pooled[0, 0])                # [n_frames, D]
        else:
            # Very short chunk — repeat-pad to n_frames
            repeats = (n_frames + T - 1) // T
            pooled_chunks.append(chunk_enc.repeat(repeats, 1)[:n_frames])

    result = torch.cat(pooled_chunks, dim=0)  # [n_chunks * n_frames, D]
    return result.cpu().numpy().astype(np.float32), duration


# ── main ──────────────────────────────────────────────────────────────────────

def find_json_dir(video_dir: Path) -> Path:
    """Walk up from video_dir until a sibling 'json/' directory is found."""
    candidate = video_dir
    for _ in range(4):
        candidate = candidate.parent
        json_dir = candidate / "json"
        if json_dir.is_dir() and any(json_dir.glob("avqa-*.json")):
            return json_dir
    raise FileNotFoundError(
        f"Could not find a 'json/' dir with avqa-*.json files relative to {video_dir}. "
        "Pass --json_dir explicitly."
    )


def get_all_video_ids(json_dir: Path) -> list[str]:
    ids = set()
    for fname in ["avqa-train.json", "avqa-val.json", "avqa-test.json"]:
        fpath = json_dir / fname
        if not fpath.exists():
            continue
        for item in json.load(open(fpath)):
            ids.add(item["video_id"])
    return sorted(ids)


def _load_durations(durations_path: Path) -> dict[str, float]:
    if durations_path.exists():
        return json.loads(durations_path.read_text())
    return {}


def _save_durations(durations_path: Path, durations: dict[str, float]) -> None:
    durations_path.write_text(json.dumps(durations, indent=2))


def preprocess(video_dir: Path, out_dir: Path, checkpoint: str = HF_REPO,
               n_frames: int = N_FRAMES_DEFAULT, device: str = "cuda",
               json_dir: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    durations_path = out_dir / DURATIONS_FILE

    if json_dir is None:
        json_dir = find_json_dir(video_dir)
    print(f"JSON dir: {json_dir}")
    all_ids = get_all_video_ids(json_dir)
    print(f"Total unique video IDs: {len(all_ids)}")

    todo = [vid for vid in all_ids if not (out_dir / f"{vid}.npy").exists()]
    print(f"To compute: {len(todo)}  (already done: {len(all_ids) - len(todo)})")
    print(
        f"Output shape per file: [n_chunks × {n_frames}, {D_MODEL}] float32  "
        f"(per-chunk pooled — 60s → [{2*n_frames}, {D_MODEL}], 30s → [{n_frames}, {D_MODEL}])"
    )

    if not todo:
        print("All features already cached.")
        return

    durations = _load_durations(durations_path)
    encoder, audio_proc = load_whisper_encoder(checkpoint, device)
    errors: list[str] = []

    for i, vid in enumerate(tqdm(todo, desc="Whisper encode (fullres)")):
        video_path = None
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            candidate = video_dir / f"{vid}{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            errors.append(vid)
            continue

        try:
            audio         = extract_audio(video_path)
            feats, dur    = get_whisper_features_fullres(
                encoder, audio_proc, audio, n_frames, device
            )
            np.save(out_dir / f"{vid}.npy", feats)
            durations[vid] = round(dur, 3)
        except Exception as e:
            errors.append(vid)
            tqdm.write(f"  ERROR {vid}: {e}")

        if (i + 1) % SAVE_EVERY == 0:
            _save_durations(durations_path, durations)

    _save_durations(durations_path, durations)

    n_done = len(todo) - len(errors)
    print(f"\nDone. Cached: {n_done}  Errors: {len(errors)}")
    print(f"Durations written to {durations_path}  ({len(durations)} entries)")
    if errors:
        err_file = out_dir / "errors.txt"
        err_file.write_text("\n".join(errors))
        print(f"Error video IDs saved to {err_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Per-chunk-pooling Whisper feature extraction for MUSIC-AVQA. "
            "Each 30s chunk is pooled to N_FRAMES tokens independently. "
            "Output shape is variable: [n_chunks * N_FRAMES, D_MODEL]. "
            "Constant resolution: 0.94s/token regardless of video length."
        )
    )
    parser.add_argument("--video_dir",  default="music_avqa_dataset/data/video/MUSIC-AVQA-videos-Real")
    parser.add_argument("--out_dir",    default="music_avqa_dataset/data/whisper_features_fullres")
    parser.add_argument("--checkpoint", default=HF_REPO,
                        help="HF repo or local path with fine-tuned model weights")
    parser.add_argument("--n_frames",   type=int, default=N_FRAMES_DEFAULT,
                        help="Output frames per chunk after stride-pooling (default 32)")
    parser.add_argument("--json_dir",   default=None,
                        help="Path to dir containing avqa-*.json files. "
                             "Auto-detected by walking up from video_dir if omitted.")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    preprocess(
        video_dir  = Path(args.video_dir),
        out_dir    = Path(args.out_dir),
        checkpoint = args.checkpoint,
        n_frames   = args.n_frames,
        device     = args.device,
        json_dir   = Path(args.json_dir) if args.json_dir else None,
    )
