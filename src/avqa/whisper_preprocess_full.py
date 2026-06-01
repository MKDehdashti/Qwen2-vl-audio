# src/avqa/whisper_preprocess_full.py
#
# Full-duration variant of whisper_preprocess.py.
#
# Key difference: audio is NOT truncated at 30s. Instead the full audio is split
# into non-overlapping 30-second chunks, the Whisper encoder runs on each chunk,
# and the encoder outputs are concatenated before stride-pooling to N_FRAMES.
#
# Output shape per file: float32 [N_FRAMES, 1280]  — identical to whisper_features/
# but covering the full video duration rather than only the first 30s.
#
# A duration sidecar (out_dir/durations.json) is also written so downstream code
# can map Whisper frame i → timestamp  i * (duration / N_FRAMES).
#
# Usage (run on GPU pod):
#   python src/avqa/whisper_preprocess_full.py \
#       --video_dir  music_avqa_dataset/data/video \
#       --out_dir    music_avqa_dataset/data/whisper_features_full \
#       --checkpoint MayaKD/qwen2-vl-audio  \
#       --n_frames   32
#
# Output dir:  whisper_features_full/{video_id}.npy
#              whisper_features_full/durations.json   {video_id: duration_secs}
#
# Compare against: whisper_features/ (30s cap, produced by whisper_preprocess.py)

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
N_FRAMES_DEFAULT    = 32
D_MODEL             = 1280
N_FFT               = 400           # Whisper frontend (hop = 160, n_fft = 400)
HOP_LENGTH          = 160
MIN_TAIL_SECS       = 5.0           # drop last chunk if shorter than this
HF_REPO             = "MayaKD/qwen2-vl-audio"
DURATIONS_FILE      = "durations.json"
SAVE_EVERY          = 100           # write durations.json checkpoint every N videos


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
    """Extract full mono waveform from video. Returns float32 array.

    No duration cap — unlike whisper_preprocess.py which truncates at 30s.
    """
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

    The last chunk is kept only if it is >= min_tail_secs (default 5s).
    Shorter tails are discarded — too small for reliable STFT computation and
    contribute negligible temporal content to the final pooled features.
    """
    chunk_len    = int(chunk_secs * sr)
    min_tail_len = int(min_tail_secs * sr)
    chunks = []
    for start in range(0, max(len(audio), 1), chunk_len):
        chunk = audio[start : start + chunk_len]
        is_last = (start + chunk_len) >= len(audio)
        if is_last and len(chunk) < min_tail_len:
            break  # discard short tail
        chunks.append(chunk)
    return chunks if chunks else [audio]  # fallback: keep at least one chunk


def _actual_enc_frames(chunk_len_samples: int) -> int:
    """Number of valid Whisper encoder frames for a chunk of given sample length.

    Whisper frontend: mel_frames = (n_samples + n_fft//2) // hop + 1
    Encoder conv stride halves this: enc_frames = mel_frames // 2
    """
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
    encoder   = model.audio_encoder.to(device)
    audio_proc = processor.audio_processor
    del model
    torch.cuda.empty_cache()
    return encoder, audio_proc


@torch.no_grad()
def get_whisper_features_full(
    encoder, audio_proc, audio: np.ndarray,
    n_frames: int, device: str = "cuda"
) -> tuple[np.ndarray, float]:
    """Run Whisper encoder on full-duration audio via 30s chunking.

    Each 30s chunk is processed independently. Encoder outputs are trimmed to
    actual content frames (avoiding encoder output over zero-padding for the last
    chunk) and concatenated along the time axis. The full sequence is then
    stride-pooled to n_frames so the output shape is identical to
    whisper_preprocess.py but temporal coverage spans the whole video.

    Returns:
        features : float32 ndarray [n_frames, D_MODEL]
        duration : float, seconds
    """
    import torch.nn.functional as F

    duration = len(audio) / WHISPER_SAMPLE_RATE
    chunks   = chunk_audio(audio)
    full_chunk_samples = int(CHUNK_SECS * WHISPER_SAMPLE_RATE)

    enc_outputs: list[torch.Tensor] = []

    for i, chunk in enumerate(chunks):
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

        # For a short last chunk, trim encoder output to frames backed by real audio.
        # For full-length chunks every frame is valid.
        is_short = len(chunk) < full_chunk_samples
        if is_short:
            valid_frames = min(_actual_enc_frames(len(chunk)), enc_out.shape[1])
            enc_outputs.append(enc_out[0, :valid_frames].float())
        else:
            enc_outputs.append(enc_out[0].float())  # all 1500 frames

    full_enc = torch.cat(enc_outputs, dim=0)  # [T_total, 1280]
    T_total  = full_enc.shape[0]

    # Stride-pool T_total → n_frames via adaptive average pooling
    if T_total >= n_frames:
        enc    = full_enc.unsqueeze(0).unsqueeze(0)       # [1, 1, T_total, D]
        pooled = F.adaptive_avg_pool2d(enc, (n_frames, D_MODEL))
        result = pooled[0, 0]                             # [n_frames, D]
    else:
        # Very short clip (T_total < n_frames) — repeat-pad then trim
        repeats = (n_frames + T_total - 1) // T_total
        result  = full_enc.repeat(repeats, 1)[:n_frames]

    return result.cpu().numpy().astype(np.float32), duration


# ── main ──────────────────────────────────────────────────────────────────────

def find_json_dir(video_dir: Path) -> Path:
    """Walk up from video_dir until a sibling 'json/' directory is found."""
    candidate = video_dir
    for _ in range(4):  # max 4 levels up
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
    all_ids  = get_all_video_ids(json_dir)
    print(f"Total unique video IDs: {len(all_ids)}")

    # Resume: skip videos already computed
    todo = [vid for vid in all_ids if not (out_dir / f"{vid}.npy").exists()]
    print(f"To compute: {len(todo)}  (already done: {len(all_ids) - len(todo)})")
    print(f"Output shape per file: [{n_frames}, {D_MODEL}] float32  (full-duration chunked)")

    if not todo:
        print("All features already cached.")
        return

    # Load existing durations so a resumed run doesn't lose previous entries
    durations = _load_durations(durations_path)

    encoder, audio_proc = load_whisper_encoder(checkpoint, device)
    errors: list[str] = []

    for i, vid in enumerate(tqdm(todo, desc="Whisper encode (full-dur)")):
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
            feats, dur    = get_whisper_features_full(encoder, audio_proc, audio, n_frames, device)
            np.save(out_dir / f"{vid}.npy", feats)
            durations[vid] = round(dur, 3)
        except Exception as e:
            errors.append(vid)
            tqdm.write(f"  ERROR {vid}: {e}")

        # Checkpoint durations periodically so progress isn't lost on interruption
        if (i + 1) % SAVE_EVERY == 0:
            _save_durations(durations_path, durations)

    # Final save
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
            "Full-duration Whisper feature extraction for MUSIC-AVQA. "
            "Unlike whisper_preprocess.py, audio is not truncated at 30s — "
            "the full video audio is chunked and processed completely."
        )
    )
    parser.add_argument("--video_dir",  default="music_avqa_dataset/data/video")
    parser.add_argument("--out_dir",    default="music_avqa_dataset/data/whisper_features_full")
    parser.add_argument("--checkpoint", default=HF_REPO,
                        help="HF repo or local path with fine-tuned model weights")
    parser.add_argument("--n_frames",   type=int, default=N_FRAMES_DEFAULT,
                        help="Output frames after stride-pooling (default 32)")
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
