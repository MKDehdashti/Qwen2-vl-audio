# src/avqa/whisper_preprocess.py
# One-time preprocessing: extract audio from each MUSIC-AVQA video, run through
# the fine-tuned Whisper encoder (from our HF checkpoint), stride-pool the output
# sequence to N_FRAMES frames, and cache as {video_id}.npy of shape [N_FRAMES, D_MODEL].
#
# This produces the music features for the Whisper-as-music-encoder ablation.
# The per-frame [N_FRAMES, D_MODEL] arrays preserve temporal order — unlike mean-pooled
# single-vector encoders (PANNs/CLAP/MERT), temporal structure is retained.
#
# Usage (run on GPU pod):
#   python src/avqa/whisper_preprocess.py \
#       --video_dir  music_avqa_dataset/data/video \
#       --out_dir    music_avqa_dataset/data/whisper_features \
#       --checkpoint MayaKD/qwen2-vl-audio  \
#       --n_frames   32
#
# Output shape per file: float32 [N_FRAMES, 1280]  (1280 = Whisper d_model)
# Requires: transformers (our fork), soundfile, imageio-ffmpeg (if system ffmpeg absent)

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

WHISPER_SAMPLE_RATE = 16000    # Whisper expects 16 kHz
N_FRAMES_DEFAULT    = 32       # stride-pool target
D_MODEL             = 1280     # Whisper encoder output dim (confirmed from config.audio_config.d_model)
HF_REPO             = "MayaKD/qwen2-vl-audio"


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


def extract_audio(video_path: Path, target_sr: int = WHISPER_SAMPLE_RATE,
                  max_secs: float = 30.0) -> np.ndarray:
    """Extract mono waveform from video (up to max_secs). Returns float32 array."""
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
                "-t", str(max_secs),
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


# ── Whisper encoder ───────────────────────────────────────────────────────────

def load_whisper_encoder(checkpoint: str, device: str = "cuda"):
    """Load just the Whisper encoder + audio processor from our fine-tuned checkpoint."""
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(
        '/workspace/projects/speech/processor'
    )
    processor.audio_processor = Qwen2VLAudioProcessor()

    print(f"Loading model from {checkpoint} ...")
    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation='flash_attention_2',
        ignore_mismatched_sizes=True,
    )
    model.eval()
    # We only need the audio encoder weights — discard LLM to save VRAM
    encoder = model.audio_encoder.to(device)
    audio_proc = processor.audio_processor
    del model
    torch.cuda.empty_cache()
    return encoder, audio_proc


@torch.no_grad()
def get_whisper_features(encoder, audio_proc, audio: np.ndarray,
                         n_frames: int, device: str = "cuda") -> np.ndarray:
    """
    Run Whisper encoder on a mono 16 kHz waveform, stride-pool to n_frames.

    Args:
        audio:    float32 numpy array, mono 16 kHz
        n_frames: number of output frames after stride pooling (e.g. 32)

    Returns:
        float32 numpy array of shape [n_frames, D_MODEL]
    """
    import torch.nn.functional as F

    # Compute mel spectrogram via the audio processor
    out = audio_proc.preprocess([{"audio": audio, "sampling_rate": WHISPER_SAMPLE_RATE}])
    input_features = torch.tensor(
        out["input_features"][0], dtype=torch.float16, device=device
    ).unsqueeze(0)   # [1, 128, T]

    # Encoder forward: [1, T_enc, D_MODEL] — T_enc proportional to actual audio duration
    encoder_out = encoder(input_features).last_hidden_state   # [1, T_enc, 1280]
    T_enc = encoder_out.shape[1]

    # Stride-pool: divide T_enc into n_frames equal-sized bins, mean-pool each bin.
    # This preserves temporal order while compressing 1500 → 32 frames.
    if T_enc >= n_frames:
        # Reshape to [1, n_frames, T_enc//n_frames (approx), D] via adaptive avg pool
        enc = encoder_out[0].float()                           # [T_enc, D]
        enc = enc.unsqueeze(0).unsqueeze(0)                    # [1, 1, T_enc, D]
        # Pool along time axis: [1, 1, n_frames, D]
        pooled = F.adaptive_avg_pool2d(enc, (n_frames, D_MODEL))
        result = pooled[0, 0]                                  # [n_frames, D]
    else:
        # Audio shorter than n_frames (very short clip) — repeat-pad
        enc = encoder_out[0].float()                           # [T_enc, D]
        repeats = (n_frames + T_enc - 1) // T_enc
        result = enc.repeat(repeats, 1)[:n_frames]             # [n_frames, D]

    return result.cpu().numpy().astype(np.float32)             # [n_frames, 1280]


# ── main ──────────────────────────────────────────────────────────────────────

def get_all_video_ids(json_dir: Path) -> list[str]:
    ids = set()
    for fname in ["avqa-train.json", "avqa-val.json", "avqa-test.json"]:
        fpath = json_dir / fname
        if not fpath.exists():
            continue
        for item in json.load(open(fpath)):
            ids.add(item["video_id"])
    return sorted(ids)


def preprocess(video_dir: Path, out_dir: Path, checkpoint: str = HF_REPO,
               n_frames: int = N_FRAMES_DEFAULT, device: str = "cuda"):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = video_dir.parent / "json"
    all_ids  = get_all_video_ids(json_dir)
    print(f"Total unique video IDs: {len(all_ids)}")

    todo = [vid for vid in all_ids if not (out_dir / f"{vid}.npy").exists()]
    print(f"To compute: {len(todo)}  (already done: {len(all_ids) - len(todo)})")
    print(f"Output shape per file: [{n_frames}, {D_MODEL}] float32")

    if not todo:
        print("All features already cached.")
        return

    encoder, audio_proc = load_whisper_encoder(checkpoint, device)
    errors = []

    for vid in tqdm(todo, desc="Whisper encode"):
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
            audio  = extract_audio(video_path)
            feats  = get_whisper_features(encoder, audio_proc, audio, n_frames, device)
            np.save(out_dir / f"{vid}.npy", feats)
        except Exception as e:
            errors.append(vid)
            tqdm.write(f"  ERROR {vid}: {e}")

    print(f"\nDone. Cached: {len(todo) - len(errors)}  Errors: {len(errors)}")
    if errors:
        err_file = out_dir / "errors.txt"
        err_file.write_text("\n".join(errors))
        print(f"Error video IDs saved to {err_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",  default="music_avqa_dataset/data/video")
    parser.add_argument("--out_dir",    default="music_avqa_dataset/data/whisper_features")
    parser.add_argument("--checkpoint", default=HF_REPO,
                        help="HF repo or local path with fine-tuned model weights")
    parser.add_argument("--n_frames",   type=int, default=N_FRAMES_DEFAULT,
                        help="Number of output frames after stride pooling (default 32)")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    preprocess(
        video_dir  = Path(args.video_dir),
        out_dir    = Path(args.out_dir),
        checkpoint = args.checkpoint,
        n_frames   = args.n_frames,
        device     = args.device,
    )
