# src/avqa/panns_preprocess.py
# One-time preprocessing: extract audio from each MUSIC-AVQA video,
# run PANNs CNN14, cache 2048-dim embeddings as {video_id}.npy.
#
# Usage (run on GPU pod after videos are downloaded):
#   python src/avqa/panns_preprocess.py \
#       --video_dir music_avqa_dataset/data/video \
#       --out_dir   music_avqa_dataset/data/panns_features \
#       --workers   4
#
# Requires: panns_inference, ffmpeg, soundfile, librosa (or torchaudio)

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PANNS_SAMPLE_RATE = 32000   # CNN14 expects 32 kHz
PANNS_EMBED_DIM   = 2048    # CNN14 embedding dimension


# ── audio extraction ─────────────────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    """Return path to ffmpeg binary (system or imageio-ffmpeg fallback)."""
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


def extract_audio(video_path: Path, target_sr: int = PANNS_SAMPLE_RATE) -> np.ndarray:
    """Extract mono waveform from video using ffmpeg. Returns float32 numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                _ffmpeg_exe(), "-y", "-i", str(video_path),
                "-vn",                          # no video
                "-acodec", "pcm_s16le",
                "-ar", str(target_sr),
                "-ac", "1",                     # mono
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


# ── PANNs model ───────────────────────────────────────────────────────────────

def load_panns(checkpoint_path: str, device: str = "cuda"):
    """Load PANNs CNN14 AudioTagging model."""
    from panns_inference import AudioTagging
    model = AudioTagging(checkpoint_path=checkpoint_path, device=device)
    model.model.eval()
    return model


@torch.no_grad()
def get_embedding(panns_model, audio: np.ndarray) -> np.ndarray:
    """
    Run PANNs on a single mono waveform (float32, 32 kHz).
    Returns (2048,) numpy embedding.
    """
    # panns_inference.AudioTagging.inference expects (batch, time) numpy
    audio_batch = audio[np.newaxis, :]          # (1, T)
    _, embedding = panns_model.inference(audio_batch)
    return embedding[0]                          # (2048,)


# ── main ──────────────────────────────────────────────────────────────────────

def get_all_video_ids(json_dir: Path) -> list[str]:
    """Collect all unique video_ids from train/val/test JSON files."""
    ids = set()
    for fname in ["avqa-train.json", "avqa-val.json", "avqa-test.json"]:
        fpath = json_dir / fname
        if not fpath.exists():
            continue
        for item in json.load(open(fpath)):
            ids.add(item["video_id"])
    return sorted(ids)


def preprocess(video_dir: Path, out_dir: Path, panns_checkpoint: str,
               workers: int = 1, device: str = "cuda"):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = video_dir.parent / "json"
    all_ids  = get_all_video_ids(json_dir)
    print(f"Total unique video IDs: {len(all_ids)}")

    # Skip already-computed
    todo = [vid for vid in all_ids if not (out_dir / f"{vid}.npy").exists()]
    print(f"To compute: {len(todo)}  (already done: {len(all_ids) - len(todo)})")

    if not todo:
        print("All embeddings already cached.")
        return

    panns = load_panns(panns_checkpoint, device=device)
    errors = []

    for vid in tqdm(todo, desc="PANNs embed"):
        # Videos may be .mp4 or .mkv — find whichever exists
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
            audio = extract_audio(video_path)
            emb   = get_embedding(panns, audio)
            np.save(out_dir / f"{vid}.npy", emb.astype(np.float32))
        except Exception as e:
            errors.append(vid)
            tqdm.write(f"  ERROR {vid}: {e}")

    print(f"\nDone. Cached: {len(todo) - len(errors)}  Errors: {len(errors)}")
    if errors:
        err_file = out_dir / "errors.txt"
        err_file.write_text("\n".join(errors))
        print(f"Error video IDs saved to {err_file}")


def load_embedding(video_id: str, panns_dir: Path) -> np.ndarray | None:
    """Load a cached PANNs embedding. Returns None if not found."""
    p = panns_dir / f"{video_id}.npy"
    if not p.exists():
        return None
    return np.load(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",  default="music_avqa_dataset/data/video")
    parser.add_argument("--out_dir",    default="music_avqa_dataset/data/panns_features")
    parser.add_argument("--checkpoint", default="music_avqa_dataset/pretrained/Cnn14_mAP=0.431.pth",
                        help="PANNs CNN14 checkpoint (.pth)")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--workers",    type=int, default=1)
    args = parser.parse_args()

    preprocess(
        video_dir        = Path(args.video_dir),
        out_dir          = Path(args.out_dir),
        panns_checkpoint = args.checkpoint,
        workers          = args.workers,
        device           = args.device,
    )
