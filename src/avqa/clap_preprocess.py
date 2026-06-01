# src/avqa/clap_preprocess.py
# One-time preprocessing: extract audio from each MUSIC-AVQA video,
# run CLAP (laion/larger_clap_music_and_speech), cache 512-dim embeddings as {video_id}.npy.
#
# Usage (run on GPU pod after videos are downloaded):
#   python src/avqa/clap_preprocess.py \
#       --video_dir music_avqa_dataset/data/video \
#       --out_dir   music_avqa_dataset/data/clap_features \
#       --model_id  laion/larger_clap_music_and_speech
#
# Requires: transformers, soundfile, imageio-ffmpeg (if system ffmpeg absent)

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CLAP_MODEL_ID  = "laion/larger_clap_music_and_speech"
CLAP_SAMPLE_RATE = 48000   # CLAP expects 48 kHz
CLAP_EMBED_DIM   = 512     # laion/larger_clap_music_and_speech audio embedding dim


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


def extract_audio(video_path: Path, target_sr: int = CLAP_SAMPLE_RATE) -> np.ndarray:
    """Extract mono waveform from video using ffmpeg. Returns float32 numpy array."""
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


# ── CLAP model ────────────────────────────────────────────────────────────────

def load_clap(model_id: str = CLAP_MODEL_ID, device: str = "cuda"):
    from transformers import ClapModel, ClapProcessor
    model     = ClapModel.from_pretrained(model_id).to(device).eval()
    processor = ClapProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def get_embedding(clap_model, clap_processor, audio: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Run CLAP on a mono waveform (float32, 48 kHz).
    Returns (512,) numpy embedding (L2-normalised).
    """
    inputs = clap_processor(
        audio=audio,
        return_tensors="pt",
        sampling_rate=CLAP_SAMPLE_RATE,
    ).to(device)
    audio_embed = clap_model.get_audio_features(**inputs)   # (1, 512)
    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
    return audio_embed[0].cpu().float().numpy()              # (512,)


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


def preprocess(video_dir: Path, out_dir: Path, model_id: str = CLAP_MODEL_ID,
               device: str = "cuda"):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = video_dir.parent / "json"
    all_ids  = get_all_video_ids(json_dir)
    print(f"Total unique video IDs: {len(all_ids)}")

    todo = [vid for vid in all_ids if not (out_dir / f"{vid}.npy").exists()]
    print(f"To compute: {len(todo)}  (already done: {len(all_ids) - len(todo)})")

    if not todo:
        print("All embeddings already cached.")
        return

    clap_model, clap_processor = load_clap(model_id, device)
    errors = []

    for vid in tqdm(todo, desc="CLAP embed"):
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
            emb   = get_embedding(clap_model, clap_processor, audio, device)
            np.save(out_dir / f"{vid}.npy", emb)
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
    parser.add_argument("--video_dir", default="music_avqa_dataset/data/video")
    parser.add_argument("--out_dir",   default="music_avqa_dataset/data/clap_features")
    parser.add_argument("--model_id",  default=CLAP_MODEL_ID)
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    preprocess(
        video_dir = Path(args.video_dir),
        out_dir   = Path(args.out_dir),
        model_id  = args.model_id,
        device    = args.device,
    )
