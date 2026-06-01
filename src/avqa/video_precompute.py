# src/avqa/video_precompute.py
# One-time preprocessing: extract N frames from each MUSIC-AVQA video and
# cache them as uint8 numpy arrays so training skips mp4 decoding entirely.
#
# Usage:
#   python src/avqa/video_precompute.py \
#       --video_dir  music_avqa_dataset/data/video \
#       --out_dir    music_avqa_dataset/data/video_frames \
#       --nframes    8  --workers 4
#
# Output: {out_dir}/{video_id}.npy  — uint8 [T, H, W, 3] at Qwen2-VL resolution
# Resume-safe: skips already-cached files.
#
# Uses ffmpeg (via imageio-ffmpeg) to seek directly to frame timestamps —
# avoids reading the entire video file. ~10-50x faster than torchvision.io.read_video.

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen-vl-utils" / "src"))
from qwen_vl_utils.vision_process import (
    FRAME_FACTOR, VIDEO_MIN_PIXELS, VIDEO_TOTAL_PIXELS, VIDEO_MAX_PIXELS,
    smart_resize,
)


def _ffmpeg_exe() -> str:
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _video_duration(video_path: Path, ffmpeg: str) -> float:
    """Return video duration in seconds using ffprobe/ffmpeg."""
    result = subprocess.run(
        [ffmpeg, "-i", str(video_path)],
        capture_output=True, text=True,
    )
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            parts = line.strip().split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Could not parse duration from {video_path}")


def _extract_one_frame(video_path: Path, timestamp: float, ffmpeg: str) -> np.ndarray:
    """Seek to timestamp and extract one frame as RGB uint8 numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-ss", f"{timestamp:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                tmp_path,
            ],
            check=True,
            capture_output=True,
        )
        return np.array(Image.open(tmp_path).convert("RGB"))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_frames(video_path: Path, nframes: int = 8, ffmpeg: str = None) -> np.ndarray:
    """Extract nframes evenly-spaced frames. Returns uint8 [T, H, W, 3] at Qwen2-VL res."""
    if ffmpeg is None:
        ffmpeg = _ffmpeg_exe()

    duration = _video_duration(video_path, ffmpeg)
    timestamps = [duration * i / (nframes - 1) for i in range(nframes)]
    # avoid seeking past end
    timestamps[-1] = min(timestamps[-1], duration - 0.1)

    raw_frames = [_extract_one_frame(video_path, t, ffmpeg) for t in timestamps]

    # resize to Qwen2-VL dynamic resolution
    h, w = raw_frames[0].shape[:2]
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR),
        VIDEO_MIN_PIXELS * 1.05,
    )
    rh, rw = smart_resize(h, w, factor=FRAME_FACTOR,
                           min_pixels=VIDEO_MIN_PIXELS, max_pixels=max_pixels)

    resized = [
        np.array(Image.fromarray(f).resize((rw, rh), Image.BICUBIC))
        for f in raw_frames
    ]
    return np.stack(resized).astype(np.uint8)  # [T, H, W, 3]


def _process_one(args):
    vid, video_path, out_path, nframes, ffmpeg = args
    try:
        frames = extract_frames(video_path, nframes=nframes, ffmpeg=ffmpeg)
        np.save(out_path, frames)
        return vid, None
    except Exception as e:
        return vid, str(e)


def get_all_video_ids(json_dir: Path) -> list[str]:
    ids = set()
    for fname in ["avqa-train.json", "avqa-val.json", "avqa-test.json"]:
        fpath = json_dir / fname
        if not fpath.exists():
            continue
        for item in json.load(open(fpath)):
            ids.add(item["video_id"])
    return sorted(ids)


def precompute(video_dir: Path, out_dir: Path, nframes: int = 8, workers: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = video_dir.parent / "json"
    all_ids  = get_all_video_ids(json_dir)
    ffmpeg   = _ffmpeg_exe()

    todo = []
    for vid in all_ids:
        out_path = out_dir / f"{vid}.npy"
        if out_path.exists():
            continue
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            candidate = video_dir / f"{vid}{ext}"
            if candidate.exists():
                todo.append((vid, candidate, out_path, nframes, ffmpeg))
                break

    done_already = len(all_ids) - len(todo) - sum(
        1 for vid in all_ids
        if not (out_dir / f"{vid}.npy").exists()
        and not any((video_dir / f"{vid}{ext}").exists()
                    for ext in [".mp4", ".mkv", ".webm", ".avi"])
    )
    print(f"Total: {len(all_ids)}  To compute: {len(todo)}  "
          f"Already done: {len(all_ids) - len(todo)}  Workers: {workers}")

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, arg): arg[0] for arg in todo}
        with tqdm(total=len(todo), desc="frames") as pbar:
            for future in as_completed(futures):
                vid, err = future.result()
                if err:
                    errors.append(vid)
                    tqdm.write(f"  ERROR {vid}: {err}")
                pbar.update(1)

    print(f"\nDone. Cached: {len(todo) - len(errors)}  Errors: {len(errors)}")
    if errors:
        (out_dir / "errors.txt").write_text("\n".join(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="music_avqa_dataset/data/video")
    parser.add_argument("--out_dir",   default="music_avqa_dataset/data/video_frames")
    parser.add_argument("--nframes",   type=int, default=8)
    parser.add_argument("--workers",   type=int, default=4)
    args = parser.parse_args()
    precompute(Path(args.video_dir), Path(args.out_dir), args.nframes, args.workers)
