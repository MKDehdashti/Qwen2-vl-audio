"""
dataset.py — MUSIC-AVQA dataset for Qwen2.5-Omni fine-tuning.

Mirrors our Qwen2-VL model's input format for a fair comparison:
  - precomputed frames (list[PIL.Image]) → Omni's native video encoder
  - video audio track (np.ndarray, 16kHz mono) → Omni's native audio encoder (music)
  - TTS WAV      → Omni's native audio encoder  (spoken question, same files as our model)
  - "Answer the question." → plain text         (same static prompt as our model)

Frames are precomputed by src/avqa/video_precompute.py → (8, H, W, 3) uint8 .npy per video.
Video audio is loaded from the original .mp4 at __getitem__ time (ffmpeg + soundfile, 16kHz mono).
TTS files are precomputed by src/avqa/tts_preprocess.py — nothing to regenerate.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.append(_src)
from avqa.dataset import fill_placeholders  # noqa: E402

DATA_DIR           = Path('/workspace/projects/speech/music_avqa_dataset/data')
DEFAULT_JSON_DIR   = DATA_DIR / 'json'
DEFAULT_FRAMES_DIR = DATA_DIR / 'video_frames'
DEFAULT_VIDEO_DIR  = DATA_DIR / 'video' / 'MUSIC-AVQA-videos-Real'
DEFAULT_TTS_DIR    = DATA_DIR / 'tts_questions'


def _ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found — install it or pip install imageio-ffmpeg")


def _load_video_audio(video_path: str, target_sr: int = 16_000) -> np.ndarray:
    """Extract mono float32 waveform from video via ffmpeg + soundfile (mirrors whisper_preprocess_fullres.py)."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
             tmp_path],
            check=True, capture_output=True,
        )
        audio, _ = sf.read(tmp_path, dtype="float32")
        return audio
    finally:
        os.unlink(tmp_path)


class AVQADatasetOmni(Dataset):
    """
    Each __getitem__ returns a plain dict:
        frames:        list[PIL.Image] — 8 precomputed RGB frames (from uint8 .npy)
        video_audio:   np.ndarray     — audio track at 16kHz mono (loaded from .mp4)
        tts_path:      str — absolute path to .wav  (spoken question, precomputed TTS)
        question:      str — filled question text    (for reference / logging only)
        answer:        str — ground truth (one of 42 closed-form answers)
        question_type: str — JSON-encoded e.g. '["Audio-Visual", "Counting"]'
        video_id:      str
        question_id:   int
    """

    def __init__(
        self,
        split: str = 'train',
        json_dir: Path = DEFAULT_JSON_DIR,
        frames_dir: Path = DEFAULT_FRAMES_DIR,
        video_dir: Path = DEFAULT_VIDEO_DIR,
        tts_dir: Path = DEFAULT_TTS_DIR,
        max_samples: Optional[int] = None,
    ):
        self.frames_dir = Path(frames_dir)
        self.video_dir  = Path(video_dir)
        self.tts_dir    = Path(tts_dir)

        raw = json.loads((Path(json_dir) / f'avqa-{split}.json').read_text())
        self.items: list[dict] = []
        skipped: dict[str, int] = {'frames': 0, 'video': 0, 'tts': 0}

        for item in raw:
            if item.get('question_deleted', 0):
                continue
            vid = item['video_id']
            qid = item['question_id']

            if not (self.frames_dir / f'{vid}.npy').exists():
                skipped['frames'] += 1
                continue
            if not (self.video_dir / f'{vid}.mp4').exists():
                skipped['video'] += 1
                continue
            if not (self.tts_dir / f'{qid}.wav').exists():
                skipped['tts'] += 1
                continue

            self.items.append({
                'frames_path':   str(self.frames_dir / f'{vid}.npy'),
                'video_path':    str(self.video_dir  / f'{vid}.mp4'),
                'tts_path':      str(self.tts_dir    / f'{qid}.wav'),
                'question':      fill_placeholders(item['question_content'], item['templ_values']),
                'answer':        item['anser'],   # dataset typo — intentional
                'question_type': item.get('type', ''),
                'video_id':      vid,
                'question_id':   qid,
            })

        if max_samples is not None:
            self.items = self.items[:max_samples]

        if any(skipped.values()):
            print(f'[AVQADatasetOmni/{split}] Skipped: {skipped}')
        print(f'[AVQADatasetOmni/{split}] {len(self.items)} samples loaded')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        frames_arr  = np.load(item['frames_path'])              # uint8 [T, H, W, 3]
        frames      = [Image.fromarray(frames_arr[i]) for i in range(len(frames_arr))]
        video_audio = _load_video_audio(item['video_path'])     # float32 [samples] at 16kHz
        return {**item, 'frames': frames, 'video_audio': video_audio}
