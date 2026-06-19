# src/avqa/dataset.py
# MUSIC-AVQA PyTorch Dataset class.
# format_data() lives in src/data_utils.py (shared with all other modalities).
# fill_placeholders() lives here — imported by tts_preprocess.py too.

import json
import re
import sys
from pathlib import Path
from typing import Optional

DURATIONS_FILE = "durations.json"  # sidecar written by whisper_preprocess_fullres.py

import numpy as np
from torch.utils.data import Dataset

# ensure src/ is on path so data_utils is importable when run from project root
_avqa_src = str(Path(__file__).resolve().parents[1] / 'asr')
if _avqa_src not in sys.path:
    sys.path.append(_avqa_src)
from data_utils import format_data  # noqa: E402

# ── path defaults ─────────────────────────────────────────────────────────────

DEFAULT_JSON_DIR   = Path("music_avqa_dataset/data/json")
DEFAULT_VIDEO_DIR  = Path("music_avqa_dataset/data/video")
DEFAULT_MUSIC_DIR  = Path("music_avqa_dataset/data/panns_features")
DEFAULT_TTS_DIR    = Path("music_avqa_dataset/data/tts_questions")
DEFAULT_FRAMES_DIR = Path("music_avqa_dataset/data/video_frames")


# ── text helpers ──────────────────────────────────────────────────────────────

def fill_placeholders(text: str, templ_values_str: str) -> str:
    """Replace <Placeholder> tokens with values; normalise underscored instrument names."""
    values = json.loads(templ_values_str) if templ_values_str else []
    for val in values:
        text = re.sub(r'<[^>]+>', val, text, count=1)
    return text.replace("_", " ").strip()


# ── dataset ───────────────────────────────────────────────────────────────────

class AVQADataset(Dataset):
    """MUSIC-AVQA PyTorch dataset.

    Each __getitem__ returns:
      messages       – list of dicts (system/user/assistant) for the processor
      music_features – np.float32 [512]  (precomputed CLAP embedding)
      answer         – ground truth answer string
      question_type  – e.g. '["Audio-Visual", "Counting"]'  (for ablations)
      video_id       – str
      question_id    – int

    The user turn in messages contains:
      {"type": "video", ...}  → Qwen2-VL vision encoder (frame extraction on-the-fly)
      {"type": "music"}       → placeholder; processor expands to n_music_tokens pads
      {"type": "audio", ...}  → TTS question bytes; Whisper encoder encodes the question

    Args:
        split:         "train", "val", or "test"
        json_dir:      directory with avqa-{split}.json
        video_dir:     directory with {video_id}.mp4
        music_dir:     directory with {video_id}.npy  (CLAP embeddings)
        tts_dir:       directory with {question_id}.wav
        frames_dir:    directory with {video_id}.npy  (precomputed frames uint8 [T,H,W,3]);
                       if set and file exists, bypasses mp4 decoding at training time
        max_samples:   cap dataset size (None = use all)
        require_video: skip items whose video file is missing
        require_music: skip items whose CLAP .npy is missing
        require_tts:   skip items whose TTS .wav is missing
    """

    def __init__(
        self,
        split: str = "train",
        json_dir: Path = DEFAULT_JSON_DIR,
        video_dir: Path = DEFAULT_VIDEO_DIR,
        music_dir: Path = DEFAULT_MUSIC_DIR,
        tts_dir: Path = DEFAULT_TTS_DIR,
        frames_dir: Path = DEFAULT_FRAMES_DIR,
        max_samples: Optional[int] = None,
        video_nframes: int = 8,
        require_video: bool = True,
        require_music: bool = True,
        require_tts: bool = True,
        add_frame_timestamps: bool = False,
        text_question: bool = False,
    ):
        self.video_dir    = Path(video_dir)
        self.music_dir    = Path(music_dir)
        self.tts_dir      = Path(tts_dir)
        self.frames_dir   = Path(frames_dir) if frames_dir is not None else None
        self.video_nframes = video_nframes
        self.add_frame_timestamps = add_frame_timestamps
        self.text_question = text_question
        if text_question:
            require_tts = False  # no .wav files needed
        self._durations: dict = {}
        if add_frame_timestamps:
            dur_path = Path(music_dir) / DURATIONS_FILE
            if not dur_path.exists():
                raise FileNotFoundError(
                    f"durations.json not found at {dur_path}. "
                    "Only whisper_features_fullres/ has this sidecar — "
                    "set music_dir=WHISPER_FULLRES_MUSIC_DIR."
                )
            self._durations = json.loads(dur_path.read_text())

        raw     = json.load(open(Path(json_dir) / f"avqa-{split}.json"))
        skipped = {"video": 0, "music": 0, "tts": 0}
        self.items = []

        for item in raw:
            if item.get("question_deleted", 0):
                continue

            vid = item["video_id"]
            qid = item["question_id"]

            has_video = self._find_video(vid) is not None
            has_frames = (self.frames_dir is not None and
                          (self.frames_dir / f"{vid}.npy").exists())
            if require_video and not has_video and not has_frames:
                skipped["video"] += 1
                continue
            if require_music and not (self.music_dir / f"{vid}.npy").exists():
                skipped["music"] += 1
                continue
            if require_tts and not (self.tts_dir / f"{qid}.wav").exists():
                skipped["tts"] += 1
                continue

            self.items.append({
                "video_id":         vid,
                "question_id":      qid,
                "answer":           item["anser"],   # dataset typo — intentional
                "question_type":    item.get("type", ""),
                "question_content": item.get("question_content", ""),
                "templ_values":     item.get("templ_values", "[]"),
            })

        if max_samples is not None:
            self.items = self.items[:max_samples]
        if any(skipped.values()):
            print(f"[AVQADataset/{split}] Skipped: {skipped}")
        print(f"[AVQADataset/{split}] {len(self.items)} samples loaded")

    def _find_video(self, video_id: str) -> Optional[Path]:
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            p = self.video_dir / f"{video_id}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        vid  = item["video_id"]
        qid  = item["question_id"]

        music_features = np.load(self.music_dir / f"{vid}.npy").astype(np.float32)

        # Use precomputed frames (list of PIL Images) if available — avoids mp4 decoding.
        frames_path = self.frames_dir / f"{vid}.npy" if self.frames_dir is not None else None
        if frames_path is not None and frames_path.exists():
            from PIL import Image
            frames_arr = np.load(frames_path)          # uint8 [T, H, W, 3]
            video_val  = [Image.fromarray(frames_arr[i]) for i in range(len(frames_arr))]
        else:
            video_val = str(self._find_video(vid))

        if self.text_question:
            question_text = fill_placeholders(item["question_content"], item["templ_values"])
            sample = {
                "video":          video_val,
                "video_nframes":  self.video_nframes,
                "music":          True,
                "question_text":  question_text,   # text token path — no Whisper encoding
                "answer":         item["answer"],
            }
        else:
            tts_bytes = (self.tts_dir / f"{qid}.wav").read_bytes()
            sample = {
                "video":          video_val,
                "video_nframes":  self.video_nframes,
                "music":          True,
                "audio":          tts_bytes,        # TTS question bytes → Whisper encoder
                "answer":         item["answer"],
            }

        messages = format_data(sample)

        if self.add_frame_timestamps:
            n_audio   = music_features.shape[0]
            duration  = self._durations.get(vid, n_audio * (30.0 / 32))  # fallback: 0.9375s/token
            tok_rate  = duration / n_audio
            nf        = self.video_nframes
            stamps    = [round(duration * i / max(nf - 1, 1), 1) for i in range(nf)]
            header    = (
                f"This is a video ({round(duration)}s, {n_audio} audio tokens, "
                f"~{tok_rate:.2f}s/token).\n"
                f"Frames at {stamps}:\n"
            )
            user_turn = next(m for m in messages if m["role"] == "user")
            user_turn["content"].insert(0, {"type": "text", "text": header})

        return {
            "messages":       messages,
            "music_features": music_features,
            "answer":         item["answer"],
            "question_type":  item["question_type"],
            "video_id":       vid,
            "question_id":    qid,
        }
