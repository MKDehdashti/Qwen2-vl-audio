from datasets import load_dataset

DATASET_NAME = "speechbrain/LargeScaleASR"


def _compute_mel(sample):
    """Module-level so pickle can find it by name → num_proc > 1 works."""
    import numpy as np
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    if not hasattr(_compute_mel, '_ap'):
        _compute_mel._ap = Qwen2VLAudioProcessor()
    out = _compute_mel._ap.preprocess([{"audio": sample["wav"]["bytes"]}])
    feat = np.array(out["input_features"][0], dtype=np.float32)  # [128, T]
    return {"mel_feat": feat, "audio_length": int(out["audio_lengths"][0])}


# ── Pool-based mel precompute (bypasses HF dataset.map entirely) ──────────────

_worker_ap = None  # one per worker process; set by _mp_init


def _mp_init():
    """Pool worker initializer: create Qwen2VLAudioProcessor once per process."""
    import sys
    sys.path.insert(0, '/workspace/projects/speech/transformers/src')
    sys.path.insert(0, '/workspace/projects/speech/qwen-vl-utils/src')
    global _worker_ap
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    _worker_ap = Qwen2VLAudioProcessor()


def _mp_compute(audio_bytes):
    """Compute mel for one clip. Called by pool.imap; returns (feat, length)."""
    import numpy as np
    out = _worker_ap.preprocess([{"audio": audio_bytes}])
    feat = np.array(out["input_features"][0], dtype=np.float32)  # [128, T]
    return feat, int(out["audio_lengths"][0])


class MelCachedDataset:
    """Wraps an HF Dataset with precomputed mel features stored in RAM.

    Returned samples are plain dicts with all original keys plus:
      mel_feat     – np.float32 [128, T]  (unpadded, real frames only)
      audio_length – int  (number of <|audio_pad|> tokens this clip expands to)

    DataLoader workers on Linux inherit the lists via fork — no extra IPC.
    """

    def __init__(self, hf_dataset, mel_feats, audio_lengths):
        self._dataset = hf_dataset
        self._mel_feats = mel_feats        # list[np.ndarray]
        self._audio_lengths = audio_lengths  # list[int]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = dict(self._dataset[idx])
        sample["mel_feat"] = self._mel_feats[idx]
        sample["audio_length"] = self._audio_lengths[idx]
        return sample


MEL_CACHE_DIR = "/workspace/.cache/mel_precompute"


def preprocess_audio_features_mp(dataset, n_workers=8, cache_dir=MEL_CACHE_DIR):
    """Precompute mel spectrograms via multiprocessing.Pool.

    Caches results to cache_dir keyed by the HF dataset fingerprint.
    On subsequent calls with the same dataset, loads from disk (~10s) instead
    of recomputing (~12 min).

    Steps printed at runtime:
      [1/3] Reading audio bytes  — silent before, now has tqdm
      [2/3] Computing mel        — Pool.imap with tqdm
      [3/3] Saving / loading cache

    Returns a MelCachedDataset (indexable, has __len__).
    """
    import multiprocessing
    import os
    import time
    import numpy as np
    from tqdm.auto import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    fingerprint = getattr(dataset, "_fingerprint", None) or f"n{len(dataset)}"
    mel_path = os.path.join(cache_dir, f"{fingerprint}_mel.npy")
    len_path = os.path.join(cache_dir, f"{fingerprint}_lengths.npy")

    if os.path.exists(mel_path) and os.path.exists(len_path):
        print(f"[mel cache] Found cache for fingerprint={fingerprint}")
        print(f"[mel cache] Loading from {cache_dir} ...")
        t0 = time.time()
        mel_feats = list(np.load(mel_path, allow_pickle=True))
        audio_lengths = list(np.load(len_path))
        print(f"[mel cache] Loaded {len(mel_feats)} samples in {time.time()-t0:.1f}s")
        return MelCachedDataset(dataset, mel_feats, audio_lengths)

    n = len(dataset)
    print(f"[mel precompute 1/3] Reading {n} audio byte arrays from Arrow ...")
    t0 = time.time()
    audio_bytes = [
        s["wav"]["bytes"]
        for s in tqdm(dataset, desc="  reading bytes", ncols=80)
    ]
    print(f"[mel precompute 1/3] Done in {time.time()-t0:.1f}s")

    # spawn context: fresh process per worker → no inherited CUDA state → no deadlock
    print(f"[mel precompute 2/3] Computing mel with {n_workers} workers ...")
    t0 = time.time()
    mel_feats, audio_lengths = [], []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_mp_init) as pool:
        for feat, length in tqdm(
            pool.imap(_mp_compute, audio_bytes, chunksize=64),
            total=n,
            desc="  mel compute",
            ncols=80,
        ):
            mel_feats.append(feat)
            audio_lengths.append(length)
    print(f"[mel precompute 2/3] Done in {time.time()-t0:.1f}s")

    print(f"[mel precompute 3/3] Saving cache (key={fingerprint}) ...")
    t0 = time.time()
    np.save(mel_path, np.array(mel_feats, dtype=object), allow_pickle=True)
    np.save(len_path, np.array(audio_lengths))
    print(f"[mel precompute 3/3] Saved in {time.time()-t0:.1f}s → {cache_dir}")

    return MelCachedDataset(dataset, mel_feats, audio_lengths)

# Stage 1: projector-only training doesn't need the full 322K-sample split.
# 20K samples ≈ 1.3h vs 20h for the full set.
TRAIN_SIZE   = 20_000
# Filter out clips longer than MAX_AUDIO_SECS to keep sequences short.
# LargeScaleASR stores audio as {"bytes": ..., "sampling_rate": ...}.
MAX_AUDIO_SECS = 15.0


def _short_enough(sample):
    """Return True if the audio clip is ≤ MAX_AUDIO_SECS."""
    import io, soundfile as sf
    info = sf.info(io.BytesIO(sample["wav"]["bytes"]))
    return info.duration <= MAX_AUDIO_SECS


def preprocess_audio_features(dataset, audio_processor):
    """Pre-compute mel spectrograms once; HF datasets caches the result on disk.

    Adds two columns to each sample:
      mel_feat    – numpy float32 [128, T]  (unpadded, real frames only)
      audio_length – int  (number of <|audio_pad|> tokens this clip expands to)

    num_proc=8 — _compute_mel is module-level so pickle can find it by name.
    Run time: ~10 min for 20K samples on 8 cores; cached after first call.
    """
    return dataset.map(
        _compute_mel,
        num_proc=8,
        desc="Precomputing mel spectrograms",
        new_fingerprint="mel_precomputed_v1",  # bypass dill hashing of audio_processor
        writer_batch_size=200,                 # flush to disk every 200 samples
    )


def load_datasets():
    """Load train and test splits of LargeScaleASR (non-streaming, for training)."""
    train = (
        load_dataset(DATASET_NAME, "small", num_proc=12)["train"]
        .filter(_short_enough, num_proc=12)
        .select(range(TRAIN_SIZE))
    )
    test = load_dataset(
        DATASET_NAME,
        data_files=["test/test-00000*"],
        num_proc=12,
    )["train"].select(range(100))
    return train, test


def format_data(sample):
    """Convert a dataset sample to OpenAI conversation format.

    Handles mixed-modality samples — all present modalities are included in
    the same user turn, so a sample with 'wav' + 'image' + 'video' keys
    produces a single message with audio, image, and video content blocks.

    Supported keys (all optional, at least one required):
      - 'messages' → pre-formatted, returned as-is (ignores all other keys)
      - 'wav'      → audio bytes  (speechbrain/LargeScaleASR: wav["bytes"])
      - 'audio'    → {"array": np.ndarray, "sampling_rate": int}  (LibriSpeech-style)
                     or raw bytes
      - 'image'    → PIL.Image, file path, URL, base64 URI, or bytes
      - 'video'    → local file path or list of PIL frames
      - 'music'    → truthy flag; inserts a <|music_start|>...<|music_end|> placeholder block.
                     Actual PANNs features are passed separately as music_features kwarg to
                     the model/processor — this key is just a signal to include the block.

    Ground truth (assistant turn) comes from the first present key among:
      'text', 'caption', 'label', 'answer'  (falls back to empty string)

    Text prompt is chosen by modality combination:
      audio only       → ASR transcription prompt
      image only       → image description prompt
      video only       → video description prompt
      music present    → AVQA prompt (answer the spoken question about the video)
      mixed (no music) → generic multimodal prompt

    Raises ValueError if none of the modality keys are found.
    """
    # Already in conversation format (e.g. pre-processed HF dataset)
    if "messages" in sample:
        return sample["messages"]

    # Collect all modality content blocks present in this sample
    user_content = []

    has_audio = "wav" in sample or "audio" in sample
    has_image = "image" in sample
    has_video = "video" in sample
    has_music = bool(sample.get("music"))   # PANNs music track placeholder

    if has_video:
        video_item = {"type": "video", "video": sample["video"]}
        if "video_nframes" in sample:
            video_item["nframes"] = sample["video_nframes"]
        user_content.append(video_item)
    if has_music:
        # Placeholder token block — processor expands to n_music_tokens <|music_pad|> tokens.
        # PANNs embedding is passed separately as music_features= to the processor/model.
        user_content.append({"type": "music"})
    if has_audio:
        if "wav" in sample:
            user_content.append({"type": "audio", "audio": sample["wav"]["bytes"]})
        else:
            # LibriSpeech-style: may be decoded {"array": np.ndarray, "sampling_rate": int}
            # or undecoded {"bytes": bytes, "path": str} (cast_column decode=False).
            # fetch_audio handles both np.ndarray and raw bytes via soundfile.
            a = sample["audio"]
            if isinstance(a, dict):
                if "array" in a:
                    user_content.append({"type": "audio", "audio": a["array"],
                                         "sampling_rate": a["sampling_rate"]})
                else:
                    user_content.append({"type": "audio", "audio": a["bytes"]})
            else:
                # raw bytes passed directly
                user_content.append({"type": "audio", "audio": a})
    if has_image:
        user_content.append({"type": "image", "image": sample["image"]})

    if not user_content:
        raise ValueError(
            f"Cannot infer modality from sample keys: {list(sample.keys())}. "
            "Expected one of: 'messages', 'wav', 'audio', 'image', 'video', 'music'."
        )

    # Choose system message and user prompt based on modalities present.
    if has_music:
        # Audio-visual QA: question is spoken (audio), music track and video are context.
        system = (
            "You are an audio-visual question answering assistant. "
            "Answer with a single word or short phrase."
        )
        prompt = "Answer the question."
    elif sum([has_audio, has_image, has_video]) > 1:
        system = "You are a multimodal assistant."
        prompt = "Describe what you see and hear."
    elif has_audio:
        system = ("You are an ASR model that transcribes speech to text. "
                  "Avoid additional explanation unless absolutely necessary.")
        prompt = "Transcribe the speech into text."
    elif has_image:
        system = "You are a vision assistant."
        prompt = "Describe the image."
    else:
        system = "You are a video assistant."
        prompt = "Describe the video."

    user_content.append({"type": "text", "text": prompt})

    label = (sample.get("text") or sample.get("caption")
             or sample.get("label") or sample.get("answer", ""))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": str(label)},
    ]
