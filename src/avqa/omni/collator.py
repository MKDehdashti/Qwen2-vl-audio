"""
collator.py — Data collator for Qwen2.5-Omni MUSIC-AVQA fine-tuning.

Input samples: {"frames": list[PIL.Image], "video_audio": np.ndarray, "tts_path": str, ...}

Chat format mirrors our Qwen2-VL model exactly:
  system: SYSTEM_PROMPT
  user:   [8 precomputed frames]  [video audio (music)]  [TTS audio (question)]  Answer the question.
  asst:   {answer}

Two audio elements per sample — processed in order by the Omni processor:
  1. video audio (music track, ~30s → ~750 audio tokens) — same role as our music encoder
  2. TTS audio   (spoken question, ~3–5s → ~75 audio tokens) — same role as our Whisper encoder

Label mask (left-padded batches):
  -100 for all system + user tokens; real token ids for assistant tokens only.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, '/workspace/projects/speech/qwen-vl-utils/src')
from qwen_vl_utils import process_audio_info, process_vision_info  # noqa: E402

SYSTEM_PROMPT = (
    "You are an audio-visual question answering assistant. "
    "Answer with a single word or short phrase."
)


def make_messages(
    frames,
    video_audio: np.ndarray,
    tts_path: str,
    answer: str | None = None,
) -> list[dict]:
    """Build Omni chat messages mirroring our model's format.

    frames:      list[PIL.Image] — precomputed frames (same 8 frames as our model)
    video_audio: np.ndarray     — video audio track at 16kHz mono (music, same source as our model)
    tts_path:    str            — path to spoken question WAV (same TTS files as our model)
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [
            {"type": "video", "video": frames},                                      # precomputed frames
            {"type": "audio", "audio": video_audio, "sampling_rate": 16_000},       # music audio track
            {"type": "audio", "audio": tts_path},                                   # spoken question
            {"type": "text",  "text": "Answer the question."},
        ]},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


@dataclass
class OmniCollator:
    """Collate AVQA samples into left-padded Qwen2.5-Omni batches with labels."""

    processor: Any  # Qwen2_5OmniProcessor

    def __call__(self, samples: list[dict]) -> dict[str, Any]:
        import torch

        texts_full:  list[str] = []
        texts_user:  list[str] = []
        vids_full:   list      = []
        vids_user:   list      = []
        auds_full:   list      = []
        auds_user:   list      = []

        for s in samples:
            msgs_full = make_messages(s['frames'], s['video_audio'], s['tts_path'], s['answer'])
            msgs_user = make_messages(s['frames'], s['video_audio'], s['tts_path'])

            texts_full.append(
                self.processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
            )
            texts_user.append(
                self.processor.apply_chat_template(msgs_user, tokenize=False, add_generation_prompt=True)
            )

            # Assistant turn has no audio/video — full and user vision/audio are identical.
            # Compute once and reuse for both the batch call and per-sample label masking.
            _, vid = process_vision_info(msgs_user)
            # process_audio_info returns [(video_audio_arr, 16000), (tts_arr, sr)] in message order
            aud    = process_audio_info(msgs_user)

            vids_full.append(vid)
            vids_user.append(vid)
            auds_full.append(aud)
            auds_user.append(aud)

        flat_vids_full = [v for sub in vids_full for v in (sub or [])]
        # Extract waveforms from (array, sr) tuples returned by process_audio_info
        flat_auds_full = [arr for sub in auds_full for arr, _ in (sub or [])]

        # Process full batch (system + user + assistant)
        inputs = self.processor(
            text=texts_full,
            videos=flat_vids_full if flat_vids_full else None,
            audio=flat_auds_full if flat_auds_full else None,
            return_tensors='pt',
            padding=True,
            pad_to_multiple_of=8,
            add_special_tokens=False,
        )

        input_ids = inputs['input_ids']   # [B, max_len]
        max_len   = input_ids.shape[1]
        pad_id    = self.processor.tokenizer.pad_token_id

        # Actual (non-padded) length per row — left-padded so scan from left.
        # nonzero() is used instead of argmax() because argmax() returns 0 for
        # all-ones tensors (the longest, unpadded row), masking silent mis-fires
        # if pad_id ever coincides with a real content token id.
        actual_lengths = []
        for i in range(len(samples)):
            non_pad    = (input_ids[i] != pad_id)
            nz         = non_pad.nonzero(as_tuple=False)
            first_real = int(nz[0, 0].item()) if len(nz) else 0
            actual_lengths.append(max_len - first_real)

        # Build labels: -100 everywhere except assistant tokens
        labels = input_ids.clone()
        for i in range(len(samples)):
            # Re-run processor on user-only messages to get exact prompt token count.
            # Video + audio token counts are variable; can't infer from text alone.
            flat_vid_i = vids_user[i] if vids_user[i] else None
            flat_aud_i = [arr for arr, _ in auds_user[i]] if auds_user[i] else None
            user_inputs = self.processor(
                text=[texts_user[i]],
                videos=flat_vid_i,
                audio=flat_aud_i,
                return_tensors='pt',
                add_special_tokens=False,
            )
            user_len      = user_inputs['input_ids'].shape[1]
            assistant_len = actual_lengths[i] - user_len
            if assistant_len <= 0:
                # Tokenization mismatch — mask entire row rather than train on corrupt labels.
                labels[i] = -100
                continue
            labels[i, : max_len - assistant_len] = -100

        # Mask modality special tokens as a safety net (Omni token names)
        tok = self.processor.tokenizer
        for token_name in (
            '<|AUDIO|>', '<|audio_bos|>', '<|audio_eos|>',
            '<|IMAGE|>', '<|VIDEO|>',
            '<|vision_bos|>', '<|vision_eos|>', '<|vision_pad|>',
        ):
            tid = tok.convert_tokens_to_ids(token_name)
            if tid is not None and tid != tok.unk_token_id:
                labels[labels == tid] = -100

        inputs['labels'] = labels
        return dict(inputs)
