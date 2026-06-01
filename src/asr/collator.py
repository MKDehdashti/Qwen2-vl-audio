"""
collator.py — Data collator for audio+text SFT training.

Label-mask strategy (left padding, padding_side="left"):
  With left-padding the batch looks like:
      row i: [PAD ... PAD | sys_tokens | user_tokens | assistant_tokens]
  Assistant tokens are always the LAST `assistant_len` tokens of each row:
      labels[i, : max_len - assistant_len] = -100
  assistant_len = actual_seq_len_i - user_len_i

  user_len_i fast path (audio-only samples):
      tokenizer(user_text) gives unexpanded length (1 <|audio_pad|> per clip)
      user_len = unexpanded_len - n_clips + sum(audio_lengths_for_sample_i)
      audio_lengths come from the batch processor call — no mel recomputation.

  user_len_i fallback (samples with images or video):
      full per-sample processor call (re-computes vision token counts).
"""

import sys

sys.path.insert(0, "/workspace/projects/speech/transformers/src")
sys.path.insert(0, "/workspace/projects/speech/qwen-vl-utils/src")

from dataclasses import dataclass
from typing import Any

from qwen_vl_utils import process_audio_info, process_vision_info

from data_utils import format_data


@dataclass
class DataCollator:
    """Collate raw dataset samples into left-padded model-ready batches.

    Supports audio, image, video, and mixed-modality samples via format_data().

    Args:
        processor: Qwen2VLProcessor with audio_processor attached.
    """

    processor: Any  # Qwen2VLProcessor

    def __call__(self, samples: list[dict]) -> dict[str, Any]:
        import torch
        texts: list[str] = []
        all_messages: list[list] = []   # cached — reused for label masking
        all_audio_inputs: list[list] = []
        all_image_inputs: list = []
        all_video_inputs: list = []

        # Detect pre-computed mel spectrograms (added by preprocess_audio_features).
        use_precomputed_mel = "mel_feat" in samples[0]

        # Detect music features (AVQA samples from AVQADataset).
        # Single-vector encoders (PANNs/CLAP/MERT): each sample is [D]  → stack to [B, D]
        # Fixed-T sequence encoders (Whisper-32):    each sample is [T, D] → stack to [B, T, D]
        # Variable-T sequence encoders (whisper_fullres): each sample is [T_i, D] → concat to
        #     [total_T, D] and record music_lengths = [T_0, T_1, ...] for the processor.
        has_music = "music_features" in samples[0]
        music_lengths: list[int] | None = None
        if has_music:
            import numpy as np
            first_feat = samples[0]["music_features"]
            if first_feat.ndim == 2:
                # Sequence encoder: always pass music_lengths so processor uses
                # actual T instead of falling back to self.n_music_tokens (default 8).
                Ts = [s["music_features"].shape[0] for s in samples]
                music_features = torch.from_numpy(
                    np.concatenate([s["music_features"] for s in samples], axis=0)
                ).float()
                music_lengths = Ts
            else:
                # Single-vector [D]: stack → [B, D]
                music_features = torch.from_numpy(
                    np.stack([s["music_features"] for s in samples])
                ).float()
        else:
            music_features = None

        for sample in samples:
            messages = format_data(sample)
            all_messages.append(messages)  # cache: avoids double format_data + fetch_audio
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            # Skip fetch_audio when mel is pre-computed — only need audio_inputs for
            # label-mask user_len calculation (fast path uses audio_lengths from batch).
            all_audio_inputs.append(
                None if use_precomputed_mel else process_audio_info(messages)
            )
            img_inputs, vid_inputs = process_vision_info(messages)
            all_image_inputs.append(img_inputs)   # None or list of PIL Images
            all_video_inputs.append(vid_inputs)   # None or list of tensors

        flat_images  = [img for sub in all_image_inputs for img in (sub or [])]
        flat_videos  = [vid for sub in all_video_inputs for vid in (sub or [])]

        if use_precomputed_mel:
            import numpy as np
            # Batch-pad per-sample mel spectrograms to the longest T in this batch.
            feats = [torch.as_tensor(s["mel_feat"], dtype=torch.float32) for s in samples]
            max_T = max(f.shape[1] for f in feats)
            padded_mels = torch.zeros(len(feats), feats[0].shape[0], max_T)
            for i, f in enumerate(feats):
                padded_mels[i, :, :f.shape[1]] = f
            audio_lengths = torch.tensor(
                [s["audio_length"] for s in samples], dtype=torch.long
            )
            # processor call: tokenise + expand audio pad tokens (no mel recomputation)
            inputs = self.processor(
                text=texts,
                precomputed_input_features=padded_mels,
                precomputed_audio_lengths=audio_lengths,
                images=flat_images      if flat_images      else None,
                videos=flat_videos      if flat_videos      else None,
                music_features=music_features,
                music_lengths=music_lengths,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            )
        else:
            flat_audios = [a for sub in all_audio_inputs for a in (sub or [])]
            inputs = self.processor(
                text=texts,
                audios=flat_audios      if flat_audios      else None,
                images=flat_images      if flat_images      else None,
                videos=flat_videos      if flat_videos      else None,
                music_features=music_features,
                music_lengths=music_lengths,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            )
        input_ids = inputs["input_ids"]   # [B, max_len]
        max_len = input_ids.shape[1]
        pad_id = self.processor.tokenizer.pad_token_id

        # Infer un-padded length per row from first non-pad position
        actual_lengths: list[int] = []
        for i in range(len(samples)):
            non_pad = (input_ids[i] != pad_id).long()
            first_real = int(non_pad.argmax().item())
            actual_lengths.append(max_len - first_real)

        # Slice batch audio_lengths into per-sample lists so we can compute
        # user_len without re-running mel spectrogram computation.
        # audio_lengths: [total_n_audios] — flat tensor from the processor call.
        batch_audio_lengths = inputs.get("audio_lengths")
        audio_offset = 0
        per_sample_audio_lengths: list = []
        if use_precomputed_mel:
            # Each sample has exactly 1 audio clip; lengths are stored per-sample.
            import torch
            for s in samples:
                per_sample_audio_lengths.append(
                    torch.tensor([s["audio_length"]], dtype=torch.long)
                )
        else:
            for sub in all_audio_inputs:
                n = len(sub) if sub else 0
                if n > 0 and batch_audio_lengths is not None:
                    per_sample_audio_lengths.append(
                        batch_audio_lengths[audio_offset : audio_offset + n]
                    )
                else:
                    per_sample_audio_lengths.append(None)
                audio_offset += n

        # Build labels: -100 for padding + user prompt; real ids for assistant
        labels = input_ids.clone()
        for i in range(len(samples)):
            messages = all_messages[i]  # use cache — no re-call to format_data/fetch_audio
            user_msgs = [m for m in messages if m["role"] != "assistant"]
            user_text = self.processor.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True
            )

            has_vision = (
                all_image_inputs[i] is not None or all_video_inputs[i] is not None
            )

            if not has_vision:
                # Fast path: tokenizer-only (no mel recomputation)
                # user_text has 1 <|audio_pad|> placeholder per clip.
                # After batch expansion each placeholder → audio_lengths[j] tokens.
                user_len = len(
                    self.processor.tokenizer.encode(user_text, add_special_tokens=False)
                )
                n_audio = 1 if use_precomputed_mel else len(all_audio_inputs[i] or [])
                if n_audio > 0 and per_sample_audio_lengths[i] is not None:
                    user_len = (
                        user_len
                        - n_audio
                        + int(per_sample_audio_lengths[i].sum().item())
                    )
            else:
                # Fallback for samples with images/videos: vision token counts
                # are not tracked separately, so re-run the processor.
                # Pass music_features[i] so <|music_pad|> is expanded correctly.
                user_audio = all_audio_inputs[i]
                # Extract per-sample music_features slice and per-sample music_lengths
                if has_music:
                    if music_lengths is not None:
                        # Variable-T: extract this sample's slice from the flat tensor
                        offset = sum(music_lengths[:i])
                        T_i = music_lengths[i]
                        mf_i = music_features[offset:offset + T_i]  # [T_i, D]
                        ml_i = [T_i]
                    elif music_features.dim() == 3:
                        mf_i = music_features[i:i+1]   # [1, T, D] fixed-T
                        ml_i = None
                    else:
                        mf_i = music_features[i:i+1]   # [1, D] single-vector
                        ml_i = None
                else:
                    mf_i = None
                    ml_i = None
                user_inputs = self.processor(
                    text=[user_text],
                    audios=user_audio              if user_audio              else None,
                    images=all_image_inputs[i]     if all_image_inputs[i]     else None,
                    videos=all_video_inputs[i]     if all_video_inputs[i]     else None,
                    music_features=mf_i,
                    music_lengths=ml_i,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                user_len = user_inputs["input_ids"].shape[1]

            assistant_len = actual_lengths[i] - user_len
            labels[i, : max_len - assistant_len] = -100

        # Mask all modality special tokens in labels as a safety net
        # (these appear in the user turn and are already covered by prompt masking,
        # but explicit masking guards against off-by-one errors)
        tok = self.processor.tokenizer
        special_ids = {
            tok.convert_tokens_to_ids("<|audio_start|>"),
            tok.convert_tokens_to_ids("<|audio_pad|>"),
            tok.convert_tokens_to_ids("<|audio_end|>"),
            tok.convert_tokens_to_ids("<|music_start|>"),
            tok.convert_tokens_to_ids("<|music_pad|>"),
            tok.convert_tokens_to_ids("<|music_end|>"),
            tok.convert_tokens_to_ids("<|vision_start|>"),
            tok.convert_tokens_to_ids("<|vision_end|>"),
            tok.convert_tokens_to_ids("<|image_pad|>"),
            tok.convert_tokens_to_ids("<|video_pad|>"),
        }
        for tid in special_ids:
            if tid is not None:
                labels[labels == tid] = -100

        inputs["labels"] = labels
        return dict(inputs)
