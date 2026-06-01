"""
init_model.py — Initialize Qwen2VLAudioForConditionalGeneration from pretrained weights.

Steps:
  1. Load Qwen/Qwen2-VL-7B-Instruct weights into Qwen2VLAudioForConditionalGeneration
     (audio_encoder + audio_projector start randomly initialized)
  2. Load openai/whisper-large-v3-turbo encoder weights into model.audio_encoder
     (only audio_projector remains random after this)
  3. Save model + processor locally to SAVE_DIR
  4. Push to HF repo
  5. Patch config.json: set _name_or_path and ensure audio_token_id is present

Usage:
  source /workspace/projects/speech/.venv/bin/activate
  source /workspace/projects/speech/.secrets
  python /workspace/projects/speech/src/init_model.py
"""

import sys, os, gc, json, torch
sys.path.insert(0, "/workspace/projects/speech/transformers/src")
sys.path.insert(0, "/workspace/projects/speech/qwen-vl-utils/src")

from huggingface_hub import login, HfApi

HF_REPO  = "MayaKD/qwen2-vl-audio"
SAVE_DIR = "/workspace/projects/speech/model_checkpoint"

login(token=os.environ.get("HF_TOKEN"))  # falls back to cached token if env var not set

# ── 1. Load Qwen2VL-7B-Instruct ───────────────────────────────────────────
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAudioForConditionalGeneration

print("Loading Qwen/Qwen2-VL-7B-Instruct …")
model = Qwen2VLAudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    # audio_encoder + audio_projector keys absent in checkpoint → stay randomly initialized
)
print("  model.model + lm_head:           loaded from Qwen2VL-7B-Instruct")
print("  audio_encoder + audio_projector: randomly initialized")

# ── 2. Load Whisper Turbo encoder ─────────────────────────────────────────
from transformers import WhisperForConditionalGeneration

print("Loading openai/whisper-large-v3-turbo encoder …")
whisper = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo",
    torch_dtype=torch.bfloat16,
)
missing, unexpected = model.audio_encoder.load_state_dict(
    whisper.model.encoder.state_dict(), strict=True
)
assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
del whisper
gc.collect()
print("  audio_encoder:    loaded from whisper-large-v3-turbo")
print("  audio_projector:  still randomly initialized (trained in Stage 1)")

# ── 3. Save locally ───────────────────────────────────────────────────────
from transformers import Qwen2VLProcessor
from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor

proc = Qwen2VLProcessor.from_pretrained("/workspace/projects/speech/processor")
proc.audio_processor = Qwen2VLAudioProcessor()

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
proc.save_pretrained(SAVE_DIR)
print(f"Saved to {SAVE_DIR}")

# ── 4. Push to HF ─────────────────────────────────────────────────────────
model.push_to_hub(HF_REPO)
proc.push_to_hub(HF_REPO)
print(f"Pushed model + processor to https://huggingface.co/{HF_REPO}")

# ── 5. Patch config.json ──────────────────────────────────────────────────
config_path = os.path.join(SAVE_DIR, "config.json")
with open(config_path) as f:
    cfg = json.load(f)

cfg["_name_or_path"] = HF_REPO
cfg["audio_token_id"] = 151658  # <|audio_pad|>

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

HfApi().upload_file(
    path_or_fileobj=config_path,
    path_in_repo="config.json",
    repo_id=HF_REPO,
)
print("config.json patched and pushed.")
print("Done.")
