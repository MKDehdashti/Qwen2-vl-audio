# create_processor.py
import os
from pathlib import Path

from huggingface_hub import HfApi, login, whoami
from transformers import Qwen2VLProcessor


# -------------------------
# Secrets loader (self-contained)
# -------------------------
def load_secrets(secrets_path: str = "/workspace/projects/speech/.secrets") -> None:
    p = Path(secrets_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        # override to avoid stale/bad env values
        os.environ[k.strip()] = v.strip()


# -------------------------
# Main
# -------------------------
SAVE_DIR = "/workspace/projects/speech/processor"
BASE_PROCESSOR = "Qwen/Qwen2-VL-7B-Instruct"
REPO_NAME = "qwen2-vl-audio"  # repo name under your user

load_secrets()

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) not found. Add it to .secrets.")

login(token=HF_TOKEN)
me = whoami()
user = me.get("name")
if not user:
    raise RuntimeError(f"Could not determine Hugging Face username from whoami(): {me}")

HF_REPO = f"{user}/{REPO_NAME}"
print("HF logged in as:", user)
print("Target repo:", HF_REPO)

processor = Qwen2VLProcessor.from_pretrained(BASE_PROCESSOR)

# Add audio special tokens (ASR) + music special tokens (AVQA)
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": [
        "<|audio_start|>", "<|audio_pad|>", "<|audio_end|>",   # ASR / TTS questions
        "<|music_start|>", "<|music_pad|>", "<|music_end|>",   # AVQA music track
    ]}
)
# Expected IDs after addition:
#   <|audio_start|> 151657  <|audio_pad|> 151658  <|audio_end|> 151659
#   <|music_start|> 151660  <|music_pad|> 151661  <|music_end|> 151662

# Chat template: handles audio (ASR / TTS questions) and music (video audio track)
AUDIO_CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% set audio_count = namespace(value=0) %}"
    "{% set music_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content.get('type') == 'image' or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content.get('type') == 'video' %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif content.get('type') == 'audio' %}"
    "{% set audio_count.value = audio_count.value + 1 %}"
    "<|audio_start|><|audio_pad|><|audio_end|>"
    "{% elif content.get('type') == 'music' %}"
    "{% set music_count.value = music_count.value + 1 %}"
    "<|music_start|><|music_pad|><|music_end|>"
    "{% elif 'text' in content %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

processor.tokenizer.chat_template = AUDIO_CHAT_TEMPLATE

# Save locally
os.makedirs(SAVE_DIR, exist_ok=True)
processor.save_pretrained(SAVE_DIR)
# Overwrite the .jinja file — save_pretrained copies the original from cache
Path(SAVE_DIR, "chat_template.jinja").write_text(AUDIO_CHAT_TEMPLATE)
print("Saved locally at:", SAVE_DIR)

# Create + upload to HF
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=SAVE_DIR, repo_id=HF_REPO, repo_type="model")

print("Uploaded processor to:", HF_REPO)
