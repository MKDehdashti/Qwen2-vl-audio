"""
train.py — Fine-tune Qwen2.5-Omni-7B on MUSIC-AVQA.

Single LoRA stage (no projector warmup — all modalities already aligned in Omni).
LoRA is applied to model.thinker only; the talker (speech synthesis) is untouched.

model.thinker = multimodal LLM (vision encoder + audio encoder + 7B LM) — generates text.
model.talker  = speech synthesis head — unused for MUSIC-AVQA text output.

Usage from scratch.ipynb:
    import sys; sys.path.insert(0, '/workspace/projects/speech/src/avqa/omni')
    from train import init_omni, run_omni_training, evaluate_omni
    model, processor, lora_config = init_omni()
    trainer = run_omni_training(model, processor, lora_config)
"""
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspace/projects/speech/transformers/src')
sys.path.insert(0, '/workspace/projects/speech/qwen-vl-utils/src')
sys.path.insert(0, '/workspace/projects/speech/src/asr')
sys.path.insert(0, '/workspace/projects/speech/src/avqa/omni')

import torch
import wandb
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from collator import OmniCollator, make_messages
from dataset import AVQADatasetOmni
from wandb_utils import init_wandb, WandBLossCallback, save_wandb_run_id, load_wandb_run_id

MODEL_ID        = 'Qwen/Qwen2.5-Omni-7B'
HF_REPO         = 'MayaKD/qwen2-vl-audio'
CHECKPOINT_BASE = '/workspace/projects/speech/avqa_omni_checkpoint'


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


# ── model loading ─────────────────────────────────────────────────────────────

def init_omni():
    """Load Qwen2.5-Omni-7B and apply LoRA r=64 to the thinker's attention layers.

    The thinker is the multimodal LLM (vision + audio encoder + language model).
    The talker (speech synthesis) is frozen and unused for MUSIC-AVQA text output.

    Returns (model, processor, lora_config).
    """
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
    )
    from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor

    print(f'Loading {MODEL_ID} ...')
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    # Ensure thinker's name_or_path is the full HF model ID, not the submodule path
    # (thinker.config.name_or_path can end up as "Qwen2.5-Omni-7B/thinker" which is invalid on HF Hub)
    model.thinker.config.name_or_path = MODEL_ID
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()

    return model, processor, lora_config


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_omni(
    model,
    processor,
    dataset: AVQADatasetOmni,
    n: int | None = None,
    tag: str = 'eval',
    step: int = 0,
) -> dict:
    """Exact-match accuracy on MUSIC-AVQA.

    Uses model.thinker.generate() for text output.
    Scoring identical to eval_utils._normalise: lowercase + strip punctuation.
    """
    from tqdm import tqdm
    from qwen_vl_utils import process_audio_info, process_vision_info

    n_items = min(n, len(dataset)) if n is not None else len(dataset)
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    n_correct = 0

    model.eval()
    device = next(model.thinker.parameters()).device

    for i in tqdm(range(n_items), desc=f'[omni_eval/{tag}]', unit='sample'):

        item = dataset[i]
        msgs = make_messages(item['frames'], item['video_audio'], item['tts_path'])
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, video_inputs = process_vision_info(msgs)
        audio_raw       = process_audio_info(msgs)
        audio_inputs    = [arr for arr, _ in audio_raw] if audio_raw else None

        inputs = processor(
            text=[text],
            videos=video_inputs if video_inputs else None,
            audio=audio_inputs,
            return_tensors='pt',
        )
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.thinker.generate(**inputs, max_new_tokens=5, do_sample=False)

        generated = out_ids[0][inputs['input_ids'].shape[1]:]
        pred    = _normalise(processor.tokenizer.decode(generated, skip_special_tokens=True))
        gt      = _normalise(item['answer'])
        correct = int(pred == gt)
        n_correct += correct

        try:
            qtype = ' / '.join(json.loads(item['question_type']))
        except Exception:
            qtype = str(item['question_type'])
        per_type[qtype][0] += correct
        per_type[qtype][1] += 1

    if n_items == 0:
        return {'accuracy': 0.0, 'n_correct': 0, 'n_total': 0, 'per_type': {}}

    overall = n_correct / n_items * 100
    print(f'\n[{tag}] Overall: {overall:.2f}%  ({n_correct}/{n_items})')
    for qtype, (c, t) in sorted(per_type.items()):
        print(f'  {qtype:<42} {c / t * 100:.2f}%  ({c}/{t})')

    if wandb.run is not None:
        wandb.log({f'{tag}/accuracy': overall / 100, f'{tag}/n_correct': n_correct}, step=step)

    return {
        'accuracy': overall / 100,
        'n_correct': n_correct,
        'n_total': n_items,
        'per_type': {k: v[0] / v[1] for k, v in per_type.items()},
    }


# ── eval callback ─────────────────────────────────────────────────────────────

class OmniEvalCallback(TrainerCallback):
    def __init__(self, model, processor, val_dataset: AVQADatasetOmni, n: int = 50, tag: str = 'epoch_eval'):
        self.model       = model
        self.processor   = processor
        self.val_dataset = val_dataset
        self.n           = n
        self.tag         = tag

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        evaluate_omni(self.model, self.processor, self.val_dataset,
                      n=self.n, tag=self.tag, step=state.global_step)
        self.model.thinker.train()


# ── training ──────────────────────────────────────────────────────────────────

def run_omni_training(
    model=None,
    processor=None,
    lora_config=None,
    train_dataset=None,
    val_dataset=None,
    output_dir=None,
    resume_from_checkpoint=None,
) -> SFTTrainer:
    """LoRA fine-tune model.thinker on MUSIC-AVQA.

    Call with no args:
        trainer = run_omni_training()
    Returns the trainer so you can push / inspect after training.
    """
    if model is None or processor is None or lora_config is None:
        model, processor, lora_config = init_omni()

    if train_dataset is None:
        train_dataset = AVQADatasetOmni('train', max_samples=8000)
        val_dataset   = AVQADatasetOmni('val')
    test_dataset = AVQADatasetOmni('test')

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f'{CHECKPOINT_BASE}_{run_id}'

    # 8K / (2×8) = 500 steps/epoch → save_steps=100 fires 5×/epoch
    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     2,   # ~2071 tokens/sample; batch=4 likely OOMs
        'per_device_eval_batch_size':      2,
        'gradient_accumulation_steps':     8,   # keeps effective batch = 16
        'learning_rate':                   2e-5,
        'warmup_steps':                    50,
        'lr_scheduler_type':               'linear',
        'bf16':                            True,
        'tf32':                            True,
        'optim':                           'adamw_torch_fused',
        'max_grad_norm':                   0.3,
        'max_seq_length':                  2048,
        'dataloader_num_workers':          4,
        'dataloader_pin_memory':           True,
        'logging_steps':                   10,
        'eval_strategy':                   'no',      # OmniEvalCallback handles accuracy; SFT loss eval is slow (collator re-runs processor 3×/batch)
        'eval_steps':                      100,
        'save_strategy':                   'steps',
        'save_steps':                      100,
        'save_total_limit':                1,
        'load_best_model_at_end':          False,
        'gradient_checkpointing':          True,
        'gradient_checkpointing_kwargs':   {'use_reentrant': False},
        'remove_unused_columns':           False,
        'report_to':                       'wandb',
        'dataset_kwargs':                  {'skip_prepare_dataset': True},
    })

    init_wandb(run_id, sft_config, lora_config=lora_config,
               stage_name='avqa_omni', exp_name='qwen2-vl-avqa-omni',
               model_id=MODEL_ID,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    collator = OmniCollator(processor=processor)

    evaluate_omni(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model.thinker,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
        callbacks=[
            WandBLossCallback(),
            OmniEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_omni(model, processor, val_dataset,
                  n=200, tag='final_val', step=trainer.state.global_step)
    evaluate_omni(model, processor, test_dataset,
                  n=len(test_dataset), tag='final_test', step=trainer.state.global_step)

    push_omni(trainer)

    if wandb.run is not None:
        wandb.finish()

    return trainer


# ── HF push ───────────────────────────────────────────────────────────────────

def push_omni(trainer: SFTTrainer, subfolder: str = 'avqa_omni') -> None:
    """Push LoRA adapter weights to HF repo."""
    import tempfile
    from huggingface_hub import HfApi
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.model.save_pretrained(tmp_dir)
        HfApi().upload_folder(folder_path=tmp_dir, repo_id=HF_REPO, path_in_repo=subfolder)
    print(f'Pushed to {HF_REPO}/{subfolder}')
