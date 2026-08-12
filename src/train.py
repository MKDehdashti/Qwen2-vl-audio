"""
train.py — Fine-tuning helpers for Qwen2VLAudioForConditionalGeneration.

Stage 1: freeze_for_stage1(), run_stage1()
Stage 2: setup_stage2(), run_stage2()  — uncomment after Stage 1 looks good.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspace/projects/speech/transformers/src')
sys.path.insert(0, '/workspace/projects/speech/qwen-vl-utils/src')
sys.path.insert(0, '/workspace/projects/speech/src')
sys.path.append('/workspace/projects/speech/src/asr')

import torch
import wandb
from trl import SFTConfig, SFTTrainer
from huggingface_hub import HfApi

from collator import DataCollator
from wandb_utils import init_wandb, WandBLossCallback, save_wandb_run_id, load_wandb_run_id
from eval_utils import evaluate_asr, ASREvalCallback

STAGE1_BASE_DIR = '/workspace/projects/speech/stage1_checkpoint'
HF_REPO         = 'MayaKD/qwen2-vl-audio'
# The repo ROOT is now the merged AVQA model (Qwen-MusicAVQA-7B). Every AVQA run must
# initialize from the ASR Stage-2 merge, which moved to this subfolder in the 2026-08-11
# reorganization. Loading the root instead would silently start from a fine-tuned model.
ASR_MERGE_SUBFOLDER = 'asr/merged_stage2'

# HF layout (see CLAUDE.md Conventions). Uploads must land in the tree, not the old flat namespace.
_HEADLINE = {'whisper_fullres_v2': 'avqa/headline/stage1',
             'whisper_fullres_v3': 'avqa/headline/stage2_qproj_frozen'}
def avqa_hf_path(experiment_tag: str, stage: int) -> str:
    """Destination subfolder for an AVQA run's checkpoint."""
    if experiment_tag == 'whisper_fullres_v2':
        return _HEADLINE['whisper_fullres_v2'] if stage == 1 else 'avqa/headline/stage2_qproj_tuned'
    if experiment_tag == 'whisper_fullres_v3':
        return 'avqa/headline/stage1' if stage == 1 else _HEADLINE['whisper_fullres_v3']
    if experiment_tag.startswith('whisper_fullres_v3_seed'):
        return f"avqa/seeds/{experiment_tag.split('_')[-1]}/stage{stage}"
    return f'avqa/ablations/{experiment_tag}/stage{stage}'


def freeze_for_stage1(model):
    """Freeze all parameters except audio_projector."""
    for name, param in model.named_parameters():
        param.requires_grad = 'audio_projector' in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.4f}%)")


def run_stage1(model=None, processor=None, train_dataset=None, test_dataset=None,
               output_dir=None, resume_from_checkpoint=None):
    """Load everything if not provided, then train Stage 1.

    Can be called with no arguments from scratch:
        trainer = run_stage1()

    Or pass already-loaded objects to skip reloading:
        trainer = run_stage1(model, processor, train_dataset, test_dataset)

    output_dir defaults to stage1_checkpoint_YYYYMMDD_HHMMSS so each run
    gets its own folder. The W&B run name uses the same timestamp.

    Returns the trainer so you can inspect / push afterwards.
    """
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAudioForConditionalGeneration
    from data_utils import load_datasets

    if processor is None:
        processor = Qwen2VLProcessor.from_pretrained(
            '/workspace/projects/speech/processor')
        processor.audio_processor = Qwen2VLAudioProcessor()

    if model is None:
        model = Qwen2VLAudioForConditionalGeneration.from_pretrained(
            'MayaKD/qwen2-vl-audio',
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation='flash_attention_2',
        )
        freeze_for_stage1(model)

    if train_dataset is None:
        train_dataset, test_dataset = load_datasets()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = f"{STAGE1_BASE_DIR}_{run_id}"

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                6,
        'per_device_train_batch_size':     8,
        'per_device_eval_batch_size':      8,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   1e-4,
        'warmup_steps':                    100,
        'lr_scheduler_type':               'linear',
        'bf16':                            True,
        'tf32':                            True,
        'optim':                           'adamw_torch_fused',
        'max_grad_norm':                   0.3,
        'dataloader_num_workers':          4,
        'dataloader_pin_memory':           True,
        'logging_steps':                   10,
        'eval_strategy':                   'steps',
        'eval_steps':                      500,
        'save_strategy':                   'steps',
        'save_steps':                      500,
        'save_total_limit':                1,
        'load_best_model_at_end':          True,
        'gradient_checkpointing':          True,
        'gradient_checkpointing_kwargs':   {'use_reentrant': False},
        'remove_unused_columns':           False,
        'report_to':                       'wandb',
        'dataset_kwargs':                  {'skip_prepare_dataset': True},
        'max_seq_length':                  1024,
    })

    train_n = len(train_dataset) if hasattr(train_dataset, '__len__') else None
    eval_n  = len(test_dataset)  if hasattr(test_dataset,  '__len__') else None

    init_wandb(
        run_id,
        sft_config,
        stage_name='stage1',
        exp_name='qwen2-vl-audio',
        model_id='MayaKD/qwen2-vl-audio',
        train_samples=train_n,
        eval_samples=eval_n,
        resume_run_id=load_wandb_run_id(output_dir),
    )
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_asr(model, processor, test_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            ASREvalCallback(model, processor, test_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_asr(
        trainer.model, processor, test_dataset,
        n=100, tag='final', step=trainer.state.global_step,
    )

    if wandb.run is not None:
        wandb.finish()

    push_stage1(trainer)

    return trainer


def push_stage1(trainer, output_dir=None, repo_id=HF_REPO):
    """Save model locally and push to HF. Only 1/4 shards will change
    (audio_projector ~4.6M params lands in a single tensor shard).

    output_dir defaults to trainer.args.output_dir (set by run_stage1).
    Processor is already saved to output_dir by run_stage1().
    """
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(STAGE1_BASE_DIR + '_', '') if STAGE1_BASE_DIR in output_dir else ''
    trainer.save_model(output_dir)
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message=f"Stage 1: audio_projector trained ({run_tag})",
    )
    print(f"Stage 1 checkpoint pushed to {repo_id}  [{output_dir}]")


# ── Stage 2 (LoRA fine-tuning of LLM backbone) ───────────────────────────────

STAGE2_BASE_DIR = '/workspace/projects/speech/stage2_checkpoint'


def setup_stage2(checkpoint_path):
    """Load Stage 1 model and build LoRA config.

    checkpoint_path: path to the Stage 1 output dir (after push_stage1 saved
                     weights there), e.g. stage1_checkpoint_20260304_190000/
                     or 'MayaKD/qwen2-vl-audio' to load from HF.

    Returns (model, processor, lora_config).
    The model is NOT yet wrapped with LoRA — call run_stage2() to do that.
    """
    from peft import LoraConfig
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained('/workspace/projects/speech/processor')
    processor.audio_processor = Qwen2VLAudioProcessor()

    model = Qwen2VLAudioForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    # Ensure PEFT README uses the HF repo ID, not a local path, as base_model.
    if not checkpoint_path.startswith(('MayaKD/', 'hf://')):
        model.config._name_or_path = HF_REPO

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        # Full attention + audio_projector kept trainable via modules_to_save
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        modules_to_save=['audio_projector'],
    )
    return model, processor, lora_config


def run_stage2(model, processor, lora_config, train_dataset=None, test_dataset=None,
               output_dir=None, resume_from_checkpoint=None):
    from peft import get_peft_model
    from data_utils import load_datasets

    if train_dataset is None:
        train_dataset, test_dataset = load_datasets()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = f"{STAGE2_BASE_DIR}_{run_id}"

    # Wrap with LoRA — freezes base weights, adds trainable adapters + modules_to_save
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 20K / 4 / 4 = 1250 steps/epoch → eval_steps=200 fires ~6x/epoch
    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                3,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   2e-5,
        'warmup_steps':                    50,
        'lr_scheduler_type':               'linear',
        'bf16':                            True,
        'tf32':                            True,
        'optim':                           'adamw_torch_fused',
        'max_grad_norm':                   0.3,
        'max_seq_length':                  1024,
        'dataloader_num_workers':          4,
        'dataloader_pin_memory':           True,
        'logging_steps':                   10,
        'eval_strategy':                   'steps',
        'eval_steps':                      200,
        'save_strategy':                   'steps',
        'save_steps':                      200,
        'save_total_limit':                1,
        'load_best_model_at_end':          True,
        'gradient_checkpointing':          True,
        'gradient_checkpointing_kwargs':   {'use_reentrant': False},
        'remove_unused_columns':           False,
        'report_to':                       'wandb',
        'dataset_kwargs':                  {'skip_prepare_dataset': True},
    })

    train_n = len(train_dataset) if hasattr(train_dataset, '__len__') else None
    eval_n  = len(test_dataset)  if hasattr(test_dataset,  '__len__') else None

    init_wandb(run_id, sft_config, lora_config=lora_config, stage_name='stage2',
               exp_name='qwen2-vl-audio', model_id=output_dir,
               train_samples=train_n, eval_samples=eval_n,
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_asr(model, processor, test_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            ASREvalCallback(model, processor, test_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_asr(trainer.model, processor, test_dataset,
                 n=100, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_stage2(trainer)

    return trainer


def push_stage2(trainer, output_dir=None, repo_id=HF_REPO):
    """Save LoRA adapter weights locally and push to HF under asr/lora_stage2/."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(STAGE2_BASE_DIR + '_', '') if STAGE2_BASE_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'asr/lora_stage2'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"Stage 2: LoRA adapter ({run_tag})",
    )
    print(f"Stage 2 LoRA adapter pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


# ── Merge Stage 2 LoRA + Stage 3 (LoRA from merged weights) ──────────────────

STAGE3_BASE_DIR = '/workspace/projects/speech/stage3_checkpoint'
MERGED_BASE_DIR = '/workspace/projects/speech/merged_stage2'


def merge_stage2(local_adapter_dir=None, merged_output_dir=None, repo_id=HF_REPO):
    """Load Stage 2 LoRA adapter, merge into base weights, save locally, push to HF.

    local_adapter_dir: local path to Stage 2 checkpoint (adapter_model.safetensors etc.)
                       or None to load from HF repo asr/lora_stage2/ subfolder (default).
    merged_output_dir: where to save merged model locally.
                       defaults to /workspace/projects/speech/merged_stage2_TIMESTAMP.

    Returns merged_output_dir (str) — pass to setup_stage2() as checkpoint_path.
    """
    from peft import PeftModel
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAudioForConditionalGeneration

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if merged_output_dir is None:
        merged_output_dir = f"{MERGED_BASE_DIR}_{run_id}"

    print("Loading base model ...")
    model = Qwen2VLAudioForConditionalGeneration.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )

    print("Loading LoRA adapter ...")
    if local_adapter_dir:
        model = PeftModel.from_pretrained(model, local_adapter_dir)
    else:
        model = PeftModel.from_pretrained(model, repo_id, subfolder='asr/lora_stage2')

    print("Merging LoRA into base weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_output_dir} ...")
    model.save_pretrained(merged_output_dir)

    print(f"Pushing merged model to {repo_id} ...")
    HfApi().upload_folder(
        folder_path=merged_output_dir,
        repo_id=repo_id,
        commit_message=f"Merge Stage 2 LoRA into base weights ({run_id})",
    )
    print(f"Merged model pushed to {repo_id}  [{merged_output_dir}]")
    return merged_output_dir


def run_stage3(model, processor, lora_config, train_dataset, test_dataset,
               output_dir=None, resume_from_checkpoint=None, num_train_epochs=3):
    """Stage 3: LoRA fine-tuning from merged Stage 2 weights on any dataset.

    train_dataset and test_dataset must be provided by the caller — load and
    cast them in the notebook before calling this function.

    Call sequence:
        merged_dir = merge_stage2()
        model, processor, lora_config = setup_stage2(merged_dir)
        trainer = run_stage3(model, processor, lora_config, train_ds, test_ds)
    """
    from peft import get_peft_model
    from datasets import Audio as HFAudio

    # Avoid torchcodec / libavutil.so.56 in DataLoader workers — return raw bytes;
    # format_data() / fetch_audio() handle the undecoded {"bytes":..., "path":...} dict.
    if "audio" in train_dataset.column_names:
        train_dataset = train_dataset.cast_column("audio", HFAudio(decode=False))
    if "audio" in test_dataset.column_names:
        test_dataset  = test_dataset.cast_column("audio",  HFAudio(decode=False))

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = f"{STAGE3_BASE_DIR}_{run_id}"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_n = len(train_dataset) if hasattr(train_dataset, '__len__') else None
    eval_n  = len(test_dataset)  if hasattr(test_dataset,  '__len__') else None

    # eval_steps tuned for LibriSpeech train-clean-100 (~28K samples):
    # 28K / batch 4 / grad_accum 4 = ~1750 steps/epoch → eval fires ~7×/epoch
    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                num_train_epochs,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   2e-5,
        'warmup_steps':                    50,
        'lr_scheduler_type':               'linear',
        'bf16':                            True,
        'tf32':                            True,
        'optim':                           'adamw_torch_fused',
        'max_grad_norm':                   0.3,
        'max_seq_length':                  1024,
        'dataloader_num_workers':          4,
        'dataloader_pin_memory':           True,
        'logging_steps':                   10,
        'eval_strategy':                   'steps',
        'eval_steps':                      250,
        'save_strategy':                   'steps',
        'save_steps':                      250,
        'save_total_limit':                1,
        'load_best_model_at_end':          True,
        'gradient_checkpointing':          True,
        'gradient_checkpointing_kwargs':   {'use_reentrant': False},
        'remove_unused_columns':           False,
        'report_to':                       'wandb',
        'dataset_kwargs':                  {'skip_prepare_dataset': True},
    })

    init_wandb(run_id, sft_config, lora_config=lora_config, stage_name='stage3',
               exp_name='qwen2-vl-audio', model_id=output_dir,
               train_samples=train_n, eval_samples=eval_n,
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_asr(model, processor, test_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            ASREvalCallback(model, processor, test_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_asr(trainer.model, processor, test_dataset,
                 n=100, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_stage3(trainer)

    return trainer


def push_stage3(trainer, output_dir=None, repo_id=HF_REPO):
    """Save Stage 3 LoRA adapter and push to HF under asr/lora_stage3/."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(STAGE3_BASE_DIR + '_', '') if STAGE3_BASE_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'asr/lora_stage3'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"Stage 3: LoRA adapter ({run_tag})",
    )
    print(f"Stage 3 LoRA adapter pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


# ── AVQA Stage 1: train music_projector only ─────────────────────────────────

AVQA_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_checkpoint'
AVQA_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_checkpoint'


def init_avqa_model(base_repo=HF_REPO, processor_path='/workspace/projects/speech/processor'):
    """Load Qwen2VLDualAudioForConditionalGeneration from the ASR checkpoint.

    Inherits trained Whisper encoder + audio_projector from the ASR model.
    music_projector is randomly initialised — trained in AVQA Stage 1.

    Returns (model, processor).
    """
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()

    print(f"Loading base model from {base_repo}/{ASR_MERGE_SUBFOLDER} ...")
    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        base_repo, subfolder=ASR_MERGE_SUBFOLDER,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
        ignore_mismatched_sizes=True,   # music_projector is new — not in checkpoint
    )
    print("music_projector randomly initialised ✓")
    return model, processor


def freeze_for_avqa_stage1(model):
    """Freeze all parameters except music_projector."""
    for name, param in model.named_parameters():
        param.requires_grad = 'music_projector' in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.4f}%)")


def run_avqa_stage1(model=None, processor=None,
                    train_dataset=None, val_dataset=None,
                    output_dir=None, resume_from_checkpoint=None):
    """Train AVQA Stage 1: music_projector only, all other weights frozen.

    Can be called with no arguments:
        trainer = run_avqa_stage1()
    """
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if model is None or processor is None:
        model, processor = init_avqa_model()
        freeze_for_avqa_stage1(model)

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000)
        val_dataset   = AVQADataset(split='val')

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_STAGE1_DIR}_{run_id}"

    # 8K / 4 / 4 = 500 steps/epoch → eval_steps=100 fires 5×/epoch
    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   1e-4,
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
        'eval_strategy':                   'steps',
        'eval_steps':                      100,
        'save_strategy':                   'steps',
        'save_steps':                      100,
        'save_total_limit':                1,
        'load_best_model_at_end':          False,  # key structure mismatch prevents correct reload
        'gradient_checkpointing':          True,
        'gradient_checkpointing_kwargs':   {'use_reentrant': False},
        'remove_unused_columns':           False,
        'report_to':                       'wandb',
        'dataset_kwargs':                  {'skip_prepare_dataset': True},
    })

    init_wandb(run_id, sft_config, stage_name='avqa_stage1_panns',
               exp_name='qwen2-vl-avqa-mert', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_stage1(trainer)
    return trainer


def push_avqa_stage1(trainer, output_dir=None, repo_id=HF_REPO):
    """Save model and push to HF under avqa/ablations/panns8/stage1/."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_STAGE1_DIR + '_', '') if AVQA_STAGE1_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/panns8/stage1'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 1 (MERT): music_projector trained ({run_tag})",
    )
    print(f"AVQA Stage 1 pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


# ── AVQA Stage 2: LoRA + both projectors ─────────────────────────────────────

def setup_avqa_stage2(checkpoint_path=HF_REPO, subfolder='avqa/ablations/panns8/stage1',
                      processor_path='/workspace/projects/speech/processor'):
    """Load AVQA Stage 1 model and build LoRA config.

    checkpoint_path: HF repo or local dir with Stage 1 weights.
    subfolder:       subfolder within the repo (default 'avqa/ablations/panns8/stage1').

    Returns (model, processor, lora_config).
    """
    from peft import LoraConfig
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    if subfolder:
        load_kwargs['subfolder'] = subfolder

    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        checkpoint_path, **load_kwargs
    )
    if not checkpoint_path.startswith(('MayaKD/', 'hf://')):
        model.config._name_or_path = HF_REPO

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        modules_to_save=['audio_projector', 'music_projector'],
    )
    return model, processor, lora_config


def run_avqa_stage2(model, processor, lora_config,
                    train_dataset=None, val_dataset=None,
                    output_dir=None, resume_from_checkpoint=None):
    from peft import get_peft_model
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000)
        val_dataset   = AVQADataset(split='val')

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_STAGE2_DIR}_{run_id}"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
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
        'eval_strategy':                   'steps',
        'eval_steps':                      100,
        'save_strategy':                   'steps',
        'save_steps':                      100,
        'save_total_limit':                1,
        'load_best_model_at_end':          False,  # key structure mismatch prevents correct reload
        'gradient_checkpointing':          True,
        'gradient_checkpointing_kwargs':   {'use_reentrant': False},
        'remove_unused_columns':           False,
        'report_to':                       'wandb',
        'dataset_kwargs':                  {'skip_prepare_dataset': True},
    })

    init_wandb(run_id, sft_config, lora_config=lora_config, stage_name='avqa_stage2_panns',
               exp_name='qwen2-vl-avqa-mert', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_stage2(trainer)
    return trainer


def push_avqa_stage2(trainer, output_dir=None, repo_id=HF_REPO):
    """Save LoRA adapter and push to HF under avqa/ablations/panns8/stage2/."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_STAGE2_DIR + '_', '') if AVQA_STAGE2_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/panns8/stage2'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 2 (MERT): LoRA + projectors ({run_tag})",
        ignore_patterns=["README.md"],
    )
    print(f"AVQA Stage 2 pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")

# ── AVQA Whisper ablation: Whisper-32 as music encoder ───────────────────────
# Architecture: video audio → Whisper encoder (offline precompute) → stride-pool
# to 32 frames → [32, 1280] .npy → music_projector (per-frame Linear(1280, H))
# → 32 × <|music_pad|> tokens.  TTS question path unchanged.
# Config: music_seq_input=True, music_embed_dim=1280, n_music_tokens=32.

AVQA_WHISPER_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_whisper32_checkpoint'
AVQA_WHISPER_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper32_checkpoint'
WHISPER_MUSIC_DIR = '/workspace/projects/speech/music_avqa_dataset/data/whisper_features'


def init_avqa_whisper_model(base_repo=HF_REPO,
                            processor_path='/workspace/projects/speech/processor',
                            seed=42):
    """Load model for Whisper-as-music-encoder ablation.

    Sets music_seq_input=True, music_embed_dim=1280, n_music_tokens=32 in config
    so music_projector = Linear(1280, H) (per-frame) instead of Linear(D, 32*H).
    music_projector is randomly initialised — trained in AVQA Stage 1.
    Whisper encoder + audio_projector inherited from ASR checkpoint (frozen).
    """
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()
    processor.n_music_tokens = 32

    print(f"Loading base model from {base_repo}/{ASR_MERGE_SUBFOLDER} ...")
    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        base_repo, subfolder=ASR_MERGE_SUBFOLDER,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
        ignore_mismatched_sizes=True,  # music_projector shape will change
    )
    # Override music config for Whisper-32 sequence input
    model.config.music_seq_input  = True
    model.config.music_embed_dim  = 1280   # Whisper d_model
    model.config.n_music_tokens   = 32

    # Re-initialise music_projector with the correct shape: Linear(1280, H)
    # seed controls the projector init → vary it for multi-seed error-bar runs.
    import torch.nn as nn
    torch.manual_seed(seed)
    llm_hidden = model.config.text_config.hidden_size
    model.music_projector = nn.Linear(1280, llm_hidden, bias=True).to(
        device=next(model.parameters()).device,
        dtype=torch.bfloat16,
    )
    print(f"music_projector re-initialised: Linear(1280, {llm_hidden}) [per-frame, 32 tokens] ✓")
    return model, processor


def freeze_for_avqa_whisper_stage1(model):
    """Freeze all parameters except music_projector."""
    for name, param in model.named_parameters():
        param.requires_grad = 'music_projector' in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.4f}%)")


def run_avqa_whisper_stage1(model=None, processor=None,
                            train_dataset=None, val_dataset=None,
                            output_dir=None, resume_from_checkpoint=None):
    """Train AVQA Whisper Stage 1: music_projector only, all other weights frozen."""
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if model is None or processor is None:
        model, processor = init_avqa_whisper_model()
        freeze_for_avqa_whisper_stage1(model)

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=WHISPER_MUSIC_DIR, require_video=False)
        val_dataset   = AVQADataset(split='val', music_dir=WHISPER_MUSIC_DIR, require_video=False)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_WHISPER_STAGE1_DIR}_{run_id}"

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   1e-4,
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
        'eval_strategy':                   'steps',
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

    init_wandb(run_id, sft_config, stage_name='avqa_stage1_whisper32',
               exp_name='qwen2-vl-avqa-whisper32', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_whisper_stage1(trainer)
    return trainer


def push_avqa_whisper_stage1(trainer, output_dir=None, repo_id=HF_REPO):
    """Save model and push to HF under avqa/ablations/whisper32/stage1/."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_WHISPER_STAGE1_DIR + '_', '') if AVQA_WHISPER_STAGE1_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/whisper32/stage1'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 1 (Whisper-32): music_projector trained ({run_tag})",
    )
    print(f"AVQA Stage 1 (Whisper) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


def setup_avqa_whisper_stage2(checkpoint_path=HF_REPO, subfolder='avqa/ablations/whisper32/stage1',
                              processor_path='/workspace/projects/speech/processor',
                              modules_to_save=None):
    """Load Whisper Stage 1 model and build LoRA config.

    modules_to_save defaults to ['audio_projector', 'music_projector'].
    Pass modules_to_save=['music_projector'] to freeze audio_projector (preserves ASR WER).
    """
    from peft import LoraConfig
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    if subfolder:
        load_kwargs['subfolder'] = subfolder

    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        checkpoint_path, **load_kwargs
    )
    if not checkpoint_path.startswith(('MayaKD/', 'hf://')):
        model.config._name_or_path = HF_REPO

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        modules_to_save=modules_to_save or ['audio_projector', 'music_projector'],
    )
    return model, processor, lora_config


def run_avqa_whisper_stage2(model, processor, lora_config,
                            train_dataset=None, val_dataset=None,
                            output_dir=None, resume_from_checkpoint=None):
    from peft import get_peft_model
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=WHISPER_MUSIC_DIR, require_video=False)
        val_dataset   = AVQADataset(split='val', music_dir=WHISPER_MUSIC_DIR, require_video=False)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_WHISPER_STAGE2_DIR}_{run_id}"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
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
        'eval_strategy':                   'steps',
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

    init_wandb(run_id, sft_config, lora_config=lora_config, stage_name='avqa_stage2_whisper32',
               exp_name='qwen2-vl-avqa-whisper32', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_whisper_stage2(trainer)
    return trainer


def push_avqa_whisper_stage2(trainer, output_dir=None, repo_id=HF_REPO):
    """Save LoRA adapter and push to HF under avqa/ablations/whisper32/stage2/."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_WHISPER_STAGE2_DIR + '_', '') if AVQA_WHISPER_STAGE2_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/whisper32/stage2'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 2 (Whisper-32): LoRA + projectors ({run_tag})",
        ignore_patterns=["README.md"],
    )
    print(f"AVQA Stage 2 (Whisper) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


# ── AVQA Whisper-32-VarLen (no pad-to-3000 dilution in preprocessing) ────────
# Same architecture as Whisper-32 (music_seq_input=True, Linear(1280,H), 32 tokens).
# Features produced by whisper_preprocess.py WITH the pad-to-3000 removed:
#   encoder runs at actual audio length → pool from real T_enc → 32 tokens.
# For ~60s clips (capped to 30s, full 3000 frames) there is no difference.
# For clips shorter than 30s the old code diluted the 32 tokens with silence-
# encoded frames; this variant does not.
# init_avqa_whisper_model() and freeze_for_avqa_whisper_stage1() are reused.

AVQA_WHISPER_VARLEN_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_whisper_varlen_checkpoint'
AVQA_WHISPER_VARLEN_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper_varlen_checkpoint'
WHISPER_VARLEN_MUSIC_DIR       = '/workspace/projects/speech/music_avqa_dataset/data/whisper_features_varlen'


def run_avqa_whisper_varlen_stage1(model=None, processor=None,
                                   train_dataset=None, val_dataset=None,
                                   output_dir=None, resume_from_checkpoint=None):
    """Train VarLen Stage 1: music_projector only (no pad-to-3000 features)."""
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if model is None or processor is None:
        model, processor = init_avqa_whisper_model()
        freeze_for_avqa_whisper_stage1(model)

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=WHISPER_VARLEN_MUSIC_DIR, require_video=False)
        val_dataset   = AVQADataset(split='val', music_dir=WHISPER_VARLEN_MUSIC_DIR, require_video=False)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_WHISPER_VARLEN_STAGE1_DIR}_{run_id}"

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   1e-4,
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
        'eval_strategy':                   'steps',
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

    init_wandb(run_id, sft_config, stage_name='avqa_stage1_whisper_varlen',
               exp_name='qwen2-vl-avqa-whisper-varlen', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_whisper_varlen_stage1(trainer)
    return trainer


def push_avqa_whisper_varlen_stage1(trainer, output_dir=None, repo_id=HF_REPO):
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_WHISPER_VARLEN_STAGE1_DIR + '_', '') if AVQA_WHISPER_VARLEN_STAGE1_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/whisper_fullres_varlen/stage1'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 1 (Whisper-VarLen): music_projector trained ({run_tag})",
    )
    print(f"AVQA Stage 1 (Whisper-VarLen) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


def setup_avqa_whisper_varlen_stage2(checkpoint_path=HF_REPO, subfolder='avqa/ablations/whisper_fullres_varlen/stage1',
                                     processor_path='/workspace/projects/speech/processor'):
    from peft import LoraConfig
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    if subfolder:
        load_kwargs['subfolder'] = subfolder

    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        checkpoint_path, **load_kwargs
    )
    if not checkpoint_path.startswith(('MayaKD/', 'hf://')):
        model.config._name_or_path = HF_REPO

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        modules_to_save=['audio_projector', 'music_projector'],
    )
    return model, processor, lora_config


def run_avqa_whisper_varlen_stage2(model, processor, lora_config,
                                   train_dataset=None, val_dataset=None,
                                   output_dir=None, resume_from_checkpoint=None):
    from peft import get_peft_model
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=WHISPER_VARLEN_MUSIC_DIR, require_video=False)
        val_dataset   = AVQADataset(split='val', music_dir=WHISPER_VARLEN_MUSIC_DIR, require_video=False)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_WHISPER_VARLEN_STAGE2_DIR}_{run_id}"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
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
        'eval_strategy':                   'steps',
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

    init_wandb(run_id, sft_config, lora_config=lora_config, stage_name='avqa_stage2_whisper_varlen',
               exp_name='qwen2-vl-avqa-whisper-varlen', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_whisper_varlen_stage2(trainer)
    return trainer


def push_avqa_whisper_varlen_stage2(trainer, output_dir=None, repo_id=HF_REPO):
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_WHISPER_VARLEN_STAGE2_DIR + '_', '') if AVQA_WHISPER_VARLEN_STAGE2_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/whisper_fullres_varlen/stage2'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 2 (Whisper-VarLen): LoRA + projectors ({run_tag})",
        ignore_patterns=["README.md"],
    )
    print(f"AVQA Stage 2 (Whisper-VarLen) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


# ── AVQA Whisper-32-Full (full-duration audio, chunked preprocessing) ─────────
# Same architecture as Whisper-32; only the feature dir and run names differ.
# Uses whisper_features_full/ produced by whisper_preprocess_full.py.
# init_avqa_whisper_model() and freeze_for_avqa_whisper_stage1() are reused as-is.

AVQA_WHISPER_FULL_STAGE1_DIR    = '/workspace/projects/speech/avqa_stage1_whisper32full_checkpoint'
AVQA_WHISPER_FULL_STAGE2_DIR    = '/workspace/projects/speech/avqa_stage2_whisper32full_checkpoint'
WHISPER_FULL_MUSIC_DIR          = '/workspace/projects/speech/music_avqa_dataset/data/whisper_features_full'

# whisper_fullres — per-chunk pooling, variable n_music_tokens (64 for 60s, 32 for 30s)
AVQA_WHISPER_FULLRES_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_whisper_fullres_checkpoint'
AVQA_WHISPER_FULLRES_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper_fullres_checkpoint'
WHISPER_FULLRES_MUSIC_DIR       = '/workspace/projects/speech/music_avqa_dataset/data/whisper_features_fullres'


def run_avqa_whisper_full_stage1(model=None, processor=None,
                                  train_dataset=None, val_dataset=None,
                                  output_dir=None, resume_from_checkpoint=None,
                                  music_dir=None, experiment_tag='whisper32_full',
                                  seed=42, push=True):
    """Train Whisper-full Stage 1: music_projector only.

    Pass music_dir=WHISPER_FULLRES_MUSIC_DIR and experiment_tag='whisper_fullres'
    to run the per-chunk-pooling variant without duplicating this function.

    seed: controls projector init + data-shuffle order (multi-seed error-bar runs).
    push: False → save Stage 1 locally only, skip HF upload (so seed runs don't
          clobber the canonical avqa_hf_path(tag, 1) subfolder on HF).
    """
    from transformers import set_seed
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    set_seed(seed)
    if music_dir is None:
        music_dir = WHISPER_FULL_MUSIC_DIR

    if model is None or processor is None:
        model, processor = init_avqa_whisper_model(seed=seed)
        freeze_for_avqa_whisper_stage1(model)

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=music_dir, require_video=False)
        val_dataset   = AVQADataset(split='val', music_dir=music_dir, require_video=False)

    # output_dir default depends on experiment_tag so resumed runs land in the right folder
    stage1_base = (AVQA_WHISPER_FULLRES_STAGE1_DIR
                   if experiment_tag == 'whisper_fullres'
                   else AVQA_WHISPER_FULL_STAGE1_DIR)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{stage1_base}_{run_id}"

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'seed':                            seed,
        'data_seed':                       seed,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   1e-4,
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
        'eval_strategy':                   'steps',
        'eval_steps':                      250,
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

    init_wandb(run_id, sft_config, stage_name=f'avqa_stage1_{experiment_tag}',
               exp_name=f'qwen2-vl-avqa-{experiment_tag}', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    if push:
        push_avqa_whisper_full_stage1(trainer, experiment_tag=experiment_tag)
    else:
        trainer.save_model(trainer.args.output_dir)
        print(f"[seed={seed}] Stage 1 saved locally (no HF push): {trainer.args.output_dir}")
    return trainer


def cleanup_after_stage1(*refs):
    """Free GPU memory between Stage 1 and Stage 2.

    Pass any objects to delete (trainer, model, processor, datasets, etc.).
    A kernel restart is the most reliable alternative, but this helps when
    staying in the same kernel.

    Usage (in notebook):
        cleanup_after_stage1(trainer, model, processor, train_dataset, val_dataset)
    """
    import gc, torch
    for obj in refs:
        try:
            name = type(obj).__name__
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    print("Cleanup done. Restart kernel if GPU memory above is insufficient for Stage 2.")


def push_avqa_whisper_full_stage1(trainer, output_dir=None, repo_id=HF_REPO,
                                   experiment_tag='whisper32_full'):
    """Save model and push to HF under avqa_hf_path(experiment_tag, 1)."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    hf_subfolder = avqa_hf_path(experiment_tag, 1)
    trainer.save_model(output_dir)
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 1 ({experiment_tag}): music_projector trained",
    )
    print(f"AVQA Stage 1 ({experiment_tag}) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


def setup_avqa_whisper_full_stage2(checkpoint_path=HF_REPO, subfolder='avqa/ablations/whisper32_full/stage1',
                                    processor_path='/workspace/projects/speech/processor'):
    """Load Whisper-32-Full Stage 1 model and build LoRA config."""
    return setup_avqa_whisper_stage2(
        checkpoint_path=checkpoint_path,
        subfolder=subfolder,
        processor_path=processor_path,
    )


def run_avqa_whisper_full_stage2(model, processor, lora_config,
                                  train_dataset=None, val_dataset=None,
                                  output_dir=None, resume_from_checkpoint=None,
                                  music_dir=None, experiment_tag='whisper32_full',
                                  num_train_epochs=1, seed=42, push=True):
    """Train Whisper-full Stage 2: LoRA fine-tune.

    Pass music_dir=WHISPER_FULLRES_MUSIC_DIR and experiment_tag='whisper_fullres'
    to run the per-chunk-pooling variant without duplicating this function.

    seed: set_seed before get_peft_model (LoRA init) + data_seed for shuffle order.
    push: False → save adapter locally only, skip HF upload (seed runs must not
          clobber the canonical avqa_hf_path(tag, 2) subfolder on HF).
    """
    from transformers import set_seed
    from peft import get_peft_model
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    set_seed(seed)
    if music_dir is None:
        music_dir = WHISPER_FULL_MUSIC_DIR

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=music_dir, require_video=False)
        val_dataset   = AVQADataset(split='val', music_dir=music_dir, require_video=False)

    stage2_base = (AVQA_WHISPER_FULLRES_STAGE2_DIR
                   if experiment_tag == 'whisper_fullres'
                   else AVQA_WHISPER_FULL_STAGE2_DIR)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{stage2_base}_{run_id}"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Default (1-epoch) runs keep the original step-based saving for crash recovery.
    # Multi-epoch runs save once per epoch and keep them all, so each epoch's checkpoint
    # can be full-tested afterwards to pick the best (guards against late-epoch overfit).
    if num_train_epochs > 1:
        save_kwargs = {'save_strategy': 'epoch', 'save_total_limit': num_train_epochs}
    else:
        save_kwargs = {'save_strategy': 'steps', 'save_steps': 100, 'save_total_limit': 1}

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'seed':                            seed,
        'data_seed':                       seed,
        'num_train_epochs':                num_train_epochs,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
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
        'eval_strategy':                   'steps',
        'eval_steps':                      250,
        **save_kwargs,
        'load_best_model_at_end':          False,
        'gradient_checkpointing':          True,
        'gradient_checkpointing_kwargs':   {'use_reentrant': False},
        'remove_unused_columns':           False,
        'report_to':                       'wandb',
        'dataset_kwargs':                  {'skip_prepare_dataset': True},
    })

    init_wandb(run_id, sft_config, lora_config=lora_config,
               stage_name=f'avqa_stage2_{experiment_tag}',
               exp_name=f'qwen2-vl-avqa-{experiment_tag}', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    if push:
        push_avqa_whisper_full_stage2(trainer, experiment_tag=experiment_tag)
    else:
        trainer.save_model(trainer.args.output_dir)
        print(f"[seed={seed}] Stage 2 adapter saved locally (no HF push): {trainer.args.output_dir}")
    return trainer


def push_avqa_whisper_full_stage2(trainer, output_dir=None, repo_id=HF_REPO,
                                   experiment_tag='whisper32_full'):
    """Save LoRA adapter and push to HF under avqa_hf_path(experiment_tag, 2)."""
    if output_dir is None:
        output_dir = trainer.args.output_dir
    hf_subfolder = avqa_hf_path(experiment_tag, 2)
    trainer.save_model(output_dir)
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 2 ({experiment_tag}): LoRA + projectors",
        ignore_patterns=["README.md"],
    )
    print(f"AVQA Stage 2 ({experiment_tag}) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


# ── AVQA PANNs-32: single CNN14 vector (2048-dim) → 32 tokens via Linear ────
# music_seq_input=False: PANNs produces one embedding per clip, not a sequence.
# music_projector = Linear(2048, 32*H) expands it to 32 LLM-space tokens.

AVQA_PANNS32_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_panns32_checkpoint'
AVQA_PANNS32_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_panns32_checkpoint'
PANNS32_MUSIC_DIR       = '/workspace/projects/speech/music_avqa_dataset/data/panns_features'


def init_avqa_panns32_model(base_repo=HF_REPO,
                            processor_path='/workspace/projects/speech/processor'):
    """Load model for PANNs-32 run.

    music_seq_input=False, music_embed_dim=2048, n_music_tokens=32.
    music_projector = Linear(2048, 32*H) — single-vector → 32 tokens.
    Whisper encoder + audio_projector inherited from ASR checkpoint (frozen).
    """
    import torch.nn as nn
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()

    print(f"Loading base model from {base_repo}/{ASR_MERGE_SUBFOLDER} ...")
    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        base_repo, subfolder=ASR_MERGE_SUBFOLDER,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
        ignore_mismatched_sizes=True,
    )
    model.config.music_seq_input = False
    model.config.music_embed_dim = 2048
    model.config.n_music_tokens  = 32

    llm_hidden = model.config.text_config.hidden_size
    model.music_projector = nn.Linear(2048, 32 * llm_hidden, bias=True).to(
        device=next(model.parameters()).device,
        dtype=torch.bfloat16,
    )
    print(f"music_projector re-initialised: Linear(2048, 32×{llm_hidden}) ✓")
    return model, processor


def freeze_for_avqa_panns32_stage1(model):
    """Freeze all parameters except music_projector."""
    for name, param in model.named_parameters():
        param.requires_grad = 'music_projector' in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.4f}%)")


def run_avqa_panns32_stage1(model=None, processor=None,
                            train_dataset=None, val_dataset=None,
                            output_dir=None, resume_from_checkpoint=None):
    """Train PANNs-32 Stage 1: music_projector only, all other weights frozen."""
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if model is None or processor is None:
        model, processor = init_avqa_panns32_model()
        freeze_for_avqa_panns32_stage1(model)

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=PANNS32_MUSIC_DIR, require_video=False)
        val_dataset   = AVQADataset(split='val',
                                    music_dir=PANNS32_MUSIC_DIR, require_video=False)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_PANNS32_STAGE1_DIR}_{run_id}"

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
        'learning_rate':                   1e-4,
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
        'eval_strategy':                   'steps',
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

    init_wandb(run_id, sft_config, stage_name='avqa_stage1_panns32',
               exp_name='qwen2-vl-avqa-panns32', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_panns32_stage1(trainer)
    return trainer


def push_avqa_panns32_stage1(trainer, output_dir=None, repo_id=HF_REPO):
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_PANNS32_STAGE1_DIR + '_', '') if AVQA_PANNS32_STAGE1_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/panns32/stage1'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 1 (PANNs-32): music_projector trained ({run_tag})",
    )
    print(f"AVQA Stage 1 (PANNs-32) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


def setup_avqa_panns32_stage2(checkpoint_path=HF_REPO, subfolder='avqa/ablations/panns32/stage1',
                              processor_path='/workspace/projects/speech/processor'):
    """Load PANNs-32 Stage 1 checkpoint and build LoRA config."""
    from peft import LoraConfig
    from transformers import Qwen2VLProcessor
    from transformers.models.qwen2_vl.audio_processing_qwen2_vl import Qwen2VLAudioProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDualAudioForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(processor_path)
    processor.audio_processor = Qwen2VLAudioProcessor()

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    if subfolder:
        load_kwargs['subfolder'] = subfolder

    model = Qwen2VLDualAudioForConditionalGeneration.from_pretrained(
        checkpoint_path, **load_kwargs
    )
    if not checkpoint_path.startswith(('MayaKD/', 'hf://')):
        model.config._name_or_path = HF_REPO

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        modules_to_save=['audio_projector', 'music_projector'],
    )
    return model, processor, lora_config


def run_avqa_panns32_stage2(model, processor, lora_config,
                            train_dataset=None, val_dataset=None,
                            output_dir=None, resume_from_checkpoint=None):
    from peft import get_peft_model
    from avqa.dataset import AVQADataset
    from avqa.eval_utils import evaluate_avqa, AVQAEvalCallback

    if train_dataset is None:
        train_dataset = AVQADataset(split='train', max_samples=8000,
                                    music_dir=PANNS32_MUSIC_DIR, require_video=False)
        val_dataset   = AVQADataset(split='val',
                                    music_dir=PANNS32_MUSIC_DIR, require_video=False)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        if resume_from_checkpoint is not None:
            output_dir = str(Path(resume_from_checkpoint).parent)
        else:
            output_dir = f"{AVQA_PANNS32_STAGE2_DIR}_{run_id}"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(**{
        'output_dir':                      output_dir,
        'num_train_epochs':                1,
        'per_device_train_batch_size':     4,
        'per_device_eval_batch_size':      4,
        'gradient_accumulation_steps':     4,
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
        'eval_strategy':                   'steps',
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

    init_wandb(run_id, sft_config, lora_config=lora_config, stage_name='avqa_stage2_panns32',
               exp_name='qwen2-vl-avqa-panns32', model_id=HF_REPO,
               train_samples=len(train_dataset), eval_samples=len(val_dataset),
               resume_run_id=load_wandb_run_id(output_dir))
    save_wandb_run_id(output_dir)

    data_collator = DataCollator(processor=processor)

    evaluate_avqa(model, processor, val_dataset, n=100, tag='baseline', step=0)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            WandBLossCallback(),
            AVQAEvalCallback(model, processor, val_dataset, n=50, tag='epoch_eval'),
        ],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_avqa(trainer.model, processor, val_dataset,
                  n=200, tag='final', step=trainer.state.global_step)

    if wandb.run is not None:
        wandb.finish()

    push_avqa_panns32_stage2(trainer)
    return trainer


def push_avqa_panns32_stage2(trainer, output_dir=None, repo_id=HF_REPO):
    if output_dir is None:
        output_dir = trainer.args.output_dir
    run_tag = output_dir.replace(AVQA_PANNS32_STAGE2_DIR + '_', '') if AVQA_PANNS32_STAGE2_DIR in output_dir else ''
    trainer.save_model(output_dir)
    hf_subfolder = 'avqa/ablations/panns32/stage2'
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        path_in_repo=hf_subfolder,
        commit_message=f"AVQA Stage 2 (PANNs-32): LoRA + projectors ({run_tag})",
        ignore_patterns=["README.md"],
    )
    print(f"AVQA Stage 2 (PANNs-32) pushed to {repo_id}/{hf_subfolder}  [{output_dir}]")


# ── AVQA Whisper-fullres-VarLen (no pad-to-3000 on partial last chunk) ────────
# Same architecture as whisper_fullres; only the feature dir and run names differ.
# Features in whisper_features_fullres_varlen/ were produced with padding=False —
# partial last chunks run at natural encoder length instead of being padded to 30s.

AVQA_WHISPER_FULLRES_VARLEN_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_whisper_fullres_varlen_checkpoint'
AVQA_WHISPER_FULLRES_VARLEN_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper_fullres_varlen_checkpoint'
WHISPER_FULLRES_VARLEN_MUSIC_DIR       = '/workspace/projects/speech/music_avqa_dataset/data/whisper_features_fullres_varlen'


def run_avqa_whisper_fullres_varlen_stage1(model=None, processor=None,
                                           train_dataset=None, val_dataset=None,
                                           output_dir=None, resume_from_checkpoint=None):
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    return run_avqa_whisper_full_stage1(
        model=model, processor=processor,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or f"{AVQA_WHISPER_FULLRES_VARLEN_STAGE1_DIR}_{run_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_VARLEN_MUSIC_DIR,
        experiment_tag='whisper_fullres_varlen',
    )


def setup_avqa_whisper_fullres_varlen_stage2(checkpoint_path=HF_REPO,
                                             subfolder='avqa/ablations/whisper_fullres_varlen/stage1',
                                             processor_path='/workspace/projects/speech/processor'):
    return setup_avqa_whisper_stage2(
        checkpoint_path=checkpoint_path,
        subfolder=subfolder,
        processor_path=processor_path,
    )


def run_avqa_whisper_fullres_varlen_stage2(model, processor, lora_config,
                                           train_dataset=None, val_dataset=None,
                                           output_dir=None, resume_from_checkpoint=None):
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    return run_avqa_whisper_full_stage2(
        model=model, processor=processor, lora_config=lora_config,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or f"{AVQA_WHISPER_FULLRES_VARLEN_STAGE2_DIR}_{run_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_VARLEN_MUSIC_DIR,
        experiment_tag='whisper_fullres_varlen',
    )


# whisper_fullres_ts — same features as whisper_fullres, adds per-video timing header to prompt
AVQA_WHISPER_FULLRES_TS_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_whisper_fullres_ts_checkpoint'
AVQA_WHISPER_FULLRES_TS_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper_fullres_ts_checkpoint'


def run_avqa_whisper_ts_stage1(model=None, processor=None,
                                train_dataset=None, val_dataset=None,
                                output_dir=None, resume_from_checkpoint=None):
    """Temporal-context variant of whisper_fullres Stage 1.

    Identical to whisper_fullres except AVQADataset is constructed with
    add_frame_timestamps=True, which prepends a per-video header:
        'This is a video (60s, 64 audio tokens, ~0.94s/token).
         Frames at [0.0s, 8.6s, ...60.0s]:'
    before the video block. Everything else (features, model, config) unchanged.
    """
    from avqa.dataset import AVQADataset
    if train_dataset is None:
        train_dataset = AVQADataset(
            split='train', max_samples=8000,
            music_dir=WHISPER_FULLRES_MUSIC_DIR,
            require_video=False,
            add_frame_timestamps=True,
        )
        val_dataset = AVQADataset(
            split='val',
            music_dir=WHISPER_FULLRES_MUSIC_DIR,
            require_video=False,
            add_frame_timestamps=True,
        )
    return run_avqa_whisper_full_stage1(
        model=model, processor=processor,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or AVQA_WHISPER_FULLRES_TS_STAGE1_DIR,
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_MUSIC_DIR,
        experiment_tag='whisper_fullres_ts',
    )


def run_avqa_whisper_ts_stage2(model, processor, lora_config,
                                train_dataset=None, val_dataset=None,
                                output_dir=None, resume_from_checkpoint=None):
    """Temporal-context variant of whisper_fullres Stage 2."""
    from avqa.dataset import AVQADataset
    if train_dataset is None:
        train_dataset = AVQADataset(
            split='train', max_samples=8000,
            music_dir=WHISPER_FULLRES_MUSIC_DIR,
            require_video=False,
            add_frame_timestamps=True,
        )
        val_dataset = AVQADataset(
            split='val',
            music_dir=WHISPER_FULLRES_MUSIC_DIR,
            require_video=False,
            add_frame_timestamps=True,
        )
    return run_avqa_whisper_full_stage2(
        model=model, processor=processor, lora_config=lora_config,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or AVQA_WHISPER_FULLRES_TS_STAGE2_DIR,
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_MUSIC_DIR,
        experiment_tag='whisper_fullres_ts',
    )


# ── AVQA whisper_fullres_v2: fullres features + n_music_tokens bug fixed + seeded init ──
# Fixes vs whisper_fullres (94.31%):
#   1. collator/eval_utils always pass music_lengths=Ts → processor uses actual T (64),
#      not the n_music_tokens=8 fallback that silently dropped 75% of audio features.
#   2. processor.n_music_tokens=32 set at init (belt-and-suspenders).
#   3. torch.manual_seed(42) before music_projector Linear → reproducible init.
# Same features as whisper_fullres (whisper_features_fullres/).

AVQA_WHISPER_FULLRES_V2_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_whisper_fullres_v2_checkpoint'
AVQA_WHISPER_FULLRES_V2_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper_fullres_v2_checkpoint'


def run_avqa_whisper_fullres_v2_stage1(model=None, processor=None,
                                       train_dataset=None, val_dataset=None,
                                       output_dir=None, resume_from_checkpoint=None,
                                       seed=42, push=True):
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    return run_avqa_whisper_full_stage1(
        model=model, processor=processor,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or f"{AVQA_WHISPER_FULLRES_V2_STAGE1_DIR}_{run_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_MUSIC_DIR,
        experiment_tag='whisper_fullres_v2',
        seed=seed, push=push,
    )


def setup_avqa_whisper_fullres_v2_stage2(checkpoint_path=HF_REPO,
                                         subfolder='avqa/headline/stage1',
                                         processor_path='/workspace/projects/speech/processor'):
    return setup_avqa_whisper_stage2(
        checkpoint_path=checkpoint_path,
        subfolder=subfolder,
        processor_path=processor_path,
    )


def run_avqa_whisper_fullres_v2_stage2(model, processor, lora_config,
                                       train_dataset=None, val_dataset=None,
                                       output_dir=None, resume_from_checkpoint=None):
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    return run_avqa_whisper_full_stage2(
        model=model, processor=processor, lora_config=lora_config,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or f"{AVQA_WHISPER_FULLRES_V2_STAGE2_DIR}_{run_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_MUSIC_DIR,
        experiment_tag='whisper_fullres_v2',
    )


# ── AVQA whisper_fullres_v3: same as v2 but audio_projector frozen in Stage 2 ──
# Change vs v2: modules_to_save=['music_projector'] only (audio_projector excluded).
# Motivation: preserves ASR WER (~4.71%) that degrades to ~18.68% when audio_projector
# is fine-tuned on the AVQA TTS-question objective.
# Stage 1 is identical to v2 — reuse the v2 Stage 1 checkpoint on HF.

AVQA_WHISPER_FULLRES_V3_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper_fullres_v3_checkpoint'


def setup_avqa_whisper_fullres_v3_stage2(checkpoint_path=HF_REPO,
                                         subfolder='avqa/headline/stage1',
                                         processor_path='/workspace/projects/speech/processor'):
    return setup_avqa_whisper_stage2(
        checkpoint_path=checkpoint_path,
        subfolder=subfolder,
        processor_path=processor_path,
        modules_to_save=['music_projector'],
    )


def run_avqa_whisper_fullres_v3_stage2(model, processor, lora_config,
                                       train_dataset=None, val_dataset=None,
                                       output_dir=None, resume_from_checkpoint=None,
                                       seed=42, push=True):
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    return run_avqa_whisper_full_stage2(
        model=model, processor=processor, lora_config=lora_config,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or f"{AVQA_WHISPER_FULLRES_V3_STAGE2_DIR}_{run_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_MUSIC_DIR,
        experiment_tag='whisper_fullres_v3',
        seed=seed, push=push,
    )


# ── AVQA whisper_fullres_notts: text questions instead of TTS→Whisper ─────────
# Motivation: reviewer objected to routing clean text through TTS→Whisper (4.71% WER).
#   Passing questions as text tokens removes that lossy channel.
# Design (revised 2026-06-11): train BOTH stages with text questions on the usual 8k
#   data budget — same as v2/v3 — so the ONLY difference vs v3 is TTS→text throughout.
#   - Stage 1 (this run) trains music_projector with text_question=True, so the projector
#     co-adapts to the text-question context it will actually operate in. (Earlier attempt
#     reused the TTS-trained avqa_stage1_whisper_fullres_v2 and got 71.36% — question-
#     ignoring; see training_history.md. Suspected cause: projector trained in TTS context,
#     mismatched at Stage 2. Training Stage 1 with text removes that confound.)
#   - Stage 2: LoRA on LLM + music_projector. audio_projector excluded from modules_to_save
#     (it is never invoked — no TTS audio path in text mode).
# Comparison: 8k/8k like every other run → drops straight into the results table vs v3.
# Expected: accuracy ≥ 97.31% if text questions are genuinely fine.

AVQA_WHISPER_FULLRES_NOTTS_STAGE1_DIR = '/workspace/projects/speech/avqa_stage1_whisper_fullres_notts_checkpoint'
AVQA_WHISPER_FULLRES_NOTTS_STAGE2_DIR = '/workspace/projects/speech/avqa_stage2_whisper_fullres_notts_checkpoint'


def run_avqa_whisper_fullres_notts_stage1(model=None, processor=None,
                                          output_dir=None, resume_from_checkpoint=None,
                                          match_wrapper=False):
    """Stage 1 for the notts variant: music_projector trained with TEXT questions.

    Builds text-question datasets (text_question=True → no TTS audio) and delegates to
    run_avqa_whisper_full_stage1. The model/architecture is identical to v2 Stage 1
    (init_avqa_whisper_model); only the question is delivered as text tokens instead of
    TTS→Whisper audio, so the music_projector co-adapts to the text context.

    match_wrapper=False (default): the bare notts arm — question text REPLACES the prompt.
      Pushes to avqa/ablations/whisper_fullres_notts/stage1 (the original notts).
    match_wrapper=True: single-variable control for sec:delivery — the text arm mirrors the
      TTS arm [video, music, text(Q), "Answer the question."], so the ONLY difference vs TTS
      is the question's pathway. Distinct tag → whisper_fullres_notts_matched
      (never clobbers the original notts). Threaded into BOTH datasets.
    """
    from avqa.dataset import AVQADataset
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag    = 'whisper_fullres_notts_matched' if match_wrapper else 'whisper_fullres_notts'
    suffix = '_matched' if match_wrapper else ''
    train_dataset = AVQADataset(split='train', max_samples=8000,
                                music_dir=WHISPER_FULLRES_MUSIC_DIR,
                                require_video=False, text_question=True,
                                match_wrapper=match_wrapper)
    val_dataset   = AVQADataset(split='val',
                                music_dir=WHISPER_FULLRES_MUSIC_DIR,
                                require_video=False, text_question=True,
                                match_wrapper=match_wrapper)
    return run_avqa_whisper_full_stage1(
        model=model, processor=processor,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or f"{AVQA_WHISPER_FULLRES_NOTTS_STAGE1_DIR}{suffix}_{run_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_MUSIC_DIR,
        experiment_tag=tag,
    )


def setup_avqa_whisper_fullres_notts_stage2(checkpoint_path=HF_REPO,
                                            subfolder=None,
                                            processor_path='/workspace/projects/speech/processor',
                                            match_wrapper=False):
    """Load the TEXT-trained notts Stage 1 checkpoint (not the TTS v2 one) for Stage 2.

    subfolder=None → defaults to the notts Stage 1 subfolder matching match_wrapper:
      avqa/ablations/whisper_fullres_notts/stage1 (bare) or
      avqa/ablations/whisper_fullres_notts_matched/stage1 (matched control).
    An explicit subfolder= always wins (e.g. resuming a specific run).
    """
    if subfolder is None:
        subfolder = avqa_hf_path(
            'whisper_fullres_notts_matched' if match_wrapper else 'whisper_fullres_notts', 1)
    return setup_avqa_whisper_stage2(
        checkpoint_path=checkpoint_path,
        subfolder=subfolder,
        processor_path=processor_path,
        modules_to_save=['music_projector'],
    )


def run_avqa_whisper_fullres_notts_stage2(model, processor, lora_config,
                                          output_dir=None, resume_from_checkpoint=None,
                                          num_train_epochs=1, match_wrapper=False):
    """Stage 2 LoRA for the notts variant. match_wrapper=True → matched control:
    datasets mirror the TTS arm and push to avqa/ablations/whisper_fullres_notts_matched/stage2
    (never clobbers the original notts). Must match the Stage 1 / setup match_wrapper."""
    from avqa.dataset import AVQADataset
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag    = 'whisper_fullres_notts_matched' if match_wrapper else 'whisper_fullres_notts'
    suffix = '_matched' if match_wrapper else ''
    train_dataset = AVQADataset(split='train', max_samples=8000,
                                music_dir=WHISPER_FULLRES_MUSIC_DIR,
                                require_video=False, text_question=True,
                                match_wrapper=match_wrapper)
    val_dataset   = AVQADataset(split='val',
                                music_dir=WHISPER_FULLRES_MUSIC_DIR,
                                require_video=False, text_question=True,
                                match_wrapper=match_wrapper)
    return run_avqa_whisper_full_stage2(
        model=model, processor=processor, lora_config=lora_config,
        train_dataset=train_dataset, val_dataset=val_dataset,
        output_dir=output_dir or f"{AVQA_WHISPER_FULLRES_NOTTS_STAGE2_DIR}{suffix}_{run_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        music_dir=WHISPER_FULLRES_MUSIC_DIR,
        experiment_tag=tag,
        num_train_epochs=num_train_epochs,
    )
