# src_combo/wandb_utils
import os
import time
import wandb
from transformers import TrainerCallback
from huggingface_hub import login as hf_login

def _load_secrets():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # up from src/asr/ → src/ → project root
    path = os.path.join(proj_root, ".secrets")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip().removeprefix("export").strip()
                    os.environ.setdefault(key, val.strip())

# Load secrets into env
_load_secrets()

# Hugging Face login
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    hf_login(token=hf_token)

# W&B login
wandb_token = os.getenv("WANDB_API_KEY")
if wandb_token:
    wandb.login(key=wandb_token)

class WandBLossCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._train_start = time.time()
            self._epoch_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            elapsed = time.time() - self._epoch_start
            wandb.log(
                {"epoch_time_min": elapsed / 60},
                step=state.global_step,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            total = time.time() - self._train_start
            wandb.log({"total_train_time_min": total / 60})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if logs is not None:
            wandb.log(logs, step=state.global_step)

def init_wandb(run_id, training_args, lora_config=None, stage_name=None, exp_name="exp1",
               model_id=None, train_samples=None, eval_samples=None, resume_run_id=None):
    def get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    bs   = get(training_args, "per_device_train_batch_size") or 0
    accum = get(training_args, "gradient_accumulation_steps") or 1

    stable_id = resume_run_id or run_id

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "qwen2-vl-audio"),
        entity=os.getenv("WANDB_ENTITY"),
        id=stable_id,
        resume="allow",
        group=f"{exp_name}_{run_id}",   # one experiment group
        job_type=stage_name,            # stage1_projector, stage2_qlora, etc.
        name=f"{stage_name}_{run_id}",  # run name (ignored on resume)
        config={
            "run_id": run_id,
            "stage": stage_name,
            "model_id": model_id,
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "learning_rate": get(training_args, "learning_rate"),
            "batch_size": bs,
            "accum_steps": accum,
            "effective_batch_size": bs * accum,
            "epochs": get(training_args, "num_train_epochs"),
            "gradient_checkpointing": get(training_args, "gradient_checkpointing"),
            "lr_scheduler": get(training_args, "lr_scheduler_type"),
            "warmup_ratio": get(training_args, "warmup_ratio"),
            "lora_r": getattr(lora_config, "r", None),
            "lora_alpha": getattr(lora_config, "lora_alpha", None),
            "lora_dropout": getattr(lora_config, "lora_dropout", None),
        },
    )


def save_wandb_run_id(output_dir: str):
    """Persist the current W&B run id to output_dir so it can be reused on resume."""
    if wandb.run is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb.run.id)


def load_wandb_run_id(output_dir: str):
    """Return a saved W&B run id if one exists, else None."""
    path = os.path.join(output_dir, "wandb_run_id.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip() or None
    return None


def init_wandb_eval(stage_name: str, exp_name: str = "qwen2-vl-avqa-clap",
                    notes: str = None):
    """Lightweight wandb.init() for standalone eval cells (no training_args needed).

    Usage:
        init_wandb_eval("avqa_stage2_clap_test_eval")
        evaluate_avqa(model, processor, test_dataset, n=len(test_dataset), tag="test")
        wandb.finish()
    """
    from datetime import datetime
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "qwen2-vl-audio"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"{stage_name}_{run_id}",
        group=exp_name,
        job_type="eval",
        notes=notes,
        config={"stage": stage_name, "exp_name": exp_name},
    )