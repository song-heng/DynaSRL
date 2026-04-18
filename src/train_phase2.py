import os
os.environ["OMP_NUM_THREADS"] = "1"  # Silence libgomp warning by providing a valid numeric value
import torch
import argparse
import logging
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from modeling_dynasrl import DynaSRLModel
from data_utils import DynaSRLDataset, DynaSRLCollator
from metrics_utils import DynaSRLMetrics
from train_log import TrainLogger
import transformers.utils.logging as hf_logging

# Use 'tf32' precision for modern GPUs (maps to 'high' and avoids backend warnings)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.conv.fp32_precision = 'tf32'

# [Optimization] Set CUDA allocation configuration to avoid fragmentation and optimize memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress transformers and peft verbosity (hides Config dump)
hf_logging.set_verbosity_warning()


# =========================================================================
# Dataset Configuration for Phase 2
# =========================================================================
# Global variable to specify the current dataset to train on
CURRENT_DATASET = "phee"  # Options: "conll2009_cn", "conll2009_en", "fire", "phee", "fabner"

# --- Global Ablation Switches ---
DISABLE_GLAD = False
DISABLE_MLP_PROJECTOR = False
DISABLE_LOGGING = False

DATA_DIR = "/root/DynaSRL/data/input"
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
PHASE1_CKPT_PATH = "/root/autodl-tmp/models/Qwen3-8B-cpb1/checkpoint-1602"

# Target datasets for phase 2 (excluding cpb1)
TARGET_DATASETS = ["conll2009_cn", "conll2009_en", "conll2012_cn", "conll2012_en",
                   "ace2005","fire", "phee", "fabner",
                   "framenet17"]

# 'cpb1-Q14B': "/root/autodl-tmp/models/Qwen3-14B-cpb1/checkpoint-3204",
# 'cpb1-Q8B': "/root/autodl-tmp/models/Qwen3-8B-cpb1/checkpoint-1602",
# 'cpb1-Q4B': "/root/autodl-tmp/models/Qwen3-4B-cpb1/checkpoint-1602",
# 'cpb1-Q1.7B': "/root/autodl-tmp/models/Qwen3-1.7B-cpb1/checkpoint-1246",
# 'cpb1-L3B': "/root/autodl-tmp/models/Llama-3.2-3B-cpb1/checkpoint-890",
# 'cpb1-L1B': "/root/autodl-tmp/models/Llama-3.2-1B-cpb1/checkpoint-1068",

# Dynamic dictionary for datasets
DATASET_CONFIG = {
    dataset: {
        "train": f"{DATA_DIR}/{dataset}/{dataset}_train_ins.jsonl",
        "dev": f"{DATA_DIR}/{dataset}/{dataset}_dev_ins.jsonl",
        "schema": f"{DATA_DIR}/{dataset}/{dataset}_schema.json",
        "output": f"{BASE_MODEL_PATH}-{dataset}"
    } for dataset in TARGET_DATASETS
}

# =========================================================================
# Hyperparameter Configuration per Dataset
# =========================================================================
# HYPERPARAM_CONFIG = {
#     "conll2009_cn": {"num_epochs": 3, "batch_size": 4, "grad_accum": 3, "learning_rate": 5e-5},
#     "conll2009_en": {"num_epochs": 3, "batch_size": 4, "grad_accum": 3, "learning_rate": 5e-5},
#     "conll2012_cn": {"num_epochs": 3, "batch_size": 4, "grad_accum": 3, "learning_rate": 5e-5},
#     "conll2012_en": {"num_epochs": 3, "batch_size": 4, "grad_accum": 3, "learning_rate": 5e-5},
#     "fire": {"num_epochs": 5, "batch_size": 4, "grad_accum": 4, "learning_rate": 5e-5},
#     "phee": {"num_epochs": 5, "batch_size": 4, "grad_accum": 4, "learning_rate": 5e-5},
#     "fabner": {"num_epochs": 5, "batch_size": 4, "grad_accum": 4, "learning_rate": 5e-5},
#     "ace2005": {"num_epochs": 5, "batch_size": 4, "grad_accum": 4, "learning_rate": 5e-5},
#     "framenet17": {"num_epochs": 5, "batch_size": 4, "grad_accum": 4, "learning_rate": 5e-5},
# }

HYPERPARAM_CONFIG = {
    "conll2009_cn": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "conll2009_en": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "conll2012_cn": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "conll2012_en": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "fire": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "phee": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "fabner": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "ace2005": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
    "framenet17": {"num_epochs": 3, "batch_size": 4, "grad_accum": 4, "learning_rate": 2e-5},
}


# =========================================================================
# GLAD Trainer (Gradient-Regularized Optimization)
# Corresponds to Paper Phase 2: Regularized Dual-Adaptation
# =========================================================================
class GladTrainer(Trainer):
    # [Modification] Add enable_glad switch parameter
    # def __init__(self, glad_rho=0.05, glad_alpha=0.5, enable_glad=True, **kwargs):
    def __init__(self, glad_rho=0.01, glad_alpha=0.1, enable_glad=True, **kwargs):

        super().__init__(**kwargs)
        self.glad_rho = glad_rho
        self.glad_alpha = glad_alpha
        self.enable_glad = enable_glad  # Save switch state

    def save_model(self, output_dir=None, _internal_call=False):
        """
        [Modification] Only save LoRA and MLP Projector instead of the full model.
        Reference: Phase 2 yesterday's final save logic.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving LoRA & Projector checkpoint to {output_dir}...")
        
        # Save the internal PeftModel (LoRA parameters) directly
        self.model.llm.save_pretrained(output_dir)

        # Save Projector
        if self.model.projector is not None:
            torch.save(self.model.projector.state_dict(), os.path.join(output_dir, "mlp_projector.bin"))
        else:
            logger.info("Skipping Projector save (Projector is None/Disabled).")

        # Save a minimal state_dict of trainable parameters to satisfy Trainer's load_best_model_at_end
        state_dict = self.model.state_dict()
        trainable_keys = [k for k, v in self.model.named_parameters() if v.requires_grad]
        trainable_state_dict = {k: state_dict[k] for k in trainable_keys if k in state_dict}
        torch.save(trainable_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        # Save training args 
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_best_model(self):
        """
        Overriding to load with strict=False to suppress 'missing keys' warnings
        caused by our lightweight (LoRA + Projector only) save strategy.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (partial state_dict)...")
        state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin"), map_location="cpu")
        
        # Suppress the "missing keys" warning from transformers/peft during partial load
        hf_logging.set_verbosity_error()
        self.model.load_state_dict(state_dict, strict=False)
        hf_logging.set_verbosity_info()

    def training_step(self, model, inputs, num_items_in_batch=None):
        '''
        GLAD Regularization: Look-ahead Gradient Step.
        :param model: The model to train
        :param inputs: Input dictionary
        :param num_items_in_batch: Number of items in the batch
        :return: Detached loss tensor
        '''
        # [New] Ablation study switch: If GLAD is disabled, use standard training_step
        if not self.enable_glad:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # --- Original GLAD logic below ---
        model.train()
        inputs = self._prepare_inputs(inputs)

        # get the gradient accumulation factor
        ga_steps = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))



        # Preserve accumulated gradients from previous micro-batches.
        # Without this, GLAD's internal two-pass backward interferes with gradient accumulation.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        accumulated_grads = []
        for p in trainable_params:
            if p.grad is not None:
                accumulated_grads.append(p.grad.detach().clone())
                p.grad = None
            else:
                accumulated_grads.append(None)

        # 1. Compute original Loss and Gradient g
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()


        self.accelerator.backward(loss / ga_steps)

        # Get all trainable parameters with gradients
        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]

        if len(params) > 0:
            # Save original gradients and parameter references
            orig_grads = [p.grad.clone() for p in params]

            # Compute global gradient norm
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in orig_grads]))

            # Compute perturbation scale = rho / ||g||
            scale = self.glad_rho / (grad_norm + 1e-12)

            # 2. Apply perturbation (In-place)
            for p, g in zip(params, orig_grads):
                p.data.add_(g * scale)  # Theta_new = Theta + epsilon
                p.grad = None  # Clear gradients for second backward pass

            # 3. Compute perturbed Loss' and Gradient g'
            with self.compute_loss_context_manager():
                loss_prime = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            if self.args.n_gpu > 1:
                loss_prime = loss_prime.mean()

            self.accelerator.backward(loss_prime / ga_steps)

            # 4. Global Gradient Orthogonal Projection Check (Safety Mechanism)
            # Compute global dot product <g, g'>
            for p, g_orig in zip(params, orig_grads):
                if p.grad is None:
                    p.grad = torch.zeros_like(g_orig)
            dot_product = sum(torch.sum(g_orig * p.grad) for p, g_orig in zip(params, orig_grads))

            if dot_product < 0:
                # <g, g> is simply grad_norm squared
                orig_norm_sq = grad_norm ** 2
                projection_scalar = dot_product / (orig_norm_sq + 1e-12)
                
                # Update g' (p.grad) to be orthogonal to g (orig_grads)
                for p, g_orig in zip(params, orig_grads):
                    p.grad.sub_(g_orig * projection_scalar)

            # 5. Gradient interpolation and parameter restoration
            for p, g_orig in zip(params, orig_grads):
                g_prime = p.grad  # Get g' (possibly projected)

                # Combined gradient: (1-alpha)*g + alpha*g'
                new_grad = (1 - self.glad_alpha) * g_orig + self.glad_alpha * g_prime

                # Restore parameters (Undo perturbation)
                p.data.sub_(g_orig * scale)

                # Set final gradient
                p.grad = new_grad

        # Restore previously accumulated gradients.
        for p, acc_g in zip(trainable_params, accumulated_grads):
            if acc_g is None:
                continue
            if p.grad is None:
                p.grad = acc_g
            else:
                p.grad.add_(acc_g)

        return loss.detach()


# =========================================================================
# Core Function: Intelligent Weight Loader
# =========================================================================
# [Modification] Add load_projector parameter
def load_checkpoint_weights(model, checkpoint_path, load_projector=True):
    '''
    Load LoRA and Projector weights separately from the lightweight checkpoint.
    :param model: The DynaSRL model instance
    :param checkpoint_path: Path to the checkpoint directory
    :param load_projector: Boolean to decide whether to load projector weights
    :return: None
    '''
    # 1. Load LoRA Weights (Look for .safetensors first, then .bin)
    lora_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(checkpoint_path, "adapter_model.bin")

    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter weights from {lora_path}...")
        if lora_path.endswith(".safetensors"):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location="cpu")
        
        # Load weights into the PeftModel (model.llm)
        msg = model.llm.load_state_dict(lora_state_dict, strict=False)
        logger.info(f"LoRA Load Results: {msg}")
    else:
        logger.warning(f"LoRA adapter weights not found in {checkpoint_path}")

    # 2. Load Projector Weights
    if load_projector and hasattr(model, 'projector') and model.projector is not None:
        mlp_path = os.path.join(checkpoint_path, "mlp_projector.bin")
        if os.path.exists(mlp_path):
            logger.info(f"Loading Projector weights from {mlp_path}...")
            projector_dict = torch.load(mlp_path, map_location="cpu")
            model.projector.load_state_dict(projector_dict, strict=True)
        else:
            logger.warning(f"No Projector weights found at {mlp_path}! Using random initialization.")
    else:
        logger.info("[Ablation] MLP Projector loading DISABLED or Model has no projector.")


# =========================================================================
# Main Execution
# =========================================================================
def main():
    global CURRENT_DATASET
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--dataset", type=str, default=CURRENT_DATASET)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    selected_dataset = bootstrap_args.dataset

    # Safety fallback if CURRENT_DATASET is mistakenly set inappropriately
    cfg = DATASET_CONFIG.get(selected_dataset)
    if not cfg:
        raise ValueError(f"Unknown CURRENT_DATASET: {selected_dataset}. Please choose from {TARGET_DATASETS}")

    parser = argparse.ArgumentParser(parents=[bootstrap_parser])

    # === Path Configuration ===
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--phase1_ckpt_path", type=str, default=PHASE1_CKPT_PATH)
    parser.add_argument("--train_data_path", type=str, default=cfg["train"])
    parser.add_argument("--dev_data_path", type=str, default=cfg["dev"])
    parser.add_argument("--schema_path", type=str, default=cfg["schema"])
    parser.add_argument("--output_dir", type=str, default=cfg["output"])
    parser.add_argument("--run_name", type=str, default=None, help="Custom identifier for this training run.")

    # === Parameter Configuration ===
    cfg_hp = HYPERPARAM_CONFIG.get(CURRENT_DATASET, HYPERPARAM_CONFIG["phee"])
    
    parser.add_argument("--num_epochs", type=int, default=cfg_hp["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=cfg_hp["batch_size"])
    parser.add_argument("--grad_accum", type=int, default=cfg_hp["grad_accum"])
    parser.add_argument("--learning_rate", type=float, default=cfg_hp["learning_rate"])
    parser.add_argument("--save_steps", type=int, default=-1, help="If -1, adaptively compute to evaluate 10 times.")
    parser.add_argument("--max_train_samples", type=int, default=3000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--glad_rho", type=float, default=0.05)
    parser.add_argument("--glad_alpha", type=float, default=0.5)

    # === [New] Ablation Experiment Switches ===
    parser.add_argument("--disable_glad", action="store_true",
                        help="Ablation: Disable GLAD regularization (use standard training).")
    parser.add_argument("--disable_mlp_projector", action="store_true",
                        help="Ablation: Do not load MLP Projector weights from Phase 1 checkpoint.")

    # === [New] Training Log Switch ===
    parser.add_argument("--disable_logging", action="store_true",
                        help="Disable JSON training log output to project log/ directory.")
    parser.add_argument("--run_sequential", action="store_true",
                        help="Run the internal sequential experiment launcher instead of a single Phase 2 job.")

    args = parser.parse_args()
    CURRENT_DATASET = args.dataset

    if args.run_sequential:
        run_sequential_training()
        return

    # =======================================================
    # Handle Global Switches and Arguments
    # =======================================================
    args.disable_glad = args.disable_glad or DISABLE_GLAD
    args.disable_mlp_projector = args.disable_mlp_projector or DISABLE_MLP_PROJECTOR
    args.disable_logging = args.disable_logging or DISABLE_LOGGING

    # [Refinement] Handle combined ablation naming
    if args.disable_glad and args.disable_mlp_projector:
        args.output_dir += "-woGLADwoMLP"
    elif args.disable_glad:
        args.output_dir += "-woGLAD"
    elif args.disable_mlp_projector:
        args.output_dir += "-woMLP"
    # =======================================================

    # 1. Load Tokenizer
    print(f"Loading Tokenizer from {args.base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Initialize Model Architecture
    print(f"Loading Base Model from {args.base_model_path}...")
    # Logic: If disable_mlp_projector is True, then use_projector is False
    model = DynaSRLModel(
        args.base_model_path,
        use_lora=False,
        use_projector=not args.disable_mlp_projector
    )

    # 3. Re-attach LoRA Config
    print("Configuring LoRA Adapter structure...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model.llm = get_peft_model(model.llm, peft_config)

    # 4. Smart Load Phase 1 Weights
    # [Modification] Pass inverse of disable_mlp_projector
    load_checkpoint_weights(
        model,
        args.phase1_ckpt_path,
        load_projector=not args.disable_mlp_projector  # If disable=True, then load=False
    )

    # 5. Ensure parameters are trainable
    model.llm.print_trainable_parameters()
    # Modification - Add null check protection, only set gradients if Projector exists
    if model.projector is not None:
        for param in model.projector.parameters():
            param.requires_grad = True

    # 6. Prepare Datasets
    print("Preparing Datasets...")
    max_samples_val = args.max_train_samples if args.max_train_samples > 0 else None
    max_eval_samples = args.max_eval_samples if args.max_eval_samples > 0 else None
    
    train_dataset = DynaSRLDataset(
        args.train_data_path,
        args.schema_path,
        tokenizer,
        max_samples=max_samples_val
    )
    dev_dataset = DynaSRLDataset(
        args.dev_data_path, 
        args.schema_path, 
        tokenizer,
        max_samples=max_eval_samples
    )

    # =======================================================
    # Adaptive Evaluation Steps Calculation
    # =======================================================
    if args.save_steps == -1:
        import math
        # Determine number of GPUs for effective batch size
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        effective_bs = args.batch_size * args.grad_accum * n_gpus
        
        # Calculate steps
        steps_per_epoch = math.ceil(len(train_dataset) / effective_bs)
        total_steps = steps_per_epoch * args.num_epochs
        
        target_evals = 6
        eval_steps = max(1, math.floor(total_steps / target_evals))
        
        print(f"[Adaptive Eval] Dataset Size: {len(train_dataset)}, Effective BS: {effective_bs}")
        print(f"[Adaptive Eval] Total Steps: {total_steps}, Target Evals: {target_evals} -> eval_steps = {eval_steps}")
        
        args.save_steps = eval_steps

    # === [Optional] Initialize Training Logger ===
    train_logger = None
    if not args.disable_logging:
        train_logger = TrainLogger(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            current_dataset_name=CURRENT_DATASET,
            run_name=args.run_name
        )
        print("[TrainLogger] Logging enabled. Will save JSON log after training.")

    # =================================================================
    # Evaluation Logic
    # =================================================================
    metric_tool = DynaSRLMetrics()

    def preprocess_logits_for_metrics(logits, labels):
        '''
        Preprocess logits for metrics to save memory.
        :param logits: Raw logits
        :param labels: Labels
        :return: Predicted IDs
        '''
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    def compute_metrics(eval_preds):
        '''
        Compute evaluation metrics.
        :param eval_preds: Tuple of (predictions, labels)
        :return: Metrics dictionary
        '''
        pred_ids, labels = eval_preds
        if isinstance(pred_ids, torch.Tensor):
            pred_ids = pred_ids.detach().cpu().to(torch.int64).numpy()

        vocab_size = tokenizer.vocab_size
        pred_ids = np.clip(pred_ids, 0, vocab_size - 1)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = labels.astype(np.int64)

        decoded_preds = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        metric_tool.reset()
        for p, l in zip(decoded_preds, decoded_labels):
            metric_tool.update(p, l)

        results = metric_tool.compute()
        print(
            f"\n[Eval Result] TP: {results['eval_tp']} | FP: {results['eval_fp']} | "
            f"FN: {results['eval_fn']} | F1: {results['f1']:.4f}"
        )
        return results

    # =================================================================

    # 7. Configure TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=8,
        report_to="none",
        log_level="warning",
        disable_tqdm=False,
        # [Critical Fix] Enable Gradient Checkpointing for larger models (e.g. 14B)
        gradient_checkpointing=False,
    )

    collator = DynaSRLCollator(tokenizer)

    # 8. Initialize GLAD Trainer
    # [Modification] Pass enable_glad parameter
    trainer = GladTrainer(
        glad_rho=args.glad_rho,
        glad_alpha=args.glad_alpha,
        enable_glad=not args.disable_glad,  # If disable=True, then enable=False
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # 9. Start Training
    if args.disable_glad:
        print("Starting Phase 2 Training (Standard Training - GLAD Disabled)...")
    else:
        print("Starting Phase 2 Training (Regularized Dual-Adaptation)...")

    try:
        trainer.train()
    finally:
        # === [Optional] Collect and Save Training Log ===
        if train_logger is not None:
            try:
                train_logger.collect_from_trainer(trainer)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                phase2_log_dir = os.path.join(project_root, "log", "phase2")
                train_logger.save(log_dir=phase2_log_dir)
            except Exception as e:
                print(f"[TrainLogger Warning] Exception occurred during log collection or saving: {e}")

    # 10. Save Final Results
    
    # Save the internal PeftModel (LoRA parameters) directly
    model.llm.save_pretrained(args.output_dir)

    # [Modification] Add null check: save only if Projector exists
    if model.projector is not None:
        torch.save(model.projector.state_dict(), os.path.join(args.output_dir, "mlp_projector.bin"))

    print(f"Phase 2 Training for {args.output_dir} Completed Successfully.")


# =========================================================================
# Sequential Training Runner
# =========================================================================
def run_sequential_training():
    """
    Run specific training tasks sequentially as per the 60-task optimization plan.
    Covers: Scalability (28), Transferability (5), Ablation (9), and Ablation II (18).
    """
    # --- 1. Scalability Configurations (28 tasks) ---
    scalability_models = [
        ("Qwen3-14B-cpb1", "/root/autodl-tmp/models/Qwen3-14B", "/root/autodl-tmp/models/Qwen3-14B-cpb1/checkpoint-3204"),
        ("Qwen3-8B-cpb1", "/root/autodl-tmp/models/Qwen3-8B", "/root/autodl-tmp/models/Qwen3-8B-cpb1/checkpoint-1602"),
        ("Qwen3-4B-cpb1", "/root/autodl-tmp/models/Qwen3-4B", "/root/autodl-tmp/models/Qwen3-4B-cpb1/checkpoint-1602"),
        ("Qwen3-1.7B-cpb1", "/root/autodl-tmp/models/Qwen3-1.7B", "/root/autodl-tmp/models/Qwen3-1.7B-cpb1/checkpoint-1246"),
        ("Llama-3.2-1B-cpb1", "/root/autodl-tmp/models/Llama-3.2-1B", "/root/autodl-tmp/models/Llama-3.2-1B-cpb1/checkpoint-1068"),
        ("Llama-3.2-3B-cpb1", "/root/autodl-tmp/models/Llama-3.2-3B", "/root/autodl-tmp/models/Llama-3.2-3B-cpb1/checkpoint-890"),
        ("Llama-3-8B-cpb1", "/root/autodl-tmp/models/Llama-3-8B", "/root/autodl-tmp/models/Llama-3-8B-cpb1/checkpoint-1782")
    ]
    scalability_datasets = ["conll2009_cn", "conll2009_en", "conll2012_cn", "conll2012_en"]
    
    scalability_tasks = []
    for name, base, ckpt in scalability_models:
        scalability_tasks.append({
            "name": name, "base": base, "ckpt": ckpt, "datasets": scalability_datasets
        })

    # --- 2. Transferability Configurations (5 tasks) ---
    transferability_tasks = [{
        "name": "Qwen3-8B-cpb1",
        "base": "/root/autodl-tmp/models/Qwen3-8B",
        "ckpt": "/root/autodl-tmp/models/Qwen3-8B-cpb1/checkpoint-1602",
        "datasets": ["ace2005", "fabner", "fire", "phee", "framenet17"]
    }]

    # --- 3. Ablation Configurations (9 tasks) ---
    all_datasets = ["conll2009_cn", "conll2009_en", "conll2012_cn", "conll2012_en", 
                    "ace2005", "fabner", "fire", "phee", "framenet17"]
    ablation_tasks = [{
        "name": "Qwen3-8B-cpb1",
        "base": "/root/autodl-tmp/models/Qwen3-8B",
        "ckpt": "/root/autodl-tmp/models/Qwen3-8B-cpb1/checkpoint-1602",
        "datasets": all_datasets,
        "disable_glad": True
    }]

    # --- 4. Ablation II Configurations (18 tasks) ---
    ablation2_tasks = [
        # woMLP
        {
            "name": "Qwen3-8B-cpb1-woMLP",
            "base": "/root/autodl-tmp/models/Qwen3-8B",
            "ckpt": "/root/autodl-tmp/models/Qwen3-8B-cpb1-woMLP/checkpoint-1782",
            "datasets": all_datasets,
            "disable_mlp_projector": True
        },
        # woGLADwoMLP
        {
            "name": "Qwen3-8B-cpb1-woMLP",
            "base": "/root/autodl-tmp/models/Qwen3-8B",
            "ckpt": "/root/autodl-tmp/models/Qwen3-8B-cpb1-woMLP/checkpoint-1782",
            "datasets": all_datasets,
            "disable_glad": True,
            "disable_mlp_projector": True
        }
    ]

    all_models_to_run = scalability_tasks + transferability_tasks + ablation_tasks + ablation2_tasks

    import sys
    import gc
    
    for model_info in all_models_to_run:
        for dataset in model_info["datasets"]:
            print("\n" + "="*80)
            tag = ""
            if model_info.get("disable_glad") and model_info.get("disable_mlp_projector"): tag = "[woGLADwoMLP]"
            elif model_info.get("disable_glad"): tag = "[woGLAD]"
            elif model_info.get("disable_mlp_projector"): tag = "[woMLP]"
            
            print(f"Starting Task: Model {model_info['name']} on Dataset {dataset} {tag}")
            print("="*80 + "\n")
            
            # Construct base output dir
            output_dir = f"/root/autodl-tmp/models/{model_info['name']}-{dataset}"
            
            # Prepare config for current dataset
            cfg = {
                "train": f"{DATA_DIR}/{dataset}/{dataset}_train_ins.jsonl",
                "dev": f"{DATA_DIR}/{dataset}/{dataset}_dev_ins.jsonl",
                "schema": f"{DATA_DIR}/{dataset}/{dataset}_schema.json",
            }
            
            # Mock sys.argv to pass arguments to main()
            args_list = [
                "train_phase2.py",
                "--base_model_path", model_info["base"],
                "--phase1_ckpt_path", model_info["ckpt"],
                "--train_data_path", cfg["train"],
                "--dev_data_path", cfg["dev"],
                "--schema_path", cfg["schema"],
                "--output_dir", output_dir,
                "--run_name", model_info["name"],
                "--max_eval_samples", "1000"
            ]
            
            if model_info.get("disable_glad"):
                args_list.append("--disable_glad")
            if model_info.get("disable_mlp_projector"):
                args_list.append("--disable_mlp_projector")
            
            sys.argv = args_list
            
            # Set CURRENT_DATASET global for loggers/configs that use it
            global CURRENT_DATASET
            CURRENT_DATASET = dataset
            
            try:
                main()
            except Exception as e:
                print(f"!!! Task Failed: Model {model_info['name']} on {dataset} !!!")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                # [Optimization] Aggressive memory clearing for sequential runs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Further force clearing
                if 'trainer' in locals(): del trainer
                if 'model' in locals(): del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\nAll 60 sequential training tasks completed.")


if __name__ == "__main__":
    main()
