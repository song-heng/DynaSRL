import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import json
import numpy as np
import logging
from transformers import AutoTokenizer, TrainingArguments, Trainer
from modeling_dynasrl import DynaSRLModel
from data_utils import DynaSRLDataset, DynaSRLCollator
from metrics_utils import DynaSRLMetrics
from train_log import TrainLogger
import transformers.utils.logging as hf_logging

# Silence TF32 deprecation warning (New PyTorch API)
torch.backends.cuda.matmul.fp32_precision = 'high'
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================================
# Global Configuration
# =========================================================================
TARGET_MODELS = [
    # "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-1.7B",
    # "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3-8B",
]

CURRENT_DATASET = "cpb1"
DATA_DIR = "/root/DynaSRL/data/input"
BASE_OUTPUT_DIR = "/root/autodl-tmp/models"

# [Ablation Control] Explicitly disable MLP at the beginning
# If True, the model will not use the MLP Projector and will not input schema embeddings.
DISABLE_MLP = True


# =========================================================================
# Custom Trainer for Saving LoRA + MLP Projector Only
# =========================================================================
class Phase1Trainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """
        Save only LoRA and MLP Projector instead of the full model.
        Resolves "layer missing" evaluation checkpoint errors by building dummy standard parameters.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving LoRA & Projector checkpoint to {output_dir}...")
        
        # Save the internal PeftModel (LoRA parameters) directly
        if hasattr(self.model, "llm"):
            self.model.llm.save_pretrained(output_dir)
        else:
            logger.warning("No llm attribute found. Skipping LoRA save.")

        # Save Projector
        if hasattr(self.model, "projector") and self.model.projector is not None:
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


# =========================================================================
# Main Execution Function per Model
# =========================================================================
def run_training_for_model(model_name_or_path, args):
    # Determine short model name for output directory
    model_short_name = model_name_or_path.split("/")[-1]
    dataset_name = args.dataset
    
    local_model_path = os.path.join(args.model_root, model_short_name)
    output_dir = os.path.join(args.model_root, f"{model_short_name}-{dataset_name}")
    
    # [New] Update output_dir for woMLP ablation
    if args.disable_mlp_projector:
        output_dir += "-woMLP"
    
    print(f"\n{'='*60}")
    print(f"Starting Training for Model: {model_name_or_path}")
    print(f"Local Model Path: {local_model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Check if local model directory actually exists before proceeding
    if not os.path.exists(local_model_path):
        logger.error(f"Local model path not found: {local_model_path}. Did you run download_model.py?")
        return
    
    train_data_path = os.path.join(args.data_dir, dataset_name, f"{dataset_name}_train_ins.jsonl")
    dev_data_path = os.path.join(args.data_dir, dataset_name, f"{dataset_name}_dev_ins.jsonl")
    schema_path = os.path.join(args.data_dir, dataset_name, f"{dataset_name}_schema.json")

    print(f"Loading Tokenizer from {local_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Model from {local_model_path}...")
    model = DynaSRLModel(
        local_model_path, 
        use_lora=True,
        use_projector=not args.disable_mlp_projector
    )
    
    print("Preparing Datasets...")
    max_eval_samples = args.max_eval_samples if args.max_eval_samples > 0 else None
    
    use_projector = not args.disable_mlp_projector
    print(f"  >>> Projector (MLP) Enabled: {use_projector}")
    print(f"  >>> Schema Embedding Input Enabled: {use_projector}")

    train_dataset = DynaSRLDataset(
        train_data_path, 
        schema_path, 
        tokenizer, 
        use_projector=use_projector
    )
    dev_dataset = DynaSRLDataset(
        dev_data_path, 
        schema_path, 
        tokenizer, 
        max_samples=max_eval_samples,
        use_projector=use_projector
    )

    print("-" * 40)
    print(f"  >>> Train Samples: {len(train_dataset)}")
    print(f"  >>> Eval Samples: {len(dev_dataset)}")
    print("-" * 40)

    # === Metric Evaluation ===
    metric_tool = DynaSRLMetrics()

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    def compute_metrics(eval_preds):
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

    # === Adaptive save_steps calculation ===
    save_steps = args.save_steps
    if save_steps == -1:
        import math
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        effective_bs = args.batch_size * args.grad_accum * n_gpus
        steps_per_epoch = math.ceil(len(train_dataset) / effective_bs)
        total_steps = steps_per_epoch * args.num_epochs
        
        target_evals = 6
        eval_steps = max(1, math.floor(total_steps / target_evals))
        
        print(f"[Adaptive Eval] Dataset Size: {len(train_dataset)}, Effective BS: {effective_bs}")
        print(f"[Adaptive Eval] Total Steps: {total_steps}, Target Evals: {target_evals} -> eval_steps = {eval_steps}")
        save_steps = eval_steps

    print("Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        group_by_length=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        logging_dir=os.path.join(output_dir, "runs")
    )

    trainer = Phase1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DynaSRLCollator(tokenizer, use_projector=use_projector),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    print("Starting Training...")
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            last_checkpoint = True
            print(f"Resuming from checkpoint in {output_dir}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"Saving best model to {output_dir}...")
    trainer.save_model(output_dir)

    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    log_history = trainer.state.log_history
    log_path = os.path.join(output_dir, "loss_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    # === Structured TrainLogger output to /log/phase1/ ===
    try:
        train_logger = TrainLogger(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            current_dataset_name=dataset_name,
            run_name=model_short_name,
            phase="phase1"
        )
        train_logger.collect_from_trainer(trainer)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        phase1_log_dir = os.path.join(project_root, "log", "phase1")
        train_logger.save(log_dir=phase1_log_dir)
    except Exception as e:
        print(f"[TrainLogger Warning] Exception during log saving: {e}")

    print(f"Training finished for {model_name_or_path}! Total Time: {metrics['train_runtime']:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(TARGET_MODELS),
                        help="Comma-separated base model repo IDs or local model names.")
    parser.add_argument("--dataset", type=str, default=CURRENT_DATASET)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--model_root", type=str, default=BASE_OUTPUT_DIR)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_accum", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=-1, help="If -1, adaptively compute to evaluate 10 times.")
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    
    # === [New] Ablation Switch ===
    parser.add_argument("--disable_mlp_projector", action="store_true", default=DISABLE_MLP,
                        help="Ablation: Do not use/train MLP Projector.")
    
    args = parser.parse_args()

    model_paths = [item.strip() for item in args.models.split(",") if item.strip()]
    for model_path in model_paths:
        run_training_for_model(model_path, args)


if __name__ == "__main__":
    main()
