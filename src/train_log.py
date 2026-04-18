"""
Train Logger Module for DynaSRL Phase 1 & Phase 2.
Provides structured JSON logging for training experiments,
capturing run info, config, hardware details, and step-level loss records.

Usage:
    from train_log import TrainLogger

    # Initialize after args are parsed and datasets are loaded
    tlogger = TrainLogger(args, train_dataset, dev_dataset)

    # After training, collect logs from trainer and save
    tlogger.collect_from_trainer(trainer)
    tlogger.save()
"""

import os
import sys
import json
import time
import platform
from datetime import datetime, timezone, timedelta


def _get_hardware_info():
    """Collect hardware and software environment information."""
    try:
        import torch
        info = {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
        }

        # GPU information
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024 ** 3), 2),
                })
            info["gpus"] = gpus
            info["gpu_summary"] = gpus[0]["name"] if len(gpus) == 1 else f"{len(gpus)}x {gpus[0]['name']}"
            info["cuda_version"] = torch.version.cuda or "N/A"
        else:
            info["gpu_count"] = 0
            info["gpus"] = []
            info["gpu_summary"] = "CPU only"
            info["cuda_version"] = "N/A"

        # Transformers version
        try:
            import transformers
            info["transformers_version"] = transformers.__version__
        except ImportError:
            info["transformers_version"] = "N/A"

        # PEFT version
        try:
            import peft
            info["peft_version"] = peft.__version__
        except ImportError:
            info["peft_version"] = "N/A"

    except Exception as e:
        import traceback
        try:
            info["hardware_info_error"] = str(e)
            info["traceback"] = traceback.format_exc()
        except UnboundLocalError:
            info = {
                "hardware_info_error": str(e),
                "traceback": traceback.format_exc()
            }


    return info


class TrainLogger:
    """
    Lightweight JSON logger for DynaSRL Phase 1 & Phase 2 training experiments.

    Design principles:
        - Zero overhead during training (only collects logs post-training from Trainer's log_history)
        - Decoupled from Trainer internals: no monkey-patching or callback injection
        - Single .save() call at the end to write the JSON file
    """

    def __init__(self, args, train_dataset, dev_dataset,
                 current_dataset_name=None, run_name=None, random_seed=42, phase="phase2"):
        """
        Initialize the logger with run configuration.

        :param args: Parsed argparse.Namespace from train_phase1.py or train_phase2.py
        :param train_dataset: Training dataset instance (for sample count)
        :param dev_dataset: Dev dataset instance (for sample count)
        :param current_dataset_name: Name string of the current dataset (e.g. 'phee')
        :param run_name: Optional explicit run name (e.g. model_info['name'])
        :param random_seed: Random seed used in training
        :param phase: Training phase identifier ('phase1' or 'phase2')
        """
        self._start_time = time.time()
        self._timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S")

        self._args = args
        self._phase = phase
        self._dataset_name = current_dataset_name or "unknown"
        self._run_name = run_name
        self._random_seed = random_seed
        self._train_samples = len(train_dataset)
        self._dev_samples = len(dev_dataset)

        # Derive base model name from path (last component)
        # Phase 1 may use a different attribute name, so use safe getattr
        base_path = getattr(args, "base_model_path", None) or getattr(args, "model_path", "unknown")
        self._base_model_name = os.path.basename(str(base_path).rstrip("/\\"))
        
        # [Improvement] Capture descriptive Phase 1 checkpoint name (safe for Phase 1 which has no phase1_ckpt_path)
        ckpt_path = getattr(args, "phase1_ckpt_path", None)
        if ckpt_path:
            ckpt_path = ckpt_path.rstrip("/\\")
            last_part = os.path.basename(ckpt_path)
            if last_part.startswith("checkpoint-"):
                self._phase1_ckpt_name = os.path.basename(os.path.dirname(ckpt_path))
            else:
                self._phase1_ckpt_name = last_part
        else:
            self._phase1_ckpt_name = "N/A"

        # Ablation flags
        self._glad_enabled = not getattr(args, "disable_glad", False)
        self._mlp_enabled = not getattr(args, "disable_mlp_projector", False)

        # Will be populated after training
        self._step_logs = []
        self._eval_logs = []
        self._training_summary = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def collect_from_trainer(self, trainer):
        """
        Extract step-level and eval-level logs from HuggingFace Trainer's log_history.
        This should be called after trainer.train() completes.

        :param trainer: The GladTrainer / Trainer instance after training
        """
        for entry in trainer.state.log_history:
            if "loss" in entry and "eval_loss" not in entry:
                # Training step log
                self._step_logs.append({
                    "step": entry.get("step"),
                    "epoch": round(entry.get("epoch", 0), 4),
                    "loss": round(entry.get("loss", 0), 6),
                    "learning_rate": entry.get("learning_rate"),
                    "grad_norm": round(entry.get("grad_norm", 0), 6) if "grad_norm" in entry else None,
                })

            elif "eval_loss" in entry:
                # Evaluation log
                eval_record = {
                    "step": entry.get("step"),
                    "epoch": round(entry.get("epoch", 0), 4),
                    "eval_loss": round(entry.get("eval_loss", 0), 6),
                }
                # Capture DynaSRL-specific metrics if present
                for key in ("eval_tp", "eval_fp", "eval_fn", "f1", "eval_f1"):
                    if key in entry:
                        val = entry[key]
                        eval_record[key] = round(val, 6) if isinstance(val, float) else val
                self._eval_logs.append(eval_record)

        # Build training summary
        elapsed_sec = round(time.time() - self._start_time, 2)
        self._training_summary["train_runtime_sec"] = elapsed_sec

        if self._step_logs:
            self._training_summary["total_steps"] = self._step_logs[-1]["step"]
            self._training_summary["final_loss"] = self._step_logs[-1]["loss"]

        if self._eval_logs:
            self._training_summary["total_eval_steps"] = len(self._eval_logs)
            # Find best F1
            f1_entries = [e for e in self._eval_logs if "f1" in e or "eval_f1" in e]
            if f1_entries:
                best = max(f1_entries, key=lambda x: x.get("eval_f1", x.get("f1", 0)))
                self._training_summary["best_step"] = best["step"]
                self._training_summary["best_f1"] = best.get("eval_f1", best.get("f1"))

        if hasattr(trainer.state, "num_train_epochs"):
            self._training_summary["total_epochs"] = trainer.state.num_train_epochs

    def build_log_dict(self):
        """
        Assemble the full log dictionary.

        :return: Complete log dictionary ready for JSON serialization
        """
        try:
            # Better experiment name: use run_name if available, else dataset-based

            exp_name = self._run_name if self._run_name else f"dynasrl_{self._phase}_{self._dataset_name}"
            
            log = {
                "run_info": {
                    "experiment_name": exp_name,
                    "run_id": self._run_name,
                    "phase": self._phase,
                    "timestamp": self._timestamp,
                    "random_seed": self._random_seed,
                    "framework": "pytorch+transformers+peft",
                },
                "environment": _get_hardware_info(),
                "model_config": {
                    "base_model_name": self._base_model_name,
                    "base_model_path": getattr(self._args, "base_model_path", "unknown"),
                    "phase1_checkpoint_name": self._phase1_ckpt_name,
                    "phase1_checkpoint_path": getattr(self._args, "phase1_ckpt_path", "unknown"),
                    "lora_rank": 64,
                    "lora_alpha": 128,
                    "lora_dropout": 0.05,
                    "lora_target_modules": [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ],
                },
                "ablation": {
                    "glad_enabled": self._glad_enabled,
                    "mlp_projector_enabled": self._mlp_enabled,
                },
                "training_hyperparams": {
                    "num_epochs": getattr(self._args, "num_epochs", None),
                    "batch_size": getattr(self._args, "batch_size", None),
                    "gradient_accumulation_steps": getattr(self._args, "grad_accum", None),
                    "learning_rate": getattr(self._args, "learning_rate", None),
                    "lr_scheduler": "cosine",
                    "warmup_ratio": 0.05,
                    "bf16": True,
                    "glad_rho": getattr(self._args, "glad_rho", None) if self._glad_enabled else None,
                    "glad_alpha": getattr(self._args, "glad_alpha", None) if self._glad_enabled else None,
                    "save_steps": getattr(self._args, "save_steps", None),
                    "max_train_samples": getattr(self._args, "max_train_samples", None),
                    "logging_steps": 10,
                },
                "dataset_info": {
                    "dataset_name": self._dataset_name,
                    "train_data_path": getattr(self._args, "train_data_path", "unknown"),
                    "dev_data_path": getattr(self._args, "dev_data_path", "unknown"),
                    "schema_path": getattr(self._args, "schema_path", "unknown"),
                    "train_samples": self._train_samples,
                    "dev_samples": self._dev_samples,
                    "max_seq_length": 2048,
                },
                "training_summary": self._training_summary,
                "step_logs": self._step_logs,
                "eval_logs": self._eval_logs,
            }
            return log
        except Exception as e:
            import traceback
            print(f"[TrainLogger] Error building full log dict: {e}")
            # Fallback barebone dictionary
            return {
                "run_info": {
                    "experiment_name": f"dynasrl_{self._phase}_{self._dataset_name}",
                    "timestamp": self._timestamp,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                },
                "training_summary": getattr(self, "_training_summary", {}),
                "step_logs": getattr(self, "_step_logs", []),
                "eval_logs": getattr(self, "_eval_logs", [])
            }

    def save(self, log_dir=None):
        """
        Write the log JSON file to disk.
        File naming: {base_model}_{dataset}[_woGLAD][_woMLP]_{timestamp}.json

        :param log_dir: Optional custom log directory. Defaults to <project_root>/log/
        :return: Absolute path of the saved log file
        """
        if log_dir is None:
            # Default: project_root/log/
            # Assuming src/ is one level below project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(project_root, "log")

        os.makedirs(log_dir, exist_ok=True)

        # Build filename components
        # Priority: run_name > base_model + dataset
        if self._run_name:
            parts = [self._run_name, self._dataset_name]
        else:
            parts = [self._base_model_name, self._dataset_name]

        # Build ablation tag matching project convention (woGLADwoMLP, not woGLAD-woMLP)
        ablation_tag = ""
        if not self._glad_enabled:
            ablation_tag += "woGLAD"
        if not self._mlp_enabled:
            ablation_tag += "woMLP"
        if ablation_tag:
            parts.append(ablation_tag)

        ts_str = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
        parts.append(ts_str)

        filename = "-".join(parts) + ".json"
        filepath = os.path.join(log_dir, filename)

        log_data = self.build_log_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"[TrainLogger] Log saved to: {filepath}")
        return filepath
