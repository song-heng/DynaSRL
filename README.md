# DynaSRL: Toward Schema-Aware Universal Semantic Role Labeling via Dynamic Instruction Following

![Python](https://img.shields.io/badge/Python-3.12-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-orange)

This repository contains the public core implementation of **DynaSRL**, a schema-aware semantic role labeling framework that reformulates SRL as a dynamic instruction-following task.

<div align="center">
  <img src="dynasrl_overview.png" alt="DynaSRL overview" width="900"/>
</div>

## Introduction

DynaSRL decouples role semantics from the model architecture by treating schema definitions as explicit inputs. Instead of fixing label meaning inside a task-specific head, the framework conditions a large language model on natural-language schema descriptions and adapts it to new SRL domains through a two-stage training pipeline.

### Key Features

- **Dynamic Schema Adaptation:** Decouples role semantics from the model architecture, breaking the "Schema Barrier".
- **Schema-Conditioned Instruction Mechanism:** Projects natural language definitions into continuous latent vectors to guide the LLM's attention.
- **Regularized Dual-Adaptation:** A specialized optimizer that seeks flat minima to prevent overfitting during few-shot adaptation on target domains.

## Project Structure

```text
.
|-- instruction/
|   |-- data_prep.py
|   |-- data_statistics.py
|   `-- instruction_gen.py
|-- src/
|   |-- data_utils.py
|   |-- inference.py
|   |-- metrics_cal.py
|   |-- metrics_utils.py
|   |-- modeling_dynasrl.py
|   |-- train_log.py
|   |-- train_phase1.py
|   `-- train_phase2.py
|-- .gitignore
|-- README.md
|-- download_model.py
|-- dynasrl_overview.png
`-- requirements.txt
```

The repository only keeps public scripts and the overview figure. Datasets, checkpoints, logs, plots, and temporary research files are intentionally excluded from version control.

### Expected Local Data Layout

After local data preparation, place processed files under `data/input/` with one subdirectory per dataset. The current codebase supports:

- `cpb1`
- `conll2009_cn`
- `conll2009_en`
- `conll2012_cn`
- `conll2012_en`
- `ace2005`
- `fire`
- `phee`
- `fabner`
- `framenet17`

`instruction/data_prep.py` currently includes raw-data preprocessing for `cpb1`, `conll2009`, `fire`, `phee`, `fabner`, and `ace2005`. `conll2012_cn`, `conll2012_en`, and `framenet17` can be used once their unified JSON files and schema files are prepared under `data/input/<dataset>/`.

## Setup

The public scripts were aligned to the verified AutoDL environment with **Python 3.12.3** and **PyTorch 2.9.0**.

```bash
conda create -n dynasrl python=3.12.3 -y
conda activate dynasrl
pip install -r requirements.txt
```

## Usage

### 1. Download Base Models

`download_model.py` no longer stores any credential in code. It reads `HF_TOKEN` from the environment if you need authenticated downloads.

```bash
export HF_TOKEN=your_hf_token
python download_model.py \
  --models Qwen/Qwen3-8B \
  --target_dir /root/autodl-tmp/models
```

### 2. Prepare Unified Data Files

Convert raw datasets into the unified JSON format used by DynaSRL:

```bash
python instruction/data_prep.py \
  --datasets cpb1,conll2009,fire,phee,fabner,ace2005
```

Then build schema-aware instruction data:

```bash
python instruction/instruction_gen.py \
  --tasks cpb1,conll2009_cn,conll2009_en,conll2012_cn,conll2012_en,ace2005,fire,phee,fabner,framenet17
```

Dataset names passed to `--tasks` are expanded automatically to their train/dev/test splits when matching configs are available.

### 3. Phase 1 Training

Train the source-domain instruction-following model on `cpb1`:

```bash
python src/train_phase1.py \
  --models Qwen/Qwen3-8B \
  --dataset cpb1 \
  --data_dir ./data/input \
  --model_root /root/autodl-tmp/models \
  --learning_rate 2e-5 \
  --num_epochs 2 \
  --batch_size 6 \
  --grad_accum 3
```

### 4. Phase 2 Adaptation

Adapt the Phase 1 checkpoint to a target dataset such as `fire`:

```bash
python src/train_phase2.py \
  --dataset fire \
  --base_model_path /root/autodl-tmp/models/Qwen3-8B \
  --phase1_ckpt_path /root/autodl-tmp/models/Qwen3-8B-cpb1/checkpoint-1602 \
  --train_data_path ./data/input/fire/fire_train_ins.jsonl \
  --dev_data_path ./data/input/fire/fire_dev_ins.jsonl \
  --schema_path ./data/input/fire/fire_schema.json \
  --output_dir /root/autodl-tmp/models/Qwen3-8B-fire \
  --learning_rate 2e-5 \
  --num_epochs 3 \
  --glad_rho 0.05 \
  --glad_alpha 0.5
```

For internal large-scale ablation runs, `src/train_phase2.py` also keeps the optional `--run_sequential` entry point.

### 5. Inference and Evaluation

Run batch inference for a specific adapted model:

```bash
python src/inference.py \
  --model_root /root/autodl-tmp/models \
  --model Qwen3-8B-fire \
  --dataset fire \
  --max_new_tokens 512
```

Evaluate any prediction file against a gold JSON file:

```bash
python src/metrics_cal.py \
  --pred_path ./data/output/fire/fire_Qwen3-8B_pred.json \
  --gold_path ./data/input/fire/fire_test.json
```
