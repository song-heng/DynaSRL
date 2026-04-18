import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import torch
import traceback
import argparse
import re
import gc
import datetime
from typing import Any, Dict, List
from safetensors.torch import load_file
import regex

# ==============================================================================
# 1. Path Environment Configuration
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
print(f"[Init] Added path to system: {PROJECT_ROOT}")

try:
    from transformers import AutoTokenizer
    from modeling_dynasrl import DynaSRLModel
    print("[Init] Successfully imported DynaSRLModel from src.")
except ImportError as e:
    print(f"\n[Fatal Error] Cannot import module from src. Please check if src folder exists.\nError info: {e}")
    sys.exit(1)


# ==============================================================================
# 2. Define Inference Wrapper Class (终极完美对齐版)
# ==============================================================================
class DynaSRLInference(DynaSRLModel):
    # [核心修改 1] 接收外部传入的 attention_mask，彻底放弃容易出错的 pad_token_id 猜想
    def generate(
        self,
        input_ids,
        attention_mask,
        schema_input_ids,
        max_new_tokens=512,
        min_new_tokens=0,
        use_base_mode=False,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        # 回退 pad_token 以供生成停止判定使用
        safe_pad_id = self.llm.config.pad_token_id if self.llm.config.pad_token_id is not None else self.llm.config.eos_token_id
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "pad_token_id": safe_pad_id,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "top_k": top_k if do_sample else None,
        }

        if use_base_mode or self.projector is None:
            # Base 模式直接透传
            output_ids = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
            return output_ids[:, input_ids.shape[1]:]
            
        else:
            with torch.no_grad():
                embed_layer = self.llm.get_input_embeddings()
                s_embeds = embed_layer(schema_input_ids)
                s_role_repr = s_embeds.mean(dim=1) 
                global_schema_repr = s_role_repr.mean(dim=0, keepdim=True) 
                v_s = self.projector(global_schema_repr) 
                
                v_s = v_s.expand(input_ids.shape[0], -1, -1)
                latent_len = v_s.shape[1]

                inputs_embeds = embed_layer(input_ids)
                batch_size, seq_len, dim = inputs_embeds.shape
                
                combined_seq_len = latent_len + seq_len
                combined_embeds = torch.zeros((batch_size, combined_seq_len, dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                combined_mask = torch.zeros((batch_size, combined_seq_len), dtype=torch.long, device=self.llm.device)
                combined_input_ids = torch.ones((batch_size, combined_seq_len), dtype=torch.long, device=self.llm.device)
                
                for i in range(batch_size):
                    # [核心修改 2] 直接数 Tokenizer mask 里的 0，绝对不可能出错！
                    p = (attention_mask[i] == 0).sum().item()
                    
                    # A. 左侧 Padding 区域
                    if p > 0:
                        combined_embeds[i, :p] = inputs_embeds[i, :p]
                        combined_mask[i, :p] = 0
                        combined_input_ids[i, :p] = input_ids[i, :p]
                    
                    # B. Schema 向量区域 (v_s) 紧贴在真实文本前方
                    combined_embeds[i, p : p + latent_len] = v_s[i]
                    combined_mask[i, p : p + latent_len] = 1
                    combined_input_ids[i, p : p + latent_len] = 1 # 安全占位符
                    
                    # C. 真实文本区域
                    combined_embeds[i, p + latent_len :] = inputs_embeds[i, p:]
                    combined_mask[i, p + latent_len :] = 1
                    combined_input_ids[i, p + latent_len :] = input_ids[i, p:]

            output_ids = self.llm.generate(
                input_ids=combined_input_ids, 
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                **gen_kwargs
            )
            
            # 返回纯净的新 Token
            return output_ids[:, combined_seq_len:]


# ==============================================================================
# 3. Model Loading and Utility Functions
# ==============================================================================
def load_model_and_tokenizer(base_model_path, checkpoint_dir, use_base_mode=False, is_wo_mlp=False):
    print(f"Loading Tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing Model Structure...")

    if use_base_mode:
        print("\n[Mode] Loading Base Model...")
        model = DynaSRLInference(base_model_path, use_lora=False, use_projector=False)
    else:
        effective_use_projector = not is_wo_mlp
        mode_desc = "(LoRA + MLP)" if effective_use_projector else "(LoRA only)"
        print(f"\n[Mode] Loading DynaSRL {mode_desc}...")
        
        from peft import PeftModel
        model = DynaSRLInference(base_model_path, use_lora=False, use_projector=effective_use_projector)
        print(f"Loading LoRA weights from {checkpoint_dir}...")
        model.llm = PeftModel.from_pretrained(model.llm, checkpoint_dir)

        if effective_use_projector:
            mlp_path = os.path.join(checkpoint_dir, "mlp_projector.bin")
            if os.path.exists(mlp_path):
                print(f"Loading MLP Projector from {mlp_path}...")
                model.projector.load_state_dict(torch.load(mlp_path, map_location="cpu"))
            else:
                model.projector = None

        print("Weights Loaded Successfully.")

    model = model.cuda()
    model.eval()
    return tokenizer, model

SCHEMA_TENSOR_CACHE = {}

def prepare_schema_tensor(keys, schema_map, tokenizer):
    schema_def_ids = []
    for key in keys:
        if key in SCHEMA_TENSOR_CACHE:
            schema_def_ids.append(SCHEMA_TENSOR_CACHE[key])
            continue

        raw_def = schema_map.get(key, "Unknown")
        def_text = normalize_schema_def(raw_def)
        def_enc = tokenizer(def_text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
        tid = def_enc.input_ids.squeeze(0)
        SCHEMA_TENSOR_CACHE[key] = tid
        schema_def_ids.append(tid)
        
    if not schema_def_ids:
        return torch.zeros((1, 64), dtype=torch.long).cuda()
    return torch.stack(schema_def_ids).cuda()

def normalize_schema_def(raw_def):
    if isinstance(raw_def, list):
        parts = [str(x).strip() for x in raw_def if str(x).strip()]
        return " | ".join(parts) if parts else "Unknown"
    if isinstance(raw_def, dict):
        parts = []
        for value in raw_def.values():
            txt = str(value).strip()
            if txt:
                parts.append(txt)
        return " | ".join(parts) if parts else "Unknown"
    text = str(raw_def).strip()
    return text if text else "Unknown"

def build_schema_map(schema_obj):
    """
    Build role->definition map from multiple schema styles:
      1) flat:   [{"role": "...", "def": ...}, ...]
      2) nested: [{"meta_info": {...}, "schema": [{"role": "...", "def": ...}, ...]}, ...]
    """
    role_defs = {}

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return

        if not isinstance(node, dict):
            return

        role = str(node.get("role", "")).strip()
        if role:
            role_defs.setdefault(role, normalize_schema_def(node.get("def", "Unknown")))

        for nested_key in ("schema", "roles", "items"):
            nested = node.get(nested_key)
            if isinstance(nested, (list, dict)):
                visit(nested)

    visit(schema_obj)
    return role_defs

def load_json_or_jsonl_items(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        items = []
        bad_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                    else:
                        bad_lines += 1
                except Exception:
                    bad_lines += 1
                    if bad_lines <= 3:
                        print(f"    [Warn] Bad JSONL line at {path}:{line_no}")
        return items, bad_lines

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        dict_items = [x for x in data if isinstance(x, dict)]
        return dict_items, len(data) - len(dict_items)
    return [], 1

def resolve_test_data_path(project_root, dataset_name, task_name):
    base = os.path.join(project_root, "data", "input", dataset_name)
    candidates = [
        os.path.join(base, f"{task_name}_test_ins.jsonl"),
        os.path.join(base, f"{task_name}_test_ins.json"),
        os.path.join(base, f"{task_name}_test.jsonl"),
        os.path.join(base, f"{task_name}_test.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

def resolve_batch_schema_keys(batch_items, schema_map):
    keys = []
    seen = set()
    for item in batch_items:
        raw_keys = item.get("aux_schema_keys", [])
        if isinstance(raw_keys, list):
            for raw_key in raw_keys:
                key = str(raw_key).strip()
                if key and key not in seen:
                    seen.add(key)
                    keys.append(key)
    if keys:
        return keys
    return list(schema_map.keys())

def resolve_instruction_text(item):
    instruction = item.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction

    sentence = item.get("sentence")
    if isinstance(sentence, str) and sentence.strip():
        return f"Input Sentence: {sentence}\n\nResponse:"

    return ""

def is_cuda_oom_error(exc):
    msg = str(exc).lower()
    return (
        isinstance(exc, RuntimeError)
        and ("out of memory" in msg or "cuda error" in msg)
    )

def generate_decoded_batch(
    model,
    tokenizer,
    text_inputs,
    schema_tensor,
    max_new_tokens,
    min_new_tokens,
    use_base_mode,
    task_tag,
):
    try:
        tokenizer.padding_side = "left"
        encodings = tokenizer(text_inputs, return_tensors="pt", padding=True).to("cuda")
        output_ids = model.generate(
            encodings.input_ids,
            encodings.attention_mask,
            schema_tensor,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            use_base_mode=use_base_mode,
        )
        return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]
    except Exception as exc:
        if not is_cuda_oom_error(exc):
            raise
        print(f"\n    [Warn] CUDA OOM on batch ({task_tag}), fallback to single-sample mode.")
        torch.cuda.empty_cache()
        gc.collect()

        decoded = []
        for idx, text in enumerate(text_inputs):
            try:
                tokenizer.padding_side = "left"
                single_enc = tokenizer([text], return_tensors="pt", padding=True).to("cuda")
                single_out = model.generate(
                    single_enc.input_ids,
                    single_enc.attention_mask,
                    schema_tensor,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    use_base_mode=use_base_mode,
                )
                decoded.append(tokenizer.decode(single_out[0], skip_special_tokens=True).strip())
            except Exception as single_exc:
                if is_cuda_oom_error(single_exc):
                    print(f"    [Warn] Single-sample OOM at index {idx} ({task_tag}), emit empty output.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    decoded.append("")
                else:
                    raise
        return decoded

def extract_sentence_from_instruction(instruction):
    pattern = r"Input Sentence:\s*(.*?)\s*(?:Target:|Response:|Output:)"
    match = re.search(pattern, instruction, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "Input Sentence:" in instruction:
        return instruction.split("Input Sentence:")[-1].strip()
    return ""

def parse_model_response(response_str):
    roles = []
    pattern = r"\(([^,]+),\s*(?P<text>(?:[^()]+|\((?P>text)\))*)\)"
    matches = regex.findall(pattern, response_str)
    for match in matches:
        roles.append([match[0].strip(), match[1].strip()])
    return roles

def format_roles_as_text(roles):
    if not roles:
        return ""
    return " ".join([f"({role}, {span})" for role, span in roles])

def strip_think_block(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def build_base_chat_input(tokenizer, raw_instruction):
    """
    Build base-model input with thinking disabled and strict SRL tuple output constraints.
    """
    system_prompt = (
        "You are a semantic role labeling extractor.\n"
        "Return ONLY tuples in exact format: (Role, Span)\n"
        "Rules:\n"
        "1. Output only zero or more tuples separated by one space.\n"
        "2. No explanation, no reasoning, no markdown, no JSON, no extra text.\n"
        "3. Keep each Span exactly from the input sentence.\n"
        "4. If no role exists, return an empty string."
    )

    user_content = raw_instruction
    if "/no_think" not in user_content:
        user_content += "\n/no_think"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizer API fallback (no enable_thinking arg).
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return raw_instruction
    except Exception:
        return raw_instruction

# ==============================================================================
# 4. Constants and Model Name Parser
# ==============================================================================
BASE_MODELS = [
    "Llama-3-8B", "Llama-3.2-1B", "Llama-3.2-3B",
    "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
]

KNOWN_DATASETS = [
    "cpb1", "conll2009_cn", "conll2009_en", "conll2012_cn", "conll2012_en",
    "ace2005", "fabner", "fire", "framenet17", "phee",
]

# --base 模式下用于对比的基座模型 (仅此一个)
BASE_COMPARISON_MODEL = "Qwen3-8B"


def resolve_min_new_tokens(model_info, dataset_name, args):
    """
    Auto-fix for known early-EOS collapse:
    Qwen3-14B on conll2009_cn may emit empty strings when min_new_tokens=0.
    """
    if args.min_new_tokens >= 0:
        return args.min_new_tokens

    if (
        model_info.get("model_type") != "base"
        and model_info.get("base_model") == "Qwen3-14B"
        and dataset_name == "conll2009_cn"
    ):
        return 8
    return 0


def find_checkpoint(model_path):
    """在模型目录中查找最新的 checkpoint 子目录"""
    if not os.path.isdir(model_path):
        return model_path
    subdirs = [d for d in os.listdir(model_path)
               if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, d))]
    if not subdirs:
        return model_path
    try:
        subdirs.sort(key=lambda x: int(x.split('-')[-1]))
        return os.path.join(model_path, subdirs[-1])
    except Exception:
        return os.path.join(model_path, subdirs[0])


def parse_model_dir_name(model_dir_name):
    """
    精准解析模型目录名，返回结构化信息。

    支持的命名模式:
      Base:             {base}                                      e.g. Qwen3-8B
      Phase 1:          {base}-cpb1                                 e.g. Qwen3-8B-cpb1
      Phase 1 woMLP:    {base}-cpb1-woMLP                           e.g. Qwen3-8B-cpb1-woMLP
      Phase 2:          {base}-cpb1-{dataset}                       e.g. Qwen3-8B-cpb1-conll2009_cn
      Phase 2 woGLAD:   {base}-cpb1-{dataset}-woGLAD                e.g. Qwen3-8B-cpb1-conll2009_cn-woGLAD
      Phase 2 woMLP:    {base}-cpb1-woMLP-{dataset}-woMLP           e.g. Qwen3-8B-cpb1-woMLP-ace2005-woMLP
      Phase 2 woGLADwoMLP: {base}-cpb1-woMLP-{dataset}-woGLADwoMLP  e.g. Qwen3-8B-cpb1-woMLP-ace2005-woGLADwoMLP

    Returns:
        dict with keys: base_model, model_type, dataset, ablation_tag, is_wo_mlp, is_wo_glad
        None if the name cannot be parsed.
    """
    # --- 1. 检查是否为 Base Model ---
    if model_dir_name in BASE_MODELS:
        return {
            'base_model': model_dir_name,
            'model_type': 'base',
            'dataset': None,
            'ablation_tag': '',
            'is_wo_mlp': False,
            'is_wo_glad': False,
        }

    # --- 2. 匹配 Base Model 前缀 (按长度降序，避免 Llama-3 匹配到 Llama-3.2) ---
    base_model = None
    for bm in sorted(BASE_MODELS, key=len, reverse=True):
        if model_dir_name.startswith(bm + "-"):
            base_model = bm
            break

    if not base_model:
        return None

    remainder = model_dir_name[len(base_model) + 1:]  # 剥离 "{base}-"

    # --- 3. 所有微调模型必须以 cpb1 开头 ---
    if not remainder.startswith("cpb1"):
        return None

    remainder = remainder[4:]  # 剥离 "cpb1"

    # --- 4. Phase 1: {base}-cpb1 (remainder 为空) ---
    if not remainder:
        return {
            'base_model': base_model,
            'model_type': 'phase1',
            'dataset': None,
            'ablation_tag': '',
            'is_wo_mlp': False,
            'is_wo_glad': False,
        }

    remainder = remainder[1:]  # 剥离 "-"

    # --- 5. Phase 1 woMLP: {base}-cpb1-woMLP (remainder == "woMLP") ---
    if remainder == "woMLP":
        return {
            'base_model': base_model,
            'model_type': 'phase1',
            'dataset': None,
            'ablation_tag': '_woMLP',
            'is_wo_mlp': True,
            'is_wo_glad': False,
        }

    # --- 6. 检测 woMLP 前缀 (Phase 2 woMLP 系列) ---
    is_from_womlp_phase1 = False
    if remainder.startswith("woMLP-"):
        is_from_womlp_phase1 = True
        remainder = remainder[6:]  # 剥离 "woMLP-"

    # --- 7. 剥离尾部消融后缀 ---
    ablation_tag = ""
    is_wo_glad = False
    is_wo_mlp = is_from_womlp_phase1

    if remainder.endswith("-woGLADwoMLP"):
        ablation_tag = "_woGLADwoMLP"
        is_wo_glad = True
        is_wo_mlp = True
        remainder = remainder[:-len("-woGLADwoMLP")]
    elif remainder.endswith("-woGLAD"):
        ablation_tag = "_woGLAD"
        is_wo_glad = True
        remainder = remainder[:-len("-woGLAD")]
    elif remainder.endswith("-woMLP"):
        ablation_tag = "_woMLP"
        is_wo_mlp = True
        remainder = remainder[:-len("-woMLP")]

    # --- 8. 剩余部分即为数据集名称 ---
    dataset = remainder

    if dataset not in KNOWN_DATASETS:
        return None  # 未知数据集，跳过

    return {
        'base_model': base_model,
        'model_type': 'phase2',
        'dataset': dataset,
        'ablation_tag': ablation_tag,
        'is_wo_mlp': is_wo_mlp,
        'is_wo_glad': is_wo_glad,
    }


# ==============================================================================
# 5. Main Function
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="DynaSRL Batch Inference Pipeline")
    parser.add_argument("--base", action="store_true", help="Include base model (woDyna) inference")
    parser.add_argument("--model_root", type=str, default="/root/models", help="Root directory for models")
    parser.add_argument("--model", type=str, default=None, help="Run inference for a specific model folder only")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by a specific dataset")
    parser.add_argument("--include_phase1", action="store_true", help="Include Phase 1 models in batch mode")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation length.")
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=-1,
        help="If >=0, force min_new_tokens. If -1, auto-choose (Qwen3-14B+conll2009_cn -> 8 else 0).",
    )
    parser.add_argument(
        "--disable_empty_retry",
        action="store_true",
        help="Disable one-step retry (sampling) when parsed roles are empty.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Default inference batch size for non-framenet17 datasets.",
    )
    parser.add_argument(
        "--framenet_batch_size",
        type=int,
        default=1,
        help="Inference batch size for framenet17 non-base runs.",
    )
    args = parser.parse_args()

    # ---------- 路径与日志初始化 ----------
    project_root = PROJECT_ROOT

    log_dir = os.path.join(project_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inference_tasks.log")

    def log_status(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(message)

    log_status("\n" + "=" * 80)
    log_status("DynaSRL Batch Inference Pipeline Started")
    log_status(f"  model_root  = {args.model_root}")
    log_status(f"  --model     = {args.model or '(all)'}")
    log_status(f"  --dataset   = {args.dataset or '(auto)'}")
    log_status(f"  --base      = {args.base}")
    log_status(f"  --include_phase1 = {args.include_phase1}")
    log_status(f"  --max_new_tokens = {args.max_new_tokens}")
    log_status(f"  --min_new_tokens = {args.min_new_tokens}")
    log_status(f"  --disable_empty_retry = {args.disable_empty_retry}")
    log_status(f"  --batch_size = {args.batch_size}")
    log_status(f"  --framenet_batch_size = {args.framenet_batch_size}")
    log_status("=" * 80)

    if not os.path.exists(args.model_root):
        log_status(f"[Fatal] Model root directory not found: {args.model_root}")
        return

    # ---------- 扫描所有模型目录并解析 ----------
    all_model_dirs = sorted([
        d for d in os.listdir(args.model_root)
        if os.path.isdir(os.path.join(args.model_root, d))
    ])
    log_status(f"\n[Scan] Found {len(all_model_dirs)} directories in {args.model_root}")

    # ---------- 构建任务列表 ----------
    # 每个任务: { 'model_dir': str, 'info': dict, 'dataset': str }
    tasks_to_run = []

    def add_tasks_for_model(model_dir, model_info, dataset_override=None):
        """根据模型类型和参数为模型添加推理任务"""
        if model_info['model_type'] == 'base':
            # Base 模型: 在指定的或全部已知数据集上推理
            ds_list = [dataset_override] if dataset_override else KNOWN_DATASETS
            for ds in ds_list:
                tasks_to_run.append({
                    'model_dir': model_dir,
                    'info': model_info,
                    'dataset': ds,
                })
        elif model_info['model_type'] == 'phase1':
            # Phase 1 模型: 仅在 cpb1 上推理 (Phase 1 的训练数据集)
            ds = dataset_override if dataset_override else "cpb1"
            tasks_to_run.append({
                'model_dir': model_dir,
                'info': model_info,
                'dataset': ds,
            })
        else:
            # Phase 2 模型: 仅在其训练的目标数据集上推理
            ds = model_info['dataset']
            if dataset_override and dataset_override != ds:
                return  # --dataset 指定了不匹配的数据集，跳过
            tasks_to_run.append({
                'model_dir': model_dir,
                'info': model_info,
                'dataset': ds,
            })

    if args.model:
        # ---- 单模型模式 ----
        info = parse_model_dir_name(args.model)
        if info is None:
            log_status(f"[Fatal] Cannot parse model directory name: {args.model}")
            return
        add_tasks_for_model(args.model, info, dataset_override=args.dataset)
        log_status(f"[Manual] Single model mode: {args.model}")
    else:
        # ---- 批量模式 ----
        for model_dir in all_model_dirs:
            info = parse_model_dir_name(model_dir)
            if info is None:
                log_status(f"  [Skip] Unrecognized directory: {model_dir}")
                continue

            if info['model_type'] == 'base':
                if args.base and model_dir == BASE_COMPARISON_MODEL:
                    add_tasks_for_model(model_dir, info, dataset_override=args.dataset)
                elif not args.base:
                    log_status(f"  [Skip] Base model (use --base): {model_dir}")
                else:
                    log_status(f"  [Skip] Base model (only {BASE_COMPARISON_MODEL} for --base): {model_dir}")
                continue

            if info['model_type'] == 'phase1':
                if args.include_phase1:
                    add_tasks_for_model(model_dir, info, dataset_override=args.dataset)
                else:
                    log_status(f"  [Skip] Phase 1 model (use --include_phase1): {model_dir}")
                continue

            # Phase 2 模型
            add_tasks_for_model(model_dir, info, dataset_override=args.dataset)

    if not tasks_to_run:
        log_status("[Done] No tasks to run. Check arguments.")
        return

    # ---------- 按模型分组 (避免重复加载同一模型) ----------
    from collections import OrderedDict
    model_task_groups = OrderedDict()
    for task in tasks_to_run:
        key = task['model_dir']
        if key not in model_task_groups:
            model_task_groups[key] = []
        model_task_groups[key].append(task)

    # ---------- 打印执行计划 ----------
    log_status(f"\n{'='*60}")
    log_status(f"[Plan] {len(tasks_to_run)} tasks across {len(model_task_groups)} models")
    log_status(f"{'='*60}")
    for m_dir, m_tasks in model_task_groups.items():
        info = m_tasks[0]['info']
        ds_names = [t['dataset'] for t in m_tasks]
        tag = info['ablation_tag'] or "(standard)"
        log_status(f"  {m_dir}  [{info['model_type']}]  ablation={tag}")
        log_status(f"    -> datasets: {ds_names}")
    log_status("")

    # ---------- 执行推理 ----------
    completed, failed, skipped = 0, 0, 0

    for m_idx, (model_dir_name, model_task_list) in enumerate(model_task_groups.items()):
        model_info = model_task_list[0]['info']
        full_model_path = os.path.join(args.model_root, model_dir_name)

        try:
            base_model_path = os.path.join(args.model_root, model_info['base_model'])
            checkpoint_dir = find_checkpoint(full_model_path)

            is_model_base = (model_info['model_type'] == 'base')
            effective_base_mode = is_model_base  # 仅 base model 使用 base mode，--base 只控制是否包含

            log_status(f"\n{'='*60}")
            log_status(f"[{m_idx+1}/{len(model_task_groups)}] Loading Model: {model_dir_name}")
            log_status(f"    Type: {model_info['model_type']}  |  Base: {model_info['base_model']}")
            log_status(f"    Ablation: {model_info['ablation_tag'] or 'None'}  |  woMLP: {model_info['is_wo_mlp']}")
            log_status(f"    Checkpoint: {checkpoint_dir}")
            log_status(f"    effective_base_mode: {effective_base_mode}")

            tokenizer, model = load_model_and_tokenizer(
                base_model_path, checkpoint_dir,
                use_base_mode=effective_base_mode,
                is_wo_mlp=model_info['is_wo_mlp'],
            )

            for task_item in model_task_list:
                d_name = task_item['dataset']
                try:
                    # conll2009_en 拥有 wsj 和 brown 两个测试子集
                    sub_tasks = (
                        ["conll2009_en_wsj", "conll2009_en_brown"]
                        if d_name == "conll2009_en"
                        else [d_name]
                    )

                    for task in sub_tasks:
                        try:
                            schema_path = os.path.join(
                                project_root, f"data/input/{d_name}/{d_name}_schema.json"
                            )
                            test_data_path = resolve_test_data_path(project_root, d_name, task)

                            # === 生成有区分度的输出文件名 ===
                            if effective_base_mode:
                                output_name = f"{task}_woDyna_{model_info['base_model']}_pred.json"
                            elif model_info['model_type'] == 'phase1':
                                # Phase 1 模型加 _phase1 标记避免与 Phase 2 冲突
                                output_name = f"{task}_phase1{model_info['ablation_tag']}_{model_info['base_model']}_pred.json"
                            else:
                                output_name = f"{task}{model_info['ablation_tag']}_{model_info['base_model']}_pred.json"

                            output_dir = os.path.join(project_root, f"data/output/{d_name}")
                            output_file_path = os.path.join(output_dir, output_name)

                            # 检查数据文件是否存在
                            if not os.path.exists(test_data_path):
                                log_status(f"    [Skip] Test data not found: {test_data_path}")
                                skipped += 1
                                continue
                            if not os.path.exists(schema_path):
                                log_status(f"    [Skip] Schema not found: {schema_path}")
                                skipped += 1
                                continue

                            effective_min_new_tokens = resolve_min_new_tokens(model_info, d_name, args)
                            log_status(
                                f"    [Task] {task} -> {output_name} "
                                f"(min_new_tokens={effective_min_new_tokens})"
                            )

                            with open(schema_path, "r", encoding="utf-8") as f:
                                schema_raw = json.load(f)
                            schema_map = build_schema_map(schema_raw)
                            if not schema_map:
                                log_status(f"    [Skip] Parsed empty schema map: {schema_path}")
                                skipped += 1
                                continue

                            dataset_items, bad_line_count = load_json_or_jsonl_items(test_data_path)
                            if bad_line_count > 0:
                                log_status(
                                    f"    [Warn] Ignored {bad_line_count} malformed records in {test_data_path}"
                                )
                            if not dataset_items:
                                log_status(f"    [Skip] No usable items found in {test_data_path}")
                                skipped += 1
                                continue

                            all_predictions = []
                            primary_empty_count = 0
                            recovered_by_retry = 0
                            final_empty_count = 0
                            current_batch_size = (
                                args.framenet_batch_size
                                if (d_name == "framenet17" and not effective_base_mode)
                                else args.batch_size
                            )
                            SCHEMA_TENSOR_CACHE.clear()

                            for i in range(0, len(dataset_items), current_batch_size):
                                batch_items = dataset_items[i : i + current_batch_size]

                                keys = resolve_batch_schema_keys(batch_items, schema_map)
                                schema_tensor = prepare_schema_tensor(keys, schema_map, tokenizer)

                                text_inputs = []
                                for item in batch_items:
                                    raw_instruction = resolve_instruction_text(item)
                                    if effective_base_mode:
                                        text_input = build_base_chat_input(tokenizer, raw_instruction)
                                    else:
                                        text_input = raw_instruction
                                    text_inputs.append(text_input)

                                decoded_outputs = generate_decoded_batch(
                                    model=model,
                                    tokenizer=tokenizer,
                                    text_inputs=text_inputs,
                                    schema_tensor=schema_tensor,
                                    max_new_tokens=args.max_new_tokens,
                                    min_new_tokens=effective_min_new_tokens,
                                    use_base_mode=effective_base_mode,
                                    task_tag=f"{task}:{i}-{i + len(batch_items) - 1}",
                                )

                                processed_samples = min(i + current_batch_size, len(dataset_items))
                                print(
                                    f"      - Progress: {processed_samples}/{len(dataset_items)} samples processed...",
                                    end="\r",
                                )

                                for j, item in enumerate(batch_items):
                                    clean_response = decoded_outputs[j] if j < len(decoded_outputs) else ""
                                    if effective_base_mode:
                                        clean_response = strip_think_block(clean_response)
                                    parsed_roles = parse_model_response(clean_response)
                                    had_primary_empty = len(parsed_roles) == 0
                                    if had_primary_empty:
                                        primary_empty_count += 1

                                    # Retry once with sampling if non-base output is empty.
                                    if (
                                        had_primary_empty
                                        and (not effective_base_mode)
                                        and (not args.disable_empty_retry)
                                    ):
                                        retry_instruction = resolve_instruction_text(item)
                                        if retry_instruction:
                                            try:
                                                tokenizer.padding_side = "left"
                                                retry_encoding = tokenizer(
                                                    [retry_instruction], return_tensors="pt", padding=True
                                                ).to("cuda")
                                                retry_ids = model.generate(
                                                    retry_encoding.input_ids,
                                                    retry_encoding.attention_mask,
                                                    schema_tensor,
                                                    max_new_tokens=args.max_new_tokens,
                                                    min_new_tokens=effective_min_new_tokens,
                                                    use_base_mode=False,
                                                    do_sample=True,
                                                    temperature=0.3,
                                                    top_p=0.9,
                                                    top_k=50,
                                                )
                                                retry_response = tokenizer.decode(
                                                    retry_ids[0], skip_special_tokens=True
                                                ).strip()
                                                retry_roles = parse_model_response(retry_response)
                                                if retry_roles:
                                                    parsed_roles = retry_roles
                                                    clean_response = retry_response
                                                    recovered_by_retry += 1
                                            except Exception as retry_exc:
                                                if is_cuda_oom_error(retry_exc):
                                                    torch.cuda.empty_cache()
                                                    gc.collect()
                                                    log_status(
                                                        f"    [Warn] Retry OOM at sample {i + j} in task {task}, keep empty."
                                                    )
                                                else:
                                                    raise

                                    if len(parsed_roles) == 0:
                                        final_empty_count += 1

                                    instruction_text = resolve_instruction_text(item)
                                    extracted_sentence = extract_sentence_from_instruction(instruction_text)
                                    if not extracted_sentence and isinstance(item.get("sentence"), str):
                                        extracted_sentence = item.get("sentence", "")

                                    if j == 0:
                                        print(
                                            f"\n      [Sample {processed_samples - len(batch_items) + 1}] "
                                            f"Sentence: {extracted_sentence[:30]}..."
                                        )
                                        print(
                                            f"      [Sample {processed_samples - len(batch_items) + 1}] "
                                            f"Response: {format_roles_as_text(parsed_roles) if effective_base_mode else clean_response}"
                                        )

                                    all_predictions.append({
                                        "sentence": extracted_sentence,
                                        "roles": parsed_roles,
                                    })

                            os.makedirs(output_dir, exist_ok=True)
                            with open(output_file_path, 'w', encoding='utf-8') as f:
                                json.dump(all_predictions, f, indent=2, ensure_ascii=False)
                            log_status(
                                f"    [OK] {task} completed -> {output_file_path} ({len(all_predictions)} samples)"
                            )
                            log_status(
                                f"    [Stats] primary_empty={primary_empty_count}, "
                                f"recovered_by_retry={recovered_by_retry}, final_empty={final_empty_count}"
                            )
                            completed += 1

                        except Exception as task_e:
                            log_status(f"    [FAIL] Task '{task}' error: {task_e}")
                            traceback.print_exc()
                            failed += 1

                except Exception as dataset_e:
                    log_status(f"    [FAIL] Dataset '{d_name}' error: {dataset_e}")
                    traceback.print_exc()
                    failed += 1

            # 释放 GPU 显存
            try:
                del model, tokenizer
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()
            log_status(f"    [Cleanup] GPU memory released for {model_dir_name}")

        except Exception as model_e:
            log_status(f"[FAIL] Model '{model_dir_name}' loading error: {model_e}")
            traceback.print_exc()
            failed += len(model_task_list)
            torch.cuda.empty_cache()
            gc.collect()

    # ---------- 最终统计 ----------
    log_status("\n" + "=" * 80)
    log_status("Batch Inference Pipeline Finished")
    log_status(f"  Completed : {completed}")
    log_status(f"  Failed    : {failed}")
    log_status(f"  Skipped   : {skipped}")
    log_status("=" * 80 + "\n")


if __name__ == "__main__":
    main()
