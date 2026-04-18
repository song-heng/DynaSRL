import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random


from typing import Dict, List, Optional, Tuple

PRED_DEFAULT_DEF = [
    "surface string of the verb",
    "The predicate or main lexical unit triggering the semantic relations",
    "谓词 / 核心动词",
    "动词的表层形式（语义关系的触发词）",
]


class SchemaRepository:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema_data = json.load(f)

        self.schema_type = self._detect_schema_type(self.schema_data)
        self.flat_schema_map: Dict[str, List[str]] = {}
        self.multi_schema_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        self.frame_index: Dict[str, List[Dict[str, List[str]]]] = {}
        self.lexical_unit_index: Dict[str, List[Dict[str, List[str]]]] = {}

        if self.schema_type == "flat":
            self.flat_schema_map = self._build_flat_schema_map(self.schema_data)
        else:
            self.multi_schema_map = self._build_multi_schema_map(self.schema_data)

    @staticmethod
    def _detect_schema_type(schema_data):
        if isinstance(schema_data, list) and schema_data:
            first_item = schema_data[0]
            if isinstance(first_item, dict) and "role" in first_item:
                return "flat"
            if isinstance(first_item, dict) and "meta_info" in first_item and "schema" in first_item:
                return "multi"
        raise ValueError("Unsupported schema format")

    @staticmethod
    def _normalize_definitions(defs):
        if isinstance(defs, list):
            return [str(x) for x in defs if str(x).strip()]
        if defs is None:
            return []
        text = str(defs).strip()
        return [text] if text else []

    def _build_flat_schema_map(self, schema_data):
        schema_map = {}
        for item in schema_data:
            role = item["role"]
            schema_map[role] = self._normalize_definitions(item.get("def", []))
        return schema_map

    def _build_multi_schema_map(self, schema_data):
        schema_map = {}
        for entry in schema_data:
            meta_info = entry.get("meta_info", {})
            lexical_unit = meta_info.get("lexical_unit")
            frame = meta_info.get("frame")
            if not lexical_unit or not frame:
                continue

            frame_schema_map = {}
            for role_item in entry.get("schema", []):
                role = role_item.get("role")
                if not role:
                    continue
                frame_schema_map[role] = self._normalize_definitions(role_item.get("def", []))

            key = (lexical_unit, frame)
            schema_map[key] = frame_schema_map
            self.frame_index.setdefault(frame, []).append(frame_schema_map)
            self.lexical_unit_index.setdefault(lexical_unit, []).append(frame_schema_map)

        return schema_map

    def get_schema_map(self, sample: Dict) -> Dict[str, List[str]]:
        if self.schema_type == "flat":
            return self.flat_schema_map

        meta_info = sample.get("aux_meta_info", {})
        if not meta_info and "meta_info" in sample:
            meta_info = sample["meta_info"]

        lexical_unit = meta_info.get("lexical_unit")
        frame = meta_info.get("frame")

        if lexical_unit and frame:
            key = (lexical_unit, frame)
            if key in self.multi_schema_map:
                return self.multi_schema_map[key]

        if frame and frame in self.frame_index and len(self.frame_index[frame]) == 1:
            return self.frame_index[frame][0]

        if lexical_unit and lexical_unit in self.lexical_unit_index and len(self.lexical_unit_index[lexical_unit]) == 1:
            return self.lexical_unit_index[lexical_unit][0]

        return {"PRED": list(PRED_DEFAULT_DEF)}


class DynaSRLDataset(Dataset):
    '''
    Dataset class for DynaSRL, handling instruction loading and tokenization.
    :param data_path: Path to the JSONL data file
    :param schema_path: Path to the schema definition JSON file
    :param tokenizer: Tokenizer instance for text encoding
    :param max_length: Maximum sequence length for truncation
    :param max_samples: Optional limit on the number of samples to load
    '''

    def __init__(self, data_path, schema_path, tokenizer, max_length=2048, max_samples=None, use_projector=True):
        self.data = []
        self.use_projector = use_projector

        # 1. Read all raw lines
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]

        # [New] Sample truncation logic
        if max_samples is not None and max_samples > 0:
            if len(lines) > max_samples:
                print(f"[Dataset] Truncating data from {len(lines)} to {max_samples} samples.")
                random.seed(42)  # Fixed seed for reproducibility
                random.shuffle(lines)
                lines = lines[:max_samples]

        # 2. Parse JSON
        for line in lines:
            self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

        # [Modification] Use SchemaRepository to handle both flat and multi-schema formats
        self.schema_repo = SchemaRepository(schema_path)

    def __len__(self):
        '''
        Return the total number of samples.
        :return: Integer count of samples
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Get a single sample from the dataset.
        :param idx: Index of the sample
        :return: Dictionary containing input_ids, attention_mask, labels, and schema_def_ids
        '''
        item = self.data[idx]
        instruction = item['instruction']
        output = item['output']
        keys = item['aux_schema_keys']

        # 1. Build full text
        full_text = instruction + output + self.tokenizer.eos_token

        # 2. Tokenizer encoding
        # [Modification] Removed padding="max_length", only keep truncation
        # Returns variable length input_ids (e.g. 500, 800, 1200...)
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)

        # 3. Build Labels
        labels = input_ids.clone()

        # Mask instruction part
        ins_enc = self.tokenizer(instruction, add_special_tokens=False)
        ins_len = len(ins_enc.input_ids)
        if ins_len < len(input_ids):
            labels[:ins_len] = -100

        # Note: No need to mask padding here because there is no padding yet, collator will handle it

        # 4. Schema Processing
        # [Modification] Dynamically retrieve schema map based on sample metadata
        schema_def_tensor = None
        if self.use_projector:
            schema_map = self.schema_repo.get_schema_map(item)
            schema_def_ids = []
            for key in keys:
                defs = schema_map.get(key, ["Unknown"])
                selected_def = random.choice(defs)
                # Schema definition is short, fixed padding to max_length=64 is fine for stacking
                def_enc = self.tokenizer(
                    selected_def,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True
                )
                schema_def_ids.append(def_enc.input_ids.squeeze(0))

            if len(schema_def_ids) > 0:
                schema_def_tensor = torch.stack(schema_def_ids)
            else:
                schema_def_tensor = torch.zeros((1, 64), dtype=torch.long)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if schema_def_tensor is not None:
            res["schema_def_ids"] = schema_def_tensor
        return res

class DynaSRLCollator:
    '''
    Data collator for dynamic padding of batches.
    '''

    def __init__(self, tokenizer, use_projector=True):
        '''
        Initialize the collator with a tokenizer.
        :param tokenizer: Tokenizer instance
        '''
        self.tokenizer = tokenizer
        self.use_projector = use_projector
        # Ensure pad_token exists, usually Qwen/Llama use eos_token as pad
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        '''
        Process the batch by padding sequences to the longest in the batch.
        :param batch: List of samples from dataset
        :return: Dictionary containing padded input_ids, attention_mask, labels, and schema_input_ids_list
        '''
        # Extract input_ids, labels
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]

        # 1. Dynamic Padding (batch_first=True)
        # Automatically pad to the longest sequence in this batch, not 2048
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Labels padding with -100 (ignored in loss calculation)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        # 2. Regenerate Attention Mask
        # 1 where tokens exist, 0 where padded
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if self.use_projector:
            res["schema_input_ids_list"] = [item['schema_def_ids'] for item in batch]
        return res
