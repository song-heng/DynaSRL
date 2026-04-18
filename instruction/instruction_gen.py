import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =========================================================================
# Global Configuration (for easy modification)
# =========================================================================
# Set this to a dataset name (e.g., "phee") or a comma-separated list of tasks.
# If empty, the script will require command-line arguments.
CURRENT_TASK = "conll2009_cn"  # Options: cpb1, conll2009_cn, phee, etc.

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

        meta_info = sample.get("meta_info", {}) if isinstance(sample.get("meta_info"), dict) else {}
        lexical_unit = meta_info.get("lexical_unit") or sample.get("lexical_unit")
        frame = meta_info.get("frame") or sample.get("frame") or sample.get("predicate_sense")

        if lexical_unit and frame:
            key = (lexical_unit, frame)
            if key in self.multi_schema_map:
                return self.multi_schema_map[key]

        if frame and frame in self.frame_index and len(self.frame_index[frame]) == 1:
            return self.frame_index[frame][0]

        if lexical_unit and lexical_unit in self.lexical_unit_index and len(self.lexical_unit_index[lexical_unit]) == 1:
            return self.lexical_unit_index[lexical_unit][0]

        # Fallback: keep target mode usable even when schema key is missing.
        return {"PRED": list(PRED_DEFAULT_DEF)}


class DynaSRLInstructionGenerator:
    def __init__(self, schema_path, src_path, predicate_keys, shuffle_schema=True):
        self.schema_path = schema_path
        self.src_path = src_path
        self.predicate_keys = predicate_keys
        self.shuffle_schema = shuffle_schema
        self.schema_repo = SchemaRepository(schema_path)

    @staticmethod
    def _read_source_data(path: str):
        if path.endswith(".jsonl"):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def linearize_schema(self, schema_map: Dict[str, List[str]]):
        schema_items = list(schema_map.items())
        if self.shuffle_schema:
            random.shuffle(schema_items)

        schema_text_parts = []
        for role, def_list in schema_items:
            effective_defs = def_list if def_list else [""]
            selected_def = random.choice(effective_defs)
            schema_text_parts.append(f"[{role}]: {selected_def}")

        return "\n".join(schema_text_parts), [role for role, _ in schema_items]

    @staticmethod
    def linearize_output(roles_data):
        return " ".join([f"({role_label}, {span})" for role_label, span in roles_data])

    def extract_predicate(self, roles_data):
        for p_key in self.predicate_keys:
            for role_label, span in roles_data:
                if role_label == p_key:
                    return span, role_label
        return None, None

    @staticmethod
    def get_prompt_templates(mode, target_pred=None):
        if mode == "target_based":
            identity = (
                "You are DynaSRL, an advanced reasoning agent specialized in Target-Based Information Extraction. "
                "Your task is to analyze the sentence strictly focusing on a **Specific Target** (e.g., a predicate or trigger word)."
            )
            task_desc = (
                f"Task: Identify semantic arguments specifically associated with the Target: \"{target_pred}\".\n"
                "Execution Steps:\n"
                "1. Locate the Target in the input sentence.\n"
                "2. Extract text spans that serve as arguments defined in the Schema for this Target only.\n"
                "3. Ignore entities or relations that are not directly relevant to this Target.\n"
                "Output Format: (Role Label, Text Span)."
            )
            demo = (
                "Here is a demonstration:\n"
                "Sample Input: Yesterday Zhang San bought a book in Shanghai.\n"
                "Target: \"bought\"\n"
                "Sample Output: (ARGM-TMP, Yesterday) (ARG0, Zhang San) (PRED, bought) (ARG1, a book) (ARGM-LOC, in Shanghai)"
            )
            return identity, task_desc, demo

        identity = (
            "You are DynaSRL, an advanced reasoning agent specialized in Global Information Extraction. "
            "Your task is to scan the **Entire Sentence** and extract all relevant entities or roles defined in the Schema."
        )
        task_desc = (
            "Task: Analyze the full Input Sentence globally.\n"
            "Execution Steps:\n"
            "1. Read the Schema Definitions carefully.\n"
            "2. Scan the entire text to find all spans that match any of the Schema definitions.\n"
            "3. Extract all valid mentions without omitting any parts of the sentence.\n"
            "Output Format: (Role Label, Text Span)."
        )
        demo = (
            "Here is a demonstration:\n"
            "Sample Input: Steve Jobs is the founder of Apple.\n"
            "Sample Output: (Person, Steve Jobs) (Company, Apple) (Title, founder)"
        )
        return identity, task_desc, demo

    def generate_instruction(self, sample):
        sentence = sample["sentence"]
        roles_data = sample["roles"]
        schema_map = self.schema_repo.get_schema_map(sample)

        target_span, _ = self.extract_predicate(roles_data)
        mode = "target_based" if target_span else "global"
        focus_instruction = f'Target: "{target_span}"' if target_span else ""

        identity_prompt, task_description, demo_text = self.get_prompt_templates(mode, target_span)
        schema_text, schema_keys = self.linearize_schema(schema_map)

        input_block = f"Input Sentence: {sentence}"
        if mode == "target_based":
            input_block += f"\n{focus_instruction}"

        full_input = (
            f"{identity_prompt}\n\n"
            f"{task_description}\n\n"
            f"{demo_text}\n\n"
            f"Schema Definitions:\n{schema_text}\n\n"
            f"{input_block}\n\n"
            f"Response:"
        )

        return {
            "instruction": full_input,
            "output": self.linearize_output(roles_data),
            "aux_schema_keys": schema_keys,
            "aux_meta_info": sample.get("meta_info", {}),
        }

    def process_and_save(self, output_file):
        raw_data = self._read_source_data(self.src_path)
        processed_data = [self.generate_instruction(sample) for sample in raw_data]

        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Generated: {len(processed_data)} -> {output_file}")


DEFAULT_DATASET_CONFIGS = {
    # --- Framenet 1.7 ---
    # this is a multi-schema dataset
    "framenet17_multischema_train": {
        "schema": "data/input/framenet17/framenet17_schema.json",
        "src": "data/input/framenet17/framenet17_train.json",
        "out": "data/input/framenet17/framenet17_train_ins.jsonl",
    },
    "framenet17_multischema_dev": {
        "schema": "data/input/framenet17/framenet17_schema.json",
        "src": "data/input/framenet17/framenet17_dev.json",
        "out": "data/input/framenet17/framenet17_dev_ins.jsonl",
    },
    "framenet17_multischema_test": {
        "schema": "data/input/framenet17/framenet17_schema.json",
        "src": "data/input/framenet17/framenet17_test.json",
        "out": "data/input/framenet17/framenet17_test_ins.jsonl",
    },
    # --- CPB1 ---
    "cpb1_train": {
        "schema": "data/input/cpb1/cpb1_schema.json",
        "src": "data/input/cpb1/cpb1_train.json",
        "out": "data/input/cpb1/cpb1_train_ins.jsonl",
    },
    "cpb1_dev": {
        "schema": "data/input/cpb1/cpb1_schema.json",
        "src": "data/input/cpb1/cpb1_dev.json",
        "out": "data/input/cpb1/cpb1_dev_ins.jsonl",
    },
    "cpb1_test": {
        "schema": "data/input/cpb1/cpb1_schema.json",
        "src": "data/input/cpb1/cpb1_test.json",
        "out": "data/input/cpb1/cpb1_test_ins.jsonl",
    },
    # --- ConLL2009 CN ---
    "conll2009_cn_train": {
        "schema": "data/input/conll2009_cn/conll2009_cn_schema.json",
        "src": "data/input/conll2009_cn/conll2009_cn_train.json",
        "out": "data/input/conll2009_cn/conll2009_cn_train_ins.jsonl",
    },
    "conll2009_cn_dev": {
        "schema": "data/input/conll2009_cn/conll2009_cn_schema.json",
        "src": "data/input/conll2009_cn/conll2009_cn_dev.json",
        "out": "data/input/conll2009_cn/conll2009_cn_dev_ins.jsonl",
    },
    "conll2009_cn_test": {
        "schema": "data/input/conll2009_cn/conll2009_cn_schema.json",
        "src": "data/input/conll2009_cn/conll2009_cn_test.json",
        "out": "data/input/conll2009_cn/conll2009_cn_test_ins.jsonl",
    },
    # --- ConLL2012 CN ---
    "conll2012_cn_train": {
        "schema": "data/input/conll2012_cn/conll2012_cn_schema.json",
        "src": "data/input/conll2012_cn/conll2012_cn_train.json",
        "out": "data/input/conll2012_cn/conll2012_cn_train_ins.jsonl",
    },
    "conll2012_cn_dev": {
        "schema": "data/input/conll2012_cn/conll2012_cn_schema.json",
        "src": "data/input/conll2012_cn/conll2012_cn_dev.json",
        "out": "data/input/conll2012_cn/conll2012_cn_dev_ins.jsonl",
    },
    "conll2012_cn_test": {
        "schema": "data/input/conll2012_cn/conll2012_cn_schema.json",
        "src": "data/input/conll2012_cn/conll2012_cn_test.json",
        "out": "data/input/conll2012_cn/conll2012_cn_test_ins.jsonl",
    },
    # --- ConLL2009 EN ---
    "conll2009_en_train": {
        "schema": "data/input/conll2009_en/conll2009_en_schema.json",
        "src": "data/input/conll2009_en/conll2009_en_train.json",
        "out": "data/input/conll2009_en/conll2009_en_train_ins.jsonl",
    },
    "conll2009_en_dev": {
        "schema": "data/input/conll2009_en/conll2009_en_schema.json",
        "src": "data/input/conll2009_en/conll2009_en_dev.json",
        "out": "data/input/conll2009_en/conll2009_en_dev_ins.jsonl",
    },
    "conll2009_en_brown_test": {
        "schema": "data/input/conll2009_en/conll2009_en_schema.json",
        "src": "data/input/conll2009_en/conll2009_en_brown_test.json",
        "out": "data/input/conll2009_en/conll2009_en_brown_test_ins.jsonl",
    },
    "conll2009_en_wsj_test": {
        "schema": "data/input/conll2009_en/conll2009_en_schema.json",
        "src": "data/input/conll2009_en/conll2009_en_wsj_test.json",
        "out": "data/input/conll2009_en/conll2009_en_wsj_test_ins.jsonl",
    },
    # --- ConLL2012 EN ---
    "conll2012_en_train": {
        "schema": "data/input/conll2012_en/conll2012_en_schema.json",
        "src": "data/input/conll2012_en/conll2012_en_train.json",
        "out": "data/input/conll2012_en/conll2012_en_train_ins.jsonl",
    },
    "conll2012_en_dev": {
        "schema": "data/input/conll2012_en/conll2012_en_schema.json",
        "src": "data/input/conll2012_en/conll2012_en_dev.json",
        "out": "data/input/conll2012_en/conll2012_en_dev_ins.jsonl",
    },
    "conll2012_en_test": {
        "schema": "data/input/conll2012_en/conll2012_en_schema.json",
        "src": "data/input/conll2012_en/conll2012_en_test.json",
        "out": "data/input/conll2012_en/conll2012_en_test_ins.jsonl",
    },
    # --- PHEE ---
    "phee_train": {
        "schema": "data/input/phee/phee_schema.json",
        "src": "data/input/phee/phee_train.json",
        "out": "data/input/phee/phee_train_ins.jsonl",
    },
    "phee_dev": {
        "schema": "data/input/phee/phee_schema.json",
        "src": "data/input/phee/phee_dev.json",
        "out": "data/input/phee/phee_dev_ins.jsonl",
    },
    "phee_test": {
        "schema": "data/input/phee/phee_schema.json",
        "src": "data/input/phee/phee_test.json",
        "out": "data/input/phee/phee_test_ins.jsonl",
    },
    # --- FIRE ---
    "fire_train": {
        "schema": "data/input/fire/fire_schema.json",
        "src": "data/input/fire/fire_train.json",
        "out": "data/input/fire/fire_train_ins.jsonl",
    },
    "fire_dev": {
        "schema": "data/input/fire/fire_schema.json",
        "src": "data/input/fire/fire_dev.json",
        "out": "data/input/fire/fire_dev_ins.jsonl",
    },
    "fire_test": {
        "schema": "data/input/fire/fire_schema.json",
        "src": "data/input/fire/fire_test.json",
        "out": "data/input/fire/fire_test_ins.jsonl",
    },
    # --- FabNER ---
    "fabner_train": {
        "schema": "data/input/fabner/fabner_schema.json",
        "src": "data/input/fabner/fabner_train.json",
        "out": "data/input/fabner/fabner_train_ins.jsonl",
    },
    "fabner_dev": {
        "schema": "data/input/fabner/fabner_schema.json",
        "src": "data/input/fabner/fabner_dev.json",
        "out": "data/input/fabner/fabner_dev_ins.jsonl",
    },
    "fabner_test": {
        "schema": "data/input/fabner/fabner_schema.json",
        "src": "data/input/fabner/fabner_test.json",
        "out": "data/input/fabner/fabner_test_ins.jsonl",
    },
    # --- ACE2005 ---
    "ace2005_train": {
        "schema": "data/input/ace2005/ace2005_schema.json",
        "src": "data/input/ace2005/ace2005_train.json",
        "out": "data/input/ace2005/ace2005_train_ins.jsonl",
    },
    "ace2005_dev": {
        "schema": "data/input/ace2005/ace2005_schema.json",
        "src": "data/input/ace2005/ace2005_dev.json",
        "out": "data/input/ace2005/ace2005_dev_ins.jsonl",
    },
    "ace2005_test": {
        "schema": "data/input/ace2005/ace2005_schema.json",
        "src": "data/input/ace2005/ace2005_test.json",
        "out": "data/input/ace2005/ace2005_test_ins.jsonl",
    },
}


def load_dataset_configs(config_path: Optional[str]):
    if not config_path:
        return DEFAULT_DATASET_CONFIGS
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(project_root: Path, p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    # Check if relative path already handles direct path or needs root prefix
    full_path = project_root.parent / path
    if full_path.exists():
        return str(full_path.resolve())
    return str((project_root / path).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional dataset config JSON path")
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated task names",
    )
    parser.add_argument("--predicate_keys", type=str, default="PRED,rel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle_schema", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    # Script directory
    script_dir = Path(__file__).resolve().parent
    # project_root should be the parent directory of 'instruction/'
    project_root = script_dir.parent
    dataset_configs = load_dataset_configs(args.config)

    # Priorities: Command line > Global Variable
    task_input = args.tasks if args.tasks else CURRENT_TASK
    
    if not task_input:
        print("Usage error: Please specify tasks via --tasks or modify CURRENT_TASK in the script.")
        print(f"Available specific tasks: {list(dataset_configs.keys())}")
        return

    # Handle dataset-level shortcuts (e.g., "phee" -> ["phee_train", "phee_dev", "phee_test"])
    raw_tasks = [t.strip() for t in task_input.split(",") if t.strip()]
    tasks_to_run = []
    
    for t in raw_tasks:
        if t in dataset_configs:
            tasks_to_run.append(t)
        else:
            # Check if it's a prefix (dataset name)
            matching = [k for k in dataset_configs.keys() if (k == t or k.startswith(t + "_"))]
            if matching:
                tasks_to_run.extend(matching)
            else:
                print(f"!!! Error: Unknown task or dataset '{t}'")
                print(f"Available tasks: {list(dataset_configs.keys())}")
                return

    predicate_keys = [k.strip() for k in args.predicate_keys.split(",") if k.strip()]

    print(f"Current Target Keys Config: {predicate_keys}")
    print(f"Scheduled Tasks: {tasks_to_run}\n")

    for task_name in tasks_to_run:
        config = dataset_configs.get(task_name)
        if not config:
            print(f"Skipped unknown task: {task_name}")
            continue

        # Using script_dir as baseline for resolve_path to stay consistent with original logic
        schema_path = str((project_root / config["schema"]).resolve())
        src_path = str((project_root / config["src"]).resolve())
        out_path = str((project_root / config["out"]).resolve())

        print(f">>> Start Processing: {task_name}")
        if not os.path.exists(schema_path):
            print(f"!!! Skipped: Schema file not found {schema_path}")
            continue
        if not os.path.exists(src_path):
            print(f"!!! Skipped: Source file not found {src_path}")
            continue

        generator = DynaSRLInstructionGenerator(
            schema_path=schema_path,
            src_path=src_path,
            predicate_keys=predicate_keys,
            shuffle_schema=not args.no_shuffle_schema,
        )
        generator.process_and_save(output_file=out_path)
        print("-" * 50)

    print("\nAll selected tasks processed successfully.")


if __name__ == "__main__":
    main()
