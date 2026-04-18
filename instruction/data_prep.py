import argparse
import os
import json


# ==========================================================
# 1. CPB1 Processing Logic
# ==========================================================
def prep_cpb1_file(input_path, output_path):
    '''
    Read a single CPB1 format file and store roles in [[label, value], ...] format.
    :param input_path: Path to the input file
    :param output_path: Path to the output file
    :return: None
    '''
    processed_data = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            raw_tokens = line.split()
            sentence_words, roles_list = [], []
            temp_words, temp_label = [], None

            for token in raw_tokens:
                parts = token.rsplit('/', 2)
                if len(parts) != 3: continue
                word, pos, label = parts
                sentence_words.append(word)

                if label == 'O':
                    if temp_label is not None:
                        roles_list.append([temp_label, "".join(temp_words)])
                        temp_words, temp_label = [], None
                    continue

                prefix, base_label = label.split('-', 1) if '-' in label else ('', label)
                if prefix == 'B':
                    if temp_label is not None:
                        roles_list.append([temp_label, "".join(temp_words)])
                    temp_label, temp_words = base_label, [word]
                elif prefix == 'I':
                    if temp_label == base_label:
                        temp_words.append(word)
                    else:
                        if temp_label is not None:
                            roles_list.append([temp_label, "".join(temp_words)])
                        temp_label, temp_words = base_label, [word]
                elif prefix == 'E':
                    if temp_label == base_label:
                        temp_words.append(word)
                        roles_list.append([temp_label, "".join(temp_words)])
                        temp_words, temp_label = [], None
                    else:
                        if temp_label is not None:
                            roles_list.append([temp_label, "".join(temp_words)])
                        roles_list.append([base_label, word])
                        temp_words, temp_label = [], None
                elif prefix in ['S', '']:
                    if temp_label is not None:
                        roles_list.append([temp_label, "".join(temp_words)])
                        temp_words, temp_label = [], None
                    roles_list.append([base_label, word])

            if temp_label is not None:
                roles_list.append([temp_label, "".join(temp_words)])
            processed_data.append({"sentence": "".join(sentence_words), "roles": roles_list})

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, ensure_ascii=False, indent=2)
        print(f"[CPB1] Saved successfully: {output_path} (Total {len(processed_data)} items)")
    except Exception as e:
        print(f"[CPB1] Processing Error {input_path}: {e}")


# ==========================================================
# 2. ConLL2009 Processing Logic
# ==========================================================
def prep_conll2009_dataset(project_root, output_root):
    '''
    Process the ConLL2009 dataset.
    :param project_root: Root directory of the project
    :param output_root: Output root directory
    :return: None
    '''

    def get_subtree_indices(head_id, children_map):
        '''
        Recursively get indices of the subtree.
        :param head_id: The head node ID
        :param children_map: Map of children nodes
        :return: List of indices in the subtree
        '''
        indices = [head_id]
        if head_id in children_map:
            for child in children_map[head_id]:
                indices.extend(get_subtree_indices(child, children_map))
        return indices

    drop_role_by_lang = {
        'cn': {
            'PRD', 'C-A2', 'C-ARGM-TMP', 'CRD', 'QTY', 'VOC', 'A5', 'ARGM', 'C-A3'
        },
        'en': {
            'A5', 'AM-PRD', 'C-A3', 'C-AM-ADV', 'C-AM-MNR', 'C-A4', 'C-AM-TMP',
            'C-AM-CAU', 'R-AM-PNC', 'AM', 'AM-REC', 'C-AM-EXT', 'C-AM-LOC',
            'C-AM-DIS', 'AM-PRT', 'AM-TM', 'C-AM-DIR', 'R-A3'
        }
    }

    def parse_conll_sentence(sentence_lines, lang='cn', drop_roles=None):
        '''
        Parse a single ConLL sentence block.
        :param sentence_lines: List of lines for one sentence
        :param lang: Language code ('cn' or 'en')
        :param drop_roles: A set of role labels. If any appears, this sample is dropped.
        :return: A dictionary containing the parsed sentence and roles, or None
        '''
        if not sentence_lines: return None
        tokens, word_list, pred_list, children_map = [], [], [], {}
        root_node_info = None

        for idx, line in enumerate(sentence_lines):
            cols = line.strip().split()
            tokens.append(cols)
            curr_id, word = int(cols[0]), cols[1]
            word_list.append(word)
            head_id = int(cols[8]) if cols[8].isdigit() else 0
            if head_id == 0: root_node_info = (idx, curr_id)
            children_map.setdefault(head_id, []).append(curr_id)
            if len(cols) > 13 and cols[13] != '_':
                pred_list.append((idx, cols[13]))

        if not root_node_info: return None
        root_row_idx, root_id = root_node_info
        found_pred_row_idx = -1
        if len(tokens[root_row_idx]) > 13 and tokens[root_row_idx][13] != '_':
            found_pred_row_idx = root_row_idx
        elif lang == 'en':
            candidates = [ri for ri, t in enumerate(tokens) if int(t[8]) == root_id and len(t) > 13 and t[13] != '_']
            if candidates:
                verb_cands = [ri for ri in candidates if tokens[ri][4].startswith('V')]
                found_pred_row_idx = verb_cands[0] if verb_cands else candidates[0]

        if found_pred_row_idx == -1: return None
        target_pred_info = next(((p_idx, p_label) for p_idx, p_label in pred_list if p_idx == found_pred_row_idx), None)
        target_pred_rank = next(
            (rank for rank, (p_idx, p_label) in enumerate(pred_list) if p_idx == found_pred_row_idx), -1)

        if target_pred_info is None: return None
        roles_list = [["rel", target_pred_info[1].split('.')[0]]]
        arg_col_idx = 14 + target_pred_rank

        for token_cols in tokens:
            if len(token_cols) <= arg_col_idx: continue
            arg_label = token_cols[arg_col_idx]
            if arg_label != '_':
                if drop_roles and arg_label in drop_roles:
                    # Drop this sample immediately if it contains any blocked role.
                    return None
                curr_id = int(token_cols[0])
                span_ids = sorted(get_subtree_indices(curr_id, children_map))
                span_words = [tokens[sid - 1][1] for sid in span_ids if sid - 1 < len(tokens)]
                sep = " " if lang == 'en' else ""
                roles_list.append([arg_label, sep.join(span_words)])

        sep = " " if lang == 'en' else ""
        return {"sentence": sep.join(word_list), "roles": roles_list}

    source_configs = [
        ('cn', 'conll2009_cn', ['conll2009_cn_train_src.txt', 'conll2009_cn_dev_src.txt', 'conll2009_cn_test_src.txt']),
        ('en', 'conll2009_en',
         ['conll2009_en_train_src.txt', 'conll2009_en_dev_src.txt', 'conll2009_en_brown_test_src.txt',
          'conll2009_en_wsj_test_src.txt'])
    ]

    for lang_code, folder_name, file_list in source_configs:
        drop_roles = drop_role_by_lang.get(lang_code, set())
        source_dir = os.path.join(project_root, 'data', 'source', folder_name)
        target_output_dir = os.path.join(output_root, folder_name)
        os.makedirs(target_output_dir, exist_ok=True)

        for file_name in file_list:
            input_file = os.path.join(source_dir, file_name)
            output_file = os.path.join(target_output_dir, file_name.replace('_src.txt', '.json'))
            if not os.path.exists(input_file): continue

            processed_data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                buffer = []
                for line in f:
                    if not line.strip():
                        if buffer:
                            res = parse_conll_sentence(buffer, lang=lang_code, drop_roles=drop_roles)
                            if res: processed_data.append(res)
                            buffer = []
                    else:
                        buffer.append(line)
                if buffer:
                    res = parse_conll_sentence(buffer, lang=lang_code, drop_roles=drop_roles)
                    if res: processed_data.append(res)

            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(processed_data, f_out, ensure_ascii=False, indent=2)
            print(f"[ConLL2009] Saved successfully ({lang_code}): {output_file} (Total {len(processed_data)} items)")


# ==========================================================
# 4. FIRE Processing Logic
# ==========================================================
def prep_fire_file(input_path, output_path):
    '''
    Process the FIRE dataset file.
    :param input_path: Path to the input file
    :param output_path: Path to the output file
    :return: None
    '''
    processed_data = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                raw_data = [json.loads(line) for line in f if line.strip()]

        for entry in raw_data:
            roles_list = []
            for ent in entry.get("entities", []):
                role_key, role_val = ent.get("type"), ent.get("text")
                if role_key and role_val:
                    roles_list.append([role_key, role_val])
            processed_data.append({"sentence": " ".join(entry.get("tokens", [])), "roles": roles_list})

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, ensure_ascii=False, indent=2)
        print(f"[FIRE] Saved successfully: {output_path} (Total {len(processed_data)} items)")
    except Exception as e:
        print(f"[FIRE] Processing Error {input_path}: {e}")


# ==========================================================
# 5. PHEE Processing Logic
# ==========================================================
def prep_phee_file(input_path, output_path):
    '''
    Process the PHEE dataset file.
    It reads concatenated JSON objects, extracts the 'text' as sentence,
    and extracts 'trigger' and 'arguments' from 'event_mentions' as roles.
    :param input_path: Path to the input file
    :param output_path: Path to the output file
    :return: None
    '''
    processed_data = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Using JSONDecoder.raw_decode to handle concatenated JSON objects (e.g. {obj}{obj}...)
    # which is common in PHEE source files.
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            # Skip any whitespace/newlines between objects
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break

            try:
                obj, next_pos = decoder.raw_decode(content, pos)
                pos = next_pos

                # Main Extraction Logic
                sentence_text = obj.get('text', '')
                roles_list = []

                # Iterate over event mentions
                for evt in obj.get('event_mentions', []):
                    # 1. Extract Trigger
                    trigger = evt.get('trigger', {})
                    if trigger and trigger.get('text'):
                        # Using 'Trigger' as the generic label for the event anchor
                        roles_list.append(['Trigger', trigger['text']])

                    # 2. Extract Arguments
                    for arg in evt.get('arguments', []):
                        role_type = arg.get('role')
                        role_text = arg.get('text')
                        if role_type and role_text:
                            roles_list.append([role_type, role_text])

                if sentence_text:
                    processed_data.append({"sentence": sentence_text, "roles": roles_list})

            except json.JSONDecodeError as e:
                print(f"[PHEE] Warning: JSON decode error at position {pos} in {input_path}: {e}")
                break

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, ensure_ascii=False, indent=2)
        print(f"[PHEE] Saved successfully: {output_path} (Total {len(processed_data)} items)")

    except Exception as e:
        print(f"[PHEE] Processing Error {input_path}: {e}")

# ==========================================================
# 6. FabNER Processing Logic
# ==========================================================
def prep_fabner_file(input_path, output_path):
    '''
    Process the FabNER dataset file.
    :param input_path: Path to the input file
    :param output_path: Path to the output file
    :return: None
    '''
    processed_data = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                raw_data = [json.loads(line) for line in f if line.strip()]

        for entry in raw_data:
            roles_list = []
            for ent in entry.get("entities", []):
                role_key, role_val = ent.get("type"), ent.get("name")
                if role_key and role_val:
                    roles_list.append([role_key, role_val])
            processed_data.append({"sentence": entry.get("sentence", ""), "roles": roles_list})

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, ensure_ascii=False, indent=2)
        print(f"[FabNER] Saved successfully: {output_path} (Total {len(processed_data)} items)")
    except Exception as e:
        print(f"[FabNER] Processing Error {input_path}: {e}")


# ==========================================================
# 7. ACE2005 Processing Logic
# ==========================================================
def prep_ace2005_file(input_path, output_path):
    '''
    Process the ACE2005 dataset file.
    :param input_path: Path to the input file
    :param output_path: Path to the output file
    :return: None
    '''
    LABEL_MAP = {
        "organization": "ORG",
        "person": "PER",
        "geographical social political": "GPE",
        "vehicle": "VEH",
        "location": "LOC",
        "weapon": "WEA",
        "facility": "FAC"
    }
    processed_data = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for entry in raw_data:
            roles_list = []
            for ent in entry.get("entities", []):
                raw_type = ent.get("type")
                # Map to standard uppercase schema roles
                role_key = LABEL_MAP.get(raw_type, raw_type.upper() if raw_type else "UNK")
                role_val = ent.get("name")
                if role_key and role_val:
                    roles_list.append([role_key, role_val])
            processed_data.append({"sentence": entry.get("sentence", ""), "roles": roles_list})

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, ensure_ascii=False, indent=2)
        print(f"[ACE2005] Saved successfully: {output_path} (Total {len(processed_data)} items)")
    except Exception as e:
        print(f"[ACE2005] Processing Error {input_path}: {e}")



# ==========================================================
# 6. Task Dispatcher Wrapper
# ==========================================================

def run_cpb1_task(project_root):
    '''
    Wrapper to run the CPB1 task.
    :param project_root: Root directory of the project
    :return: None
    '''
    src = os.path.join(project_root, 'data', 'source', 'cpb1')
    out = os.path.join(project_root, 'data', 'input', 'cpb1')
    print("\n>>> Start processing CPB1 dataset...")
    for f in ['cpb1_train_src.txt', 'cpb1_dev_src.txt', 'cpb1_test_src.txt']:
        prep_cpb1_file(os.path.join(src, f), os.path.join(out, f.replace('_src.txt', '.json')))


def run_conll2009_task(project_root):
    '''
    Wrapper to run the ConLL2009 task.
    :param project_root: Root directory of the project
    :return: None
    '''
    print("\n>>> Start processing ConLL2009 dataset...")
    prep_conll2009_dataset(project_root, os.path.join(project_root, 'data', 'input'))



def run_fire_task(project_root):
    '''
    Wrapper to run the FIRE task.
    :param project_root: Root directory of the project
    :return: None
    '''
    src = os.path.join(project_root, 'data', 'source', 'fire')
    out = os.path.join(project_root, 'data', 'input', 'fire')
    print("\n>>> Start processing FIRE dataset...")
    for f in ['fire_train_src.json', 'fire_dev_src.json', 'fire_test_src.json']:
        prep_fire_file(os.path.join(src, f), os.path.join(out, f.replace('_src.json', '.json')))

def run_phee_task(project_root):
    '''
    Wrapper to run the PHEE task.
    :param project_root: Root directory of the project
    :return: None
    '''
    src = os.path.join(project_root, 'data', 'source', 'phee')
    out = os.path.join(project_root, 'data', 'input', 'phee')
    print("\n>>> Start processing PHEE dataset...")
    for f in ['phee_train_src.json', 'phee_dev_src.json', 'phee_test_src.json']:
        prep_phee_file(os.path.join(src, f), os.path.join(out, f.replace('_src.json', '.json')))

def run_fabner_task(project_root):
    '''
    Wrapper to run the FabNER task.
    :param project_root: Root directory of the project
    :return: None
    '''
    src = os.path.join(project_root, 'data', 'source', 'fabner')
    out = os.path.join(project_root, 'data', 'input', 'fabner')
    print("\n>>> Start processing FabNER dataset...")
    for f in ['fabner_train_src.json', 'fabner_dev_src.json', 'fabner_test_src.json']:
        prep_fabner_file(os.path.join(src, f), os.path.join(out, f.replace('_src.json', '.json')))


def run_ace2005_task(project_root):
    '''
    Wrapper to run the ACE2005 task.
    :param project_root: Root directory of the project
    :return: None
    '''
    src = os.path.join(project_root, 'data', 'source', 'ace2005')
    out = os.path.join(project_root, 'data', 'input', 'ace2005')
    print("\n>>> Start processing ACE2005 dataset...")
    for f in ['ace2005_train.json', 'ace2005_dev.json', 'ace2005_test.json']:
        input_file = os.path.join(src, f)
        output_file = os.path.join(out, f)
        prep_ace2005_file(input_file, output_file)

# ==========================================================
# 7. Main Entry Point
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="conll2009",
        help="Comma-separated dataset groups to preprocess. Choices: cpb1, conll2009, fire, phee, fabner, ace2005.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DISPATCHER = {
        'cpb1': run_cpb1_task,
        'conll2009': run_conll2009_task,
        'fire': run_fire_task,
        'phee': run_phee_task,
        'fabner': run_fabner_task,
        'ace2005': run_ace2005_task
    }

    datasets_to_run = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets_to_run:
        raise ValueError("No datasets were provided. Use --datasets to select at least one preprocessing target.")

    print(f"Current selected tasks: {', '.join(datasets_to_run)}")

    for task_name in datasets_to_run:
        if task_name in DISPATCHER:
            DISPATCHER[task_name](project_root)
        else:
            print(f"Error: Unknown task name '{task_name}'")

    print("\n>>> All selected tasks processed successfully.")
