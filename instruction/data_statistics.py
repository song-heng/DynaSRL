import os
import json
import re


CJK_CHAR_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
EN_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def count_lines_in_jsonl(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_items_in_json(file_path):
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                return len(data.keys())
    except Exception as e:
        print(f"Warning: Failed to parse JSON {file_path}: {e}")
    return 0


def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse JSON {file_path}: {e}")
        return None


def get_split_aliases(dataset_name, split):
    # conll2009_en has no direct test split file
    if dataset_name == "conll2009_en" and split == "test":
        return ["wsj_test", "brown_test"]
    return [split]


def get_count(dir_path, dataset_name, split):
    total = 0
    for real_split in get_split_aliases(dataset_name, split):
        # Prefer instruction jsonl for compatibility with current logic
        jsonl_path = os.path.join(dir_path, f"{dataset_name}_{real_split}_ins.jsonl")
        if os.path.exists(jsonl_path):
            total += count_lines_in_jsonl(jsonl_path)
            continue

        # Fallback to raw json
        json_path = os.path.join(dir_path, f"{dataset_name}_{real_split}.json")
        if os.path.exists(json_path):
            total += count_items_in_json(json_path)
    return total


def iter_records(dir_path, dataset_name, split):
    for real_split in get_split_aliases(dataset_name, split):
        json_path = os.path.join(dir_path, f"{dataset_name}_{real_split}.json")
        if os.path.exists(json_path):
            data = load_json_file(json_path)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, dict):
                        yield value
            continue

        jsonl_path = os.path.join(dir_path, f"{dataset_name}_{real_split}_ins.jsonl")
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict):
                                yield item
                        except Exception:
                            continue
            except Exception as e:
                print(f"Warning: Failed to parse JSONL {jsonl_path}: {e}")


def collect_roles_from_schema(obj, role_set):
    if isinstance(obj, dict):
        role = obj.get("role")
        if isinstance(role, str) and role:
            role_set.add(role)
        for value in obj.values():
            collect_roles_from_schema(value, role_set)
    elif isinstance(obj, list):
        for item in obj:
            collect_roles_from_schema(item, role_set)


def collect_roles_from_instance(record, role_set):
    roles = record.get("roles")
    if not isinstance(roles, list):
        return
    for role_item in roles:
        role_name = None
        if isinstance(role_item, (list, tuple)) and role_item:
            role_name = role_item[0]
        elif isinstance(role_item, dict):
            role_name = role_item.get("role")
        elif isinstance(role_item, str):
            role_name = role_item
        if isinstance(role_name, str) and role_name:
            role_set.add(role_name)


def count_sentence_units(sentence):
    if not isinstance(sentence, str) or not sentence.strip():
        return 0

    cjk_count = len(CJK_CHAR_PATTERN.findall(sentence))
    if cjk_count > 0:
        # Chinese: each Han character counts as one unit.
        # If English words exist in mixed text, count them as words.
        non_cjk_text = CJK_CHAR_PATTERN.sub(" ", sentence)
        en_word_count = len(EN_WORD_PATTERN.findall(non_cjk_text))
        return cjk_count + en_word_count

    # English: count words
    return len(EN_WORD_PATTERN.findall(sentence))


def get_dataset_stats(dir_path, dataset_name):
    role_set = set()
    total_units = 0
    total_sentences = 0

    for split in ["train", "dev", "test"]:
        for record in iter_records(dir_path, dataset_name, split):
            collect_roles_from_instance(record, role_set)
            sentence = record.get("sentence")
            if not isinstance(sentence, str):
                continue
            total_units += count_sentence_units(sentence)
            total_sentences += 1

    # Fallback role source if no role found in instances
    if not role_set:
        schema_path = os.path.join(dir_path, f"{dataset_name}_schema.json")
        if os.path.exists(schema_path):
            schema_data = load_json_file(schema_path)
            if schema_data is not None:
                collect_roles_from_schema(schema_data, role_set)

    return len(role_set), total_units, total_sentences


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "input"))

    if not os.path.exists(input_dir):
        print(f"Error: Dataset directory does not exist at {input_dir}")
        return

    datasets = sorted(
        [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    )

    if not datasets:
        print(f"No datasets found in {input_dir}")
        return

    header = (
        f"| {'Dataset Name':<15} | {'Role #':<8} | {'Avg Sent Len':<12} | "
        f"{'Train':<10} | {'Dev':<10} | {'Test':<10} | {'Total':<10} |"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    total_train = 0
    total_dev = 0
    total_test = 0
    total_sent_units = 0
    total_sentences = 0

    for ds in datasets:
        ds_dir = os.path.join(input_dir, ds)

        train_count = get_count(ds_dir, ds, "train")
        dev_count = get_count(ds_dir, ds, "dev")
        test_count = get_count(ds_dir, ds, "test")
        role_count, sent_units, sent_num = get_dataset_stats(ds_dir, ds)
        avg_sent_len = (sent_units / sent_num) if sent_num > 0 else 0.0

        ds_total = train_count + dev_count + test_count

        total_train += train_count
        total_dev += dev_count
        total_test += test_count
        total_sent_units += sent_units
        total_sentences += sent_num

        print(
            f"| {ds:<15} | {role_count:<8} | {avg_sent_len:<12.2f} | "
            f"{train_count:<10} | {dev_count:<10} | {test_count:<10} | {ds_total:<10} |"
        )

    overall_total = total_train + total_dev + total_test
    overall_avg_sent_len = (total_sent_units / total_sentences) if total_sentences > 0 else 0.0
    print(separator)
    print(
        f"| {'ALL DATASETS':<15} | {'-':<8} | {overall_avg_sent_len:<12.2f} | "
        f"{total_train:<10} | {total_dev:<10} | {total_test:<10} | {overall_total:<10} |"
    )
    print(separator + "\n")


if __name__ == "__main__":
    main()
