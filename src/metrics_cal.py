import argparse
import json
import os
import sys
import difflib
import re
from collections import Counter
from datetime import datetime

# ==========================================================
# Auto-locate project root: Assume metrics_cal.py is in src directory
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Get parent of src, i.e., project root

# Ensure metrics_utils can be found in src directory
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from metrics_utils import DynaSRLMetrics
except ImportError:
    print("Error: Cannot import DynaSRLMetrics. Please ensure metrics_utils.py is in the src directory.")
    sys.exit(1)


# ==========================================================
# [New] Relaxed Match Evaluator Class
# Inherits from DynaSRLMetrics, overrides update method to support fuzzy matching
# ==========================================================
class RelaxedDynaSRLMetrics(DynaSRLMetrics):
    def __init__(self, relax_match=False, threshold=0.80):
        super().__init__()
        self.relax_match = relax_match
        self.threshold = threshold

    @staticmethod
    def _tokenize_span(text):
        """
        Tokenize span for overlap calculation.
        Keep alphanumeric tokens and simple hyphen compounds.
        """
        text = str(text).lower().strip()
        return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text)

    def _overlap_rate(self, pred_span, gold_span):
        """
        Gold-coverage overlap rate:
            overlap_rate = overlap_tokens / gold_tokens
        where overlap is computed by multiset token intersection.
        """
        pred_counter = Counter(self._tokenize_span(pred_span))
        gold_counter = Counter(self._tokenize_span(gold_span))

        gold_token_count = sum(gold_counter.values())
        if gold_token_count == 0:
            # If gold span is empty after tokenization, only empty pred is a full match.
            return 1.0 if sum(pred_counter.values()) == 0 else 0.0

        overlap_count = sum((pred_counter & gold_counter).values())
        return overlap_count / gold_token_count

    def update(self, preds_text, labels_text):
        '''
        Core calculation logic: Support Exact Match and Relaxed Match.
        :param preds_text: Prediction text string
        :param labels_text: Gold label text string
        :return: None
        '''
        # 1. Basic parsing (Get set, auto-deduplication)
        pred_set = self.parse_srl_output(preds_text)
        gold_set = self.parse_srl_output(labels_text)

        # If relaxed match is not enabled, use parent class logic (Exact Match Only)
        if not self.relax_match:
            super().update(preds_text, labels_text)
            return

        # ================= [Relaxed Match Logic Start] =================

        # 2. First round: Exact Match
        # Intersection of two sets is completely correct
        exact_tp_set = pred_set & gold_set

        # 3. Filter unmatched candidates
        # Remaining unmatched predictions
        candidates_pred = pred_set - exact_tp_set
        # Remaining unfound gold standards
        candidates_gold = gold_set - exact_tp_set

        relaxed_tp_count = 0
        matched_gold_candidates = set()  # Record matched gold to prevent one-to-many mapping

        # 4. Second round: Relaxed Match
        for p in candidates_pred:
            p_label, p_span = p

            # Find best match in remaining gold
            best_ratio = 0
            best_g = None

            for g in candidates_gold:
                if g in matched_gold_candidates:
                    continue  # Already matched, skip

                g_label, g_span = g

                # Condition 1: Label must match exactly
                if p_label == g_label:
                    # Condition 2: Token overlap rate (w.r.t. gold span) >= threshold
                    ratio = self._overlap_rate(p_span, g_span)
                    if ratio >= self.threshold and ratio > best_ratio:
                        best_ratio = ratio
                        best_g = g

            # If valid fuzzy match found
            if best_g:
                relaxed_tp_count += 1
                matched_gold_candidates.add(best_g)
                # print(f"[Relaxed Match] Pred: '{p_span}' ~= Gold: '{best_g[1]}' (Sim: {best_ratio:.2f})") # For debug

        # 5. Summary statistics
        total_tp = len(exact_tp_set) + relaxed_tp_count

        # FP = Total Pred - (Exact TP + Relaxed TP)
        total_fp = len(pred_set) - total_tp

        # FN = Total Gold - (Exact TP + Relaxed TP)
        total_fn = len(gold_set) - total_tp

        self.tp += total_tp
        self.fp += total_fp
        self.fn += total_fn


def roles_to_srl_string(roles_data):
    '''
    Convert 2D list to string parseable by metrics_utils.
    Input format: [["label1", "span1"], ["label2", "span2"]]
    Output format: "(label1, span1), (label2, span2)"
    :param roles_data: List of roles and spans
    :return: Formatted string representation
    '''
    if not roles_data:
        return ""
    items = [f"({label}, {span})" for label, span in roles_data]
    return ", ".join(items)


def run_evaluation(task_name, pred_rel_path, gold_rel_path, save_path=None, relax_match=False):
    '''
    Execute evaluation logic.
    :param task_name: Name of the task
    :param pred_rel_path: Relative path to prediction file
    :param gold_rel_path: Relative path to gold standard file
    :param save_path: Path to save the log
    :param relax_match: [New] Whether to enable relaxed matching
    :return: None
    '''
    pred_path = os.path.join(PROJECT_ROOT, pred_rel_path)
    gold_path = os.path.join(PROJECT_ROOT, gold_rel_path)

    header_msg = f"\n>>> Evaluating Task: {task_name}"
    print(header_msg)

    # [Modification] Use new evaluator class and pass config
    metrics = RelaxedDynaSRLMetrics(relax_match=relax_match, threshold=0.80)

    if not os.path.exists(pred_path):
        err_msg = f"  [Error] Prediction file not found: {pred_path}"
        print(err_msg)
        if save_path:
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(header_msg + "\n" + err_msg + "\n")
        return
    if not os.path.exists(gold_path):
        err_msg = f"  [Error] Gold label file not found: {gold_path}"
        print(err_msg)
        if save_path:
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(header_msg + "\n" + err_msg + "\n")
        return

    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            preds_data = json.load(f)
        with open(gold_path, 'r', encoding='utf-8') as f:
            golds_data = json.load(f)
    except Exception as e:
        err_msg = f"  [Error] Failed to load JSON: {e}"
        print(err_msg)
        if save_path:
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(header_msg + "\n" + err_msg + "\n")
        return

    for pred_entry, gold_entry in zip(preds_data, golds_data):
        pred_str = roles_to_srl_string(pred_entry.get("roles", []))
        gold_str = roles_to_srl_string(gold_entry.get("roles", []))
        metrics.update(pred_str, gold_str)

    results = metrics.compute()

    # Result Display
    match_mode_str = "Mode: [Relaxed Match (>80%)]" if relax_match else "Mode: [Exact Match]"

    result_log = (
        f"{'-' * 40}\n"
        f"{match_mode_str}\n"  # Print current mode
        f"{'-' * 40}\n"
        f"{'Metric':<15} | {'Value':<10}\n"
        f"{'-' * 40}\n"
        f"{'Precision':<15} | {results['precision']:.4f}\n"
        f"{'Recall':<15} | {results['recall']:.4f}\n"
        f"{'F1 Score':<15} | {results['f1']:.4f}\n"
        f"{'-' * 40}\n"
        f"TP: {int(results['eval_tp'])}, FP: {int(results['eval_fp'])}, FN: {int(results['eval_fn'])}\n"
        f"{'-' * 40}\n"
    )

    print(result_log)

    if save_path:
        try:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(header_msg + "\n")
                f.write(result_log)
        except Exception as e:
            print(f"  [Warning] Cannot write to log file: {e}")


if __name__ == "__main__":
    RELAX_MATCH = True

    EVAL_CONFIG = {
        "cpb1": {
            "pred": "data/output/cpb1/cpb1_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "cpb1-Q8B": {
            "pred": "data/output/cpb1/cpb1-Q8B_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "cpb1-Q14B": {
            "pred": "data/output/cpb1/cpb1-Q14B_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "cpb1-Q4B": {
            "pred": "data/output/cpb1/cpb1-Q4B_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "cpb1-Q1.7B": {
            "pred": "data/output/cpb1/cpb1-Q1.7B_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "cpb1-L3B": {
            "pred": "data/output/cpb1/cpb1-L3B_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "cpb1-L1B": {
            "pred": "data/output/cpb1/cpb1-L1B_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "conll2009_cn": {
            "pred": "data/output/conll2009_cn/conll2009_cn_pred.json",
            "gold": "data/input/conll2009_cn/conll2009_cn_test.json"
        },
        "conll2009_en_wsj": {
            "pred": "data/output/conll2009_en/conll2009_en_wsj_pred.json",
            "gold": "data/input/conll2009_en/conll2009_en_wsj_test.json"
        },
        "conll2009_en_brown": {
            "pred": "data/output/conll2009_en/conll2009_en_brown_pred.json",
            "gold": "data/input/conll2009_en/conll2009_en_brown_test.json"
        },
        "discourseee": {
            "pred": "data/output/discourseee/discourseee_pred.json",
            "gold": "data/input/discourseee/discourseee_test.json"
        },
        "fire": {
            "pred": "data/output/fire/fire_pred.json",
            "gold": "data/input/fire/fire_test.json"
        },
        "phee": {
            "pred": "data/output/phee/phee_pred.json",
            "gold": "data/input/phee/phee_test.json"
        },
        "fabner": {
            "pred": "data/output/fabner/fabner_pred.json",
            "gold": "data/input/fabner/fabner_test.json"
        },
        "cpb1_vanilla": {
            "pred": "data/output/cpb1/cpb1_vanilla_pred.json",
            "gold": "data/input/cpb1/cpb1_test.json"
        },
        "conll2009_cn_vanilla": {
            "pred": "data/output/conll2009_cn/conll2009_cn_vanilla_pred.json",
            "gold": "data/input/conll2009_cn/conll2009_cn_test.json"
        },
        "conll2009_en_wsj_vanilla": {
            "pred": "data/output/conll2009_en/conll2009_en_wsj_vanilla_pred.json",
            "gold": "data/input/conll2009_en/conll2009_en_wsj_test.json"
        },
        "conll2009_en_brown_vanilla": {
            "pred": "data/output/conll2009_en/conll2009_en_brown_vanilla_pred.json",
            "gold": "data/input/conll2009_en/conll2009_en_brown_test.json"
        },
        "conll2009_cn_glad_disabled_pred": {
            "pred": "data/output/conll2009_cn/conll2009_cn_glad_disabled_pred.json",
            "gold": "data/input/conll2009_cn/conll2009_cn_test.json"
        },
    }
    DEFAULT_TASKS_TO_RUN = [
        "cpb1-Q14B",
        "cpb1-Q8B",
        "cpb1-Q4B",
        "cpb1-Q1.7B",
        "cpb1-L3B",
        "cpb1-L1B",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS_TO_RUN),
        help="Comma-separated task names from EVAL_CONFIG.",
    )
    parser.add_argument("--pred_path", type=str, default=None, help="Custom prediction JSON path.")
    parser.add_argument("--gold_path", type=str, default=None, help="Custom gold JSON path.")
    parser.add_argument("--label", type=str, default="custom", help="Task label used with custom paths.")
    parser.add_argument("--exact_match", action="store_true", help="Disable relaxed matching and use exact match only.")
    args = parser.parse_args()

    relax_match = not args.exact_match

    current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join(PROJECT_ROOT, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    prefix = "metrics_stat_rm" if relax_match else "metrics_stat"
    log_filename = f"{prefix}_{current_time_str}.txt"
    log_file_path = os.path.join(output_dir, log_filename)

    print(
        f"[*] Evaluation results ({'Relaxed Match' if relax_match else 'Exact Match'}) will be saved to: {log_file_path}")

    if args.pred_path and args.gold_path:
        run_evaluation(args.label, args.pred_path, args.gold_path, save_path=log_file_path, relax_match=relax_match)
    else:
        tasks_to_run = [item.strip() for item in args.tasks.split(",") if item.strip()]
        for task in tasks_to_run:
            if task in EVAL_CONFIG:
                conf = EVAL_CONFIG[task]
                run_evaluation(task, conf["pred"], conf["gold"], save_path=log_file_path, relax_match=relax_match)
            else:
                print(f"Error: Task '{task}' not registered in EVAL_CONFIG.")

    print(f"\n>>> All evaluation tasks completed. Log saved: {log_file_path}")
