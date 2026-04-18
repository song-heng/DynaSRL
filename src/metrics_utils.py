import re


class DynaSRLMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        '''
        Reset all counters.
        :return: None
        '''
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives (False Alarm)
        self.fn = 0  # False Negatives (Missed)

    def parse_srl_output(self, text):
        '''
        Parse tuple output. Ensures exact extraction of Label and Span.
        :param text: Input text string
        :return: Set of tuples {('Label', 'Span'), ...}
        '''
        # Match (Label, Span) format
        pattern = r"\(([^,]+),\s*([^)]+)\)"
        matches = re.findall(pattern, text)
        # Return set: {('Agent', 'John'), ('Predicate', 'Eat'), ...}
        return set([(m[0].strip(), m[1].strip()) for m in matches])

    def update(self, preds_text, labels_text):
        '''
        Update metrics based on prediction and gold standard text.
        Logic aligns with strict set comparison.
        :param preds_text: Prediction string
        :param labels_text: Gold label string
        :return: None
        '''
        pred_set = self.parse_srl_output(preds_text)
        gold_set = self.parse_srl_output(labels_text)

        # 1. TP: Intersection of two sets (Exact Match)
        current_tp = len(pred_set & gold_set)

        # 2. FP: In prediction but not in gold (False Alarm)
        # Includes: wrong label, wrong boundary, hallucination
        current_fp = len(pred_set - gold_set)

        # 3. FN: In gold but not in prediction (Missed)
        # Includes: missed prediction, or lost gold role due to wrong prediction
        current_fn = len(gold_set - pred_set)

        self.tp += current_tp
        self.fp += current_fp
        self.fn += current_fn

    def compute(self):
        '''
        Compute and return the full metrics dictionary.
        :return: Dictionary containing Precision, Recall, F1, and raw counts
        '''
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # Calculate simple accuracy (Optional logic placeholder)
        # total_elements = self.tp + self.fp + self.fn
        # accuracy = self.tp / total_elements if total_elements > 0 else 0

        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            # "accuracy": accuracy,
            "eval_tp": float(self.tp),
            "eval_fp": float(self.fp),
            "eval_fn": float(self.fn)
        }
