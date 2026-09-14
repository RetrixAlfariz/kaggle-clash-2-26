"""Exact slot and unordered entity metrics; no fuzzy credit in scores."""
from collections import Counter, defaultdict

LABELS = ("NAME", "DATE", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "JOB_TITLE")


def triples(entities):
    return sorted((e["start"], e["end"], e["label"]) for e in entities)


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def relation(gold, available):
    if gold in available:
        return "exact"
    if any(gold[:2] == p[:2] for p in available):
        return "same_boundary_wrong_label"
    if any(overlap(gold, p) and gold[2] == p[2] for p in available):
        return "overlap_same_label"
    if any(overlap(gold, p) for p in available):
        return "overlap_wrong_label"
    return "none"


def ratio(a, b):
    return a / b if b else 0.0


def prf(correct, predicted, gold):
    return {"precision": ratio(correct, predicted), "recall": ratio(correct, gold),
            "f1": ratio(2 * correct, predicted + gold)}


class Evaluation:
    def __init__(self):
        self.counts = Counter()
        self.labels = {label: Counter() for label in LABELS}
        self.candidate_errors = Counter()
        self.output_errors = Counter()
        self.label_output_errors = defaultdict(Counter)
        self.confusion_exact_boundary = Counter()

    def add(self, text, gold, predicted, candidates):
        for spans in (gold, predicted):
            if len(set(spans)) != len(spans) or list(spans) != sorted(spans):
                raise ValueError("Duplicate or unsorted spans")
            last_end = 0
            for start, end, label in spans:
                if not (last_end <= start < end <= len(text)) or label not in LABELS:
                    raise ValueError("Invalid, overlapping or unknown-label spans")
                last_end = end
        actual, expected = set(predicted), set(gold)
        c = self.counts
        c.update({"documents": 1, "gold": len(gold), "predicted": len(predicted),
                  "entity_correct": len(actual & expected), "boundary_correct": len({s[:2] for s in actual} & {s[:2] for s in expected}),
                  "slot_correct": sum(a == b for a, b in zip(gold, predicted)),
                  "exact_documents": gold == predicted, "count_correct_documents": len(gold) == len(predicted),
                  "under_count_documents": len(predicted) < len(gold), "over_count_documents": len(predicted) > len(gold),
                  "missing_count": max(0, len(gold) - len(predicted)), "excess_count": max(0, len(predicted) - len(gold)),
                  "candidate_gold_found": len(expected & set(candidates))})
        for span in predicted:
            self.labels[span[2]]["predicted"] += 1
        rows = []
        for i, span in enumerate(gold):
            label = self.labels[span[2]]
            label.update({"gold": 1, "correct": span in actual,
                          "slot_correct": i < len(predicted) and predicted[i] == span,
                          "candidate_found": span in candidates})
            stage = relation(span, candidates)
            stage = "exact_candidate" if stage == "exact" else "no_overlapping_candidate" if stage == "none" else stage
            output = relation(span, actual)
            if output == "exact":
                output = "correct_slot" if i < len(predicted) and predicted[i] == span else "exact_span_wrong_slot"
            elif output == "none":
                output = "no_overlapping_prediction"
            self.candidate_errors[stage] += 1
            self.output_errors[output] += 1
            self.label_output_errors[span[2]][output] += 1
            if span in candidates and span not in actual:
                c["exact_candidate_not_selected"] += 1
            for p in predicted:
                if span[:2] == p[:2]:
                    self.confusion_exact_boundary[f"{span[2]}->{p[2]}"] += 1
            rows.append({"gold_slot": i + 1, "gold": span, "candidate_category": stage, "output_category": output})
        return rows

    def report(self):
        c = self.counts
        return {"counts": dict(c), "slot_accuracy": ratio(c["slot_correct"], c["gold"]),
                "entity_micro": prf(c["entity_correct"], c["predicted"], c["gold"]),
                "boundary_micro": prf(c["boundary_correct"], c["predicted"], c["gold"]),
                "candidate_recall": ratio(c["candidate_gold_found"], c["gold"]),
                "exact_document_rate": ratio(c["exact_documents"], c["documents"]),
                "count_correct_document_rate": ratio(c["count_correct_documents"], c["documents"]),
                "per_label": {label: {**dict(v), **prf(v["correct"], v["predicted"], v["gold"]),
                                       "slot_accuracy": ratio(v["slot_correct"], v["gold"]),
                                       "candidate_recall": ratio(v["candidate_found"], v["gold"])} for label, v in self.labels.items()},
                "candidate_stage": dict(self.candidate_errors), "output_stage": dict(self.output_errors),
                "output_stage_by_label": {k: dict(v) for k, v in self.label_output_errors.items()},
                "exact_boundary_label_confusion": dict(self.confusion_exact_boundary)}
