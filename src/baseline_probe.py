# /// script
# requires-python = ">=3.13"
# dependencies = ["pyarrow>=18"]
# ///
"""Small, deterministic gold-only experiment. Run: uv run src/baseline_probe.py."""
import bisect
import collections
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
END = None


def reconstruct():
    segments = collections.defaultdict(list)
    with (ROOT / "Data/train/train_segments.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            segments[row["document_id"]].append((int(row["segment_index"]), row["text"]))
    metadata = pq.read_table(ROOT / "Data/train/train_metadata.parquet").to_pylist()
    assert {r["document_id"] for r in metadata} == set(segments)
    texts = {}
    for row in metadata:
        parts = sorted(segments[row["document_id"]])
        assert [i for i, _ in parts] == list(range(row["n_segments"]))
        text = "\n".join(s for _, s in parts)
        assert len(text) == row["n_chars"], row["document_id"]
        texts[row["document_id"]] = text
    labels = collections.defaultdict(list)
    with (ROOT / "Data/train/train_labels.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            a, b = int(row["start_offset"]), int(row["end_offset"])
            assert 0 <= a < b <= len(texts[row["document_id"]])
            labels[row["document_id"]].append((a, b, row["label"]))
    for spans in labels.values():
        spans.sort()
    return texts, labels


def matches(text, trie):
    for start in range(len(text)):
        if start and text[start - 1].isalnum() and text[start].isalnum():
            continue
        node = trie
        end = start
        while end < len(text) and text[end] in node:
            node = node[text[end]]
            end += 1
            if END in node and not (end < len(text) and text[end - 1].isalnum() and text[end].isalnum()):
                yield start, end, node[END]


def decode(candidates, count=None):
    """Weighted interval DP; count mode maximizes total confidence at fixed K."""
    items = sorted(candidates.items(), key=lambda item: (item[0][1], item[0][0], item[0][2]))
    ends = [span[1] for span, _ in items]
    prior = [bisect.bisect_right(ends, span[0], hi=i) for i, (span, _) in enumerate(items)]
    if count is None:
        values = [0.0]
        paths = [()]
        for i, (span, score) in enumerate(items):
            weight = math.log(max(score, 1e-6) / max(1 - score, 1e-6))
            take = values[prior[i]] + weight
            if take > values[-1]:
                values.append(take)
                paths.append(paths[prior[i]] + (span,))
            else:
                values.append(values[-1])
                paths.append(paths[-1])
        return sorted(paths[-1])
    table = [[0.0] + [-math.inf] * count]
    paths = [[()] + [None] * count]
    for i, (span, score) in enumerate(items):
        values = table[-1].copy()
        selected = paths[-1].copy()
        for k in range(1, count + 1):
            take = table[prior[i]][k - 1] + score
            if take > values[k]:
                values[k] = take
                selected[k] = paths[prior[i]][k - 1] + (span,)
        table.append(values)
        paths.append(selected)
    feasible = max(k for k, path in enumerate(paths[-1]) if path is not None)
    return sorted(paths[-1][feasible])


def main():
    # Decoder must prefer two compatible spans over one overlapping span.
    assert decode({(0, 5, "NAME"): .99, (0, 2, "NAME"): .8, (3, 5, "NAME"): .8}, 2) == [(0, 2, "NAME"), (3, 5, "NAME")]
    texts, gold = reconstruct()
    print(f"Verified metadata reconstruction for {len(texts):,} training documents.", flush=True)
    groups = collections.defaultdict(list)
    for doc, text in texts.items():
        groups[hashlib.sha256(" ".join(text.split()).encode()).hexdigest()].append(doc)
    keys = sorted(groups)
    random.Random(2026).shuffle(keys)
    train = [doc for key in keys[:10000] for doc in groups[key]]
    valid = [doc for key in keys[10000:12000] for doc in groups[key]]
    trie = {}
    phrase_labels = collections.defaultdict(collections.Counter)
    for doc in train:
        for a, b, label in gold[doc]:
            phrase_labels[texts[doc][a:b]][label] += 1
    for phrase in phrase_labels:
        node = trie
        for char in phrase:
            node = node.setdefault(char, {})
        node[END] = phrase
    occurrences = collections.Counter()
    matched_positives = collections.defaultdict(collections.Counter)
    for doc in train:
        annotations = {(a, b): label for a, b, label in gold[doc]}
        for a, b, phrase in matches(texts[doc], trie):
            occurrences[phrase] += 1
            if (a, b) in annotations:
                matched_positives[phrase][annotations[a, b]] += 1
    confidence = {}
    for phrase, counts in phrase_labels.items():
        label = (matched_positives[phrase] or counts).most_common(1)[0][0]
        positive = matched_positives[phrase][label]
        confidence[phrase] = (label, (positive + 1) / (occurrences[phrase] + 2))
        assert 0 < confidence[phrase][1] < 1
    patterns = [
        ("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), .85),
        ("PHONE", re.compile(r"(?<!\w)(?:\+1[ -]?)?\(\d{3}\)[ -]?\d{3}-\d{4}(?!\d)"), .8),
        ("DATE", re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b|\b\d{4}-\d{2}-\d{2}\b"), .8),
    ]
    results = {mode: collections.Counter() for mode in ("unconstrained", "known_count")}
    per_label = collections.defaultdict(collections.Counter)
    errors = []
    for doc in valid:
        text = texts[doc]
        expected = gold[doc]
        candidates = {}
        for a, b, phrase in matches(text, trie):
            label, probability = confidence[phrase]
            candidates[a, b, label] = probability
        for label, pattern, probability in patterns:
            for match in pattern.finditer(text):
                span = match.start(), match.end(), label
                # Keep the observed train precision when this exact candidate is known.
                candidates.setdefault(span, probability)
        for mode, count in (("unconstrained", None), ("known_count", len(expected))):
            predicted = decode(candidates, count)
            stats = results[mode]
            stats["gold"] += len(expected)
            stats["predicted"] += len(predicted)
            stats["set_correct"] += len(set(expected) & set(predicted))
            stats["slot_correct"] += sum(a == b for a, b in zip(expected, predicted))
            stats["exact_documents"] += predicted == expected
            stats["count_correct_documents"] += len(predicted) == len(expected)
            stats["candidate_gold_found"] += sum(s in candidates for s in expected)
            if mode == "known_count":
                for i, span in enumerate(expected):
                    c = per_label[span[2]]
                    c["gold"] += 1
                    c["candidate_found"] += span in candidates
                    c["set_correct"] += span in predicted
                    c["slot_correct"] += i < len(predicted) and predicted[i] == span
                if predicted != expected and len(errors) < 12:
                    show = lambda spans: [{"start": a, "end": b, "label": label, "text": text[a:b]} for a, b, label in spans]
                    errors.append({"document_id": doc, "text": text, "gold": show(expected), "predicted": show(predicted)})
    for stats in results.values():
        stats["slot_accuracy"] = stats["slot_correct"] / stats["gold"]
        stats["entity_set_f1"] = 2 * stats["set_correct"] / (stats["gold"] + stats["predicted"])
        stats["candidate_recall"] = stats["candidate_gold_found"] / stats["gold"]
    report = {
        "seed": 2026, "train_documents": len(train), "validation_documents": len(valid),
        "reconstruction_verified_documents": len(texts), "dictionary_phrases": len(phrase_labels),
        "method": "Train-only exact phrase trie with smoothed occurrence precision, three regex fallbacks, weighted non-overlapping interval decoding.",
        "limitations": ["Single small split; normalized duplicates grouped, near duplicates not grouped.", "Known-count validation uses gold counts because test supplies counts in sample_submission.", "Missing predictions score zero; insufficient candidates are not fabricated into a submission.", "Regex confidence values are fixed heuristics, not calibrated; weak data unused."],
        "results": results, "known_count_by_label": per_label,
        "split": {"train": train, "validation": valid},
    }
    output = ROOT / "output/baseline_probe"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "split"}, indent=2))


if __name__ == "__main__":
    main()
