"""Frozen full-train/dev dictionary+regex baseline. uv run --frozen python src/run_m0.py"""
import argparse
import collections
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from baseline_probe import matches, decode, END
from prepared_loader import PreparedData
from ner_evaluation import Evaluation, triples, overlap

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    ("EMAIL", r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", .85),
    ("PHONE", r"(?<!\w)(?:\+1[ -]?)?\(\d{3}\)[ -]?\d{3}-\d{4}(?!\d)", .8),
    ("DATE", r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b|\b\d{4}-\d{2}-\d{2}\b", .8),
]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit(train):
    phrase_labels = collections.defaultdict(collections.Counter)
    for row in train:
        for e in row["entities"]:
            phrase_labels[e["text"]][e["label"]] += 1
    trie = {}
    for phrase in sorted(phrase_labels):
        node = trie
        for char in phrase:
            node = node.setdefault(char, {})
        node[END] = phrase
    occurrences = collections.Counter()
    positives = collections.defaultdict(collections.Counter)
    for i, row in enumerate(train, 1):
        gold = {(e["start"], e["end"]): e["label"] for e in row["entities"]}
        for a, b, phrase in matches(row["full_text"], trie):
            occurrences[phrase] += 1
            if (a, b) in gold:
                positives[phrase][gold[a, b]] += 1
        if i % 10000 == 0:
            print(f"Fitted occurrence counts: {i} train documents", flush=True)
    dictionary = []
    for phrase in sorted(phrase_labels):
        counts = positives[phrase] or phrase_labels[phrase]
        label = min(counts, key=lambda k: (-counts[k], k))
        score = (positives[phrase][label] + 1) / (occurrences[phrase] + 2)
        dictionary.append({"phrase": phrase, "label": label, "score": score,
                           "positive_occurrences": positives[phrase][label], "occurrences": occurrences[phrase]})
    return trie, dictionary


def candidates_for(text, trie, confidence):
    candidates = {}
    for a, b, phrase in matches(text, trie):
        label, score = confidence[phrase]
        candidates[a, b, label] = score
    for label, pattern, score in PATTERNS:
        for match in re.finditer(pattern, text):
            candidates.setdefault((match.start(), match.end(), label), score)
    return candidates


def entities(spans):
    return [{"start": a, "end": b, "label": label} for a, b, label in spans]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/m0/v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError("M0 output already exists; choose a new --output to preserve prior runs")
    sources = ["output/prepared/v1/train.parquet", "output/prepared/v1/dev.parquet", "output/prepared/v1/manifest.json",
               "src/run_m0.py", "src/ner_evaluation.py", "src/baseline_probe.py", "src/prepared_loader.py",
               "note/M0_EVALUATION_PROTOCOL.md", "pyproject.toml", "uv.lock", ".python-version"]
    hashes = {p: sha(ROOT / p) for p in sources}
    out.mkdir(parents=True)
    config = {"run": "m0-v1", "fitting": "all v1 train", "evaluation": "all v1 dev", "dev_tuning": False,
              "regex": PATTERNS, "known_count": "expected_entity_count", "dictionary_tie": "frequency descending then label alphabetical",
              "sources": hashes, "runtime": {"python": platform.python_version(), "pyarrow": pa.__version__}}
    (out / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.monotonic()
    data = PreparedData()
    train = data.load_train().to_pylist()
    trie, dictionary = fit(train)
    pq.write_table(pa.Table.from_pylist(dictionary), out / "dictionary.parquet", compression="zstd")
    confidence = {r["phrase"]: (r["label"], r["score"]) for r in dictionary}
    train_count = len(train)
    del train
    dev = data.load_dev("evaluation").to_pylist()
    evaluations = {mode: Evaluation() for mode in ("known_count", "unconstrained")}
    predictions, examples = [], []
    example_counts = collections.Counter()
    candidate_total = 0
    for i, row in enumerate(dev, 1):
        text = row["full_text"]
        candidates = candidates_for(text, trie, confidence)
        candidate_total += len(candidates)
        # Predict with text and exposed K; only then use gold for evaluation.
        outputs = {"known_count": decode(candidates, row["expected_entity_count"]), "unconstrained": decode(candidates)}
        gold = triples(row["entities"])
        record = {"document_id": row["document_id"], "expected_entity_count": row["expected_entity_count"], "candidate_count": len(candidates)}
        for mode, predicted in outputs.items():
            diagnostic = evaluations[mode].add(text, gold, predicted, candidates)
            record[mode] = entities(predicted)
            if mode == "known_count":
                for error in diagnostic:
                    category = error["output_category"]
                    if category == "correct_slot" or example_counts[category] >= 4:
                        continue
                    span = error["gold"]
                    lo, hi = max(0, span[0] - 100), min(len(text), span[1] + 100)
                    examples.append({"document_id": row["document_id"], **error, "gold_text": text[span[0]:span[1]],
                                     "context_start": lo, "context": text[lo:hi],
                                     "predicted_at_gold_slot": predicted[error["gold_slot"] - 1] if error["gold_slot"] <= len(predicted) else None,
                                     "local_predictions": entities([p for p in predicted if overlap(p, (lo, hi, ""))]),
                                     "local_candidates": [{"start": s[0], "end": s[1], "label": s[2], "score": score} for s, score in sorted(candidates.items()) if overlap(s, (lo, hi, ""))]})
                    example_counts[category] += 1
        predictions.append(record)
        if i % 1000 == 0:
            print(f"Evaluated {i} dev documents", flush=True)
    pq.write_table(pa.Table.from_pylist(predictions), out / "dev_predictions.parquet", compression="zstd")
    (out / "error_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("Inputs/code changed during M0 run")
    report = {"train_documents": train_count, "dev_documents": len(dev), "dictionary_phrases": len(dictionary),
              "candidate_total": candidate_total, "elapsed_seconds": time.monotonic() - start,
              "results": {k: v.report() for k, v in evaluations.items()}, "sources": hashes,
              "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
              "limitations": ["Dev evaluation, not blind holdout; no dev hyperparameter tuning in this run.",
                              "Fixed regex scores are heuristic, not calibrated probabilities.", "Known-count missing slots are errors, not fabricated submission values.",
                              "Shared-template generalization remains unresolved; no neural model/tokenizer used."]}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({mode: {k: v for k, v in value.items() if k in ("slot_accuracy", "entity_micro", "candidate_recall", "counts")} for mode, value in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
