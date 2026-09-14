# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1", "scikit-learn==1.9.1"]
# ///
"""From-scratch contextual candidate classifier; no pretrained weights/corpora."""
import argparse
import collections
import hashlib
import json
import math
import pickle
import platform
import random
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from baseline_probe import decode
from ner_evaluation import Evaluation, triples
from prepared_loader import PreparedData
from run_m0 import candidates_for
from submit_m0 import load_dictionary, format_prediction, validate_rows

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
EPOCHS = 2
FEATURES = 2 ** 20
TITLE = re.compile(r"^(?:Mr|Mrs|Ms|Miss|Dr|Prof|Mx)\.?\s+")
USERNAME_CONTEXT = re.compile(r"\b(?:username|user\s+name|user\s+id|login\s+id)\s*(?:is\b|was\b|:|=)?\s*['\"`]?([A-Za-z0-9][A-Za-z0-9_.@-]{2,63})", re.I)
STOP_USER = {"is", "was", "for", "and", "the", "your", "will", "should", "has", "can"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def propose(text, trie, confidence):
    result = candidates_for(text, trie, confidence)
    for m in USERNAME_CONTEXT.finditer(text):
        a, b = m.span(1)
        while b > a and text[b - 1] in ".,;:":
            b -= 1
        if b > a and text[a:b].lower() not in STOP_USER:
            result.setdefault((a, b, "USERNAME"), .5)
    return result


def features(text, span, prior):
    a, b, label = span
    value = text[a:b]
    lower = value.lower()
    left, right = text[max(0, a - 100):a], text[b:b + 70]
    left_words = re.findall(r"[\w@.-]+|[^\w\s]", left.lower())[-5:]
    right_words = re.findall(r"[\w@.-]+|[^\w\s]", right.lower())[:4]
    prefix = label + "|"
    f = {prefix + "bias": 1., prefix + "prior_logit": math.log(max(prior, 1e-6) / max(1 - prior, 1e-6)) / 4,
         prefix + "chars": min(len(value), 150) / 30, prefix + "words": min(len(value.split()), 15) / 3,
         prefix + "position": a / max(1, len(text)), prefix + "newlines": min(value.count("\n"), 3),
         prefix + "starts_title": float(bool(TITLE.match(value))), prefix + "starts_space": float(value[0].isspace()),
         prefix + "ends_space": float(value[-1].isspace()), prefix + "has_digit": float(any(c.isdigit() for c in value)),
         prefix + "has_underscore": float("_" in value), prefix + "has_at": float("@" in value),
         prefix + "all_lower": float(value.islower()), prefix + "all_upper": float(value.isupper()),
         prefix + "title_case": float(value.istitle()),
         prefix + "left_identifier": float(a > 0 and (text[a - 1].isalnum() or text[a - 1] == "_")),
         prefix + "right_identifier": float(b < len(text) and (text[b].isalnum() or text[b] == "_")),
         prefix + "left_title": float(bool(re.search(r"\b(?:mr|mrs|ms|dr|prof)\.?\s*$", left, re.I)))}
    # All lexical weights are learned from competition train, no external lexicon.
    f[prefix + "phrase=" + lower] = 1.
    f[prefix + "first=" + lower.split()[0] if lower.split() else prefix + "empty"] = 1.
    f[prefix + "last=" + lower.split()[-1] if lower.split() else prefix + "empty"] = 1.
    for i, word in enumerate(reversed(left_words)):
        f[prefix + f"L{i}=" + word] = 1.
    for i, word in enumerate(right_words):
        f[prefix + f"R{i}=" + word] = 1.
    f[prefix + "Lbigram=" + " ".join(left_words[-2:])] = 1.
    f[prefix + "Rbigram=" + " ".join(right_words[:2])] = 1.
    f[prefix + "left_char=" + (text[a - 1] if a else "BOS")] = 1.
    f[prefix + "right_char=" + (text[b] if b < len(text) else "EOS")] = 1.
    return f


def score(text, candidates, model, hasher):
    spans = sorted(candidates)
    if not spans:
        return {}
    matrix = hasher.transform(features(text, s, candidates[s]) for s in spans)
    probabilities = model.predict_proba(matrix)[:, 1]
    return dict(zip(spans, map(float, probabilities)))


def model_sources():
    paths = ["src/train_m1.py", "src/train_m1.py.lock", "src/run_m0.py", "src/baseline_probe.py", "src/ner_evaluation.py", "src/prepared_loader.py", "src/submit_m0.py",
             "output/m0/v1/dictionary.parquet", "output/prepared/v1/train.parquet", "output/prepared/v1/dev.parquet", "note/M1_PROTOCOL.md"]
    return {p: sha(ROOT / p) for p in paths}


def train(out):
    if out.exists():
        raise ValueError("Existing M1 run is immutable; choose a new --output")
    out.mkdir(parents=True)
    source_hashes = model_sources()
    config = {"method": "binary logistic SGD contextual span candidate classifier", "pretrained": False,
              "seed": SEED, "epochs": EPOCHS, "hash_features": FEATURES, "alpha": 1e-6,
              "learning_rate": "constant", "eta0": .02, "average": True, "batch_documents": 256,
              "candidate_pool": "frozen M0 train dictionary/regex plus fixed username context regex",
              "training": "all v1 train candidates, gold exact triple positives, all other candidates negatives",
              "dev_tuning": False, "runtime": {"python": platform.python_version(), "pyarrow": pa.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
              "sources": source_hashes}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data = PreparedData()
    rows = data.load_train().to_pylist()
    trie, confidence = load_dictionary(ROOT / "output/m0/v1/dictionary.parquet")
    hasher = FeatureHasher(n_features=FEATURES, input_type="dict", alternate_sign=True)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-6, learning_rate="constant", eta0=.02,
                          average=True, random_state=SEED, shuffle=True)
    start = time.monotonic()
    epochs = []
    for epoch in range(EPOCHS):
        order = list(range(len(rows)))
        random.Random(SEED + epoch).shuffle(order)
        candidate_count = positives = 0
        for offset in range(0, len(order), 256):
            vectors, y = [], []
            for index in order[offset:offset + 256]:
                row = rows[index]
                text = row["full_text"]
                gold = set(triples(row["entities"]))
                candidates = propose(text, trie, confidence)
                for span in sorted(candidates):
                    vectors.append(features(text, span, candidates[span]))
                    y.append(int(span in gold))
            model.partial_fit(hasher.transform(vectors), np.asarray(y), classes=np.array([0, 1]))
            candidate_count += len(y)
            positives += sum(y)
            if offset % (256 * 40) == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, train docs {min(offset + 256, len(rows))}/{len(rows)}", flush=True)
        epochs.append({"epoch": epoch + 1, "candidates": candidate_count, "positives": positives})
    train_documents = len(rows)
    del rows
    with (out / "model.pkl").open("wb") as stream:
        pickle.dump({"model": model, "hasher": hasher}, stream, protocol=5)
    evaluator = Evaluation()
    predictions = []
    for row in data.load_dev("evaluation").to_pylist():
        text = row["full_text"]
        candidates = propose(text, trie, confidence)
        scored = score(text, candidates, model, hasher)
        predicted = decode(scored, row["expected_entity_count"])
        evaluator.add(text, triples(row["entities"]), predicted, candidates)
        predictions.append({"document_id": row["document_id"], "predicted": [{"start": a, "end": b, "label": label} for a, b, label in predicted]})
    pq.write_table(pa.Table.from_pylist(predictions), out / "dev_predictions.parquet", compression="zstd")
    if model_sources() != source_hashes:
        raise ValueError("Source changed during M1 training")
    report = {"train_documents": train_documents, "epochs": epochs, "seconds": time.monotonic() - start,
              "dev": evaluator.report(), "pretrained": False,
              "model_sha256": sha(out / "model.pkl"), "prediction_sha256": sha(out / "dev_predictions.parquet")}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"dev_slot_accuracy": report["dev"]["slot_accuracy"], "entity_micro": report["dev"]["entity_micro"], "candidate_recall": report["dev"]["candidate_recall"]}, indent=2))


def submit_file(out):
    report = json.loads((out / "report.json").read_text())
    config = json.loads((out / "config.json").read_text())
    if model_sources() != config["sources"] or sha(out / "model.pkl") != report["model_sha256"]:
        raise ValueError("Model/source hash mismatch")
    target = out / "submission.csv"
    if target.exists():
        raise ValueError("Submission file already exists")
    # Only load a local pickle whose hash matches this run's trusted report.
    with (out / "model.pkl").open("rb") as stream:
        fitted = pickle.load(stream)
    trie, confidence = load_dictionary(ROOT / "output/m0/v1/dictionary.parquet")
    data = PreparedData()
    test = data.load_test().to_pylist()
    slots = data.load_test_slots().to_pylist()
    by_doc = {}
    for i, row in enumerate(test, 1):
        text = row["full_text"]
        spans = decode(score(text, propose(text, trie, confidence), fitted["model"], fitted["hasher"]), row["expected_entity_count"])
        if len(spans) != row["expected_entity_count"]:
            raise ValueError("Insufficient candidates for exposed K; no placeholder spans generated")
        by_doc[row["document_id"]] = [format_prediction(s) for s in spans]
        if i % 5000 == 0:
            print(f"Predicted {i} test documents", flush=True)
    final = [{"row_id": row["row_id"], "document_id": row["document_id"], "Predicted": by_doc[row["document_id"]][row["slot"] - 1]} for row in slots]
    validate_rows(final, {r["document_id"]: r["full_text"] for r in test}, {r["document_id"]: r["expected_entity_count"] for r in test})
    import csv
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "Predicted"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final)
    (out / "submission_manifest.json").write_text(json.dumps({"rows": len(final), "sha256": sha(target), "model_sha256": report["model_sha256"], "test_hash": sha(ROOT / "output/prepared/v1/test.parquet"), "slots_hash": sha(ROOT / "output/prepared/v1/test_slots.parquet"), "pretrained": False}, indent=2), encoding="utf-8")
    print(json.dumps({"submission": str(target), "rows": len(final)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/m1/v1")
    parser.add_argument("--submission-only", action="store_true")
    args = parser.parse_args()
    submit_file(args.output.resolve()) if args.submission_only else train(args.output.resolve())
