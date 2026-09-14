# /// script
# requires-python = "==3.13.2"
# dependencies = ["lightgbm==4.6.0", "pyarrow==25.0.1", "scikit-learn==1.9.1"]
# ///
"""bitrase-1.algo-2: from-scratch boosted candidate scorer.

The candidate pool and split are inherited from M1.  LightGBM is used only as
a supervised tabular classifier; no pretrained model or external corpus is
loaded.
"""
import hashlib, json, math, pickle, random, re, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd() / "src"))

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ner_evaluation import Evaluation, triples
from prepared_loader import PreparedData
from train_m1 import propose, sha
from slot_decoder import decode
from submit_m0 import load_dictionary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/bitrase-1.algo-2/v1"
SEED = 2026
MAX_NEGATIVES = 30
MAX_TRAIN_DOCUMENTS = 12000
LABELS = {x: i for i, x in enumerate(("NAME", "DATE", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "JOB_TITLE"))}
WORD = re.compile(r"[\w@.-]+", re.UNICODE)
TITLE = re.compile(r"\b(?:mr|mrs|ms|miss|dr|prof|mx)\.?\s*$", re.I)


def _token(text, at, direction):
    words = list(WORD.finditer(text))
    if direction < 0:
        vals = [m.group(0).lower() for m in words if m.end() <= at]
        return vals[-1] if vals else ""
    vals = [m.group(0).lower() for m in words if m.start() >= at]
    return vals[0] if vals else ""


def dense_features(text, span, prior, frequencies=None):
    a, b, label = span
    value = text[a:b]
    low = value.lower()
    left = text[max(0, a - 100):a]
    right = text[b:b + 100]
    first = low.split()[0] if low.split() else ""
    last = low.split()[-1] if low.split() else ""
    f = [LABELS[label], math.log(max(prior, 1e-6) / max(1 - prior, 1e-6)),
         min(len(value), 200), min(len(value.split()), 20), a / max(1, len(text)),
         b / max(1, len(text)), float("\n" in value), float(any(c.isdigit() for c in value)),
         float("_" in value), float("@" in value), float(value.islower()), float(value.isupper()),
         float(value.istitle()), float(a > 0 and (text[a-1].isalnum() or text[a-1] == "_")),
         float(b < len(text) and (text[b].isalnum() or text[b] == "_")),
         float(bool(TITLE.search(left))), float(bool(re.search(r"\b(?:username|user\s+name|user\s+id|login\s+id)\b", left, re.I))),
         float(bool(re.search(r"\b(?:email|e-mail|phone|mobile|address|date|job|title)\b", left, re.I))),
         float(bool(re.search(r"[:=]\s*$", left))),
         math.log1p((frequencies or {}).get("p:" + low, 0)),
         math.log1p((frequencies or {}).get("f:" + first, 0)),
         math.log1p((frequencies or {}).get("l:" + last, 0)),
         math.log1p((frequencies or {}).get("L:" + _token(text, a, -1), 0)),
         math.log1p((frequencies or {}).get("R:" + _token(text, b, 1), 0))]
    # A small fixed vocabulary of context indicators is learned only as
    # binary features, avoiding any external lexical resource.
    for w in ("name", "named", "email", "phone", "mobile", "address", "date", "username", "user", "login", "job", "title", "born", "called"):
        f.extend((float(bool(re.search(r"\b" + w + r"\b", left, re.I))), float(bool(re.search(r"\b" + w + r"\b", right, re.I)))))
    return f


def build_train(rows, trie, confidence):
    frequencies = {}
    cached = []
    for row in rows:
        pool = propose(row["full_text"], trie, confidence)
        gold = set(triples(row["entities"]))
        cached.append((row, pool, gold))
        for a, b, label in pool:
            value = row["full_text"][a:b].lower()
            for key in ("p:" + value, "f:" + (value.split()[0] if value.split() else ""), "l:" + (value.split()[-1] if value.split() else "")):
                frequencies[key] = frequencies.get(key, 0) + 1
    xs, ys = [], []
    for row, pool, gold in cached:
        positives = [s for s in sorted(pool) if s in gold]
        negatives = [s for s in sorted(pool) if s not in gold]
        rng = random.Random(SEED + int(hashlib.sha1(row["document_id"].encode()).hexdigest()[:8], 16))
        rng.shuffle(negatives)
        keep = positives + negatives[:MAX_NEGATIVES]
        for span in keep:
            xs.append(dense_features(row["full_text"], span, pool[span], frequencies))
            ys.append(int(span in gold))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8), frequencies


def evaluate_model(model, frequencies, trie, confidence):
    ev = Evaluation(); predictions = []
    for row in PreparedData().load_dev("evaluation").to_pylist():
        text = row["full_text"]; pool = propose(text, trie, confidence); spans = sorted(pool)
        x = np.asarray([dense_features(text, s, pool[s], frequencies) for s in spans], dtype=np.float32)
        probs = model.booster_.predict(x) if spans else []
        pred = decode(dict(zip(spans, map(float, probs))), row["expected_entity_count"], "slot_mbr")
        ev.add(text, triples(row["entities"]), pred, pool)
        predictions.append({"document_id": row["document_id"], "predicted": [{"start": a, "end": b, "label": l} for a,b,l in pred]})
    return ev.report(), predictions


def main():
    if OUT.exists(): raise ValueError("Run exists; experiment is immutable")
    out = OUT; out.mkdir(parents=True)
    protocol = {"experiment": "bitrase-1.algo-2", "parent": "bitrase-1", "method": "LightGBM binary boosted trees from scratch", "pretrained": False, "candidate_pool": "frozen M1 propose", "negative_cap_per_document": MAX_NEGATIVES, "train_document_cap": MAX_TRAIN_DOCUMENTS, "trees": 350, "seed": SEED, "dev_tuning": False, "sources": {p: sha(ROOT/p) for p in ("src/run_boosted_candidates.py", "src/train_m1.py", "src/run_m0.py", "src/slot_decoder.py", "output/m0/v1/dictionary.parquet", "output/prepared/v1/train.parquet", "output/prepared/v1/dev.parquet")}}
    (out / "config.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    data = PreparedData(); train = data.load_train().to_pylist(); train = sorted(train, key=lambda r: hashlib.sha1(r["document_id"].encode()).hexdigest())[:MAX_TRAIN_DOCUMENTS]; trie, confidence = load_dictionary(ROOT / "output/m0/v1/dictionary.parquet")
    x, y, frequencies = build_train(train, trie, confidence)
    model = lgb.LGBMClassifier(n_estimators=350, learning_rate=.035, num_leaves=31, max_depth=8, min_child_samples=30, subsample=.85, colsample_bytree=.9, reg_lambda=2., random_state=SEED, n_jobs=8, verbosity=-1)
    start = time.monotonic(); model.fit(x, y)
    with (out / "model.pkl").open("wb") as f: pickle.dump({"model": model, "frequencies": frequencies}, f, protocol=5)
    report, predictions = evaluate_model(model, frequencies, trie, confidence)
    pq.write_table(pa.Table.from_pylist(predictions), out / "dev_predictions.parquet", compression="zstd")
    result = {"config": protocol, "train_candidates": int(len(y)), "train_positives": int(y.sum()), "dev": report, "seconds": time.monotonic()-start, "model_sha256": sha(out/"model.pkl"), "prediction_sha256": sha(out/"dev_predictions.parquet"), "pretrained": False}
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"slot_accuracy": report["slot_accuracy"], "entity_f1": report["entity_micro"]["f1"], "candidates": len(y)}, indent=2))


if __name__ == "__main__": main()
