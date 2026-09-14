"""Independent saved-prediction verification and boundary diagnostics; dev only."""
import collections
import hashlib
import json
import re
from pathlib import Path
import pyarrow.parquet as pq
from prepared_loader import PreparedData

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/m0/v1"
TITLE = re.compile(r"^(?:Mr|Mrs|Ms|Miss|Dr|Prof|Mx)\.?\s+")


def main():
    report = json.loads((OUT / "report.json").read_text())
    config = json.loads((OUT / "frozen_config.json").read_text())
    dictionary = {r["phrase"]: r["label"] for r in pq.read_table(OUT / "dictionary.parquet").to_pylist()}
    rows = PreparedData().load_dev("evaluation").to_pylist()
    predictions = pq.read_table(OUT / "dev_predictions.parquet").to_pylist()
    by_id = {r["document_id"]: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["document_id"] for r in rows}:
        raise ValueError("Prediction document coverage mismatch")
    counts = {mode: collections.Counter() for mode in ("known_count", "unconstrained")}
    boundary = collections.defaultdict(collections.Counter)
    rank_shifts = collections.Counter()
    title_examples = []
    candidate_by_label = collections.Counter()
    for row in rows:
        text = row["full_text"]
        gold = sorted((e["start"], e["end"], e["label"]) for e in row["entities"])
        regex_matches = {(m.start(), m.end(), label) for label, pattern, _ in config["regex"] for m in re.finditer(pattern, text)}
        for a, b, label in gold:
            left_ok = a == 0 or not (text[a-1].isalnum() and text[a].isalnum())
            right_ok = b == len(text) or not (text[b-1].isalnum() and text[b].isalnum())
            found = ((dictionary.get(text[a:b]) == label and left_ok and right_ok) or (a, b, label) in regex_matches)
            candidate_by_label[label] += found
        for mode in counts:
            pred = [(e["start"], e["end"], e["label"]) for e in by_id[row["document_id"]][mode]]
            if pred != sorted(pred) or len(set(pred)) != len(pred):
                raise ValueError("Invalid serialized predictions")
            c = counts[mode]
            c.update({"gold": len(gold), "predicted": len(pred), "entity_correct": len(set(gold) & set(pred)),
                      "slot_correct": sum(g == p for g, p in zip(gold, pred)),
                      "boundary_correct": len({g[:2] for g in gold} & {p[:2] for p in pred})})
            if mode != "known_count":
                continue
            position = {p: i for i, p in enumerate(pred)}
            for index, g in enumerate(gold):
                if g in position:
                    if position[g] != index:
                        rank_shifts[str(position[g] - index)] += 1
                    continue
                if any(g[:2] == p[:2] for p in pred):
                    continue
                overlaps = [p for p in pred if g[2] == p[2] and g[0] < p[1] and p[0] < g[1]]
                if not overlaps:
                    continue
                # One diagnostic representative, not a one-to-one metric match.
                p = min(overlaps, key=lambda p: (-(min(g[1], p[1]) - max(g[0], p[0])), p))
                kind = "prediction_inside_gold" if g[0] <= p[0] and p[1] <= g[1] else "prediction_contains_gold" if p[0] <= g[0] and g[1] <= p[1] else "crossing_boundaries"
                boundary[g[2]][kind] += 1
                gt, pt = text[g[0]:g[1]], text[p[0]:p[1]]
                if g[2] == "NAME" and gt != pt and TITLE.sub("", gt) == TITLE.sub("", pt):
                    boundary["NAME"]["title_only_difference"] += 1
                    if len(title_examples) < 5:
                        title_examples.append({"document_id": row["document_id"], "gold": gt, "predicted": pt, "gold_span": g, "predicted_span": p})
    mismatches = []
    for mode, c in counts.items():
        for key, actual in c.items():
            if actual != report["results"][mode]["counts"][key]:
                mismatches.append(f"{mode}:{key}")
    for label, actual in candidate_by_label.items():
        if actual != report["results"]["known_count"]["per_label"][label]["candidate_found"]:
            mismatches.append(f"candidate_recall:{label}")
    if sum(rank_shifts.values()) != report["results"]["known_count"]["output_stage"]["exact_span_wrong_slot"]:
        mismatches.append("rank_shift_total")
    for path, expected in {**report["sources"], **{str((OUT / p).relative_to(ROOT)): h for p, h in report["artifacts"].items()}}.items():
        with (ROOT / path).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                mismatches.append(f"hash:{path}")
    result = {"metric_mismatches": mismatches, "dev_documents_checked": len(rows),
              "recomputed_counts": {m: dict(v) for m, v in counts.items()},
              "independent_candidate_found_by_label": dict(candidate_by_label),
              "rank_shift_predicted_index_minus_gold_index": dict(sorted(rank_shifts.items(), key=lambda x: int(x[0]))),
              "boundary_diagnostics": {k: dict(v) for k, v in boundary.items()}, "title_only_examples": title_examples,
              "definitions": {"boundary_representative": "For same-label overlaps with no exact predicted triple/boundaries, choose largest character intersection then tuple order. Diagnostics can be many-to-one.",
                              "title_only_difference": TITLE.pattern, "candidate_check": "Direct dictionary lookup at gold substring plus independent boundary predicate OR exact regex match. Does not call trie matcher."}}
    (OUT / "validation_and_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
