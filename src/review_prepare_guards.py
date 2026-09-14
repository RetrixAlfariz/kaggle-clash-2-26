"""Small isolated fault fixtures for current builder validation; no data edits."""
import ast
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "output/preparation_audit/source_snapshots/prepare_data.py"
expected = json.loads((ROOT / "output/prepared/v1/manifest.json").read_text())["sources"]["src/prepare_data.py"]
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise ValueError("Historical builder snapshot hash mismatch")
tree = ast.parse(source.read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "part" and any(isinstance(x, ast.Name) and x.id == "loaded" for x in ast.walk(n)))
end = next(i for i in range(start + 1, len(main.body)) if isinstance(main.body[i], ast.For))
block = ast.Module(body=main.body[start:end], type_ignores=[])


def run_case(loaded_gold, optimize):
    def read_table(path):
        rows = loaded_gold if path.name in {"train.parquet", "dev.parquet", "holdout.parquet"} else []
        return SimpleNamespace(to_pylist=lambda: rows)
    env = {"pq": SimpleNamespace(read_table=read_table), "staging": Path("synthetic"), "train_text": {"d": "abc"}, "entities": {"d": []}, "partition": {"d": "train"}, "test_rows": [], "slots": [], "weak_rows": [], "totals": {"train": {"documents": 1, "entities": 0}}}
    try:
        exec(compile(block, "builder_readback_fixture", "exec", optimize=optimize), env)
        return "accepted"
    except AssertionError:
        return "rejected"


result = {
    "scope": "Historical v1 builder snapshot verified against v1 manifest; synthetic reader. Does not describe the hardened current builder.",
    "assert_count_in_builder": sum(isinstance(n, ast.Assert) for n in ast.walk(tree)),
    "assert_count_compiled_optimized": "Python compile(optimize=1) removes assert statements",
    "missing_all_gold_rows_normal_mode": run_case([], 0),
    "wrong_gold_text_normal_mode": run_case([{"document_id": "d", "full_text": "wrong", "entities": [], "expected_entity_count": 0}], 0),
    "wrong_gold_text_optimized_mode": run_case([{"document_id": "d", "full_text": "wrong", "entities": [], "expected_entity_count": 0}], 1),
    "interpretation": "Missing rows pass normal readback because no loaded ID-set/count equality is checked. Wrong text is rejected normally but accepted under optimization. These are guard defects, not observed v1 corruption."
}
out = ROOT / "output/preparation_audit/guard_review.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
