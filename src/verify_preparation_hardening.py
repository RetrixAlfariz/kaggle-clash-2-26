"""Record reproducible integration checks; run with uv run --frozen python."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepared_loader import PreparedData, PreparedDataError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/hardening_validation"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    v1 = ROOT / "output/prepared/v1"
    new = OUT / "build"
    manifest = json.loads((v1 / "manifest.json").read_text())
    original_audit = json.loads((ROOT / "output/preparation_audit/data_checks.json").read_text())
    output_hashes = {p: {"v1_unchanged": sha(v1 / p) == info["sha256"], "hardened_bytes_equal_v1": sha(v1 / p) == sha(new / p)} for p, info in manifest["files"].items()}
    manifest_unchanged = sha(v1 / "manifest.json") == original_audit["build_manifest_sha256"]
    tests = {}
    for mode, flags in (("normal", []), ("optimized", ["-O"])):
        proc = subprocess.run([sys.executable, *flags, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, text=True, capture_output=True)
        tests[mode] = {"returncode": proc.returncode, "output": proc.stdout + proc.stderr}
    loader = PreparedData(v1)
    reads = {name: getattr(loader, f"load_{name}")().num_rows for name in ("train", "dev", "test", "test_slots")}
    try:
        loader.load_holdout()
        holdout_blocked = False
    except PreparedDataError:
        holdout_blocked = True
    audits = {name: json.loads((OUT / name / "data_checks.json").read_text())["failed_checks"] for name in ("audit_v1", "audit_build")}
    passed = manifest_unchanged and all(all(v.values()) for v in output_hashes.values()) and all(t["returncode"] == 0 for t in tests.values()) and holdout_blocked and not any(audits.values())
    report = {"passed": passed, "v1_manifest_unchanged": manifest_unchanged, "artifacts": output_hashes, "unit_tests": tests, "loader_actual_rows": reads, "default_holdout_blocked": holdout_blocked, "independent_audit_failed_checks": audits, "code_hashes": {p: sha(ROOT / p) for p in ("src/prepare_data.py", "src/prepared_loader.py", "src/prepare_data.py.lock", "pyproject.toml", "uv.lock", ".python-version")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"unit_tests", "code_hashes"}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
