"""Paired EVAL2 evaluation for frozen control/challenger span-score caches.

The evaluator re-mines each final cache at evaluation time, decodes exact-K
MAP and Slot-MBR (tau=1), and reports paired document/family-bootstrap gains.
It does not load source data or train models.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .scale_ranking_audit import LABELS, _to_char, load_split, local_span_ranking, ranking_record, sha256
    from .slot_diagnostics import compare
    from .slot_mbr import decode
except ImportError:
    from scale_ranking_audit import LABELS, _to_char, load_split, local_span_ranking, ranking_record, sha256
    from slot_diagnostics import compare
    from slot_mbr import decode

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260919
MIN_GAIN = 0.002  # +0.20 percentage points


def _ordered_correct(gold: tuple, pred: tuple) -> int:
    return sum(j < len(pred) and tuple(g) == tuple(pred[j]) for j, g in enumerate(gold))


def _named(rows: tuple) -> tuple:
    return tuple((int(a), int(b), LABELS[int(label)]) for a, b, label in rows)


def _family_bootstrap(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Cluster bootstrap families; K-weighted pooled slot accuracy difference."""
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        families.setdefault(row['group_id'], []).append(row)
    keys = sorted(families)
    if not keys:
        return {'families': 0, 'docs': 0, 'gain': None, 'ci95': None,
                'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
                'minimum_gain': MIN_GAIN, 'gate_pass': False}
    numerator = {key: sum(r[f'{metric}_correct_challenger'] - r[f'{metric}_correct_control'] for r in families[key]) for key in keys}
    denominator = {key: sum(r['K'] for r in families[key]) for key in keys}
    total_n = sum(numerator.values()); total_d = sum(denominator.values())
    gain = total_n / total_d if total_d else None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(keys), size=len(keys))
        ns = sum(numerator[keys[j]] for j in chosen)
        ds = sum(denominator[keys[j]] for j in chosen)
        samples[i] = ns / ds if ds else 0.0
    ci = [float(v) for v in np.quantile(samples, [0.025, 0.975])]
    return {'families': len(keys), 'docs': len(rows), 'gain': float(gain),
            'gain_percentage_points': float(gain * 100), 'ci95': ci,
            'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
            'minimum_gain': MIN_GAIN,
            'gate_pass': bool(gain >= MIN_GAIN and ci[0] > 0)}


def _evaluate_doc(control: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    if (control['document_id'], control['n'], control['k'], control['gold_char'], control['group_id']) != \
       (challenger['document_id'], challenger['n'], challenger['k'], challenger['gold_char'], challenger['group_id']):
        raise ValueError(f"control/challenger metadata mismatch for {control.get('document_id')}")
    if not np.array_equal(control['pairs'], challenger['pairs']) or not np.array_equal(control['boundaries'], challenger['boundaries']):
        raise ValueError(f"control/challenger candidate/token-boundary mismatch for {control['document_id']}")
    predictions: dict[str, dict[str, tuple]] = {}
    rankings: dict[str, dict[str, Any]] = {}
    locals_: dict[str, list] = {}
    for name, doc in (('control', control), ('challenger', challenger)):
        mbr, diag = decode(doc['n'], doc['pairs'], doc['energies'], doc['k'])
        mp = [tuple(x) for x in diag['map_prediction']]
        predictions[name] = {
            'map': _to_char(mp, doc['boundaries']),
            'mbr_tau1': _to_char([tuple(x) for x in mbr], doc['boundaries']),
        }
        rankings[name] = ranking_record(doc['n'], doc['pairs'], doc['energies'], doc['k'], doc['gold'])
        locals_[name] = local_span_ranking(doc['pairs'], doc['energies'], doc['gold']) if doc['supported'] else []
    gold = tuple(sorted(control['gold_char']))
    row: dict[str, Any] = {
        'document_id': control['document_id'], 'group_id': control['group_id'], 'K': control['k'],
        'gold_supported': bool(control['supported'] and challenger['supported']),
        'map_control': predictions['control']['map'], 'map_challenger': predictions['challenger']['map'],
        'mbr_control': predictions['control']['mbr_tau1'], 'mbr_challenger': predictions['challenger']['mbr_tau1'],
        'map_correct_control': _ordered_correct(gold, predictions['control']['map']),
        'map_correct_challenger': _ordered_correct(gold, predictions['challenger']['map']),
        'mbr_correct_control': _ordered_correct(gold, predictions['control']['mbr_tau1']),
        'mbr_correct_challenger': _ordered_correct(gold, predictions['challenger']['mbr_tau1']),
        'gold_slots': gold,
        'ranking': rankings,
        'local_span_ranking': locals_,
        'map_change': compare(_named(gold), _named(predictions['control']['map']), _named(predictions['challenger']['map'])),
        'mbr_change': compare(_named(gold), _named(predictions['control']['mbr_tau1']), _named(predictions['challenger']['mbr_tau1'])),
    }
    return row


def _entity_f1(rows: list[dict[str, Any]], decoder: str, arm: str) -> dict[str, Any]:
    tp = pred_n = gold_n = 0
    for row in rows:
        gold = set(map(tuple, row['gold_slots']))
        pred = set(map(tuple, row[f'{decoder}_{arm}']))
        tp += len(gold & pred); pred_n += len(pred); gold_n += len(gold)
    precision = tp / pred_n if pred_n else (1.0 if not gold_n else 0.0)
    recall = tp / gold_n if gold_n else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'true_positive': tp, 'predicted': pred_n, 'gold': gold_n,
            'precision': precision, 'recall': recall, 'f1': f1}


def _gap_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    supported = [r['ranking'][arm] for r in rows if r['ranking'][arm]['gold_supported']]
    gaps = [r['gold_minus_wrong_gap'] for r in supported if r['gold_minus_wrong_gap'] is not None]
    return {'supported_docs': len(supported),
            'map_gold_docs': sum(bool(r['map_is_gold']) for r in supported),
            'strict_win': sum(x > 1e-12 for x in gaps),
            'tie': sum(abs(x) <= 1e-12 for x in gaps),
            'loss': sum(x < -1e-12 for x in gaps),
            'mean_gold_minus_strongest_wrong_gap': float(np.mean(gaps)) if gaps else None}


def _local_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    slots = [slot for row in rows for slot in row['local_span_ranking'][arm]]
    supported = [slot for slot in slots if slot.get('supported')]
    compared = [slot for slot in supported if slot.get('competitors', 0) > 0 and
                slot.get('gold_minus_alternative_gap') is not None]
    gaps = [slot['gold_minus_alternative_gap'] for slot in compared]
    return {
        'slots': len(slots), 'supported_slots': len(supported),
        'slots_with_feasible_alternatives': len(compared),
        'gold_strict_win': sum(x > 1e-12 for x in gaps),
        'tie': sum(abs(x) <= 1e-12 for x in gaps),
        'gold_strict_loss': sum(x < -1e-12 for x in gaps),
        'mean_gold_minus_strongest_alternative_gap': float(np.mean(gaps)) if gaps else None,
    }


def run(control_path: Path, challenger_path: Path, out_path: Path, workers: int = 4) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1]
    source_paths = ('src/evaluate_global_ranking.py', 'src/scale_ranking_audit.py',
                    'src/slot_mbr.py', 'src/slot_diagnostics.py')
    source_hashes = {path: sha256(source_root / path) for path in source_paths}
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f'refusing existing output directory: {out_path}')
    control, control_hashes = load_split(control_path)
    challenger, challenger_hashes = load_split(challenger_path)
    if len(control) != len(challenger):
        raise ValueError('control/challenger document counts differ')
    if any(not isinstance(d['group_id'], str) or not d['group_id'] for d in control + challenger):
        raise ValueError('every EVAL2 document must have a nonempty group_id')
    if workers < 1:
        raise ValueError('workers must be positive')
    by_id = {d['document_id']: d for d in challenger}
    if len(by_id) != len(challenger) or {d['document_id'] for d in control} != set(by_id):
        raise ValueError('control/challenger document ID sets differ')
    pairs = [(d, by_id[d['document_id']]) for d in control]
    rows: list[dict[str, Any] | None] = [None] * len(pairs)
    if workers == 1:
        for i, pair in enumerate(pairs):
            rows[i] = _evaluate_doc(*pair)
            if (i + 1) % 100 == 0:
                print(f'EVAL decoded {i + 1}/{len(pairs)}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {}; next_i = 0; done_count = 0; limit = workers * 2
            while next_i < len(pairs) or pending:
                while next_i < len(pairs) and len(pending) < limit:
                    fut = pool.submit(_evaluate_doc, *pairs[next_i]); pending[fut] = next_i; next_i += 1
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    ix = pending.pop(fut); rows[ix] = fut.result(); done_count += 1
                    if done_count % 100 == 0:
                        print(f'EVAL decoded {done_count}/{len(pairs)}', flush=True)
    rows = [row for row in rows if row is not None]
    total_slots = sum(r['K'] for r in rows)
    if total_slots <= 0:
        raise ValueError('EVAL2 has no gold slots')
    map_boot = _family_bootstrap(rows, 'map')
    mbr_boot = _family_bootstrap(rows, 'mbr')
    totals = {}
    for decoder in ('map', 'mbr'):
        totals[decoder] = {}
        for arm in ('control', 'challenger'):
            correct = sum(r[f'{decoder}_correct_{arm}'] for r in rows)
            totals[decoder][arm] = {'correct_slots': correct, 'all_slots': total_slots,
                                    'accuracy': correct / total_slots,
                                    'entity_f1': _entity_f1(rows, decoder, arm)}
    report = {
        'schema': 'bitrase-2.algo-4.global-ranking-eval.v1',
        'eval_docs': len(rows), 'eval_slots': total_slots,
        'families': len({r['group_id'] for r in rows}),
        'cache_hashes': {'control': control_hashes, 'challenger': challenger_hashes},
        'metrics': totals,
        'paired_family_bootstrap': {'map': map_boot, 'mbr_tau1': mbr_boot},
        'gates': {'map_pass': map_boot['gate_pass'], 'mbr_tau1_pass': mbr_boot['gate_pass']},
        'gold_vs_strongest_wrong': {'control': _gap_summary(rows, 'control'),
                                    'challenger': _gap_summary(rows, 'challenger')},
        'local_span_ranking_supported_docs': sum(bool(r['gold_supported']) for r in rows),
        'local_span_ranking': {arm: _local_summary(rows, arm) for arm in ('control', 'challenger')},
        'bootstrap': {'unit': 'whole group_id family, sampled with replacement; singleton documents form singleton clusters',
                      'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
                      'gate': 'gain >= 0.002 slot accuracy (+0.20 percentage points) and bootstrap 95% lower bound > 0'},
        'claims_limit': 'One paired EVAL2 comparison of the frozen final heads. Gains are descriptive for this split and do not guarantee external Kaggle performance.',
        'source_hashes': source_hashes,
    }
    after = {path: sha256(source_root / path) for path in source_paths}
    if after != source_hashes:
        raise RuntimeError('evaluator/decoder source changed during evaluation')
    out_path.mkdir(parents=True)
    report['source_hashes_after'] = after
    (out_path / 'eval_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (out_path / 'eval_documents.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--challenger', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.control, args.challenger, args.out, args.workers), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
