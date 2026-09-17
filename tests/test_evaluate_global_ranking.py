import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.evaluate_global_ranking import _local_summary, run


def write_split(root: Path, name: str, scores: list[float]) -> None:
    root.mkdir(parents=True)
    pairs = np.array([(0, 1), (0, 2), (1, 2)], dtype=np.int32)
    logits = np.zeros((3, 8), dtype=np.float32)
    # Gold span (0,1) has NAME index 0; all alternatives score lower.
    logits[0, 1] = scores[0]
    logits[1, 1] = scores[1]
    logits[2, 1] = scores[2]
    bounds = np.array([[0, 1], [2, 3]], dtype=np.int32)
    np.save(root / 'logits.npy', logits)
    np.save(root / 'pairs.npy', pairs)
    np.save(root / 'span_offsets.npy', np.array([0, 3], dtype=np.int64))
    np.save(root / 'token_offsets.npy', np.array([0, 2], dtype=np.int64))
    np.save(root / 'boundaries.npy', bounds)
    (root / 'metadata.json').write_text(json.dumps([{
        'document_id': name, 'group_id': f'family-{name}', 'expectedK': 1,
        'token_count': 2, 'gold': [{'start': 0, 'end': 1, 'label': 'NAME'}],
    }]), encoding='utf-8')
    names = ('logits.npy', 'pairs.npy', 'span_offsets.npy', 'token_offsets.npy', 'boundaries.npy', 'metadata.json')
    files = [{'name': item, 'sha256': hashlib.sha256((root / item).read_bytes()).hexdigest()} for item in names]
    (root / 'manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')


class EvaluateGlobalRankingTests(unittest.TestCase):
    def test_local_gap_counts_use_the_actual_gap_field(self):
        rows = [{'local_span_ranking': {'control': [
            {'supported': True, 'competitors': 2, 'gold_minus_alternative_gap': 0.5},
            {'supported': True, 'competitors': 1, 'gold_minus_alternative_gap': -0.25},
            {'supported': True, 'competitors': 1, 'gold_minus_alternative_gap': 0.0},
            {'supported': True, 'competitors': 0, 'gold_minus_alternative_gap': None},
            {'supported': False, 'competitors': 0},
        ]}}]
        summary = _local_summary(rows, 'control')
        self.assertEqual(summary['gold_strict_win'], 1)
        self.assertEqual(summary['gold_strict_loss'], 1)
        self.assertEqual(summary['tie'], 1)
        self.assertEqual(summary['slots_with_feasible_alternatives'], 3)

    def test_paired_eval_runs_remines_and_writes_family_bootstrap(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for arm, values in (('control', [3, 0, 0]), ('challenger', [4, 0, 0])):
                for idx in range(2):
                    write_split(tmp / arm / f'doc{idx}', f'doc{idx}', values)
            # Split caches are represented as one multi-document cache in production;
            # merge synthetic tiny cache arrays to exercise that exact loader format.
            for arm in ('control', 'challenger'):
                base = tmp / arm
                docs = [base / f'doc{i}' for i in range(2)]
                logits = np.concatenate([np.load(d / 'logits.npy') for d in docs])
                pairs = np.concatenate([np.load(d / 'pairs.npy') for d in docs])
                bounds = np.concatenate([np.load(d / 'boundaries.npy') for d in docs])
                np.save(base / 'logits.npy', logits); np.save(base / 'pairs.npy', pairs)
                np.save(base / 'span_offsets.npy', np.array([0, 3, 6], dtype=np.int64))
                np.save(base / 'token_offsets.npy', np.array([0, 2, 4], dtype=np.int64))
                np.save(base / 'boundaries.npy', bounds)
                metadata = [json.loads((d / 'metadata.json').read_text())[0] for d in docs]
                (base / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
                names = ('logits.npy', 'pairs.npy', 'span_offsets.npy', 'token_offsets.npy', 'boundaries.npy', 'metadata.json')
                files = [{'name': item, 'sha256': hashlib.sha256((base / item).read_bytes()).hexdigest()} for item in names]
                (base / 'manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')
            report = run(tmp / 'control', tmp / 'challenger', tmp / 'out', workers=4)
            self.assertEqual(report['eval_docs'], 2)
            self.assertEqual(report['paired_family_bootstrap']['map']['replicates'], 2000)
            self.assertEqual(report['metrics']['map']['control']['correct_slots'], 2)
            self.assertTrue((tmp / 'out' / 'eval_documents.json').is_file())
            saved = json.loads((tmp / 'out' / 'eval_documents.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved[0]['map_control']), 1)
            self.assertEqual(len(saved[0]['mbr_challenger']), 1)

    def test_family_bootstrap_resamples_clusters_not_rows(self):
        from src.evaluate_global_ranking import _family_bootstrap
        rows = [
            {'group_id': 'a', 'K': 1, 'map_correct_control': 0, 'map_correct_challenger': 1},
            {'group_id': 'a', 'K': 1, 'map_correct_control': 0, 'map_correct_challenger': 0},
            {'group_id': 'b', 'K': 1, 'map_correct_control': 1, 'map_correct_challenger': 1},
        ]
        first = _family_bootstrap(rows, 'map')
        second = _family_bootstrap(rows, 'map')
        self.assertEqual(first, second)
        self.assertEqual(first['families'], 2)
        self.assertEqual(first['replicates'], 2000)


if __name__ == '__main__':
    unittest.main()
