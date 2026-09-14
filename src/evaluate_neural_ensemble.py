"""Fixed equal-logit average of scratch checkpoints 4 and 8."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from prepared_loader import PreparedData
from neural_sequence import build_model, encode_document
from run_neural_sequence import evaluate, sha, source_hashes

ROOT = Path(__file__).resolve().parents[1]

def main():
    run = ROOT / 'output/bitrase-1.algo-6/v1'
    out = ROOT / 'output/bitrase-1.algo-6/ensemble_4_8'
    if out.exists():
        raise ValueError('Preserve previous experiment')
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    if source_hashes() != config['sources']:
        raise ValueError('Source mismatch')
    vocab = json.loads((run / 'vocab.json').read_text(encoding='utf-8'))
    if sha(run / 'vocab.json') != report['vocab_sha256']:
        raise ValueError('Vocabulary mismatch')
    models = []
    for epoch in (4, 8):
        checkpoint = run / f'epoch{epoch}.pt'
        if sha(checkpoint) != report['epochs'][str(epoch)]['model_sha256']:
            raise ValueError('Checkpoint mismatch')
        model = build_model(len(vocab)).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        models.append(model)
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.models = torch.nn.ModuleList(models)
        def forward(self, *args):
            return sum(m(*args).float() for m in self.models) / len(self.models)
    out.mkdir(parents=True)
    provenance = {'epochs': [4, 8], 'weights': [0.5, 0.5], 'combination': 'logits',
                  'source_sha256': sha(Path(__file__)), 'parent_report_sha256': sha(run / 'report.json')}
    (out / 'config.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    torch.set_num_threads(4)
    dev = PreparedData().load_dev('evaluation').to_pylist()
    encoded = []
    for row in dev:
        ts, ids, chars, flags = encode_document(row['full_text'], vocab)
        encoded.append((ts, ids, chars, flags, np.full(len(ids), -100, dtype=np.int64)))
    metrics, predictions = evaluate(Ensemble(), dev, encoded, config['batch_size'])
    for method, rows in predictions.items():
        pq.write_table(pa.Table.from_pylist(rows), out / f'{method}_dev_predictions.parquet', compression='zstd')
    (out / 'report.json').write_text(json.dumps({'dev': metrics, 'provenance': provenance}, indent=2), encoding='utf-8')
    print(json.dumps({m: r['slot_accuracy'] for m, r in metrics.items()}), flush=True)

if __name__ == '__main__':
    main()
