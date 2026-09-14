# /// script
# requires-python = ">=3.13"
# dependencies = ["pyarrow>=18", "numpy", "scipy", "pandas", "matplotlib", "nbformat", "nbclient", "nbconvert", "ipykernel"]
# ///
"""Build and execute the five-stage data-understanding notebook and HTML preview."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/data_understanding"


def main():
    manifests = []
    files = [*sorted((ROOT / "Data").rglob("*")),
             *[ROOT / "src" / n for n in ("audit_reconstruction.py", "audit_annotations.py", "audit_metadata.py", "baseline_probe.py")],
             *[OUT / n for n in ("reconstruction.json", "annotations.json", "annotations_examples.json", "metadata.json")]]
    for path in files:
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            manifests.append({"path": path.relative_to(ROOT).as_posix(), "sha256": h.hexdigest()})
    (OUT / "manifest.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "files": manifests}, indent=2), encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("""# Kaggle Clash 2 — Data understanding

## tl;dr
The five-stage audit covers reconstruction, annotation conventions, label context, metadata associations, and observable train/test distributions. Read [REPORT.md](../../note/REPORT.md) for the integrated findings and decisions. This notebook supplies executed tables, figures, and independent reconciliations; audit scripts retain the full calculations and examples.

## Context & Methods
All input is the supplied local competition corpus. Character offsets use Python Unicode characters with an exclusive end. Gold annotations belong to train only; test slots disclose entity counts, not labels. No training or submission is performed here.

### Key Assumptions
- Gold annotations define the target convention; unusual annotations are not automatically mistakes.
- Header/signature regions and identifier formats are explicit heuristics, not ground truth.
- Distribution similarity is descriptive, not a guarantee of generalization. Operational timestamps are synthetic.
- Exact duplicates are checked; no exhaustive near-duplicate semantic search is claimed.

### Reproduction
From the repository root: `uv run src/audit_reconstruction.py`, `uv run src/audit_annotations.py`, `uv run src/audit_metadata.py`, then `uv run src/build_understanding_notebook.py`.
The saved notebook checks SHA-256 hashes before using existing audit results. Change `RUN_AUDITS` below to recompute them in a kernel with the dependencies declared in the builder script. After source changes, rerun the audits and builder to generate a new manifest.
""")
    code("""from pathlib import Path
import collections, csv, hashlib, importlib.metadata, json, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from IPython.display import display, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'Data/README.txt').exists())
OUT = ROOT / 'output/data_understanding'
RUN_AUDITS = False
if RUN_AUDITS:
    for name in ['audit_reconstruction.py', 'audit_annotations.py', 'audit_metadata.py']:
        result = subprocess.run([sys.executable, str(ROOT / 'src' / name)], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
        print(name, 'completed')
manifest = json.loads((OUT / 'manifest.json').read_text())
for entry in manifest['files']:
    # Generated JSON may change when audits rerun (execution timestamp); raw inputs and code must still match.
    if RUN_AUDITS and entry['path'].startswith('output/'):
        continue
    h = hashlib.sha256()
    with (ROOT / entry['path']).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == entry['sha256'], f"Changed input/artifact: {entry['path']}"
recon = json.loads((OUT / 'reconstruction.json').read_text())
ann = json.loads((OUT / 'annotations.json').read_text())
meta = json.loads((OUT / 'metadata.json').read_text())
examples = json.loads((OUT / 'annotations_examples.json').read_text())
plt.rcParams.update({'figure.figsize': (10, 4.5), 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlepad': 14, 'savefig.bbox': 'tight'})
BLUE, GOLD = '#275D8C', '#AA6A13'
print('Input and artifact hashes verified:', len(manifest['files']))
print({p: importlib.metadata.version(p) for p in ['pyarrow','numpy','pandas','scipy','matplotlib']})
""")
    md("""## Data
Full-scan sources: shuffled segment CSVs, gold label CSV, both metadata Parquets, submission slots, and weak JSONL structural checks. Weak labels are excluded from convention/context analysis. Rows below are file-level provenance, not inferred freshness.""")
    code("""display(pd.DataFrame(recon['sources'])[['path', 'bytes']])
train_meta = pq.read_table(ROOT / 'Data/train/train_metadata.parquet').to_pandas()
test_meta = pq.read_table(ROOT / 'Data/test/test_metadata.parquet').to_pandas()
gold = pd.read_csv(ROOT / 'Data/train/train_labels.csv', keep_default_na=False)
counts = gold.groupby('label').size().sort_values(ascending=False)
assert int(counts.sum()) == ann['denominators']['annotations'] == meta['scope']['train_labels']
""")
    md("""## Results
### 1. Reconstruction verification
The checks use the exact source text without stripping, normalization, or blank-row removal. Hash normalization is used only to detect duplicates, never to replace model input. A structural pass cannot prove that every gold annotation is semantically correct.""")
    code("""checks = pd.DataFrame(recon['checks'])
display(checks[['name','status','failures','denominator']])
display(pd.DataFrame({s: {k: recon['splits'][s][k] for k in ['documents','segments','empty_segments','documents_with_non_ascii']} for s in ['train','test']}).T)
print('Identical training-document groups:', recon['splits']['train']['exact_duplicate_groups'])
print('Duplicate-text annotation conflicts:', recon['duplicate_text_annotation_conflicts'])
""")
    md("""### 2. Annotation convention audit
The first chart gives the scoring weight of each category in train. The boundary table distinguishes characters inside a span from characters adjacent to it. Detailed exceptions retain document IDs and exact offsets in `annotations_examples.json`.""")
    code("""fig, ax = plt.subplots()
counts.sort_values().plot.barh(ax=ax, color=BLUE)
ax.set(title='Gold annotation counts by label', xlabel='Annotated spans — full training corpus', ylabel='')
for i, value in enumerate(counts.sort_values()):
    ax.text(value + counts.max() * .01, i, f'{value:,}', va='center', fontsize=10)
ax.set_xlim(0, counts.max() * 1.22)
fig.tight_layout(); fig.savefig(OUT / 'label_counts.png'); plt.show()
boundary = pd.DataFrame(ann['boundary_by_label']).T
display(boundary[['count','leading_whitespace','trailing_whitespace','internal_newline','leading_punctuation','trailing_punctuation']])
display(pd.DataFrame(ann['label_profile']).T.fillna(0))
display(pd.DataFrame(ann['cross_label_values_top'][:8]))
""")
    md("""### 3. Label-context profiling
Each row below sums to 100% within a gold label. Region assignments are position/text heuristics; `signature_or_tail` must not be read as confirmed signatures. The full report also retains channel/domain counts and frequent neighboring fragments.""")
    code("""regions = pd.DataFrame(ann['region_by_label']).T.fillna(0)
assert (regions.sum(axis=1).astype(int) == counts.reindex(regions.index)).all()
shares = regions.div(regions.sum(axis=1), axis=0) * 100
display(shares.round(2))
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(shares.to_numpy(), cmap='Blues', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(shares.columns)), [c.replace('_', '\\n') for c in shares.columns])
ax.set_yticks(range(len(shares)), shares.index)
ax.set_title('Gold labels by heuristic document region')
for i in range(len(shares)):
    for j in range(len(shares.columns)):
        value = shares.iloc[i,j]
        ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color='white' if value > 55 else '#222222')
fig.colorbar(im, ax=ax, label='Percent of spans within label')
fig.tight_layout(); fig.savefig(OUT / 'label_regions.png'); plt.show()
display(pd.Series(ann['repetition'], name='Observed count').to_frame())
""")
    md("""### 4. Metadata signal audit
These are associations in the observed training corpus, not held-out predictive utility. Text length/count metadata is a redundant description of the reconstructed string. Operational group means are recomputed below directly from gold counts and metadata to cross-check the audit artifact.""")
    code("""doc_counts = gold.groupby('document_id').size().rename('entity_count')
joined = train_meta.merge(doc_counts, left_on='document_id', right_index=True, how='left', validate='one_to_one')
assert joined['entity_count'].notna().all()
source_summary = joined.groupby('source_system')['entity_count'].agg(['size','mean','std'])
for source, row in source_summary.iterrows():
    expected = meta['operational_associations_train']['source_system'][str(source)]
    assert int(row['size']) == expected['documents']
    assert np.isclose(row['mean'], expected['gold_count_mean'])
display(source_summary.round(4))
correlations = joined[['n_chars','n_segments','n_words','n_nonempty_segments','entity_count']].corr()
display(correlations.round(3))
for label, chunk in meta['categorical_comparison'].items():
    print(label, 'train/test TVD =', round(chunk['total_variation_distance'], 6))
""")
    md("""### 5. Train/test distribution comparison
Both splits use the same definitions. Entity count is measured from gold spans in train and exposed slots in test. Small standardized differences and small KS distances support similarity only for the variables actually measured; test entity labels remain unavailable.""")
    code("""rows = []
for field, item in meta['numeric_comparison'].items():
    rows.append({'field': field, 'train_mean': item['train']['mean'], 'test_mean': item['test']['mean'],
                 'standardized_difference': item['standardized_mean_difference'], 'KS_distance': item.get('ks_statistic')})
display(pd.DataFrame(rows).round(5))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for label, data, color, linestyle in [('Train',train_meta,BLUE,'-'),('Test',test_meta,GOLD,'--')]:
    values = np.sort(data['n_chars'].to_numpy())
    axes[0].plot(values, np.arange(1, len(values)+1)/len(values), label=label, color=color, linestyle=linestyle)
axes[0].set(title='Document length distribution', xlabel='Python characters per document', ylabel='Cumulative document share', ylim=(0,1))
axes[0].legend()
domains = sorted(set(train_meta.domain) | set(test_meta.domain))
x = np.arange(len(domains))
for label, data, color, offset in [('Train',train_meta,BLUE,-.18),('Test',test_meta,GOLD,.18)]:
    values = data.domain.value_counts(normalize=True).reindex(domains, fill_value=0) * 100
    axes[1].barh(x + offset, values, .36, label=label, color=color)
axes[1].set_yticks(x, domains)
axes[1].set(title='Domain composition', xlabel='Percent of documents', ylabel='')
axes[1].legend()
fig.tight_layout(); fig.savefig(OUT / 'train_test_comparison.png'); plt.show()
display(pd.DataFrame(meta['formatting']).T)
display(pd.Series(meta['text_overlap'], name='Value').to_frame())
""")
    md("""## Takeaways
Use the integrated [report](../../note/REPORT.md) to distinguish verified findings, ambiguous conventions, and next tests. Preserve raw strings; group identical documents during validation; inspect labels in context before writing normalization rules. Use metadata to describe and stratify evaluation before claiming feature value. Test distribution similarity does not establish hidden-label similarity.

The weak-data scan establishes structural validity only; it does not measure semantic precision or recall. The sample submission contains placeholders, not labels. This notebook does not change the baseline, train a model, or submit predictions.
""")
    nb.cells = cells
    reconstruction = json.loads((OUT / 'reconstruction.json').read_text())
    annotations = json.loads((OUT / 'annotations.json').read_text())
    nb.cells[0].source = nb.cells[0].source.replace(
        'The five-stage audit covers reconstruction, annotation conventions, label context, metadata associations, and observable train/test distributions.',
        f"The full reconstruction audit passed {reconstruction['summary']['passed']}/{reconstruction['summary']['checks']} checks across train and test. "
        f"The gold corpus contains {annotations['cross_label_distinct_values']:,} exact values assigned more than one category across occurrences, warranting context and annotation-noise review. "
        'The five-stage audit covers annotation conventions, label context, metadata associations, and observable train/test distributions.')
    nbf.validate(nb)
    path = OUT / "data_understanding.ipynb"
    nbf.write(nb, path)
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.validate(nb)
    nbf.write(nb, path)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    (OUT / "data_understanding.html").write_text(body, encoding="utf-8")
    print(f"Executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells: {path}")


if __name__ == "__main__":
    main()
