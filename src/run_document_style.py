# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1", "scikit-learn==1.9.1"]
# ///
"""Fixed train-only document honorific style classifier and CRF adjustment."""
from __future__ import annotations
import collections, hashlib, json, pickle, re, time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from prepared_loader import PreparedData
from ner_evaluation import Evaluation, triples

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output/bitrase-1.algo-5'
TITLE_RE=re.compile(r'(?i)(Mr|Mrs|Ms|Miss|Dr|Prof|Mx)(?:\.)?[ \t\r\n]+$')
START_RE=re.compile(r'(?i)^(Mr|Mrs|Ms|Miss|Dr|Prof|Mx)(?:\.)?[ \t\r\n]+')
LABELS={'NAME','DATE','EMAIL','PHONE','ADDRESS','USERNAME','JOB_TITLE'}

def audit_obs(text, entities):
    obs=[]
    for e in entities:
        if e['label']!='NAME': continue
        m=START_RE.match(e['text'])
        if m: obs.append(('included',m.group(1).title()))
        else:
            m=TITLE_RE.search(text[:e['start']])
            if m: obs.append(('excluded',m.group(1).title()))
    return obs

def adjust(text,predicted,policy):
    result=[]
    for a,b,label in predicted:
        if label=='NAME' and policy=='included':
            m=TITLE_RE.search(text[:a])
            if m: a=m.start()
        elif label=='NAME' and policy=='excluded':
            m=START_RE.match(text[a:b])
            if m: a += m.end()
        if not a<b: raise ValueError('empty honorific adjustment')
        result.append((a,b,label))
    result=sorted(result)
    if any(x[1]>y[0] for x,y in zip(result,result[1:])): raise ValueError('honorific adjustment overlap')
    return result

def main():
    if OUT.exists(): raise ValueError('Preserve prior experiment')
    out=OUT.resolve(); out.mkdir(parents=True)
    protocol=out/'protocol.md'
    protocol.write_text('# bitrase-1.algo-5 document honorific style\n\nFixed TF-IDF word 1-2 gram max_features=150000 min_df=2 and SGD logistic classifier, random_state=2026. Fit uses only prepared v1 train documents with at least two classified honorific observations; ties resolve to included. Dev style accuracy is descriptive. The sole adjustment uses predicted probability >= 0.5 for included and applies only to NAME spans in CRF known_k predictions. No holdout/test, refit, threshold sweep, or gold-based adjustment. Mixed documents remain a caveat.\n',encoding='utf-8')
    data=PreparedData(); train=data.load_train('train_fitting').to_pylist()
    train_x=[]; train_y=[]; style_docs=0; style_counts=collections.Counter();
    for r in train:
        obs=audit_obs(r['full_text'],r['entities'])
        if len(obs)<2: continue
        inc=sum(k=='included' for k,_ in obs); label='included' if inc>=len(obs)-inc else 'excluded'
        train_x.append(r['full_text']); train_y.append(label); style_counts[label]+=1; style_docs+=1
    model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=150000,min_df=2)),('clf',SGDClassifier(loss='log_loss',random_state=2026,max_iter=1000,tol=1e-3))])
    model.fit(train_x,train_y)
    with (out/'model.pkl').open('wb') as f: pickle.dump(model,f)
    vectorizer=model.named_steps['tfidf']; (out/'vocabulary.json').write_text(json.dumps(sorted(vectorizer.vocabulary_)),encoding='utf-8')
    dev=data.load_dev('evaluation').to_pylist(); crf=pq.read_table(ROOT/'output/bitrase-1.algo-3/crf12k/known_k_dev_predictions.parquet').to_pylist(); pred={r['document_id']:[(int(e['start']),int(e['end']),e['label']) for e in r['predicted']] for r in crf}
    style_total=style_correct=0; eval_base=Evaluation(); eval_adj=Evaluation(); adjusted=[]; mixed=collections.Counter(); invalid=0
    for r in dev:
        obs=audit_obs(r['full_text'],r['entities'])
        if len(obs)>=2:
            probs=model.predict_proba([r['full_text']])[0]; classes=model.named_steps['clf'].classes_
            # use explicit probability lookup to avoid class ordering assumptions
            policy='included' if ('included' in classes and probs[list(classes).index('included')]>=.5) else 'excluded'
            style_correct += policy == ('included' if sum(k=='included' for k,_ in obs)>=sum(k=='excluded' for k,_ in obs) else 'excluded'); style_total += 1
        else: policy='excluded'
        base=pred[r['document_id']];
        try: adj=adjust(r['full_text'],base,policy)
        except ValueError: adj=base; invalid+=1
        gold=triples(r['entities']); eval_base.add(r['full_text'],gold,base,base); eval_adj.add(r['full_text'],gold,adj,adj)
        adjusted.append({'document_id':r['document_id'],'predicted':[{'start':a,'end':b,'label':l} for a,b,l in adj]})
    report={'experiment':'bitrase-1.algo-5','train_style_documents':style_docs,'train_style_counts':dict(style_counts),'dev_style_documents':style_total,'dev_style_accuracy':style_correct/style_total if style_total else 0,'mixed_train_policy_caveat':'Document majority labels collapse mixed observations; no mixed docs are relabeled.','invalid_adjustments':invalid,'baseline_known_k':eval_base.report(),'adjusted_known_k':eval_adj.report(),'threshold':0.5,'pretrained':False,'holdout_read':False,'test_read':False}
    pq.write_table(pa.Table.from_pylist(adjusted),out/'adjusted_known_k_dev_predictions.parquet',compression='zstd'); (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'dev_style_accuracy':report['dev_style_accuracy'],'baseline_slot':report['baseline_known_k']['slot_accuracy'],'adjusted_slot':report['adjusted_known_k']['slot_accuracy'],'invalid_adjustments':invalid},indent=2))

if __name__=='__main__': main()
