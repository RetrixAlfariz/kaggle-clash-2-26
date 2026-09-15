"""Bounded train-only residual structure probes; never loads dev/test/holdout."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/bitrase-2/structure-diagnostic'
NOTE=ROOT/'note/bitrase-2/structure-diagnostic'
CACHE=ROOT/'output/bitrase-2/p0/cache/train'
P0=ROOT/'output/bitrase-2/p0/seed2026'
SEED=20260916
LABELS=('NAME','DATE','EMAIL','PHONE','ADDRESS','USERNAME','JOB_TITLE')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value,indent=2,default=default,allow_nan=False),encoding='utf-8')


def hashes():
    names=['src/run_structure_diagnostic.py','src/document_structure.py','src/structure_probe.py',
           'src/structure_families.py','src/span_probe.py','src/audit_template_similarity.py',
           'tests/test_document_structure.py','tests/test_structure_probe.py','tests/test_structure_families.py',
           'tests/test_structure_diagnostic.py','note/bitrase-2/structure-diagnostic/protocol.md']
    return {n:sha(ROOT/n) for n in names}


def freeze():
    from document_structure import CONTROL_NAMES,STRUCTURE_NAMES
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('Preserve frozen protocol')
    before=hashes()
    command=[sys.executable,'-B','-m','unittest','tests.test_document_structure','tests.test_structure_probe',
             'tests.test_structure_families','tests.test_structure_diagnostic','-v']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (OUT/'verification.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout+result.stderr,flush=True)
    if result.returncode or hashes()!=before:raise ValueError('Verification failed/source changed')
    schema={'control':[f'logit_{l}' for l in ('NONE',*LABELS)]+['candidate_minus_NONE','candidate_minus_best_other',
             'margin_hinge_minus4','margin_hinge_zero','margin_hinge_plus4']+CONTROL_NAMES,
            'additional_structure':STRUCTURE_NAMES,'per_label_probes':list(LABELS),'seed':SEED,
            'ridge_mean_loss_lambda':.001,'folds':3,'bootstrap_replicates':1000,
            'minimum_useful':{'auc_delta':.005,'relative_logloss_reduction':.01,'pair_accuracy_delta':.01}}
    write(NOTE/'feature_schema.json',schema)
    write(OUT/'frozen.json',{'sources':before,'feature_schema_sha256':sha(NOTE/'feature_schema.json'),
                            'verification_log_sha256':sha(OUT/'verification.log'),'tests_passed':True,'schema':schema})


def check():
    frozen=read(OUT/'frozen.json')
    if frozen['sources']!=hashes() or frozen['feature_schema_sha256']!=sha(NOTE/'feature_schema.json'):
        raise ValueError('Frozen specification/source mismatch')
    return frozen


def triples(items):return [(e['start'],e['end'],e['label']) for e in items]


def extract():
    import torch
    from prepared_loader import PreparedData
    from span_probe import SpanHead,universe,span_index,padded_hidden
    from document_structure import DocumentLayout
    from structure_families import group_and_sample
    frozen=check();target=OUT/'data'
    if target.exists():raise FileExistsError(target)
    started=time.monotonic();rows=PreparedData().load_train().to_pylist()
    # No cache helper that hashes other splits is used in this diagnostic.
    manifest=read(CACHE/'manifest.json')
    if not manifest['complete'] or manifest['split']!='train':raise ValueError('Invalid train cache')
    p0_config=read(P0/'config.json')
    if p0_config['sources']['output/bitrase-2/p0/cache/train/manifest.json']!=sha(CACHE/'manifest.json'):
        raise ValueError('Train cache manifest differs from frozen P0 training provenance')
    for name,info in manifest['files'].items():
        if sha(CACHE/name)!=info['sha256']:raise ValueError(f'Train cache changed: {name}')
    for name,digest in manifest['sources'].items():
        if name.endswith('/dev.parquet'):continue
        if sha(ROOT/name)!=digest:raise ValueError(f'Train source changed: {name}')
    headhash=sha(P0/'epoch2.pt')
    if headhash!='520646a7295797707e11fa24a8f06a5670c9d3f3201e81cf911ff11ec15b3e06':
        raise ValueError('Selected P0 checkpoint changed')
    print('Grouping train documents only',flush=True)
    selected,groups,folds,group_report=group_and_sample(rows)
    target.mkdir(parents=True);write(OUT/'grouping.json',group_report)
    meta=read(CACHE/'metadata.json');offs=np.load(CACHE/'offsets.npy',mmap_mode='r')
    hidden=np.load(CACHE/'hidden.npy',mmap_mode='r');bounds=np.load(CACHE/'boundaries.npy',mmap_mode='r')
    if len(rows)!=len(meta) or any(r['document_id']!=m['document_id'] for r,m in zip(rows,meta)):
        raise ValueError('Train cache document association mismatch')
    oldrecords=pq.read_table(CACHE/'train_known_k_predictions.parquet').to_pylist()
    old={r['document_id']:triples(r['predicted']) for r in oldrecords}
    if len(old)!=len(rows) or len(oldrecords)!=len(old):raise ValueError('Old prediction coverage')
    head=SpanHead().cuda();head.load_state_dict(torch.load(P0/'epoch2.pt',map_location='cuda',weights_only=True));head.eval();torch.set_num_threads(4)
    documents=[];plans=[];unaligned=long=0
    group_names=sorted(set(groups));group_index={g:i for i,g in enumerate(group_names)}
    for j,(i,group,fold) in enumerate(zip(selected,groups,folds)):
        r=rows[i];bd=bounds[offs[i]:offs[i+1]];n=len(bd);allpairs=universe(n)
        starts={int(x):k for k,(x,y) in enumerate(bd)};ends={int(y):k+1 for k,(x,y) in enumerate(bd)}
        gold=set(triples(r['entities']));entries={};positive=[]
        seed=int.from_bytes(hashlib.sha256(f'{SEED}:{r["document_id"]}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
        def add(a,b,label,tag):entries[(a,b,label)]=entries.get((a,b,label),0)|tag
        for a,b,l in sorted(gold):
            if a not in starts or b not in ends:unaligned+=1;continue
            x,y=starts[a],ends[b]
            if y-x>16:long+=1;continue
            label=LABELS.index(l);add(x,y,label,1);positive.append((x,y,label))
            near=sorted({(s,t) for delta in (-2,-1,1,2) for s,t in ((x+delta,y),(x,y+delta)) if 0<=s<t<=n and t-s<=16})
            for ix in rng.choice(len(near),min(2,len(near)),replace=False):add(*near[ix],label,2)
        for a,b,l in old[r['document_id']]:
            if a in starts and b in ends and ends[b]-starts[a]<=16:add(starts[a],ends[b],LABELS.index(l),4)
        randompairs=[tuple(map(int,allpairs[k])) for k in rng.choice(len(allpairs),min(16,len(allpairs)),replace=False)]
        intervals=sorted({(a,b) for a,b,l in entries}|set(randompairs))
        documents.append({'document_id':r['document_id'],'train_index':int(i),'group':group,'group_index':group_index[group],
                          'fold':int(fold),'tokens':n,'gold':len(gold)})
        plans.append((entries,positive,randompairs,intervals,gold))
    control=[];structure=[];labels=[];targets=[];docids=[];sources=[];candidate_rows=[];pair_pos=[];pair_neg=[]
    order=sorted(range(len(selected)),key=lambda j:documents[j]['tokens'])
    with torch.no_grad():
        for begin in range(0,len(order),32):
            ids=order[begin:begin+32]
            h,prefix=padded_hidden([hidden[offs[selected[j]]:offs[selected[j]+1]] for j in ids])
            pp=np.concatenate([np.asarray(plans[j][3],dtype=np.int64) for j in ids]);dd=np.repeat(np.arange(len(ids)),[len(plans[j][3]) for j in ids])
            chunks=[]
            for start in range(0,len(pp),32768):
                p=pp[start:start+32768];d=dd[start:start+32768]
                with torch.autocast('cuda',dtype=torch.float16):
                    z=head(h,prefix,torch.as_tensor(d,device='cuda'),torch.as_tensor(p[:,0],device='cuda'),torch.as_tensor(p[:,1],device='cuda'))
                chunks.append(z.float().cpu().numpy())
            zs=np.concatenate(chunks);cursor=0
            for j in ids:
                entries,positive,randompairs,intervals,gold=plans[j];z=zs[cursor:cursor+len(intervals)];cursor+=len(intervals)
                zmap={p:z[k].astype(np.float64) for k,p in enumerate(intervals)}
                if not np.isfinite(z).all():raise ValueError('Nonfinite P0 score')
                for a,b,l in positive:
                    zz=zmap[a,b][1:].copy();zz[l]=-np.inf;other=int(zz.argmax());key=(a,b,other);entries[key]=entries.get(key,0)|8
                for a,b in randompairs:
                    l=int(zmap[a,b][1:].argmax());key=(a,b,l);entries[key]=entries.get(key,0)|16
                i=selected[j];r=rows[i];bd=bounds[offs[i]:offs[i+1]];layout=DocumentLayout(r['full_text'],bd)
                local=[]
                for (a,b,l),tag in sorted(entries.items()):
                    zz=zmap[a,b];score=zz[l+1]-zz[0];other=np.delete(zz[1:],l).max();c,s=layout.features(a,b)
                    features=np.r_[zz,score,zz[l+1]-other,np.maximum(score-np.array([-4.,0.,4.]),0),c]
                    correct=(int(bd[a,0]),int(bd[b-1,1]),LABELS[l]) in gold
                    index=len(control);control.append(features);structure.append(s);labels.append(l);targets.append(correct);docids.append(j);sources.append(tag)
                    candidate_rows.append({'document_index':j,'start_token':a,'end_token':b,'label_index':l,'target':correct,'source_bits':tag})
                    local.append((index,l,correct,tag,score))
                for l in range(7):
                    positives=[x for x in local if x[1]==l and x[2]]
                    negatives=sorted([x for x in local if x[1]==l and not x[2] and x[3]&14],key=lambda x:(-x[4],x[0]))[:4]
                    for g in positives:
                        for neg in negatives:pair_pos.append(g[0]);pair_neg.append(neg[0])
            if begin%1024==0:print(f'Extracted {min(begin+32,len(order))}/{len(order)} documents',flush=True)
    arrays={'control':np.asarray(control,dtype=np.float64),'structure':np.asarray(structure,dtype=np.float64),
            'labels':np.asarray(labels,dtype=np.int8),'target':np.asarray(targets,dtype=np.int8),'document':np.asarray(docids,dtype=np.int32),
            'source_bits':np.asarray(sources,dtype=np.int8),'pair_pos':np.asarray(pair_pos,dtype=np.int64),'pair_neg':np.asarray(pair_neg,dtype=np.int64)}
    for name,value in arrays.items():np.save(target/f'{name}.npy',value)
    write(target/'documents.json',documents);pq.write_table(pa.Table.from_pylist(candidate_rows),target/'candidates.parquet',compression='zstd')
    for name in ('control','structure'):
        if not np.isfinite(arrays[name]).all():raise ValueError('Nonfinite extracted features')
    schema=frozen['schema']
    if arrays['control'].shape[1]!=len(schema['control']) or arrays['structure'].shape[1]!=len(schema['additional_structure']):raise ValueError('Feature schema mismatch')
    if hashes()!=frozen['sources']:raise ValueError('Sources changed during extraction')
    write(target/'manifest.json',{'complete':True,'train_sha256':sha(ROOT/'output/prepared/v1/train.parquet'),
          'head_sha256':headhash,'train_cache_manifest_sha256':sha(CACHE/'manifest.json'),'frozen_sha256':sha(OUT/'frozen.json'),
          'files':{p.name:sha(p) for p in target.iterdir()},'documents':len(selected),'families':len(group_names),
          'candidates':len(control),'pairs':len(pair_pos),'unaligned_gold':unaligned,'long_gold':long,
          'seconds':time.monotonic()-started,'positive':int(arrays['target'].sum()),'negative':int((1-arrays['target']).sum())})
    print(json.dumps({k:read(target/'manifest.json')[k] for k in ('documents','families','candidates','pairs','seconds')}),flush=True)


def data():
    check();m=read(OUT/'data/manifest.json')
    if not m['complete'] or m['frozen_sha256']!=sha(OUT/'frozen.json'):raise ValueError('Extraction gate')
    for name,digest in m['files'].items():
        if sha(OUT/'data'/name)!=digest:raise ValueError(f'Diagnostic data hash mismatch:{name}')
    return {p.stem:np.load(p,mmap_mode='r') for p in (OUT/'data').glob('*.npy')},read(OUT/'data/documents.json')


def fit():
    from structure_probe import fit_predict
    x,docs=data();dest=OUT/'probes'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir();n=len(x['target']);pred=np.full((n,2),np.nan);f=np.array([d['fold'] for d in docs])[x['document']]
    records=[];started=time.monotonic()
    for label in range(7):
        for fold in range(3):
            train=(x['labels']==label)&(f!=fold);test=(x['labels']==label)&(f==fold)
            if not test.any() or len(np.unique(x['target'][train]))<2:raise ValueError(f'Unsupported label/fold:{label}/{fold}; inconclusive')
            for variant in range(2):
                design=x['control'] if variant==0 else np.column_stack([x['control'],x['structure']])
                p,info=fit_predict(design[train],x['target'][train],design[test])
                if not info['converged']:raise ValueError(f'Probe nonconvergence:{label}/{fold}/{variant}')
                pred[test,variant]=p;records.append({'label':LABELS[label],'fold':fold,'variant':variant,'fit':info})
            print(f'Fit {LABELS[label]} fold{fold}',flush=True)
    if not np.isfinite(pred).all():raise ValueError('Missing OOF predictions')
    np.save(dest/'oof.npy',pred);write(dest/'fits.json',records)
    write(dest/'manifest.json',{'complete':True,'seconds':time.monotonic()-started,'oof_sha256':sha(dest/'oof.npy'),
         'data_manifest_sha256':sha(OUT/'data/manifest.json'),'sources':hashes(),'fits':len(records)})


class FastAUC:
    def __init__(self,y,p,g):
        order=np.argsort(p,kind='stable');scores=p[order]
        self.g=np.asarray(g)[order];self.y=np.asarray(y,dtype=np.float64)[order]
        self.starts=np.r_[0,np.flatnonzero(np.diff(scores)!=0)+1]
    def value(self,counts):
        w=counts[self.g];pos=np.add.reduceat(w*self.y,self.starts);neg=np.add.reduceat(w*(1-self.y),self.starts)
        denom=pos.sum()*neg.sum()
        if not denom:return np.nan
        return float(np.dot(pos,np.cumsum(neg)-.5*neg)/denom)


def summarize(mask,pairmask,x,pred,groups,draws=None):
    y=np.asarray(x['target'][mask]);g=groups[mask];N=int(groups.max())+1
    if len(np.unique(y))<2:return {'supported':False}
    models=[FastAUC(y,pred[mask,v],g) for v in range(2)]
    loss=[]
    for v in range(2):
        p=np.clip(pred[mask,v],1e-12,1-1e-12)
        loss.append(np.bincount(g,weights=-(y*np.log(p)+(1-y)*np.log1p(-p)),minlength=N))
    n=np.bincount(g,minlength=N);ones=np.ones(N);pp=x['pair_pos'][pairmask];pn=x['pair_neg'][pairmask]
    pg=groups[pp];pairs=np.bincount(pg,minlength=N);wins=[]
    for v in range(2):wins.append(np.bincount(pg,weights=(pred[pp,v]>pred[pn,v])+.5*(pred[pp,v]==pred[pn,v]),minlength=N))
    def calc(weights):
        auc=[m.value(weights) for m in models];ll=[float(weights@l/(weights@n)) for l in loss]
        pair=[float(weights@w/(weights@pairs)) if weights@pairs else np.nan for w in wins]
        return auc,ll,pair
    auc,ll,pair=calc(ones)
    result={'supported':True,'candidates':int(len(y)),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),
            'families':int(len(set(g))),'pairs':int(len(pp)),'control':{'auc':auc[0],'logloss':ll[0],'pair_accuracy':pair[0]},
            'structural':{'auc':auc[1],'logloss':ll[1],'pair_accuracy':pair[1]},
            'improvement':{'auc':auc[1]-auc[0],'logloss':ll[0]-ll[1],'relative_logloss':(ll[0]-ll[1])/ll[0],'pair_accuracy':pair[1]-pair[0]}}
    if draws is not None:
        samples=[]
        for counts in draws:
            a,l,p=calc(counts);samples.append([a[1]-a[0],l[0]-l[1],(l[0]-l[1])/l[0],p[1]-p[0]])
        samples=np.asarray(samples)
        if not np.isfinite(samples).all():raise ValueError('Bootstrap class/pair support failed; inconclusive')
        result['ci95']={name:np.quantile(samples[:,i],[.025,.975]).tolist() for i,name in enumerate(('auc','logloss','relative_logloss','pair_accuracy'))}
    return result


def evaluate():
    x,docs=data();manifest=read(OUT/'probes/manifest.json')
    if manifest['sources']!=hashes() or manifest['oof_sha256']!=sha(OUT/'probes/oof.npy'):raise ValueError('OOF provenance mismatch')
    if (OUT/'results.json').exists():raise FileExistsError('Preserve results')
    pred=np.load(OUT/'probes/oof.npy');groups=np.array([d['group_index'] for d in docs])[x['document']]
    folds=np.array([d['fold'] for d in docs])[x['document']];G=int(groups.max())+1
    rng=np.random.default_rng(SEED);draws=np.asarray([np.bincount(rng.integers(0,G,size=G),minlength=G) for _ in range(1000)],dtype=np.int32)
    start=time.monotonic();results={}
    for label in [-1,*range(7)]:
        mask=np.ones(len(pred),dtype=bool) if label<0 else x['labels']==label
        pairmask=mask[x['pair_pos']]
        name='pooled' if label<0 else LABELS[label]
        results[name]=summarize(mask,pairmask,x,pred,groups,draws)
        print(f'Evaluated/bootstrap {name}: {results[name]["improvement"]}',flush=True)
    byfold={str(f):summarize(folds==f,folds[x['pair_pos']]==f,x,pred,groups) for f in range(3)}
    bysource={}
    for name,bit in [('gold_boundary',1),('boundary_alternative',2),('old_model',4),('alternative_label',8),('random',16)]:
        mask=(x['source_bits']&bit)!=0
        bysource[name]={'candidates':int(mask.sum()),'positive':int(x['target'][mask].sum()),'negative':int(mask.sum()-x['target'][mask].sum())}
    primary=results['pooled'];delta=primary['improvement'];ci=primary['ci95']
    support=[l for l in LABELS if min(results[l]['positive'],results[l]['negative'],results[l]['pairs'])>=50 and results[l]['families']>=20]
    gate={'auc_minimum':delta['auc']>=.005,'logloss_minimum':delta['relative_logloss']>=.01,
          'pair_minimum':delta['pair_accuracy']>=.01,'positive_intervals':all(ci[k][0]>0 for k in ('auc','logloss','pair_accuracy')),
          'all_folds_positive':all(r['improvement']['logloss']>0 and r['improvement']['pair_accuracy']>0 for r in byfold.values()),
          'five_labels_positive':sum(results[l]['improvement']['pair_accuracy']>0 for l in support)>=5}
    report={'results':results,'folds':byfold,'source_counts_overlapping':bysource,'gate':gate,'advance':all(gate.values()),
            'supported_labels':support,'bootstrap':{'replicates':1000,'seed':SEED,'unit':'approximate train family','refit':False},
            'seconds':time.monotonic()-start,'train_only':True,'P0_train_exposure':True,'dev_accessed':False,'production_change':False,
            'frozen_sha256':sha(OUT/'frozen.json'),'oof_sha256':sha(OUT/'probes/oof.npy')}
    write(OUT/'results.json',report)
    print(json.dumps({'advance':report['advance'],'gate':gate,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','extract','fit','evaluate'));args=p.parse_args()
    globals()[args.stage]()
