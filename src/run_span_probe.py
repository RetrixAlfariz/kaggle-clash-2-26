"""Train/evaluate the frozen direct-span P0 head. Never imports holdout/test."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from prepared_loader import PreparedData
from ner_evaluation import Evaluation,triples
from run_neural_sequence import sha,source_hashes,clean_report
from span_probe import SpanHead,universe,sampling_plan,sample_plan,padded_hidden,decode_spans,character_spans

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/bitrase-2/p0'
CHUNK=32768

def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def write_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

class Cache:
    def __init__(self,split):
        self.path=BASE/'cache'/split;self.manifest=read_json(self.path/'manifest.json')
        if not self.manifest['complete'] or self.manifest['split']!=split:raise ValueError('Incomplete/wrong cache')
        if self.manifest['generator_sha256']!=sha(ROOT/'src/build_span_cache.py'):raise ValueError('Cache generator changed')
        for name,info in self.manifest['files'].items():
            if sha(self.path/name)!=info['sha256']:raise ValueError(f'Cache hash mismatch {name}')
        if self.manifest['sources']!=source_hashes():raise ValueError('Parent cache provenance mismatch')
        self.hidden=np.load(self.path/'hidden.npy',mmap_mode='r')
        self.bounds=np.load(self.path/'boundaries.npy',mmap_mode='r')
        self.offsets=np.load(self.path/'offsets.npy',mmap_mode='r')
        self.meta=read_json(self.path/'metadata.json')
        if len(self.offsets)!=len(self.meta)+1 or self.offsets[-1]!=len(self.hidden):raise ValueError('Cache index contract')
        if np.any(np.diff(self.offsets)<=0) or self.hidden.shape[1]!=256 or self.bounds.shape!=(len(self.hidden),2):raise ValueError('Empty/malformed cached document')
        if any(int(self.offsets[i+1]-self.offsets[i])!=m['token_count'] for i,m in enumerate(self.meta)):raise ValueError('Cached token count mismatch')
    def h(self,i):return self.hidden[int(self.offsets[i]):int(self.offsets[i+1])]
    def b(self,i):return self.bounds[int(self.offsets[i]):int(self.offsets[i+1])]
    def check_rows(self,rows):
        if len(rows)!=len(self.meta):raise ValueError('Rows/cache mismatch')
        for row,m in zip(rows,self.meta):
            if row['document_id']!=m['document_id'] or row['expected_entity_count']!=m['expectedK']:raise ValueError('Cache row association')

def provenance():
    files=['src/span_probe.py','src/run_span_probe.py','src/build_span_cache.py','note/bitrase-2/p0/protocol.md',
           'output/bitrase-2/p0/cache/train/manifest.json','output/bitrase-2/p0/cache/dev/manifest.json',
           'output/bitrase-1.algo-6/v1/epoch4_known_k_dev_predictions.parquet',
           'output/bitrase-1.algo-6/ensemble_4_8/known_k_dev_predictions.parquet']
    return {**source_hashes(),**{f:sha(ROOT/f) for f in files}}

def bootstrap(documents,baseline,seed=2026):
    base={r['document_id']:triples(r['predicted']) for r in pq.read_table(baseline).to_pylist()}
    if set(base)!={r['document_id'] for r in documents}:raise ValueError('Bootstrap baseline coverage')
    k=np.asarray([r['K'] for r in documents],dtype=np.int64)
    current=np.asarray([r['correct'] for r in documents],dtype=np.int64)
    old=np.asarray([sum(g==p for g,p in zip(r['gold'],base[r['document_id']])) for r in documents],dtype=np.int64)
    delta=current-old;rng=np.random.default_rng(seed);draws=[]
    for begin in range(0,10000,128):
        idx=rng.integers(0,len(k),size=(min(128,10000-begin),len(k)))
        draws.extend((delta[idx].sum(1)/k[idx].sum(1)).tolist())
    return {'observed_delta':float(delta.sum()/k.sum()),'median':float(np.median(draws)),
            'ci95':np.quantile(draws,[.025,.975]).tolist(),'replicates':10000,'paired_unit':'document',
            'baseline_sha256':sha(baseline),'selection_bias_corrected':False}

def evaluate(head,cache,rows,out,epoch):
    head.eval();ev=Evaluation();predictions=[];documents=[]
    order=sorted(range(len(rows)),key=lambda i:cache.meta[i]['token_count']);t0=time.monotonic()
    with torch.no_grad():
        for begin in range(0,len(order),32):
            ids=order[begin:begin+32];h,prefix=padded_hidden([cache.h(i) for i in ids])
            pairs=[universe(len(cache.h(i))) for i in ids];counts=[len(x) for x in pairs]
            allpairs=np.concatenate(pairs);di=np.repeat(np.arange(len(ids)),counts)
            values=[];labels=[]
            for offset in range(0,len(allpairs),CHUNK):
                pp=allpairs[offset:offset+CHUNK];dd=di[offset:offset+CHUNK]
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=head(h,prefix,torch.as_tensor(dd,device='cuda'),torch.as_tensor(pp[:,0],device='cuda'),torch.as_tensor(pp[:,1],device='cuda'))
                z=logits.float().cpu().numpy()
                if not np.isfinite(z).all():raise ValueError('Nonfinite logits')
                lab=z[:,1:].argmax(1)+1
                values.append(z[np.arange(len(z)),lab]-z[:,0]);labels.append(lab)
            values=np.concatenate(values);labels=np.concatenate(labels);offset=0
            for i,pp,count in zip(ids,pairs,counts):
                row=rows[i];gold=triples(row['entities']);k=row['expected_entity_count']
                pred=character_spans(decode_spans(len(cache.h(i)),values[offset:offset+count],labels[offset:offset+count],k),cache.b(i))
                ev.add(row['full_text'],gold,pred,pred)
                documents.append({'document_id':row['document_id'],'K':k,'correct':sum(a==b for a,b in zip(gold,pred)),'gold':gold})
                predictions.append({'document_id':row['document_id'],'predicted':[{'start':a,'end':b,'label':l} for a,b,l in pred]})
                offset+=count
            if begin%(32*50)==0:print(f'Epoch{epoch} dev {min(begin+32,len(order))}/{len(order)}',flush=True)
    file=out/f'epoch{epoch}_dev_predictions.parquet';pq.write_table(pa.Table.from_pylist(predictions),file,compression='zstd')
    docfile=out/f'epoch{epoch}_documents.parquet'
    pq.write_table(pa.Table.from_pylist([{k:v for k,v in r.items() if k!='gold'} for r in documents]),docfile,compression='zstd')
    result={'dev':{'known_k':clean_report(ev.report())},'prediction_sha256':sha(file),'seconds':time.monotonic()-t0,
            'model_sha256':sha(out/f'epoch{epoch}.pt')}
    write_json(out/f'epoch{epoch}_report.json',result)
    print(json.dumps({'epoch':epoch,'slot_accuracy':result['dev']['known_k']['slot_accuracy'],'seconds':result['seconds']}),flush=True)
    return result,documents

def optimize_batch(head,opt,scaler,h,prefix,pairs,dd,targets,tags,ww,denom):
    """Retry the same batch/dropout draw if float16 loss scaling overflows.

    GradScaler skips a nonfinite step and reduces its scale. Never omit the
    batch from training, and never count failed attempts in loss statistics.
    """
    state=torch.cuda.get_rng_state()
    for attempt in range(16):
        torch.cuda.set_rng_state(state);opt.zero_grad(set_to_none=True)
        total=0.;strata={name:[0.,0,0] for name in ('positive','near','old_fp','random')}
        for offset in range(0,len(pairs),CHUNK):
            pp=pairs[offset:offset+CHUNK];di=dd[offset:offset+CHUNK];target=torch.as_tensor(targets[offset:offset+CHUNK],device='cuda');weight=torch.as_tensor(ww[offset:offset+CHUNK],device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                logits=head(h,prefix,torch.as_tensor(di,device='cuda'),torch.as_tensor(pp[:,0],device='cuda'),torch.as_tensor(pp[:,1],device='cuda'))
                ce=torch.nn.functional.cross_entropy(logits,target,reduction='none');loss=(ce*weight).sum()/denom
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward();total+=float(loss.detach())*denom
            ces=ce.detach().float().cpu().numpy();correct=(logits.detach().argmax(1)==target).cpu().numpy();tag=tags[offset:offset+CHUNK]
            for name,bit in [('positive',1),('near',2),('old_fp',4),('random',8)]:
                mask=(tag&bit)!=0;strata[name][0]+=float(ces[mask].sum());strata[name][1]+=int(mask.sum());strata[name][2]+=int(correct[mask].sum())
        scaler.unscale_(opt)
        finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None]).all())
        if finite:
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite unscaled norm despite finite gradients')
            scaler.step(opt);scaler.update();return total,strata,attempt
        before=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()>=before:raise ValueError('Loss scaler failed to reduce overflow')
        print(f'AMP overflow; retry same batch at scale{scaler.get_scale()}',flush=True)
    raise ValueError('Persistent nonfinite gradient after16 scale reductions')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=2026);parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--replication',action='store_true');a=parser.parse_args()
    if a.seed not in (2026,3407,1337) or not 1<=a.epochs<=3:raise ValueError('Outside protocol budget')
    if not a.replication and (a.seed!=2026 or a.epochs!=3):raise ValueError('Discovery protocol fixed')
    if a.replication:
        discovery=read_json(BASE/'seed2026/report.json')
        if not discovery['research_positive'] or a.epochs!=discovery['selected_epoch'] or a.seed==2026:raise ValueError('Replication gate')
    out=BASE/f'seed{a.seed}'
    if out.exists():raise ValueError('Preserve prior run')
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    traincache=Cache('train');devcache=Cache('dev');data=PreparedData()
    train=data.load_train().to_pylist();dev=data.load_dev('evaluation').to_pylist();traincache.check_rows(train);devcache.check_rows(dev)
    hashes=provenance();out.mkdir(parents=True)
    config={'seed':a.seed,'epochs':a.epochs,'replication':a.replication,'sources':hashes,'pretrained':False,
            'batch_documents':32,'chunk_spans':CHUNK,'lr':.001,'weight_decay':.01,'clip':1.,'optimizer':'AdamW',
            'initial_loss_scale':1024,'overflow_retry_limit':16,
            'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'cache_dtype':str(traincache.hidden.dtype)}
    write_json(out/'config.json',config);started=time.monotonic()
    oldrows=pq.read_table(traincache.path/'train_known_k_predictions.parquet').to_pylist()
    old={r['document_id']:triples(r['predicted']) for r in oldrows}
    if len(oldrows)!=len(old) or set(old)!={r['document_id'] for r in train}:raise ValueError('Train predictions coverage')
    plans=[]
    for i,row in enumerate(train):
        plans.append(sampling_plan(traincache.b(i),row['entities'],old[row['document_id']]))
        if (i+1)%10000==0:print(f'Sampling plans {i+1}/{len(train)}',flush=True)
    weights=[max(1,len(p['remaining'])/256) for p in plans]
    sampling_documents=[{'document_id':traincache.meta[i]['document_id'],'original_index':i,
                         'positive':len(p['positive']),'forced_negative':len(p['forced']),
                         'remaining':len(p['remaining']),'random_draws':min(256,len(p['remaining'])),
                         'random_inclusion_probability':1/weights[i],
                         'population':p['population'],'raw_population':p['raw_population'],
                         'unaligned':p['unaligned'],'long':p['long']} for i,p in enumerate(plans)]
    pq.write_table(pa.Table.from_pylist(sampling_documents),out/'sampling_documents.parquet',compression='zstd')
    del sampling_documents
    stats={'documents':len(plans),'positive':sum(len(p['positive']) for p in plans),'forced_negatives':sum(len(p['forced']) for p in plans),
           'unaligned':sum(p['unaligned'] for p in plans),'long':sum(p['long'] for p in plans),
           'raw_population':sum(p['raw_population'] for p in plans),'eligible_population':sum(p['population'] for p in plans),
           'random_stratum_weight_document_quantiles':np.quantile(weights,[0,.5,.95,1]).tolist(),
           'min_q':1/max(weights),'plan_seed_index':'original prepared document row'}
    write_json(out/'sampling_report.json',stats);print(json.dumps(stats),flush=True)
    # Coverage diagnostics retain all dev gold, even though absent from the universe.
    devcoverage={'gold':0,'representable':0,'unaligned':0,'long':0}
    for i,row in enumerate(dev):
        b=devcache.b(i);starts={int(x):j for j,(x,y) in enumerate(b)};ends={int(y):j+1 for j,(x,y) in enumerate(b)}
        for g in row['entities']:
            devcoverage['gold']+=1
            if g['start'] not in starts or g['end'] not in ends:devcoverage['unaligned']+=1
            elif ends[g['end']]-starts[g['start']]>16:devcoverage['long']+=1
            else:devcoverage['representable']+=1
    del train,old,oldrows
    head=SpanHead().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    config['trainable_parameters']=sum(p.numel() for p in head.parameters());write_json(out/'config.json',config)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024);logs=[];results={};selected_documents={}
    for epoch in range(1,a.epochs+1):
        head.train();order=list(range(len(plans)));random.Random(a.seed+epoch).shuffle(order)
        total=0.;population=0;retries=0;strata={name:[0.,0,0] for name in ('positive','near','old_fp','random')};epstart=time.monotonic()
        for begin in range(0,len(order),32):
            ids=order[begin:begin+32];h,prefix=padded_hidden([traincache.h(i) for i in ids]);samples=[sample_plan(plans[i],a.seed,epoch,i) for i in ids]
            pairs=np.concatenate([universe(len(traincache.h(i)))[s[0]] for i,s in zip(ids,samples)])
            dd=np.repeat(np.arange(len(ids)),[len(s[0]) for s in samples]);targets=np.concatenate([s[1] for s in samples]);tags=np.concatenate([s[2] for s in samples]);ww=np.concatenate([s[3] for s in samples])
            denom=sum(plans[i]['population'] for i in ids)
            if denom<=0:raise ValueError('Empty eligible batch')
            batch_total,batch_strata,attempts=optimize_batch(head,opt,scaler,h,prefix,pairs,dd,targets,tags,ww,denom)
            total+=batch_total;retries+=attempts;population+=denom
            for name,v in batch_strata.items():
                for j in range(3):strata[name][j]+=v[j]
            if begin%(32*100)==0:print(f'Epoch{epoch} train {min(begin+32,len(order))}/{len(order)} weighted_ce={total/population:.5f}',flush=True)
        log={'epoch':epoch,'weighted_ce':total/population,'seconds':time.monotonic()-epstart,'amp_retries':retries,'final_scale':scaler.get_scale(),
             'strata':{k:{'ce':v[0]/v[1] if v[1] else None,'accuracy':v[2]/v[1] if v[1] else None,'count':v[1]} for k,v in strata.items()}}
        logs.append(log);write_json(out/'training_log.json',logs);torch.save(head.state_dict(),out/f'epoch{epoch}.pt')
        print(json.dumps(log),flush=True)
        if a.replication and epoch!=a.epochs:continue
        result,documents=evaluate(head,devcache,dev,out,epoch);results[str(epoch)]=result;selected_documents[str(epoch)]=documents
    selected=max(results,key=lambda e:results[e]['dev']['known_k']['slot_accuracy']);metric=results[selected]['dev']['known_k']['slot_accuracy']
    boot={}
    for label,file in [('BIO4',ROOT/'output/bitrase-1.algo-6/v1/epoch4_known_k_dev_predictions.parquet'),('ensemble',ROOT/'output/bitrase-1.algo-6/ensemble_4_8/known_k_dev_predictions.parquet')]:
        boot[label]=bootstrap(selected_documents[selected],file)
    if provenance()!=hashes:raise ValueError('Sources changed during training')
    # Error decomposition must still be reviewed before calling the numerical research gate accepted.
    final={'selected_epoch':int(selected),'epochs':results,'bootstrap':boot,'sampling':stats,'dev_coverage':devcoverage,
           'research_positive':metric>=.7942251804631105+.005,'promotion_candidate':metric>.8040569044717353,
           'seconds':time.monotonic()-started,'selected_model_sha256':sha(out/f'epoch{selected}.pt'),'pretrained':False,'uploaded':False}
    write_json(out/'report.json',final);print(json.dumps({k:v for k,v in final.items() if k not in ('epochs','sampling')}),flush=True)

if __name__=='__main__':main()
