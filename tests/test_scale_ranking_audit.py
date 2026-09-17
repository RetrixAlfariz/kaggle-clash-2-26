import itertools
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.scale_ranking_audit import _bootstrap, _ordered_matches, fit_beta, partition_stats, ranking_record, run, _best_two


def worlds(pairs, energies, k):
    out=[]
    for ids in itertools.combinations(range(len(pairs)),k):
        ordered=sorted(ids,key=lambda i:(pairs[i,1],pairs[i,0]))
        if any(pairs[ordered[j-1],1]>pairs[ordered[j],0] for j in range(1,k)):
            continue
        for labs in itertools.product(range(energies.shape[1]),repeat=k):
            path=tuple((int(pairs[i,0]),int(pairs[i,1]),int(l)) for i,l in zip(ordered,labs))
            out.append((sum(float(energies[i,l]) for i,l in zip(ordered,labs)),path))
    return out


class ScaleRankingAuditTests(unittest.TestCase):
    def test_calibration_matches_analytic_interior_and_boundary_optima(self):
        p=np.array([[0,1]]);e=np.array([[0.,1.]])
        def doc(score):
            return {'n':1,'pairs':p,'energies':e,'k':1,'supported':True,'gold_energy':score}
        fit=fit_beta([doc(1.),doc(1.),doc(0.)])
        self.assertAlmostEqual(fit['beta'],math.log(2),delta=1e-5)
        self.assertLess(fit['cal_nll_per_slot_fitted'],fit['cal_nll_per_slot_beta1'])
        self.assertEqual(fit_beta([doc(1.)])['boundary_solution'],'upper')
        self.assertEqual(fit_beta([doc(0.)])['boundary_solution'],'lower')

    def test_top_two_match_all_labeled_worlds_including_ties(self):
        p=np.array([(a,b) for b in range(1,4) for a in range(b)])
        rng=np.random.default_rng(0)
        for trial in range(30):
            e=rng.integers(-2,3,(len(p),3)).astype(float)
            for k in (1,2,3):
                expected=sorted(worlds(p,e,k),key=lambda x:(-x[0],x[1]))[:2]
                self.assertEqual(_best_two(3,p,e,k),expected)

    @staticmethod
    def _write_split(root, doc_id, group_id, gold):
        root.mkdir()
        pairs=np.array([(a,b) for b in range(1,4) for a in range(max(0,b-16),b)],dtype=np.int32)
        logits=np.zeros((len(pairs),8),dtype=np.float32)
        logits[:,1:]=np.arange(len(pairs)*7,dtype=np.float32).reshape(len(pairs),7)/100
        bounds=np.array([[0,1],[2,3],[4,5]],dtype=np.int32)
        np.save(root/'logits.npy',logits);np.save(root/'pairs.npy',pairs)
        np.save(root/'span_offsets.npy',np.array([0,len(pairs)],dtype=np.int64))
        np.save(root/'token_offsets.npy',np.array([0,3],dtype=np.int64));np.save(root/'boundaries.npy',bounds)
        (root/'metadata.json').write_text(json.dumps([{'document_id':doc_id,'group_id':group_id,'expectedK':2,
            'token_count':3,'gold':gold}]),encoding='utf-8')
        names=('logits.npy','pairs.npy','span_offsets.npy','token_offsets.npy','boundaries.npy','metadata.json')
        files=[{'name':n,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in names]
        (root/'manifest.json').write_text(json.dumps({'files':files}),encoding='utf-8')

    def test_partition_and_expected_energy_match_exhaustive_gradient(self):
        pairs=np.array([[0,1],[1,2],[2,3],[0,2],[1,3]])
        energies=np.array([[.5,-.1],[.2,.7],[-.2,.3],[.4,-.3],[.1,.2]])
        beta=1.7;k=2
        actual=energies*beta; ws=worlds(pairs,actual,k)
        mx=max(x[0] for x in ws); logz=mx+math.log(sum(math.exp(x[0]-mx) for x in ws))
        probs=[math.exp(x[0]-logz) for x in ws]
        expected=sum(pr*sum(energies[next(i for i,p in enumerate(pairs) if tuple(p)==span[:2]),span[2]] for span in w[1])
                     for pr,w in zip(probs,ws))
        z,mean=partition_stats(3,pairs,energies,k,beta)
        self.assertAlmostEqual(z,logz,places=11)
        self.assertAlmostEqual(mean,expected,places=11)
        eps=1e-5
        zp=partition_stats(3,pairs,energies,k,beta+eps)[0]
        zm=partition_stats(3,pairs,energies,k,beta-eps)[0]
        self.assertAlmostEqual((zp-zm)/(2*eps),mean,places=7)

    def test_positive_scale_preserves_exact_structure_ranking(self):
        pairs=np.array([[0,1],[1,2],[0,2]])
        energies=np.zeros((3,7));energies[0,0]=1.2;energies[1,1]=.4;energies[2,2]=1.0
        gold=((0,1,0),(1,2,1))
        a=ranking_record(2,pairs,energies,2,gold)
        b=ranking_record(2,pairs,energies*3.7,2,gold)
        self.assertEqual(a['map_is_gold'],b['map_is_gold'])
        self.assertAlmostEqual(b['gold_minus_wrong_gap'],3.7*a['gold_minus_wrong_gap'],places=10)

    def test_calibration_fit_stays_in_frozen_bounds(self):
        pairs=np.array([[0,1],[1,2],[0,2]])
        energies=np.zeros((3,7));energies[0,0]=1.;energies[1,1]=.8;energies[2,2]=.2
        d={'n':2,'pairs':pairs,'energies':energies,'k':1,'supported':True,'gold_energy':.8}
        result=fit_beta([d])
        self.assertGreaterEqual(result['beta'],.25)
        self.assertLessEqual(result['beta'],4.)
        self.assertEqual(result['iterations'],18)
        self.assertTrue(math.isfinite(result['cal_nll_per_slot_fitted']))

    def test_second_best_may_change_only_the_label(self):
        pairs=np.array([[0,1],[1,2]])
        energies=np.zeros((2,7));energies[0,0]=1.;energies[0,1]=.9;energies[1,2]=.8
        result=ranking_record(2,pairs,energies,2,((0,1,0),(1,2,2)))
        self.assertTrue(result['map_is_gold'])
        self.assertAlmostEqual(result['gold_minus_wrong_gap'],.1,places=10)

    def test_unsupported_gold_is_separated_from_ranking_failure(self):
        result=ranking_record(1,np.array([[0,1]]),np.zeros((1,7)),1,((0,1,0),(0,1,1)))
        self.assertFalse(result['gold_supported'])
        self.assertIsNone(result['gold_minus_wrong_gap'])

    def test_end_to_end_cache_reports_ordered_slots_and_pooled_gates(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);gold=[{'start':0,'end':1,'label':'NAME'},{'start':4,'end':5,'label':'DATE'}]
            self._write_split(base/'cal','cal-doc','family-cal',gold)
            self._write_split(base/'eval','eval-doc','family-eval',gold)
            run(base/'cal',base/'eval',base/'out',workers=2)
            rows=json.loads((base/'out'/'eval_documents.json').read_text(encoding='utf-8'))
            row=rows[0]
            self.assertEqual(row['mbr_tau1_correct'],sum(a==b for a,b in zip(row['gold_slots'],row['mbr_tau1_prediction'])))
            summary=json.loads((base/'out'/'eval_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['mbr']['all_eval_slots'],2)
            self.assertEqual(summary['mbr']['paired_doc_bootstrap']['n'],1)
            self.assertTrue(all(row['map_replay_equal_under_scaling'] for row in rows))

    def test_slot_order_and_heterogeneous_document_bootstrap_use_pooled_counts(self):
        gold=[(0,1,0),(4,5,1)];reversed_predictions=[gold[1],gold[0]]
        self.assertEqual(len(set(gold)&set(reversed_predictions)),2)
        self.assertEqual(sum(_ordered_matches(gold,reversed_predictions)),0)
        # One correct slot in K=1 plus zero in K=9 is 10% pooled gain,
        # not the 50% average of two document-level rates.
        self.assertAlmostEqual(_bootstrap([1,0],[1,9])['mean'],.1)


if __name__=='__main__': unittest.main()
