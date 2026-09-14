import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sequence_crf import tokens,gold_tags,spans_from_tags,decode_k


class SequenceTests(unittest.TestCase):
    def test_alignment_unicode(self):
        text='Dear Éva Chen, login: eva_3.'
        ts=tokens(text);a=text.index('Éva');b=text.index(',')
        tags,bad=gold_tags(ts,[{'start':a,'end':b,'label':'NAME'}])
        self.assertEqual(bad,0);self.assertEqual(spans_from_tags(ts,tags),[(a,b,'NAME')])
        _,bad=gold_tags(ts,[{'start':a,'end':a+1,'label':'NAME'}]);self.assertEqual(bad,1)

    def test_counted_viterbi_exhaustive(self):
        rng=np.random.default_rng(2026);labels=['O','B-NAME','I-NAME','B-DATE','I-DATE']
        for _ in range(30):
            n=4;k=int(rng.integers(0,4));em=rng.normal(size=(n,5));tr=rng.normal(size=(5,5));best=-np.inf
            for path in itertools.product(range(5),repeat=n):
                tags=[labels[j] for j in path]
                if sum(t.startswith('B-') for t in tags)!=k:continue
                if any(t.startswith('I-') and (i==0 or tags[i-1] not in ('B-'+t[2:],t)) for i,t in enumerate(tags)):continue
                score=sum(em[i,j] for i,j in enumerate(path))+sum(tr[a,b] for a,b in zip(path,path[1:]));best=max(best,score)
            tags=decode_k(em,tr,labels,k);path=[labels.index(t) for t in tags]
            score=sum(em[i,j] for i,j in enumerate(path))+sum(tr[a,b] for a,b in zip(path,path[1:]))
            self.assertAlmostEqual(score,best)


if __name__=='__main__':unittest.main()
