import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neural_sequence import build_vocab, encode_chars, encode_document, collate_encoded, LABEL_TO_ID,build_model
class NeuralSequenceTests(unittest.TestCase):
    def test_unicode_byte_encoding_is_capped_and_deterministic(self):
        self.assertEqual(encode_chars('é😀',32).dtype,np.uint8); self.assertEqual(len(encode_chars('x'*100,32)),32)
    def test_train_vocab_lowercases_and_unknowns(self):
        v=build_vocab(['Alice ALICE']); _,ids,_,_=encode_document('alice unseen',v); self.assertEqual(ids[0],v['alice']); self.assertEqual(ids[1],1)
    def test_collate_padding_and_mask(self):
        v=build_vocab(['A B']); a=encode_document('A B',v); b=encode_document('A',v); batch=collate_encoded([(a[0],a[1],a[2],a[3],np.array([LABEL_TO_ID['O']]*2)),(b[0],b[1],b[2],b[3],np.array([LABEL_TO_ID['O']]))]); self.assertEqual(tuple(batch['words'].shape),(2,2)); self.assertEqual(batch['mask'].tolist(),[[True,True],[True,False]]); self.assertEqual(batch['labels'][1,1].item(),-100)
    def test_padding_does_not_change_valid_logits(self):
        import torch
        torch.set_num_threads(2);torch.manual_seed(2026)
        v=build_vocab(['A B C']);docs=[]
        for text in ('A B C','A'):
            ts,ids,ch,fl=encode_document(text,v);docs.append((ts,ids,ch,fl,np.zeros(len(ids),dtype=np.int64)))
        model=build_model(len(v));model.eval()
        with torch.no_grad():
            batch=collate_encoded(docs);both=model(batch['words'],batch['chars'],batch['flags'],batch['mask'].sum(1))
            batch=collate_encoded(docs[1:]);single=model(batch['words'],batch['chars'],batch['flags'],batch['mask'].sum(1))
        torch.testing.assert_close(both[1,:1],single[0],atol=1e-6,rtol=1e-5)
if __name__=='__main__': unittest.main()
