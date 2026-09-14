"""From-scratch compact token/byte BiLSTM BIO model utilities."""
from __future__ import annotations
import collections
import numpy as np
from sequence_crf import tokens, gold_tags, spans_from_tags, decode_k

LABELS = ("O",) + tuple(x for label in ("NAME", "DATE", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "JOB_TITLE") for x in ("B-" + label, "I-" + label))
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}

def build_vocab(texts, max_size=50000):
    counts = collections.Counter(w.lower() for text in texts for _, _, w in tokens(text))
    words = [w for w, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_size-2]]
    return {"<PAD>": 0, "<UNK>": 1, **{w: i + 2 for i, w in enumerate(words)}}

def encode_chars(word, width=32):
    out = np.zeros(width, dtype=np.uint8)
    raw = word.encode("utf-8")[:width]
    out[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out

def encode_document(text, vocab, char_width=32):
    ts = tokens(text)
    ids = np.asarray([vocab.get(w.lower(), 1) for _, _, w in ts], dtype=np.int64)
    chars = np.asarray([encode_chars(w, char_width) for _, _, w in ts], dtype=np.uint8)
    flags = np.asarray([[float(w.isupper()), float(w.isdigit()), float(w.istitle()),
                         float('\n' in text[ts[i-1][1] if i else 0:a]),
                         float('\n' in text[b:ts[i+1][0] if i+1<len(ts) else len(text)]),
                         float(i>0 and ts[i-1][1]==a),float(i+1<len(ts) and b==ts[i+1][0])]
                        for i,(a,b,w) in enumerate(ts)], dtype=np.float32)
    return ts, ids, chars, flags

def tags_to_ids(ts, entities):
    tags, unaligned = gold_tags(ts, entities)
    return np.asarray([LABEL_TO_ID[t] if t in LABEL_TO_ID else -100 for t in tags], dtype=np.int64), unaligned

def collate_encoded(batch, pad_id=0):
    import torch
    n = len(batch); length = max((x[1].shape[0] for x in batch), default=0); width = batch[0][2].shape[1] if batch else 32
    words = np.full((n, length), pad_id, dtype=np.int64); chars = np.zeros((n, length, width), dtype=np.uint8); flags = np.zeros((n, length, 7), dtype=np.float32); labels = np.full((n, length), -100, dtype=np.int64); mask = np.zeros((n, length), dtype=bool)
    for i, (ts, ids, ch, fl, y) in enumerate(batch):
        m = len(ids); words[i,:m], chars[i,:m], flags[i,:m], labels[i,:m], mask[i,:m] = ids, ch, fl, y, True
    return {"words": torch.from_numpy(words), "chars": torch.from_numpy(chars), "flags": torch.from_numpy(flags), "labels": torch.from_numpy(labels), "mask": torch.from_numpy(mask)}

def decode_logits(logits, lengths, documents, expected_counts):
    """Decode logits with legal BIO transitions and zero transition weights."""
    result=[]
    for i, length in enumerate(lengths):
        ts = documents[i][0]; emission = logits[i,:length].detach().cpu().numpy();
        raw_ids = emission.argmax(axis=1); raw_tags = [LABELS[x] for x in raw_ids]
        raw = spans_from_tags(ts, raw_tags); valid = all(t == "O" or not t.startswith("I-") or (j and raw_tags[j-1] in ("B-"+t[2:], t)) for j,t in enumerate(raw_tags))
        if valid and len(raw) == expected_counts[i]: chosen = raw
        else:
            chosen = spans_from_tags(ts, decode_k(emission, np.zeros((len(LABELS),len(LABELS))), LABELS, expected_counts[i]))
        result.append(chosen)
    return result

def build_model(vocab_size, char_vocab_size=256, embedding_dim=128, char_dim=16, char_channels=64, hidden=128, flags_dim=7, labels=len(LABELS)):
    import torch
    import torch.nn as nn
    class ByteCNNBiLSTM(nn.Module):
        def __init__(self):
            super().__init__(); self.word=nn.Embedding(vocab_size,embedding_dim,padding_idx=0); self.char=nn.Embedding(char_vocab_size,char_dim,padding_idx=0); self.conv=nn.Conv1d(char_dim,char_channels,3,padding=1); self.lstm=nn.LSTM(embedding_dim+char_channels+flags_dim,hidden,num_layers=2,bidirectional=True,dropout=.2,batch_first=True); self.out=nn.Linear(hidden*2,labels)
        def forward(self, words, chars, flags, lengths=None):
            b,t,w=chars.shape; c=self.char(chars.long().reshape(b*t,w)).transpose(1,2); c=torch.relu(self.conv(c)).amax(dim=2).reshape(b,t,-1); x=torch.cat((self.word(words),c,flags),dim=-1)
            if lengths is not None:
                packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False); packed,_=self.lstm(packed); x,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=t)
            else: x,_=self.lstm(x)
            return self.out(x)
    return ByteCNNBiLSTM()
