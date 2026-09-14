"""Train-from-scratch word/punctuation BIO CRF helpers and exactly-K decoder."""
import itertools
import re
import numpy as np

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokens(text):
    return [(m.start(), m.end(), m.group()) for m in TOKEN.finditer(text)]


def shape(word):
    raw = ''.join('X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in word)
    return ''.join(c for c, _ in itertools.groupby(raw))


def features(text, ts):
    words = [w.lower() for a, b, w in ts]
    shapes = [shape(w) for a, b, w in ts]
    seq = []
    for i, (a, b, word) in enumerate(ts):
        lower = words[i]
        f = ['bias', 'w=' + lower, 'shape=' + shapes[i], 'len=' + str(min(len(word), 20)),
             'upper=' + str(word.isupper()), 'title=' + str(word.istitle()), 'digit=' + str(word.isdigit())]
        if len(word) > 1:
            for n in (2, 3, 4):
                f.extend(['pre' + str(n) + '=' + lower[:n], 'suf' + str(n) + '=' + lower[-n:]])
        for delta in (-2, -1, 1, 2):
            j = i + delta
            if 0 <= j < len(ts):
                f.extend([f'w{delta}=' + words[j], f's{delta}=' + shapes[j]])
            else: f.append(f'edge{delta}')
        left_gap = text[ts[i-1][1]:a] if i else text[:a]
        right_gap = text[b:ts[i+1][0]] if i+1 < len(ts) else text[b:]
        f.extend(['left_newline=' + str('\n' in left_gap), 'right_newline=' + str('\n' in right_gap),
                  'left_joined=' + str(not left_gap), 'right_joined=' + str(not right_gap)])
        seq.append(f)
    return seq


def gold_tags(ts, entities):
    starts = {a: i for i, (a, b, w) in enumerate(ts)}
    ends = {b: i for i, (a, b, w) in enumerate(ts)}
    tags = ['O'] * len(ts)
    unaligned = 0
    for e in entities:
        if e['start'] not in starts or e['end'] not in ends:
            unaligned += 1
            continue
        i, j = starts[e['start']], ends[e['end']]
        if i > j or any(t != 'O' for t in tags[i:j+1]): raise ValueError('Invalid gold overlap')
        tags[i] = 'B-' + e['label']
        tags[i+1:j+1] = ['I-' + e['label']] * (j-i)
    return tags, unaligned


def spans_from_tags(ts, tags):
    spans = []
    current = None
    for (a, b, word), tag in zip(ts, tags):
        if tag == 'O':
            if current is not None: spans.append(tuple(current)); current = None
        elif tag.startswith('B-') or current is None or current[2] != tag[2:]:
            if current is not None: spans.append(tuple(current))
            current = [a, b, tag[2:]]
        else: current[1] = b
    if current is not None: spans.append(tuple(current))
    return spans


def model_arrays(tagger):
    labels = tagger.labels()
    index = {s: i for i, s in enumerate(labels)}
    info = tagger.info()
    lookup = {}
    for (feature, label), weight in info.state_features.items():
        lookup.setdefault(feature, []).append((index[label], weight))
    transitions = np.zeros((len(labels), len(labels)))
    for (a, b), w in info.transitions.items(): transitions[index[a], index[b]] = w
    return labels, lookup, transitions


def emissions(seq, lookup, nlabels):
    result = np.zeros((len(seq), nlabels))
    for i, fs in enumerate(seq):
        for f in fs:
            for j, w in lookup.get(f, ()):
                result[i, j] += w
    return result


def decode_k(emission, transitions, labels, count):
    """Maximum CRF path score with legal BIO transitions and exactly K B tags."""
    n, l = emission.shape
    if count < 0 or count > n: raise ValueError('Infeasible entity count')
    if not n:
        if count: raise ValueError('Empty tokens')
        return []
    trans = transitions.copy()
    is_b = np.array([s.startswith('B-') for s in labels])
    is_i = np.array([s.startswith('I-') for s in labels])
    for j, b in enumerate(labels):
        if b.startswith('I-'):
            for i, a in enumerate(labels):
                if a not in ('B-' + b[2:], b): trans[i, j] = -np.inf
    prev = np.full((count + 1, l), -np.inf)
    for j in range(l):
        k = int(is_b[j])
        if not is_i[j] and k <= count: prev[k, j] = emission[0, j]
    back = np.zeros((n, count + 1, l), dtype=np.int16)
    for t in range(1, n):
        all_scores = prev[:, :, None] + trans[None, :, :]
        arg = all_scores.argmax(axis=1)
        best = np.take_along_axis(all_scores, arg[:, None, :], axis=1)[:, 0, :]
        current = np.full_like(prev, -np.inf)
        current[:, ~is_b] = best[:, ~is_b] + emission[t, ~is_b]
        back[t][:, ~is_b] = arg[:, ~is_b]
        if count:
            current[1:, is_b] = best[:-1, is_b] + emission[t, is_b]
            back[t][1:, is_b] = arg[:-1, is_b]
        prev = current
    j = int(prev[count].argmax())
    if not np.isfinite(prev[count, j]): raise ValueError('No feasible BIO path')
    result = [j]
    k = count
    for t in range(n-1, 0, -1):
        new_j = int(back[t, k, j])
        k -= int(is_b[j])
        j = new_j
        result.append(j)
    return [labels[j] for j in reversed(result)]
