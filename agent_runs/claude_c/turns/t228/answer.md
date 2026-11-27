This is the key finding of the breakdown: numeric ×5's cost sits almost entirely in readings within 1.5 h after a norepinephrine record. There it costs finger glucose +0.007, lab glucose +0.0125 and arterial MAP +0.006. Readings shortly after insulin cost about nothing. So amplifying insulin's dose features is nearly free; amplifying norepinephrine's is what hurts.

The numeric columns are shared by both drugs. The per-drug handle could be the z-scores, which are scaled per code. If norepinephrine's dose or rate z-scores span a much wider range than insulin's, one column may cause the cost and could be left alone. I'll check the normalisation constants and the observed z-score ranges for both drugs in the vocab.

```python
nv = VOCAB.get('normalisation', VOCAB.get('normalization', None)); print(list(VOCAB.keys())[:12])
key_ = [k_ for k_ in VOCAB.keys() if 'norm' in k_.lower()]; print(key_)
nrm_ = VOCAB[key_[0]] if key_ else None
print(json.dumps({k_: v_ for k_, v_ in (nrm_.items() if isinstance(nrm_, dict) else []) if any(s in str(k_) for s in ['insulin','norepi'])})[:800])
```