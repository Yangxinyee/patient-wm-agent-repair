The action bundle at hour 1.083 holds 4 atoms: the start and end records for each drug. My helper `atom_vec` takes code, kind, event kind, unit row, z-scores, presence flags and the fine bins of value, dose and rate, so I can rebuild action atoms if I get the vocabulary conventions right.

I'll print the vocab entries needed for that: the top-level keys, the code list, and the normalisation entries for the two drugs (robust z-score centre and scale for rate and dose, and the bin edges). This gives the exact inputs the action encoder sees, including how they change when a rate is scaled. If the normalisation is linear in the rate, a rate-proportional channel can start right at the embedding.

```python
print(list(VOCAB.keys()))
print('codes', VOCAB['codes'] if not isinstance(VOCAB['codes'], dict) else list(VOCAB['codes'].items())[:14])
for k in VOCAB:
    v = VOCAB[k]
    if isinstance(v, dict) and 'insulin' in v:
        print(k, '| insulin:', str(v['insulin'])[:200])
```