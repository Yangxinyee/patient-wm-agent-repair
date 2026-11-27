dt is in seconds: the decay timescales run from 573 s (about 10 minutes) to 5.9×10^7 s (about 1.9 years). a_bar is 0.29-0.43 and softplus(delta_in) is about 1. Next I'll re-derive the encoder inputs from the vocabulary rather than from guessed indices. I'll print VOCAB's keys, its config, the code entries and any unit or category tables. If VOCAB gives an explicit unit-index mapping, it replaces the indices I chose by search last session, which may explain why the z-posterior match wasn't exact.

```python
print(list(VOCAB.keys()))
print(VOCAB['config'])
print([ (c['name'], c.get('id'), c.get('event_kind')) for c in VOCAB['codes'] ])
for k in VOCAB:
    if k not in ('codes','config','numeric','value_bin_centres'):
        print(k, str(VOCAB[k])[:300])
```