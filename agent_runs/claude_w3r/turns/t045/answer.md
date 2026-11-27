The tensor list gives the full structure.

- The value head is Linear(192→192), an activation, then Linear(192→17). Its last layer is linear in the hidden activations, so a rank-one change there could later tilt the glucose distribution.
- The core is a pre-norm transformer prelude, a 2-layer recurrent block (self-attention over slots, cross-attention to memory, FFN) and a 1-layer coda over the slots. Writing a persistent dose memory into it would need a local implementation that matches the states.
- `last_numeric` has 8 inputs, not the 4 the doc lists. There are several categorical tables I never added: action, route, status, discharged_to, text and kq. Omitted constant terms like these could explain why my z posterior didn't match.

Next I will check which embedding tables have an all-zero row 0, meaning index 0 is padding for "absent", and read the data README and the vocab's code and categorical entries. That tells me which terms each atom really gets. If, say, the categorical tables have non-zero row 0, then every atom gets a constant term I left out. That would be a likely cause of the mismatch.

```python
from safetensors.numpy import load_file
for n in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight']+[n for n in NAMES if n.startswith('embed.categorical')]:
    W_=load(n); print(n, list(W_.shape), 'row norms', np.round(np.linalg.norm(W_,axis=1),2)[:8])
print(open('/data/README.md').read()[:3000])
print(list(VOCAB.keys()))
print({k:VOCAB[k] for k in VOCAB if k not in ('numeric','value_bin_centres','codes')})
print(VOCAB['codes'])
```