The rebuilt bundles are lists of (time, atoms) with code, kind and value. The config says 4 attention heads, FFN width ×4, one prelude layer, two recurrent layers, one coda layer, a memory of 128 and z of 32 dimensions.

To re-implement the embedding I need the exact normalisation (how z-scores are computed) and the bin rules. Next I'll print the vocab's numeric section, notes and code list, kept short.

```python
for k_ in ['codes','event_kinds','atom_kinds','numeric_note','value_bin_note']:
    print(k_, json.dumps(VOCAB[k_])[:600])
num=VOCAB['numeric']; print(type(num).__name__, list(num.keys())[:20] if isinstance(num,dict) else len(num))
k0=list(num.keys())[0] if isinstance(num,dict) else 0
print(json.dumps(num[k0])[:700])
```