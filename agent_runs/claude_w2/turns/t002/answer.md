The architecture is now clear. Doses enter only through the action encoder (`encode_action.*`, producing u_k) and the embedding terms for rate and dose (`embed.numeric` columns for z_dose and z_rate, `embed.bins.dose` and `embed.bins.rate`). Glucose and MAP predictions come from the value head, which reads `out` and z, plus the last-value register. So any dose-to-outcome pathway runs from u_k (and the action atom embeddings) through the core into later `out` states.

Next I'll read the data README and the vocabulary (codes, normalisation, bin edges and centres) and list the tensors with their shapes. This will tell me which records I can use to estimate the true drug effects, and which weights carry the dose signal.

```python
print(open('/data/README.md').read())
V = VOCAB
print(V.keys())
for k in V:
    s = json.dumps(V[k])
    print(k, ':', s[:1500])
    print()
```