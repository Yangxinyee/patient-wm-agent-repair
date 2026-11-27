For the Xf rows, uf is the observed value, pf the previous value, lbf the last bin, cf the code and sf the stay. So I can compute the replica's factual NLL on 70,738 train rows. Next I will build a head function, logits = W2·gelu(W1·x + b1) + b2 + transition[code, last bin], and check its per-code NLL against the tool's setA baseline (finger 1.53, lab 1.26, arterial 1.35, cuff 1.75).

```python
print('CODE',CODE); print('VOCAB keys',list(VOCAB.keys()))
vb=VOCAB.get('value_bin_edges', None); print(type(vb).__name__, (list(vb.keys())[:6] if isinstance(vb,dict) else ''))
```