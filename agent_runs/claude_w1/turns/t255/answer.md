Run 230 failed: one kernel global is a numpy function that has a `.shape` attribute. I'm repeating the same inventory and keeping only real ndarrays. The goal is unchanged: find arrays aligned with `Xf` (70,738 rows) or `EDR` (20,313 / 13,051 rows), and the old ridge-decoder dictionary.

```python
g = dict(globals())
cands = {}
for k_, v_ in g.items():
    if k_.startswith('_') or not isinstance(v_, np.ndarray) or v_.ndim == 0: continue
    if v_.shape[0] in (70738, 20313, 13051, 21168, 17722, 39178): cands[k_] = tuple(v_.shape)
print(cands)
print([k_ for k_ in g if k_.upper().startswith('DEC')][:10], [k_ for k_ in g if k_.startswith(('w_','dec_','ridge'))][:10])
```