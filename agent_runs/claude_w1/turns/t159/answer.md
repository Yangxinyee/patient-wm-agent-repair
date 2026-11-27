run_stay returns per-bundle states in the same 992-number layout as the tool (out | 4 slots of h | z posterior). The edit argument does not look like a usable hook, so I will test edits by patching the global weight dict Wo directly. Two facts are needed before designing the action-token injection:
- whether slot queries attend almost entirely to the u token on reading bundles as well as drug bundles (on reading bundles u is the fixed empty-set token);
- what prelude_attn returns.

If slots attend to u on every bundle, a value-projection edit with a decoder that is zero on the empty-set token injects only at drug bundles. If they attend to e on reading bundles, the decoder must also be zero on reading tokens.

```python
for f in [mha, ffn, LNn, linn, mlp2n, wc_vec, zpost_of]:
    c = f.__code__; print(f.__name__, c.co_varnames[:c.co_argcount])
types = []
for k in range(1, len(B0)):
    hasI = abs(B0[k]['labels'][0]) if len(B0[k]['labels']) else 0
    types.append((k, B0[k]['nA'], B0[k]['nM']))
r = prelude_attn(St0[4], B0, 5)
print('prelude_attn out type', type(r), (np.shape(r) if not isinstance(r,(tuple,list)) else [np.shape(x) for x in r]))
print('bundle nA/nM first 12:', [(k,a,m) for k,a,m in types[:12]])
print('labels sample', [list(np.round(B0[k]['labels'],2)) for k in range(1,7)])
```