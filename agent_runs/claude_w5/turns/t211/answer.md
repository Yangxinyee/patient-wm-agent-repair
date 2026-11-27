run_tf returns, per bundle, a dict of the core-step inputs (out0, h0, dt, e, u, Mv, …), so I can replay any bundle through cstep_h myself. A bundle is (time in hours, list of atoms). A run-185 `rows` entry holds a (4×768) tensor (probably the last-layer FFN hidden activations per slot) plus labels.

Next I print the full key list of a step dict, call cstep_h on one bundle with `rec={}`, and list what it records (names and shapes) and what it returns. If rec gives me the prelude internals (attention tokens, n3, the slot residual P, E) and h_new, I can build both edit stages and the injection Jacobian from it.

```python
d = res[1]
print('step keys', {k: desc(v) for k, v in d.items()})
rec = {}
o = cstep_h(d['out0'], d['h0'], d['dt'], d['e'], d['u'], d['Mv'], d.get('Mage'), rec=rec)
print('ret', desc(o), ([desc(x) for x in o] if isinstance(o, (list, tuple)) else ''))
print('rec', {k: desc(v) for k, v in rec.items()})
```