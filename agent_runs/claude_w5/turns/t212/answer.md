The step dict names the memory ages 'Ma', not 'Mage', so my call passed None. It also carries 'res', the 6-tuple the core step returned. Next I retry the probe with d['Ma'] and list what cstep_h records and returns. I expect a 6-tuple (out, y, z, E, hf, …) and a rec dict of internal activations.

```python
rec = {}
o = cstep_h(d['out0'], d['h0'], d['dt'], d['e'], d['u'], d['Mv'], d['Ma'], rec=rec)
print('ret', desc(o), ([desc(x) for x in o] if isinstance(o, (list, tuple)) else ''))
print('rec', {k: desc(v) for k, v in rec.items()})
print('res stored', [desc(x) for x in d['res']])
print('match out', float((o[0]-d['res'][0]).abs().max()) if isinstance(o,(list,tuple)) else '')
```