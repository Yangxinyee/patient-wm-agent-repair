Time-flow timescales run from 587 s to 6.4e7 s, so dt is in seconds, which is what my replica assumes. The states come from fwd_2 (original model, batch1 stays), with bundles at integer hours (readings) and hour+0.0833 (infusion starts). Next I'll find the functions I wrote after the encoder fix (unit mapping, bare END atoms) so I can reuse them rather than rewrite them. A short listing of function names is enough.

```python
fn = [k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None) in (None,'__main__')]
print(fn)
print('UNIT_FIX' in globals() and UNIT_FIX, 'ACTMAP', ACTMAP)
```