The context is coming back. effects.json also holds an NE impulse response on MAP (gam_ne: +3.16, +2.63, +1.78, +0.95, +0.46 at lags 1–5, presumably mmHg per mcg/kg/min). The kernel has helper functions (load_file, zsc, fbin, lin, mlp, ln, atom_v and more past the cut), so I seem to have been re-implementing parts of the model in numpy. Next I'll print the rest of the globals, the fwd_2 summary, and the source of my helper functions, to see what run 38's 'gaponly' / 'bias+gap' search measured.

```python
import inspect, types
gl = [(k,v) for k,v in list(globals().items()) if not k.startswith('_') and not isinstance(v, types.ModuleType)]
keys = [k for k,_ in gl]
i0 = keys.index('atom_v')
for k,v in gl[i0:]:
    if callable(v) and not hasattr(v,'shape'):
        print('FUNC', k); continue
    s=''
    if hasattr(v,'shape'): s=str(tuple(v.shape))
    elif isinstance(v,(list,dict,tuple,set)): s='len=%d'%len(v)
    elif isinstance(v,(int,float,str)): s=repr(v)[:100]
    print(k, type(v).__name__, s)
print("=====fwd_2 summary=====")
print(open('/work/fwd_2/summary.json').read()[:1500])
print("=====fwd_1 summary=====")
print(open('/work/fwd_1/summary.json').read()[:1500])
for fn in ['load_file','zsc','fbin','lin','mlp','ln','atom_v']:
    try:
        print('-----',fn); print(inspect.getsource(globals()[fn]))
    except Exception as e:
        print(fn, 'ERR', e)
```