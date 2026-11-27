Run 40 failed because the earlier listing was cut mid-name: 'atom_v' was probably the start of a longer name. I'll match by prefix instead and print everything after it, plus the summaries and helper sources.

```python
import inspect, types
gl = [(k,v) for k,v in list(globals().items()) if not k.startswith('_') and not isinstance(v, types.ModuleType)]
keys = [k for k,_ in gl]
i0 = [i for i,k in enumerate(keys) if k.startswith('atom_v')][0]
for k,v in gl[i0:]:
    if callable(v) and not hasattr(v,'shape'):
        print('FUNC', k); continue
    s=''
    if hasattr(v,'shape'): s=str(tuple(v.shape))
    elif isinstance(v,(list,dict,tuple,set)): s='len=%d'%len(v)
    elif isinstance(v,(int,float,str)): s=repr(v)[:100]
    print(k, type(v).__name__, s)
print("=====fwd_2 summary=====")
print(open('/work/fwd_2/summary.json').read()[:1200])
print("=====fwd_1 summary=====")
print(open('/work/fwd_1/summary.json').read()[:1200])
for fn in ['load_file','zsc','fbin','lin','mlp','ln']+[k for k in keys if k.startswith('atom_v')]:
    try:
        print('-----',fn); print(inspect.getsource(globals()[fn]))
    except Exception as e:
        print(fn, 'ERR', e)
```