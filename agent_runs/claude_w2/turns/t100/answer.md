The variable list was cut off at 10,000 characters, just where the most recent objects from runs 82 to 93 begin. Those include my verification functions (z_check, z_check2 and whatever produced the "e_route" result). Next I print only the variables defined after `kw`, plus the source of the newest functions if `inspect` can retrieve it. That shows exactly what the local re-implementation covers. If the source is retrievable I can reuse it directly. If not, I will rebuild the needed parts using ARCHITECTURE.md.

```python
keys=list(globals().keys())
i0=keys.index('kw')
tail=[k for k in keys[i0:] if not k.startswith('_')]
for k in tail:
    v=globals()[k]
    t=type(v).__name__
    info=''
    if hasattr(v,'shape'): info=str(tuple(v.shape))
    elif isinstance(v,(list,tuple,dict,set)): info='len=%d'%len(v)
    elif callable(v):
        try: info=str(inspect.signature(v))[:80]
        except Exception: info='?'
    print(k,t,info)
# try to get source of the most recent functions
for fnn in [k for k in tail if callable(globals()[k]) and isinstance(globals()[k], types.FunctionType)][-6:]:
    try:
        src=inspect.getsource(globals()[fnn]); print('----',fnn,len(src))
    except Exception as ex: print('----',fnn,'nosrc',ex)
```