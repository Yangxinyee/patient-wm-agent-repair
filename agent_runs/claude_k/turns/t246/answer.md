Run 229 shows `mk_best` fails whichever way I call it. Like `L` before, a global its helper `atom_vec` depends on has probably been overwritten during this session.

I'll list the global names that `atom_vec`, `fbin` and `zsc` use, with each one's current type. Any name that should be a function or dict but now holds a string, int or list is the culprit to restore.

```python
for fn_n in ['atom_vec','fbin','zsc']:
    f_=globals()[fn_n]; cn=f_.__code__.co_names
    print(fn_n, [(c,type(globals()[c]).__name__) for c in cn if c in globals() and not isinstance(globals()[c],(types.ModuleType,types.BuiltinFunctionType))][:20])
print('CODE sample',list(CODE.items())[:4])
```