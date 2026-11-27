The inventory was cut off at the function `lagm`. The rest of the functions (e.g. `run_stay`), the arrays and lists, and the head of the notebook are still missing. I will print only those parts, compactly. I am looking for the objects from runs 100–125: any per-stay generated samples, decoders, or coefficient estimates. I need them to decide whether to continue with the plan in the notes (a head-level edit with policy-centred dose innovations) or rebuild it.

```python
rows2 = []
for k, v in g.items():
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    t = type(v).__name__
    if t in ('DataFrame','Series','dict','bool','float','float64','Tensor','ABCMeta','type','builtin_function_or_method','Counter','DataFrameGroupBy','Generator','Index','Interval','MultiIndex','NoneType','NpzFile','TextIOWrapper','GroupKFold'): continue
    if t == 'function' and k < 'lagm': continue
    try:
        if isinstance(v, types.FunctionType):
            c = v.__code__; info = f"{c.co_varnames[:c.co_argcount]}"
        elif hasattr(v, 'shape'): info = str(tuple(v.shape))
        elif isinstance(v, (list, tuple, set)): info = f'len={len(v)} first={type(next(iter(v))).__name__ if len(v) else None}'
        elif isinstance(v, (int, str)): info = repr(v)[:50]
        else: info = ''
    except Exception: info = '?'
    rows2.append(f"{t}:{k}{info}")
print(' | '.join(sorted(rows2)))
s = open('/work/notes.md').read()
print('----- notes len', len(s))
print(s[:max(0, len(s)-8000)][:4500])
```