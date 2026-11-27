Every call pattern fails now, though the same call raised no error in run 174. Something used by `view_from` or its helpers must differ from what I assume. Candidates are the global `L` used by `mlpf` and the restored `T`.

To see where it breaks, I'll print the full stack of frames (function names and line numbers) and a few objects `mlpf` depends on (`L`, `gelu`, the `T` entries for `gap`).

```python
try:
    view_from(X8o[kidx[0]][None,:], np.array([3600.0]))
except Exception:
    tb_=traceback.extract_tb(sys.exc_info()[2]) if 'sys' in globals() else None
    import sys
    tb_=traceback.extract_tb(sys.exc_info()[2])
    for fr in tb_: print(fr.name, fr.lineno)
print('L is', L, getattr(L,'__code__',None) and (L.__code__.co_varnames[:L.__code__.co_argcount], L.__code__.co_names))
print('gelu', gelu, '| T gap.0.weight', T['gap.0.weight'].shape, T['gap.0.weight'].dtype)
```