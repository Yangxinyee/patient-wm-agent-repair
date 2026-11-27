The architecture is back in view, but the variable listing crashed on a numpy function object. Two details matter for the edit:
- The value head reads a "last value" register directly, so predicted glucose depends heavily on the previous reading.
- The state update takes the z posterior mean through `z_in(z_k)` in the prelude.

This run lists the kernel variables safely. I also want my helper functions back before building the injection.

```python
import types
out = []
for k, v in list(globals().items()):
    if k.startswith('_') or k in ('In','Out','exit','quit','get_ipython'):
        continue
    try:
        if isinstance(v, (types.ModuleType,)):
            continue
        if callable(v) and not hasattr(v, 'shape'):
            out.append(f"{k}()")
        elif hasattr(v, 'shape'):
            out.append(f"{k}:{type(v).__name__}{tuple(v.shape)}")
        elif isinstance(v, (int, float, str, bool)):
            s = str(v)
            out.append(f"{k}={s[:30]}")
        elif isinstance(v, (list, tuple, dict, set)):
            out.append(f"{k}:{type(v).__name__}[{len(v)}]")
        else:
            out.append(f"{k}:{type(v).__name__}")
    except Exception as e:
        out.append(f"{k}:?")
print(' | '.join(out))
```