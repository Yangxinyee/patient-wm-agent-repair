Restoring `T` was not enough: `view_from` still fails inside, with an inhomogeneous array error. Most likely one of its helpers (`mlpf` or `LNf`) expects a different structure in `T`, or `dt` has the wrong shape.

Instead of guessing call conventions, I'll inspect the helpers' names and constants and print the full traceback. If it's only a shape issue I'll call `view_from` correctly. Otherwise I'll write a small view function directly from the architecture formula and verify it against `Q99`.

```python
import traceback
for fn__ in [mlpf, LNf]:
    co=fn__.__code__; print(fn__.__name__, 'args', co.co_varnames[:co.co_argcount], '| globals', co.co_names, '| consts', [c for c in co.co_consts if not hasattr(c,'co_code')][:20])
kp=kidx[0]
try:
    qv=view_from(X8o[kp][None,:], np.array([3600.0]))
except Exception:
    print(traceback.format_exc()[-1500:])
print([n_ for n_ in NAMES if 'gap' in n_ or 'z_prior' in n_ or 'z_head' in n_ or 'query_norm' in n_ or 'type_emb' in n_])
```