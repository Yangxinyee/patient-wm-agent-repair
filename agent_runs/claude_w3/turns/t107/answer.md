I have the architecture again. The step that matters most is the per-bundle update. The z posterior depends on `out`, the time-decayed slots of h, the gap term and the bundle vectors e_k and u_k, so the ~0.08 errors could come from the encoders or from the recurrence.

Next I will check whether Python can still show the source of my validation helpers, and list the keys of `variants`, keeping the output short.

```python
import inspect
for fn in ['zpost_check2','set_enc','bundle_vecs']:
    try:
        src = inspect.getsource(globals()[fn])
        print('==', fn, len(src.splitlines()), 'lines'); print(src[:1500])
    except Exception as e:
        print('==', fn, 'no source:', type(e).__name__)
print('variants keys:', list(variants.keys()))
```