The kernel still has all the verified pieces. The block weights sit one level deeper than my failed helper assumed: `core.blocks.0.0.*` and `core.blocks.0.1.*`, i.e. two block layers. The prelude and coda each have one layer.

Next, before building on these functions:
- read the function signatures;
- look at one bundle's structure (time field, atom fields).

If a bundle carries time, atoms, and drug and reading fields as I expect, I can write a teacher-forced loop that records, at every bundle, the intermediate quantities (E, h_new, y) I need for the injection design.

```python
import inspect
for k in ['cstep','enc2','tf_check','probs_from_v0','avec','senc']:
    try: print(k, inspect.signature(globals()[k]))
    except Exception as ex: print(k,'?',ex)
sid0=list(bundles_B)[0]; B=bundles_B[sid0]
b=B[3]; print(type(b), (list(b.keys()) if isinstance(b,dict) else len(b)))
print(str(b)[:600])
```