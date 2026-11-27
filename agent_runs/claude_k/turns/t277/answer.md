`e_read` doesn't take a DataFrame. Its code refers to a global `RB`, probably a lookup from bundle key to the bundle's atoms, plus `mk_best` to build atoms. I'll check what `RB` is and its key format, then try `e_read` on one key and time it. I'll compute e for all 5244 bundles only if one call is fast.

```python
print(type(RB).__name__, len(RB))
k0=list(RB.keys())[:2] if isinstance(RB,dict) else None; print('keys',k0)
v0=RB[k0[0]] if k0 else RB[0]; print(type(v0).__name__, (v0 if not hasattr(v0,'shape') else v0.shape) if not isinstance(v0,pd.DataFrame) else v0.columns.tolist())
import time as _t; t0=_t.time()
try: ev=np.asarray(e_read(k0[0] if k0 else 0)); print('e_read', ev.shape, round(float(np.linalg.norm(ev)),2), 'sec %.3f'%(_t.time()-t0))
except Exception as ex: print('err', repr(ex)[:200])
```