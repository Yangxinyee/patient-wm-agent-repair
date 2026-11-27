I found the config the earlier rebuild used: `CFG`, with keys act, unit, rate_unit, end_attrs, kq and empty_e. I will run the z posterior check on a stay whose states I have (the first stay of sel1, from fwd_2). That shows how far the rebuilt encoders and z posterior are from the tool.

- **Errors near float precision:** the bundle and action encoders are right.
- **Large errors:** the earlier rebuild never matched, and a full core rebuild is not worth the remaining runs.

```python
import time
t0=time.time()
try:
    r_=zpost_check2(sel1[0], CFG, list(range(1,40)))
    if isinstance(r_,(list,tuple,np.ndarray)):
        a_=np.asarray(r_,dtype=float); print('errs shape',a_.shape); print(np.round(a_[:40],4))
    else: print(type(r_), r_)
except Exception as ex:
    import traceback; traceback.print_exc()
print('CFG',{k:(v if not hasattr(v,'shape') else v.shape) for k,v in CFG.items()})
print(time.time()-t0)
```