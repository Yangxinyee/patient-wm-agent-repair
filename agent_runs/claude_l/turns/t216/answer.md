The kernel has `run_batch` and `compare` (probably the batch rollout and its comparison with the tool's CSV) and `true_shift` (probably my record-based "truth" for each readout row). Their source can't be read back, so next I'll print their signatures. I'll also run `run_batch` with the current κ = 4 weights on a few stays to see its output format. That way the next run can compute all four readouts for the whole batch, with the truth alongside.

```python
for f in [run_batch, compare, true_shift, evalF, evalH]:
    try: print(f.__name__, inspect.signature(f))
    except Exception as ex: print(f.__name__, ex)
t0=time.time()
try:
    rb=run_batch(Wk, keys[:3]) 
    print(type(rb).__name__, (list(rb.keys())[:10] if isinstance(rb,dict) else (len(rb), type(rb[0]).__name__ if len(rb) else None)))
    if isinstance(rb,dict):
        for kk,vv in list(rb.items())[:6]: print(kk, type(vv).__name__, getattr(vv,'shape',None) if not isinstance(vv,(int,float)) else vv)
except Exception as ex: print('run_batch err', repr(ex)[:300])
print('%.1fs'%(time.time()-t0))
```