Run 232 shows that `run_cf` converts tensors to float64 and calls `zpost` and `step` through their global names, so the patch should take effect. Something else is wrong: either the key lookup fails on the float64 u, or `amp_eval` uses cached or precomputed results. Next I will call `run_cf` directly on one stay with injection 0 and −30, compare the output states and count lookup hits. If the states differ, the lever works and `amp_eval` was the problem. If not, the lookup is the culprit.

```python
CNT.update(z=0,hit=0,b=0)
rr={}
for iz in [0.0,-30.0]:
    INJ_Z=iz; CACHE.clear(); INS_KEY.clear(); rr[iz]=run_cf(s0,{},300)
    print(iz, dict(CNT), type(rr[iz]).__name__, len(rr[iz]))
O0,O1=rr[0.0][0],rr[-30.0][0]; print('O diff %.4f'%float((O0-O1).abs().max()), tuple(O0.shape))
INJ_Z=0.0
```