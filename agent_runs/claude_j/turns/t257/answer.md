Run 229 shows exactly zero effect, even with a −30 mg/dL injection. The patch never fired, so this result says nothing about the model. Either the u that reaches `zpost` does not match the key recorded in `bvd`, or `run_cf` does not call the patched functions.

Next I will count how often the patched `zpost` and `bvd` are called on one stay and how often the insulin lookup succeeds. That will show which link failed.

```python
CNT={'z':0,'hit':0,'b':0}
def zpost_inj(o,h,dt,e,u,unit=1.0):
    CNT['z']+=1; z=zpost_orig(o,h,dt,e,u,unit)
    if round(float(u.sum()),5) in INS_KEY:
        CNT['hit']+=1
        if INJ_Z: z=z+INJ_Z*gz_t
    return z
_b0=bvd_inj
def bvd_cnt(evs):
    CNT['b']+=1; return _b0(evs)
bvd=bvd_cnt; zpost=zpost_inj; INS_KEY.clear(); CACHE.clear()
_=amp_eval(1.0,[s0]); print(CNT, len(INS_KEY), list(INS_KEY.items())[:2], len(insmin[s0]))
```