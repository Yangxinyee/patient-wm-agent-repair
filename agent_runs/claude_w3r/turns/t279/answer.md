The gate read is clean: residual 0.007, at most 0.14 on non-u tokens. The rate reads are not. Through norm1's LayerNorm the rate information saturates:
- Insulin reads roughly linearly over 0.5–5 but flattens above 5.
- Norepinephrine breaks above its data maximum of 1.0: rate 1.5 reads as 0.58, rate 2.0 as 0.79. So the norepinephrine ×2 edit would read wrongly.

A single linear read therefore cannot serve norepinephrine ×2. I will decode with gated GELU units, using dead units of the prelude FFN, fitted by least squares on tokens that include the four dose edits (UX). That needs natural slot-token statistics at the prelude FFN input. This run probes what `prelude_att` and `run_stay` return, so I can collect those statistics.

```python
s0=SST[0] if 'SST' in globals() else None
try:
    Bq,rq,Eq=stay_data(int(s0)); print('stay',s0,'nb',len(Bq))
    o=prelude_att([Bq],[rq],[Eq],3,ret_tokens=True); print('prelude_att', type(o), [ (type(t).__name__, getattr(t,'shape',None)) for t in (o if isinstance(o,(tuple,list)) else [o])])
except Exception as ex: print('prelude_att err',repr(ex)[:300])
try:
    o2=prelude_att(Bq,rq,Eq,3,ret_tokens=True); print('prelude_att2', type(o2), [ (type(t).__name__, getattr(t,'shape',None)) for t in (o2 if isinstance(o2,(tuple,list)) else [o2])])
except Exception as ex: print('prelude_att2 err',repr(ex)[:300])
try:
    o3=run_stay(Bq,rq,Eq,'std',ret=True); print('run_stay', type(o3), [ (k if isinstance(o3,dict) else i, getattr(v,'shape',type(v).__name__)) for i,(k,v) in enumerate(o3.items() if isinstance(o3,dict) else enumerate(o3))][:12])
except Exception as ex: print('run_stay err',repr(ex)[:300])
```