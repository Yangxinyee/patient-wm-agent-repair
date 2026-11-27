The x1/2 shortfall is not uniform. It depends strongly on the insulin rate.

| Recorded rate (u/h) | x2 shift | x1/2 shift | Truth (x2 / x1/2, roughly) |
|---|---|---|---|
| ≤1 | −2.1 | −0.6 (wrong sign) | −7 / +3.5 |
| 1–2 | −12.9 | +2.8 | −15 / +7.5 |
| 2–3 | −24.7 | +16.9 (too strong) | −25 / +12.5 |

So the edit's response is convex in the dose. It is nearly flat when the edited dose ends up below about 1.5 u/h and steep above.

There are two candidate causes:
- **The gate:** my insulin readout is multiplied by an "insulin in the previous bundle" gate. If that gate weakens for small doses, halving would partly switch the readout off.
- **The state:** it may barely distinguish small doses at all.

To tell them apart, I will tabulate the gate and the readout separately against the previous insulin dose on recorded data. First I need to see what per-row data I still hold: head inputs, gate and readout weights, and row tables with doses.

```python
allbig=[(k,type(v).__name__,getattr(v,'shape',None)) for k,v in globals().items()
     if not k.startswith('_') and isinstance(v,(np.ndarray,pd.DataFrame)) and k not in ('NAMES','SHAPES','VOCAB')]
print(allbig[80:])
for k in ['A_','WCall','datasets','base_out','Gi3','ROi5','gvec']:
    v=globals().get(k)
    if v is None: print(k,'missing'); continue
    if isinstance(v,pd.DataFrame): print(k,v.shape,list(v.columns))
    elif isinstance(v,dict): print(k,{kk:getattr(vv,'shape',vv) for kk,vv in v.items()})
    elif isinstance(v,(list,tuple)): print(k,len(v),[type(x).__name__ for x in v[:3]])
    else: print(k,getattr(v,'shape',type(v)))
print(inspect.signature(gated3), inspect.signature(ins_only), inspect.signature(pair), inspect.signature(pair_out), inspect.signature(head_run))
```