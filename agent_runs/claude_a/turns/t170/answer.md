`ages` was overwritten with a (3,2) tensor. It should be the per-reading age of the last same-code value (5802 entries). The kernel still has `age_h` (5802, hours) and `age_mult=3600`, so `ages` was most likely `age_h·3600` seconds.

I'll rebuild it that way and rerun `tf_nll(PO)`. A match with the tool's 1.40394 (within z-sampling noise) confirms the reconstruction and the offline factual evaluator. Then I'll measure the scaffold `PS` cost.

```python
ages = torch.tensor(age_h*3600.0, dtype=torch.float32)
t0=time.time(); r0=tf_nll(PO, ret=True); t1=time.time()
print(type(r0).__name__, (len(r0) if isinstance(r0,(tuple,list)) else ''), '%.1fs'%(t1-t0))
nll0 = r0[0] if isinstance(r0,(tuple,list)) else r0
print('PO', float(nll0) if not torch.is_tensor(nll0) or nll0.numel()==1 else float(nll0.mean()))
if isinstance(r0,(tuple,list)):
    for i,x in enumerate(r0): print(i, type(x).__name__, getattr(x,'shape',None))
```