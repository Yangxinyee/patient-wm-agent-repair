The carrier and replay helpers are available. I’ll now check the readout order, transport gains and attention calculation so that any new fit preserves factual updates while imposing dose responses with the signs and magnitudes estimated from the records.

```python
print("=== PRELUDE CALCULATION ===")
print(logic224(prelude155))
print("=== PHYSIOLOGICAL READOUT ===")
print(logic224(phys171))
print("=== CURRENT CALIBRATION ===")
for nm in ['J186','Tmain187','Jmain187','rates187','directions186','directions187','cols199','rows199']:
    x225=globals().get(nm)
    if x225 is not None:
        a225=np.asarray(x225)
        if a225.size <=24: print(nm, a225.tolist())
        else: print(nm, a225.shape, 'column norm',np.linalg.norm(a225,axis=0).round(5).tolist(),
                    'mean',np.mean(a225,axis=0).round(5).tolist())
print("J186 @ directions187",np.round(J186@directions187,6))
print("=== OVERRIDES ===")
for nm in ['zero169','refit220']:
    o225=globals()[nm]
    print(nm)
    for k225,v225 in o225.items():
        if isinstance(v225,dict):
            print(k225,{kn: tuple(vn.shape) if hasattr(vn,'shape') else type(vn).__name__
                        for kn,vn in v225.items()})
        else: print(k225,tuple(v225.shape) if hasattr(v225,'shape') else type(v225).__name__)
print("=== BASELINE AND COMPLETE LOCAL RESPONSES ===")
print({str(k):np.asarray(v).mean(axis=0).round(5).tolist()
       for k,v in replay219.items()})
```
