EDR holds, for each edit, 20,313 head-input rows (gi indexes into Xf), the change in head input dX, and the dose change at lags 1-4 (dd). Next I will run diag on the stored decoders vI (glucose rows) and vN (MAP rows). This gives each decoder's factual noise and its response per unit of dose change under each edit. It will show whether the decoders are good enough to build the weight edit on.

```python
for nm,v_,rw,oc in [('vI',vI,ins_rows,0),('vN',vN,map_rows,2)]:
    try:
        r_=diag(v_, rw, oc, keysI+keysN); print(nm, str(r_)[:600])
    except Exception as e: print(nm,'ERR',repr(e)[:300])
```