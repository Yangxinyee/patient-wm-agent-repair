The code objects are enough to use the functions. `stay_data(s)` returns (B, rows, EU) for a stay, using the records `recs` and the model states. `step_at(B, rows, EU, k, du, de)` runs one local core step at bundle k from the model's true previous state, with optional perturbations of u/e, and returns (y, out, E). Next I'll check the globals these depend on (`recs`, `SX`, `SST`, `key_k`). Then I'll call `stay_data` and `step_at` on one stay with drug bundles. Success means sensible shapes and a small error against the stored model state.

```python
for k in ['recs','SX','SST','key_k','scs','rows0','B0','EU0']:
    v = globals().get(k)
    print(k, type(v).__name__, getattr(v,'shape',None) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else v))
print(recs.head(3)); print(recs.stay_id.nunique(), recs.event.unique())
s0 = int(recs.stay_id.iloc[0])
B_, rows_, EU_ = stay_data(s0)
print(type(B_), len(B_), type(rows_), len(rows_), type(EU_), len(EU_))
print('bundle0:', B_[0] if not hasattr(B_[0],'keys') else list(B_[0].keys()))
```