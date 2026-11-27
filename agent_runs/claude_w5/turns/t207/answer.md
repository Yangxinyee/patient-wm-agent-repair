The inventory confirms the kernel still holds the full toolkit. The notebook also gives the per-lag g-estimation effects that I need as calibration targets:
- insulin: about -2 mg/dL per unit/h at lags 1–3, cumulative -12.7;
- norepinephrine: +3.2, +2.8, +1.6 … per mcg/kg/min, cumulative +8.6.

The function and array lists were cut off before the most recent items, which are the hooked core (cstep_h), the head pieces and the run-185 feature collection. I need those exact names before building the edit. Next I print only the tail of the function list with signatures, plus the later-defined variables. I expect to find cstep, cstep_h, enc2, probs_from_v0, W64 and the 14,866-row collection. If any are missing, I will rebuild them, using the exactly verified recipes in the notebook.

```python
fn_items = [(k, v) for k, v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')]
tail = fn_items[90:]
print("FN tail", len(tail))
print(' | '.join(f"{k}{str(inspect.signature(v))[:50]}" for k, v in tail)[:2600])
arr_items = [(k, v) for k, v in globals().items() if isinstance(v, (np.ndarray, torch.Tensor)) and not k.startswith('_')]
print("ARR tail")
print(' '.join(f"{k}{tuple(v.shape)}" for k, v in arr_items[-140:])[:2600])
for nm in ['W64','bundles_B','H','fixed','TRr','cen','SIDS','rows','feat']:
    if nm in globals():
        v = globals()[nm]; print(nm, type(v).__name__, (len(v) if hasattr(v,'__len__') else ''))
```