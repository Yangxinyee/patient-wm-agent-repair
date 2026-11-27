The value head has reserved drug-response neurons and existing sensor-specific centering terms. I will inspect their exact layout and the head-input builder before fitting a replacement center. A successful recalibration should remove the trial’s predictable factual offset without removing its response to changed doses.

```python
import inspect
import pandas as pd

def _brief273(v):
    if isinstance(v, dict):
        return {str(k): _brief273(x) for k, x in list(v.items())[:10]}
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = np.asarray(v)
        return {"shape": a.shape, "sample": a.ravel()[:5].tolist()}
    if isinstance(v, pd.DataFrame):
        return {"shape": v.shape, "columns": v.columns.tolist()[:25]}
    if isinstance(v, (list, tuple)):
        return [_brief273(x) for x in v[:8]]
    return repr(v)[:130]

for k in ["sensor_neurons", "head_audit", "r245_obj", "r249_sensor_controls",
          "r249_constants", "_params257", "history_audit"]:
    print(k, _brief273(globals().get(k)))

print("\nHEAD BUILDER")
try:
    print(inspect.getsource(make_head_inputs212)[:6500])
except Exception as e:
    print(type(e).__name__, str(e)[:100])
    print("arguments:", inspect.signature(make_head_inputs212))
    print("dependencies:", make_head_inputs212.__code__.co_names)
    print("constants:", [str(c)[:250] for c in make_head_inputs212.__code__.co_consts])

print("\nOTHER DATA OBJECTS")
for k, v in list(globals().items()):
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, v.columns.tolist()[:16])
print("fresh states:", {k: v.shape for k, v in _s271.items()})
print("state files:", sorted(str(p) for p in Path("/work").glob("fwd_*/states_*.npz")))

print("\nRESPONSE-TARGET MEANS")
for drug in ["insulin", "norepinephrine"]:
    vals = [float(v) for k, v in _target257.items() if k[2] == drug]
    print(drug, "n", len(vals), "x2 target mean", np.mean(vals) if vals else None)
```
