The architecture confirms that dose information passes through normalization and sampled-reading feedback, either of which could distort the response. The earlier record estimates support the observed directions, but do not explain the asymmetry. I’ll inspect the current calibration objects and rollout files next; no weights will change in this diagnostic.

```python
import torch
import inspect

recent151 = []
for k151, v151 in snapshot150[-130:]:
    if k151.startswith("_"):
        continue
    if isinstance(v151, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        d151 = f"{k151}: {type(v151).__name__} {tuple(v151.shape)}"
    elif isinstance(v151, dict):
        d151 = f"{k151}: dict({len(v151)}) keys={list(v151)[:8]}"
    elif isinstance(v151, (list, tuple)):
        d151 = f"{k151}: {type(v151).__name__}({len(v151)})"
    elif isinstance(v151, (str, int, float, bool, np.number)):
        d151 = f"{k151}: {str(v151)[:120]}"
    elif inspect.isfunction(v151):
        d151 = f"{k151}: function {inspect.signature(v151)}"
    else:
        d151 = f"{k151}: {type(v151).__name__}"
    recent151.append(d151)
print("=== MOST RECENT PERSISTENT OBJECTS ===")
print("\n".join(recent151)[:6800])

dose151 = pd.read_csv("/work/fwd_18/dose_edit_edited.csv")
print("=== ROLLOUT CSV ===")
print(dose151.shape, dose151.columns.tolist())
print(dose151.head(3).to_string(index=False))
print("=== CALIBRATION FUNCTION NAMES ===")
print([
    k for k, v in snapshot150
    if inspect.isfunction(v) and re.search(
        r"install|calib|tilt|head|proxy|transport|effect|predict|feature|pair", k, re.I
    )
])
```
