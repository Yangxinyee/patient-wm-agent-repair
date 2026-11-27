The architecture confirms that this can be tested entirely within the value head, without changing the recurrent dynamics. I’ll identify the modified neurons and cache interfaces, then fit cancellation from training states only. Held-out loss will test generalization; a forward rollout will test whether the drug response survives.

```python
import torch, numpy as np
from pathlib import Path
torch.set_num_threads(8)
torch.set_grad_enabled(False)

r239_keys = [
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"
]
r239_orig = {k: np.asarray(load_original(k), dtype=np.float32) for k in r239_keys}
r239_safe = {k: np.asarray(load(k), dtype=np.float32).copy() for k in r239_keys}
r239_rows = np.flatnonzero(
    np.max(np.abs(r239_safe[r239_keys[0]] - r239_orig[r239_keys[0]]), axis=1) > 1e-7
)
print("=== MODIFIED HIDDEN ROWS ===")
print(r239_rows.tolist())
for j in r239_rows:
    print(int(j), "W0 norm", float(np.linalg.norm(r239_safe[r239_keys[0]][j])),
          "b0", float(r239_safe[r239_keys[1]][j]),
          "W2", np.round(r239_safe[r239_keys[2]][:, j], 5).tolist())

print("\n=== CACHE INTERFACES ===")
for cname in ("c210_train", "c210_tune"):
    c = globals()[cname]
    print(cname, {k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else None)
                  for k, v in c.items()})
for name in sorted(list(globals())):
    if name.startswith(("c210_", "c209_", "c207_")):
        obj = globals()[name]
        if isinstance(obj, (dict, list, tuple)) and not callable(obj):
            if isinstance(obj, dict):
                print(name, "keys", list(obj)[:15])
            else:
                print(name, type(obj).__name__, "length", len(obj))
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))

r239_result = c210_eval(r239_safe, c210_train, True)
print("\n=== EVALUATOR RETURN ===")
if isinstance(r239_result, dict):
    print({k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
           for k, v in r239_result.items()})
elif isinstance(r239_result, (tuple, list)):
    print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
           for v in r239_result])
else:
    print(type(r239_result).__name__, getattr(r239_result, "shape", None))

print("\n=== RECORD-DERIVED TARGET HISTORY ===")
r239_notes = Path("/work/notes.md").read_text()
r239_sections = r239_notes.split("\n## ")
r239_selected = [s for s in r239_sections
                 if any(t in s.lower() for t in
                        ("long-horizon", "causal target", "record-derived target",
                         "cross-fitted drug", "run 234:", "run 236:"))]
for s in r239_selected[-6:]:
    print(("## " + s)[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 239: exact cancellation interfaces\n"
            "Recovered changed hidden rows, original/safe head weights, cache types "
            "and evaluator outputs. Cancellation will be fitted on training features "
            "rather than tuning labels. No weights changed.\n")
```