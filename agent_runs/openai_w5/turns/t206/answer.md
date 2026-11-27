The replay error is about 7% of the hidden-state scale, and the attention variants tested so far haven’t removed it. I’ll first verify that the saved states came from the same checkpoint as the replay weights. A stale checkpoint could explain why both implementations agree with each other but not with the GPU.

```python
import os, json, numpy as np, torch
g181 = dict(globals())

print("=== STATE FILE PROVENANCE ===")
print("s4 type", type(s4).__name__)
for folder181 in ["fwd_13", "fwd_15"]:
    for model181 in ["original", "edited"]:
        path181 = f"/work/{folder181}/states_{model181}.npz"
        if not os.path.exists(path181):
            continue
        with np.load(path181) as f181:
            x181 = f181["x"]
            if x181.shape == s4["x"].shape:
                same_rows181 = np.array_equal(f181["stay_id"], s4["stay_id"])
                print(folder181, model181, "same stay rows", same_rows181,
                      "x RMSE", float(np.sqrt(np.mean((x181-s4["x"])**2))),
                      "max difference", float(np.max(np.abs(x181-s4["x"]))))
            else:
                print(folder181, model181, "shape", x181.shape)
    summary_path181 = f"/work/{folder181}/summary.json"
    if os.path.exists(summary_path181):
        summary181 = json.load(open(summary_path181))
        print("summary", folder181, repr(summary181)[:700])

print("\n=== CHANNEL CONFIGURATION OBJECTS ===")
count181 = 0
for nm, obj in sorted(g181.items()):
    if not any(word in nm.lower() for word in ["pair", "private", "coord", "channel"]):
        continue
    if callable(obj):
        print(nm, "function",
              obj.__code__.co_varnames[:obj.__code__.co_argcount]
              if hasattr(obj, "__code__") else "")
    elif isinstance(obj, dict):
        print(nm, "dict keys", list(obj)[:12])
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(nm, "shape", tuple(obj.shape),
              "values", obj.tolist() if obj.size <= 12 and isinstance(obj, np.ndarray)
              else obj.tolist() if isinstance(obj, torch.Tensor) and obj.numel() <= 12
              else "")
    elif isinstance(obj, (list, tuple)):
        print(nm, repr(obj)[:220])
    elif isinstance(obj, (str, int, float)):
        print(nm, repr(obj)[:180])
    else:
        continue
    count181 += 1
    if count181 >= 35:
        break

print("\n=== CURRENT PRIVATE-ROW STRUCTURE ===")
idx181 = [61, 21, 15, 66]
for nm in ["core.c_out.weight", "core.b_in.weight",
           "core.coda.0.attn.out.weight", "core.coda.0.ffn.2.weight",
           "core.out_norm.weight", "core.out_norm.bias"]:
    a181 = load(nm)
    a181 = a181.detach().cpu().numpy() if isinstance(a181, torch.Tensor) else np.asarray(a181)
    print(nm, "private row norms/values",
          np.linalg.norm(a181[idx181], axis=1).tolist()
          if a181.ndim == 2 else a181[idx181].tolist())
for j181 in [2, 3, 4]:
    x181 = np.asarray(s4["x"][j181])
    print("bundle", j181, "hour", s4["hour"][j181],
          "out private", x181[:192][idx181].tolist(),
          "mean h private", x181[192:960].reshape(4,192).mean(0)[idx181].tolist())
```