The architecture confirms that the latent token can carry a dose signal into the recurrent state. The current edit still carries no such signal, so it cannot yet repair the rollouts. I’ll recover only the essential replay helpers and carrier definitions; they must reproduce the saved states closely before I trust their calibration.

```python
import inspect, numpy as np, torch, re
print("=== RELEVANT HELPERS ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if inspect.isfunction(obj) and (
        re.search(r"(15[3-9]|16[0-4])$", key)
        or re.search(r"(readout|core_step|memory|prelude|predict_mean)", key)
    ):
        print(key, str(inspect.signature(obj)),
              "uses:", ",".join(obj.__code__.co_names))

print("\n=== ESSENTIAL ARRAYS ===")
for key in ["post154", "zin154", "zb154", "carrier154", "carrier_scale154",
            "reserved154", "rows154", "payload154", "policy154",
            "value_extract154", "W147", "overrides147",
            "tokens155", "indices155", "positions155", "best162",
            "readrows121", "codes97", "attr_static105", "transition97",
            "centres105", "r40_info", "idx81"]:
    if key not in globals():
        continue
    obj = globals()[key]
    if isinstance(obj, dict):
        print(key, {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                    for k, v in obj.items()})
    elif hasattr(obj, "shape"):
        arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
        print(key, arr.shape, "RMS", float(np.sqrt(np.mean(arr.astype(float)**2)))
              if np.issubdtype(arr.dtype, np.number) else "non-numeric")
        if arr.size <= 16:
            print(arr)
    elif isinstance(obj, (list, tuple)):
        print(key, "length", len(obj), "item types",
              [type(v).__name__ for v in obj[:4]])
    else:
        print(key, str(obj)[:150])

print("\n=== RECORD-IDENTIFICATION NOTES (bounded) ===")
text166 = open("/work/notes.md").read()
selected166 = [p for p in text166.split("\n\n")
               if re.search(r"(identif|instrument|confound|kernel|randomiz)", p, re.I)]
print("\n\n".join(selected166)[:3800])
print("\n=== TRANSPORT CONSTRUCTION NOTES ===")
selected166 = [p for p in text166.split("\n\n")
               if re.search(r"Python run[s]? (14[5-9]|15[0-6])", p)]
print("\n\n".join(selected166)[-2200:])
```
