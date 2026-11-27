The architecture confirms that edited doses must affect recurrent state; a head-only change cannot reliably carry their effects forward. The records support provisional effects in both drug channels, while the original rollouts are much weaker. I’ll inspect the reusable code and weight layout next to locate a targeted recurrent edit.

```python
import inspect, numpy as np, torch

print("=== REUSABLE FUNCTIONS ===")
fnlist = []
for k, v in list(globals().items()):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            fnlist.append((k, str(inspect.signature(v))))
        except Exception:
            pass
print("\n".join(f"{k}{sig}" for k, sig in fnlist))

print("\n=== LARGE ANALYSIS OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 1 and v.shape[0] >= 100:
        print(k, tuple(v.shape), str(v.dtype))
    elif isinstance(v, dict) and any(s in k.lower() for s in
                                    ["rec", "cfg", "fit", "resp", "probe", "action", "corr", "state", "cand"]):
        print(k, list(v.keys())[:12])

print("\n=== KEY TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ["embed.numeric", "encode_action", "core.prelude", "core.b_in",
                            "core.delta_in", "core.a_in", "core.flow", "attr_heads.value",
                            "last_numeric", "core.c_out", "core.out_norm"]):
        print(n, SHAPES[n])

print("\n=== NOTEBOOK RUNS 8–15 ===")
start = notes.find("## Sensor checks")
end = notes.find("## Reconstruction conventions")
print(notes[start:end][:5000])
```
