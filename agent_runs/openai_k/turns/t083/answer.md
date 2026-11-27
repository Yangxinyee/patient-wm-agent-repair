The retained posterior edits barely strengthen either drug response, despite preserving local factual fit. I’ll verify the boundary-safe reconstruction and recover the stronger prior-head route. That route is worth pursuing only if its signal can survive the state update while keeping factual predictions nearly unchanged.

```python
import torch, numpy as np

print("EARLY IDENTIFICATION NOTES")
print(Path("/work/notes.md").read_text()[:4200])

print("\nSCRATCH WEIGHT DIFFERENCES")
scratch77 = []
for name in NAMES:
    original = torch.as_tensor(load_original(name), dtype=torch.float32)
    current = torch.as_tensor(W0[name], dtype=torch.float32)
    delta = current - original
    if float(delta.abs().max()) > 1e-7:
        scratch77.append((name, float(delta.norm()), float(delta.abs().max())))
print(scratch77)
print("mlp0:", getattr(mlp0, "__name__", type(mlp0).__name__))

print("\nCANDIDATE/HELPER INVENTORY")
lines77 = []
for name in sorted(list(globals())):
    if any(s in name for s in ("65", "66", "73", "74")):
        obj = globals()[name]
        if isinstance(obj, dict):
            desc = "dict keys=" + repr(list(obj.keys())[:10])
        elif isinstance(obj, (list, tuple)):
            desc = type(obj).__name__ + " length=" + str(len(obj))
            if len(obj):
                first = obj[0]
                desc += " first=" + (
                    str(tuple(first.shape)) if hasattr(first, "shape")
                    else repr(first)[:150]
                )
        elif callable(obj):
            desc = str(inspect.signature(obj))
        elif hasattr(obj, "shape") and ("carrier" in name or "dose" in name):
            desc = str(tuple(obj.shape)) + " " + repr(arr71(obj))[:300]
        else:
            continue
        lines77.append(name + ": " + desc)
print("\n".join(lines77)[:3000])

print("\nBOUNDARY-SAFE CORE CHECK")
probe77 = np.flatnonzero(valid76)[::max(1, int(valid76.sum()) // 16)][:16]
out77, h77 = r59_core(probe77)
print("out max error:", float(np.max(np.abs(arr71(out77) - sx[probe77, :192]))))
print("h max error:", float(np.max(np.abs(arr71(h77).reshape(-1,768) - sx[probe77,192:960]))))
print("r59_features globals:", r59_features.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 77: boundary-safe core and prior-route audit\n"
            "- Posterior-only candidates preserve local fit but do not recover identified dose slopes.\n"
            "- Checking scratch weights, exact state reconstruction, and prior-head candidate interfaces.\n"
            "- Forward passes: 4; Python runs: 77.\n")
```