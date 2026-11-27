The whole-stay test suggests the nonlinear detector can distinguish norepinephrine dose changes with limited factual disturbance and little insulin leakage. That is promising, but it has not yet improved the model’s rollouts. I’ll inspect the retained coefficients and weight shapes before constructing a conservative edit. Improved norepinephrine readouts with factual NLL within the limit would support this approach; worsening either would refute it.

```python
import os, numpy as np
print("ARCHITECTURE\n", open("/model/ARCHITECTURE.md").read())
print("\nRELEVANT WEIGHT SHAPES")
for n in NAMES:
    if any(s in n.lower() for s in ("prelude", "b_in", "ffn")):
        print(n, SHAPES[n])

print("\nRETAINED OBJECTS")
items = list(globals().items())
matches = []
for k, v in items:
    if k.startswith("_") or not any(s in k.lower() for s in
        ("244", "243", "242", "nonlin", "prelude", "detector", "ne_dir", "ne_push", "joint236")):
        continue
    if callable(v):
        desc = f"function args={v.__code__.co_varnames[:v.__code__.co_argcount]}" if hasattr(v, "__code__") else "callable"
    elif isinstance(v, dict):
        desc = f"dict keys={list(v.keys())[:24]}"
    elif isinstance(v, (list, tuple)):
        desc = f"{type(v).__name__} len={len(v)}"
    else:
        desc = f"{type(v).__name__} shape={getattr(v, 'shape', None)}"
    matches.append((k, desc))
for k, desc in matches[:100]:
    print(k, desc)

print("\nSAVED CONTRACTS")
for fn in ("joint236.npz", "ne_candidate151.npz", "ne_detector81.npz"):
    with np.load("/work/" + fn, allow_pickle=True) as z:
        print(fn)
        for k in z.files:
            a = z[k]
            print(" ", k, a.shape, a.dtype,
                  repr(a.tolist())[:350] if a.size <= 16 else "")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 245: recover nonlinear edit contracts\n"
            "Run244 whole-stay validation shows positive NE doubling/halving sensitivity "
            "with small insulin leakage; nonlinear weight edit is not yet tested. "
            "Inspecting retained coefficients and architecture before a conservative GPU trial. "
            "Joint236 remains installed. Python runs: 245; forwards: 29.\n")
```
