The assertion checks that a bundle and its predecessor belong to the same stay. The failure likely came from including stay-start rows, not from incorrect weights. I’ll inspect that indexing and the retained candidates; removing boundary rows should resolve it if this diagnosis is right.

```python
import numpy as np, inspect, dis

print("DIAGNOSTICS73")
print(repr(globals().get("diagnostics73", "missing"))[:3500])

print("\nRETAINED OBJECTS")
selected = []
for name in sorted(list(globals())):
    if name.endswith(("59", "65", "66", "73", "74")):
        obj = globals()[name]
        if hasattr(obj, "shape"):
            desc = str(tuple(obj.shape))
        elif isinstance(obj, dict):
            desc = "keys=" + repr(list(obj.keys())[:8])
        elif isinstance(obj, (list, tuple)):
            desc = "length=" + str(len(obj))
        elif callable(obj):
            desc = str(inspect.signature(obj))
        else:
            continue
        selected.append(name + ": " + desc)
print("\n".join(selected)[:3800])

print("\nPRIOR_FEATURES74")
if callable(globals().get("prior_features74")):
    print(dis.Bytecode(prior_features74).dis()[:4300])

print("\nSTATE-BOUNDARY COUNTS")
n76 = len(rr33_stays)
valid76 = np.r_[False, np.asarray(rr33_stays)[1:] == np.asarray(rr33_stays)[:-1]]
print("bundles:", n76, "noninitial:", int(valid76.sum()),
      "stay starts:", np.flatnonzero(~valid76)[:12].tolist())
print("W0 tensors:", len(W0), "submitted edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 76: stay-boundary diagnosis\n"
            "- Run 74's assertion checks adjacent stay IDs, not numerical reconstruction accuracy.\n"
            "- Inspecting candidate diagnostics and prior feature indexing; initial bundles require separate handling.\n"
            "- Forward passes: 4; Python runs: 76.\n")
```