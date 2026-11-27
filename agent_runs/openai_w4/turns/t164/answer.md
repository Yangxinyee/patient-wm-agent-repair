The architecture confirms that doses pass through nonlinear encoders before affecting a recurrent state. That leaves both dose compression and faulty response dynamics plausible. I’ll locate the action-building functions and recover the numerical record-based targets, keeping the output short enough to avoid another truncation.

```python
import inspect
import numpy as np
import torch

print("=== ACTION / CHANNEL FUNCTIONS ===")
funcs145 = []
for k145, v145 in list(globals().items()):
    if callable(v145) and hasattr(v145, "__code__"):
        if any(s145 in k145.lower() for s145 in
               ["action", "atom", "channel", "posterior", "innov", "convol", "kernel"]):
            funcs145.append((k145, str(inspect.signature(v145))))
for row145 in sorted(funcs145)[:45]:
    print(*row145)

print("\n=== RELEVANT OBJECT NAMES ===")
objects145 = []
for k145, v145 in list(globals().items()):
    if k145.startswith("_") or callable(v145):
        continue
    if any(s145 in k145.lower() for s145 in
           ["channel", "decoder", "kernel", "target", "137", "136", "action"]):
        if isinstance(v145, dict):
            desc145 = "keys=" + repr(list(v145)[:10])
        elif hasattr(v145, "shape"):
            desc145 = "shape=" + str(tuple(v145.shape))
        else:
            desc145 = type(v145).__name__
        objects145.append((k145, desc145))
for row145 in sorted(objects145)[:75]:
    print(*row145)

print("\n=== LOCAL-PROJECTION ESTIMATES ===")
for drug145, res145 in lp_new136.items():
    for lag145 in [1, 2, 3, 6, 12]:
        rr145 = res145.get(lag145)
        if rr145 is None:
            continue
        ss145 = {}
        for kk145, vv145 in rr145.items():
            if np.isscalar(vv145):
                ss145[kk145] = vv145
            elif hasattr(vv145, "shape"):
                aa145 = np.asarray(vv145)
                ss145[kk145] = (
                    aa145.round(4).tolist() if aa145.size <= 13
                    else {"shape": aa145.shape}
                )
        print(drug145, lag145, ss145)

print("\n=== VOCAB STRUCTURE ===")
print("top-level:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
print("\nEDITED:", edited())

with open("/work/notes.md", "a") as f145:
    f145.write(
        "\n## Run 145\n"
        "Located retained action/channel functions and compact longitudinal-effect estimates. "
        "Architecture allows both encoder compression and recurrent-dynamics failure; "
        "neither is established yet. No weights changed.\n"
    )
```