The paired activations reproduce the decoder probabilities accurately. That supports testing a small closed-form correction, but it must improve held-out factual fit without erasing the dose response. I’ll first recover the exact buffers and decoder contract so the fit uses correctly aligned readings.

```python
import inspect
import numpy as np
import torch
from pathlib import Path

ns226 = dict(globals())
print("Current edits:", edited())

print("\nRelevant persistent buffers:")
keys226 = [
    k for k in ns226
    if any(s in k.lower() for s in
           ["225", "224", "hidden", "train_meta162", "baseline77", "decode"])
]
for k in sorted(keys226)[:85]:
    v = ns226[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape), str(v.dtype))
    elif hasattr(v, "columns") and hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape, list(v.columns))
    elif callable(v):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, "function", sig)
        if hasattr(v, "__code__"):
            print("  dependencies:", v.__code__.co_names)
            print("  arguments/locals:", v.__code__.co_varnames[:24])
    else:
        text = repr(v)
        print(k, type(v).__name__, text[:180])

print("\nValue-head tensor names:")
for n in NAMES:
    if any(s in n.lower() for s in ["value", "decoder", "head"]):
        print(n, SHAPES[n])

print("\nBaseline archive:")
with np.load("/work/baseline77.npz", allow_pickle=False) as a:
    for k in a.files:
        print(k, a[k].shape, a[k].dtype)

print("\nCore helper contracts:")
for k in ["get", "linear", "W0"]:
    if k not in ns226:
        continue
    v = ns226[k]
    if callable(v):
        print(k, inspect.signature(v))
        print(" dependencies:", v.__code__.co_names)
        print(" constants:", repr(v.__code__.co_consts)[:1800])
    elif isinstance(v, dict):
        print(k, "dict", len(v), "entries")
    else:
        print(k, type(v).__name__, getattr(v, "shape", None))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 226: recover paired-head correction contracts\n"
        "Run225 accurately reconstructed paired decoder probabilities. "
        "Inspecting aligned activation buffers and exact value-head helpers "
        "before fitting a closed-form correction. No weights changed. "
        "Python runs: 226; forwards: 26.\n"
    )
```