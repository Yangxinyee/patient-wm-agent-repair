The coda replay is accurate enough to use as an anchor, but the full recurrent reconstruction is not. I’ll recover the detector and weight-mapping details before testing a nonlinear repair. A useful direction should move MAP without materially moving glucose; paired factual validation must still meet the NLL limit.

```python
import os, re, numpy as np, torch, inspect

print("ARCHITECTURE")
print(open("/model/ARCHITECTURE.md").read())

print("\nSAVED CANDIDATE CONTRACTS")
for fn in ["ne_nonlinear250.npz", "joint236.npz", "baseline77.npz",
           "ne_candidate151.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as ar:
        print(fn)
        for k in ar.files:
            a = ar[k]
            if np.issubdtype(a.dtype, np.number):
                print(" ", k, a.shape, "norm", float(np.linalg.norm(a)),
                      "range", (float(a.min()), float(a.max())))
            else:
                print(" ", k, a.shape, str(a)[:180])

print("\nRELEVANT KERNEL CONTRACTS")
snapshot252 = list(globals().items())
for k, v in snapshot252:
    if re.search(r"(134|242|243|244|250)$", k):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict keys", list(v)[:20])
        elif callable(v) and hasattr(v, "__code__"):
            print(k, "args", v.__code__.co_varnames[:v.__code__.co_argcount],
                  "names", v.__code__.co_names)
for k, v in snapshot252:
    if callable(v) and hasattr(v, "__code__") and (
        any(s in k.lower() for s in ["prelude", "coda", "core_layer", "readout"])
        or k in ["expect45", "predict45", "norm0", "linear0", "mlp0"]
    ):
        print("FUNCTION", k, "args",
              v.__code__.co_varnames[:v.__code__.co_argcount],
              "constants", v.__code__.co_consts,
              "names", v.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 252: recover exact nonlinear-edit interfaces\n"
            "Joint236 retained. GPU-anchored coda replay is accurate; full recurrent "
            "absolute reconstruction is not trusted. Inspecting architecture, saved "
            "nonlinear detector, and callable contracts before a forward-difference "
            "MAP-specific direction fit. No weights changed.\n"
            "Python runs: 252; forwards: 29.\n")
```