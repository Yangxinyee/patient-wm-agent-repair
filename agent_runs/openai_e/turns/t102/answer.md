The recovered objects include factual features, both dose contrasts, and admission-null constraints. I’ll examine their units and candidate trade-offs next. The goal is to find a detector with stronger doubling sensitivity but less factual and insulin activation; if that trade-off is unavailable, this rank-one approach needs revision.

```python
import inspect
import numpy as np
import torch

print("RUN81/82 OBJECT INVENTORY")
for k87, v87 in sorted(list(globals().items())):
    if k87.endswith(("81", "82")):
        if isinstance(v87, (np.ndarray, torch.Tensor)):
            print(k87, type(v87).__name__, tuple(v87.shape))
        elif isinstance(v87, dict):
            print(k87, "dict", list(v87)[:16])
        elif isinstance(v87, (int, float, str, tuple, list)):
            print(k87, repr(v87)[:200])

print("\nCONTRAST UNITS")
for k87 in D81:
    x87 = array80(D81[k87])
    y87 = array80(Y81[k87])
    print(k87, "D", x87.shape, "Y", y87.shape,
          "Y min/max/rms", float(np.min(y87)), float(np.max(y87)),
          float(np.sqrt(np.mean(y87*y87))))

print("\nRAW FEATURE/MASK INVENTORY")
for k87, v87 in sorted(list(globals().items())):
    if any(s in k87.lower() for s in
           ["slot", "train57", "test57", "mask", "src45", "hnew", "fit_drug",
            "initial", "core_result", "core_out", "delta_in"]):
        if isinstance(v87, (np.ndarray, torch.Tensor)):
            print(k87, type(v87).__name__, tuple(v87.shape))
        elif isinstance(v87, dict):
            print(k87, "dict", [(str(k), getattr(v, "shape", None))
                               for k, v in list(v87.items())[:5]])
        elif isinstance(v87, (int, float, str, tuple, list)):
            print(k87, repr(v87)[:250])

print("\nLOCAL CODA")
for name87 in ["coda46", "core_lin", "core_ffn", "tt45"]:
    f87 = globals()[name87]
    print(name87, inspect.signature(f87), "vars", f87.__code__.co_varnames,
          "refs", f87.__code__.co_names)

print("\nSELECTED CANDIDATE RESPONSE")
c87 = np.load("/work/ne_candidate82.npz")["feature"]
for key87 in slot_deltas47:
    dd87 = array80(slot_deltas47[key87])
    yy87 = array80(targets57[key87])
    response87 = dd87 @ c87
    print(key87, "feature", response87.shape, "target", yy87.shape,
          "feature rms", float(np.sqrt(np.mean(response87**2))))

print("\nNOTEBOOK HEADERS WITH INSULIN IDENTIFICATION / BASELINE")
for s87 in sections85:
    if ("1.637" in s87 or "Run 68" in s87 or "Forward 9" in s87 or
        ("insulin" in s87.lower() and any(t in s87.lower() for t in
          ["placebo", "bootstrap", "persistence", "target quantile"]))):
        print(s87[:650].strip())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 87: detector feasibility audit\n"
            "Inspecting contrast targets, masks, raw features and local coda helpers before "
            "closed-form selective refitting. No new weight changes; baseline77 remains installed. "
            "Python runs: 87; forwards: 12.\n")
```