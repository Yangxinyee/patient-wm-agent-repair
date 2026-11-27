The record estimates support an immediate glucose decrease of about 2.05 per insulin-rate unit and a MAP increase of about 3.14 per norepinephrine-rate unit, subject to the identification assumptions. I will recover the sustained targets and establish a current-edit replay baseline. Then an isolated response injection can test persistence independently of dose encoding: a fading response would support the dynamics hypothesis.

```python
import time, inspect, numpy as np, torch

print("=== DIRECT KERNELS AND MATCHED READOUT TARGETS ===")
for drug in kernels137:
    print(drug, "kernel:", np.round(np.asarray(kernels137[drug]), 5))
    obj = readout_estimates137[drug]
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (np.ndarray, torch.Tensor, list, tuple)):
                a = np.asarray(v)
                print(" ", k, np.round(a, 4) if a.size <= 16 else {"shape": a.shape})
            elif isinstance(v, dict):
                print(" ", k, {kk: vv for kk, vv in v.items()
                              if isinstance(vv, (int, float, np.number))})
            else:
                print(" ", k, v)
    else:
        print(obj)

print("\n=== ACTION PROBE STRUCTURES ===")
for drug, obj in channel_probe147.items():
    print(drug, "keys:", list(obj))
    for k in ["indices", "rate", "records"]:
        if k in obj:
            print(k, getattr(obj[k], "shape", type(obj[k]).__name__))
    for mult in [1.0, 2.0]:
        print("multiplier", mult,
              {k: getattr(v, "shape", type(v).__name__)
               for k, v in obj["results"][mult].items()})

print("\n=== TRANSITION ALIGNMENT ===")
print("next:", type(next).__name__, getattr(next, "shape", None),
      np.asarray(next)[:12])
print("E65/U65:", E65.shape, U65.shape)
print("candidate:", candidate150[:6])
print("candidate next:", np.asarray(next)[candidate150[:6]])
print("candidate hours:", repair_bank["hour"][candidate150[:6]])
for name in ["head_features_fixed", "probe_means95", "pre_context_repair"]:
    fn = globals()[name]
    print(name, inspect.signature(fn), "dependencies:", fn.__code__.co_names)
for name, fn in list(globals().items()):
    if inspect.isfunction(fn) and any(s in name.lower() for s in ["nll", "reading", "means"]):
        print(name, inspect.signature(fn))

weights_current153 = {name: torch.as_tensor(load(name), dtype=torch.float32)
                      for name in NAMES}
t153 = time.time()
baseline153 = replay_factual125(weights_current153)
print("\n=== CURRENT-EDIT FACTUAL REPLAY ===")
print("seconds:", round(time.time()-t153, 2))
if isinstance(baseline153, dict):
    print({k: getattr(v, "shape", type(v).__name__) for k, v in baseline153.items()})
elif isinstance(baseline153, (tuple, list)):
    print([(type(v).__name__, getattr(v, "shape", None)) for v in baseline153])
else:
    print(type(baseline153).__name__, getattr(baseline153, "shape", None))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 153\nRecovered direct record-based kernels/readout targets and "
            "action-probe alignment; computed factual replay baseline for the currently "
            "saved scale-0.5 edit. Immediate longitudinal moments are approximately "
            "-2.0468 insulin→glucose and +3.1429 NE→MAP per rate unit, conditional on "
            "the existing identification assumptions. No weights changed.\n")
```