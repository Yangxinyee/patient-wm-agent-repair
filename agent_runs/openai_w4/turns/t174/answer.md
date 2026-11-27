The last run showed an insulin impulse peaking around lag 2 and then decaying, but a display error stopped execution before the replay baseline. No weights changed. I will fix that error and complete the baseline and alignment checks. The subsequent impulse test will confirm or refute whether the edited model loses this sustained response.

```python
import time, inspect, numpy as np, torch

def brief154(v, depth=0):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return np.round(v, 5).tolist() if v.size <= 16 else {"shape": v.shape}
    if isinstance(v, dict):
        if depth >= 4:
            return {"keys": list(v)[:10]}
        return {str(k): brief154(x, depth+1) for k, x in list(v.items())[:10]}
    if isinstance(v, (list, tuple)):
        return [brief154(x, depth+1) for x in v[:10]]
    if isinstance(v, np.generic):
        return v.item()
    return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)[:100]

print("=== DIRECT KERNELS AND MATCHED READOUT TARGETS ===")
print("kernels:", brief154(kernels137))
print("readouts:", brief154(readout_estimates137))
print("quadratic:", brief154(quadratic137))

print("\n=== ACTION PROBE STRUCTURES ===")
for drug, obj in channel_probe147.items():
    print(drug, "keys:", list(obj))
    for k in ["indices", "rate", "records"]:
        if k in obj:
            print(k, getattr(obj[k], "shape", type(obj[k]).__name__))
    for mult in [1.0, 2.0]:
        res = obj["results"][mult]
        print("multiplier", mult,
              {k: getattr(v, "shape", type(v).__name__) for k, v in res.items()}
              if isinstance(res, dict) else brief154(res))

print("\n=== TRANSITION ALIGNMENT ===")
print("next:", type(next).__name__, getattr(next, "shape", None),
      np.asarray(next)[:12])
print("E65/U65:", E65.shape, U65.shape)
print("candidate:", candidate150[:6])
print("candidate hours:", repair_bank["hour"][candidate150[:6]])
for name in ["head_features_fixed", "probe_means95", "pre_context_repair"]:
    fn = globals()[name]
    print(name, inspect.signature(fn), "dependencies:", fn.__code__.co_names)
    print("constants:", brief154(fn.__code__.co_consts))

print("\n=== READING-EVALUATION FUNCTIONS ===")
for name, fn in list(globals().items()):
    if inspect.isfunction(fn) and any(s in name.lower() for s in ["nll", "reading", "means"]):
        print(name, inspect.signature(fn))

weights_current154 = {name: torch.as_tensor(load(name), dtype=torch.float32)
                      for name in NAMES}
t154 = time.time()
baseline154 = replay_factual125(weights_current154)
print("\n=== CURRENT-EDIT FACTUAL REPLAY ===")
print("seconds:", round(time.time()-t154, 2))
print(brief154(baseline154))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 154\nRun 153 stopped on a heterogeneous-array display error before "
            "computing its proposed baseline. Retried with recursive compact display and "
            "computed current-edit factual replay. No saved weights changed.\n")
```