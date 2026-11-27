The records support a negative insulin effect and a positive norepinephrine effect. The current repair passes the factual-loss limit, but its rollout effects remain inaccurate, especially for halved insulin. I’ll compare the cached factual and changed-dose states directly; matching the caches will let me fit the combined update rather than adjust its pieces separately.

```python
import numpy as np, torch, inspect, re
def np265(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
print("=== CACHE INTERFACES ===")
for nm in ["features220", "parts259", "oparts259", "fresh_tokens262", "value254"]:
    v = globals().get(nm)
    if isinstance(v, dict):
        print(nm, [(str(k), tuple(x.shape) if hasattr(x, "shape") else type(x).__name__) for k, x in list(v.items())[:10]])
    elif isinstance(v, tuple):
        print(nm, [tuple(x.shape) if hasattr(x, "shape") else type(x).__name__ for x in v])
for nm in ["J255", "target255", "directions255", "directions_primary255", "dd193", "primary255", "T4_228"]:
    v = globals().get(nm)
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = np265(v)
        print(nm, a.tolist() if a.size <= 20 else
              {"shape": a.shape, "norms": np.linalg.norm(a, axis=0).tolist()})
print("\n=== TOKEN AND FEATURE ALIGNMENT ===")
print("F81 shape", tuple(F81.shape), "tokens259 shape", tuple(tokens259.shape))
print("first-six token RMS", float(torch.mean((torch.as_tensor(F81)[:,:6]-tokens259[:,:6])**2).sqrt()))
pbase265 = parts254(F81, original=True)
pcur265 = parts254(tokens259)
for j in range(4):
    a, b = np265(pbase265[j]), np265(pcur265[j])
    print("part", j, "shape", a.shape, "original/current RMS", float(np.sqrt(np.mean(a*a))),
          float(np.sqrt(np.mean(b*b))), "difference", float(np.sqrt(np.mean((a-b)**2))))
print("H220 hidden RMS difference",
      float(np.sqrt(np.mean((np265(pcur265[1]).reshape(-1,768)-np265(H220))**2))))
print("\n=== DOSE / HOLDOUT ARRAYS ===")
for nm, v in sorted(dict(globals()).items()):
    if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim and (
        nm.startswith(("rate", "drug", "stay", "split", "train", "mask", "Y220", "X220", "fit220", "val220"))
        or (v.shape[0] in [4213,16852] and v.ndim == 1)):
        a = np265(v)
        print(nm, a.shape, "range", (float(np.nanmin(a)),float(np.nanmax(a))) if np.issubdtype(a.dtype,np.number) else "")
print("\n=== REPLAY HELPERS ===")
for nm, fn in sorted(dict(globals()).items()):
    if callable(fn) and re.search(r"(replay|advance|step|future|prelude|transition)", nm, re.I):
        try: print(nm, inspect.signature(fn))
        except Exception: pass
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 265; forwards 20. Checking factual/cache alignment and physical-dose interfaces before joint regression. Record-derived signs remain insulin negative, NE positive; checkpoint222 remains installed and factually validated.\n")
```