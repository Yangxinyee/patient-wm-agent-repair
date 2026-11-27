The output was cut off again, and `F81` does not match the current input tokens, so I won’t use it as the factual reference. I’ll verify the already-cached original and current prelude outputs instead. Matching those caches would give us a safe basis for the joint fit.

```python
import numpy as np, torch, inspect, re
print("=== PRELUDE OUTPUT SCALES ===")
for nm in ["parts259", "oparts259", "pcur265", "pbase265"]:
    v = globals().get(nm)
    if isinstance(v, tuple):
        print(nm, [(tuple(x.shape), round(float(np.sqrt(np.mean(np265(x)**2))),4)) for x in v])
print("parts259 versus fresh current:",
      [float(np.sqrt(np.mean((np265(a)-np265(b))**2))) for a,b in zip(parts259,pcur265)])
print("\n=== POSSIBLE INPUT TOKENS ===")
for nm,v in sorted(dict(globals()).items()):
    if isinstance(v,(np.ndarray,torch.Tensor)) and tuple(v.shape) in [(4213,7,192),(4213,4,192),(4213,32)]:
        print(nm, tuple(v.shape), round(float(np.sqrt(np.mean(np265(v)**2))),3))
print("\n=== DOSE AND STAY INTERFACES ===")
for nm in ["rate88","drug88","stay88","stays88","ids88","mask220","train220","valid220","X220","Y220","H220","features193","features220"]:
    v=globals().get(nm)
    if isinstance(v,(np.ndarray,torch.Tensor)):
        a=np265(v)
        print(nm, a.shape, "first", a.reshape(-1)[:6].tolist())
    elif isinstance(v,dict):
        print(nm, "keys", list(v)[:10])
print("\n=== REPLAY FUNCTIONS ===")
for nm,fn in sorted(dict(globals()).items()):
    if callable(fn) and re.search(r"(replay|advance|step|future|prelude)",nm,re.I):
        try: print(nm,inspect.signature(fn))
        except Exception: pass
print("\n=== IMPLEMENTATION NOTES ===")
for ln in open("/work/notes.md").read().splitlines():
    if any(f"Python run {n};" in ln or f"Python runs: {n};" in ln for n in [81,88,154,155,193,254,255,259,260]):
        print(ln[:500])
with open("/work/notes.md","a") as f:
    f.write("\nPython run 266; forwards 20. F81 is not aligned with current input tokens and is excluded as a regression reference. Verifying cached original/current prelude outputs and physical doses. No weights changed.\n")
```