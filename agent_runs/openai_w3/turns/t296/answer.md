The changed-dose tokens reconstruct exactly, so a token-construction error looks unlikely. I’ll inspect how the attention and feed-forward edits combine before fitting a correction to their total response. Success will require better rollout effects without exceeding the factual-loss limit.

```python
import os, re, inspect, numpy as np, torch
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== RECORD DOCUMENTATION ===")
print(open("/data/README.md").read())
print("\n=== RECENT HELPERS AND ARRAYS ===")
_g263 = dict(globals())
for k, v in sorted(_g263.items()):
    if re.search(r"(24[5-9]|25[0-9]|26[0-2])$", k):
        if callable(v):
            try: detail = str(inspect.signature(v))
            except Exception: detail = ""
            print(k, detail)
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape), str(v.dtype))
        elif isinstance(v, dict):
            print(k, "dict", len(v), "keys", list(v)[:12])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))
print("\n=== HELPER SOURCE ===")
for k in ["prelude254", "posterior153", "token255", "tokens255"]:
    f = _g263.get(k)
    if callable(f):
        try:
            print(k, inspect.getsource(f))
        except Exception:
            print(k, "referenced names:", f.__code__.co_names,
                  "constants:", [x for x in f.__code__.co_consts if isinstance(x, (str, int, float))][:30])
print("\n=== EARLY IDENTIFICATION NOTES ===")
_nt263 = open("/work/notes.md").read()
print(_nt263[:6500])
print("\n=== CHECKPOINT222 NOTES ===")
for line in _nt263.splitlines():
    if any(s in line.lower() for s in ["python run 222", "forward 15", "checkpoint222", "factual budget", "12.955"]):
        print(line[:1100])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 263; forwards 20. Full changed-dose token reconstruction is exact, ruling out the suspected token bug. Recovering architecture and joint raw-state feature interfaces for a total-response constrained fit. Checkpoint222 unchanged.\n")
```