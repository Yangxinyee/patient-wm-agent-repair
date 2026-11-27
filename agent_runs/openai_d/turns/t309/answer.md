The current edit’s pooled NLL increase across 300 tuning stays is about 0.00539, still above budget. The records support glucose-lowering insulin and MAP-raising norepinephrine effects, despite misleading unadjusted associations. I’ll recover the rollout targets and candidate interfaces next; a modest reduction may regain factual margin without losing most of the repair.

```python
import inspect, numpy as np, pandas as pd, re
from pathlib import Path

print("=== RECORD-DERIVED ROLLOUT TARGETS AND FALLBACK ===")
parts = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
wanted = ["## Run 7", "## Run 8", "## Run 10", "## Run 11", "## Run 12",
          "## Forward 29", "## Forward 32"]
out = "\n".join(s for s in parts if any(s.startswith(w) for w in wanted))
print(out[:5000])

print("\n=== CANDIDATE/CANCELLATION INTERFACES ===")
for nm in ["r246_candidate", "r246_loss", "r252_nll",
           "r252_fit_table", "r252_table_rows", "c210_eval", "r240_prepare"]:
    fn = globals().get(nm)
    if fn is None:
        continue
    print(nm, inspect.signature(fn))
    try:
        src = inspect.getsource(fn)
        print(src[:2200])
    except Exception:
        print("names:", fn.__code__.co_names)
        print("constants:", [x for x in fn.__code__.co_consts
                             if isinstance(x, (str, int, float, tuple))][:30])

print("\n=== CACHE STRUCTURES ===")
for nm in ["r240_pre", "r237_caches", "r252_logits", "r252_orig",
           "r252_base", "r252_fit", "r252_val", "r252_dmean",
           "r246_safe_np", "r246_orig", "r246_base", "r246_table",
           "r257_table", "r257_rows"]:
    obj = globals().get(nm)
    if isinstance(obj, dict):
        print(nm, {k: (type(v).__name__,
                       tuple(v.shape) if hasattr(v, "shape") else
                       list(v.keys())[:12] if isinstance(v, dict) else str(v)[:65])
                   for k,v in list(obj.items())[:8]})
    elif hasattr(obj, "shape"):
        print(nm, type(obj).__name__, obj.shape)
    elif obj is not None:
        print(nm, type(obj).__name__, str(obj)[:120])

n261_rows = [r for r in n260_recent if r[0] in [36, 37, 38] and r[3]=="nll"]
n261_pooled = sum(r[1]*r[4] for r in n261_rows)/sum(r[1] for r in n261_rows)
print("\nCURRENT 300-STAY POOLED DELTA:", n261_pooled)
with open("/work/notes.md", "a") as f:
    f.write(f"\n## Runs 260–261: pooled factual audit\n"
            f"Current checkpoint 8898bdc457 has paired factual deltas .006259, "
            f".003666, .006231 on tuning cohorts 0–99, 100–199, 200–299; "
            f"reading-weighted pooled delta {n261_pooled:.8f}. "
            "It is not accepted for submission. No weights changed. "
            "Recovering candidate and training-only cancellation interfaces "
            "to reduce strength with an explicit factual safety margin.\n")
```
