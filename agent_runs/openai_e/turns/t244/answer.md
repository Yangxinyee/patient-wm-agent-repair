The saved history confirms that simply strengthening or balancing the insulin edit can violate factual fit. I’ll now compare all actual GPU results and inspect the detector and decoder helpers. My next hypothesis is that factual compensation must account for accumulated state changes, not just local perturbations; it is supported only if that reduces held-out NLL without erasing dose sensitivity.

```python
import inspect, json, numpy as np, pandas as pd
from pathlib import Path

print("=== ACTUAL GPU FRONTIER ===")
for n in range(1, 24):
    p = Path(f"/work/fwd_{n}")
    facts = {}
    doses = {}
    for model in ("original", "edited"):
        fp = p / f"factual_{model}.csv"
        dp = p / f"dose_edit_{model}.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            facts[model] = float(df.nll.mean())
        if dp.exists():
            df = pd.read_csv(dp)
            df = df[df.reached.astype(str).str.lower().isin(["true", "1"])]
            doses[model] = {
                e: round(float(g.groupby("stay_id").shift.mean().mean()), 3)
                for e, g in df.groupby("edit")
            }
    if facts:
        print(n, "factual", {k: round(v, 6) for k,v in facts.items()},
              "delta", round(facts.get("edited", 0)-facts.get("original", 0), 6)
              if len(facts) == 2 else "")
    if doses:
        print(n, "stay-weighted dose", doses)

print("\n=== IMPORTANT HELPER CONTRACTS ===")
for name in ("prelude46", "core_update_local", "fit_detector197",
             "decode_states175", "static_for_states162", "fit_ridge182"):
    fn = globals().get(name)
    if not callable(fn):
        print(name, "not found")
        continue
    print(name, inspect.signature(fn))
    try:
        src = inspect.getsource(fn)
        print(src[:1700] if name in ("fit_detector197", "fit_ridge182") else src[:700])
    except Exception:
        print("dependencies:", fn.__code__.co_names)

print("\n=== PRELUDE DATA CONTRACT ===")
print("F197", F197.shape, "B198", B198.shape)
for key, data in C197.items():
    print(key, {k: (getattr(v, "shape", None) or type(v).__name__) for k,v in data.items()})
print("\n=== CURRENT BASELINE IDENTITY ===")
cur212 = load("core.b_in.weight")
cur212 = cur212.detach().cpu().numpy() if hasattr(cur212, "detach") else np.asarray(cur212)
orig212 = load_original("core.b_in.weight")
orig212 = orig212.detach().cpu().numpy() if hasattr(orig212, "detach") else np.asarray(orig212)
with np.load("/work/baseline77.npz") as z:
    print("max baseline mismatch:", np.max(np.abs(cur212-orig212-z["delta"])))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 212: measured-frontier aggregation and helper contracts\n"
            "Aggregating actual stay-weighted GPU interventions and factual costs; inspecting "
            "detector/decoder interfaces for accumulation-aware guards or stronger factual "
            "distillation. No weights changed. Python runs: 212; forwards: 23.\n")
```
