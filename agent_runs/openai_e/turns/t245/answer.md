The aggregation stopped because `shift` is also a pandas method; this was a reporting error, not a model result. No weights changed. I’ll rerun with explicit column indexing so the comparison uses the evaluation’s stay-weighted averaging. The resulting table will determine whether any prior candidate is genuinely better under the factual-fit constraint.

```python
import inspect, numpy as np, pandas as pd
from pathlib import Path

print("=== ACTUAL GPU FRONTIER ===")
for n in range(1, 24):
    p = Path(f"/work/fwd_{n}")
    facts, doses = {}, {}
    for model in ("original", "edited"):
        fp, dp = p/f"factual_{model}.csv", p/f"dose_edit_{model}.csv"
        if fp.exists():
            facts[model] = float(pd.read_csv(fp)["nll"].mean())
        if dp.exists():
            df = pd.read_csv(dp)
            df = df[df["reached"].astype(str).str.lower().isin(["true", "1"])]
            doses[model] = {
                e: round(float(g.groupby("stay_id")["shift"].mean().mean()), 3)
                for e,g in df.groupby("edit")
            }
    if facts:
        print(n, "NLL", {k: round(v,6) for k,v in facts.items()},
              "delta", round(facts["edited"]-facts["original"],6)
              if len(facts)==2 else "")
    if doses:
        print(n, "dose", doses)

print("\n=== HELPER CONTRACTS ===")
for name in ("prelude46", "core_update_local", "fit_detector197",
             "decode_states175", "static_for_states162", "fit_ridge182"):
    fn = globals().get(name)
    if not callable(fn):
        print(name, "not found")
        continue
    print(name, inspect.signature(fn))
    try:
        print(inspect.getsource(fn)[:1500 if name in ("fit_detector197","fit_ridge182") else 650])
    except Exception:
        print("dependencies:", fn.__code__.co_names)

print("\n=== PRELUDE DATA ===")
print("F197", F197.shape, "B198", B198.shape)
for key,data in C197.items():
    print(key, {k: getattr(v,"shape",type(v).__name__) for k,v in data.items()})

print("\n=== CURRENT BASELINE IDENTITY ===")
def arr213(x):
    return x.detach().cpu().numpy() if hasattr(x,"detach") else np.asarray(x)
with np.load("/work/baseline77.npz") as z:
    print("max mismatch", np.max(np.abs(arr213(load("core.b_in.weight")) -
          arr213(load_original("core.b_in.weight")) - z["delta"])))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs 212–213: measured GPU frontier and helper recovery\n"
            "Run212 stopped on pandas shift-method collision; fixed with explicit column indexing. "
            "Aggregating factual costs and stay-weighted dose readouts; recovering decoder/detector "
            "contracts. No weights changed. Python runs: 213; forwards: 23.\n")
```
