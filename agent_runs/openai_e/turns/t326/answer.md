The edit changes only the recurrent input weights; the architecture leaves the reading heads unchanged. The two stronger candidates both exceeded the factual limit on the same cohort. The audit output was truncated before showing the reduced edit’s other result, so I’ll summarize validation more compactly. Losses below 0.005 across both cohorts would support further rollout testing.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import inspect

print("FACTUAL AUDIT: forward, stays, readings, original NLL, edited NLL, delta")
for k in range(1, 38):
    po = Path(f"/work/fwd_{k}/factual_original.csv")
    pe = Path(f"/work/fwd_{k}/factual_edited.csv")
    if po.exists() and pe.exists():
        ao, ae = pd.read_csv(po), pd.read_csv(pe)
        print(k, ao.stay_id.nunique(), len(ao),
              round(ao.nll.mean(), 7), round(ae.nll.mean(), 7),
              round(ae.nll.mean() - ao.nll.mean(), 7))
        if k in [36, 37]:
            print("  cohort range", int(ao.stay_id.min()), int(ao.stay_id.max()))
            print("  event deltas",
                  (ae.groupby("event").nll.mean() -
                   ao.groupby("event").nll.mean()).round(7).to_dict())

print("\nDOSE FILE SCHEMA")
for k in [1, 35]:
    for m in ["original", "edited"]:
        p = Path(f"/work/fwd_{k}/dose_edit_{m}.csv")
        if p.exists():
            d = pd.read_csv(p)
            print(k, m, "shape", d.shape, "columns", list(d.columns))
            print(d.head(2).to_dict("records"))

notes276 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-IDENTIFICATION NOTES")
print(notes276[:5200])

print("\nRELEVANT RETAINED CONTRACTS")
keywords276 = ("insulin", "ins_", "ne_", "delta", "causal", "target", "impulse", "blend")
selected276 = [k for k in sorted(globals())
               if any(s in k.lower() for s in keywords276) and not k.startswith("_")]
for k in selected276[:65]:
    v = globals()[k]
    if callable(v):
        try:
            print(k, str(inspect.signature(v)))
        except Exception:
            print(k, "callable")
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (int, float, str, np.number)):
        print(k, str(v)[:100])
    elif isinstance(v, dict):
        print(k, "dict keys", list(v)[:12])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 276: compact audit after truncated run275\n"
            "Architecture confirms the current edit acts solely through core.b_in.weight. "
            "Joint236/268 exceeded .005 on tuning100..199. Recovering compact paired "
            "results, rollout schema and record-identification contracts. No edits changed. "
            "Python runs: 276; forwards: 37.\n")
```