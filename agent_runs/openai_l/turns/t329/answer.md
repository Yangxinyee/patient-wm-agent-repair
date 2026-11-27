The new head remains within the offline factual-loss budget, but its stronger local norepinephrine response has not yet been tested in rollouts. I’ll recover the exact audit stays and saved alternatives before paired GPU validation. Improvement would require larger, record-consistent treatment effects while keeping factual NLL within +0.005 nats.

```python
import os, glob, json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RECENT FACTUAL AUDITS ===")
recent288 = sorted(
    glob.glob("/work/fwd_*/factual_edited.csv"),
    key=os.path.getmtime
)[-3:]
for path in recent288:
    df288 = pd.read_csv(path)
    print(path, "rows:", len(df288), "columns:", list(df288.columns))
    print("stay range:", df288["stay_id"].min(), df288["stay_id"].max(),
          "unique stays:", df288["stay_id"].nunique())
audit_ids288 = sorted(pd.read_csv(recent288[-1])["stay_id"].unique().astype(int).tolist())
print("AUDIT_STAYS_JSON:", json.dumps(audit_ids288))

def compact288(x, depth=0):
    if isinstance(x, dict):
        if depth >= 2:
            return {"keys": [str(k) for k in list(x)[:12]]}
        return {str(k): compact288(v, depth + 1) for k, v in list(x.items())[:30]}
    if isinstance(x, (tuple, list)):
        return [compact288(v, depth + 1) for v in x[:10]]
    if hasattr(x, "shape"):
        if np.size(x) <= 8:
            return np.asarray(x).tolist()
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if callable(x):
        c = getattr(x, "__code__", None)
        return {"callable": True, "arguments": list(c.co_varnames[:c.co_argcount]) if c else []}
    return repr(x)[:180]

print("\n=== SAVED CANDIDATES ===")
for name288 in ["grid286", "weighted281", "weighted_fit281",
                "backup287", "backup285", "backup273",
                "record_row_targets25", "ar2_predictions19"]:
    if name288 in globals():
        print(name288, compact288(globals()[name288]))

print("\n=== LATER IDENTIFICATION EVIDENCE ===")
notes288 = open("/work/notes.md").read()
paragraphs288 = notes288.split("\n\n")
evidence288 = [
    p for p in paragraphs288
    if any(s in p.lower() for s in ["dml", "threshold", "discontinuity", "identification"])
]
for p in evidence288[-10:]:
    print(p[:1300], "\n")

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 288 / audit recovery\n"
            "No weights changed. Recovering exact paired-audit stays, the saved "
            "multi-dose head grid, and accepted record-identification evidence "
            "before GPU validation of grid286(.35,.3,8).\n")
```
