The factual cost is not confined to insulin: MAP predictions also worsen. More importantly, the edit changes the first glucose prediction substantially, before any infusion could affect it. I’ll check whether early, untreated bundles account for much of the cost and whether they were missing from the dose-decoder fit. If so, explicitly constraining those bundles to zero could free up factual-fit budget.

```python
import numpy as np, pandas as pd
from pathlib import Path

base144 = dfs143["original"]
for label144 in ["fallback", "stronger"]:
    cur144 = dfs143[label144]
    assert base144[["stay_id", "hour", "event"]].equals(cur144[["stay_id", "hour", "event"]])
    q144 = base144[["stay_id", "hour", "event"]].copy()
    q144["dnll"] = cur144.nll.to_numpy() - base144.nll.to_numpy()
    q144["dmean"] = cur144.predicted_mean.to_numpy() - base144.predicted_mean.to_numpy()
    q144["period"] = pd.cut(q144.hour, [-np.inf, -0.001, 0.5, 4.5, 11.5, 24.5, np.inf])
    print("=== PAIRED FACTUAL SHIFTS:", label144, "===")
    print(q144.groupby("period", observed=True).agg(
        n=("dnll", "size"), mean_dnll=("dnll", "mean"),
        total_dnll=("dnll", "sum"), mean_shift=("dmean", "mean"),
        rms_shift=("dmean", lambda x: np.sqrt(np.mean(x*x)))).to_string())
    print(q144.groupby(["event", "period"], observed=True).agg(
        n=("dnll", "size"), mean_dnll=("dnll", "mean"),
        mean_shift=("dmean", "mean")).to_string())

print("=== FEATURE / STATE OBJECTS ===")
for k144 in ["rec119", "E119", "S_before119", "S_object120", "e119", "u119",
             "h119", "z119", "fit119", "raw_fits119", "target112",
             "residual_targets102", "tuning", "tune_panel"]:
    if k144 in globals():
        v144 = globals()[k144]
        print(k144, type(v144).__name__, getattr(v144, "shape", None))
        if isinstance(v144, pd.DataFrame):
            print(v144.columns.tolist())
            print(v144.head(2).to_string(index=False))
        elif isinstance(v144, dict):
            print({k: getattr(v, "shape", type(v).__name__) for k, v in v144.items()})
print("=== INITIAL-STATE TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES
       if any(s in n.lower() for s in ["initial", "init", "h0", "start", "empty"])])
print("=== ENCODING / BUNDLE HELPERS ===")
print([k for k in globals() if callable(globals()[k])
       and any(s in k.lower() for s in ["bundle", "atom", "encode", "prelude", "posterior", "context"])])
with Path("/work/notes.md").open("a") as f144:
    f144.write("\n## Runs 142–144: factual-cost attribution\n"
               "Canonical E decoder Wdose134 and steering V128 recovered; fallback [-5,+8] remains installed. "
               "Stronger [-10,+12] costs glucose_finger +0.00815 and map_arterial +0.00815 per sensor. "
               "Fallback changes the first glucose reading at hour -1 from 143.99 to 154.79, and stronger to 168.85, "
               "despite no previous drug exposure. Investigating omitted initial-bundle zero-dose constraints "
               "before accepting a simple stronger gain compromise. Record-based target uncertainty retained.\n")
```
