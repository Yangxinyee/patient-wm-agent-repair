The insulin baseline is restored exactly. Earlier NE candidates improved rollout responses but exceeded the factual-fit budget; the records suggest targets near **+4.2 to +4.6** for NE doubling and **−2.1 to −2.3** for halving. I’ll compare all measured candidates and inspect the failed calibration work so the next test addresses a documented failure.

```python
import inspect
import pandas as pd
from pathlib import Path
import numpy as np

print("=== MEASURED FACTUAL TRADE-OFFS ===")
for k in range(1, 24):
    p = Path(f"/work/fwd_{k}")
    fo, fe = p/"factual_original.csv", p/"factual_edited.csv"
    if fo.exists() and fe.exists():
        o, e = pd.read_csv(fo), pd.read_csv(fe)
        if len(o) == len(e):
            diff = e["nll"].to_numpy() - o["nll"].to_numpy()
            sd = pd.DataFrame({"stay": o.stay_id, "diff": diff}).groupby("stay")["diff"].mean()
            print(k, "n=", len(o), "original=", round(o.nll.mean(), 6),
                  "cost=", round(diff.mean(), 6),
                  "stay_SE=", round(sd.std()/np.sqrt(len(sd)), 6),
                  "event_cost=", dict(pd.DataFrame({"event": o.event, "diff": diff})
                                     .groupby("event")["diff"].mean().round(5)))

print("\n=== MEASURED ROLLOUT READOUTS (REACHED, STAY-AVERAGED) ===")
for k in range(1, 24):
    p = Path(f"/work/fwd_{k}")
    fe = p/"dose_edit_edited.csv"
    if fe.exists():
        d = pd.read_csv(fe)
        d = d[d.reached.astype(bool)]
        a = d.groupby(["edit", "stay_id"])["shift"].mean().groupby("edit").mean()
        print(k, a.round(4).to_dict())

print("\n=== RECENT CALIBRATION FINDINGS ===")
ss = Path("/work/notes.md").read_text().split("\n## ")
for s in ss:
    h = s.splitlines()[0]
    if any(t in h for t in ["Run 159:", "Run 171:", "Run 172:", "Run 173:",
                            "Run 174:", "Run 175:", "Run 176:", "Run 177:",
                            "Run 181:", "Run 182:"]):
        print("\n## " + s[:650])

print("\n=== AVAILABLE RECONSTRUCTION/FITTING HELPERS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name + sig)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 202: audit measured frontier and prior calibration\n"
            "Insulin baseline remains installed. Aggregating all actual GPU factual costs "
            "and stay-averaged reached-hour dose effects; recovering calibration findings "
            "and exact reconstruction helpers to avoid repeating rejected approaches. "
            "Python runs: 202; forwards: 23.\n")
```
