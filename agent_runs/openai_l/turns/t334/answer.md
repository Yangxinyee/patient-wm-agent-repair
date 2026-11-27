The second factual audit increased NLL by 0.00399, giving about +0.00440 across the two batches. The candidate therefore remains plausible, though norepinephrine effects still undershoot. I’ll compare the saved regression solutions and recover the causal-channel construction before deciding whether a stronger edit is justified.

```python
import re
import numpy as np
import pandas as pd

backup289 = {name: load(name).copy() for name in edited()}
current_T289 = np.vstack([
    load("attr_heads.value_bin.2.weight").T,
    load("attr_heads.value_bin.2.bias")[None, :]
])
print("CURRENT_HEAD_RECONSTRUCTION_ERROR:",
      np.max(np.abs(current_T289 - grid286[(0.35, 0.3, 8.0)]["T"])))

print("\n=== ALL MULTI-DOSE CANDIDATES ===")
print("key | offline factual deltas | four immediate slopes | proxy")
for key, candidate in sorted(grid286.items()):
    s = candidate["score"]
    print(key,
          "loss:", [round(float(v), 6) for v in s[:3]],
          "slopes:", [round(float(v), 3) for v in s[3]],
          "proxy:", round(float(candidate["proxy"]), 6))

print("\n=== REGRESSION FUNCTION DEPENDENCIES ===")
for name in ["weighted_fit281", "score263", "score269", "nll228"]:
    obj = globals().get(name)
    if callable(obj) and hasattr(obj, "__code__"):
        c = obj.__code__
        print(name, "args:", c.co_varnames[:c.co_argcount],
              "dependencies:", c.co_names)
print("RELEVANT_LIVE_NAMES:",
      [n for n in list(globals())
       if any(s in n.lower() for s in ["wire", "quiet", "pair4", "arch263", "fisher"])])

notes289 = open("/work/notes.md").read()
parts289 = notes289.split("\n\n")
print("\n=== EARLY CORRECTED IDENTIFICATION ===")
dml289 = [p for p in parts289 if "dml" in p.lower()]
for p in dml289[:5]:
    print(p[:1000], "\n")
print("\n=== QUIET-CHANNEL CONSTRUCTION ===")
wire289 = [p for p in parts289
           if any(s in p.lower() for s in ["quiet wire", "quiet-wire", "causal channel", "action wire"])]
for p in wire289[:5]:
    print(p[:1000], "\n")

print("\n=== PAIRED FACTUAL DELTAS ===")
for directory in ["/work/fwd_30", "/work/fwd_33"]:
    e = pd.read_csv(directory + "/factual_edited.csv")
    o = pd.read_csv(directory + "/factual_original.csv")
    assert e[["stay_id", "hour", "event"]].equals(o[["stay_id", "hour", "event"]])
    d = e["nll"].to_numpy() - o["nll"].to_numpy()
    by_stay = pd.DataFrame({"stay": e.stay_id, "delta": d}).groupby("stay").delta.mean()
    print(directory, "readings:", len(d), "delta:", float(d.mean()),
          "stay_mean_SE:", float(by_stay.std(ddof=1) / np.sqrt(len(by_stay))))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 289 / current candidate audit and alternatives\n"
            "grid286(.35,.3,8), hash df0778bbce: Forward29 factual "
            "+.00481 on tuning100–199; Forward30 rollout "
            "[-10.201,+4.921,+2.423,-1.151]. Forward31 factual "
            "+.00399 on tuning300–399; pooled factual delta about +.00440. "
            "No weights changed; backup289 preserves all current edits. "
            "Comparing saved closed-form alternatives and recovering quiet-channel "
            "construction before any further strengthening.\n")
```
