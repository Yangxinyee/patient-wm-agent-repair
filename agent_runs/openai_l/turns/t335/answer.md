The diagnostic stopped while copying a tensor; no weights changed and no new evidence was produced. I’ll correct the tensor handling and rerun it. The purpose remains to identify a stronger candidate without spending the remaining factual-loss margin blindly.

```python
import numpy as np
import pandas as pd

backup290 = {name: load(name).clone() for name in edited()}
current_T290 = np.vstack([
    load("attr_heads.value_bin.2.weight").detach().cpu().numpy().T,
    load("attr_heads.value_bin.2.bias").detach().cpu().numpy()[None, :]
])
print("CURRENT_HEAD_RECONSTRUCTION_ERROR:",
      np.max(np.abs(current_T290 - grid286[(0.35, 0.3, 8.0)]["T"])))

print("\n=== ALL MULTI-DOSE CANDIDATES ===")
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
       if any(s in n.lower() for s in ["wire", "quiet", "pair4", "arch263", "fisher"])][:40])

parts290 = open("/work/notes.md").read().split("\n\n")
print("\n=== EARLY CORRECTED IDENTIFICATION ===")
for p in [p for p in parts290 if "dml" in p.lower()][:3]:
    print(p[:650], "\n")
print("\n=== QUIET-CHANNEL CONSTRUCTION ===")
for p in [p for p in parts290
          if any(s in p.lower() for s in ["quiet wire", "quiet-wire", "causal channel", "action wire"])][:3]:
    print(p[:650], "\n")

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
    f.write("\n\n## Runs 289–290 / current candidate audit and alternatives\n"
            "Run289 stopped at Tensor.copy; no edits. grid286(.35,.3,8), "
            "hash df0778bbce: Forward29 factual +.00481 on tuning100–199; "
            "Forward30 rollout [-10.201,+4.921,+2.423,-1.151]. "
            "Forward31 factual +.00399 on tuning300–399; pooled factual "
            "delta about +.00440. No weights changed; backup290 preserves "
            "all current edits. Comparing closed-form alternatives and "
            "recovering quiet-channel construction before strengthening.\n")
```
