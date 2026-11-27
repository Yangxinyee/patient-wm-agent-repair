The discrepancy is explained: the old fit averaged five dose factors, while the new fit emphasizes doubling and halving norepinephrine. The candidate reproduces exactly and is now staged. I’ll prepare its GPU audit and fit unsaved alternatives that retain constraints for stopping, x1.5 and x3, so improvement is not limited to the four visible rollouts.

```python
import pandas as pd
import json

audit_ids286 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique()[100:200]
print("GPU AUDIT STAYS", json.dumps(audit_ids286.tolist()))

recheck286 = score234(arch263, T285, stats263["Ftune"])
print("Cached candidate score reproduced:", recheck286)
assert np.max(np.abs(np.asarray(recheck286[:3]) -
                     np.asarray(weighted281[(0.3, 0.25, 8.0)]["score"][:3]))) < 1e-8

def fit_multidose286(etaI, etaN, half_ratio, all_factor_anchor=0.05):
    GN = (all_factor_anchor * stats263["GD"]["norepinephrine"]
          + 0.5 * etaN * (partial281[("norepinephrine", 2.0)][0]
                          + half_ratio * partial281[("norepinephrine", 0.5)][0]))
    RN = (all_factor_anchor * stats263["RD"]["norepinephrine"]
          + 0.5 * etaN * (partial281[("norepinephrine", 2.0)][1]
                          + half_ratio * partial281[("norepinephrine", 0.5)][1]))
    A = stats263["GF"] + etaI * stats263["GD"]["insulin"] + GN + 1e-4 * np.eye(193)
    B = stats263["RF"] + etaI * stats263["RD"]["insulin"] + RN
    return (stats263["T0"]
            + solve281(A, B, assume_a="pos", check_finite=False)
              / stats263["scale"][:, None])

grid286 = {}
for etaI in [0.25, 0.35, 0.45]:
    for etaN in [0.3, 0.4, 0.5, 0.6]:
        for half_ratio in [4.0, 8.0]:
            key = (etaI, etaN, half_ratio)
            T = fit_multidose286(*key)
            score = score234(arch263, T, stats263["Ftune"])
            proxy = float(np.mean(((np.asarray(score[3]) - anchor281)
                                   / np.maximum(np.abs(anchor281), 1e-6))**2))
            grid286[key] = {"T": T, "score": score, "proxy": proxy}

feasible286 = sorted(
    [(key, val) for key, val in grid286.items() if max(val["score"][:3]) < 0.0043],
    key=lambda kv: kv[1]["proxy"]
)
print("=== MULTI-DOSE-ANCHORED ALTERNATIVES (NOT SAVED) ===")
for key, val in feasible286[:8]:
    print(key, "NLL", np.round(val["score"][:3], 6),
          "slopes", np.round(val["score"][3], 3),
          "proxy", round(val["proxy"], 5))
print("Current edits remain weighted281 (.3,.25,8).")
print("Forward-file names", [p.name for p in Path("/work/fwd_26").iterdir()])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 286 / multi-dose-anchored alternatives\n"
            "Reproduced current offline scores. Prepared tuning stays100–199 for GPU "
            "audit. Fitted 24 unsaved closed-form heads with all-five NE-factor "
            "anchor .05 plus separate x2/half constraints; retained in grid286. "
            "Current saved tensors remain weighted281(.3,.25,8).\n")
```