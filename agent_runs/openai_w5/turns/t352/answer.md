The recovered notes support substantial drug effects in the records, while the original model’s rollouts barely respond to dose changes. The output was truncated before the calibration audit, so it provides no new acceptance evidence. I’ll print a compact paired-loss comparison and the exact stay groups. Lower validation loss without crossing the 0.005 limit would support keeping the calibration.

```python
import numpy as np
import pandas as pd
import json, re
from pathlib import Path

def nllcol297(d):
    for c in ("nll", "value_nll", "fine_bin_nll", "fine_nll", "nll_value"):
        if c in d.columns:
            return c
    cs = [c for c in d.columns if "nll" in c.lower()]
    if len(cs) != 1:
        raise ValueError(("Ambiguous NLL columns", cs))
    return cs[0]

def staycol297(d):
    return next(c for c in d.columns if "stay" in c.lower())

print("=== FACTUAL FILE AUDIT ===")
for (k, model), d in sorted(REC296["factual"].items()):
    c = nllcol297(d)
    print(k, model, len(d), round(float(d[c].mean()), 8))

print("\n=== CURRENT FACTUAL SCHEMA ===")
d297 = REC296["factual"][(41, "edited")]
print(list(d297.columns))
print(d297.head(2).to_string(index=False))

PRE297, POST297 = {}, {}
print("\n=== STAY GROUPS AND PAIRED CALIBRATION COMPARISONS ===")
for gi, g in enumerate(REC296["groups"]):
    ids = frozenset(g["ids"])
    candidates = []
    for (k, model), e in REC296["factual"].items():
        if model != "edited" or (k, "original") not in REC296["factual"]:
            continue
        s = staycol297(e)
        if frozenset(e[s].unique()) == ids:
            candidates.append(k)
    before = max([k for k in candidates if k <= 37], default=None)
    after = max([k for k in candidates if k >= 39], default=None)
    print("GROUP", gi, "before", before, "after", after)
    print("IDS", json.dumps(g["ids"], default=str))
    for phase, k, store in (("before", before, PRE297), ("after", after, POST297)):
        if k is None:
            continue
        o = REC296["factual"][(k, "original")]
        e = REC296["factual"][(k, "edited")]
        s = staycol297(o)
        assert len(o) == len(e) and np.array_equal(o[s].values, e[s].values)
        loss = e[nllcol297(e)].to_numpy() - o[nllcol297(o)].to_numpy()
        paired = pd.DataFrame({"stay": o[s].values, "delta": loss})
        store[gi] = paired
        sidnum = pd.to_numeric(o[s], errors="coerce")
        even = np.asarray((sidnum % 2) == 0)
        print(phase, "delta", round(float(loss.mean()), 8),
              "even_ID_delta", round(float(loss[even].mean()), 8),
              "odd_ID_delta", round(float(loss[~even].mean()), 8))
    if before is not None and after is not None:
        print("CALIBRATION_CHANGE",
              round(float(POST297[gi]["delta"].mean() -
                          PRE297[gi]["delta"].mean()), 8))

def pool297(parts, label):
    if not parts:
        return
    a = pd.concat(list(parts.values()), ignore_index=True)
    z = a.groupby("stay")["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(297)
    ix = rng.integers(0, len(z), size=(2000, len(z)))
    boot = z["sum"].to_numpy()[ix].sum(1) / z["count"].to_numpy()[ix].sum(1)
    print(label, "groups", len(parts), "stays", len(z), "readings", len(a),
          "pooled_delta", float(a["delta"].mean()),
          "bootstrap_5_95", np.quantile(boot, [.05, .95]).tolist())

print("\n=== POOLED LOSSES (ONLY MATCHING SNAPSHOTS) ===")
pool297(PRE297, "PRE_CALIBRATION")
pool297(POST297, "POST_CALIBRATION_AVAILABLE")

print("\n=== LATEST ROLLOUT FILES ===")
for key in sorted(REC296["dose"])[-3:]:
    d = REC296["dose"][key]
    print(key, "rows", len(d), "columns", list(d.columns))
    print(d.head(3).to_string(index=False))

print("\n=== EDIT-CONTROL VARIABLE NAMES ===")
names297 = [n for n in sorted(list(globals()))
            if re.search(r"(BASE|center|control|signal|target|truth|coef|beta|tau|295|289|285|900|1250)", n)]
print(", ".join(names297)[:2200])
for name in ("BASE289", "BASE295", "calibration295", "offsets295",
             "targets", "target_shifts", "centers", "control_neurons"):
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "keys", list(obj)[:15])
        elif hasattr(obj, "shape"):
            print(name, "shape", tuple(obj.shape))
        else:
            print(name, repr(obj)[:400])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 297: compact paired calibration audit\n"
            "- Computed pre/post-calibration paired losses on matching stay groups, "
            "including even-ID validation subsets and stay-bootstrap uncertainty.\n"
            "- Recovered exact group IDs and latest rollout schemas. No weights changed.\n"
            "- Missing post-calibration groups must be evaluated before claiming "
            "full 400-stay factual feasibility.\n")
```
