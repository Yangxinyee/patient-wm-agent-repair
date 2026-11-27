The current detector responds almost as strongly to insulin edits as to norepinephrine edits, which could explain the interference. I’ll compare the retained candidates’ signal-to-leakage ratios and check whether the factual damage is mostly a systematic bias. A substantially cleaner detector would justify another recurrent edit; otherwise, I’ll change the approach.

```python
import numpy as np
import pandas as pd
import torch

print("NE CANDIDATE FRONTIER")
tab88 = pd.DataFrame(records82)
cols88 = ["cross", "half_w", "anchor", "s2.0", "s0.5",
          "unit_factual", "unit_cross", "half_ratio", "score"]
print(tab88.sort_values("score")[cols88].head(12).round(5).to_string(index=False))
print("\nLOWEST CROSS LEAKAGE AMONG POSITIVE DOUBLE/HALF SLOPES")
pos88 = tab88[(tab88["s2.0"] > 0) & (tab88["s0.5"] > 0)]
print(pos88.sort_values("unit_cross")[cols88].head(8).round(5).to_string(index=False))

print("\nBASIS/UNIT CHECKS")
raw88 = array80(slot_deltas47[("norepinephrine", 2.0)]).reshape(-1, 192)
for name88 in ["B64", "N64", "basis63"]:
    b88 = array80(globals()[name88])
    for mode88, xx88 in [("raw", raw88), ("scaled", raw88 / slot_scale47)]:
        err88 = np.linalg.norm(xx88 @ b88 - D81[("norepinephrine", 2.0)])
        print(name88, mode88, "difference norm", float(err88))
print("fit_drug57", np.unique(fit_drug57, return_counts=True))
print("train_slots81", np.unique(train_slots81, return_counts=True))
for key88 in targets57:
    yy88 = array80(targets57[key88])
    print(key88, yy88.shape, "min", yy88.min(axis=0), "max", yy88.max(axis=0))
print("local hnew46", tuple(hnew46.shape), "gain range",
      float(gain81.min()), float(gain81.max()))

print("\nLOCAL STATE INPUTS")
for k88, v88 in sorted(list(globals().items())):
    if k88.startswith(("cf_", "core_")) and not callable(v88):
        if isinstance(v88, (np.ndarray, torch.Tensor)):
            print(k88, tuple(v88.shape))
        elif isinstance(v88, dict):
            print(k88, "dict", [(str(k), getattr(v, "shape", None))
                               for k, v in list(v88.items())[:3]])
        elif isinstance(v88, pd.DataFrame):
            print(k88, "DataFrame", v88.shape)

o88 = pd.read_csv("/work/fwd_11/factual_original.csv")
e88 = pd.read_csv("/work/fwd_11/factual_edited.csv")
print("\nFACTUAL SCHEMA", list(o88.columns))
if len(o88) == len(e88):
    keycols88 = [c for c in ["stay_id", "hour", "event", "code"] if c in o88]
    assert all(o88[c].equals(e88[c]) for c in keycols88)
    mean_col88 = next((c for c in ["pred_mean", "mean", "predicted_mean"] if c in o88), None)
    nll_col88 = next((c for c in ["nll", "value_nll"] if c in o88), None)
    event_col88 = next((c for c in ["event", "code"] if c in o88), None)
    diag88 = o88[keycols88].copy()
    if mean_col88:
        diag88["mean_delta"] = e88[mean_col88] - o88[mean_col88]
    if nll_col88:
        diag88["nll_delta"] = e88[nll_col88] - o88[nll_col88]
    if event_col88 and len(diag88.columns) > len(keycols88):
        print(diag88.groupby(event_col88).agg(["mean", "std"]).round(4).to_string())
    else:
        print(o88.head(1).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 88: NE signal/leakage frontier\n"
            "Comparing retained NE feature fits, checking basis and raw-dose units, and auditing "
            "paired factual mean/NLL shifts. Baseline77 remains installed. "
            "Python runs: 88; forwards: 12.\n")
```