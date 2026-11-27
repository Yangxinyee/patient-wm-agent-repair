Only the two correction output columns change with strength, so further scaling leaves the gate and decoder intact. I’ll trial strength 2.6, which should bring insulin closer to the target, while retaining the strength-2 fallback. A fresh factual check below the 0.005 limit would support keeping it; a larger loss would require reducing strength.

```python
import torch
import numpy as np
import pandas as pd

print("=== SAVED LOCAL STRENGTH FRONTIER ===")
for s116, v116 in r111_results.items():
    print(s116, repr(v116)[:500])
print("correction neurons", pair104.tolist(), "theta", float(theta104))
print("tilt", np.round(tilt104, 4))

f116_o = pd.read_csv("/work/fwd_12/factual_original.csv")
f116_e = pd.read_csv("/work/fwd_12/factual_edited.csv")
f116 = f116_o.merge(f116_e, on=["stay_id", "hour", "event", "observed"],
                   suffixes=("_o", "_e"), validate="one_to_one")
f116["delta"] = f116.nll_e - f116.nll_o
s116_stay = f116.groupby("stay_id").agg(
    delta_sum=("delta", "sum"), count=("delta", "size"))
s116_mean = s116_stay.delta_sum.sum() / s116_stay["count"].sum()
s116_influence = (
    s116_stay.delta_sum - s116_mean * s116_stay["count"]
) / s116_stay["count"].mean()
print("strength-2 factual delta", float(s116_mean),
      "cluster SE", float(s116_influence.std(ddof=1) / np.sqrt(len(s116_stay))))

r116_strength = 2.6
with np.load("/work/insulin_candidate_v1.npz") as f116_b:
    r116_candidate = {
        n: torch.from_numpy(np.array(f116_b[n], copy=True))
        for n in f116_b.files
    }
r116_candidate["attr_heads.value_bin.2.weight"][:, pair104] *= r116_strength
for n116, t116 in r116_candidate.items():
    save_edit(n116, t116)

print("\n=== NE RECOVERY DETAILS ===")
print("aligned rows", aligned99.columns.tolist())
print(aligned99.head(2).to_string(index=False))
for k116, v116 in fits99.items():
    print("fit", k116, type(v116).__name__,
          list(v116.keys()) if isinstance(v116, dict) else repr(v116)[:350])
print("state/panel/kernel variables",
      sorted(k for k in globals()
             if not k.startswith("_") and
             any(w in k.lower() for w in ["panel", "kernel", "states", "impulse"])))
print("installed insulin strength", r116_strength)

with open("/work/notes.md", "a") as f116_n:
    f116_n.write(
        "\n## Run 116: insulin strength-2.6 trial\n"
        "Installed strength 2.6 by scaling only value-head output columns "
        "107/109 of the saved strength-1 construction. Gate, decoder, and "
        "removed-neuron mean compensation are unchanged. "
        "Strength-2 fallback remains in r115_strength2. "
        "This trial is not accepted pending a fresh paired factual check.\n"
    )
```
