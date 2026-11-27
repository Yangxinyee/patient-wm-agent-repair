The fourth cohort’s factual loss is about 0.00250 nats, so the reduced edit passes on all 400 tuning stays. This suggests room for a stronger repair. I’ll try increasing the insulin component while reducing the norepinephrine component, prioritizing the main readout without abandoning either drug’s inferred effect. The trial will be rejected if factual loss exceeds 0.005 or insulin rollouts fail to improve.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

parts278 = []
for k in [36, 37, 38, 39]:
    ao = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    ae = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    keys = ["stay_id", "hour", "event", "observed"]
    assert ao[keys].equals(ae[keys])
    d = ao[keys].copy()
    d["original_nll"] = ao.nll.to_numpy()
    d["edited_nll"] = ae.nll.to_numpy()
    d["delta"] = d.edited_nll - d.original_nll
    parts278.append(d)
all_factual278 = pd.concat(parts278, ignore_index=True)
assert all_factual278.stay_id.nunique() == 400
clusters278 = all_factual278.groupby("stay_id").agg(
    delta_sum=("delta", "sum"), n=("delta", "size"))
rng278 = np.random.default_rng(278)
ix278 = rng278.integers(0, len(clusters278), size=(4000, len(clusters278)))
bs278 = (clusters278.delta_sum.to_numpy()[ix278].sum(axis=1) /
         clusters278.n.to_numpy()[ix278].sum(axis=1))
print("CURRENT JOINT274, ALL 400 TUNING STAYS")
print({"readings": len(all_factual278),
       "original_nll": float(all_factual278.original_nll.mean()),
       "edited_nll": float(all_factual278.edited_nll.mean()),
       "delta": float(all_factual278.delta.mean()),
       "cluster_bootstrap_95": np.quantile(bs278, [.025, .975]).tolist(),
       "by_event_delta": all_factual278.groupby("event").delta.mean().to_dict()})

with np.load("/work/joint268.npz") as a:
    insulin_unit278 = a["insulin_delta"].astype(np.float64) / float(a["insulin_scale"])
with np.load("/work/ne_candidate151.npz") as a:
    ne_unit278 = a["ne_delta"].astype(np.float64)
with np.load("/work/joint274.npz") as a:
    reference_delta278 = a["delta"].copy()
original_b278 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
current_b278 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
print("Current edit archive error",
      float(np.max(np.abs(current_b278 - (original_b278 + reference_delta278)))))
assert np.max(np.abs(current_b278 - (original_b278 + reference_delta278))) < 1e-4

insulin_scale278 = .90
ne_scale278 = .25
insulin_delta278 = (insulin_scale278 * insulin_unit278).astype(np.float32)
ne_delta278 = (ne_scale278 * ne_unit278).astype(np.float32)
joint_delta278 = insulin_delta278 + ne_delta278
save_edit("core.b_in.weight",
          torch.from_numpy(original_b278 + joint_delta278))
np.savez("/work/joint278.npz", delta=joint_delta278,
         insulin_delta=insulin_delta278, ne_delta=ne_delta278,
         insulin_scale=insulin_scale278, insulin_blend=.60,
         ne_scale=ne_scale278)
print("INSTALLED UNACCEPTED TRIAL",
      {"insulin_scale": insulin_scale278, "insulin_blend": .60,
       "ne_scale": ne_scale278, "delta_norm": float(np.linalg.norm(joint_delta278)),
       "edited_tensors": edited()})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 278: full-cohort safety and insulin-prioritized trial\n"
            f"Joint274 all400 factual delta={all_factual278.delta.mean():.9f}; "
            f"cluster bootstrap95={np.quantile(bs278, [.025, .975]).tolist()}. "
            "All four 100-stay cohorts were individually below .005. "
            "Installed unaccepted joint278: I_scale=.90, I_blend=.60, "
            "NE_scale=.25; only core.b_in.weight. Closed-form recombination "
            "of existing record-derived rank corrections, no gradients/optimizer. "
            "Joint274 remains archived as the validated factual fallback. "
            "Python runs: 278; forwards: 39.\n")
```