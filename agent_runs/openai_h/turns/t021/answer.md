The final batch increased NLL by only 0.00025 nats, so all four batches remain comfortably within tolerance. I’ll now compute the pooled factual change and inspect the existing delayed-effect estimates. Consistent delayed responses would challenge the near-zero-effect hypothesis; otherwise, the minimal edit remains supported.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pprint import pprint

audit_pairs = []
for audit_k in (3, 5, 6, 7):
    audit_o = pd.read_csv(f"/work/fwd_{audit_k}/factual_original.csv")
    audit_e = pd.read_csv(f"/work/fwd_{audit_k}/factual_edited.csv")
    audit_keys = ["stay_id", "hour", "event", "observed"]
    pd.testing.assert_frame_equal(audit_o[audit_keys], audit_e[audit_keys])
    audit_pair = audit_o.copy()
    audit_pair = audit_pair.rename(columns={"nll": "nll_original"})
    audit_pair["nll_edited"] = audit_e["nll"].to_numpy()
    audit_pair["delta"] = audit_pair["nll_edited"] - audit_pair["nll_original"]
    audit_pairs.append(audit_pair)

audit_all = pd.concat(audit_pairs, ignore_index=True)
audit_stay = audit_all.groupby("stay_id").agg(
    delta_sum=("delta", "sum"), n=("delta", "size")
)
audit_delta = float(audit_all["delta"].mean())
audit_g = len(audit_stay)
audit_influence = audit_stay["delta_sum"] - audit_delta * audit_stay["n"]
audit_se = float(
    np.sqrt(audit_g / (audit_g - 1) * np.square(audit_influence).sum())
    / len(audit_all)
)
audit_summary = {
    "stays": audit_g,
    "readings": len(audit_all),
    "original_nll": float(audit_all["nll_original"].mean()),
    "edited_nll": float(audit_all["nll_edited"].mean()),
    "delta_nll": audit_delta,
    "stay_cluster_se": audit_se,
    "approx_95pct_interval": [
        audit_delta - 1.96 * audit_se,
        audit_delta + 1.96 * audit_se,
    ],
    "allowed_increase": 0.005,
}
print("=== POOLED FACTUAL VALIDATION ===")
print(json.dumps(audit_summary, indent=2))
print("\nBy event:")
print(audit_all.groupby("event").agg(
    n=("delta", "size"),
    original=("nll_original", "mean"),
    edited=("nll_edited", "mean"),
    delta=("delta", "mean"),
).to_string())
print("\nBefore/after hour 12:")
print(audit_all.groupby(audit_all["hour"] >= 13).agg(
    n=("delta", "size"), delta=("delta", "mean")
).to_string())

print("\n=== EXISTING DELAYED-EFFECT ESTIMATES ===")
for audit_h in (2, 3, 6, 12):
    for audit_event in ("glucose_finger", "map_arterial"):
        print("Horizon", audit_h, audit_event)
        pprint(delayed_results[(audit_h, audit_event)], width=110, compact=True)

print("\n=== DOSE VARIATION AFTER STRONG ADJUSTMENT ===")
audit_mask = np.asarray(held, dtype=bool) & np.asarray(eligible, dtype=bool)
for audit_j, audit_drug in enumerate(("insulin", "norepinephrine")):
    audit_d = D[audit_mask, audit_j]
    audit_r = audit_d - dose_pred_strong[audit_mask, audit_j]
    print(audit_drug, {
        "n": len(audit_d),
        "dose_quantiles": np.quantile(audit_d, [0, .25, .5, .75, .95, 1]).tolist(),
        "residual_sd": float(np.std(audit_r)),
    })

print("\n=== WEIGHT-EDIT INVARIANCE CHECK ===")
for audit_attr in ("dose", "rate"):
    audit_name = f"embed.bins.{audit_attr}.weight"
    audit_w = np.asarray(load(audit_name))
    audit_wo = np.asarray(load_original(audit_name))
    print(audit_name, {
        "present_rows_max_difference": float(np.max(np.abs(audit_w[1:] - audit_w[1]))),
        "absent_row_change": float(np.max(np.abs(audit_w[0] - audit_wo[0]))),
    })
audit_nw = np.asarray(load("embed.numeric.weight"))
audit_nwo = np.asarray(load_original("embed.numeric.weight"))
print("Numeric magnitude columns max absolute weight:",
      float(np.max(np.abs(audit_nw[:, [2, 4]]))))
print("Numeric changes by input column:",
      np.linalg.norm(audit_nw - audit_nwo, axis=0).tolist())
print("Edited tensors:", edited())

with Path("/work/notes.md").open("a") as audit_f:
    audit_f.write(
        "\n## Python run 13 / forward passes 7: complete tuning factual validation\n"
        "- Forward 3/5/6/7 cover all 400 tuning stays with unchanged edit hash e723af07c9.\n"
        "- Pooled factual results: " + json.dumps(audit_summary) + "\n"
        "- Forward 4 confirms all four edited-model dose shifts are exactly zero "
        "at every tested eligible stay-hour; original shifts are nonzero.\n"
        "- Auditing delayed-effect estimates and remaining dose variation before "
        "accepting the training-record-supported near-zero-effect hypothesis.\n"
    )
```