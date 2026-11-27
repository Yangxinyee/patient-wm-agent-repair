The recurrent path now reconstructs closely across all probes. Existing posterior features recover some insulin dose variation but essentially none for norepinephrine, so that candidate is not yet convincing. I’ll check their responses to dose multipliers: strong, drug-specific responses with little factual noise would support an edit; weak or cross-drug responses would require different features.

```python
import numpy as np
import torch
from types import SimpleNamespace
from pathlib import Path
import pandas as pd

def rr35_scaled_action_encoding(indices, drug, factor):
    sets = []
    for i in np.asarray(indices, dtype=int):
        atoms = []
        for row, kind in bundle_records_audit[state_keys_audit[int(i)]]:
            if kind == 0 and row.event in drug_codes:
                if row.event == drug:
                    fields = {c: getattr(row, c, np.nan) for c in df.columns}
                    fields["rate"] = float(row.rate) * factor
                    fields["dose"] = float(row.dose) * factor
                    row = SimpleNamespace(**fields)
                atoms.append(embed_consistent_audit(row, kind))
        sets.append(atoms)
    return encode_sets_audit(sets, "encode_action", zero_empty=False)

dose_feature_counterfactual = {}
dose_feature_response_summary = {}
for rr35_j, rr35_drug in enumerate(("insulin", "norepinephrine")):
    rr35_idx = rr34_rows[np.asarray(state_drug_rate0)[:, rr35_j] > 0]
    rr35_input = post_inputs_audit[rr35_idx].clone()
    rr35_h0 = act0(lin0(rr35_input, "z_post.0"), "gelu")
    rr35_phi0 = rr35_h0 @ dose_feature_coef + dose_feature_bias
    rr35_u1 = rr35_scaled_action_encoding(rr35_idx, rr35_drug, 1.0)
    rr35_u_error = float((rr35_u1 - rr35_input[:, 384:576]).square().mean().sqrt())
    print("\n", rr35_drug, "n =", len(rr35_idx),
          "mean rate =", float(rr34_rates[rr35_idx, rr35_j].mean()),
          "action reconstruction RMS =", rr35_u_error)
    if rr35_u_error > 1e-4:
        raise RuntimeError("Scaled-action reconstruction is not validated.")
    dose_feature_counterfactual[rr35_drug] = {
        "indices": rr35_idx, "baseline": rr35_phi0.numpy()
    }
    dose_feature_response_summary[rr35_drug] = {}
    for rr35_factor in (0., .5, 1.5, 2., 3.):
        rr35_cf_input = rr35_input.clone()
        rr35_cf_input[:, 384:576] = rr35_scaled_action_encoding(
            rr35_idx, rr35_drug, rr35_factor
        )
        rr35_cf_hidden = act0(lin0(rr35_cf_input, "z_post.0"), "gelu")
        rr35_cf_phi = rr35_cf_hidden @ dose_feature_coef + dose_feature_bias
        rr35_delta = (rr35_cf_phi - rr35_phi0).numpy()
        rr35_true_delta = (rr35_factor - 1) * rr34_rates[rr35_idx, rr35_j]
        rr35_slope = float(np.dot(rr35_delta[:, rr35_j], rr35_true_delta) /
                          max(np.dot(rr35_true_delta, rr35_true_delta), 1e-12))
        rr35_summary = {
            "mean_feature_change": rr35_delta.mean(0).tolist(),
            "mean_rate_change": float(rr35_true_delta.mean()),
            "own_feature_slope": rr35_slope,
            "own_delta_quantiles": np.quantile(
                rr35_delta[:, rr35_j], [.05, .5, .95]
            ).tolist()
        }
        dose_feature_counterfactual[rr35_drug][rr35_factor] = rr35_cf_phi.numpy()
        dose_feature_response_summary[rr35_drug][rr35_factor] = rr35_summary
        print("factor", rr35_factor, rr35_summary)

print("\n=== FACTUAL OUTPUT SCHEMA ===")
for rr35_path in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    rr35_table = pd.read_csv(rr35_path)
    print(str(rr35_path), rr35_table.columns.tolist())
    print(rr35_table.head(2).to_string(index=False))

print("\n=== NUMERIC SPECIFICATION SCHEMA ===")
print("num_spec0 keys:", list(num_spec0)[:12])
for rr35_key in list(num_spec0)[:2]:
    print(rr35_key, num_spec0[rr35_key])
print("action_empty_mode_audit:", globals().get("action_empty_mode_audit"))

with open("/work/notes.md", "a") as rr35_f:
    rr35_f.write("\n## Run 35: off-policy dose-feature assay\n"
                 "- Existing posterior features had active-dose CV R2 .168 (insulin) and -.010 (NE); no edit yet.\n"
                 "- Scaled-action helper rr35_scaled_action_encoding reconstructs factor-one actions before multiplier tests.\n"
                 "- Counterfactual scalar responses: " + repr(dose_feature_response_summary) + "\n"
                 "- Python runs: 35; forward passes: 4; no weights edited.\n")
```