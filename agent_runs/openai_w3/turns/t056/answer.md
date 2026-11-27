The original rollout is nearly insensitive to insulin and predicts a MAP decrease when norepinephrine is doubled, unlike the history-adjusted estimates from the records. I’ll check the retained fits for a correction that is inactive without a drug and responds to dose deviations from the recorded policy. Accurate held-out decoding and sensible response directions would justify a pilot, not yet a final edit.

```python
import numpy as np
import torch

def brief_tree(obj, depth=0):
    if isinstance(obj, dict):
        return {str(k): brief_tree(v, depth+1) for k, v in obj.items()}
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        z = as_numpy(obj)
        return {"shape": list(z.shape), "norm": float(np.linalg.norm(z)),
                "range": [float(np.min(z)), float(np.max(z))]} if z.size else {"shape": list(z.shape)}
    if isinstance(obj, (list, tuple)) and len(obj) < 8:
        return [brief_tree(v, depth+1) for v in obj]
    return str(obj)[:160]

print("=== RETAINED FITS ===")
for name in ("r41_policies", "r46_models", "r47_diagnostics",
             "r45_sensor_directions", "r46_units"):
    print(name, brief_tree(globals()[name]))

print("\n=== ACTION DECODER ON FACTUAL BUNDLES ===")
repair_decoder_raw = r41_action_coef / r41_uscale[None, :]
repair_decoded = r47_actual_u @ repair_decoder_raw.T
for j, drug in enumerate(r41_drug_names):
    on = r41_present[:, j] > 0.5
    off = ~on
    print(drug,
          "true rate:", np.quantile(r41_rates[on, j], [0, .1, .5, .9, 1]),
          "decoded rate RMSE:", np.sqrt(np.mean(
              (repair_decoded[on, j] - r41_rates[on, j])**2)),
          "presence on:", np.quantile(repair_decoded[on, j+2], [0, .01, .5, .99, 1]),
          "presence off:", np.quantile(repair_decoded[off, j+2], [0, .01, .5, .99, 1]))
    pol = r41_policies[drug]
    print("policy fields:", list(pol))

print("\n=== POSTERIOR UNIT IMPORTANCE ===")
print("units:", r46_units)
print("score quantiles:", np.quantile(r46_score, [0, .1, .5, .9, 1]))
print("chosen scores:", r46_score[r46_units])
print("state-direction z cosine:",
      float(np.dot(r46_vg, r46_vm) /
            (np.linalg.norm(r46_vg)*np.linalg.norm(r46_vm))))

print("\n=== BASELINE READOUTS ===")
for edit, group in roll0.groupby("edit"):
    print(edit, "all:", group["shift"].mean(),
          "reached:", group.loc[group["reached"], "shift"].mean(),
          "reference:", matched_refs.get(edit))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 51; forwards 3. Assessing gated posterior-mean repair "
            "using physical action decoding and context-predicted dose. "
            "Original response is much weaker than record-derived insulin "
            "effect and has the wrong doubled-NE sign. No edits yet.\n")
```