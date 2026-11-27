The fresh decoder has factual active-dose RMSEs of 0.24 for insulin and 0.12 for norepinephrine, and its presence estimates separate drug starts from inactive bundles. The two latent reading directions are also nearly orthogonal. I’ll try a small, gated posterior correction, while compensating for the four neurons it replaces. It remains a hypothesis: factual NLL and rollout responses must validate it.

```python
import numpy as np
import torch
from scipy.special import ndtr

# Preserve originals and construct a closed-form compression of four hidden units.
repair_names = ["z_post.0.weight", "z_post.0.bias",
                "z_post.2.weight", "z_post.2.bias"]
repair_original = {n: as_numpy(load_original(n)).astype(np.float64).copy()
                   for n in repair_names}
repair_units = np.asarray(r46_units, dtype=int).copy()
repair_keep = np.setdiff1d(np.arange(192), repair_units)
repair_H = as_numpy(r46_hidden).astype(np.float64)
repair_comp_X = np.column_stack([repair_H[:, repair_keep],
                                np.ones(len(repair_H))])
repair_comp_Y = repair_H[:, repair_units] @ (
    repair_original["z_post.2.weight"][:, repair_units].T)
repair_comp_penalty = np.eye(repair_comp_X.shape[1]) * .01
repair_comp_penalty[-1, -1] = 0.
repair_comp_coef = np.linalg.solve(
    repair_comp_X.T @ repair_comp_X + repair_comp_penalty,
    repair_comp_X.T @ repair_comp_Y)
repair_comp_error = repair_comp_X @ repair_comp_coef - repair_comp_Y
print("compression posterior-output RMSE:",
      float(np.sqrt(np.mean(repair_comp_error**2))),
      "max absolute:", float(np.max(np.abs(repair_comp_error))))

repair_template = {n: v.copy() for n,v in repair_original.items()}
repair_template["z_post.2.weight"][:, repair_keep] += repair_comp_coef[:-1].T
repair_template["z_post.2.bias"] += repair_comp_coef[-1]
repair_template["z_post.2.weight"][:, repair_units] = 0.

repair_gate_scale = 20.
repair_pilot_strength = np.array([-2., 3.3], dtype=np.float64)
repair_zdirs = np.column_stack([r46_vg, r46_vm]).astype(np.float64)
repair_centering = {}

for j, drug in enumerate(r41_drug_names):
    unit_a, unit_b = repair_units[2*j:2*j+2]
    policy_w, policy_b = repair_policy_w[drug]
    on = r41_present[:, j] > .5
    residual = (repair_actual_pred[:, j] -
                repair_context @ policy_w - policy_b)
    center = float(residual[on].mean())
    repair_centering[drug] = center
    policy_b += center

    gate_row = np.zeros(576)
    gate_row[384:] = repair_gate_scale * repair_action_w[j+2]
    gate_bias = -repair_gate_scale * .5
    residual_row = np.zeros(576)
    residual_row[:192] = -policy_w
    residual_row[384:] = repair_action_w[j]

    repair_template["z_post.0.weight"][unit_a] = gate_row + residual_row
    repair_template["z_post.0.bias"][unit_a] = gate_bias - policy_b
    repair_template["z_post.0.weight"][unit_b] = gate_row
    repair_template["z_post.0.bias"][unit_b] = gate_bias
    direction = repair_zdirs[:, j] * repair_pilot_strength[j]
    repair_template["z_post.2.weight"][:32, unit_a] = direction
    repair_template["z_post.2.weight"][:32, unit_b] = -direction

    gate = repair_gate_scale * (repair_actual_pred[:, j+2] - .5)
    innovation = residual - center
    correction = (gate+innovation)*ndtr(gate+innovation)-gate*ndtr(gate)
    print(drug, "centering:", center,
          "active innovation quantiles:", np.round(np.quantile(
              innovation[on], [.01,.5,.99]), 4),
          "inactive correction RMS:", float(np.sqrt(np.mean(
              correction[~on]**2))))
    augmented_gate = repair_X @ repair_action_w[j+2]
    augmented_on = repair_Y[:, j+2] > .5
    print("augmented presence quantiles:",
          np.round(np.quantile(augmented_gate[augmented_on],
                               [.001,.01,.5,.99,.999]), 4),
          "fraction below gate threshold:",
          float(np.mean(augmented_gate[augmented_on] < .5)))

for name in repair_names:
    save_edit(name, torch.tensor(repair_template[name], dtype=torch.float32))
print("edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 53; forwards 3. First pilot submitted: four low-importance "
        "posterior hidden units replaced by two gated GELU differences. "
        "Each writes a glucose/MAP latent direction proportional to decoded "
        "dose minus contextual expected dose, inactive without a WORLD drug. "
        "Other posterior outputs compensated by closed-form hidden-feature "
        "regression. Pilot strengths -2 insulin/+3.3 NE, from record-derived "
        "immediate effects. Four z_post tensors edited; factual and rollout "
        "validation still required.\n"
    )
```