Changing the world-view terms did not resolve the mismatch. The remaining clue is that `last_numeric` takes eight inputs, while the documentation describes only four. I’ll test the placement of the last-value and presence fields without changing their values. A correct layout should remove the bias for all sensors, not just improve one.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF

numeric_without = fresh_register - TF.linear(
    fresh_head["numbers"], wt("last_numeric.weight"), wt("last_numeric.bias")
)
layout_rows = []
best_layout = None

for flow_factor in (1., 2.):
    layout_head = reconstruct_reading_head(hm, flow_factor=flow_factor)
    for z_slot in range(2, 8):
        for presence_slot in (None, 2, 3, 4, 5, 6, 7):
            if presence_slot == z_slot:
                continue
            layout_numbers = torch.zeros(len(hm), 8)
            layout_numbers[:, 0] = torch.tensor(
                hm.has_last.to_numpy(), dtype=torch.float32
            )
            layout_numbers[:, 1] = torch.tensor(
                np.log1p(hm.age_s.to_numpy()) / 10, dtype=torch.float32
            )
            layout_numbers[:, z_slot] = torch.tensor(
                hm.zlast.to_numpy().clip(-10, 10), dtype=torch.float32
            )
            if presence_slot is not None:
                layout_numbers[:, presence_slot] = layout_numbers[:, 0]
            layout_register = numeric_without + TF.linear(
                layout_numbers, wt("last_numeric.weight"), wt("last_numeric.bias")
            )
            scores, errors = score_view_local(
                layout_head["view"], register=layout_register
            )
            row = dict(flow=flow_factor, z_slot=z_slot,
                       presence_slot=str(presence_slot), **scores)
            layout_rows.append(row)
            if best_layout is None or scores["scaled_mean_rmse"] < best_layout["score"]:
                best_layout = {
                    "score": scores["scaled_mean_rmse"], "row": row,
                    "numbers": layout_numbers.clone(),
                    "register": layout_register.clone(),
                    "head": layout_head,
                    "errors": errors.copy(),
                }

layout_table = pd.DataFrame(layout_rows)
print("=== REGISTER LAYOUT DIAGNOSTIC ===")
print(layout_table.nsmallest(12, "scaled_mean_rmse").to_string(index=False))
print("\n=== BEST LAYOUT BY SENSOR ===")
for event in MEAS:
    selection = hm.event.to_numpy() == event
    errors = best_layout["errors"][selection]
    print(event, "n", int(selection.sum()),
          "RMSE", float(np.sqrt(np.mean(errors**2))),
          "bias", float(errors.mean()))

layout_context = best_layout["head"]["context"]
layout_prior = mlp(layout_context, "z_prior", "gelu")
layout_z = (
    layout_prior[None, :, :32] +
    layout_prior[None, :, 32:].clamp(-5, 2).exp() * fresh_eps
)
layout_view = layout_context[None] + TF.linear(
    layout_z, wt("z_head.weight"), wt("z_head.bias")
)
layout_q = ln(layout_view + wt("type_emb.weight")[1], "query_norm")
layout_p = (
    mlp(layout_q + best_layout["register"][None],
        "attr_heads.value_bin", "gelu") + fresh_transition[None]
).softmax(-1).mean(0)
layout_mean = (layout_p * fresh_centres).sum(-1).numpy()
layout_nll = -layout_p[
    torch.arange(len(hm)), fresh_targets
].log().numpy()
layout_error = (layout_mean - target_mean_local) / scale_local
print("\nBEST LAYOUT WITH 128 LATENT DRAWS:",
      "RMSE", float(np.sqrt(np.mean(layout_error**2))),
      "bias", float(layout_error.mean()),
      "NLL RMSE", float(np.sqrt(np.mean(
          (layout_nll - target_nll_local)**2
      ))))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n- Runs 27–32: fresh prior scales are ordinary; latent averaging does not explain head bias. World-view coefficient checks still favor documented terms, though double flow reduces error. Exhaustively checking the undocumented extra last_numeric fields before any edit. No edits; five forward passes.\n"
    )
```
