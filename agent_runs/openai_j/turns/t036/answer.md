Latent averaging did not remove the bias, and the freshly computed prior has ordinary scales. That rules out the stale diagnostic values as the explanation. I’ll now isolate the time-flow and world-view terms while holding the register fixed; a correct convention should approach the repeated-pass noise level across all four sensors.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF

print("=== EXISTING BEST DIAGNOSTICS ===")
for name in ("check_table", "integration_table", "reconstruction_table"):
    table = globals().get(name)
    if isinstance(table, pd.DataFrame):
        print(name)
        print(table.nsmallest(5, "scaled_mean_rmse").to_string(index=False))

print("\nlast_numeric column norms:",
      wt("last_numeric.weight").norm(dim=0).numpy())
print("flow rates:", np.quantile(wt("core.flow.a").exp().numpy(),
                                 [0, .1, .5, .9, 1]))

state_local = fresh_head["state"]
out_local = state_local[:, :192]
slots_local = state_local[:, 192:960].reshape(-1, 4, 192)
dt_local = fresh_head["dt"]
gap_local = mlp(torch.log1p(dt_local)[:, None] / 10, "gap", "gelu")
target_mean_local = hm.original_mean.to_numpy()
target_nll_local = hm.original_nll.to_numpy()
scale_local = hm.scale.to_numpy()

def score_view_local(view, register=fresh_register):
    query = ln(view + wt("type_emb.weight")[1], "query_norm")
    logits = mlp(query + register, "attr_heads.value_bin", "gelu") + fresh_transition
    p = logits.softmax(-1)
    mean = (p * fresh_centres).sum(-1).numpy()
    nll = -p[torch.arange(len(hm)), fresh_targets].log().numpy()
    error = (mean - target_mean_local) / scale_local
    return {
        "scaled_mean_rmse": float(np.sqrt(np.mean(error**2))),
        "scaled_bias": float(error.mean()),
        "nll_rmse": float(np.sqrt(np.mean((nll - target_nll_local)**2))),
    }, error

view_grid = []
best_view_local = None
for flow_factor in (0., .5, 1., 2., 4., 8.):
    flowed_local = slots_local * torch.exp(
        -flow_factor * dt_local[:, None, None] *
        wt("core.flow.a").exp()[None, None]
    )
    summary_local = ln(flowed_local.mean(1), "core.out_norm")
    context_local = out_local + summary_local + gap_local
    z_local = mlp(context_local, "z_prior", "gelu")[:, :32]
    latent_local = TF.linear(z_local, wt("z_head.weight"), wt("z_head.bias"))
    for out_coef, summary_coef, gap_coef in (
        (1., 1., 1.), (0., 1., 1.), (1., 0., 1.),
        (.5, .5, 1.), (1., 2., 1.), (2., 1., 1.),
        (1., 1., 0.), (1., 1., .5), (1., 1., 2.),
    ):
        view_local = (
            out_coef * out_local + summary_coef * summary_local
            + gap_coef * gap_local + latent_local
        )
        scores, error_local = score_view_local(view_local)
        row = dict(flow=flow_factor, out=out_coef,
                   summary=summary_coef, gap=gap_coef, **scores)
        view_grid.append(row)
        if best_view_local is None or scores["scaled_mean_rmse"] < best_view_local[0]:
            best_view_local = (
                scores["scaled_mean_rmse"], row,
                view_local.clone(), error_local.copy()
            )

view_grid_table = pd.DataFrame(view_grid)
print("\n=== WORLD-VIEW GRID ===")
print(view_grid_table.nsmallest(12, "scaled_mean_rmse").to_string(index=False))
print("\n=== SENSOR ERRORS: DOCUMENTED / BEST GRID ===")
documented_scores, documented_error = score_view_local(fresh_head["view"])
for event in MEAS:
    selection = hm.event.to_numpy() == event
    for label, errors in (
        ("documented", documented_error),
        ("best_grid", best_view_local[3]),
    ):
        print(event, label, "n", int(selection.sum()),
              "RMSE", float(np.sqrt(np.mean(errors[selection]**2))),
              "bias", float(errors[selection].mean()))
```
