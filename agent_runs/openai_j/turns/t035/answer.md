The stored numeric inputs look consistent with the register format, but the printed prior scales may belong to an earlier diagnostic. I’ll recompute the prior from the fresh context and average its latent draws. If the mismatch comes from evaluating only the prior mean, this should remove the bias; otherwise, the head convention still needs work.

```python
import numpy as np
import torch
import torch.nn.functional as TF

torch.set_num_threads(8)
print("event_kinds:", VOCAB["event_kinds"])
print("atom_kinds:", VOCAB["atom_kinds"])

fresh_ctx = fresh_head["context"]
fresh_prior = mlp(fresh_ctx, "z_prior", "gelu")
fresh_pm = fresh_prior[:, :32]
fresh_ls = fresh_prior[:, 32:].clamp(-5, 2)
fresh_posterior = fresh_head["state"][:, 960:]
for label, x in (
    ("context", fresh_ctx), ("prior mean", fresh_pm),
    ("prior logsd", fresh_ls), ("saved posterior", fresh_posterior),
    ("world view", fresh_head["view"]),
):
    print(label, "quantiles",
          np.round(np.quantile(x.numpy(), [0, .1, .5, .9, 1]), 4))

fresh_mean_view = fresh_ctx + TF.linear(
    fresh_pm, wt("z_head.weight"), wt("z_head.bias")
)
print("fresh mean-view discrepancy:",
      float((fresh_mean_view - fresh_head["view"]).abs().max()))

g = torch.Generator().manual_seed(731)
fresh_eps = torch.randn(128, len(hm), 32, generator=g)
fresh_centres = torch.tensor(
    [[0.] + VOCAB["value_bin_centres"][e] for e in hm.event],
    dtype=torch.float32
)
fresh_codes = torch.tensor(hm.code.to_numpy(), dtype=torch.long)
fresh_bins = torch.tensor(hm.last_bin.to_numpy(), dtype=torch.long)
fresh_targets = torch.tensor(hm.target_bin.to_numpy(), dtype=torch.long)
fresh_transition = wt("embed_transition.value_bin.weight")[
    fresh_codes * 17 + fresh_bins
]
fresh_register = fresh_head["register"]
fresh_rows = []

for sd_scale in (0., .5, 1., 2.):
    z = fresh_pm[None] + sd_scale * fresh_ls.exp()[None] * fresh_eps
    view_mc = fresh_ctx[None] + TF.linear(
        z, wt("z_head.weight"), wt("z_head.bias")
    )
    q_mc = ln(view_mc + wt("type_emb.weight")[1], "query_norm")
    x_mc = q_mc + fresh_register[None]
    logits_mc = mlp(x_mc, "attr_heads.value_bin", "gelu") + fresh_transition[None]
    p_mc = logits_mc.softmax(-1).mean(0)
    predicted = (p_mc * fresh_centres).sum(-1).numpy()
    predicted_nll = -p_mc[
        torch.arange(len(hm)), fresh_targets
    ].log().numpy()
    error = (predicted - hm.original_mean.to_numpy()) / hm.scale.to_numpy()
    fresh_rows.append({
        "sd_scale": sd_scale,
        "scaled_mean_rmse": float(np.sqrt(np.mean(error**2))),
        "scaled_bias": float(error.mean()),
        "nll_rmse": float(np.sqrt(np.mean(
            (predicted_nll - hm.original_nll.to_numpy())**2
        ))),
        "mean_nll": float(predicted_nll.mean()),
        "mean_p_absent": float(p_mc[:, 0].mean()),
    })

import pandas as pd
print(pd.DataFrame(fresh_rows).to_string(index=False))
```
