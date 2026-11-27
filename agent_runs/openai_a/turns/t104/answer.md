The posterior reconstruction is not close enough yet: its error is large compared with the saved latent variation. Unit choices barely change that error, suggesting a missing computational detail. I’ll test empty-bundle handling and latent parameterization next. Only near-exact agreement—not the lowest error among poor candidates—will validate the reconstruction.

```python
import itertools
import numpy as np
import torch

rm_b = len(rm_pure_idx)
rm_zero_e = torch.zeros(rm_b, 192)
rm_empty_e = rm_ln(
    rm_mlp(rm_lin(torch.zeros(rm_b, 1), "encode.size"), "encode.mlp2"),
    "encode.norm")
rm_context_scores = []
rm_candidate_cache = {}

for enc in rm_encoding_scores[:6]:
    _, unit_id, rate_unit_id, side, use_kq, activation = enc
    rm_atoms = rm_insulin_atoms(unit_id, rate_unit_id, side, use_kq, activation)
    rm_u = rm_set(rm_atoms, "encode_action", activation)
    for flow_mult, gap_style, empty_style in itertools.product(
            (0.0, 1.0, 2.0),
            ("seconds_div10", "seconds", "hours_div10", "none"),
            ("zero", "encoded")):
        h = rm_prev[:, 192:960].reshape(-1, 4, 192) * torch.exp(
            -flow_mult * rm_dt[:, :, None]
            * torch.exp(rm_W["core.flow.a"])[None, None, :])
        if gap_style == "seconds_div10":
            gap_x = torch.log1p(rm_dt)/10
        elif gap_style == "seconds":
            gap_x = torch.log1p(rm_dt)
        elif gap_style == "hours_div10":
            gap_x = torch.log1p(rm_dt/3600)/10
        else:
            gap_x = None
        ctx = rm_prev[:, :192] + rm_ln(h.mean(1), "core.out_norm")
        if gap_x is not None:
            ctx = ctx + rm_mlp(gap_x, "gap", activation)
        e = rm_zero_e if empty_style == "zero" else rm_empty_e
        raw = rm_mlp(torch.cat([ctx, e, rm_u], 1), "z_post", activation)[:, :32]
        prior_mu = rm_mlp(ctx, "z_prior", activation)[:, :32]
        for transform, pred in (
                ("linear", raw), ("tanh", torch.tanh(raw)),
                ("half", 0.5*raw), ("prior_plus", prior_mu+raw)):
            err = float(torch.sqrt(torch.mean((pred-rm_target_z)**2)))
            key = (unit_id, rate_unit_id, side, use_kq, activation,
                   flow_mult, gap_style, empty_style, transform)
            rm_context_scores.append((err, key))
            if err < 0.03:
                rm_candidate_cache[key] = pred.clone()

rm_context_scores.sort()
print("empty-world encoded vector norm:", float(rm_empty_e[0].norm()))
print("best posterior reconstructions:")
for result in rm_context_scores[:12]:
    print(result)

rm_best_enc = rm_encoding_scores[0]
_, ui, ri, bs, kq, act = rm_best_enc
rm_u0 = rm_set(rm_insulin_atoms(ui, ri, bs, kq, act), "encode_action", act)
rm_raw0 = rm_mlp(torch.cat([rm_ctx, rm_zero_e, rm_u0], 1), "z_post", act)[:, :32]
a = rm_raw0.numpy().ravel()
b = rm_target_z.numpy().ravel()
slope, intercept = np.linalg.lstsq(
    np.column_stack([a, np.ones_like(a)]), b, rcond=None)[0]
print("initial reconstruction diagnostics:", {
    "prediction_sd": float(a.std()), "target_sd": float(b.std()),
    "correlation": float(np.corrcoef(a, b)[0, 1]),
    "best_common_scale": float(slope), "intercept": float(intercept),
    "scaled_rmse": float(np.sqrt(np.mean((slope*a+intercept-b)**2)))
})
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Posterior reconstruction diagnosis; Python 90\n"
            f"- Best tested reconstruction: {rm_context_scores[0]}.\n"
            "- Synthetic-state edit remains blocked until reconstruction validates.\n")
```