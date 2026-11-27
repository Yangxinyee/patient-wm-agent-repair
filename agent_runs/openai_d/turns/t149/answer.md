The test stopped at a tensor-name lookup, before computing results or changing weights. The prior is named `z_prior`, not `core.z_prior`. I’ll correct that and rerun the same test. It will support a candidate only if reconstruction is consistent and intervention sensitivity generalizes without excessive factual variation.

```python
r129_pp_o = r124_mlp(r127_ctx_o, "z_prior")
r129_pp_e = r124_mlp(r127_ctx_e, "z_prior")
print("Prior log-SD range:",
      float(r129_pp_o[:, 32:].min()), float(r129_pp_o[:, 32:].max()))

def r129_queries(ctx, pp):
    mu, log_sd = pp[:, :32], pp[:, 32:]
    z = mu[None] + torch.exp(log_sd.clamp(-6, 2))[None] * eps98
    view = ctx[None] + r124_lin(z, "z_head")
    return r124_norm(view + r124_W["type_emb.weight"][1], "query_norm")

r129_q_o = r129_queries(r127_ctx_o, r129_pp_o)
r129_q_e = r129_queries(r127_ctx_e, r129_pp_e)
r129_register = r124_x - r129_q_o
r129_register_error = float(
    torch.max(torch.abs(r129_register - r129_register[0:1]))
)
print("Register consistency across draws:", r129_register_error)
print("X99 agreement:",
      float(np.max(np.abs(np.asarray(X99) - r124_x.numpy()))))
print("Rates:", {k: r126_desc(v) for k, v in rates99.items()})
print("Gate objects:", {str(k): r126_desc(v) for k, v in gates99.items()})

if r129_register_error < 0.001:
    r129_x_o = r124_x.numpy()
    r129_x_e = (r129_q_e + r129_register).numpy()
    r129_dx = r129_x_e - r129_x_o
    r129_family = meta96.event.isin(["map_arterial", "map_cuff"]).to_numpy()
    r129_active = np.asarray(rates99["norepinephrine"]) > 0
    r129_fit = r129_family & trainmask99
    r129_val = r129_family & valmask99
    r129_response_fit = r129_fit & r129_active
    r129_response_val = r129_val & r129_active

    r129_trainX = r129_x_o[fitdraws99][:, r129_fit].reshape(-1, 192).astype(np.float64)
    r129_mean = r129_trainX.mean(axis=0)
    r129_centered = r129_trainX - r129_mean
    r129_cov = r129_centered.T @ r129_centered / len(r129_centered)
    r129_dmean = r129_dx[:, r129_response_fit].mean(axis=(0, 1)).astype(np.float64)
    print("Map rows train/validation; active:",
          int(r129_fit.sum()), int(r129_val.sum()),
          int(r129_response_fit.sum()), int(r129_response_val.sum()))
    print("Mean active probe-direction norm:", float(np.linalg.norm(r129_dmean)))

    r129_candidates = {}
    for ridge in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        v = sla.solve(r129_cov + ridge*np.eye(192), r129_dmean, assume_a="pos")
        v /= max(float(v @ r129_dmean), 1e-30)
        b = -float(r129_mean @ v)
        r129_candidates[ridge] = (v, b)
        scores = r129_x_o[:, r129_val] @ v + b
        response = r129_dx[:, r129_response_val] @ v
        print({
            "ridge": ridge,
            "train_SD_for_unit_probe_response": float(np.sqrt(v @ r129_cov @ v)),
            "validation_score_mean_SD": [float(scores.mean()), float(scores.std())],
            "validation_probe_response_mean_SD":
                [float(response.mean()), float(response.std())],
            "weight_norm": float(np.linalg.norm(v))
        })
else:
    print("STOP: reconstruction must be corrected; no edits installed.")

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 129–130: controlled NE head-direction test\n"
            "Run 129 stopped on the incorrect core.z_prior prefix; no edits. "
            f"Run 130 register consistency error {r129_register_error:.8g}. "
            "Testing constrained directions on held-out stays; no new edits.\n")
```