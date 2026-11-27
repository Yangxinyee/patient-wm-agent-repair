Exact vector matching aligned every cached row, and every preceding full state also matched. Correctly timed observation encodings now agree exactly with the cache. The remaining error is in posterior reconstruction, possibly empty-action handling or context computation. I’ll isolate that difference before calibrating any edit; it should disappear across both reading and drug bundles.

```python
print("=== CONTEXT TENSORS ===")
r74_gap_prefixes = [n[:-len(".0.weight")] for n in NAMES
                   if "gap" in n and n.endswith(".0.weight")]
r74_norm_prefixes = [n[:-len(".weight")] for n in NAMES
                    if "out_norm" in n and n.endswith(".weight")]
print("gap prefixes", r74_gap_prefixes, "norm prefixes", r74_norm_prefixes)

r74_seconds = np.round(
    (r72_hours[r73_raw]-r72_hours[r73_raw-1])*60)*60
r74_dt = torch.tensor(r74_seconds, dtype=torch.float32).clamp_min(1e-3)
print("actual/stored dt quantiles",
      np.quantile(r74_seconds, [0, .5, 1]),
      np.quantile(r36_dt.numpy(), [0, .5, 1]))
r74_prev = torch.tensor(r72_x[r73_raw-1])
r74_hprev = r74_prev[:, 192:960].reshape(-1, 4, 192)
r74_flow = r74_hprev * torch.exp(
    -r74_dt[:, None, None]*torch.exp(r33_weights["core.flow.a"])[None, None, :])
r74_contexts = {"cached": r36_ctx}
for norm_prefix in r74_norm_prefixes:
    base = r74_prev[:, :192] + r33_ln(r74_flow.mean(1), norm_prefix)
    for gap_prefix in r74_gap_prefixes:
        for act in ("gelu", "silu"):
            r74_contexts[f"{norm_prefix}/{gap_prefix}/{act}"] = (
                base + r33_mlp(torch.log(r74_dt)[:, None]/10, gap_prefix, act))
r74_us = {
    "zero-empty": r36_set_encode("encode_action", r71_action, True),
    "encoded-empty": r36_set_encode("encode_action", r71_action, False),
}
r74_active = torch.tensor(r36_has_start.any(1))
r74_obs = torch.zeros(len(r73_raw), dtype=torch.bool)
r74_obs[np.asarray(r73_measurements)] = True

print("\n=== POSTERIOR AGREEMENT GRID ===")
r74_best = None
for ctx_label, ctx in r74_contexts.items():
    for u_label, u in r74_us.items():
        hidden = r33_act(r33_linear(torch.cat((ctx, r36_e, u), 1),
                                   "z_post.0"), "gelu")
        z = r33_linear(hidden, "z_post.2")[:, :32]
        diff = z-r36_latent_target
        rms = float(diff.square().mean().sqrt())
        print(ctx_label, u_label,
              "overall/reading/drug RMSE", np.round(
                  [rms, float(diff[r74_obs].square().mean().sqrt()),
                   float(diff[r74_active].square().mean().sqrt())], 6))
        if r74_best is None or rms < r74_best[0]:
            r74_best = (rms, ctx_label, u_label)
            r74_ctx, r74_u, r74_hidden = ctx.clone(), u.clone(), hidden.clone()
print("BEST", r74_best)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 74\n"
            "Run 73 exactly aligned all 8443 cached posterior vectors and all previous states. "
            "Cache stay IDs were correct; cached hours were floored. Integer observation encodings match exactly. "
            "Posterior mismatch remains; testing true-time context and empty-action handling. "
            "No edits. Counts: 74 Python runs, 6 forward passes.\n")
```